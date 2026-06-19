"""
Model Training Pipeline for the Disease Prediction System.

For each disease (diabetes, heart, kidney, liver):
  1. Load dataset from CSV
  2. EDA: check nulls, outliers, class imbalance
  3. Preprocess: impute, scale, encode
  4. Handle class imbalance with SMOTE
  5. Train & compare: Logistic Regression, Random Forest, XGBoost
  6. Pick best model by F1-score (macro)
  7. Save model + scaler + feature metadata as .pkl / .json

Run: python models/train.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
)

# XGBoost import with graceful fallback
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("⚠  XGBoost not available, will use Random Forest as fallback.")

# SMOTE for class imbalance
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print("⚠  imbalanced-learn not available, skipping SMOTE.")

# Suppress convergence warnings during training
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def load_dataset(disease_key):
    """
    Load a disease dataset from CSV.

    Args:
        disease_key: Key from config.DISEASES (e.g., 'diabetes')

    Returns:
        tuple: (DataFrame, feature_columns, target_column)
    """
    disease_info = config.DISEASES[disease_key]
    csv_path = os.path.join(config.DATA_DIR, disease_info["dataset_file"])

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            f"Run 'python data/download_datasets.py' first."
        )

    df = pd.read_csv(csv_path)
    target_col = disease_info["target_column"]
    feature_cols = [c for c in df.columns if c != target_col]

    print(f"\n{'='*60}")
    print(f"  Dataset: {disease_info['display_name']}")
    print(f"{'='*60}")
    print(f"  Shape: {df.shape}")
    print(f"  Target column: {target_col}")
    print(f"  Class distribution:\n{df[target_col].value_counts().to_string()}")
    print(f"  Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0].to_string() or '  None'}")

    return df, feature_cols, target_col


def preprocess_data(df, feature_cols, target_col):
    """
    Preprocess dataset: handle missing values, encode categoricals, scale features.

    Args:
        df: Raw DataFrame
        feature_cols: List of feature column names
        target_col: Name of target column

    Returns:
        tuple: (X_scaled, y, scaler, feature_names, feature_metadata)
    """
    df = df.copy()

    # ── Handle missing values ────────────────────────────────
    for col in feature_cols:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ["float64", "int64", "float32", "int32"]:
                # Impute numeric columns with median (robust to outliers)
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  Imputed {col} with median = {median_val:.2f}")
            else:
                # Impute categorical columns with mode
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"  Imputed {col} with mode = {mode_val}")

    # ── Encode categorical features ──────────────────────────
    label_encoders = {}
    for col in feature_cols:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            print(f"  Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # ── Separate features and target ─────────────────────────
    X = df[feature_cols].values.astype(np.float64)
    y = df[target_col].values.astype(int)

    # ── Cap outliers using IQR method ────────────────────────
    for i, col in enumerate(feature_cols):
        q1 = np.percentile(X[:, i], 25)
        q3 = np.percentile(X[:, i], 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outlier_count = np.sum((X[:, i] < lower) | (X[:, i] > upper))
        if outlier_count > 0:
            X[:, i] = np.clip(X[:, i], lower, upper)
            print(f"  Capped {outlier_count} outliers in {col}")

    # ── Scale features ───────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Build feature metadata (for UI tooltips and LIME) ────
    feature_metadata = {}
    for i, col in enumerate(feature_cols):
        feature_metadata[col] = {
            "index": i,
            "min": float(np.min(df[col])),
            "max": float(np.max(df[col])),
            "mean": float(np.mean(df[col])),
            "median": float(np.median(df[col])),
            "std": float(np.std(df[col])),
        }

    return X_scaled, y, scaler, feature_cols, feature_metadata


def apply_smote(X, y, disease_key):
    """
    Apply SMOTE to balance classes if imbalanced.

    Args:
        X: Feature array
        y: Target array
        disease_key: For logging purposes

    Returns:
        tuple: (X_resampled, y_resampled)
    """
    if not HAS_SMOTE:
        print("  ⚠ SMOTE unavailable, using original data.")
        return X, y

    # Check class balance
    unique, counts = np.unique(y, return_counts=True)
    ratio = min(counts) / max(counts)

    if ratio < 0.8:  # Apply SMOTE only if minority class < 80% of majority
        print(f"  Class imbalance detected (ratio={ratio:.2f}), applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        new_unique, new_counts = np.unique(y_res, return_counts=True)
        print(f"  After SMOTE: {dict(zip(new_unique, new_counts))}")
        return X_res, y_res
    else:
        print(f"  Classes are balanced (ratio={ratio:.2f}), skipping SMOTE.")
        return X, y


def train_and_compare(X_train, y_train, X_test, y_test, disease_key):
    """
    Train Logistic Regression, Random Forest, and XGBoost.
    Compare and return the best model by F1-score.

    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        disease_key: For logging

    Returns:
        tuple: (best_model, best_model_name, results_dict)
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, class_weight="balanced"
        ),
    }

    if HAS_XGBOOST:
        models["XGBoost"] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False,
        )

    results = {}
    best_f1 = -1
    best_model = None
    best_name = None

    print(f"\n  Training models for {config.DISEASES[disease_key]['display_name']}...")
    print(f"  {'Model':<25} {'F1 (macro)':<12} {'ROC-AUC':<12} {'Accuracy':<12}")
    print(f"  {'-'*60}")

    for name, model in models.items():
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            f1 = f1_score(y_test, y_pred, average="macro")
            roc = roc_auc_score(y_test, y_proba)
            acc = np.mean(y_pred == y_test)

            results[name] = {"f1": f1, "roc_auc": roc, "accuracy": acc}
            print(f"  {name:<25} {f1:<12.4f} {roc:<12.4f} {acc:<12.4f}")

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_name = name

        except Exception as e:
            print(f"  {name:<25} FAILED: {e}")
            results[name] = {"error": str(e)}

    print(f"\n  🏆 Best model: {best_name} (F1={best_f1:.4f})")

    # Print detailed classification report for best model
    y_pred_best = best_model.predict(X_test)
    print(f"\n  Classification Report ({best_name}):")
    print(classification_report(y_test, y_pred_best))
    print(f"  Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_best)
    print(f"    {cm}")

    return best_model, best_name, results


def save_model_artifacts(disease_key, model, model_name, scaler, feature_cols, feature_metadata, X_train):
    """
    Save trained model, scaler, feature info, and training data sample.

    Args:
        disease_key: e.g., 'diabetes'
        model: Trained model object
        model_name: e.g., 'XGBoost'
        scaler: Fitted StandardScaler
        feature_cols: List of feature names
        feature_metadata: Dict with min/max/mean etc.
        X_train: Training data (for LIME/SHAP background)
    """
    # Save model
    model_path = os.path.join(config.MODELS_DIR, f"{disease_key}_model.pkl")
    joblib.dump(model, model_path)
    print(f"  Saved model → {model_path}")

    # Save scaler
    scaler_path = os.path.join(config.MODELS_DIR, f"{disease_key}_scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"  Saved scaler → {scaler_path}")

    # Save a sample of training data (for SHAP/LIME background)
    # Keep 100 samples max to reduce file size
    sample_size = min(100, X_train.shape[0])
    sample_indices = np.random.choice(X_train.shape[0], sample_size, replace=False)
    train_sample = X_train[sample_indices]
    sample_path = os.path.join(config.MODELS_DIR, f"{disease_key}_train_sample.pkl")
    joblib.dump(train_sample, sample_path)
    print(f"  Saved training sample → {sample_path}")

    # Save feature metadata as JSON
    meta = {
        "disease_key": disease_key,
        "model_name": model_name,
        "feature_names": feature_cols,
        "feature_metadata": feature_metadata,
        "display_name": config.DISEASES[disease_key]["display_name"],
        "positive_label": config.DISEASES[disease_key]["positive_label"],
        "negative_label": config.DISEASES[disease_key]["negative_label"],
    }
    meta_path = os.path.join(config.MODELS_DIR, f"{disease_key}_features.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved feature metadata → {meta_path}")


def train_disease_model(disease_key):
    """
    Full training pipeline for a single disease.

    Args:
        disease_key: Key from config.DISEASES

    Returns:
        dict: Training results summary
    """
    # Step 1: Load data
    df, feature_cols, target_col = load_dataset(disease_key)

    # Step 2: Preprocess
    X, y, scaler, feature_names, feature_metadata = preprocess_data(
        df, feature_cols, target_col
    )

    # Step 3: Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")

    # Step 4: Apply SMOTE on training data only
    X_train_balanced, y_train_balanced = apply_smote(X_train, y_train, disease_key)

    # Step 5: Train and compare models
    best_model, best_name, results = train_and_compare(
        X_train_balanced, y_train_balanced, X_test, y_test, disease_key
    )

    # Step 6: Save all artifacts
    save_model_artifacts(
        disease_key, best_model, best_name, scaler,
        feature_names, feature_metadata, X_train
    )

    return {
        "disease": disease_key,
        "best_model": best_name,
        "results": results,
    }


def main():
    """Train models for all configured diseases."""
    print("=" * 60)
    print("  Disease Prediction System — Model Training Pipeline")
    print("=" * 60)

    # First, generate datasets if they don't exist
    diabetes_csv = os.path.join(config.DATA_DIR, "diabetes.csv")
    if not os.path.exists(diabetes_csv):
        print("\n📊 Datasets not found. Generating...")
        # Import and run dataset generator
        sys.path.insert(0, os.path.join(config.BASE_DIR, "data"))
        from download_datasets import (
            create_diabetes_dataset,
            create_heart_dataset,
            create_kidney_dataset,
            create_liver_dataset,
        )
        create_diabetes_dataset()
        create_heart_dataset()
        create_kidney_dataset()
        create_liver_dataset()
        print("✅ Datasets generated.\n")

    # Train all disease models
    all_results = []
    for disease_key in config.DISEASES:
        try:
            result = train_disease_model(disease_key)
            all_results.append(result)
        except Exception as e:
            print(f"\n❌ Failed to train {disease_key}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("  TRAINING SUMMARY")
    print("=" * 60)
    for r in all_results:
        best = r["results"].get(r["best_model"], {})
        print(
            f"  {config.DISEASES[r['disease']]['display_name']:<20} "
            f"→ {r['best_model']:<25} "
            f"F1={best.get('f1', 0):.4f}  "
            f"AUC={best.get('roc_auc', 0):.4f}"
        )

    print(f"\n✅ All models saved to {config.MODELS_DIR}")
    print("   Run 'python app.py' to start the web application.")


if __name__ == "__main__":
    main()
