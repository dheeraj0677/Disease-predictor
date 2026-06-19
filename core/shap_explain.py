"""
SHAP Explainability Module for the Disease Prediction System.

Uses SHAP (SHapley Additive exPlanations) to explain individual predictions:
  - Waterfall plot: shows each feature's contribution to a single prediction
  - Summary (beeswarm) plot: global feature importance across all training data

Supports TreeExplainer (for XGBoost/RF) and LinearExplainer (for LogReg).
"""

import os
import sys
import uuid
import joblib
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Attempt to import shap — graceful fallback if unavailable
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print("⚠  SHAP library not available. Explanations will be limited.")


def load_model_artifacts(disease_key):
    """
    Load the trained model, scaler, feature metadata, and training sample.

    Args:
        disease_key: e.g., 'diabetes'

    Returns:
        tuple: (model, scaler, feature_meta, train_sample)
    """
    model = joblib.load(os.path.join(config.MODELS_DIR, f"{disease_key}_model.pkl"))
    scaler = joblib.load(os.path.join(config.MODELS_DIR, f"{disease_key}_scaler.pkl"))
    train_sample = joblib.load(os.path.join(config.MODELS_DIR, f"{disease_key}_train_sample.pkl"))

    meta_path = os.path.join(config.MODELS_DIR, f"{disease_key}_features.json")
    with open(meta_path, "r") as f:
        feature_meta = json.load(f)

    return model, scaler, feature_meta, train_sample


def get_explainer(model, train_sample, model_name=None):
    """
    Create the appropriate SHAP explainer based on model type.

    Args:
        model: Trained model
        train_sample: Background data sample for the explainer
        model_name: Name of the model (for type detection)

    Returns:
        shap.Explainer instance
    """
    if not HAS_SHAP:
        return None

    model_type = type(model).__name__

    # Use TreeExplainer for tree-based models
    if model_type in ["XGBClassifier", "RandomForestClassifier", "GradientBoostingClassifier"]:
        try:
            return shap.TreeExplainer(model)
        except Exception:
            # Fallback to KernelExplainer if TreeExplainer fails
            predict_fn = lambda x: model.predict_proba(x)
            return shap.KernelExplainer(predict_fn, shap.sample(train_sample, 50))

    # Use LinearExplainer for linear models
    elif model_type in ["LogisticRegression"]:
        try:
            return shap.LinearExplainer(model, train_sample)
        except Exception:
            predict_fn = lambda x: model.predict_proba(x)
            return shap.KernelExplainer(predict_fn, shap.sample(train_sample, 50))

    # Fallback: KernelExplainer works with any model
    else:
        predict_fn = lambda x: model.predict_proba(x)
        return shap.KernelExplainer(predict_fn, shap.sample(train_sample, 50))


def generate_waterfall_plot(disease_key, input_scaled, prediction_label):
    """
    Generate a SHAP waterfall plot for a single prediction.

    The waterfall plot shows how each feature pushes the prediction
    from the base value (average prediction) toward the actual prediction.

    Args:
        disease_key: e.g., 'diabetes'
        input_scaled: Scaled input features as 1D numpy array
        prediction_label: The prediction result string (for the title)

    Returns:
        str: File path to the saved PNG chart, or None if failed
    """
    if not HAS_SHAP:
        return _generate_fallback_chart(disease_key, input_scaled, "waterfall")

    try:
        model, scaler, feature_meta, train_sample = load_model_artifacts(disease_key)
        explainer = get_explainer(model, train_sample, feature_meta.get("model_name"))

        if explainer is None:
            return None

        # Ensure input is 2D
        if input_scaled.ndim == 1:
            input_2d = input_scaled.reshape(1, -1)
        else:
            input_2d = input_scaled

        # Calculate SHAP values
        shap_values = explainer.shap_values(input_2d)

        # Handle multi-output (binary classification returns list of 2 arrays)
        if isinstance(shap_values, list):
            # Use the positive class (index 1) SHAP values
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        elif shap_values.ndim == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        # Get base value
        if hasattr(explainer, 'expected_value'):
            base_val = explainer.expected_value
            if isinstance(base_val, (list, np.ndarray)):
                base_val = base_val[1] if len(base_val) > 1 else base_val[0]
        else:
            base_val = 0.5

        feature_names = feature_meta["feature_names"]

        # Create SHAP Explanation object for waterfall plot
        explanation = shap.Explanation(
            values=sv,
            base_values=float(base_val),
            data=input_2d[0],
            feature_names=feature_names,
        )

        # Generate waterfall plot
        fig, ax = plt.subplots(figsize=(10, 6))
        plt.sca(ax)
        shap.plots.waterfall(explanation, max_display=10, show=False)
        plt.title(f"SHAP Explanation — {prediction_label}", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()

        # Save chart
        chart_id = uuid.uuid4().hex[:8]
        filename = f"{disease_key}_{chart_id}_waterfall.png"
        filepath = os.path.join(config.CHARTS_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print(f"  ✓ SHAP waterfall chart saved → {filepath}")
        return filename

    except Exception as e:
        print(f"  ⚠ SHAP waterfall generation failed: {e}")
        import traceback
        traceback.print_exc()
        return _generate_fallback_chart(disease_key, input_scaled, "waterfall")


def generate_summary_plot(disease_key):
    """
    Generate a SHAP summary (beeswarm) plot showing global feature importance.

    Args:
        disease_key: e.g., 'diabetes'

    Returns:
        str: File path to saved PNG, or None
    """
    if not HAS_SHAP:
        return None

    try:
        model, scaler, feature_meta, train_sample = load_model_artifacts(disease_key)
        explainer = get_explainer(model, train_sample, feature_meta.get("model_name"))

        if explainer is None:
            return None

        # Calculate SHAP values for entire training sample
        shap_values = explainer.shap_values(train_sample)

        # Handle multi-output
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        elif shap_values.ndim == 3:
            sv = shap_values[:, :, 1]
        else:
            sv = shap_values

        feature_names = feature_meta["feature_names"]

        # Generate summary plot
        fig, ax = plt.subplots(figsize=(10, 7))
        plt.sca(ax)
        shap.summary_plot(
            sv, train_sample,
            feature_names=feature_names,
            show=False,
            max_display=15,
        )
        display_name = config.DISEASES[disease_key]["display_name"]
        plt.title(f"Global Feature Importance — {display_name}", fontsize=14, fontweight="bold", pad=20)
        plt.tight_layout()

        # Save
        chart_id = uuid.uuid4().hex[:8]
        filename = f"{disease_key}_{chart_id}_summary.png"
        filepath = os.path.join(config.CHARTS_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print(f"  ✓ SHAP summary chart saved → {filepath}")
        return filename

    except Exception as e:
        print(f"  ⚠ SHAP summary generation failed: {e}")
        return None


def _generate_fallback_chart(disease_key, input_scaled, chart_type):
    """
    Generate a simple bar chart as a fallback when SHAP is unavailable.
    Uses the model's feature_importances_ attribute (available for RF/XGB).

    Args:
        disease_key: e.g., 'diabetes'
        input_scaled: Scaled input features
        chart_type: 'waterfall' or 'summary'

    Returns:
        str: File path to saved PNG, or None
    """
    try:
        model = joblib.load(os.path.join(config.MODELS_DIR, f"{disease_key}_model.pkl"))
        meta_path = os.path.join(config.MODELS_DIR, f"{disease_key}_features.json")
        with open(meta_path, "r") as f:
            feature_meta = json.load(f)

        feature_names = feature_meta["feature_names"]

        # Try to get feature importances
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return None

        # Sort by importance
        indices = np.argsort(importances)[::-1][:10]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]

        # Create horizontal bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))
        ax.barh(range(len(top_features)), top_importances[::-1], color=colors[::-1])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features[::-1])
        ax.set_xlabel("Feature Importance")
        display_name = config.DISEASES[disease_key]["display_name"]
        ax.set_title(f"Feature Importance — {display_name}", fontsize=14, fontweight="bold")
        plt.tight_layout()

        chart_id = uuid.uuid4().hex[:8]
        filename = f"{disease_key}_{chart_id}_{chart_type}.png"
        filepath = os.path.join(config.CHARTS_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        return filename

    except Exception as e:
        print(f"  ⚠ Fallback chart generation failed: {e}")
        return None
