"""
LIME Explainability Module for the Disease Prediction System.

Uses LIME (Local Interpretable Model-agnostic Explanations) to explain predictions:
  - Generates per-prediction explanations showing top contributing features
  - Returns human-readable text like "Glucose 148 → increases diabetes risk by 32%"
  - Optionally saves a LIME visualization chart

LIME works by perturbing the input and observing how predictions change,
building a local linear approximation around the prediction.
"""

import os
import sys
import uuid
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Attempt to import LIME
try:
    from lime.lime_tabular import LimeTabularExplainer
    HAS_LIME = True
except ImportError:
    HAS_LIME = False
    print("⚠  LIME library not available. Text explanations will be limited.")


def generate_lime_explanation(disease_key, input_scaled, input_original, prediction_label):
    """
    Generate a LIME explanation for a single prediction.

    Args:
        disease_key: e.g., 'diabetes'
        input_scaled: Scaled feature values (1D numpy array)
        input_original: Original (unscaled) feature values (dict: feature_name → value)
        prediction_label: The prediction result string

    Returns:
        dict: {
            'text_explanations': list of human-readable strings,
            'chart_path': filename of saved chart (or None),
            'feature_weights': list of (feature, weight) tuples
        }
    """
    try:
        # Load model artifacts
        model = joblib.load(os.path.join(config.MODELS_DIR, f"{disease_key}_model.pkl"))
        train_sample = joblib.load(os.path.join(config.MODELS_DIR, f"{disease_key}_train_sample.pkl"))

        meta_path = os.path.join(config.MODELS_DIR, f"{disease_key}_features.json")
        with open(meta_path, "r") as f:
            feature_meta = json.load(f)

        feature_names = feature_meta["feature_names"]
        positive_label = feature_meta["positive_label"]
        display_name = feature_meta["display_name"]

        if not HAS_LIME:
            # Fallback: generate explanation from feature importances
            return _generate_fallback_explanation(
                model, feature_names, input_original, positive_label, display_name
            )

        # Create LIME explainer using training sample as background
        lime_explainer = LimeTabularExplainer(
            training_data=train_sample,
            feature_names=feature_names,
            class_names=[feature_meta["negative_label"], feature_meta["positive_label"]],
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )

        # Ensure input is 1D
        if input_scaled.ndim > 1:
            input_1d = input_scaled.flatten()
        else:
            input_1d = input_scaled

        # Generate explanation — explain positive class (index 1)
        explanation = lime_explainer.explain_instance(
            input_1d,
            model.predict_proba,
            num_features=min(8, len(feature_names)),
            top_labels=1,
        )

        # Extract feature weights
        # Get explanation for the predicted class
        try:
            exp_list = explanation.as_list(label=1)
        except Exception:
            exp_list = explanation.as_list()

        # Build human-readable explanations
        text_explanations = _build_readable_explanations(
            exp_list, feature_names, input_original, positive_label, display_name
        )

        # Build feature weights list for programmatic use
        feature_weights = [(feat, weight) for feat, weight in exp_list]

        # Generate and save LIME chart
        chart_filename = _save_lime_chart(
            explanation, disease_key, prediction_label, feature_names
        )

        return {
            "text_explanations": text_explanations,
            "chart_path": chart_filename,
            "feature_weights": feature_weights,
        }

    except Exception as e:
        print(f"  ⚠ LIME explanation failed: {e}")
        import traceback
        traceback.print_exc()

        # Return a basic fallback
        return {
            "text_explanations": [
                f"Explanation temporarily unavailable for this prediction."
            ],
            "chart_path": None,
            "feature_weights": [],
        }


def _build_readable_explanations(exp_list, feature_names, input_original, positive_label, display_name):
    """
    Convert LIME feature weights into human-readable text explanations.

    Args:
        exp_list: List of (feature_condition, weight) tuples from LIME
        feature_names: List of feature names
        input_original: Dict of original (unscaled) input values
        positive_label: e.g., "Diabetic"
        display_name: e.g., "Diabetes"

    Returns:
        list: Human-readable explanation strings
    """
    # Feature display names for medical context
    feature_display = {
        # Diabetes
        "Pregnancies": "Number of pregnancies",
        "Glucose": "Glucose level",
        "BloodPressure": "Blood pressure",
        "SkinThickness": "Skin thickness",
        "Insulin": "Insulin level",
        "BMI": "BMI",
        "DiabetesPedigreeFunction": "Diabetes pedigree function",
        "Age": "Age",
        # Heart
        "age": "Age",
        "sex": "Sex",
        "cp": "Chest pain type",
        "trestbps": "Resting blood pressure",
        "chol": "Cholesterol level",
        "fbs": "Fasting blood sugar",
        "restecg": "Resting ECG result",
        "thalach": "Max heart rate",
        "exang": "Exercise-induced angina",
        "oldpeak": "ST depression",
        "slope": "ST slope",
        "ca": "Number of major vessels",
        "thal": "Thalassemia",
        # Kidney
        "bp": "Blood pressure",
        "sg": "Specific gravity",
        "al": "Albumin",
        "su": "Sugar",
        "bgr": "Blood glucose random",
        "bu": "Blood urea",
        "sc": "Serum creatinine",
        "sod": "Sodium",
        "pot": "Potassium",
        "hemo": "Hemoglobin",
        "pcv": "Packed cell volume",
        "wc": "White blood cell count",
        "rc": "Red blood cell count",
        "htn": "Hypertension",
        "dm": "Diabetes mellitus",
        "appet": "Appetite",
        "ane": "Anemia",
        # Liver
        "Gender": "Gender",
        "Total_Bilirubin": "Total bilirubin",
        "Direct_Bilirubin": "Direct bilirubin",
        "Alkaline_Phosphotase": "Alkaline phosphatase",
        "Alamine_Aminotransferase": "ALT enzyme",
        "Aspartate_Aminotransferase": "AST enzyme",
        "Total_Protiens": "Total proteins",
        "Albumin": "Albumin",
        "Albumin_and_Globulin_Ratio": "A/G ratio",
        "Dataset": "Dataset",
    }

    explanations = []

    for feature_condition, weight in exp_list:
        # Parse the feature name from the LIME condition string
        # LIME conditions look like "Glucose > 120.50" or "3.50 < BMI <= 35.00"
        matched_feature = None
        for fname in feature_names:
            if fname in feature_condition:
                matched_feature = fname
                break

        if matched_feature is None:
            continue

        display = feature_display.get(matched_feature, matched_feature)
        original_val = input_original.get(matched_feature, "?")

        # Determine direction (increases or decreases risk)
        abs_weight = abs(weight)
        percentage = min(round(abs_weight * 100, 1), 99)  # Cap at 99% for readability

        if weight > 0:
            direction = "increases"
            arrow = "↑"
        else:
            direction = "decreases"
            arrow = "↓"

        # Build the explanation string
        risk_term = f"{positive_label.lower()} risk"
        explanation_text = (
            f"{display} = {original_val} {arrow} {direction} {risk_term} by {percentage}%"
        )
        explanations.append(explanation_text)

    # Sort by absolute impact (most important first)
    return explanations[:5]  # Return top 5


def _save_lime_chart(explanation, disease_key, prediction_label, feature_names):
    """
    Save LIME explanation as a horizontal bar chart.

    Args:
        explanation: LIME Explanation object
        disease_key: e.g., 'diabetes'
        prediction_label: e.g., 'Diabetic'
        feature_names: List of feature names

    Returns:
        str: Filename of saved chart, or None
    """
    try:
        # Get the LIME figure
        fig = explanation.as_pyplot_figure(label=1)
        fig.set_size_inches(10, 6)
        plt.title(
            f"LIME Explanation — {prediction_label}",
            fontsize=14, fontweight="bold", pad=20,
        )
        plt.tight_layout()

        chart_id = uuid.uuid4().hex[:8]
        filename = f"{disease_key}_{chart_id}_lime.png"
        filepath = os.path.join(config.CHARTS_DIR, filename)
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        print(f"  ✓ LIME chart saved → {filepath}")
        return filename

    except Exception as e:
        print(f"  ⚠ LIME chart save failed: {e}")
        return None


def _generate_fallback_explanation(model, feature_names, input_original, positive_label, display_name):
    """
    Generate a basic explanation when LIME is unavailable.
    Uses feature importances from the model.

    Args:
        model: Trained model
        feature_names: List of feature names
        input_original: Dict of original feature values
        positive_label: e.g., "Diabetic"
        display_name: e.g., "Diabetes"

    Returns:
        dict: Same format as generate_lime_explanation return value
    """
    explanations = []

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return {
            "text_explanations": ["Feature importance data unavailable."],
            "chart_path": None,
            "feature_weights": [],
        }

    # Get top 5 features by importance
    indices = np.argsort(importances)[::-1][:5]

    for idx in indices:
        fname = feature_names[idx]
        val = input_original.get(fname, "?")
        imp = round(importances[idx] * 100, 1)
        explanations.append(f"{fname} = {val} → contributes {imp}% to prediction")

    return {
        "text_explanations": explanations,
        "chart_path": None,
        "feature_weights": [(feature_names[i], float(importances[i])) for i in indices],
    }
