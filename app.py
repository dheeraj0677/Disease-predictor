"""
Flask Application — Disease Prediction System with Explainability.

Routes:
  GET  /             → Homepage with tabbed patient input forms
  POST /predict      → Run model + SHAP + LIME, save to DB, redirect to result
  GET  /result/<id>  → Show prediction result with explanations
  GET  /dashboard    → Doctor dashboard with filters and charts
  GET  /api/stats    → JSON stats endpoint
  POST /api/predict  → REST API for external integrations
  GET  /export/csv   → Export all predictions as CSV
  POST /api/whatif   → What-If Analysis endpoint
"""

import os
import sys
import json
import csv
import io
import traceback
import numpy as np
import joblib
from datetime import datetime
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
    jsonify,
    Response,
)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from database.db import init_db, save_prediction, get_prediction_by_id, get_all_predictions, get_stats
from explainability.shap_explain import generate_waterfall_plot, generate_summary_plot
from explainability.lime_explain import generate_lime_explanation

# ─── Flask App Setup ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = config.SECRET_KEY
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

# Initialize database on startup
with app.app_context():
    init_db()


# ─── Feature Definitions ────────────────────────────────────────
# Maps each disease to its input fields with metadata for the form
FEATURE_DEFINITIONS = {
    "diabetes": {
        "display_name": "Diabetes Prediction",
        "icon": "🩸",
        "fields": [
            {"name": "Pregnancies", "label": "Number of Pregnancies", "type": "number", "min": 0, "max": 17, "step": 1, "default": 1, "tooltip": "Total number of pregnancies"},
            {"name": "Glucose", "label": "Glucose Level (mg/dL)", "type": "number", "min": 0, "max": 200, "step": 1, "default": 110, "tooltip": "Plasma glucose concentration at 2 hours in an oral glucose tolerance test"},
            {"name": "BloodPressure", "label": "Blood Pressure (mm Hg)", "type": "number", "min": 0, "max": 140, "step": 1, "default": 70, "tooltip": "Diastolic blood pressure measurement"},
            {"name": "SkinThickness", "label": "Skin Thickness (mm)", "type": "number", "min": 0, "max": 100, "step": 1, "default": 20, "tooltip": "Triceps skin fold thickness"},
            {"name": "Insulin", "label": "Insulin (μU/mL)", "type": "number", "min": 0, "max": 900, "step": 1, "default": 80, "tooltip": "2-hour serum insulin level"},
            {"name": "BMI", "label": "BMI (kg/m²)", "type": "number", "min": 10, "max": 70, "step": 0.1, "default": 25.0, "tooltip": "Body Mass Index — weight(kg) / height(m)²"},
            {"name": "DiabetesPedigreeFunction", "label": "Diabetes Pedigree Function", "type": "number", "min": 0, "max": 2.5, "step": 0.001, "default": 0.5, "tooltip": "Diabetes hereditary risk score based on family history"},
            {"name": "Age", "label": "Age (years)", "type": "number", "min": 1, "max": 120, "step": 1, "default": 30, "tooltip": "Patient's age in years"},
        ],
    },
    "heart": {
        "display_name": "Heart Disease Prediction",
        "icon": "❤️",
        "fields": [
            {"name": "age", "label": "Age (years)", "type": "number", "min": 1, "max": 120, "step": 1, "default": 50, "tooltip": "Patient's age in years"},
            {"name": "sex", "label": "Sex", "type": "select", "options": [{"value": 1, "label": "Male"}, {"value": 0, "label": "Female"}], "default": 1, "tooltip": "Biological sex"},
            {"name": "cp", "label": "Chest Pain Type", "type": "select", "options": [{"value": 0, "label": "Typical Angina"}, {"value": 1, "label": "Atypical Angina"}, {"value": 2, "label": "Non-anginal Pain"}, {"value": 3, "label": "Asymptomatic"}], "default": 0, "tooltip": "Type of chest pain experienced"},
            {"name": "trestbps", "label": "Resting Blood Pressure (mm Hg)", "type": "number", "min": 80, "max": 220, "step": 1, "default": 130, "tooltip": "Resting blood pressure on admission"},
            {"name": "chol", "label": "Cholesterol (mg/dL)", "type": "number", "min": 100, "max": 600, "step": 1, "default": 240, "tooltip": "Serum cholesterol level"},
            {"name": "fbs", "label": "Fasting Blood Sugar > 120 mg/dL", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}], "default": 0, "tooltip": "Whether fasting blood sugar exceeds 120 mg/dL"},
            {"name": "restecg", "label": "Resting ECG Result", "type": "select", "options": [{"value": 0, "label": "Normal"}, {"value": 1, "label": "ST-T Abnormality"}, {"value": 2, "label": "Left Ventricular Hypertrophy"}], "default": 0, "tooltip": "Resting electrocardiographic results"},
            {"name": "thalach", "label": "Max Heart Rate", "type": "number", "min": 60, "max": 220, "step": 1, "default": 150, "tooltip": "Maximum heart rate achieved during exercise test"},
            {"name": "exang", "label": "Exercise-Induced Angina", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}], "default": 0, "tooltip": "Whether exercise induced chest pain (angina)"},
            {"name": "oldpeak", "label": "ST Depression", "type": "number", "min": 0, "max": 7, "step": 0.1, "default": 1.0, "tooltip": "ST segment depression induced by exercise relative to rest"},
            {"name": "slope", "label": "ST Slope", "type": "select", "options": [{"value": 0, "label": "Upsloping"}, {"value": 1, "label": "Flat"}, {"value": 2, "label": "Downsloping"}], "default": 1, "tooltip": "Slope of the peak exercise ST segment"},
            {"name": "ca", "label": "Major Vessels Colored", "type": "select", "options": [{"value": 0, "label": "0"}, {"value": 1, "label": "1"}, {"value": 2, "label": "2"}, {"value": 3, "label": "3"}], "default": 0, "tooltip": "Number of major vessels colored by fluoroscopy (0-3)"},
            {"name": "thal", "label": "Thalassemia", "type": "select", "options": [{"value": 0, "label": "Normal"}, {"value": 1, "label": "Fixed Defect"}, {"value": 2, "label": "Reversible Defect"}], "default": 0, "tooltip": "Thalassemia blood disorder type"},
        ],
    },
    "kidney": {
        "display_name": "Kidney Disease Prediction",
        "icon": "🫘",
        "fields": [
            {"name": "age", "label": "Age (years)", "type": "number", "min": 1, "max": 120, "step": 1, "default": 50, "tooltip": "Patient's age in years"},
            {"name": "bp", "label": "Blood Pressure (mm Hg)", "type": "number", "min": 40, "max": 200, "step": 1, "default": 75, "tooltip": "Diastolic blood pressure"},
            {"name": "sg", "label": "Specific Gravity", "type": "select", "options": [{"value": 1.005, "label": "1.005"}, {"value": 1.010, "label": "1.010"}, {"value": 1.015, "label": "1.015"}, {"value": 1.020, "label": "1.020"}, {"value": 1.025, "label": "1.025"}], "default": 1.020, "tooltip": "Urine specific gravity — measures kidney's ability to concentrate urine"},
            {"name": "al", "label": "Albumin Level", "type": "select", "options": [{"value": 0, "label": "0 (None)"}, {"value": 1, "label": "1"}, {"value": 2, "label": "2"}, {"value": 3, "label": "3"}, {"value": 4, "label": "4"}, {"value": 5, "label": "5 (High)"}], "default": 0, "tooltip": "Albumin in urine (0-5 scale)"},
            {"name": "su", "label": "Sugar Level", "type": "select", "options": [{"value": 0, "label": "0 (None)"}, {"value": 1, "label": "1"}, {"value": 2, "label": "2"}, {"value": 3, "label": "3"}, {"value": 4, "label": "4"}, {"value": 5, "label": "5 (High)"}], "default": 0, "tooltip": "Sugar in urine (0-5 scale)"},
            {"name": "bgr", "label": "Blood Glucose Random (mg/dL)", "type": "number", "min": 50, "max": 500, "step": 1, "default": 120, "tooltip": "Random blood glucose level"},
            {"name": "bu", "label": "Blood Urea (mg/dL)", "type": "number", "min": 5, "max": 400, "step": 0.1, "default": 40, "tooltip": "Blood urea nitrogen level"},
            {"name": "sc", "label": "Serum Creatinine (mg/dL)", "type": "number", "min": 0.3, "max": 20, "step": 0.1, "default": 1.2, "tooltip": "Serum creatinine — key kidney function marker"},
            {"name": "sod", "label": "Sodium (mEq/L)", "type": "number", "min": 100, "max": 170, "step": 1, "default": 138, "tooltip": "Blood sodium level"},
            {"name": "pot", "label": "Potassium (mEq/L)", "type": "number", "min": 2, "max": 50, "step": 0.1, "default": 4.5, "tooltip": "Blood potassium level"},
            {"name": "hemo", "label": "Hemoglobin (g/dL)", "type": "number", "min": 3, "max": 18, "step": 0.1, "default": 13.0, "tooltip": "Hemoglobin level in blood"},
            {"name": "pcv", "label": "Packed Cell Volume (%)", "type": "number", "min": 5, "max": 55, "step": 1, "default": 40, "tooltip": "Volume percentage of red blood cells"},
            {"name": "wc", "label": "White Blood Cell Count (/cumm)", "type": "number", "min": 2000, "max": 30000, "step": 100, "default": 8000, "tooltip": "White blood cell count"},
            {"name": "rc", "label": "Red Blood Cell Count (millions/cmm)", "type": "number", "min": 2, "max": 8, "step": 0.1, "default": 4.8, "tooltip": "Red blood cell count"},
            {"name": "htn", "label": "Hypertension", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}], "default": 0, "tooltip": "Whether patient has hypertension"},
            {"name": "dm", "label": "Diabetes Mellitus", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}], "default": 0, "tooltip": "Whether patient has diabetes"},
            {"name": "appet", "label": "Appetite", "type": "select", "options": [{"value": 1, "label": "Good"}, {"value": 0, "label": "Poor"}], "default": 1, "tooltip": "Patient's appetite status"},
            {"name": "ane", "label": "Anemia", "type": "select", "options": [{"value": 0, "label": "No"}, {"value": 1, "label": "Yes"}], "default": 0, "tooltip": "Whether patient has anemia"},
        ],
    },
    "liver": {
        "display_name": "Liver Disease Prediction",
        "icon": "🫀",
        "fields": [
            {"name": "Age", "label": "Age (years)", "type": "number", "min": 1, "max": 120, "step": 1, "default": 40, "tooltip": "Patient's age in years"},
            {"name": "Gender", "label": "Gender", "type": "select", "options": [{"value": 1, "label": "Male"}, {"value": 0, "label": "Female"}], "default": 1, "tooltip": "Biological gender"},
            {"name": "Total_Bilirubin", "label": "Total Bilirubin (mg/dL)", "type": "number", "min": 0.1, "max": 80, "step": 0.1, "default": 1.0, "tooltip": "Total bilirubin level — elevated in liver damage"},
            {"name": "Direct_Bilirubin", "label": "Direct Bilirubin (mg/dL)", "type": "number", "min": 0.1, "max": 20, "step": 0.1, "default": 0.3, "tooltip": "Direct (conjugated) bilirubin level"},
            {"name": "Alkaline_Phosphotase", "label": "Alkaline Phosphatase (IU/L)", "type": "number", "min": 50, "max": 2500, "step": 1, "default": 200, "tooltip": "Alkaline phosphatase enzyme level"},
            {"name": "Alamine_Aminotransferase", "label": "ALT / SGPT (IU/L)", "type": "number", "min": 5, "max": 2500, "step": 1, "default": 25, "tooltip": "Alanine Aminotransferase — liver enzyme (ALT/SGPT)"},
            {"name": "Aspartate_Aminotransferase", "label": "AST / SGOT (IU/L)", "type": "number", "min": 5, "max": 5000, "step": 1, "default": 30, "tooltip": "Aspartate Aminotransferase — liver enzyme (AST/SGOT)"},
            {"name": "Total_Protiens", "label": "Total Proteins (g/dL)", "type": "number", "min": 2, "max": 10, "step": 0.1, "default": 6.5, "tooltip": "Total protein level in blood"},
            {"name": "Albumin", "label": "Albumin (g/dL)", "type": "number", "min": 0.5, "max": 6, "step": 0.1, "default": 3.2, "tooltip": "Albumin level — low in liver disease"},
            {"name": "Albumin_and_Globulin_Ratio", "label": "A/G Ratio", "type": "number", "min": 0.1, "max": 3, "step": 0.01, "default": 0.9, "tooltip": "Albumin to Globulin ratio — measures liver function"},
        ],
    },
}


def get_risk_level(confidence):
    """
    Determine risk level based on prediction confidence.

    Args:
        confidence: float between 0.0 and 1.0

    Returns:
        str: 'Low', 'Medium', or 'High'
    """
    if confidence < config.RISK_THRESHOLDS["low"]:
        return "Low"
    elif confidence < config.RISK_THRESHOLDS["medium"]:
        return "Medium"
    else:
        return "High"


def extract_age_from_form(disease_key, form_data):
    """
    Extract the patient age from form data regardless of the field name.

    Args:
        disease_key: e.g., 'diabetes'
        form_data: dict of form values

    Returns:
        int or None
    """
    # Different datasets use different names for age
    age_fields = ["Age", "age"]
    for field in age_fields:
        if field in form_data:
            try:
                return int(float(form_data[field]))
            except (ValueError, TypeError):
                pass
    return None


# ─── Routes ──────────────────────────────────────────────────────

@app.route("/")
def index():
    """Homepage — tabbed patient input form for all diseases."""
    return render_template(
        "index.html",
        diseases=config.DISEASES,
        feature_definitions=FEATURE_DEFINITIONS,
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Run a disease prediction:
    1. Parse and validate form inputs
    2. Load model + scaler
    3. Scale inputs and run predict_proba()
    4. Generate SHAP waterfall chart
    5. Generate LIME text explanation
    6. Save everything to database
    7. Redirect to result page
    """
    try:
        # Get disease type from form
        disease_key = request.form.get("disease_type", "").strip().lower()
        if disease_key not in config.DISEASES:
            flash(f"Unknown disease type: {disease_key}", "error")
            return redirect(url_for("index"))

        disease_info = config.DISEASES[disease_key]
        fields = FEATURE_DEFINITIONS[disease_key]["fields"]

        # ── Parse form inputs ────────────────────────────────
        input_original = {}  # Original values (for display and LIME)
        input_values = []    # Ordered values for model input

        for field in fields:
            raw_val = request.form.get(field["name"], "").strip()
            if raw_val == "":
                raw_val = str(field.get("default", 0))

            try:
                val = float(raw_val)
            except ValueError:
                flash(f"Invalid value for {field['label']}: {raw_val}", "error")
                return redirect(url_for("index"))

            input_original[field["name"]] = val
            input_values.append(val)

        # ── Load model and scaler ────────────────────────────
        model_path = os.path.join(config.MODELS_DIR, f"{disease_key}_model.pkl")
        scaler_path = os.path.join(config.MODELS_DIR, f"{disease_key}_scaler.pkl")

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            flash(
                f"Model for {disease_info['display_name']} not found. "
                f"Run 'python models/train.py' first.",
                "error",
            )
            return redirect(url_for("index"))

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        # ── Scale inputs and predict ─────────────────────────
        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        # Get prediction and probability
        prediction_idx = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]

        # Positive class probability (index 1)
        confidence = float(probabilities[1])

        # Determine prediction label
        if prediction_idx == 1:
            prediction_label = disease_info["positive_label"]
        else:
            prediction_label = disease_info["negative_label"]

        risk_level = get_risk_level(confidence)

        # ── Generate SHAP explanation ────────────────────────
        shap_chart = None
        try:
            shap_chart = generate_waterfall_plot(
                disease_key, input_scaled[0], prediction_label
            )
        except Exception as e:
            print(f"SHAP generation failed: {e}")
            traceback.print_exc()

        # ── Generate LIME explanation ────────────────────────
        lime_result = {"text_explanations": [], "chart_path": None}
        try:
            lime_result = generate_lime_explanation(
                disease_key, input_scaled[0], input_original, prediction_label
            )
        except Exception as e:
            print(f"LIME generation failed: {e}")
            traceback.print_exc()

        # ── Save to database ─────────────────────────────────
        patient_age = extract_age_from_form(disease_key, input_original)

        pred_data = {
            "disease_type": disease_key,
            "patient_age": patient_age,
            "patient_data": input_original,
            "prediction": prediction_label,
            "confidence": confidence,
            "risk_level": risk_level,
            "shap_chart_path": shap_chart,
            "lime_chart_path": lime_result.get("chart_path"),
            "lime_text": lime_result.get("text_explanations", []),
        }

        pred_id = save_prediction(pred_data)

        # Redirect to result page
        return redirect(url_for("result", pred_id=pred_id))

    except Exception as e:
        print(f"Prediction error: {e}")
        traceback.print_exc()
        flash(f"An error occurred during prediction: {str(e)}", "error")
        return redirect(url_for("index"))


@app.route("/result/<int:pred_id>")
def result(pred_id):
    """Show prediction result with SHAP chart and LIME explanations."""
    prediction = get_prediction_by_id(pred_id)

    if prediction is None:
        flash("Prediction not found.", "error")
        return redirect(url_for("index"))

    # Get disease display info
    disease_key = prediction["disease_type"]
    disease_info = config.DISEASES.get(disease_key, {})

    return render_template(
        "result.html",
        prediction=prediction,
        disease_info=disease_info,
        feature_definitions=FEATURE_DEFINITIONS.get(disease_key, {}),
    )


@app.route("/dashboard")
def dashboard():
    """Doctor dashboard with filters, stats, and prediction history."""
    # Get filter parameters
    filters = {}
    disease_filter = request.args.get("disease_type", "")
    prediction_filter = request.args.get("prediction", "")
    date_from = request.args.get("date_from", "")
    date_to = request.args.get("date_to", "")

    if disease_filter:
        filters["disease_type"] = disease_filter
    if prediction_filter:
        filters["prediction"] = prediction_filter
    if date_from:
        filters["date_from"] = date_from
    if date_to:
        filters["date_to"] = date_to

    predictions = get_all_predictions(filters if filters else None)
    stats = get_stats()

    return render_template(
        "dashboard.html",
        predictions=predictions,
        stats=stats,
        diseases=config.DISEASES,
        filters={
            "disease_type": disease_filter,
            "prediction": prediction_filter,
            "date_from": date_from,
            "date_to": date_to,
        },
    )


@app.route("/api/stats")
def api_stats():
    """REST API endpoint for dashboard statistics."""
    try:
        stats = get_stats()
        return jsonify({"success": True, "data": stats})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    REST API endpoint for external integrations.
    Accepts JSON body and returns prediction as JSON.

    Expected JSON body:
    {
        "disease": "diabetes",
        "features": {
            "Glucose": 148,
            "BMI": 33.6,
            ...
        }
    }
    """
    try:
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No JSON body provided"}), 400

        disease_key = data.get("disease", "").strip().lower()
        if disease_key not in config.DISEASES:
            return jsonify({"success": False, "error": f"Unknown disease: {disease_key}"}), 400

        features = data.get("features", {})
        if not features:
            return jsonify({"success": False, "error": "No features provided"}), 400

        disease_info = config.DISEASES[disease_key]
        fields = FEATURE_DEFINITIONS[disease_key]["fields"]

        # Build input array in correct order
        input_values = []
        input_original = {}
        for field in fields:
            val = features.get(field["name"], field.get("default", 0))
            input_values.append(float(val))
            input_original[field["name"]] = float(val)

        # Load model and predict
        model = joblib.load(os.path.join(config.MODELS_DIR, f"{disease_key}_model.pkl"))
        scaler = joblib.load(os.path.join(config.MODELS_DIR, f"{disease_key}_scaler.pkl"))

        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction_idx = model.predict(input_scaled)[0]
        probabilities = model.predict_proba(input_scaled)[0]
        confidence = float(probabilities[1])

        if prediction_idx == 1:
            prediction_label = disease_info["positive_label"]
        else:
            prediction_label = disease_info["negative_label"]

        risk_level = get_risk_level(confidence)

        # Generate LIME explanation
        lime_result = generate_lime_explanation(
            disease_key, input_scaled[0], input_original, prediction_label
        )

        return jsonify({
            "success": True,
            "prediction": prediction_label,
            "confidence": round(confidence, 4),
            "risk_level": risk_level,
            "explanations": lime_result.get("text_explanations", []),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/whatif", methods=["POST"])
def api_whatif():
    """
    What-If Analysis endpoint.
    Doctor modifies feature values and sees how the prediction changes.

    Expected JSON body:
    {
        "disease": "diabetes",
        "features": { ... modified features ... }
    }
    """
    try:
        data = request.get_json()
        disease_key = data.get("disease", "").strip().lower()
        features = data.get("features", {})

        if disease_key not in config.DISEASES:
            return jsonify({"success": False, "error": "Unknown disease"}), 400

        disease_info = config.DISEASES[disease_key]
        fields = FEATURE_DEFINITIONS[disease_key]["fields"]

        # Build input
        input_values = []
        for field in fields:
            val = features.get(field["name"], field.get("default", 0))
            input_values.append(float(val))

        model = joblib.load(os.path.join(config.MODELS_DIR, f"{disease_key}_model.pkl"))
        scaler = joblib.load(os.path.join(config.MODELS_DIR, f"{disease_key}_scaler.pkl"))

        input_array = np.array(input_values).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        probabilities = model.predict_proba(input_scaled)[0]
        confidence = float(probabilities[1])
        prediction_idx = int(np.argmax(probabilities))

        if prediction_idx == 1:
            prediction_label = disease_info["positive_label"]
        else:
            prediction_label = disease_info["negative_label"]

        return jsonify({
            "success": True,
            "prediction": prediction_label,
            "confidence": round(confidence, 4),
            "risk_level": get_risk_level(confidence),
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/export/csv")
def export_csv():
    """Export all predictions as a CSV download."""
    try:
        predictions = get_all_predictions()

        # Create CSV in memory
        output = io.StringIO()
        writer = csv.writer(output)

        # Header
        writer.writerow([
            "ID", "Timestamp", "Disease Type", "Patient Age",
            "Prediction", "Confidence", "Risk Level", "Patient Data",
        ])

        # Rows
        for p in predictions:
            writer.writerow([
                p["id"],
                p["timestamp"],
                p["disease_type"],
                p["patient_age"],
                p["prediction"],
                f"{p['confidence']:.4f}",
                p["risk_level"],
                json.dumps(p.get("patient_data", {})),
            ])

        # Return as downloadable CSV
        csv_content = output.getvalue()
        return Response(
            csv_content,
            mimetype="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            },
        )

    except Exception as e:
        flash(f"Export failed: {str(e)}", "error")
        return redirect(url_for("dashboard"))


# ─── Error Handlers ──────────────────────────────────────────────

@app.errorhandler(404)
def page_not_found(e):
    """Handle 404 errors."""
    flash("Page not found.", "error")
    return redirect(url_for("index"))


@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors."""
    flash("An internal error occurred. Please try again.", "error")
    return redirect(url_for("index"))


# ─── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Disease Prediction System")
    print("  http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(debug=config.DEBUG, host="0.0.0.0", port=5000)
