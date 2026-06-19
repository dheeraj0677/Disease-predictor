"""
Configuration constants for the Disease Prediction System.
All paths are constructed relative to the project root using os.path.
"""

import os

# ─── Project Root ────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── Data & Model Paths ─────────────────────────────────────────
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
DATA_DIR = STORAGE_DIR
MODELS_DIR = STORAGE_DIR
CHARTS_DIR = os.path.join(BASE_DIR, "static", "charts")
DATABASE_DIR = STORAGE_DIR
DATABASE_PATH = os.path.join(DATABASE_DIR, "predictions.db")

# ─── Supported Diseases ─────────────────────────────────────────
DISEASES = {
    "diabetes": {
        "display_name": "Diabetes",
        "dataset_file": "diabetes.csv",
        "target_column": "Outcome",
        "positive_label": "Diabetic",
        "negative_label": "Not Diabetic",
        "description": "Predicts Type 2 diabetes risk based on glucose, BMI, and other metabolic indicators.",
    },
    "heart": {
        "display_name": "Heart Disease",
        "dataset_file": "heart.csv",
        "target_column": "target",
        "positive_label": "Heart Disease",
        "negative_label": "No Heart Disease",
        "description": "Predicts coronary heart disease based on cholesterol, blood pressure, and cardiac markers.",
    },
    "kidney": {
        "display_name": "Kidney Disease",
        "dataset_file": "kidney.csv",
        "target_column": "classification",
        "positive_label": "Chronic Kidney Disease",
        "negative_label": "No Kidney Disease",
        "description": "Predicts chronic kidney disease using blood cell counts, albumin, and other renal indicators.",
    },
    "liver": {
        "display_name": "Liver Disease",
        "dataset_file": "liver.csv",
        "target_column": "Dataset",
        "positive_label": "Liver Disease",
        "negative_label": "No Liver Disease",
        "description": "Predicts liver disease based on bilirubin, enzyme levels, and protein ratios.",
    },
}

# ─── Risk Level Thresholds ──────────────────────────────────────
RISK_THRESHOLDS = {
    "low": 0.40,       # confidence < 40% → Low risk
    "medium": 0.70,    # 40% ≤ confidence < 70% → Medium risk
    # confidence ≥ 70% → High risk
}

# ─── Flask Settings ─────────────────────────────────────────────
SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")
DEBUG = os.environ.get("FLASK_DEBUG", "true").lower() == "true"

# ─── Ensure directories exist ───────────────────────────────────
for d in [STORAGE_DIR, CHARTS_DIR]:
    os.makedirs(d, exist_ok=True)
