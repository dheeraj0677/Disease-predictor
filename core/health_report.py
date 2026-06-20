"""
Health Report Generator Module.

Generates comprehensive health report data including:
- Normal range comparisons for each patient value
- Radar chart data for visualization
- Personalized health recommendations

Normal ranges are medically-sourced reference ranges used for
educational/demo purposes only.
"""


# ─── Normal Reference Ranges ────────────────────────────────────
# Each key maps a (disease, field_name) to normal range + display info.
# "low" and "high" define the normal range boundaries.
# "display_min" and "display_max" define the full scale for the bar chart.

NORMAL_RANGES = {
    # ── Diabetes ──────────────────────────────────────
    ("diabetes", "Pregnancies"): {
        "low": 0, "high": 6,
        "display_min": 0, "display_max": 17,
        "label": "Pregnancies",
    },
    ("diabetes", "Glucose"): {
        "low": 70, "high": 140,
        "display_min": 0, "display_max": 200,
        "label": "Glucose (mg/dL)",
    },
    ("diabetes", "BloodPressure"): {
        "low": 60, "high": 90,
        "display_min": 0, "display_max": 140,
        "label": "Blood Pressure (mm Hg)",
    },
    ("diabetes", "SkinThickness"): {
        "low": 10, "high": 40,
        "display_min": 0, "display_max": 100,
        "label": "Skin Thickness (mm)",
    },
    ("diabetes", "Insulin"): {
        "low": 16, "high": 166,
        "display_min": 0, "display_max": 900,
        "label": "Insulin (μU/mL)",
    },
    ("diabetes", "BMI"): {
        "low": 18.5, "high": 25.0,
        "display_min": 10, "display_max": 70,
        "label": "BMI (kg/m²)",
    },
    ("diabetes", "DiabetesPedigreeFunction"): {
        "low": 0.0, "high": 0.8,
        "display_min": 0, "display_max": 2.5,
        "label": "Diabetes Pedigree",
    },
    ("diabetes", "Age"): {
        "low": 18, "high": 65,
        "display_min": 1, "display_max": 120,
        "label": "Age (years)",
    },

    # ── Heart Disease ─────────────────────────────────
    ("heart", "age"): {
        "low": 18, "high": 65,
        "display_min": 1, "display_max": 120,
        "label": "Age (years)",
    },
    ("heart", "trestbps"): {
        "low": 90, "high": 130,
        "display_min": 80, "display_max": 220,
        "label": "Resting BP (mm Hg)",
    },
    ("heart", "chol"): {
        "low": 125, "high": 200,
        "display_min": 100, "display_max": 600,
        "label": "Cholesterol (mg/dL)",
    },
    ("heart", "thalach"): {
        "low": 100, "high": 180,
        "display_min": 60, "display_max": 220,
        "label": "Max Heart Rate",
    },
    ("heart", "oldpeak"): {
        "low": 0, "high": 1.5,
        "display_min": 0, "display_max": 7,
        "label": "ST Depression",
    },

    # ── Kidney Disease ────────────────────────────────
    ("kidney", "age"): {
        "low": 18, "high": 65,
        "display_min": 1, "display_max": 120,
        "label": "Age (years)",
    },
    ("kidney", "bp"): {
        "low": 60, "high": 90,
        "display_min": 40, "display_max": 200,
        "label": "Blood Pressure (mm Hg)",
    },
    ("kidney", "bgr"): {
        "low": 70, "high": 140,
        "display_min": 50, "display_max": 500,
        "label": "Blood Glucose (mg/dL)",
    },
    ("kidney", "bu"): {
        "low": 7, "high": 20,
        "display_min": 5, "display_max": 400,
        "label": "Blood Urea (mg/dL)",
    },
    ("kidney", "sc"): {
        "low": 0.6, "high": 1.2,
        "display_min": 0.3, "display_max": 20,
        "label": "Serum Creatinine (mg/dL)",
    },
    ("kidney", "sod"): {
        "low": 135, "high": 145,
        "display_min": 100, "display_max": 170,
        "label": "Sodium (mEq/L)",
    },
    ("kidney", "pot"): {
        "low": 3.5, "high": 5.0,
        "display_min": 2, "display_max": 50,
        "label": "Potassium (mEq/L)",
    },
    ("kidney", "hemo"): {
        "low": 12.0, "high": 17.0,
        "display_min": 3, "display_max": 18,
        "label": "Hemoglobin (g/dL)",
    },
    ("kidney", "pcv"): {
        "low": 36, "high": 48,
        "display_min": 5, "display_max": 55,
        "label": "Packed Cell Volume (%)",
    },
    ("kidney", "wc"): {
        "low": 4000, "high": 11000,
        "display_min": 2000, "display_max": 30000,
        "label": "WBC Count (/cumm)",
    },
    ("kidney", "rc"): {
        "low": 4.0, "high": 5.5,
        "display_min": 2, "display_max": 8,
        "label": "RBC Count (M/cmm)",
    },

    # ── Liver Disease ─────────────────────────────────
    ("liver", "Age"): {
        "low": 18, "high": 65,
        "display_min": 1, "display_max": 120,
        "label": "Age (years)",
    },
    ("liver", "Total_Bilirubin"): {
        "low": 0.1, "high": 1.2,
        "display_min": 0.1, "display_max": 80,
        "label": "Total Bilirubin (mg/dL)",
    },
    ("liver", "Direct_Bilirubin"): {
        "low": 0.0, "high": 0.3,
        "display_min": 0.1, "display_max": 20,
        "label": "Direct Bilirubin (mg/dL)",
    },
    ("liver", "Alkaline_Phosphotase"): {
        "low": 44, "high": 147,
        "display_min": 50, "display_max": 2500,
        "label": "Alk. Phosphatase (IU/L)",
    },
    ("liver", "Alamine_Aminotransferase"): {
        "low": 7, "high": 56,
        "display_min": 5, "display_max": 2500,
        "label": "ALT / SGPT (IU/L)",
    },
    ("liver", "Aspartate_Aminotransferase"): {
        "low": 10, "high": 40,
        "display_min": 5, "display_max": 5000,
        "label": "AST / SGOT (IU/L)",
    },
    ("liver", "Total_Protiens"): {
        "low": 6.0, "high": 8.3,
        "display_min": 2, "display_max": 10,
        "label": "Total Proteins (g/dL)",
    },
    ("liver", "Albumin"): {
        "low": 3.5, "high": 5.5,
        "display_min": 0.5, "display_max": 6,
        "label": "Albumin (g/dL)",
    },
    ("liver", "Albumin_and_Globulin_Ratio"): {
        "low": 1.0, "high": 2.5,
        "display_min": 0.1, "display_max": 3,
        "label": "A/G Ratio",
    },
}


# ─── Recommendations Database ───────────────────────────────────
# Recommendations keyed by (disease, field_name, status)
# where status is "high" or "low" relative to normal range.

RECOMMENDATIONS = {
    # ── Diabetes ──────────────────────────────────────
    ("diabetes", "Glucose", "high"): {
        "icon": "🩸",
        "title": "Elevated Blood Glucose",
        "description": "Your glucose level is above normal range. Consider reducing refined sugar and carbohydrate intake. Regular exercise can help improve insulin sensitivity. Monitor HbA1c levels periodically.",
        "severity": "high",
    },
    ("diabetes", "BMI", "high"): {
        "icon": "⚖️",
        "title": "High Body Mass Index",
        "description": "BMI above 25 indicates overweight. A balanced diet rich in vegetables, lean proteins, and regular physical activity (150 min/week) can help achieve a healthier weight.",
        "severity": "medium",
    },
    ("diabetes", "BloodPressure", "high"): {
        "icon": "💓",
        "title": "Elevated Blood Pressure",
        "description": "High blood pressure increases cardiovascular risk. Reduce sodium intake, manage stress, and consider the DASH diet. Regular monitoring is recommended.",
        "severity": "medium",
    },
    ("diabetes", "Insulin", "high"): {
        "icon": "💉",
        "title": "Elevated Insulin Level",
        "description": "High insulin may indicate insulin resistance. Focus on low-glycemic foods, increase fiber intake, and maintain regular exercise to improve insulin sensitivity.",
        "severity": "medium",
    },
    ("diabetes", "DiabetesPedigreeFunction", "high"): {
        "icon": "🧬",
        "title": "Family History Risk Factor",
        "description": "A high pedigree function indicates significant family history of diabetes. While genetic risk cannot be changed, lifestyle modifications can significantly reduce your risk.",
        "severity": "info",
    },

    # ── Heart Disease ─────────────────────────────────
    ("heart", "chol", "high"): {
        "icon": "🫀",
        "title": "High Cholesterol Level",
        "description": "Cholesterol above 200 mg/dL increases heart disease risk. Increase omega-3 fatty acids, reduce saturated fats, and consider adding soluble fiber to your diet.",
        "severity": "high",
    },
    ("heart", "trestbps", "high"): {
        "icon": "💓",
        "title": "Elevated Resting Blood Pressure",
        "description": "High resting blood pressure strains the heart. Limit sodium to 2,300 mg/day, exercise regularly, manage stress, and limit alcohol consumption.",
        "severity": "high",
    },
    ("heart", "thalach", "low"): {
        "icon": "🏃",
        "title": "Low Maximum Heart Rate",
        "description": "A lower-than-expected max heart rate during exercise may indicate reduced cardiovascular fitness. Gradual aerobic exercise under medical guidance can improve this.",
        "severity": "medium",
    },
    ("heart", "oldpeak", "high"): {
        "icon": "📈",
        "title": "Significant ST Depression",
        "description": "Elevated ST depression during exercise may indicate myocardial ischemia. Further cardiac evaluation such as stress testing or angiography may be recommended.",
        "severity": "high",
    },

    # ── Kidney Disease ────────────────────────────────
    ("kidney", "sc", "high"): {
        "icon": "🫘",
        "title": "Elevated Serum Creatinine",
        "description": "High creatinine may indicate impaired kidney function. Stay well-hydrated, reduce protein intake, avoid NSAIDs, and consult a nephrologist for further evaluation.",
        "severity": "high",
    },
    ("kidney", "bu", "high"): {
        "icon": "🧪",
        "title": "High Blood Urea",
        "description": "Elevated blood urea nitrogen can indicate kidney stress. Ensure adequate hydration, moderate protein intake, and regular kidney function monitoring.",
        "severity": "high",
    },
    ("kidney", "hemo", "low"): {
        "icon": "🩸",
        "title": "Low Hemoglobin — Anemia Risk",
        "description": "Low hemoglobin is common in kidney disease. Iron supplementation, erythropoiesis-stimulating agents, and iron-rich foods may be recommended by your doctor.",
        "severity": "medium",
    },
    ("kidney", "sod", "low"): {
        "icon": "🧂",
        "title": "Low Sodium Level",
        "description": "Hyponatremia may cause fatigue, nausea, and confusion. Monitor fluid intake and consult your doctor about adjusting fluid restriction if needed.",
        "severity": "medium",
    },
    ("kidney", "pot", "high"): {
        "icon": "⚡",
        "title": "Elevated Potassium",
        "description": "Hyperkalemia can affect heart rhythm. Limit high-potassium foods (bananas, oranges, potatoes) and consult your doctor about medication adjustments.",
        "severity": "high",
    },

    # ── Liver Disease ─────────────────────────────────
    ("liver", "Total_Bilirubin", "high"): {
        "icon": "🟡",
        "title": "Elevated Total Bilirubin",
        "description": "High bilirubin can indicate liver dysfunction or bile duct obstruction. Avoid alcohol, maintain a healthy diet, and get further hepatic evaluation.",
        "severity": "high",
    },
    ("liver", "Alamine_Aminotransferase", "high"): {
        "icon": "🔬",
        "title": "Elevated ALT / SGPT Enzyme",
        "description": "Elevated ALT indicates liver cell damage. Avoid alcohol and hepatotoxic substances, maintain healthy weight, and consider further liver function testing.",
        "severity": "high",
    },
    ("liver", "Aspartate_Aminotransferase", "high"): {
        "icon": "🧫",
        "title": "Elevated AST / SGOT Enzyme",
        "description": "High AST levels suggest liver or muscle damage. Combined with elevated ALT, this strongly suggests liver pathology. Further evaluation recommended.",
        "severity": "high",
    },
    ("liver", "Albumin", "low"): {
        "icon": "📉",
        "title": "Low Albumin Level",
        "description": "Low albumin may indicate chronic liver disease or malnutrition. Increase protein-rich foods, and consult your doctor for nutritional supplementation.",
        "severity": "medium",
    },
    ("liver", "Albumin_and_Globulin_Ratio", "low"): {
        "icon": "⚖️",
        "title": "Low Albumin/Globulin Ratio",
        "description": "A low A/G ratio may indicate liver disease, chronic inflammation, or kidney disease. Further testing including liver biopsy may be considered.",
        "severity": "medium",
    },
    ("liver", "Alkaline_Phosphotase", "high"): {
        "icon": "📊",
        "title": "Elevated Alkaline Phosphatase",
        "description": "High ALP can indicate liver or bone disease. Bile duct obstruction should be ruled out. An ultrasound or CT scan may be recommended.",
        "severity": "medium",
    },
}


def _clamp(val, lo, hi):
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, val))


def _pct_on_scale(value, display_min, display_max):
    """Convert a value to a percentage position on a display scale."""
    rng = display_max - display_min
    if rng == 0:
        return 50.0
    return _clamp(((value - display_min) / rng) * 100, 0, 100)


def generate_range_comparisons(disease_key, patient_data):
    """
    Compare each patient value to its normal range.

    Args:
        disease_key: str, e.g. 'diabetes'
        patient_data: dict of field_name → value

    Returns:
        list of dicts, each with:
            - label, value, status ('normal', 'warning', 'abnormal')
            - normal_low, normal_high
            - display_min, display_max
            - value_pct, normal_start_pct, normal_width_pct (for bar visualization)
    """
    comparisons = []

    for field_name, value in patient_data.items():
        key = (disease_key, field_name)
        if key not in NORMAL_RANGES:
            continue

        ref = NORMAL_RANGES[key]
        low = ref["low"]
        high = ref["high"]
        d_min = ref["display_min"]
        d_max = ref["display_max"]
        val = float(value)

        # Determine status
        margin = (high - low) * 0.15  # 15% margin for "warning" zone
        if low <= val <= high:
            status = "normal"
        elif (low - margin) <= val < low or high < val <= (high + margin):
            status = "warning"
        else:
            status = "abnormal"

        # Calculate percentages for bar chart
        value_pct = _pct_on_scale(val, d_min, d_max)
        normal_start_pct = _pct_on_scale(low, d_min, d_max)
        normal_end_pct = _pct_on_scale(high, d_min, d_max)
        normal_width_pct = normal_end_pct - normal_start_pct

        comparisons.append({
            "field_name": field_name,
            "label": ref["label"],
            "value": val,
            "status": status,
            "normal_low": low,
            "normal_high": high,
            "display_min": d_min,
            "display_max": d_max,
            "value_pct": round(value_pct, 1),
            "normal_start_pct": round(normal_start_pct, 1),
            "normal_width_pct": round(max(normal_width_pct, 1), 1),
        })

    return comparisons


def generate_radar_data(range_comparisons):
    """
    Generate radar chart data from range comparisons.

    Normalizes each patient value to 0-100 scale relative to display range,
    and creates a parallel "normal upper bound" line.

    Args:
        range_comparisons: list from generate_range_comparisons()

    Returns:
        dict with keys: labels, patient_values, normal_upper, point_colors
    """
    labels = []
    patient_values = []
    normal_upper = []
    point_colors = []

    color_map = {
        "normal": "rgba(16, 185, 129, 0.9)",
        "warning": "rgba(245, 158, 11, 0.9)",
        "abnormal": "rgba(239, 68, 68, 0.9)",
    }

    for comp in range_comparisons:
        labels.append(comp["label"])
        patient_values.append(comp["value_pct"])
        normal_upper.append(
            _pct_on_scale(comp["normal_high"], comp["display_min"], comp["display_max"])
        )
        point_colors.append(color_map.get(comp["status"], "rgba(59, 130, 246, 0.9)"))

    return {
        "labels": labels,
        "patient_values": patient_values,
        "normal_upper": [round(v, 1) for v in normal_upper],
        "point_colors": point_colors,
    }


def generate_recommendations(disease_key, patient_data, prediction_label, is_positive):
    """
    Generate personalized health recommendations based on the patient's values.

    Args:
        disease_key: str
        patient_data: dict
        prediction_label: str, the model's prediction label
        is_positive: bool, whether the prediction is positive for the disease

    Returns:
        list of recommendation dicts with: icon, title, description, severity
    """
    recs = []

    # Check each value against normal ranges
    for field_name, value in patient_data.items():
        key = (disease_key, field_name)
        if key not in NORMAL_RANGES:
            continue

        ref = NORMAL_RANGES[key]
        val = float(value)

        # Check if value is above normal
        if val > ref["high"]:
            rec_key = (disease_key, field_name, "high")
            if rec_key in RECOMMENDATIONS:
                recs.append(RECOMMENDATIONS[rec_key])

        # Check if value is below normal
        elif val < ref["low"]:
            rec_key = (disease_key, field_name, "low")
            if rec_key in RECOMMENDATIONS:
                recs.append(RECOMMENDATIONS[rec_key])

    # Add general recommendation based on overall prediction
    if is_positive:
        recs.append({
            "icon": "🏥",
            "title": "Schedule a Follow-Up",
            "description": f"The model predicts '{prediction_label}'. We strongly recommend scheduling a comprehensive consultation with a specialist for further diagnostic testing and personalized treatment planning.",
            "severity": "high",
        })
    else:
        recs.append({
            "icon": "✅",
            "title": "Maintain Healthy Lifestyle",
            "description": "Your results are encouraging! Continue with regular health checkups, balanced nutrition, adequate sleep, stress management, and at least 150 minutes of moderate exercise per week.",
            "severity": "low",
        })

    # Always add preventive care recommendation
    recs.append({
        "icon": "🩺",
        "title": "Regular Health Screening",
        "description": "Annual health screenings help catch potential issues early. Discuss appropriate screening tests with your healthcare provider based on your age, family history, and risk factors.",
        "severity": "info",
    })

    # Sort: high severity first, then medium, then low, then info
    severity_order = {"high": 0, "medium": 1, "low": 2, "info": 3}
    recs.sort(key=lambda r: severity_order.get(r["severity"], 4))

    return recs
