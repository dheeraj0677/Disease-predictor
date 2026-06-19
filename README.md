# 🧬 MedPredict AI — Disease Prediction System with Explainability

A production-ready web application that uses machine learning to predict diseases and provides **transparent, explainable predictions** using SHAP and LIME. Built for doctors and healthcare professionals.

---

## ✨ Features

- [x] **4 Disease Prediction Models** — Diabetes, Heart Disease, Kidney Disease, Liver Disease
- [x] **SHAP Explainability** — Waterfall plots showing each feature's contribution to the prediction
- [x] **LIME Explanations** — Human-readable text explanations (e.g., "Glucose 148 → increases risk by 32%")
- [x] **Confidence Scoring** — Probability percentage with animated circular progress bar
- [x] **Risk Level Assessment** — Low / Medium / High risk classification
- [x] **Doctor Dashboard** — Historical predictions, Plotly charts, filterable table
- [x] **What-If Analysis** — Adjust feature values and see real-time prediction changes
- [x] **REST API** — JSON endpoints for hospital system integration
- [x] **CSV Export** — Download all prediction data
- [x] **Dark Medical Theme** — Premium glassmorphism UI with micro-animations

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Backend | Python, Flask |
| ML Models | Scikit-learn, XGBoost |
| Explainability | SHAP, LIME |
| Database | SQLite |
| Frontend | HTML5, CSS3, JavaScript |
| Charts | Plotly.js, Matplotlib |
| Data Processing | Pandas, NumPy |
| Class Balancing | SMOTE (imbalanced-learn) |

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/disease-predictor.git
cd disease-predictor
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Generate Datasets & Train Models
```bash
python data/download_datasets.py
python models/train.py
```

### 5. Run the Application
```bash
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 📊 How Explainability Works

### SHAP (SHapley Additive exPlanations)
SHAP uses game theory (Shapley values) to explain individual predictions. For each prediction, it calculates how much each feature contributes to pushing the prediction away from the average. The waterfall plot visualizes this: red bars indicate features that increase disease risk, while blue bars indicate features that decrease it. This gives doctors a transparent view of *why* the model made its prediction, not just *what* it predicted.

### LIME (Local Interpretable Model-agnostic Explanations)
LIME works by creating slight variations of the patient's data and observing how the prediction changes. It builds a simple, interpretable model around the specific prediction to identify which features matter most locally. The result is human-readable explanations like "Glucose level 148 → increases diabetes risk by 32%", making it easy for non-technical medical staff to understand the model's reasoning.

---

## 📁 Project Structure

```
disease-predictor/
├── app.py                  # Flask application & routes
├── config.py               # Central configuration
├── models/
│   ├── train.py            # Training pipeline for all diseases
│   ├── *_model.pkl         # Trained models (generated)
│   ├── *_scaler.pkl        # Feature scalers (generated)
│   └── *_features.json     # Feature metadata (generated)
├── data/
│   ├── download_datasets.py # Dataset generator
│   └── *.csv               # Disease datasets (generated)
├── explainability/
│   ├── shap_explain.py     # SHAP waterfall & summary plots
│   └── lime_explain.py     # LIME text & chart explanations
├── database/
│   ├── db.py               # SQLite helper functions
│   └── predictions.db      # Database (generated)
├── static/
│   ├── css/style.css       # Design system
│   ├── js/app.js           # Frontend interactivity
│   └── charts/             # Generated SHAP/LIME charts
├── templates/
│   ├── base.html           # Base template
│   ├── index.html          # Patient input form
│   ├── result.html         # Prediction results
│   └── dashboard.html      # Doctor dashboard
├── requirements.txt
└── README.md
```

---

## 📡 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Homepage with prediction forms |
| `/predict` | POST | Submit prediction (form data) |
| `/result/<id>` | GET | View prediction result |
| `/dashboard` | GET | Doctor dashboard |
| `/api/stats` | GET | JSON statistics |
| `/api/predict` | POST | REST API prediction (JSON) |
| `/api/whatif` | POST | What-If Analysis |
| `/export/csv` | GET | Download CSV export |

---

## 📚 Dataset Sources

| Disease | Dataset | Samples | Features |
|---------|---------|---------|----------|
| Diabetes | Pima Indians Diabetes | 768 | 8 |
| Heart Disease | Cleveland Heart Disease (UCI) | 303 | 13 |
| Kidney Disease | Chronic Kidney Disease (UCI) | 400 | 18 |
| Liver Disease | Indian Liver Patient (UCI) | 583 | 10 |

---

## 🔮 Future Improvements

- [ ] PDF report generation with ReportLab
- [ ] Doctor authentication with Flask-Login
- [ ] Patient risk score combining multiple disease predictions
- [ ] Time-series tracking for recurring patients
- [ ] Deployment to Render / Railway with Docker
- [ ] More disease models (lung cancer, Parkinson's, etc.)
- [ ] Medical image analysis integration

---

## ⚠️ Disclaimer

This system is for **research and educational purposes only**. It should not be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for medical decisions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
