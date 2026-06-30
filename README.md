<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/Flask-3.0+-000000?style=for-the-badge&logo=flask&logoColor=white" alt="Flask"/>
  <img src="https://img.shields.io/badge/XGBoost-2.0+-FF6600?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"/>
  <img src="https://img.shields.io/badge/SHAP-Explainability-4B8BBE?style=for-the-badge" alt="SHAP"/>
  <img src="https://img.shields.io/badge/LIME-Explainability-10B981?style=for-the-badge" alt="LIME"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">🧬 MedPredict AI</h1>
<h3 align="center">Disease Prediction System with Explainable AI</h3>

<p align="center">
  <em>An intelligent, production-ready web app that predicts diseases using machine learning<br/>and explains every prediction transparently with SHAP & LIME — built for doctors and researchers.</em>
</p>

<p align="center">
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-features">Features</a> •
  <a href="#-tech-stack">Tech Stack</a> •
  <a href="#-api-reference">API</a> •
  <a href="#-how-explainability-works">Explainability</a> •
  <a href="#-health-report">Health Report</a>
</p>

---

## 🎯 What Is This?

MedPredict AI is a **full-stack disease prediction system** that doesn't just tell you the result — it **explains why**. Doctors and healthcare professionals can input patient data, get instant predictions across 4 diseases, and see exactly which factors drove the AI's decision through interactive SHAP waterfall charts and LIME text explanations.

> **💡 Key Differentiator:** Unlike black-box AI tools, every prediction comes with a full explainability breakdown and a comprehensive health report comparing patient values against medical reference ranges.

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🤖 Prediction Engine
- **4 Disease Models** — Diabetes, Heart, Kidney, Liver
- **XGBoost** classifier with SMOTE balancing

</td>
<td width="50%">

### 📊 Explainable AI
- **SHAP waterfall plots** — feature-level contributions
- **LIME text explanations** — human-readable insights

</td>
</tr>
<tr>
<td width="50%">

### 📄 Health Reports *(NEW)*
- **Normal range comparison** with color-coded status bars
- **Radar chart** health profile visualization


</td>
<td width="50%">

### 🏥 Doctor Dashboard
- **Historical predictions** with filters & search
- **Interactive Plotly charts** — disease & risk distribution
- **CSV export** for data analysis


</td>
</tr>
</table>

### Additional Highlights

| Feature | Description |
|:--------|:------------|
| 🎨 **Premium Dark UI** | Glassmorphism design with floating orbs, micro-animations, and gradient accents |
| 📱 **Fully Responsive** | Works on desktop, tablet, and mobile |
| 🔒 **Secure** | Flask secret key, input validation, error handling |


---

## 🛠️ Tech Stack

<table>
<tr>
<td align="center" width="96">
  <strong>Python</strong><br/>
  <sub>3.10+</sub>
</td>
<td align="center" width="96">
  <strong>Flask</strong><br/>
  <sub>Web Framework</sub>
</td>
<td align="center" width="96">
  <strong>XGBoost</strong><br/>
  <sub>ML Engine</sub>
</td>
<td align="center" width="96">
  <strong>SHAP</strong><br/>
  <sub>Explainability</sub>
</td>
<td align="center" width="96">
  <strong>LIME</strong><br/>
  <sub>Explainability</sub>
</td>
<td align="center" width="96">
  <strong>SQLite</strong><br/>
  <sub>Database</sub>
</td>
<td align="center" width="96">
  <strong>Chart.js</strong><br/>
  <sub>Radar Charts</sub>
</td>
<td align="center" width="96">
  <strong>Plotly</strong><br/>
  <sub>Dashboard</sub>
</td>
</tr>
</table>

| Layer | Technologies |
|:------|:-------------|
| **Backend** | Python 3.10+, Flask 3.x, Jinja2 |
| **Machine Learning** | Scikit-learn, XGBoost, SMOTE (imbalanced-learn) |
| **Explainability** | SHAP (Shapley values), LIME (local surrogate models) |
| **Database** | SQLite with indexed queries |
| **Frontend** | HTML5, CSS3 (custom design system), Vanilla JS |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10 or higher
- pip package manager

### 1️⃣ Clone & Setup

```bash
git clone https://github.com/dheeraj0677/Disease-predictor.git
cd Disease-predictor
```

### 2️⃣ Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Download Datasets & Train Models

```bash
python core/download_datasets.py
python core/train.py
```

### 5️⃣ Launch the Application

```bash
python app.py
```

🌐 Open **http://localhost:5000** in your browser and start predicting!

---

## 📄 Health Report *(New Feature)*

The **Patient Health Report** is a comprehensive medical summary page available at `/report/<id>` after any prediction. It includes:

| Section | What It Shows |
|:--------|:--------------|
| **📊 Lab Values vs Normal Ranges** | Color-coded comparison bars (🟢 Normal · 🟡 Borderline · 🔴 Abnormal) |
| **🎯 Health Profile Radar** | Interactive Chart.js radar chart comparing patient values to normal upper bounds |
| **💡 Personalized Recommendations** | Auto-generated health advice based on which values are outside normal range |


> Every recommendation is severity-sorted (🔴 High → 🟡 Medium → 🟢 Low → 🔵 Info) and includes medically-informed actionable guidance.

---

## 🧠 How Explainability Works

### SHAP — *SHapley Additive exPlanations*

```
┌─────────────────────────────────────────────────────┐
│  Base Value (Average)                               │
│  ══════════════════                                 │
│  + Glucose: 148        ████████████  (+0.32)  🔴    │
│  + BMI: 33.6           ██████████    (+0.28)  🔴    │
│  - BloodPressure: 72   ████          (-0.08)  🔵    │
│  - Age: 31             ██            (-0.05)  🔵    │
│  ══════════════════                                 │
│  → Final Prediction: Diabetic (87.3% confidence)   │
└─────────────────────────────────────────────────────┘
```

SHAP uses **Shapley values** from cooperative game theory. For each prediction, it calculates every feature's marginal contribution — how much it pushes the prediction away from the average. **Red bars** = increase risk, **Blue bars** = decrease risk.

### LIME — *Local Interpretable Model-agnostic Explanations*

LIME creates **perturbations** of the patient's data, observes how predictions change, and fits a simple interpretable model around that specific prediction. The result is plain-English explanations:

> *"Glucose level 148 → increases diabetes risk by 32%"*
> *"BMI 33.6 → increases diabetes risk by 28%"*
> *"Blood Pressure 72 → decreases diabetes risk by 8%"*

This makes AI decisions understandable even for non-technical medical staff.

---

## 📡 API Reference

### Web Routes

| Route | Method | Description |
|:------|:-------|:------------|
| `/` | `GET` | 🏠 Homepage — tabbed patient input forms |
| `/predict` | `POST` | 🔍 Run prediction (form submit) |
| `/result/<id>` | `GET` | 📋 Prediction result with SHAP + LIME |
| `/report/<id>` | `GET` | 📄 Comprehensive health report *(NEW)* |
| `/dashboard` | `GET` | 📊 Doctor dashboard with analytics |

### REST API Endpoints

| Endpoint | Method | Description |
|:---------|:-------|:------------|
| `/api/predict` | `POST` | JSON prediction for integrations |
| `/api/whatif` | `POST` | What-If analysis (adjust & re-predict) |

<details>
<summary><strong>📝 API Usage Example</strong></summary>

```bash
# Predict diabetes via REST API
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "disease": "diabetes",
    "features": {
      "Pregnancies": 6,
      "Glucose": 148,
      "BloodPressure": 72,
      "SkinThickness": 35,
      "Insulin": 0,
      "BMI": 33.6,
      "DiabetesPedigreeFunction": 0.627,
      "Age": 50
    }
  }'
```

**Response:**
```json
{
  "success": true,
  "prediction": "Diabetic",
  "confidence": 0.8734,
  "risk_level": "High",
  "explanations": [
    "Glucose = 148.0 → increases Diabetic risk",
    "BMI = 33.6 → increases Diabetic risk",
    "Age = 50.0 → increases Diabetic risk"
  ]
}
```
</details>

---

## 📁 Project Structure

```
disease-predictor/
│
├── 📄 app.py                       # Flask application & all routes
├── ⚙️ config.py                     # Central configuration & paths
├── 📋 requirements.txt              # Python dependencies
├── 📖 README.md                     # You are here!
│
├── 🧠 core/                         # Core logic modules
│   ├── train.py                    #   Training pipeline (XGBoost + SMOTE)
│   ├── download_datasets.py        #   Dataset generator / downloader
│   ├── db.py                       #   SQLite database operations
│   ├── shap_explain.py             #   SHAP waterfall & summary plots
│   ├── lime_explain.py             #   LIME text & chart explanations
│   └── health_report.py            #   Health report generator (NEW)
│
├── 📦 storage/                       # Generated data (gitignored)
│   ├── *.csv                       #   Disease datasets
│   ├── *_model.pkl                 #   Trained XGBoost models
│   ├── *_scaler.pkl                #   Feature scalers
│   ├── *_features.json             #   Feature metadata
│   └── predictions.db              #   SQLite database
│
├── 🎨 static/
│   ├── css/
│   │   ├── style.css               #   Main design system (1100+ lines)
│   │   └── report.css              #   Health report styles (NEW)
│   ├── js/
│   │   └── app.js                  #   Frontend interactivity
│   └── charts/                     #   Generated SHAP/LIME chart images
│
└── 🖼️ templates/
    ├── base.html                   #   Base layout (nav, footer, flash)
    ├── index.html                  #   Patient input forms (tabbed)
    ├── result.html                 #   Prediction result + explanations
    ├── report.html                 #   Health report page (NEW)
    └── dashboard.html              #   Doctor dashboard
```

---

## 📚 Datasets

| Disease | Source | Samples | Features | Target |
|:--------|:-------|:-------:|:--------:|:-------|
| 🩸 Diabetes | Pima Indians (NIDDK) | 768 | 8 | Binary (Diabetic / Not) |
| ❤️ Heart Disease | Cleveland UCI | 303 | 13 | Binary (Disease / No Disease) |
| 🫘 Kidney Disease | UCI CKD | 400 | 18 | Binary (CKD / Not CKD) |
| 🫀 Liver Disease | Indian Liver Patient (UCI) | 583 | 10 | Binary (Disease / No Disease) |

All datasets are auto-downloaded and preprocessed by `core/download_datasets.py`. Models are trained with **XGBoost** + **SMOTE** oversampling for class balance.

---

## 🔮 Roadmap

- [x] Multi-disease prediction with XGBoost
- [x] SHAP waterfall charts
- [x] LIME text explanations
- [x] Doctor dashboard with Plotly charts
- [x] What-If analysis
- [x] REST API endpoints
- [x] CSV export
- [x] Patient Health Report with normal ranges *(NEW)*
- [ ] Doctor authentication with Flask-Login
- [ ] Patient risk score combining multiple disease predictions
- [ ] Time-series tracking for recurring patients
- [ ] Deployment to Render / Railway with Docker
- [ ] More disease models (lung cancer, Parkinson's, stroke)
- [ ] Medical image analysis integration

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

---

## ⚠️ Disclaimer

> **This system is for research and educational purposes only.**
> It should not be used as a substitute for professional medical diagnosis, advice, or treatment.
> Always consult a qualified healthcare provider for medical decisions.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/dheeraj0677">Dheeraj</a>
</p>

<p align="center">
  <sub>🧬 MedPredict AI — Where AI meets transparency in healthcare</sub>
</p>
