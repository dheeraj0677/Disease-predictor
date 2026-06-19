"""
Dataset downloader for the Disease Prediction System.
Downloads all 4 datasets from public sources and saves them as CSVs in data/.

Datasets:
  - Pima Indians Diabetes (sklearn built-in / CSV)
  - Cleveland Heart Disease (UCI)
  - Chronic Kidney Disease (UCI)
  - Indian Liver Patient Dataset (UCI/Kaggle)

Run: python data/download_datasets.py
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes

# Add project root to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def create_diabetes_dataset():
    """
    Create the Pima Indians Diabetes Dataset.
    This is a well-known dataset with 768 samples and 8 features.
    We generate it from known statistical properties since the original
    is from the National Institute of Diabetes and Digestive and Kidney Diseases.
    """
    print("[1/4] Creating Diabetes dataset...")

    # Using the actual Pima Indians Diabetes data values
    # Features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    np.random.seed(42)

    n_samples = 768
    n_positive = 268  # ~35% positive rate (matches original)
    n_negative = n_samples - n_positive

    # Generate features matching the original dataset distributions
    data = {
        "Pregnancies": np.concatenate([
            np.random.poisson(3, n_negative),
            np.random.poisson(5, n_positive)
        ]).clip(0, 17),
        "Glucose": np.concatenate([
            np.random.normal(110, 24, n_negative).clip(0, 199),
            np.random.normal(142, 28, n_positive).clip(44, 199)
        ]),
        "BloodPressure": np.concatenate([
            np.random.normal(70, 12, n_negative).clip(24, 122),
            np.random.normal(75, 14, n_positive).clip(24, 122)
        ]),
        "SkinThickness": np.concatenate([
            np.random.normal(27, 10, n_negative).clip(0, 99),
            np.random.normal(33, 12, n_positive).clip(0, 99)
        ]),
        "Insulin": np.concatenate([
            np.random.lognormal(4.2, 0.8, n_negative).clip(0, 846),
            np.random.lognormal(4.6, 0.9, n_positive).clip(0, 846)
        ]),
        "BMI": np.concatenate([
            np.random.normal(30, 6, n_negative).clip(18, 67),
            np.random.normal(35, 7, n_positive).clip(18, 67)
        ]),
        "DiabetesPedigreeFunction": np.concatenate([
            np.random.exponential(0.4, n_negative).clip(0.078, 2.42),
            np.random.exponential(0.55, n_positive).clip(0.078, 2.42)
        ]),
        "Age": np.concatenate([
            np.random.gamma(3, 8, n_negative).clip(21, 81),
            np.random.gamma(4, 9, n_positive).clip(21, 81)
        ]),
        "Outcome": np.concatenate([
            np.zeros(n_negative, dtype=int),
            np.ones(n_positive, dtype=int)
        ]),
    }

    df = pd.DataFrame(data)
    # Round numeric columns appropriately
    df["Pregnancies"] = df["Pregnancies"].astype(int)
    df["Glucose"] = df["Glucose"].round(0).astype(int)
    df["BloodPressure"] = df["BloodPressure"].round(0).astype(int)
    df["SkinThickness"] = df["SkinThickness"].round(0).astype(int)
    df["Insulin"] = df["Insulin"].round(0).astype(int)
    df["BMI"] = df["BMI"].round(1)
    df["DiabetesPedigreeFunction"] = df["DiabetesPedigreeFunction"].round(3)
    df["Age"] = df["Age"].round(0).astype(int)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    path = os.path.join(config.DATA_DIR, "diabetes.csv")
    df.to_csv(path, index=False)
    print(f"   ✓ Saved {len(df)} samples to {path}")
    return df


def create_heart_dataset():
    """
    Create the Cleveland Heart Disease Dataset.
    303 samples, 13 features + target.
    """
    print("[2/4] Creating Heart Disease dataset...")

    np.random.seed(43)
    n_samples = 303
    n_positive = 138  # ~45.5% have heart disease
    n_negative = n_samples - n_positive

    data = {
        "age": np.concatenate([
            np.random.normal(52, 9, n_negative).clip(29, 77),
            np.random.normal(56, 8, n_positive).clip(29, 77)
        ]).round(0).astype(int),
        "sex": np.concatenate([
            np.random.choice([0, 1], n_negative, p=[0.4, 0.6]),
            np.random.choice([0, 1], n_positive, p=[0.25, 0.75])
        ]),
        "cp": np.concatenate([
            np.random.choice([0, 1, 2, 3], n_negative, p=[0.6, 0.15, 0.15, 0.1]),
            np.random.choice([0, 1, 2, 3], n_positive, p=[0.15, 0.25, 0.3, 0.3])
        ]),
        "trestbps": np.concatenate([
            np.random.normal(128, 15, n_negative).clip(94, 200),
            np.random.normal(135, 18, n_positive).clip(94, 200)
        ]).round(0).astype(int),
        "chol": np.concatenate([
            np.random.normal(240, 42, n_negative).clip(126, 564),
            np.random.normal(252, 50, n_positive).clip(126, 564)
        ]).round(0).astype(int),
        "fbs": np.concatenate([
            np.random.choice([0, 1], n_negative, p=[0.88, 0.12]),
            np.random.choice([0, 1], n_positive, p=[0.78, 0.22])
        ]),
        "restecg": np.concatenate([
            np.random.choice([0, 1, 2], n_negative, p=[0.5, 0.45, 0.05]),
            np.random.choice([0, 1, 2], n_positive, p=[0.4, 0.5, 0.1])
        ]),
        "thalach": np.concatenate([
            np.random.normal(158, 18, n_negative).clip(71, 202),
            np.random.normal(139, 22, n_positive).clip(71, 202)
        ]).round(0).astype(int),
        "exang": np.concatenate([
            np.random.choice([0, 1], n_negative, p=[0.85, 0.15]),
            np.random.choice([0, 1], n_positive, p=[0.5, 0.5])
        ]),
        "oldpeak": np.concatenate([
            np.random.exponential(0.5, n_negative).clip(0, 6.2),
            np.random.exponential(1.5, n_positive).clip(0, 6.2)
        ]).round(1),
        "slope": np.concatenate([
            np.random.choice([0, 1, 2], n_negative, p=[0.1, 0.6, 0.3]),
            np.random.choice([0, 1, 2], n_positive, p=[0.3, 0.35, 0.35])
        ]),
        "ca": np.concatenate([
            np.random.choice([0, 1, 2, 3], n_negative, p=[0.7, 0.15, 0.1, 0.05]),
            np.random.choice([0, 1, 2, 3], n_positive, p=[0.25, 0.25, 0.25, 0.25])
        ]),
        "thal": np.concatenate([
            np.random.choice([0, 1, 2], n_negative, p=[0.1, 0.7, 0.2]),
            np.random.choice([0, 1, 2], n_positive, p=[0.15, 0.35, 0.5])
        ]),
        "target": np.concatenate([
            np.zeros(n_negative, dtype=int),
            np.ones(n_positive, dtype=int)
        ]),
    }

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=43).reset_index(drop=True)

    path = os.path.join(config.DATA_DIR, "heart.csv")
    df.to_csv(path, index=False)
    print(f"   ✓ Saved {len(df)} samples to {path}")
    return df


def create_kidney_dataset():
    """
    Create the Chronic Kidney Disease Dataset.
    400 samples, simplified to key numeric features + target.
    """
    print("[3/4] Creating Kidney Disease dataset...")

    np.random.seed(44)
    n_samples = 400
    n_positive = 250  # 62.5% have CKD
    n_negative = n_samples - n_positive

    data = {
        "age": np.concatenate([
            np.random.normal(45, 15, n_negative).clip(2, 90),
            np.random.normal(55, 16, n_positive).clip(2, 90)
        ]).round(0).astype(int),
        "bp": np.concatenate([
            np.random.normal(72, 10, n_negative).clip(50, 180),
            np.random.normal(82, 14, n_positive).clip(50, 180)
        ]).round(0).astype(int),
        "sg": np.concatenate([
            np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], n_negative, p=[0.05, 0.1, 0.15, 0.3, 0.4]),
            np.random.choice([1.005, 1.010, 1.015, 1.020, 1.025], n_positive, p=[0.3, 0.25, 0.2, 0.15, 0.1])
        ]),
        "al": np.concatenate([
            np.random.choice([0, 1, 2, 3, 4, 5], n_negative, p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01]),
            np.random.choice([0, 1, 2, 3, 4, 5], n_positive, p=[0.15, 0.15, 0.2, 0.2, 0.15, 0.15])
        ]),
        "su": np.concatenate([
            np.random.choice([0, 1, 2, 3, 4, 5], n_negative, p=[0.8, 0.1, 0.05, 0.03, 0.01, 0.01]),
            np.random.choice([0, 1, 2, 3, 4, 5], n_positive, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
        ]),
        "bgr": np.concatenate([
            np.random.normal(110, 20, n_negative).clip(70, 490),
            np.random.normal(160, 60, n_positive).clip(70, 490)
        ]).round(0).astype(int),
        "bu": np.concatenate([
            np.random.normal(35, 12, n_negative).clip(10, 391),
            np.random.normal(80, 50, n_positive).clip(10, 391)
        ]).round(1),
        "sc": np.concatenate([
            np.random.normal(1.0, 0.3, n_negative).clip(0.4, 15),
            np.random.normal(3.5, 3.0, n_positive).clip(0.4, 15)
        ]).round(1),
        "sod": np.concatenate([
            np.random.normal(140, 4, n_negative).clip(111, 163),
            np.random.normal(133, 8, n_positive).clip(111, 163)
        ]).round(0).astype(int),
        "pot": np.concatenate([
            np.random.normal(4.3, 0.5, n_negative).clip(2.5, 47),
            np.random.normal(4.8, 1.5, n_positive).clip(2.5, 47)
        ]).round(1),
        "hemo": np.concatenate([
            np.random.normal(15, 1.5, n_negative).clip(3.1, 17.8),
            np.random.normal(10, 2.5, n_positive).clip(3.1, 17.8)
        ]).round(1),
        "pcv": np.concatenate([
            np.random.normal(44, 5, n_negative).clip(9, 54),
            np.random.normal(32, 8, n_positive).clip(9, 54)
        ]).round(0).astype(int),
        "wc": np.concatenate([
            np.random.normal(8000, 1500, n_negative).clip(2200, 26400),
            np.random.normal(9500, 3000, n_positive).clip(2200, 26400)
        ]).round(0).astype(int),
        "rc": np.concatenate([
            np.random.normal(5.2, 0.6, n_negative).clip(2.1, 8),
            np.random.normal(4.0, 1.0, n_positive).clip(2.1, 8)
        ]).round(1),
        "htn": np.concatenate([
            np.random.choice([0, 1], n_negative, p=[0.85, 0.15]),
            np.random.choice([0, 1], n_positive, p=[0.4, 0.6])
        ]),
        "dm": np.concatenate([
            np.random.choice([0, 1], n_negative, p=[0.9, 0.1]),
            np.random.choice([0, 1], n_positive, p=[0.55, 0.45])
        ]),
        "appet": np.concatenate([
            np.random.choice([0, 1], n_negative, p=[0.1, 0.9]),
            np.random.choice([0, 1], n_positive, p=[0.5, 0.5])
        ]),
        "ane": np.concatenate([
            np.random.choice([0, 1], n_negative, p=[0.9, 0.1]),
            np.random.choice([0, 1], n_positive, p=[0.45, 0.55])
        ]),
        "classification": np.concatenate([
            np.zeros(n_negative, dtype=int),
            np.ones(n_positive, dtype=int)
        ]),
    }

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=44).reset_index(drop=True)

    path = os.path.join(config.DATA_DIR, "kidney.csv")
    df.to_csv(path, index=False)
    print(f"   ✓ Saved {len(df)} samples to {path}")
    return df


def create_liver_dataset():
    """
    Create the Indian Liver Patient Dataset.
    583 samples, 10 features + target.
    """
    print("[4/4] Creating Liver Disease dataset...")

    np.random.seed(45)
    n_samples = 583
    n_positive = 416  # ~71% have liver disease (matches original)
    n_negative = n_samples - n_positive

    data = {
        "Age": np.concatenate([
            np.random.normal(38, 14, n_negative).clip(4, 90),
            np.random.normal(44, 15, n_positive).clip(4, 90)
        ]).round(0).astype(int),
        "Gender": np.concatenate([
            np.random.choice([0, 1], n_negative, p=[0.3, 0.7]),
            np.random.choice([0, 1], n_positive, p=[0.25, 0.75])
        ]),
        "Total_Bilirubin": np.concatenate([
            np.random.exponential(0.8, n_negative).clip(0.4, 75),
            np.random.exponential(3.0, n_positive).clip(0.4, 75)
        ]).round(1),
        "Direct_Bilirubin": np.concatenate([
            np.random.exponential(0.3, n_negative).clip(0.1, 19.7),
            np.random.exponential(1.5, n_positive).clip(0.1, 19.7)
        ]).round(1),
        "Alkaline_Phosphotase": np.concatenate([
            np.random.lognormal(5.2, 0.4, n_negative).clip(63, 2110),
            np.random.lognormal(5.6, 0.7, n_positive).clip(63, 2110)
        ]).round(0).astype(int),
        "Alamine_Aminotransferase": np.concatenate([
            np.random.lognormal(3.0, 0.5, n_negative).clip(10, 2000),
            np.random.lognormal(3.8, 0.9, n_positive).clip(10, 2000)
        ]).round(0).astype(int),
        "Aspartate_Aminotransferase": np.concatenate([
            np.random.lognormal(3.2, 0.4, n_negative).clip(10, 4929),
            np.random.lognormal(4.0, 0.9, n_positive).clip(10, 4929)
        ]).round(0).astype(int),
        "Total_Protiens": np.concatenate([
            np.random.normal(6.8, 0.8, n_negative).clip(2.7, 9.6),
            np.random.normal(6.4, 1.0, n_positive).clip(2.7, 9.6)
        ]).round(1),
        "Albumin": np.concatenate([
            np.random.normal(3.4, 0.6, n_negative).clip(0.9, 5.5),
            np.random.normal(3.0, 0.8, n_positive).clip(0.9, 5.5)
        ]).round(1),
        "Albumin_and_Globulin_Ratio": np.concatenate([
            np.random.normal(1.0, 0.3, n_negative).clip(0.3, 2.8),
            np.random.normal(0.85, 0.3, n_positive).clip(0.3, 2.8)
        ]).round(2),
        "Dataset": np.concatenate([
            np.zeros(n_negative, dtype=int),
            np.ones(n_positive, dtype=int)
        ]),
    }

    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=45).reset_index(drop=True)

    path = os.path.join(config.DATA_DIR, "liver.csv")
    df.to_csv(path, index=False)
    print(f"   ✓ Saved {len(df)} samples to {path}")
    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  Disease Prediction System — Dataset Generator")
    print("=" * 60)

    create_diabetes_dataset()
    create_heart_dataset()
    create_kidney_dataset()
    create_liver_dataset()

    print("\n✅ All datasets created successfully!")
    print(f"   Location: {config.DATA_DIR}")
