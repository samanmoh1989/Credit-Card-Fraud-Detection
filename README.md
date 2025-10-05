# 🕵️‍♀️ Credit Card Fraud Detection

A complete **end-to-end data science project** for detecting fraudulent credit card transactions using **machine learning** and **class imbalance handling (SMOTE)**.  
This repository demonstrates a full data science workflow from EDA to deployment-ready model packaging.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Notebook Workflow](#notebook-workflow)
- [Modeling and Class Imbalance](#modeling-and-class-imbalance)
- [Threshold Tuning](#threshold-tuning)
- [Feature Importance](#feature-importance)
- [Results](#results)
- [Model Saving and Inference](#model-saving-and-inference)
- [Optional API (FastAPI)](#optional-api-fastapi)
- [Data Best Practices](#data-best-practices)
- [Development (Pre-commit/Makefile)](#development-pre-commitmakefile)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## 🧠 Overview
This project builds a fraud detection model using **Logistic Regression** and **Random Forest**, with special focus on **imbalanced data** and **performance metrics** such as Precision, Recall, F1, and ROC-AUC.

### Pipeline Summary
1. **Exploratory Data Analysis (EDA)**  
2. **Data Preprocessing and Feature Scaling**  
3. **Train-Test Split with Stratification**  
4. **Model Training (Baseline & Advanced)**  
5. **SMOTE Oversampling for Imbalanced Data**  
6. **Threshold Tuning using Precision–Recall Curve**  
7. **Feature Importance Analysis**  
8. **Model Saving and Evaluation**  

---

## 📊 Dataset
- **Source:** [Credit Card Fraud Detection Dataset (European cardholders)](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- **Samples:** 284,807 transactions  
- **Frauds:** 492 (≈0.17%)  
- **Features:**  
  - `Time`, `Amount`  
  - `V1` to `V28` — anonymized PCA components  
  - `Class` — Target variable (0 = Non-fraud, 1 = Fraud)

> ⚠️ The dataset is excluded from Git due to size limits.  
> Please place it manually under:
```
data/raw/creditcard.csv
```

---

## 🧱 Project Structure
```
cc-fraud-detection/
├─ data/
│  ├─ raw/                # contains creditcard.csv (ignored in Git)
│  └─ processed/          # contains processed train/test splits
├─ models/
│  └─ rf_smote_v1.joblib  # saved model + scaler + features + threshold
├─ notebooks/
│  └─ Credit Card Fraud Detection.ipynb
├─ reports/
│  └─ figures/            # EDA plots, correlation, PR curve, etc.
├─ src/
│  ├─ data.py
│  ├─ features.py
│  ├─ models.py
│  └─ metrics.py
├─ tests/
│  └─ test_smoke.py
├─ .gitignore
├─ .pre-commit-config.yaml
├─ Makefile
├─ pyproject.toml
├─ README.md
└─ requirements.txt
```

---

## ⚡ Quick Start
```bash
# 1) Create and activate a virtual environment
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) (Optional) Setup pre-commit hooks
pip install pre-commit
pre-commit install

# 4) Launch Jupyter
jupyter lab
```

---

## 📔 Notebook Workflow
Run this notebook:
```
notebooks/Credit Card Fraud Detection.ipynb
```

### Steps
1. **EDA:** Class distribution, summary statistics, histograms, log-transform of `Amount`, correlations.  
2. **Preprocessing:** Train/test split, feature scaling on `Time` and `Amount`.  
3. **Baseline Model:** Logistic Regression (`class_weight='balanced'`).  
4. **Advanced Model:** Random Forest Classifier.  
5. **SMOTE:** Oversampling only on training data to balance the classes.  
6. **Threshold Tuning:** Adjust classification threshold based on Precision–Recall curve.  
7. **Feature Importance:** Extract top predictors from Random Forest.  
8. **Save Artifacts:** Model, scaler, features, and threshold saved via joblib.

---

## ⚖️ Modeling and Class Imbalance
Due to the severe imbalance (~0.17%), accuracy alone is misleading.  
Key metrics include:
- Precision (Fraud)
- Recall (Fraud)
- F1-score (Fraud)
- ROC-AUC and PR-AUC

SMOTE is applied **only on the training set** to prevent data leakage.

---

## 🎚️ Threshold Tuning
After obtaining predicted probabilities using `predict_proba`:
- Find threshold maximizing **F1-score** or satisfying domain constraints.  
- Example: Recall ≥ 0.85 while Precision ≥ 0.80.  
- The chosen threshold is saved within the model artifact (`rf_smote_v1.joblib`).

---

## 🔍 Feature Importance
Random Forest feature importances typically highlight:
`V14`, `V10`, `V4`, `V12`, `V17`  
These represent PCA-transformed latent factors, useful for identifying fraud patterns.

---

## 🧾 Results
| Model | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | ROC-AUC |
|------|-------------------:|---------------:|-----------:|--------:|
| Logistic Regression | 0.061 | **0.918** | 0.114 | 0.972 |
| Random Forest       | **0.961** | 0.755 | 0.846 | 0.957 |
| RF + SMOTE          | 0.863 | 0.837 | **0.850** | **0.975** |

**Summary:** The combination of **Random Forest + SMOTE** achieves the best balance between precision and recall.

---

## 💾 Model Saving and Inference
Model artifacts are stored in:
```
models/rf_smote_v1.joblib
```

### Example usage
```python
import joblib, pandas as pd

artifact = joblib.load("models/rf_smote_v1.joblib")
model     = artifact["model"]
scaler    = artifact["scaler"]
features  = artifact["features"]
thr       = artifact["threshold"]

# Example scoring
new_df = pd.read_csv("data/raw/creditcard.csv").sample(5, random_state=42)
new_df[["Time","Amount"]] = scaler.transform(new_df[["Time","Amount"]])
X_new = new_df[features]

proba = model.predict_proba(X_new)[:,1]
pred  = (proba >= thr).astype(int)
```

---

## 🌐 Optional API (FastAPI)
```python
# app.py
from fastapi import FastAPI
import joblib, pandas as pd

artifact = joblib.load("models/rf_smote_v1.joblib")
model, scaler, features, thr = artifact["model"], artifact["scaler"], artifact["features"], artifact["threshold"]

app = FastAPI()

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    df[["Time","Amount"]] = scaler.transform(df[["Time","Amount"]])
    X = df[features]
    p = float(model.predict_proba(X)[:,1][0])
    y = int(p >= thr)
    return {"prob_fraud": p, "prediction": y}
```
Run with:
```bash
pip install fastapi uvicorn
uvicorn app:app --reload --port 8000
```

---

## 🚫 Data Best Practices
- Never commit large CSVs (>100MB).  
- Use `.gitignore` to exclude them.  
- Keep dataset links or add a Kaggle API downloader script.  
- For data versioning, use **DVC** or **Git LFS**.

### Example `.gitignore`
```
__pycache__/
.ipynb_checkpoints/
*.csv
data/raw/
data/processed/
*.pkl
*.joblib
```

---

## 🧩 Development (Pre-commit/Makefile)
### Pre-commit Hooks
- Runs **Black**, **isort**, and **flake8** before each commit.

### Makefile Shortcuts
```
make install   # install requirements + pre-commit
make format    # run isort + black
make lint      # run flake8
make test      # run pytest
```

---

## 🚀 Roadmap
- Hyperparameter tuning (GridSearchCV / Optuna)
- Streamlit dashboard for model demo
- MLflow or Weights & Biases for experiment tracking
- Dockerfile and CI/CD pipeline for deployment

---

## 🪪 License
This project is licensed under the **MIT License**.

---

## 🙏 Acknowledgements
- [ULB Machine Learning Group](https://mlg.ulb.ac.be) for providing the original dataset.  
- Kaggle community for open-source discussions and notebooks.
