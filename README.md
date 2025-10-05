# 🕵️‍♀️ Credit Card Fraud Detection

This project demonstrates how to detect fraudulent credit card transactions using machine learning.  
It includes the full Data Science lifecycle: **EDA → Preprocessing → Modeling → Handling Imbalance (SMOTE) → Threshold Tuning → Feature Importance → Saving Artifacts**.

---

## Project Structure
```
cc-fraud-detection/
├─ data/
│  ├─ raw/
│  └─ processed/
├─ models/
│  └─ rf_smote_v1.joblib
├─ notebooks/
│  └─ Credit Card Fraud Detection.ipynb
├─ reports/
│  └─ figures/
├─ src/
│  ├─ data.py
│  ├─ features.py
│  ├─ models.py
│  └─ metrics.py
├─ tests/
├─ LICENSE
├─ Makefile
├─ pyproject.toml
├─ README.md
└─ requirements.txt
```

---

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows
pip install -r requirements.txt
jupyter lab
```
