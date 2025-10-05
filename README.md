# 🕵️‍♀️ Credit Card Fraud Detection

**تشخیص تقلب در تراکنش‌های کارت اعتباری** با یادگیری ماشین و مدیریت عدم‌توازن شدید کلاس‌ها.  
این مخزن چرخه‌ی کامل یک پروژه‌ی دیتا ساینس را پوشش می‌دهد:  
**EDA → Preprocessing → Modeling → Handling Imbalance (SMOTE) → Threshold Tuning → Feature Importance → Saving & Inference**

> **Note:** All source code and comments are in English; this README is Persian-first for clarity.

---

## فهرست مطالب
- [مرور کلی](#مرور-کلی)
- [دیتاست](#دیتاست)
- [ساختار پروژه](#ساختار-پروژه)
- [شروع سریع (Quick Start)](#شروع-سریع-quick-start)
- [اجرای نوت‌بوک](#اجرای-نوتبوک)
- [مدل‌سازی و عدم‌توازن](#مدلسازی-و-عدمتوازن)
- [تنظیم آستانه (Threshold Tuning)](#تنظیم-آستانه-threshold-tuning)
- [اهمیت ویژگی‌ها](#اهمیت-ویژگیها)
- [نتایج نمونه](#نتایج-نمونه)
- [ذخیره و استفاده از مدل](#ذخیره-و-استفاده-از-مدل)
- [API اختیاری (FastAPI)](#api-اختیاری-fastapi)
- [بهترین‌عمل‌ها برای داده‌ها](#بهترینعملها-برای-دادهها)
- [توسعه‌دهی (Pre-commit/Makefile)](#توسعهدهی-pre-commitmakefile)
- [نقشه راه](#نقشه-راه)
- [مجوز](#مجوز)
- [سپاس](#سپاس)

---

## مرور کلی
هدف: ساخت مدلی عملی برای شناسایی تقلب با **Precision/Recall** متعادل.  
الگوریتم‌ها:
- Baseline: **Logistic Regression** (با `class_weight="balanced"`)
- Advanced: **Random Forest**
- Imbalance: **SMOTE** روی داده‌ی آموزش
- Tuning: منحنی **Precision–Recall** برای تعیین آستانه‌ی تصمیم
- Explainability: **Feature Importances**

---

## دیتاست
- نام: *Credit Card Fraud Detection (European cardholders)*  
- اندازه: `284,807` تراکنش؛ تقلب‌ها: `492` (~0.17%)  
- ستون‌ها: `Time`, `Amount`, `V1..V28` (ویژگی‌های ناشناس‌شده با PCA)، `Class` (۰=Normal، ۱=Fraud)

> 🎯 فایل اصلی را دانلود کرده و در مسیر زیر قرار دهید (به‌صورت پیش‌فرض در گیت **رد** می‌شود):  
```
data/raw/creditcard.csv
```

---

## ساختار پروژه
```
cc-fraud-detection/
├─ data/
│  ├─ raw/                # creditcard.csv (قرار دهید؛ commit نکنید)
│  └─ processed/          # X_train/X_test/y_train/y_test (خروجی split؛ commit نکنید)
├─ models/
│  └─ rf_smote_v1.joblib  # مدل ذخیره‌شده + scaler + features + threshold
├─ notebooks/
│  └─ Credit Card Fraud Detection.ipynb
├─ reports/
│  └─ figures/            # نمودارهای EDA/نتایج
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

## شروع سریع (Quick Start)
```bash
# 1) Create & activate venv
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 2) Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3) (Optional) Enable pre-commit hooks
pip install pre-commit
pre-commit install

# 4) Launch Jupyter
jupyter lab
```

---

## اجرای نوت‌بوک
نوت‌بوک زیر را باز و سلول‌ها را به‌ترتیب اجرا کنید:  
```
notebooks/Credit Card Fraud Detection.ipynb
```
**جریان کار:**
1) **EDA:** توزیع کلاس‌ها، آمار توصیفی، هیستوگرام `Amount` (نسخه‌ی log)، همبستگی‌ها  
2) **Preprocessing:** `train_test_split` با `stratify`، استانداردسازی `Time/Amount`  
3) **Baseline:** Logistic Regression  
4) **Random Forest:** مدل پیشرفته و مقایسه  
5) **SMOTE:** oversampling فقط روی Train  
6) **Threshold Tuning:** انتخاب آستانه با **PR Curve**  
7) **Feature Importance:** تحلیل ویژگی‌های مهم  
8) **Saving:** ذخیره‌ی artifact (مدل + اسکیلر + لیست فیچرها + آستانه)

---

## مدل‌سازی و عدم‌توازن
- **Accuracy به‌تنهایی کافی نیست** (به‌دلیل کلاس اقلیت ~0.17%)  
- متریک‌های اصلی: **Precision، Recall، F1، ROC-AUC، PR-AUC**  
- **SMOTE** فقط روی `X_train, y_train` اعمال می‌شود (جلوگیری از **Data Leakage**)

---

## تنظیم آستانه (Threshold Tuning)
پس از گرفتن احتمال‌ها با `predict_proba`:
- آستانه‌ی با **بیشترین F1** را بیابید، یا
- آستانه‌ای که **قیود کسب‌وکار** را برآورده کند (مثلاً: Recall ≥ 0.85 و Precision ≥ 0.80)
- آستانه‌ی انتخابی در artifact ذخیره می‌شود (`models/rf_smote_v1.joblib`)

---

## اهمیت ویژگی‌ها
با استفاده از **RandomForest feature_importances_**، معمولاً `V14`, `V10`, `V4`, `V12`, `V17` از مهم‌ترین‌ها هستند.  
(این‌ها مؤلفه‌های نهفته‌ی PCAاند؛ قابل تفسیر دامنه‌ای مستقیم نیستند، اما الگوهای تفکیک‌کننده را حمل می‌کنند.)

---

## نتایج نمونه
*(ارقام نمونه—با نتایج خودتان به‌روزرسانی کنید)*

| Model | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) | ROC-AUC |
|------|-------------------:|---------------:|-----------:|--------:|
| Logistic Regression | 0.061 | **0.918** | 0.114 | 0.972 |
| Random Forest       | **0.961** | 0.755 | 0.846 | 0.957 |
| RF + SMOTE          | 0.863 | 0.837 | **0.850** | **0.975** |

**جمع‌بندی:** پیکربندی **Random Forest + SMOTE** توازن بهتری بین Precision/Recall ایجاد کرده و برای کاربرد عملی مناسب‌تر است.

---

## ذخیره و استفاده از مدل
مدل و ملحقات در یک artifact ذخیره می‌شود:
```
models/rf_smote_v1.joblib
```
کد نمونه برای بارگذاری و امتیازدهی (scoring):
```python
import joblib, pandas as pd

artifact = joblib.load("models/rf_smote_v1.joblib")
model     = artifact["model"]
scaler    = artifact["scaler"]
features  = artifact["features"]
thr       = artifact["threshold"]  # tuned threshold

new_df = pd.read_csv("data/raw/creditcard.csv").sample(5, random_state=42).copy()
new_df[["Time","Amount"]] = scaler.transform(new_df[["Time","Amount"]])
X_new = new_df[features]

proba = model.predict_proba(X_new)[:,1]
pred  = (proba >= thr).astype(int)
```

---

## API اختیاری (FastAPI)
```python
# app.py
from fastapi import FastAPI
import joblib, pandas as pd

art = joblib.load("models/rf_smote_v1.joblib")
model, scaler, feats, thr = art["model"], art["scaler"], art["features"], art["threshold"]

app = FastAPI()

@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    df[["Time","Amount"]] = scaler.transform(df[["Time","Amount"]])
    X = df[feats]
    p = float(model.predict_proba(X)[:,1][0])
    y = int(p >= thr)
    return {"prob_fraud": p, "pred": y}
```
```bash
pip install fastapi uvicorn
uvicorn app:app --reload --port 8000
```

---

## بهترین‌عمل‌ها برای داده‌ها
- فایل‌های بزرگ (`*.csv`) را **کامیت نکنید**؛ از `.gitignore` استفاده کنید.  
- لینک دیتاست را در README بگذارید و/یا اسکریپت دانلود (Kaggle API) اضافه کنید.  
- برای نسخه‌بندی داده‌ها: **DVC** یا **Git LFS** (در صورت نیاز واقعی).

نمونه `.gitignore` حداقلی:
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

## توسعه‌دهی (Pre-commit/Makefile)
- **pre-commit**: فرمت خودکار (Black/Isort) و چک‌ها قبل از هر commit  
- **Makefile**: میانبرهای توسعه
```
make install   # pip install -r requirements.txt + pre-commit install
make format    # isort . && black .
make lint      # flake8 src tests
make test      # pytest
```

---

## نقشه راه
- Hyperparameter Tuning (Grid/Random/Optuna)  
- Streamlit Dashboard برای دمو  
- MLflow/W&B برای ردیابی آزمایش‌ها  
- Dockerfile + CI/CD برای دیپلوی

---

## مجوز
این مخزن تحت مجوز **MIT** منتشر شده است (فایل `LICENSE`).

---

## سپاس
سپاس از ULB Machine Learning Group بابت انتشار دیتاست مرجع و جامعه‌ی Kaggle برای آموزش‌های مرتبط.
