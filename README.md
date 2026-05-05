# 🏠 House Prices — Advanced Regression Techniques

> **Kaggle Public Score: 0.11893** | Top ~12% on the Leaderboard

A complete end-to-end machine learning project built from scratch — from raw data exploration to a fully optimized stacking ensemble. Every step was designed to be understandable, reproducible, and production-aware.

---

## 📊 Score Progression

| Version | Approach | Public Score |
|---------|----------|-------------|
| v1 | Baseline (first submission) | 0.13640 |
| v2 | Stacking: XGB + LGBM + GBM | 0.12399 |
| v3 | + Feature Selection (92 cols removed) | 0.12346 |
| v4 | + Target Encoding | 0.12223 |
| v5 | + ElasticNet added to stack | 0.12112 |
| v6 | + CatBoost added to stack | 0.12046 |
| v7 | + Optuna 200 trials | 0.11921 |
| **v8** | **+ Skewness Transformation** | **0.11893** |

Total improvement: **0.0175 points** from start to finish.

---

## 🔧 Project Pipeline

```
Raw Data
   │
   ├── 1. Exploratory Data Analysis (EDA)
   │       ├── SalePrice distribution & log transform
   │       ├── Missing value analysis (missingno)
   │       ├── Categorical variable analysis
   │       └── Rare category analysis with TARGET_MEAN
   │
   ├── 2. Data Preparation
   │       ├── Missing value imputation (None / 0 / group median / mode)
   │       └── Outlier capping (IQR winsorization, q1=0.05, q3=0.95)
   │
   ├── 3. Feature Engineering (14 new features)
   │       ├── Area-based: TotalSF, TotalBath, TotalPorchSF, LivAreaPerRoom
   │       ├── Time-based: HouseAge, RemodAge, IsRemodeled, IsNew, GarageAge
   │       ├── Binary (has/hasn't): HasPool, HasGarage, HasFireplace, ...
   │       └── Quality scores: OverallScore, GarageScore, BsmtScore
   │
   ├── 4. Encoding
   │       ├── Ordinal: Quality columns (Ex=5, Gd=4, TA=3, Fa=2, Po=1, None=0)
   │       ├── Label: Binary/ordered columns (CentralAir, Alley, Fence...)
   │       ├── Target Encoding: Neighborhood, Exterior, SaleCondition, MSZoning
   │       └── One-Hot Encoding: Remaining nominal columns
   │
   ├── 5. Feature Selection
   │       └── LightGBM importance — removed 92 zero-importance columns
   │
   ├── 6. Skewness Transformation
   │       └── log1p applied to features with skewness > 0.75
   │
   ├── 7. Hyperparameter Optimization (Optuna — 200 trials each)
   │       ├── XGBoost
   │       ├── LightGBM
   │       ├── GradientBoosting
   │       └── CatBoost
   │
   └── 8. Stacking Ensemble
           ├── Base models: XGB + LGBM + GBM + CatBoost + ElasticNetCV
           └── Meta model: RidgeCV
```

---

## 🧠 Key Techniques & Learnings

### Missing Value Strategy
Missing values were treated based on domain knowledge from the dataset documentation — not just statistical imputation. For example, `PoolQC = NaN` means "no pool", not "unknown pool quality". This distinction is critical.

| Strategy | Columns | Reason |
|----------|---------|--------|
| Fill `None` | PoolQC, Alley, Fence, FireplaceQu, Garage*, Bsmt* | "NA" = feature doesn't exist |
| Fill `0` | GarageArea, MasVnrArea, BsmtFinSF*, etc. | No area = 0 sq ft |
| Group median | LotFrontage | Neighbors share similar frontage |
| Mode | MSZoning, Electrical, KitchenQual, etc. | Rare test-set missings |

### Target Encoding vs One-Hot Encoding
High-cardinality columns like `Neighborhood` (25 unique values) were target-encoded instead of one-hot encoded. This preserves the ordinal relationship between neighborhoods and their average sale prices. **Critical:** target encoding must be fit only on train data to prevent data leakage.

### Why Stacking Over Blending?
Manual blending with fixed weights (0.25/0.25/0.20...) was tested but performed worse (0.12012 vs 0.11921). Stacking with `RidgeCV` as meta-model learns optimal weights from data — consistently better on this dataset.

### Optuna Hyperparameter Optimization
Bayesian optimization with 200 trials per model. Standard GridSearch was not used — with 6+ parameters, it would require thousands of evaluations. Optuna uses prior results to intelligently guide the search.

---

## 📦 Tech Stack

| Tool | Purpose |
|------|---------|
| `pandas` / `numpy` | Data manipulation |
| `scikit-learn` | Models, stacking, cross-validation |
| `xgboost` | Gradient boosting |
| `lightgbm` | Fast gradient boosting |
| `catboost` | Categorical-aware boosting |
| `optuna` | Bayesian hyperparameter optimization |
| `category_encoders` | Target encoding |
| `missingno` | Missing value visualization |
| `scipy` | Skewness calculation |

---

## 🚀 How to Run

```bash
# 1. Clone the repository
git clone https://github.com/codelones/house-prices-ml-kaggle.git
cd house-prices-ml-kaggle

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add dataset
# Download train.csv and test.csv from:
# https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
# Place them in: datasets/

# 5. Run
python house_price_predict_final.py
```

---

## 📁 Repository Structure

```
house-prices-ml-kaggle/
│
├── datasets/
│   ├── train.csv
│   └── test.csv
│
├── house_price_predict_final.py   # Main pipeline
├── requirements.txt
└── README.md
```

---

## 📋 Requirements

```
pandas
numpy
matplotlib
seaborn
missingno
scikit-learn
xgboost
lightgbm
catboost
optuna
category_encoders
scipy
```

---

## 👤 About

**Hasan Pireci** — Statistics student at Eskişehir Osmangazi University (ESOGÜ), aspiring data scientist.

This project was built as a hands-on learning experience — covering the full ML pipeline from EDA to deployment-ready code. Key learnings include feature engineering strategies, encoding techniques, Bayesian hyperparameter optimization with Optuna, and ensemble methods (stacking vs blending).

- 🐙 GitHub: [codelones](https://github.com/codelones)
- 📧 hasanpireci92@gmail.com
- 📍 Eskişehir, Turkey

---

## 📄 License

MIT License — feel free to use, modify, and share.
