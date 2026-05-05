import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import missingno as msno
import warnings
from scipy.stats import skew
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, ElasticNetCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
import category_encoders as ce
import optuna

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

##############################################################
# 1. DATA LOADING
##############################################################

df_train = pd.read_csv('datasets/train.csv')
df_test = pd.read_csv('datasets/test.csv')

y = np.log1p(df_train['SalePrice'])
train_ID = df_train['Id']
test_ID = df_test['Id']

##############################################################
# 2. HELPER FUNCTIONS
##############################################################

def first_check_df(dataframe, plotna=False):
    print("Dataset Shape")
    print(f"Observations : {dataframe.shape[0]}\nFeatures     : {dataframe.shape[1]}")
    print("**********************************************")
    print("Feature Types")
    print(dataframe.dtypes.value_counts())
    print("**********************************************")
    print("Descriptive Statistics (Numerical)")
    print(dataframe.describe(include=['int64', 'float64']))
    print("**********************************************")
    print("Descriptive Statistics (Categorical)")
    print(dataframe.describe(include=['object']))
    print("**********************************************")
    print("First 5 Observations")
    print(dataframe.head())
    print("**********************************************")
    print("Last 5 Observations")
    print(dataframe.tail())
    print("**********************************************")
    print(f"Duplicate Rows: {dataframe.duplicated().sum()}")
    print("**********************************************")
    missing = dataframe.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    missing_pct = (missing / len(dataframe) * 100).round(2)
    if missing.empty:
        print("No Missing Values")
    else:
        print("Missing Value Ratio (%)")
        print(missing_pct)
    print("**********************************************")
    if plotna:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        msno.bar(dataframe, ax=axes[0], fontsize=8)
        msno.matrix(dataframe, ax=axes[1], fontsize=8)
        plt.tight_layout()
        plt.show(block=True)


def analyze_col_names(dataframe, cat_th=20, car_th=20):
    num_cols = dataframe.select_dtypes(include=['int64', 'float64']).columns.tolist()
    num_but_cat = [col for col in num_cols if dataframe[col].nunique() < cat_th]
    num_cols = [col for col in num_cols if col not in num_but_cat]
    cat_cols = dataframe.select_dtypes(include=['object', 'category']).columns.tolist()
    cat_but_car = [col for col in cat_cols if dataframe[col].nunique() > car_th]
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    cat_cols = cat_cols + num_but_cat
    print(f"Numerical Features        : {len(num_cols)}")
    print(f"Categorical Features      : {len(cat_cols)}")
    print(f"Numerical but Categorical : {len(num_but_cat)}")
    print(f"Cardinal Features         : {len(cat_but_car)}")
    print(f"Total Features            : {len(dataframe.columns)}")
    total = len(num_cols) + len(cat_cols) + len(cat_but_car)
    if total < len(dataframe.columns):
        print("WARNING: Some features are not captured!")
    else:
        print("All features successfully captured.")
    return num_cols, cat_cols, num_but_cat, cat_but_car


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({
        col_name: dataframe[col_name].value_counts(),
        "Ratio (%)": dataframe[col_name].value_counts() / len(dataframe) * 100
    }))
    print("*" * 50)


def rare_analyzer(dataframe, target, cat_cols):
    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({
            "COUNT": dataframe[col].value_counts(),
            "Ratio (%)": dataframe[col].value_counts() / len(dataframe) * 100,
            "TARGET_MEAN": dataframe.groupby(col)[target].mean()
        }), end="\n\n")


def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    Q1 = dataframe[col_name].quantile(q1)
    Q3 = dataframe[col_name].quantile(q3)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return lower, upper


def check_outliers(dataframe, col_name, q1=0.05, q3=0.95):
    lower, upper = outlier_thresholds(dataframe, col_name, q1, q3)
    return dataframe[(dataframe[col_name] < lower) | (dataframe[col_name] > upper)].shape[0] > 0


def replace_with_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    lower, upper = outlier_thresholds(dataframe, col_name, q1, q3)
    lower = max(int(lower), 0)
    upper = int(upper)
    dataframe.loc[dataframe[col_name] < lower, col_name] = lower
    dataframe.loc[dataframe[col_name] > upper, col_name] = upper


def fill_missing_values(dataframe):
    # Missing = "No feature" — fill with None
    none_cols = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu',
                 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',
                 'BsmtFinType2', 'MasVnrType']
    for col in none_cols:
        dataframe[col] = dataframe[col].fillna('None')

    # Missing = 0 — no area/count
    zero_cols = ['GarageYrBlt', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2',
                 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',
                 'GarageCars', 'GarageArea']
    for col in zero_cols:
        dataframe[col] = dataframe[col].fillna(0)

    # LotFrontage — group imputation by neighborhood median
    dataframe['LotFrontage'] = dataframe.groupby('Neighborhood')['LotFrontage'].transform(
        lambda x: x.fillna(x.median()))

    # Remaining categoricals — fill with mode
    mode_cols = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd',
                 'KitchenQual', 'Functional', 'SaleType', 'Electrical']
    for col in mode_cols:
        dataframe[col] = dataframe[col].fillna(dataframe[col].mode()[0])

    return dataframe


def apply_skew_transform(X_train, X_test, threshold=0.75):
    # Only apply to non-binary, non-uint8 columns
    numeric_feats = X_train.dtypes[X_train.dtypes != 'uint8'].index
    skewed_feats = X_train[numeric_feats].apply(lambda x: skew(x)).sort_values(ascending=False)
    skewed_feats = skewed_feats[skewed_feats > threshold]
    binary_cols = [col for col in skewed_feats.index if X_train[col].nunique() == 2]
    skewed_feats = skewed_feats.drop(binary_cols)
    print(f"Skew transform applied to {len(skewed_feats)} features")

    for col in skewed_feats.index:
        X_train[col] = np.log1p(X_train[col])
        X_test[col] = np.log1p(X_test[col])

    # Clean up any infinity values produced by log transform
    for df in [X_train, X_test]:
        inf_cols = df.columns[np.isinf(df).any()].tolist()
        for col in inf_cols:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].fillna(X_train[col].median())

    return X_train, X_test


def rmse_cv(model, X, y, cv=5):
    scores = cross_val_score(model, X, y,
                             scoring='neg_root_mean_squared_error',
                             cv=cv)
    return -scores.mean(), scores.std()

##############################################################
# 3. EXPLORATORY DATA ANALYSIS (EDA)
##############################################################

# Uncomment to run EDA
# first_check_df(df_train, plotna=True)

# fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# axes[0].hist(df_train['SalePrice'], bins=50, edgecolor='black')
# axes[0].set_title('SalePrice Distribution')
# axes[1].hist(np.log1p(df_train['SalePrice']), bins=50, edgecolor='black', color='orange')
# axes[1].set_title('Log(SalePrice) Distribution')
# plt.tight_layout()
# plt.show()

num_cols, cat_cols, num_but_cat, cat_but_car = analyze_col_names(df_train)
num_cols = [col for col in num_cols if col not in ['Id', 'SalePrice']]
num_but_cat = [col for col in num_but_cat if col != 'PoolArea']
num_cols.append('PoolArea')

# Uncomment to run category analysis
# for col in cat_cols:
#     cat_summary(df_train, col)
# rare_analyzer(df_train, 'SalePrice', cat_cols)

##############################################################
# 4. DATA PREPARATION
##############################################################

df_train.drop(['Id', 'SalePrice'], axis=1, inplace=True)
df_test.drop(['Id'], axis=1, inplace=True)
all_data = pd.concat([df_train, df_test], axis=0, ignore_index=True)
print(f"Combined dataset shape: {all_data.shape}")

all_data = fill_missing_values(all_data)
print(f"Missing values after filling: {all_data.isnull().sum().sum()}")

# Outlier capping with winsorization (q1=0.05, q3=0.95)
cols_to_cap = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1',
               'BsmtFinSF2', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea',
               'WoodDeckSF', 'OpenPorchSF']
for col in cols_to_cap:
    replace_with_thresholds(all_data, col, q1=0.05, q3=0.95)

##############################################################
# 5. FEATURE ENGINEERING
##############################################################

# Area-based features
all_data['TotalSF'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['TotalBath'] = (all_data['FullBath'] + all_data['HalfBath'] * 0.5 +
                         all_data['BsmtFullBath'] + all_data['BsmtHalfBath'] * 0.5)
all_data['TotalPorchSF'] = (all_data['OpenPorchSF'] + all_data['EnclosedPorch'] +
                            all_data['3SsnPorch'] + all_data['ScreenPorch'] +
                            all_data['WoodDeckSF'])
all_data['LivAreaPerRoom'] = all_data['GrLivArea'] / all_data['TotRmsAbvGrd']

# Time-based features
all_data['HouseAge'] = all_data['YrSold'] - all_data['YearBuilt']
all_data['RemodAge'] = all_data['YrSold'] - all_data['YearRemodAdd']
all_data['IsRemodeled'] = (all_data['YearBuilt'] != all_data['YearRemodAdd']).astype(int)
all_data['IsNew'] = (all_data['YearBuilt'] == all_data['YrSold']).astype(int)
all_data['GarageAge'] = all_data['YrSold'] - all_data['GarageYrBlt']

# Binary (has/hasn't) features
all_data['HasPool'] = (all_data['PoolArea'] > 0).astype(int)
all_data['HasGarage'] = (all_data['GarageArea'] > 0).astype(int)
all_data['HasBasement'] = (all_data['TotalBsmtSF'] > 0).astype(int)
all_data['HasFireplace'] = (all_data['Fireplaces'] > 0).astype(int)
all_data['Has2ndFloor'] = (all_data['2ndFlrSF'] > 0).astype(int)
all_data['HasWoodDeck'] = (all_data['WoodDeckSF'] > 0).astype(int)
all_data['HasPorch'] = (all_data['TotalPorchSF'] > 0).astype(int)
all_data['HasMasVnr'] = (all_data['MasVnrArea'] > 0).astype(int)
all_data['HasAlley'] = (all_data['Alley'] != 'None').astype(int)
all_data['HasFence'] = (all_data['Fence'] != 'None').astype(int)

# Quality score features
all_data['OverallScore'] = all_data['OverallQual'] * all_data['OverallCond']
all_data['GarageScore'] = all_data['GarageCars'] * all_data['GarageArea']
all_data['BsmtScore'] = all_data['TotalBsmtSF'] * all_data['BsmtFullBath']

##############################################################
# 6. ENCODING
##############################################################

# Ordinal encoding — quality columns
qual_map = {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5}
qual_cols = ['ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
             'HeatingQC', 'KitchenQual', 'FireplaceQu',
             'GarageQual', 'GarageCond', 'PoolQC']
for col in qual_cols:
    all_data[col] = all_data[col].map(qual_map)

all_data['BsmtExposure'] = all_data['BsmtExposure'].map(
    {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4})
all_data['GarageFinish'] = all_data['GarageFinish'].map(
    {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3})
all_data['Functional'] = all_data['Functional'].map(
    {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3,
     'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7})
all_data['PavedDrive'] = all_data['PavedDrive'].map({'N': 0, 'P': 1, 'Y': 2})
all_data['LotShape'] = all_data['LotShape'].map(
    {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3})

bsmt_fin_map = {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6}
all_data['BsmtFinType1'] = all_data['BsmtFinType1'].map(bsmt_fin_map)
all_data['BsmtFinType2'] = all_data['BsmtFinType2'].map(bsmt_fin_map)

# Label encoding
all_data['CentralAir'] = all_data['CentralAir'].map({'N': 0, 'Y': 1})
all_data['Alley'] = all_data['Alley'].map({'None': 0, 'Grvl': 1, 'Pave': 2})
all_data['LandSlope'] = all_data['LandSlope'].map({'Gtl': 0, 'Mod': 1, 'Sev': 2})
all_data['Fence'] = all_data['Fence'].map({'None': 0, 'MnWw': 1, 'MnPrv': 2, 'GdWo': 3, 'GdPrv': 4})

# Drop uninformative columns
drop_cols = ['Street', 'Utilities', 'PoolQC',
             'YearBuilt', 'YearRemodAdd', 'GarageYrBlt',
             'MoSold', 'YrSold']
all_data.drop(drop_cols, axis=1, inplace=True)

# Target encoding — MUST be done BEFORE one-hot encoding
target_enc_cols = ['Neighborhood', 'Exterior1st', 'Exterior2nd',
                   'SaleCondition', 'MSZoning']
te = ce.TargetEncoder(cols=target_enc_cols, smoothing=10)
train_part = all_data[:len(y)]
te.fit(train_part[target_enc_cols], y)
all_data[target_enc_cols] = te.transform(all_data[target_enc_cols])

# One-hot encoding — nominal columns
ohe_cols = ['LandContour', 'LotConfig', 'Condition1', 'Condition2',
            'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl',
            'MasVnrType', 'Foundation', 'Heating', 'Electrical',
            'GarageType', 'MiscFeature', 'SaleType']
all_data = pd.get_dummies(all_data, columns=ohe_cols, drop_first=True)
print(f"Shape after encoding: {all_data.shape}")

##############################################################
# 7. TRAIN / TEST SPLIT
##############################################################

X_train = all_data[:len(y)]
X_test = all_data[len(y):]
print(f"X_train : {X_train.shape}")
print(f"X_test  : {X_test.shape}")
print(f"y       : {y.shape}")

##############################################################
# 8. BASELINE MODEL COMPARISON
##############################################################

models = {
    'Ridge'       : Ridge(alpha=1.0),
    'Lasso'       : Lasso(alpha=0.001),
    'ElasticNet'  : ElasticNet(alpha=0.001, l1_ratio=0.5),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'GBM'         : GradientBoostingRegressor(n_estimators=100, random_state=42),
    'XGBoost'     : XGBRegressor(n_estimators=100, random_state=42, verbosity=0),
    'LightGBM'    : LGBMRegressor(n_estimators=100, random_state=42, verbose=-1),
}

results = {}
for name, model in models.items():
    mean, std = rmse_cv(model, X_train, y)
    results[name] = mean
    print(f"{name:<15} RMSE: {mean:.4f} ± {std:.4f}")
print(f"\nBest baseline model: {min(results, key=results.get)}")

##############################################################
# 9. HYPERPARAMETER OPTIMIZATION (OPTUNA)
##############################################################

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective_xgb(trial):
    params = {
        'n_estimators'     : trial.suggest_int('n_estimators', 100, 1000),
        'max_depth'        : trial.suggest_int('max_depth', 3, 8),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_weight' : trial.suggest_int('min_child_weight', 1, 10),
        'random_state'     : 42,
        'verbosity'        : 0
    }
    mean, _ = rmse_cv(XGBRegressor(**params), X_train, y)
    return mean

def objective_lgbm(trial):
    params = {
        'n_estimators'     : trial.suggest_int('n_estimators', 100, 1000),
        'max_depth'        : trial.suggest_int('max_depth', 3, 8),
        'learning_rate'    : trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample'        : trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree' : trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'num_leaves'       : trial.suggest_int('num_leaves', 20, 100),
        'random_state'     : 42,
        'verbose'          : -1
    }
    mean, _ = rmse_cv(LGBMRegressor(**params), X_train, y)
    return mean

def objective_gbm(trial):
    params = {
        'n_estimators'    : trial.suggest_int('n_estimators', 100, 1000),
        'max_depth'       : trial.suggest_int('max_depth', 3, 8),
        'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
        'max_features'    : trial.suggest_float('max_features', 0.3, 1.0),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'random_state'    : 42
    }
    mean, _ = rmse_cv(GradientBoostingRegressor(**params), X_train, y)
    return mean

def objective_cat(trial):
    params = {
        'iterations'   : trial.suggest_int('iterations', 100, 1000),
        'depth'        : trial.suggest_int('depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'subsample'    : trial.suggest_float('subsample', 0.6, 1.0),
        'l2_leaf_reg'  : trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_seed'  : 42,
        'verbose'      : 0
    }
    mean, _ = rmse_cv(CatBoostRegressor(**params), X_train, y)
    return mean

study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=200)
print(f"XGBoost  → RMSE: {study_xgb.best_value:.4f} | Params: {study_xgb.best_params}")

study_lgbm = optuna.create_study(direction='minimize')
study_lgbm.optimize(objective_lgbm, n_trials=200)
print(f"LightGBM → RMSE: {study_lgbm.best_value:.4f} | Params: {study_lgbm.best_params}")

study_gbm = optuna.create_study(direction='minimize')
study_gbm.optimize(objective_gbm, n_trials=200)
print(f"GBM      → RMSE: {study_gbm.best_value:.4f} | Params: {study_gbm.best_params}")

study_cat = optuna.create_study(direction='minimize')
study_cat.optimize(objective_cat, n_trials=200)
print(f"CatBoost → RMSE: {study_cat.best_value:.4f} | Params: {study_cat.best_params}")

##############################################################
# 10. FEATURE SELECTION
##############################################################

lgbm_best = LGBMRegressor(**study_lgbm.best_params, random_state=42, verbose=-1)
lgbm_best.fit(X_train, y)

importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': lgbm_best.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 10 important features:")
print(importance.head(10))
print(f"\nZero importance features: {(importance['importance'] == 0).sum()}")

zero_importance_cols = importance[importance['importance'] == 0]['feature'].tolist()
X_train_final = X_train.drop(columns=zero_importance_cols)
X_test_final = X_test.drop(columns=zero_importance_cols)
print(f"Features after selection: {X_train_final.shape[1]}")

##############################################################
# 11. SKEWNESS TRANSFORMATION
##############################################################

X_train_final, X_test_final = apply_skew_transform(X_train_final, X_test_final, threshold=0.75)

##############################################################
# 12. STACKING
##############################################################

xgb_best = XGBRegressor(**study_xgb.best_params, random_state=42, verbosity=0)
lgbm_best = LGBMRegressor(**study_lgbm.best_params, random_state=42, verbose=-1)
gbm_best = GradientBoostingRegressor(**study_gbm.best_params, random_state=42)
cat_best = CatBoostRegressor(**study_cat.best_params, random_seed=42, verbose=0)
elastic_best = ElasticNetCV(
    l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
    alphas=[0.0001, 0.001, 0.01, 0.1],
    cv=5,
    max_iter=10000
)

stack_model = StackingRegressor(
    estimators=[
        ('xgb', xgb_best),
        ('lgbm', lgbm_best),
        ('gbm', gbm_best),
        ('cat', cat_best),
        ('elastic', elastic_best)
    ],
    final_estimator=RidgeCV(),
    cv=5
)

mean, std = rmse_cv(stack_model, X_train_final, y)
print(f"Stacking CV RMSE: {mean:.4f} ± {std:.4f}")

##############################################################
# 13. FINAL PREDICTION & SUBMISSION
##############################################################

stack_model.fit(X_train_final, y)
y_pred = np.expm1(stack_model.predict(X_test_final))

submission = pd.DataFrame({'Id': test_ID, 'SalePrice': y_pred})
submission.to_csv('submission_final.csv', index=False)
print(f"submission_final.csv ready | Shape: {submission.shape}")
