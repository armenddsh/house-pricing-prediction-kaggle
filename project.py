import random
import pandas as pd
import seaborn as sns

import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor

import warnings
warnings.filterwarnings('ignore')

import logging
import sys

def setup_logging(log_file="training.log"):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers (important in notebooks)
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # ---- File handler ----
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # ---- Stdout handler ----
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger

logger = setup_logging("kaggle_training.log")

def clean_data(df):
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
    df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
    
    drop_columns = [
        "Id",
        "Alley",
        # "MasVnrType",
        "PoolQC",
        "Fence",
        "MiscFeature"   
    ]

    df = df.drop(drop_columns, axis=1)
    
    df["HasGarage"] = df["GarageType"].notna().astype(int)

    cat_cols = ["GarageQual", "GarageCond", "GarageFinish", "GarageType"]
    df[cat_cols] = df[cat_cols].fillna("None")

    df["GarageYrBltMissing"] = df["GarageYrBlt"].isna().astype(int)

    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(0)

    num_cols = ["GarageCars", "GarageArea"]
    df[num_cols] = df[num_cols].fillna(0)

    cat_cols = ["BsmtCond", "BsmtQual", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]

    df[cat_cols] = df[cat_cols].fillna("None")

    num_cols = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
    df[num_cols] = df[num_cols].fillna(0)

    df['OpenPorchSF'] = df['OpenPorchSF'].fillna(0)
    df["MasVnrType"] = df["MasVnrType"].fillna("None")
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    df['MSZoning'] = df['MSZoning'].fillna(df['MSZoning'].mode()[0])
    df["Electrical"] = df["Electrical"].fillna(df["Electrical"].mode()[0])

    return df

def categorize_columns(df):
    categorical_cols_to_encode = [
        'Street', 'Alley', 'BsmtCond', 'MasVnrType',
        'BsmtQual', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
        'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish',
        'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature',
        'MSZoning', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',
        'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',
        'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd',
        'ExterQual', 'ExterCond', 'Foundation', 'Heating', 'HeatingQC',
        'CentralAir', 'KitchenQual', 'Functional', 'PavedDrive',
        'SaleType', 'SaleCondition'
    ]

    df = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)
    
    return df


def prepare_clean_data(df):
  df = df.drop_duplicates()

  df['BsmtCond'] = df['BsmtCond'].fillna('NA')
  df['BsmtCond'].unique()

  median_frontage = df['LotFrontage'].median()
  df['LotFrontage'] = df['LotFrontage'].fillna(median_frontage)

  masVnrArea_mean = df['MasVnrArea'].mean()
  df['MasVnrArea'] = df['MasVnrArea'].fillna(masVnrArea_mean)

  garageYrBlt_mean = df['GarageYrBlt'].mean()
  df['GarageYrBlt'] = df['GarageYrBlt'].fillna(garageYrBlt_mean)
  
  df['BsmtHalfBath'] = df['BsmtHalfBath'].fillna(0)  
  df['1stFlrSF'] = df['1stFlrSF'].fillna(0)  
  df['TotalBsmtSF'] = df['TotalBsmtSF'].fillna(0)  
  df['BsmtUnfSF'] = df['BsmtUnfSF'].fillna(0)  
  df['BsmtFinSF2'] = df['BsmtFinSF2'].fillna(0)


  df['MasVnrType'] = df['MasVnrType'].fillna('None')
  df['BsmtQual'] = df['BsmtQual'].fillna('None')
  df['BsmtExposure'] = df['BsmtExposure'].fillna('None')
  df['BsmtFinType1'] = df['BsmtFinType1'].fillna('None')
  df['BsmtFinType2'] = df['BsmtFinType2'].fillna('None')
  df['Electrical'] = df['Electrical'].fillna('None')
  df['FireplaceQu'] = df['FireplaceQu'].fillna('None')
  df['GarageType'] = df['GarageType'].fillna('None')
  df['GarageFinish'] = df['GarageFinish'].fillna('None')
  df['GarageQual'] = df['GarageQual'].fillna('None')
  df['GarageCond'] = df['GarageCond'].fillna('None')
  df['PoolQC'] = df['PoolQC'].fillna('None')
  df['Fence'] = df['Fence'].fillna('None')
  df['MiscFeature'] = df['MiscFeature'].fillna('None')
  df['Alley'] = df['Alley'].fillna('None')

  return df

def remove_outliers(df, columns):
    for col in columns:
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        upper_bound = Q3 + 1.5 * IQR
        lower_bound = Q3 - 1.5 * IQR

        median_value = df[col].median()

        outliers = (df[col] > upper_bound) | (df[col] < lower_bound)

        df.loc[outliers, col] = median_value
    
    return df

columns = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "GarageArea",
    "TotalBsmtSF",
    "1stFlrSF",
    "FullBath",
    "YearBuilt",
    "GarageYrBlt",
    "YearRemodAdd",
    "TotRmsAbvGrd",
    "MasVnrArea",
    "Fireplaces",
    "LotArea",
    "OpenPorchSF",
    "BsmtFinSF1",
    "LotFrontage",
    "2ndFlrSF",
    "WoodDeckSF",
    "HalfBath",
    "BsmtFullBath",
    "BsmtUnfSF",
    "BedroomAbvGr",
    "OverallCond",
    "MSSubClass",
    "MoSold",
    "YrSold"
]


while True:
    random_num = random.randint(0, 8)
    columns_combination = set()
    
    for _ in range(random_num):
        columns_combination.add(random.choice(columns))
    else:
        columns_to_test = list(columns_combination)
        
        logger.info(f'\nColumns to test: {columns_combination}')

        df_test = pd.read_csv('./data/test.csv')
        df_test.head()

        df_train = pd.read_csv('./data/train.csv')
        df_train.head()
        
        df_train = prepare_clean_data(df_train)
        df_test = prepare_clean_data(df_test)

        df_train = categorize_columns(df_train)
        df_test = categorize_columns(df_test)

        df_train = remove_outliers(df_train, columns_to_test)
        df_test = remove_outliers(df_test, columns_to_test)

        feature_columns = [
            "OverallQual",
            "GrLivArea",
            "GarageCars",
            "GarageArea",
            "TotalBsmtSF",
            "1stFlrSF",
            "FullBath",
            "YearBuilt",
            "GarageYrBlt",
            "YearRemodAdd",
            "TotRmsAbvGrd",
            "MasVnrArea",
            "Fireplaces",
            "LotArea",
            "OpenPorchSF",
            "BsmtFinSF1",
            "LotFrontage",
            "2ndFlrSF",
            "WoodDeckSF",
            "HalfBath",
            "BsmtFullBath",
            "BsmtUnfSF",
            "BedroomAbvGr",
            "OverallCond",
            "MSSubClass",
            "MoSold",
            "YrSold"
        ]

        target_column = "SalePrice"

        X = df_train[feature_columns]
        y = np.log1p(df_train[target_column])  # log transform

        numeric_preprocess = Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_preprocess, feature_columns)
        ])

        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", GradientBoostingRegressor(random_state=42))
        ])

        param_grid = {
            "model__n_estimators": [800], # 1200, 1600
            "model__learning_rate": [0.01], # 0.05
            "model__max_depth": [3], # 4
            "model__min_samples_leaf": [5] # 10
        }

        grid = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="neg_root_mean_squared_error",
            cv=5,
            n_jobs=-1,
            verbose=1
        )

        grid.fit(X, y)

        logger.info(f"Best CV RMSE: {-grid.best_score_}")
        logger.info(f"Best params: {grid.best_params_}")

        best_model = grid.best_estimator_
        best_model.fit(X, y)

        X_test = df_test[feature_columns]
        test_preds_log = best_model.predict(X_test)
        test_predictions = np.expm1(test_preds_log)

        # submission = pd.DataFrame({
        #     "Id": df_test["Id"],
        #     "SalePrice": test_predictions
        # })

        # submission.to_csv("submission.csv", index=False)
        # print("Submission rows:", len(submission))