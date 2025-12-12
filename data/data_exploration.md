# Data Exploration

## Data Description

Okay, here's an analysis of the provided DataFrame information:

**Overall Structure:**

*   The DataFrame has 1459 rows (observations) and 80 columns (features).
*   It appears to represent data about houses or properties, likely for a housing price prediction task (based on the column names).

**Data Types:**

*   **Numerical:** There are 37 numerical columns (int64 and float64). These likely represent measurements like area, number of rooms, year built, etc.
*   **Categorical/Object:** There are 43 columns with the 'object' data type. These are likely categorical features (e.g., neighborhood, building type, sale condition).

**Missing Values:**
### Train Data missing values

Should be dropped because of high percentage of missing values
```txt
PoolQC: 1456 / 1459
MiscFeature: 1408 / 1459
Alley: 1352 / 1459
Fence: 1169 / 1459
```

Should be imputed
```txt
MasVnrType: 894 / 1459
FireplaceQu: 730 / 1459
LotFrontage: 227 / 1459
GarageQual: 78 / 1459
GarageCond: 78 / 1459
GarageYrBlt: 78 / 1459
GarageFinish: 78 / 1459
GarageType: 76 / 1459
BsmtCond: 45 / 1459
BsmtQual: 44 / 1459
BsmtExposure: 44 / 1459
BsmtFinType1: 42 / 1459
BsmtFinType2: 42 / 1459
MasVnrArea: 15 / 1459
MSZoning: 4 / 1459
Utilities: 2 / 1459
Functional: 2 / 1459
BsmtFullBath: 2 / 1459
BsmtHalfBath: 2 / 1459
SaleType: 1 / 1459
BsmtUnfSF: 1 / 1459
BsmtFinSF1: 1 / 1459
BsmtFinSF2: 1 / 1459
GarageCars: 1 / 1459
GarageArea: 1 / 1459
Exterior1st: 1 / 1459
Exterior2nd: 1 / 1459
TotalBsmtSF: 1 / 1459
KitchenQual: 1 / 1459
```

### Test Data missing values

Should be dropped because of high percentage of missing values

```txt
PoolQC: 1456 / 1459
MiscFeature: 1408 / 1459
Alley: 1352 / 1459
Fence: 1169 / 1459
```

Should be imputed
```txt
MasVnrType: 894 / 1459
FireplaceQu: 730 / 1459
LotFrontage: 227 / 1459
GarageQual: 78 / 1459
GarageCond: 78 / 1459
GarageYrBlt: 78 / 1459
GarageFinish: 78 / 1459
GarageType: 76 / 1459
BsmtCond: 45 / 1459
BsmtQual: 44 / 1459
BsmtExposure: 44 / 1459
BsmtFinType1: 42 / 1459
BsmtFinType2: 42 / 1459
MasVnrArea: 15 / 1459
MSZoning: 4 / 1459
Utilities: 2 / 1459
Functional: 2 / 1459
BsmtFullBath: 2 / 1459
BsmtHalfBath: 2 / 1459
SaleType: 1 / 1459
BsmtUnfSF: 1 / 1459
BsmtFinSF1: 1 / 1459
BsmtFinSF2: 1 / 1459
GarageCars: 1 / 1459
GarageArea: 1 / 1459
Exterior1st: 1 / 1459
Exterior2nd: 1 / 1459
TotalBsmtSF: 1 / 1459
KitchenQual: 1 / 1459
```

# Column Analysis

## MasVnrType Analysis

```python
df_train['MasVnrType'] = df_train['MasVnrType'].fillna('None')
df_test['MasVnrType'] = df_test['MasVnrType'].fillna('None')
```

## FireplaceQu Analysis

```python
df_train['FireplaceQu'] = df_train['FireplaceQu'].fillna('None')
df_test['FireplaceQu'] = df_test['FireplaceQu'].fillna('None')
```


## LotFrontage Analysis

```python
df_train['LotFrontage'] = df_train['LotFrontage'].fillna(df_train['LotFrontage'].median())
df_test['LotFrontage'] = df_test['LotFrontage'].fillna(df_test['LotFrontage'].median())
```

or maybe dropping the column because it has a lot of missing values and it's not a very important feature


# GarageQual Analysis

# GarageCond Analysis

# GarageYrBlt Analysis

# GarageFinish Analysis

# GarageType Analysis

# BsmtCond Analysis

# BsmtQual Analysis

# BsmtExposure Analysis

# BsmtFinType1 Analysis

# BsmtFinType2 Analysis

# MasVnrArea Analysis

# MSZoning Analysis

# Utilities Analysis

# Functional Analysis

# BsmtFullBath Analysis

# BsmtHalfBath Analysis

# SaleType Analysis

# BsmtUnfSF Analysis

# BsmtFinSF1 Analysis

# BsmtFinSF2 Analysis

# GarageCars Analysis

# GarageArea Analysis

# Exterior1st Analysis

# Exterior2nd Analysis

# TotalBsmtSF Analysis

# KitchenQual Analysis


**Checking for Outliers:**

