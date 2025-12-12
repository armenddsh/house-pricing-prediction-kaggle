---

# 🏠 House Prices - Advanced Regression Techniques
## Predict sales prices and practice feature engineering, RFs, and gradient boosting


## 🎯 Target Variable

* **SalePrice** — The property's sale price in dollars (target variable).

---

## 🏗️ Property & Building Information

* **MSSubClass** — Building class
* **MSZoning** — General zoning classification
* **LotFrontage** — Linear feet of street connected to property
* **LotArea** — Lot size (square feet)
* **Street** — Road access type
* **Alley** — Alley access type
* **LotShape** — General lot shape
* **LandContour** — Flatness of the property
* **Utilities** — Utilities available
* **LotConfig** — Lot configuration
* **LandSlope** — Slope of property
* **Neighborhood** — Physical location within Ames
* **Condition1** — Proximity to main road or railroad
* **Condition2** — Additional proximity feature (if present)
* **BldgType** — Type of dwelling
* **HouseStyle** — Dwelling style

---

## 🧱 Structure Quality & Condition

* **OverallQual** — Overall material and finish quality
* **OverallCond** — Overall condition
* **YearBuilt** — Original construction year
* **YearRemodAdd** — Remodel year
* **RoofStyle** — Roof style
* **RoofMatl** — Roof material
* **Exterior1st** — Primary exterior covering
* **Exterior2nd** — Secondary exterior covering
* **MasVnrType** — Masonry veneer type
* **MasVnrArea** — Masonry veneer area
* **ExterQual** — Exterior material quality
* **ExterCond** — Exterior condition
* **Foundation** — Foundation type

---

## 🏚️ Basement Features

* **BsmtQual** — Basement height
* **BsmtCond** — Basement condition
* **BsmtExposure** — Walkout/garden-level exposure
* **BsmtFinType1** — Type 1 finished basement area
* **BsmtFinSF1** — Finished area (Type 1, square feet)
* **BsmtFinType2** — Type 2 finished basement area
* **BsmtFinSF2** — Finished area (Type 2, square feet)
* **BsmtUnfSF** — Unfinished basement area
* **TotalBsmtSF** — Total basement square footage

---

## 🔥 Heating & Utilities

* **Heating** — Heating type
* **HeatingQC** — Heating quality/condition
* **CentralAir** — Central A/C presence
* **Electrical** — Electrical system

---

## 🏡 Living Area & Rooms

* **1stFlrSF** — First-floor square feet
* **2ndFlrSF** — Second-floor square feet
* **LowQualFinSF** — Low-quality finished area (all floors)
* **GrLivArea** — Above-grade living area (square feet)
* **BsmtFullBath** — Basement full bathrooms
* **BsmtHalfBath** — Basement half bathrooms
* **FullBath** — Above-grade full bathrooms
* **HalfBath** — Above-grade half bathrooms
* **Bedroom** — Bedrooms above basement level
* **Kitchen** — Number of kitchens
* **KitchenQual** — Kitchen quality
* **TotRmsAbvGrd** — Total rooms above grade (excluding bathrooms)
* **Functional** — Home functionality rating

---

## 🔥 Fireplaces

* **Fireplaces** — Number of fireplaces
* **FireplaceQu** — Fireplace quality

---

## 🚗 Garage Features

* **GarageType** — Garage location
* **GarageYrBlt** — Year garage was built
* **GarageFinish** — Interior garage finish
* **GarageCars** — Garage capacity (cars)
* **GarageArea** — Garage size (square feet)
* **GarageQual** — Garage quality
* **GarageCond** — Garage condition

---

## 🌳 Outdoor & Miscellaneous Features

* **PavedDrive** — Paved driveway indicator
* **WoodDeckSF** — Wood deck area (square feet)
* **OpenPorchSF** — Open porch area
* **EnclosedPorch** — Enclosed porch area
* **3SsnPorch** — Three-season porch area
* **ScreenPorch** — Screen porch area
* **PoolArea** — Pool area (square feet)
* **PoolQC** — Pool quality
* **Fence** — Fence quality
* **MiscFeature** — Miscellaneous feature
* **MiscVal** — Value of miscellaneous feature

---

## 🗓️ Sale Information

* **MoSold** — Month sold
* **YrSold** — Year sold
* **SaleType** — Type of sale
* **SaleCondition** — Condition of sale

---

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
TRAIN DATA
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1460 entries, 0 to 1459
Data columns (total 81 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             1460 non-null   int64  
 1   MSSubClass     1460 non-null   int64  
 2   MSZoning       1460 non-null   object 
 3   LotFrontage    1201 non-null   float64
 4   LotArea        1460 non-null   int64  
 5   Street         1460 non-null   object 
 6   Alley          91 non-null     object 
 7   LotShape       1460 non-null   object 
 8   LandContour    1460 non-null   object 
 9   Utilities      1460 non-null   object 
 10  LotConfig      1460 non-null   object 
 11  LandSlope      1460 non-null   object 
 12  Neighborhood   1460 non-null   object 
 13  Condition1     1460 non-null   object 
 14  Condition2     1460 non-null   object 
 15  BldgType       1460 non-null   object 
 16  HouseStyle     1460 non-null   object 
 17  OverallQual    1460 non-null   int64  
 18  OverallCond    1460 non-null   int64  
 19  YearBuilt      1460 non-null   int64  
 20  YearRemodAdd   1460 non-null   int64  
 21  RoofStyle      1460 non-null   object 
 22  RoofMatl       1460 non-null   object 
 23  Exterior1st    1460 non-null   object 
 24  Exterior2nd    1460 non-null   object 
 25  MasVnrType     588 non-null    object 
 26  MasVnrArea     1452 non-null   float64
 27  ExterQual      1460 non-null   object 
 28  ExterCond      1460 non-null   object 
 29  Foundation     1460 non-null   object 
 30  BsmtQual       1423 non-null   object 
 31  BsmtCond       1423 non-null   object 
 32  BsmtExposure   1422 non-null   object 
 33  BsmtFinType1   1423 non-null   object 
 34  BsmtFinSF1     1460 non-null   int64  
 35  BsmtFinType2   1422 non-null   object 
 36  BsmtFinSF2     1460 non-null   int64  
 37  BsmtUnfSF      1460 non-null   int64  
 38  TotalBsmtSF    1460 non-null   int64  
 39  Heating        1460 non-null   object 
 40  HeatingQC      1460 non-null   object 
 41  CentralAir     1460 non-null   object 
 42  Electrical     1459 non-null   object 
 43  1stFlrSF       1460 non-null   int64  
 44  2ndFlrSF       1460 non-null   int64  
 45  LowQualFinSF   1460 non-null   int64  
 46  GrLivArea      1460 non-null   int64  
 47  BsmtFullBath   1460 non-null   int64  
 48  BsmtHalfBath   1460 non-null   int64  
 49  FullBath       1460 non-null   int64  
 50  HalfBath       1460 non-null   int64  
 51  BedroomAbvGr   1460 non-null   int64  
 52  KitchenAbvGr   1460 non-null   int64  
 53  KitchenQual    1460 non-null   object 
 54  TotRmsAbvGrd   1460 non-null   int64  
 55  Functional     1460 non-null   object 
 56  Fireplaces     1460 non-null   int64  
 57  FireplaceQu    770 non-null    object 
 58  GarageType     1379 non-null   object 
 59  GarageYrBlt    1379 non-null   float64
 60  GarageFinish   1379 non-null   object 
 61  GarageCars     1460 non-null   int64  
 62  GarageArea     1460 non-null   int64  
 63  GarageQual     1379 non-null   object 
 64  GarageCond     1379 non-null   object 
 65  PavedDrive     1460 non-null   object 
 66  WoodDeckSF     1460 non-null   int64  
 67  OpenPorchSF    1460 non-null   int64  
 68  EnclosedPorch  1460 non-null   int64  
 69  3SsnPorch      1460 non-null   int64  
 70  ScreenPorch    1460 non-null   int64  
 71  PoolArea       1460 non-null   int64  
 72  PoolQC         7 non-null      object 
 73  Fence          281 non-null    object 
 74  MiscFeature    54 non-null     object 
 75  MiscVal        1460 non-null   int64  
 76  MoSold         1460 non-null   int64  
 77  YrSold         1460 non-null   int64  
 78  SaleType       1460 non-null   object 
 79  SaleCondition  1460 non-null   object 
 80  SalePrice      1460 non-null   int64  
dtypes: float64(3), int64(35), object(43)
memory usage: 924.0+ KB


TEST DATA
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1459 entries, 0 to 1458
Data columns (total 80 columns):
 #   Column         Non-Null Count  Dtype  
---  ------         --------------  -----  
 0   Id             1459 non-null   int64  
 1   MSSubClass     1459 non-null   int64  
 2   MSZoning       1455 non-null   object 
 3   LotFrontage    1232 non-null   float64
 4   LotArea        1459 non-null   int64  
 5   Street         1459 non-null   object 
 6   Alley          107 non-null    object 
 7   LotShape       1459 non-null   object 
 8   LandContour    1459 non-null   object 
 9   Utilities      1457 non-null   object 
 10  LotConfig      1459 non-null   object 
 11  LandSlope      1459 non-null   object 
 12  Neighborhood   1459 non-null   object 
 13  Condition1     1459 non-null   object 
 14  Condition2     1459 non-null   object 
 15  BldgType       1459 non-null   object 
 16  HouseStyle     1459 non-null   object 
 17  OverallQual    1459 non-null   int64  
 18  OverallCond    1459 non-null   int64  
 19  YearBuilt      1459 non-null   int64  
 20  YearRemodAdd   1459 non-null   int64  
 21  RoofStyle      1459 non-null   object 
 22  RoofMatl       1459 non-null   object 
 23  Exterior1st    1458 non-null   object 
 24  Exterior2nd    1458 non-null   object 
 25  MasVnrType     565 non-null    object 
 26  MasVnrArea     1444 non-null   float64
 27  ExterQual      1459 non-null   object 
 28  ExterCond      1459 non-null   object 
 29  Foundation     1459 non-null   object 
 30  BsmtQual       1415 non-null   object 
 31  BsmtCond       1414 non-null   object 
 32  BsmtExposure   1415 non-null   object 
 33  BsmtFinType1   1417 non-null   object 
 34  BsmtFinSF1     1458 non-null   float64
 35  BsmtFinType2   1417 non-null   object 
 36  BsmtFinSF2     1458 non-null   float64
 37  BsmtUnfSF      1458 non-null   float64
 38  TotalBsmtSF    1458 non-null   float64
 39  Heating        1459 non-null   object 
 40  HeatingQC      1459 non-null   object 
 41  CentralAir     1459 non-null   object 
 42  Electrical     1459 non-null   object 
 43  1stFlrSF       1459 non-null   int64  
 44  2ndFlrSF       1459 non-null   int64  
 45  LowQualFinSF   1459 non-null   int64  
 46  GrLivArea      1459 non-null   int64  
 47  BsmtFullBath   1457 non-null   float64
 48  BsmtHalfBath   1457 non-null   float64
 49  FullBath       1459 non-null   int64  
 50  HalfBath       1459 non-null   int64  
 51  BedroomAbvGr   1459 non-null   int64  
 52  KitchenAbvGr   1459 non-null   int64  
 53  KitchenQual    1458 non-null   object 
 54  TotRmsAbvGrd   1459 non-null   int64  
 55  Functional     1457 non-null   object 
 56  Fireplaces     1459 non-null   int64  
 57  FireplaceQu    729 non-null    object 
 58  GarageType     1383 non-null   object 
 59  GarageYrBlt    1381 non-null   float64
 60  GarageFinish   1381 non-null   object 
 61  GarageCars     1458 non-null   float64
 62  GarageArea     1458 non-null   float64
 63  GarageQual     1381 non-null   object 
 64  GarageCond     1381 non-null   object 
 65  PavedDrive     1459 non-null   object 
 66  WoodDeckSF     1459 non-null   int64  
 67  OpenPorchSF    1459 non-null   int64  
 68  EnclosedPorch  1459 non-null   int64  
 69  3SsnPorch      1459 non-null   int64  
 70  ScreenPorch    1459 non-null   int64  
 71  PoolArea       1459 non-null   int64  
 72  PoolQC         3 non-null      object 
 73  Fence          290 non-null    object 
 74  MiscFeature    51 non-null     object 
 75  MiscVal        1459 non-null   int64  
 76  MoSold         1459 non-null   int64  
 77  YrSold         1459 non-null   int64  
 78  SaleType       1458 non-null   object 
 79  SaleCondition  1459 non-null   object 
dtypes: float64(11), int64(26), object(43)
memory usage: 912.0+ KB
```

# Column Analysis

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


# Garage Features Analysis

```python
df_train["HasGarage"] = df_train["GarageType"].notna().astype(int)
df_test["HasGarage"] = df_test["GarageType"].notna().astype(int)

cat_cols = ["GarageQual", "GarageCond", "GarageFinish", "GarageType"]
df_train[cat_cols] = df_train[cat_cols].fillna("None")
df_test[cat_cols] = df_test[cat_cols].fillna("None")

df_train["GarageYrBltMissing"] = df_train["GarageYrBlt"].isna().astype(int)
df_test["GarageYrBltMissing"] = df_test["GarageYrBlt"].isna().astype(int)

df_train["GarageYrBlt"] = df_train["GarageYrBlt"].fillna(0)
df_test["GarageYrBlt"] = df_test["GarageYrBlt"].fillna(0)

num_cols = ["GarageCars", "GarageArea"]
df_train[num_cols] = df_train[num_cols].fillna(0)
df_test[num_cols] = df_test[num_cols].fillna(0)

```

# Bsmt Features Analysis

```python
cat_cols = ["BsmtCond", "BsmtQual", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]

df_train[cat_cols] = df_train[cat_cols].fillna("None")
df_test[cat_cols] = df_test[cat_cols].fillna("None")

num_cols = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
df_train[num_cols] = df_train[num_cols].fillna(0)
df_test[num_cols] = df_test[num_cols].fillna(0)

```

## MasVnr Features Analysis

```python
df_train["HasMasVnr"] = df_train["MasVnrType"].notna().astype(int)
df_test["HasMasVnr"] = df_test["MasVnrType"].notna().astype(int)

df_train["MasVnrType"] = df_train["MasVnrType"].fillna("None")
df_test["MasVnrType"] = df_test["MasVnrType"].fillna("None")

df_train["MasVnrArea"] = df_train["MasVnrArea"].fillna(0)
df_test["MasVnrArea"] = df_test["MasVnrArea"].fillna(0)
```


# MSZoning Analysis

```python
df_train['MSZoning'] = df_train['MSZoning'].fillna(df_train['MSZoning'].mode()[0])
df_test['MSZoning'] = df_test['MSZoning'].fillna(df_test['MSZoning'].mode()[0])
```

# Electrial Analysis

```python
df_train["Electrical"] = df_train["Electrical"].fillna(df_train["Electrical"].mode()[0])
df_test["Electrical"] = df_test["Electrical"].fillna(df_test["Electrical"].mode()[0])
```



**Checking for Outliers:**

### MSSubClass

Data doesn't have Outliers

### LotFrontage

Data does have Outliers

### OverallQual

Data does have Outliers

### OverallCond

Data does have Outliers

### YearBuilt

Data does have Outliers

### YearRemodAdd

Data doesn't have Outliers

### MasVnrArea

Data does have Outliers

### BsmtFinSF1

Data does have Outliers

### BsmtFinSF2

Data does have Outliers

### BsmtUnfSF

Data does have Outliers

### TotalBsmtSF

Data does have Outliers

### 1stFlrSF

Data does have Outliers

### 2ndFlrSF

Data does have Outliers

### LowQualFinSF

Data does have Outliers

### GrLivArea

Data does have Outliers

### BsmtFullBath

Data does have Outliers

### BsmtHalfBath

Data does have Outliers

### FullBath

Data doesn't have Outliers

### HalfBath

Data doesn't have Outliers

### BedroomAbvGr

Data does have Outliers

### KitchenAbvGr

Data does have Outliers

### TotRmsAbvGrd

Data does have Outliers

### Fireplaces

Data does have Outliers

### GarageYrBlt

Data does have Outliers

### GarageCars

Data does have Outliers

### GarageArea

Data does have Outliers

### WoodDeckSF

Data does have Outliers

### OpenPorchSF

Data does have Outliers

### EnclosedPorch

Data does have Outliers

### 3SsnPorch

Data does have Outliers

### ScreenPorch

Data does have Outliers

### PoolArea

Data does have Outliers

### MiscVal

Data does have Outliers

### MoSold

Data doesn't have Outliers

### YrSold

Data doesn't have Outliers

### SalePrice

Data does have Outliers

### HasGarage

Data does have Outliers

### GarageYrBltMissing

Data does have Outliers

### HasMasVnr

Data doesn't have Outliers


## Correlation Analysis

```

--- Features Sorted by Individual Correlation Strength with SalePrice ---
OverallQual           0.798851
GrLivArea             0.716765
GarageCars            0.643753
GarageArea            0.623376
TotalBsmtSF           0.602280
1stFlrSF              0.595744
FullBath              0.573884
TotRmsAbvGrd          0.550273
YearBuilt             0.543038
YearRemodAdd          0.516374
MasVnrArea            0.501624
Fireplaces            0.460885
HasMasVnr             0.386434
BsmtFinSF1            0.385646
2ndFlrSF              0.350759
WoodDeckSF            0.327561
LotFrontage           0.323524
OpenPorchSF           0.317200
HalfBath              0.315228
LotArea               0.263986
GarageYrBlt           0.257783
BsmtFullBath          0.234033
GarageYrBltMissing    0.230593
HasGarage             0.230593
BsmtUnfSF             0.197938
BedroomAbvGr          0.165741
EnclosedPorch         0.140782
KitchenAbvGr          0.123441
ScreenPorch           0.105800
PoolArea              0.091192
OverallCond           0.088555
MSSubClass            0.064533
3SsnPorch             0.040076
MoSold                0.039096
LowQualFinSF          0.028852
YrSold                0.025836
BsmtHalfBath          0.023321
Id                    0.023270
MiscVal               0.018621
BsmtFinSF2            0.015220
Name: SalePrice, dtype: float64

```