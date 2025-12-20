import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.impute import SimpleImputer


MODEL_PATH = "best_model.joblib"
FEATURE_COLUMNS = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "1stFlrSF",
    "YearBuilt",
    "YearRemodAdd",
    "Fireplaces",
    "LotArea",
    "BsmtFinSF1",
    "OverallCond",
]
TRAIN_PATH = "data/train.csv"


@st.cache_resource
def load_model(path: str):
    return joblib.load(path)


def ensure_preprocessor_compatibility(model, train_path: str, feature_columns):
    preprocess = getattr(model, "named_steps", {}).get("preprocess")
    if preprocess is None:
        return

    transformers = getattr(preprocess, "transformers_", None) or preprocess.transformers
    needs_fix = False
    for _, transformer, _ in transformers:
        if transformer in ("drop", "passthrough"):
            continue
        if isinstance(transformer, SimpleImputer):
            if not hasattr(transformer, "_fill_dtype"):
                needs_fix = True
        elif hasattr(transformer, "named_steps"):
            for step in transformer.named_steps.values():
                if isinstance(step, SimpleImputer) and not hasattr(step, "_fill_dtype"):
                    needs_fix = True

    if not needs_fix:
        return
    if not os.path.exists(train_path):
        st.warning(
            "Model was trained with an older scikit-learn; "
            "could not refresh imputers because data/train.csv is missing."
        )
        return

    df = pd.read_csv(train_path)
    preprocess.fit(df[feature_columns])


@st.cache_data
def load_feature_defaults(path: str):
    if not os.path.exists(path):
        return {col: 0.0 for col in FEATURE_COLUMNS}
    df = pd.read_csv(path)
    defaults = {}
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            defaults[col] = float(df[col].median())
        else:
            defaults[col] = 0.0
    return defaults


st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
st.title("House Price Predictor")
st.caption("GradientBoostingRegressor trained on Ames housing data.")

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()

model = load_model(MODEL_PATH)
if "preprocess_checked" not in st.session_state:
    ensure_preprocessor_compatibility(model, TRAIN_PATH, FEATURE_COLUMNS)
    st.session_state["preprocess_checked"] = True
defaults = load_feature_defaults(TRAIN_PATH)

st.subheader("Input features")

cols = st.columns(2)
inputs = {}
for idx, feature in enumerate(FEATURE_COLUMNS):
    col = cols[idx % 2]
    inputs[feature] = col.number_input(
        feature,
        value=float(defaults.get(feature, 0.0)),
        step=1.0,
        format="%.0f" if "Year" in feature or "Qual" in feature or "Cond" in feature else "%.2f",
    )

input_df = pd.DataFrame([inputs], columns=FEATURE_COLUMNS)

if st.button("Predict"):
    preds_log = model.predict(input_df)
    prediction = float(np.expm1(preds_log)[0])
    st.success(f"Estimated SalePrice: ${prediction:,.0f}")
