# House Prices Prediction (Kaggle)

Gradient-boosted models for the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques) competition.  
This repo bundles the raw Ames housing dataset, exploratory notebooks, feature-engineering utilities, and the scripts/notebooks that generate the final Kaggle submission.

## Repository Layout

- `data/` – Kaggle-provided `train.csv`, `test.csv`, documentation, plus intermediate cleaned CSVs.
- `project.ipynb` / `data_exploration*.ipynb` – end-to-end notebooks used for EDA, feature engineering, and exporting predictions.
- `project.py` – sequential forward feature selection with a `GradientBoostingRegressor`.
- `kaggle_training.log` – rolling log of feature-selection experiments and cross-validated scores.
- `results/submission.csv` – latest prediction file uploaded to Kaggle (sample copies in the repo root as well).

## Environment Setup

Create the environment once, then reuse it for notebooks and scripts.

### uv (recommended)
```bash
uv sync
```

### Conda
```bash
conda env create --name data-prog-env --file=environment.yaml
conda activate data-prog-env
```

## Workflow

1. **Explore & understand the data**  
   - Start in `data_exploration.ipynb` / `data_exploration_v2.ipynb` to inspect the 79 features, missingness, and correlations (a written summary also lives in `data/data_exploration.md`).
   - The notebooks rely on `pandas`, `seaborn`, and `AutoViz` for quick profile-style reports.

2. **Feature engineering & cleaning**  
   - Reusable functions live in `project.py`: `prepare_clean_data`, `clean_data`, `remove_outliers`, and `categorize_columns`.
   - The current pipeline fills high-missing categorical columns with `"None"`, imputes numeric columns with medians, adds helper flags such as `HasGarage` / `GarageYrBltMissing`, caps outliers via IQR clipping, and drops `Id` plus low-signal columns.

3. **Model search / training**  
   - Run the sequential feature-selection script:
     ```bash
     uv run python project.py
     ```
   - The script iterates through a correlation-ordered candidate list, evaluates each subset with a 5-fold CV `GradientBoostingRegressor` (log-transforming `SalePrice` via `log1p`), and keeps only improvements (`~0.0005` RMSE tolerance).  
   - Detailed progress—including the best CV log-RMSE (~0.138) and tuned hyperparameters (`n_estimators=800`, `learning_rate=0.01`, `max_depth=3`, `min_samples_leaf=5`)—is persisted to `kaggle_training.log`.

4. **Build the submission**  
   - `project.ipynb` loads the selected features, retrains on the full training set, and predicts on `data/test.csv`.  
   - Export predictions as:
     ```python
     submission.to_csv("results/submission.csv", index=False)
     ```
   - Upload `results/submission.csv` to Kaggle to score on the public leaderboard.

## Modeling Details

- **Algorithm** – `GradientBoostingRegressor` (sklearn) tuned via grid-search and evaluated with 5-fold cross-validated log-RMSE, matching Kaggle’s metric.
- **Target handling** – `SalePrice` values are transformed with `np.log1p` during training; inverse-transform before writing submissions.
- **Feature set** – Key drivers include `OverallQual`, `GrLivArea`, `GarageCars`, `TotalBsmtSF`, `YearBuilt`, etc. (`project.py` logs the exact retained subset).
- **Artifacts** – Cleaned datasets (`data/cleaned_train.csv`, `data/cleaned_test.csv`) and `submission.csv` snapshots help reproduce results without rerunning the full pipeline.

## Next Steps

- Extend `project.py` to automatically train the final model and emit submissions (mirroring the notebook flow).
- Experiment with stacked/ensemble models (`LightGBM`, `CatBoost`) using the dependencies defined in `pyproject.toml`.

## Streamlit App

Run a quick local predictor UI once the model is saved:

```bash
streamlit run app.py
```
