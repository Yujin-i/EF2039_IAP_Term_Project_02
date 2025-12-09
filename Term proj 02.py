"""
EF2039 - Term Project 02
Regression Model for Predicting Average Temperature (2020-2024)

Target:
    - avg_temperature

Inputs:
    - humidity
    - co2_emission
    - energy_consumption
    - renewable_share
    - urban_population
    - industrial_activity_index
    - energy_price
    - year, month (from date)
    - country (one-hot encoded in pipeline)

Models:
    - Linear Regression (baseline)
    - Random Forest Regressor
    - Gradient Boosting Regressor

Features:
    - EDA (correlation heatmap, distribution plot)
    - Train/Test split with time-awareness
    - Cross-validation (TimeSeriesSplit)
    - Residual analysis
    - Actual vs Predicted (parity plot)
    - Time series plot
    - Permutation importance visualization

All plots are saved into ./plots folder as PNG files.

Author: <Your Name>
"""

# ==============================
# 1. Imports & Configuration
# ==============================

import os
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")  # Non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import cross_validate, TimeSeriesSplit
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor


RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["font.size"] = 10

# Output plots folder
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def save_current_fig(filename: str):
    """Save current matplotlib figure into plots folder and close it."""
    path = os.path.join(PLOTS_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"[Saved figure] {path}")


# ==============================
# 2. Load & Inspect Data
# ==============================

data_path = "global_climate_energy_2020_2024.csv"
df = pd.read_csv(data_path)

print("=== Raw Data Info ===")
print("Shape:", df.shape)
print(df.head())
print(df.dtypes)


# ==============================
# 3. Basic Preprocessing
# ==============================

# Parse date column
df["date"] = pd.to_datetime(df["date"])

# Extract year and month
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

print("\n=== Missing Values per Column (Before) ===")
print(df.isna().sum())

# Drop rows with any NaNs (simple strategy)
df = df.dropna().reset_index(drop=True)

print("\nShape after dropping rows with NaNs:", df.shape)
print("\n=== Missing Values per Column (After) ===")
print(df.isna().sum())


# ==============================
# 4. Exploratory Data Analysis (EDA)
# ==============================

numeric_for_corr = [
    "avg_temperature",
    "humidity",
    "co2_emission",
    "energy_consumption",
    "renewable_share",
    "urban_population",
    "industrial_activity_index",
    "energy_price",
    "year",
    "month",
]

corr_matrix = df[numeric_for_corr].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
save_current_fig("01_correlation_heatmap.png")

plt.figure(figsize=(7, 4))
sns.histplot(df["avg_temperature"], kde=True, bins=30)
plt.xlabel("avg_temperature")
plt.title("Distribution of Average Temperature")
save_current_fig("02_avg_temperature_distribution.png")


# ==============================
# 5. Define Features and Target
# ==============================

target_column = "avg_temperature"

feature_columns = [
    "humidity",
    "co2_emission",
    "energy_consumption",
    "renewable_share",
    "urban_population",
    "industrial_activity_index",
    "energy_price",
    "year",
    "month",
    "country",  # categorical
]

X = df[feature_columns]
y = df[target_column]


# ==============================
# 6. Time-Aware Train-Test Split
# ==============================

df_sorted = df.sort_values(by="date").reset_index(drop=True)

X_sorted = df_sorted[feature_columns]
y_sorted = df_sorted[target_column]
dates_sorted = df_sorted["date"]

split_index = int(len(df_sorted) * 0.8)

X_train = X_sorted.iloc[:split_index]
X_test = X_sorted.iloc[split_index:]
y_train = y_sorted.iloc[:split_index]
y_test = y_sorted.iloc[split_index:]
dates_train = dates_sorted.iloc[:split_index]
dates_test = dates_sorted.iloc[split_index:]

print("\n=== Train/Test Split ===")
print("Train size:", X_train.shape, "Test size:", X_test.shape)


# ==============================
# 7. Preprocessing Pipeline
# ==============================

numeric_features = [
    "humidity",
    "co2_emission",
    "energy_consumption",
    "renewable_share",
    "urban_population",
    "industrial_activity_index",
    "energy_price",
    "year",
    "month",
]

categorical_features = ["country"]

numeric_transformer = Pipeline(
    steps=[
        ("scaler", StandardScaler())
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# ==============================
# 8. Define Models
# ==============================

models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=300,
        max_depth=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_SEED,
    ),
}


# ==============================
# 9. Evaluation Helper Function
# ==============================

def evaluate_regression_model(model_name, model, X_tr, X_te, y_tr, y_te):
    """
    Fit model, evaluate on train and test sets,
    and print MAE, RMSE, R^2.
    """
    print(f"\n===== Model: {model_name} =====")
    model.fit(X_tr, y_tr)

    y_tr_pred = model.predict(X_tr)
    y_te_pred = model.predict(X_te)

    # Train metrics
    train_mae = mean_absolute_error(y_tr, y_tr_pred)
    train_rmse = np.sqrt(mean_squared_error(y_tr, y_tr_pred))
    train_r2 = r2_score(y_tr, y_tr_pred)

    # Test metrics
    test_mae = mean_absolute_error(y_te, y_te_pred)
    test_rmse = np.sqrt(mean_squared_error(y_te, y_te_pred))
    test_r2 = r2_score(y_te, y_te_pred)

    print("Train  MAE : {:.4f}".format(train_mae))
    print("Train  RMSE: {:.4f}".format(train_rmse))
    print("Train  R^2 : {:.4f}".format(train_r2))
    print("Test   MAE : {:.4f}".format(test_mae))
    print("Test   RMSE: {:.4f}".format(test_rmse))
    print("Test   R^2 : {:.4f}".format(test_r2))

    return {
        "model_name": model_name,
        "model": model,
        "y_test_pred": y_te_pred,
    }


# ==============================
# 10. Train & Evaluate All Models
# ==============================

trained_models = {}

for model_name, base_model in models.items():
    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", base_model),
        ]
    )

    result = evaluate_regression_model(
        model_name=model_name,
        model=pipe,
        X_tr=X_train,
        X_te=X_test,
        y_tr=y_train,
        y_te=y_test,
    )

    trained_models[model_name] = result


# ==============================
# 11. Time Series Cross-Validation (RandomForest)
# ==============================

print("\n===== Time Series Cross-Validation (RandomForest) =====")

tscv = TimeSeriesSplit(n_splits=5)

rf_pipe = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", models["RandomForest"]),
    ]
)

cv_results = cross_validate(
    rf_pipe,
    X_sorted,
    y_sorted,
    cv=tscv,
    scoring=("neg_mean_absolute_error", "neg_mean_squared_error", "r2"),
    n_jobs=1,  # to avoid Unicode/multiprocessing issues on Windows
)

cv_mae = -cv_results["test_neg_mean_absolute_error"]
cv_mse = -cv_results["test_neg_mean_squared_error"]
cv_rmse = np.sqrt(cv_mse)
cv_r2 = cv_results["test_r2"]

print("CV MAE  (mean ± std): {:.4f} ± {:.4f}".format(cv_mae.mean(), cv_mae.std()))
print("CV RMSE (mean ± std): {:.4f} ± {:.4f}".format(cv_rmse.mean(), cv_rmse.std()))
print("CV R^2  (mean ± std): {:.4f} ± {:.4f}".format(cv_r2.mean(), cv_r2.std()))


# ==============================
# 12. Choose Best Model (RandomForest) for Visualization
# ==============================

best_model_name = "RandomForest"
best_model = trained_models[best_model_name]["model"]
y_test_pred = trained_models[best_model_name]["y_test_pred"]

residuals = y_test - y_test_pred


# ==============================
# 13. Visual 1: Actual vs Predicted (Parity Plot)
# ==============================

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_test_pred, alpha=0.6)
min_val = min(y_test.min(), y_test_pred.min())
max_val = max(y_test.max(), y_test_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("Actual avg_temperature")
plt.ylabel("Predicted avg_temperature")
plt.title("Actual vs Predicted")
save_current_fig("03_actual_vs_predicted_parity.png")


# ==============================
# 14. Visual 2: Time Series - Actual vs Predicted on Test Set
# ==============================

plt.figure(figsize=(10, 5))
plt.plot(dates_test, y_test.values, label="Actual", linewidth=1.5)
plt.plot(dates_test, y_test_pred, label="Predicted", linewidth=1.0)
plt.xlabel("Date")
plt.ylabel("avg_temperature")
plt.title("Actual vs Predicted Over Time")
plt.legend()
save_current_fig("04_timeseries_actual_vs_predicted.png")


# ==============================
# 15. Visual 3: Residual Plot
# ==============================

plt.figure(figsize=(8, 5))
plt.scatter(y_test_pred, residuals, alpha=0.6)
plt.axhline(0, linestyle="--")
plt.xlabel("Predicted avg_temperature")
plt.ylabel("Residuals")
plt.title("Residual Plot")
save_current_fig("05_residual_plot.png")


# ==============================
# 16. Visual 4: Permutation Importance
# ==============================

print("\n=== Computing Permutation Importance (RandomForest) ===")

perm_result = permutation_importance(
    best_model,   # full pipeline
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    n_jobs=1,
)

# 여기서는 변환된 feature가 아니라,
# 원래 입력 컬럼 기준이므로 이름도 그대로 사용
all_feature_names = numeric_features + categorical_features  # length == importances length

perm_importance_df = pd.DataFrame({
    "feature": all_feature_names,
    "importance_mean": perm_result.importances_mean,
    "importance_std": perm_result.importances_std,
}).sort_values(by="importance_mean", ascending=False)

print("\n=== Top Permutation Importances ===")
print(perm_importance_df)

plt.figure(figsize=(10, 6))
top = perm_importance_df[::-1]  # reverse for nicer barh (small -> top)
plt.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
plt.xlabel("Decrease in Model Performance")
plt.title("Permutation Importance")
plt.tight_layout()
save_current_fig("06_permutation_importance.png")
