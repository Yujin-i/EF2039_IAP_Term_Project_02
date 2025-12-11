"""
Draft v2 - Temperature Regression with Pipelines
- Load & preprocess data
- Feature engineering (year, month)
- ColumnTransformer (scaling + one-hot encoding)
- Time-aware train/test split
- 3 models: LinearRegression, RandomForest, GradientBoosting
- Basic metrics (MAE, RMSE, R^2)
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_SEED = 42

# ----------------------------
# 1. Load data
# ----------------------------
DATA_PATH = "global_climate_energy_2020_2024.csv"
df = pd.read_csv(DATA_PATH)

print("=== Raw Data Info ===")
print("Shape:", df.shape)
print(df.head())

# ----------------------------
# 2. Basic preprocessing
# ----------------------------
df["date"] = pd.to_datetime(df["date"])
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

df = df.dropna().reset_index(drop=True)
print("\nAfter dropna, shape:", df.shape)

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
    "country",
]

# ----------------------------
# 3. Time-aware train/test split
# ----------------------------
df_sorted = df.sort_values(by="date").reset_index(drop=True)

X_sorted = df_sorted[feature_columns]
y_sorted = df_sorted[target_column]

split_index = int(len(df_sorted) * 0.8)

X_train = X_sorted.iloc[:split_index]
X_test = X_sorted.iloc[split_index:]
y_train = y_sorted.iloc[:split_index]
y_test = y_sorted.iloc[split_index:]

print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)

# ----------------------------
# 4. Preprocessing pipeline
# ----------------------------
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
        ("scaler", StandardScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# ----------------------------
# 5. Define models
# ----------------------------
models = {
    "LinearRegression": LinearRegression(),
    "RandomForest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=RANDOM_SEED,
    ),
}

# ----------------------------
# 6. Train & evaluate
# ----------------------------
def rmse(y_true, y_pred):
    """Root Mean Squared Error (버전 호환용: squared 인자 안 씀)"""
    mse = mean_squared_error(y_true, y_pred)
    return mse ** 0.5

for name, base_model in models.items():
    print(f"\n===== Model: {name} (Draft v2) =====")

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", base_model),
        ]
    )

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print("Train MAE :", mean_absolute_error(y_train, y_train_pred))
    print("Train RMSE:", rmse(y_train, y_train_pred))
    print("Train R^2 :", r2_score(y_train, y_train_pred))

    print("Test  MAE :", mean_absolute_error(y_test, y_test_pred))
    print("Test  RMSE:", rmse(y_test, y_test_pred))
    print("Test  R^2 :", r2_score(y_test, y_test_pred))
