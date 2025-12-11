"""
Draft v1 - Simple Temperature Regression
- Load CSV
- Basic preprocessing (date -> year/month, drop NaNs)
- One-hot encode country
- Time-aware train/test split (80/20)
- Linear Regression + simple metrics
"""

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
# Parse date column to datetime
df["date"] = pd.to_datetime(df["date"])

# Feature engineering: year, month
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# Drop rows with missing values
df = df.dropna().reset_index(drop=True)
print("\nAfter dropna, shape:", df.shape)

# ----------------------------
# 3. Define features / target
# ----------------------------
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

# 먼저 날짜 기준으로 정렬 (time-aware split을 위해)
df_sorted = df.sort_values(by="date").reset_index(drop=True)

# 모델용 데이터프레임 만들기
df_model = df_sorted[feature_columns + [target_column]].copy()

# One-hot encode country (simple version)
df_model = pd.get_dummies(df_model, columns=["country"], drop_first=True)

# 최종 X, y
X = df_model.drop(columns=[target_column])
y = df_model[target_column]

# ----------------------------
# 4. Time-aware Train/Test Split (80/20)
# ----------------------------
split_idx = int(len(df_model) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]
y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)

# ----------------------------
# 5. Train simple Linear Regression
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# ----------------------------
# 6. Evaluate
# ----------------------------
def rmse(y_true, y_pred):
    """Root Mean Squared Error (버전 호환용: squared 인자 안 씀)"""
    mse = mean_squared_error(y_true, y_pred)
    return mse ** 0.5

print("\n===== Model: LinearRegression (Draft v1) =====")
print("Train MAE :", mean_absolute_error(y_train, y_train_pred))
print("Train RMSE:", rmse(y_train, y_train_pred))
print("Train R^2 :", r2_score(y_train, y_train_pred))

print("Test  MAE :", mean_absolute_error(y_test, y_test_pred))
print("Test  RMSE:", rmse(y_test, y_test_pred))
print("Test  R^2 :", r2_score(y_test, y_test_pred))
