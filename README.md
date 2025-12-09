# EF2039_IAP_Term_Project_02

# Global Temperature Prediction (2020–2024)

A simple machine learning pipeline that predicts average temperature using multi-country climate and energy data.
The project trains regression models from scratch (Linear Regression, RandomForest, GradientBoosting), evaluates performance, and visualizes prediction results and feature importance.

# Requirements

Python 3.10+

Dependencies:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Install all with:
pip install -r requirements.txt

# Dataset

This project uses a global climate & energy dataset (2020–2024).

Target variable: avg_temperature

Feature variables: date, country, humidity, co2_emission, energy_consumption, renewable_share, urban_population, industrial_activity_index, energy_price

plus engineered features: year, month

⚙️ Model Pipeline

The script performs:

1. Load & preprocess dataset
2. Feature engineering (year/month)
3. Time-aware train/test split (80/20)
4. Preprocessing pipeline  
    a. StandardScaler for numeric features  
    b. OneHotEncoder for country  
5. Train 3 regression models
6. Evaluate using MAE / RMSE / R²
7. TimeSeriesSplit cross-validation
8. Generate plots (saved in plots/ folder):  
    a. Correlation heatmap  
    b. Distribution of avg temperature  
    c. Actual vs predicted scatter plot  
    d. Time-series comparison  
    e. Residual plot  
    f. Permutation importance

🚀 How to Run

python climate_temp_predictor.py

All plots will be saved automatically in the plots/ directory.

📈 Model Summary

GradientBoostingRegressor achieves the best performance (Test R² ≈ 0.98, MAE ≈ 1.2°C)

The model captures seasonal and geographical temperature patterns well.

Feature interpretation shows:
- month → strongest predictor
- country → second
- other indicators have minor influence

⚠️ Limitations

Dataset covers only 2020–2024

Missing key physical climate variables (radiation, elevation, precipitation)
