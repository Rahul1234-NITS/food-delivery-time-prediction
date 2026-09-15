# Food Delivery Time Prediction

Predicting food delivery time (in minutes) using order, weather, traffic, and location data — an end-to-end regression project covering data cleaning, feature engineering, and model comparison.

## Problem Statement

Given details of a food delivery order (restaurant/customer location, weather, traffic conditions, delivery person info, vehicle type, etc.), predict how long the delivery will take.

## Dataset

- **45,593 rows**, 20 raw columns
- Contains delivery person details (age, ratings), restaurant & delivery location coordinates, weather conditions, road traffic density, vehicle type, order type, and festival/city info

## Tech Stack

- **Data handling & EDA:** Pandas, NumPy, Matplotlib, Seaborn
- **Modeling:** Scikit-learn, XGBoost
- **Model persistence:** Joblib

## Workflow

1. **Data Cleaning** — fixed malformed text fields (e.g. `"conditions Sunny"` → `Sunny`, `"(min) 24"` → `24`), handled literal `'NaN'` strings, clipped invalid rating values, imputed missing values (median for numeric, mode for categorical)
2. **EDA** — analyzed delivery time distribution and its relationship with traffic, weather, festivals, city, rider age/rating, and order hour
3. **Feature Engineering**
   - Computed `distance_km` between restaurant and delivery location using the **Haversine formula**
   - Extracted `order_hour` from order timestamps
   - Removed unrealistic distance outliers (>100 km, likely corrupt coordinates)
4. **Encoding** — ordinal encoding for `Road_traffic_density` (Low→Jam), one-hot encoding for nominal categorical features
5. **Modeling** — trained and compared 4 regression models on an 80/20 train-test split, with feature scaling via `StandardScaler` (fit on train only)
6. **Hyperparameter Tuning** — tuned XGBoost using `RandomizedSearchCV` (40 iterations, 3-fold CV)
7. **Feature Validation** — tested whether bucketing order hour into Lunch/Dinner/Normal periods improved Adjusted R²; kept the simpler feature set since the gain was negligible
8. **Model Persistence** — saved the final model and scaler with `joblib` for reuse in inference

## Results

| Model | Test R² | Test RMSE (min) | Train R² | Verdict |
|---|---:|---:|---:|---|
| Linear Regression | 0.57 | 6.21 | 0.58 | Underfits — relationships aren't linear |
| Decision Tree | 0.67 | 5.46 | 1.00 | Memorizes train set (overfits) |
| Random Forest | 0.82 | 4.03 | 0.97 | Big gain — averaging tames overfitting |
| **XGBoost (tuned)** | **0.83** | **3.91** | 0.85 | Best score with healthiest train/test gap |

**Best model:** Tuned XGBoost — predicts delivery time within ~3.9 minutes on average, explaining 83% of the variance in delivery time.

## Key Insight

Adding an engineered `order_period` (Lunch/Dinner/Normal) feature on top of the raw `order_hour` barely moved Adjusted R² (~0.829 vs baseline) — a useful reminder that not every engineered feature earns its place in the final model.

## Project Structure

```
├── Food_Delivery_Time_Prediction.ipynb   # Full analysis and modeling notebook
├── Food delivery.csv                     # Raw dataset
└── README.md
```

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
jupyter notebook Food_Delivery_Time_Prediction.ipynb
```
