

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score

from xgboost import XGBRegressor
import joblib


# ------------------------------
# Load dataset
# ------------------------------

df = pd.read_csv("data/crop_price_data_updated.csv")
df.columns = df.columns.str.strip()

print("Columns:", df.columns)
print("Original dataset shape:", df.shape)


# ------------------------------
# Data Cleaning
# ------------------------------

df = df.dropna()
df = df.drop_duplicates()




# Rename columns properly
df = df.rename(columns={
    "STATE": "State",
    "CROP": "Crop",
    "CROP_PRICE": "Price"
})
df["Crop"] = df["Crop"].str.lower().str.strip()
print("Dataset after cleaning:", df.shape)


# ------------------------------
# Convert numeric columns
# ------------------------------

num_cols = [
    "N_SOIL",
    "P_SOIL",
    "K_SOIL",
    "TEMPERATURE",
    "HUMIDITY",
    "ph",
    "RAINFALL",
    "Supply Volume (tons)",
    "Demand Volume (tons)",
    "Price"
]

for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()


# ------------------------------
# Log transform target
# ------------------------------

df["Price"] = np.log1p(df["Price"])
df["RAINFALL"] = np.log1p(df["RAINFALL"])
df["TEMPERATURE"] = np.log1p(df["TEMPERATURE"])
df["HUMIDITY"] = np.log1p(df["HUMIDITY"])
df["Supply Volume (tons)"] = np.log1p(
    df["Supply Volume (tons)"]
)

df["Demand Volume (tons)"] = np.log1p(
    df["Demand Volume (tons)"]
)


# ------------------------------
# Encode categorical variables
# ------------------------------
df["State"] = df["State"].str.lower().str.strip()
le_state = LabelEncoder()
le_crop = LabelEncoder()
le_soil = LabelEncoder()

df["State"] = le_state.fit_transform(df["State"])
df["Crop"] = le_crop.fit_transform(df["Crop"])
df["SOIL_TYPE"] = le_soil.fit_transform(df["SOIL_TYPE"])
df["N_SOIL"] = df["N_SOIL"] / 100
df["P_SOIL"] = df["P_SOIL"] / 100
df["K_SOIL"] = df["K_SOIL"] / 100

df["NPK_SUM"] = (
    df["N_SOIL"] +
    df["P_SOIL"] +
    df["K_SOIL"]
)

df["TEMP_HUMIDITY"] = (
    df["TEMPERATURE"] *
    df["HUMIDITY"]
)

df["RAIN_TEMP"] = (
    df["RAINFALL"] *
    df["TEMPERATURE"]
)
df["SUPPLY_DEMAND_RATIO"] = (
    df["Supply Volume (tons)"]
    /
    (
        df["Demand Volume (tons)"] + 1
    )
)
df["Month_sin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["Month_cos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)



# ------------------------------
# Feature Selection
# ------------------------------

features = [
    "Year",
    "Month_sin",
    "Month_cos",
    "State",
    "SOIL_TYPE",
    "Crop",
    "N_SOIL",
    "P_SOIL",
    "K_SOIL",
    "TEMPERATURE",
    "HUMIDITY",
    "ph",
    "RAINFALL",

    "Supply Volume (tons)",
    "Demand Volume (tons)",
    "SUPPLY_DEMAND_RATIO",

    "NPK_SUM",
    "TEMP_HUMIDITY",
    "RAIN_TEMP"
]

X = df[features]
y = df["Price"]


# ------------------------------
# Train Test Split
# ------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)


# ------------------------------
# Model Definition (balanced)
# ------------------------------

model = XGBRegressor(

    n_estimators=500,

    learning_rate=0.03,

    max_depth=5,

    subsample=0.8,

    colsample_bytree=0.8,

    min_child_weight=5,

    gamma=0.3,

    reg_alpha=0.8,

    reg_lambda=2.0,

    random_state=42,

    n_jobs=-1
)
# ------------------------------
# Training
# ------------------------------

print("\nTraining Price Model...")
model.fit(X_train, y_train)




# ------------------------------
# Evaluation
# ------------------------------

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# Predictions
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)


# --------------------------------
# Train Metrics
# --------------------------------

train_mae = mean_absolute_error(
    y_train,
    train_pred
)

train_mse = mean_squared_error(
    y_train,
    train_pred
)

train_rmse = np.sqrt(train_mse)

train_r2 = r2_score(
    y_train,
    train_pred
)


# --------------------------------
# Test Metrics
# --------------------------------

test_mae = mean_absolute_error(
    y_test,
    test_pred
)

test_mse = mean_squared_error(
    y_test,
    test_pred
)

test_rmse = np.sqrt(test_mse)

test_r2 = r2_score(
    y_test,
    test_pred
)


# --------------------------------
# RAE (Relative Absolute Error)
# --------------------------------

rae = (
    np.sum(np.abs(y_test - test_pred))
    /
    np.sum(
        np.abs(y_test - np.mean(y_test))
    )
)


# --------------------------------
# MAPE
# --------------------------------

mape = np.mean(
    np.abs(
        (y_test - test_pred)
        / y_test
    )
) * 100


# --------------------------------
# R2 Gap
# --------------------------------

r2_gap = train_r2 - test_r2


# --------------------------------
# Print Metrics
# --------------------------------

print("\n========================================")
print("PRICE MODEL PERFORMANCE")
print("========================================")

print(
    f"{'Metric':<20}"
    f"{'Train':>12}"
    f"{'Test':>12}"
)

print("----------------------------------------")

print(
    f"{'MAE':<20}"
    f"{train_mae:>12.4f}"
    f"{test_mae:>12.4f}"
)

print(
    f"{'MSE':<20}"
    f"{train_mse:>12.4f}"
    f"{test_mse:>12.4f}"
)

print(
    f"{'RMSE':<20}"
    f"{train_rmse:>12.4f}"
    f"{test_rmse:>12.4f}"
)

print(
    f"{'R2 Score':<20}"
    f"{train_r2:>12.4f}"
    f"{test_r2:>12.4f}"
)

print("========================================")


# --------------------------------
# Additional Metrics
# --------------------------------

print("\nAdditional Metrics")
print("----------------------------------------")

print(f"RAE        : {rae:.4f}")

print(f"MAPE (%)   : {mape:.2f}")

print(f"R2 Gap     : {r2_gap:.4f}")


# --------------------------------
# Overfitting Check
# --------------------------------

print("\nOverfitting Analysis")
print("----------------------------------------")

if r2_gap < 0.05:
    print("✅ Excellent Generalization")

elif r2_gap < 0.10:
    print("✅ Good Generalization")

elif r2_gap < 0.15:
    print("⚠️ Moderate Overfitting")

else:
    print("❌ High Overfitting")


# ------------------------------
# Feature Importance
# ------------------------------

importance_df = pd.DataFrame({

    "Feature": features,

    "Importance": model.feature_importances_

})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print("\n========================================")
print("TOP FEATURE IMPORTANCE")
print("========================================")

print(importance_df.head(10))


# ------------------------------
# Cross Validation
# ------------------------------

scores = cross_val_score(
    model,
    X,
    y,
    cv=5,
    scoring="r2"
)

print("\n========================================")
print("Cross Validation")
print("========================================")

print(
    "CV R2 Scores :",
    np.round(scores, 4)
)

print(
    "Mean CV R2   :",
    round(scores.mean(), 4)
)

print(
    "Std CV R2    :",
    round(scores.std(), 4)
)





# ------------------------------
# Save Model + Encoders
# ------------------------------

joblib.dump(model, "models/price_model.pkl")
joblib.dump(le_state, "models/price_state_encoder.pkl")
joblib.dump(le_crop, "models/price_crop_encoder.pkl")
joblib.dump(le_soil, "models/price_soil_encoder.pkl")

print("\n✅ Price model saved successfully!")



