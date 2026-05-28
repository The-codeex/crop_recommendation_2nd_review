import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold
)

from sklearn.preprocessing import (
    LabelEncoder,
    PolynomialFeatures
)

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

import joblib


# -------------------------------
# Load Dataset
# -------------------------------

df = pd.read_csv("data/clean_yield_data.csv")

print("Original dataset size:", df.shape)


# -------------------------------
# Data Cleaning
# -------------------------------

df = df.dropna()
df = df.drop_duplicates()

print("\nMissing values:")
print(df.isnull().sum())

print("Before cleaning:", df.shape)

df = df[df["Area"] > 0]
df = df[df["Production"] > 0]

# Yield validation
df = df[df["Yield"] <= 100]
df = df[df["Yield"] > 0]

print("After cleaning:", df.shape)

print("\nYield stats after cleaning:")
print(df["Yield"].describe())


# -------------------------------
# Log Transform Target
# -------------------------------

df["Yield"] = np.log1p(df["Yield"])

df["log_area"] = np.log1p(df["Area"])
df["log_production"] = np.log1p(df["Production"])


# -------------------------------
# Feature Engineering
# -------------------------------

df["fertilizer_per_hectare"] = (
    df["Fertilizer"] / df["Area"]
)

df["pesticide_per_hectare"] = (
    df["Pesticide"] / df["Area"]
)

df["input_intensity"] = (
    df["fertilizer_per_hectare"] +
    df["pesticide_per_hectare"]
)

df["log_rainfall"] = np.log1p(
    df["Annual_Rainfall"]
)

df["year_from_start"] = (
    df["Crop_Year"] -
    df["Crop_Year"].min()
)


# -------------------------------
# Encode Categorical Variables
# -------------------------------

le_state  = LabelEncoder()
le_crop   = LabelEncoder()
le_season = LabelEncoder()

df["Crop"] = (
    df["Crop"]
    .str.lower()
    .str.strip()
)

df["State"] = (
    df["State"]
    .str.lower()
    .str.strip()
)

df["Season"] = (
    df["Season"]
    .str.lower()
    .str.strip()
)

df["State"] = le_state.fit_transform(df["State"])

df["Crop"] = le_crop.fit_transform(df["Crop"])

df["Season"] = le_season.fit_transform(df["Season"])


# -------------------------------
# Base Features
# -------------------------------

features = [
    "State",
    "Crop",
    "Season",

    "year_from_start",
    

    "log_area",
    "log_production",
    

    "log_rainfall",

    "fertilizer_per_hectare",
    "pesticide_per_hectare",

    "input_intensity"
]

X = df[features]

y = df["Yield"]


# -------------------------------
# Automatic Feature Generation
# -------------------------------

poly = PolynomialFeatures(
    degree=2,
    interaction_only=True,
    include_bias=False
)

X_poly = poly.fit_transform(X)

generated_feature_names = (
    poly.get_feature_names_out(features)
)

print("\nTotal Generated Features:")
print(len(generated_feature_names))


# -------------------------------
# Automatic Feature Selection
# -------------------------------

selector_model = XGBRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

selector_model.fit(X_poly, y)

selector = SelectFromModel(
    selector_model,
    threshold="median",
    prefit=True
)

X_selected = selector.transform(X_poly)

selected_features = np.array(
    generated_feature_names
)[selector.get_support()]

print("\nSelected Features:")
print(selected_features)

# Final selected dataset
X = pd.DataFrame(
    X_selected,
    columns=selected_features
)




# -------------------------------
# Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining samples : {X_train.shape[0]}")
print(f"Testing samples  : {X_test.shape[0]}")


# -------------------------------
# Model Definition
# -------------------------------

model = XGBRegressor(
    n_estimators=900,
    learning_rate=0.025,
    max_depth=7,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.15,
    min_child_weight=4,
    reg_alpha=0.3,
    reg_lambda=1.2,
    random_state=42,
    n_jobs=-1
)
# model =  GradientBoostingRegressor(

#         n_estimators=300,

#         learning_rate=0.05,

#         max_depth=6,

#         random_state=42
#     )


# -------------------------------
# Model Training
# -------------------------------

print("\nTraining XGBoost...")

model.fit(X_train, y_train)

print("Training complete!")


# -------------------------------
# Evaluation — Train vs Test
# -------------------------------

train_preds = model.predict(X_train)

test_preds = model.predict(X_test)


# --------------------------------
# Train Metrics
# --------------------------------

train_mae = mean_absolute_error(
    y_train,
    train_preds
)

train_mse = mean_squared_error(
    y_train,
    train_preds
)

train_rmse = np.sqrt(train_mse)

train_r2 = r2_score(
    y_train,
    train_preds
)


# --------------------------------
# Test Metrics
# --------------------------------

test_mae = mean_absolute_error(
    y_test,
    test_preds
)

test_mse = mean_squared_error(
    y_test,
    test_preds
)

test_rmse = np.sqrt(test_mse)

test_r2 = r2_score(
    y_test,
    test_preds
)


# --------------------------------
# RAE (Relative Absolute Error)
# --------------------------------

rae = (
    np.sum(np.abs(y_test - test_preds))
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
        (y_test - test_preds)
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
print("YIELD MODEL PERFORMANCE")
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

    "Feature": X.columns,

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

# -------------------------------
# Cross Validation
# -------------------------------

print(f"\n{'='*40}")

print("K-Fold Cross Validation (k=5)")

print(f"{'='*40}")

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

cv_scores = cross_val_score(
    model,
    X,
    y,
    cv=kf,
    scoring="r2"
)

print(f"CV R² Scores : {np.round(cv_scores, 4)}")

print(f"Mean CV R²   : {cv_scores.mean():.4f}")

print(f"Std CV R²    : {cv_scores.std():.4f}")


# -------------------------------
# Dataset Info
# -------------------------------

print(f"\n{'='*40}")

print("Dataset Info")

print(f"{'='*40}")

print(
    f"Total unique crops  : "
    f"{df['Crop'].nunique()}"
)

print(
    f"Total unique states : "
    f"{df['State'].nunique()}"
)


# -------------------------------
# Save Model & Objects
# -------------------------------

print(f"\n{'='*40}")

print("Saving Model & Encoders")

print(f"{'='*40}")

joblib.dump(
    model,
    "models/yield_model.pkl"
)

joblib.dump(
    le_state,
    "models/state_encoder.pkl"
)

joblib.dump(
    le_crop,
    "models/crop_encoder.pkl"
)

joblib.dump(
    le_season,
    "models/season_encoder.pkl"
)

joblib.dump(
    poly,
    "models/poly_transformer.pkl"
)

joblib.dump(
    list(selected_features),
    "models/yield_selected_features.pkl"
)

print("✅ Model saved → models/yield_model.pkl")

print("🚀 All done! Model is ready.")

























