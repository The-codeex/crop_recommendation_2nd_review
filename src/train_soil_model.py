''' 

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from xgboost import XGBClassifier


# -------------------------------
# Load datasets
# -------------------------------

soil_df = pd.read_csv("data/soil_crop_dataset.csv")
yield_df = pd.read_csv("data/final_dataset.csv")

# Keep only crops available in yield dataset
valid_crops = yield_df["Crop"].unique()

soil_df = soil_df[soil_df["CROP"].isin(valid_crops)]
# Remove crops with very few samples
crop_counts = soil_df["CROP"].value_counts()

valid_crops = crop_counts[crop_counts >= 2].index

soil_df = soil_df[soil_df["CROP"].isin(valid_crops)]

print("Total crops used in soil model:", soil_df["CROP"].nunique())

# -------------------------------
# Encoding
# -------------------------------

le_state = LabelEncoder()
le_soil = LabelEncoder()
le_crop = LabelEncoder()

soil_df["STATE"] = le_state.fit_transform(soil_df["STATE"])
soil_df["SOIL_TYPE"] = le_soil.fit_transform(soil_df["SOIL_TYPE"])
soil_df["CROP"] = le_crop.fit_transform(soil_df["CROP"])

# -------------------------------
# Features
# -------------------------------

X = soil_df[[
    "STATE",
    "SOIL_TYPE",
    "N_SOIL",
    "P_SOIL",
    "K_SOIL",
    "TEMPERATURE",
    "HUMIDITY",
    "ph",
    "RAINFALL"
]]

y = soil_df["CROP"]

# -------------------------------
# Feature Scaling
# -------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

# -------------------------------
# Train Test Split
# -------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -------------------------------
# Model
# -------------------------------

model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42
)

# -------------------------------
# Train
# -------------------------------

model.fit(X_train, y_train)

# -------------------------------
# Prediction
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# Evaluation
# -------------------------------

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)

recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("\nSoil Model Evaluation")
print("---------------------------")

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# -------------------------------
# Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# -------------------------------
# Save model
# -------------------------------

joblib.dump(model, "models/soil_model.pkl")

joblib.dump(scaler, "models/soil_scaler.pkl")

joblib.dump(le_state, "models/soil_state_encoder.pkl")
joblib.dump(le_soil, "models/soil_encoder.pkl")
joblib.dump(le_crop, "models/soil_crop_encoder.pkl")

print("\nSoil recommendation model saved")

# new soil code start------------------------------------------------------------------------


Soil-based Crop Recommendation — Two-Tier Model
=================================================

Root cause of previous low accuracy:
  Many vegetable crops (Brinjal, Cabbage, Carrot, Cauliflower, Cucumber,
  Onion, Tomato, Bottle Gourd, Bitter Gourd) are agro-climatically identical
  in this dataset — they share near-identical N/P/K, temperature, pH, and
  rainfall values. No model can distinguish them reliably from soil data alone.

  Forcing a single hard prediction across all 19 crops yields ~28% accuracy.
  Restricting to the 10 crops with genuinely distinct profiles yields ~63%.

Solution — Two-Tier Architecture:
  Tier 1 (PRIMARY MODEL):  Trained on all 19 crops.
                            Used for top-3 probability recommendations.
                            Expected Top-3 accuracy: ~60-65%

  Tier 2 (CONFIDENCE CHECK): After predicting, if the top prediction is a
                            "confusable vegetable" AND model confidence is low
                            (< CONFIDENCE_THRESHOLD), return the full
                            confusable vegetable cluster as equally valid
                            recommendations rather than a misleading single answer.

  This is honest and useful: farmers see either a confident single recommendation
  or the full set of viable alternatives — never a confidently-wrong single label.

# --------------------------------------2nd old code-----------------------------------------------------

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, classification_report)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from xgboost import XGBClassifier


# -----------------------------------------------
# Config
# -----------------------------------------------

MIN_SAMPLES = 20
CONFIDENCE_THRESHOLD = 0.40   # below this → return full vegetable cluster

# Crops whose soil/climate profiles strongly overlap — cannot be distinguished
# by soil features alone. When model confidence is low and prediction falls
# in this group, we return all of them as valid recommendations.
CONFUSABLE_VEGS = {
    "Bitter Gourd", "Bottle Gourd", "Brinjal", "Cabbage",
    "Carrot", "Cauliflower", "Cucumber", "Onion", "Tomato"
}


# -----------------------------------------------
# 1. Load datasets
# -----------------------------------------------

soil_df  = pd.read_csv("data/soil_crop_dataset.csv")
yield_df = pd.read_csv("data/final_dataset.csv")

print("Original shape:", soil_df.shape)
print("Original crops:", soil_df["CROP"].nunique())


# -----------------------------------------------
# 2. Filter: intersection with yield dataset
#    AND minimum MIN_SAMPLES samples per crop
# -----------------------------------------------

crops_in_yield = set(yield_df["Crop"].unique())
soil_df = soil_df[soil_df["CROP"].isin(crops_in_yield)].reset_index(drop=True)

crop_counts = soil_df["CROP"].value_counts()
valid_crops = crop_counts[crop_counts >= MIN_SAMPLES].index
soil_df     = soil_df[soil_df["CROP"].isin(valid_crops)].reset_index(drop=True)

print(f"\nAfter intersection + >= {MIN_SAMPLES} samples filter:")
print(f"  Crops retained : {soil_df['CROP'].nunique()}")
print(f"  Total rows     : {len(soil_df)}")
print(f"  Crops          : {sorted(soil_df['CROP'].unique())}")

dropped = crops_in_yield - set(soil_df["CROP"].unique())
if dropped:
    print(f"  Dropped ({len(dropped)} crops with < {MIN_SAMPLES} samples)")


# -----------------------------------------------
# 3. Encoding
# -----------------------------------------------

le_state = LabelEncoder()
le_soil  = LabelEncoder()
le_crop  = LabelEncoder()

soil_df["STATE"]     = le_state.fit_transform(soil_df["STATE"])
soil_df["SOIL_TYPE"] = le_soil.fit_transform(soil_df["SOIL_TYPE"])
soil_df["CROP"]      = le_crop.fit_transform(soil_df["CROP"])


# -----------------------------------------------
# 4. Feature Engineering
# -----------------------------------------------

soil_df["N_P_ratio"]  = soil_df["N_SOIL"] / (soil_df["P_SOIL"] + 1)
soil_df["N_K_ratio"]  = soil_df["N_SOIL"] / (soil_df["K_SOIL"] + 1)
soil_df["P_K_ratio"]  = soil_df["P_SOIL"] / (soil_df["K_SOIL"] + 1)
soil_df["NPK_total"]  = soil_df["N_SOIL"] + soil_df["P_SOIL"] + soil_df["K_SOIL"]
soil_df["temp_humid"] = soil_df["TEMPERATURE"] * soil_df["HUMIDITY"] / 100
soil_df["rain_humid"] = soil_df["RAINFALL"]    * soil_df["HUMIDITY"] / 100


# -----------------------------------------------
# 5. Features
# -----------------------------------------------

FEATURES = [
    "STATE", "SOIL_TYPE",
    "N_SOIL", "P_SOIL", "K_SOIL",
    "TEMPERATURE", "HUMIDITY", "ph", "RAINFALL",
    "CROP_PRICE",
    "N_P_ratio", "N_K_ratio", "P_K_ratio",
    "NPK_total", "temp_humid", "rain_humid",
]

X = soil_df[FEATURES]
y = soil_df["CROP"]
num_classes = y.nunique()

print(f"\nClasses: {num_classes}  |  Features: {len(FEATURES)}")


# -----------------------------------------------
# 6. Feature Scaling
# -----------------------------------------------

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)


# -----------------------------------------------
# 7. Train / Test Split
# -----------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")


# -----------------------------------------------
# 8. Models  (class_weight="balanced", no SMOTE)
# -----------------------------------------------

rf_model = RandomForestClassifier(
    n_estimators=700,
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

et_model = ExtraTreesClassifier(
    n_estimators=700,
    max_depth=None,
    min_samples_leaf=1,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)

xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="multi:softprob",
    eval_metric="mlogloss",
    random_state=42,
    n_jobs=-1,
)

ensemble = VotingClassifier(
    estimators=[("rf", rf_model), ("et", et_model), ("xgb", xgb_model)],
    voting="soft",
    n_jobs=-1,
)


# -----------------------------------------------
# 9. Cross-Validation
# -----------------------------------------------

print("\nRunning 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, mdl in [("RandomForest", rf_model), ("ExtraTrees", et_model),
                  ("XGBoost", xgb_model), ("Ensemble", ensemble)]:
    scores = cross_val_score(mdl, X_scaled, y, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"  {name:<14}  CV: {scores.mean():.4f} ± {scores.std():.4f}")


# -----------------------------------------------
# 10. Final Training
# -----------------------------------------------

print("\nTraining final ensemble...")
ensemble.fit(X_train, y_train)


# -----------------------------------------------
# 11. Evaluation — Top-1, Top-3, Top-5
# -----------------------------------------------

y_pred  = ensemble.predict(X_test)
y_proba = ensemble.predict_proba(X_test)

top1 = accuracy_score(y_test, y_pred)
top3 = np.mean([y_test.iloc[i] in np.argsort(y_proba[i])[-3:] for i in range(len(y_test))])
top5 = np.mean([y_test.iloc[i] in np.argsort(y_proba[i])[-5:] for i in range(len(y_test))])

print("\nSoil Model Evaluation")
print("=" * 50)
print(f"Top-1 Accuracy : {top1:.4f}")
print(f"Top-3 Accuracy : {top3:.4f}  ← primary metric")
print(f"Top-5 Accuracy : {top5:.4f}")
print(f"Precision (W)  : {precision_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"Recall    (W)  : {recall_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")
print(f"F1 Score  (W)  : {f1_score(y_test, y_pred, average='weighted', zero_division=0):.4f}")

print("\nPer-class report:")
print(classification_report(y_test, y_pred, target_names=le_crop.classes_, zero_division=0))

# Show which crops model handles well vs poorly
print("\nCrop-level Top-3 accuracy:")
for i, crop_name in enumerate(le_crop.classes_):
    mask = (y_test == i)
    if mask.sum() == 0:
        continue
    crop_proba = y_proba[mask]
    crop_true  = y_test[mask]
    t3 = np.mean([crop_true.iloc[j] in np.argsort(crop_proba[j])[-3:] for j in range(len(crop_true))])
    bar = "█" * int(t3 * 20)
    print(f"  {crop_name:<15}  Top-3: {t3:.2f}  {bar}  (n={mask.sum()})")


# -----------------------------------------------
# 12. Feature Importance
# -----------------------------------------------

rf_fitted   = ensemble.estimators_[0]
importances = rf_fitted.feature_importances_
print("\nFeature Importances (Random Forest):")
for feat, imp in sorted(zip(FEATURES, importances), key=lambda x: -x[1]):
    bar = "█" * int(imp * 60)
    print(f"  {feat:<16} {imp:.4f}  {bar}")


# -----------------------------------------------
# 13. Save artefacts
# -----------------------------------------------

joblib.dump(ensemble,         "models/soil_model.pkl")
joblib.dump(scaler,           "models/soil_scaler.pkl")
joblib.dump(le_state,         "models/soil_state_encoder.pkl")
joblib.dump(le_soil,          "models/soil_encoder.pkl")
joblib.dump(le_crop,          "models/soil_crop_encoder.pkl")
joblib.dump(FEATURES,         "models/soil_features.pkl")
joblib.dump(CONFUSABLE_VEGS,  "models/soil_confusable_vegs.pkl")

print("\nAll artefacts saved.")
print(f"Classes ({num_classes}): {list(le_crop.classes_)}")


# -----------------------------------------------
# 14. Two-Tier Prediction Function
#     Use this in your prediction pipeline
# -----------------------------------------------

def predict_crops(state_enc, soil_enc, n, p, k, temp, humidity, ph,
                  rainfall, crop_price, top_n=3):
    """
    Two-tier crop recommendation.

    Returns a list of dicts: [{'crop': name, 'confidence': float, 'tier': 1|2}]
      Tier 1 = model is confident (confident distinct crop)
      Tier 2 = confusable vegetable cluster returned as equally valid options

    Usage:
        results = predict_crops(state_enc=5, soil_enc=2, n=80, p=40, k=50,
                                temp=25.5, humidity=82, ph=6.4,
                                rainfall=95, crop_price=1500)
        for r in results:
            print(r['crop'], r['confidence'], r['tier'])
    """
    n_p   = n / (p + 1)
    n_k   = n / (k + 1)
    p_k   = p / (k + 1)
    npkt  = n + p + k
    th    = temp * humidity / 100
    rh    = rainfall * humidity / 100

    row = np.array([[state_enc, soil_enc, n, p, k, temp, humidity, ph,
                     rainfall, crop_price, n_p, n_k, p_k, npkt, th, rh]])
    row_scaled = scaler.transform(row)

    proba     = ensemble.predict_proba(row_scaled)[0]
    top_idxs  = np.argsort(proba)[::-1][:top_n]
    top_crop  = le_crop.classes_[top_idxs[0]]
    top_conf  = proba[top_idxs[0]]

    # Tier 2: low confidence AND predicted crop is in confusable cluster
    if top_conf < CONFIDENCE_THRESHOLD and top_crop in CONFUSABLE_VEGS:
        # Return all confusable vegs that exist in our label set
        result = []
        for veg in sorted(CONFUSABLE_VEGS):
            if veg in le_crop.classes_:
                idx  = le_crop.transform([veg])[0]
                result.append({
                    "crop": veg,
                    "confidence": round(float(proba[idx]), 4),
                    "tier": 2,
                    "note": "Similar growing conditions — all are viable"
                })
        return sorted(result, key=lambda x: -x["confidence"])

    # Tier 1: return top-N normally
    return [
        {
            "crop":       le_crop.classes_[idx],
            "confidence": round(float(proba[idx]), 4),
            "tier":       1
        }
        for idx in top_idxs
    ]


# Demo
print("\n--- Two-Tier Prediction Demo ---")
sample_row = X_test[:1]
raw_proba  = ensemble.predict_proba(sample_row)[0]
top3_idxs  = np.argsort(raw_proba)[::-1][:3]
top_crop   = le_crop.classes_[top3_idxs[0]]
top_conf   = raw_proba[top3_idxs[0]]

print(f"Raw top prediction: {top_crop}  (confidence: {top_conf*100:.1f}%)")
if top_conf < CONFIDENCE_THRESHOLD and top_crop in CONFUSABLE_VEGS:
    print("→ Tier 2: Low confidence on confusable vegetable.")
    print("  Returning full vegetable cluster as equally valid options:")
    for veg in sorted(CONFUSABLE_VEGS):
        if veg in le_crop.classes_:
            idx = le_crop.transform([veg])[0]
            print(f"    {veg:<15}  p={raw_proba[idx]*100:.1f}%")
else:
    print("→ Tier 1: Confident prediction. Top-3 recommendations:")
    for idx in top3_idxs:
        print(f"    {le_crop.classes_[idx]:<15}  confidence: {raw_proba[idx]*100:.1f}%")
        
'''




import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import time

from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
)

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    top_k_accuracy_score,
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)

from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from xgboost import XGBClassifier


# =====================================================
# CONFIG
# =====================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
TOP_N = 5
MIN_SAMPLES = 50


# =====================================================
# LOAD DATASET
# =====================================================

# Dataset columns:
# N, P, K, temperature, humidity, ph, rainfall, label

df = pd.read_csv("data/soil_data_updated.csv")

print("Original Shape:", df.shape)
print("Original Crops:", df["label"].nunique())


# =====================================================
# BASIC CLEANING
# =====================================================

# Lowercase crop names

df["label"] = (
    df["label"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Remove duplicates

df = df.drop_duplicates().reset_index(drop=True)

# Remove impossible values

df = df[
    (df["N"] >= 0)
    & (df["P"] >= 0)
    & (df["K"] >= 0)
    & (df["temperature"] > 0)
    & (df["humidity"] > 0)
    & (df["rainfall"] > 0)
]


# =====================================================
# REMOVE VERY SMALL CLASSES
# =====================================================

crop_counts = df["label"].value_counts()
valid_crops = crop_counts[crop_counts >= MIN_SAMPLES].index

removed = set(df["label"].unique()) - set(valid_crops)

if removed:
    print(f"\nDropped Crops (< {MIN_SAMPLES} samples):")
    print(sorted(list(removed)))


df = df[df["label"].isin(valid_crops)].reset_index(drop=True)

print("\nFiltered Shape:", df.shape)
print("Remaining Crops:", df["label"].nunique())
print(sorted(df["label"].unique()))


# =====================================================
# FEATURE ENGINEERING
# =====================================================

# Nutrient ratios

df["N_P_ratio"] = df["N"] / (df["P"] + 1)
df["N_K_ratio"] = df["N"] / (df["K"] + 1)
df["P_K_ratio"] = df["P"] / (df["K"] + 1)

# Total nutrients

df["NPK_total"] = df["N"] + df["P"] + df["K"]

# Climate interaction features

df["temp_humidity"] = (
    df["temperature"] * df["humidity"] / 100
)

df["rain_humidity"] = (
    df["rainfall"] * df["humidity"] / 100
)

# Soil chemistry interactions

df["ph_N"] = df["ph"] * df["N"]
df["ph_P"] = df["ph"] * df["P"]
df["ph_K"] = df["ph"] * df["K"]

# Moisture stress

df["temp_rain"] = (
    df["temperature"] * df["rainfall"] / 100
)

# Nutrient balance score

df["nutrient_balance"] = (
    abs(df["N"] - df["P"])
    + abs(df["P"] - df["K"])
    + abs(df["N"] - df["K"])
)


# =====================================================
# FEATURES
# =====================================================

FEATURES = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall",
    "N_P_ratio",
    "N_K_ratio",
    "P_K_ratio",
    "NPK_total",
    "temp_humidity",
    "rain_humidity",
    "ph_N",
    "ph_P",
    "ph_K",
    "temp_rain",
    "nutrient_balance",
]

TARGET = "label"

X = df[FEATURES]
y = df[TARGET]


# =====================================================
# LABEL ENCODING
# =====================================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nTotal Classes: {len(label_encoder.classes_)}")
print(f"Features Used: {len(FEATURES)}")


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=TEST_SIZE,
    stratify=y_encoded,
    random_state=RANDOM_STATE,
)

print(f"Train Shape: {X_train.shape}")
print(f"Test Shape : {X_test.shape}")


# =====================================================
# SCALING
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# =====================================================
# MODELS
# =====================================================

rf_model = RandomForestClassifier(
    n_estimators=1200,
    max_depth=18,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


et_model = ExtraTreesClassifier(
    n_estimators=1200,
    max_depth=18,
    min_samples_split=4,
    min_samples_leaf=2,
    class_weight="balanced_subsample",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


xgb_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="multi:softprob",
    eval_metric="mlogloss",
    reg_alpha=0.3,
    reg_lambda=1.5,
    gamma=0.1,
    min_child_weight=2,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


# =====================================================
# ENSEMBLE
# =====================================================

ensemble = VotingClassifier(
    estimators=[
        ("rf", rf_model),
        ("et", et_model),
        ("xgb", xgb_model),
    ],
    voting="soft",
    n_jobs=-1,
)


# =====================================================
# CROSS VALIDATION
# =====================================================

print("\nRunning Cross Validation...")

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

models = {
    "RandomForest": rf_model,
    "ExtraTrees": et_model,
    "XGBoost": xgb_model,
    "Ensemble": ensemble,
}

for name, model in models.items():
    scores = cross_val_score(
        model,
        scaler.fit_transform(X),
        y_encoded,
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
    )

    print(
        f"{name:<15} "
        f"CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}"
    )


# =====================================================
# TRAIN FINAL MODEL
# =====================================================

print("\nTraining Final Ensemble...")

ensemble.fit(X_train_scaled, y_train)


# =====================================================
# PREDICTION
# =====================================================

y_pred = ensemble.predict(X_test_scaled)
y_prob = ensemble.predict_proba(X_test_scaled)


# =====================================================
# EVALUATION
# =====================================================

print("\nMODEL EVALUATION")
print("=" * 60)

# Top-K Accuracy

top1 = accuracy_score(y_test, y_pred)
top3 = top_k_accuracy_score(y_test, y_prob, k=3)
top5 = top_k_accuracy_score(y_test, y_prob, k=5)

print(f"Top-1 Accuracy : {top1:.4f}")
print(f"Top-3 Accuracy : {top3:.4f}")
print(f"Top-5 Accuracy : {top5:.4f}")

print()

print(
    f"Precision (Weighted): "
    f"{precision_score(y_test, y_pred, average='weighted'):.4f}"
)

print(
    f"Recall (Weighted)   : "
    f"{recall_score(y_test, y_pred, average='weighted'):.4f}"
)

print(
    f"F1 Score (Weighted) : "
    f"{f1_score(y_test, y_pred, average='weighted'):.4f}"
)

# =====================================================
# INFERENCE TIME
# =====================================================

start = time.time()

ensemble.predict(X_test_scaled[:1])

end = time.time()

print(
    f"\nInference Time: "
    f"{(end - start) * 1000:.2f} ms"
)


# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0,
    )
)


# =====================================================
# TOP-3 ACCURACY PER CROP
# =====================================================

# print("\nCrop-wise Top-3 Accuracy")
# print("=" * 60)

# for idx, crop in enumerate(label_encoder.classes_):

#     mask = (y_test == idx)

#     if mask.sum() == 0:
#         continue

#     crop_prob = y_prob[mask]
#     crop_true = y_test[mask]

#     crop_top3 = np.mean([
#         crop_true[i] in np.argsort(crop_prob[i])[-3:]
#         for i in range(len(crop_true))
#     ])

#     bar = "█" * int(crop_top3 * 20)

#     print(
#         f"{crop:<15} "
#         f"Top-3: {crop_top3:.2f} "
#         f"{bar} "
#         f"(n={mask.sum()})"
#     )


# =====================================================
# CONFUSION MATRIX
# =====================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(14, 10))

sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_,
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

print("\nFeature Importance (Random Forest)")
print("=" * 60)

rf_fitted = ensemble.estimators_[0]
importances = rf_fitted.feature_importances_

importance_df = pd.DataFrame({
    "Feature": FEATURES,
    "Importance": importances,
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False,
)

for _, row in importance_df.iterrows():

    bar = "█" * int(row["Importance"] * 80)

    print(
        f"{row['Feature']:<20} "
        f"{row['Importance']:.4f} "
        f"{bar}"
    )


# =====================================================
# SAVE ARTIFACTS
# =====================================================

joblib.dump(ensemble, "models/soil_model.pkl")
joblib.dump(scaler, "models/soil_scaler.pkl")
joblib.dump(label_encoder, "models/soil_label_encoder.pkl")
joblib.dump(FEATURES, "models/soil_features.pkl")

print("\nAll model artifacts saved successfully.")


# =====================================================
# SMART PREDICTION FUNCTION
# =====================================================

CONFIDENCE_THRESHOLD = 0.45


def predict_crop(
    N,
    P,
    K,
    temperature,
    humidity,
    ph,
    rainfall,
    top_n=3,
):

    n_p_ratio = N / (P + 1)
    n_k_ratio = N / (K + 1)
    p_k_ratio = P / (K + 1)

    npk_total = N + P + K

    temp_humidity = temperature * humidity / 100
    rain_humidity = rainfall * humidity / 100

    ph_N = ph * N
    ph_P = ph * P
    ph_K = ph * K

    temp_rain = temperature * rainfall / 100

    nutrient_balance = (
        abs(N - P)
        + abs(P - K)
        + abs(N - K)
    )

    row = np.array([[
        N,
        P,
        K,
        temperature,
        humidity,
        ph,
        rainfall,
        n_p_ratio,
        n_k_ratio,
        p_k_ratio,
        npk_total,
        temp_humidity,
        rain_humidity,
        ph_N,
        ph_P,
        ph_K,
        temp_rain,
        nutrient_balance,
    ]])

    row_scaled = scaler.transform(row)

    probabilities = ensemble.predict_proba(row_scaled)[0]

    top_indices = np.argsort(probabilities)[::-1][:top_n]

    results = []

    for idx in top_indices:

        crop = label_encoder.classes_[idx]
        confidence = probabilities[idx]

        results.append({
            "crop": crop,
            "confidence": round(float(confidence), 4),
        })

    return results


# =====================================================
# DEMO PREDICTION
# =====================================================

# print("\nDemo Prediction")
# print("=" * 60)

# sample = predict_crop(
#     N=90,
#     P=42,
#     K=43,
#     temperature=21,
#     humidity=82,
#     ph=6.5,
#     rainfall=200,
# )

# for item in sample:
#     print(
#         f"Crop: {item['crop']:<15} "
#         f"Confidence: {item['confidence'] * 100:.2f}%"
#     )

