import pandas as pd
import numpy as np

# -------------------------------
# Load Original Dataset
# -------------------------------
df = pd.read_csv("data/yield_data.csv")

print("Original Shape :", df.shape)

# -------------------------------
# Clean Column Names
# -------------------------------
df.columns = df.columns.str.strip()

# -------------------------------
# Remove Duplicates
# -------------------------------
df = df.drop_duplicates()

print("After Duplicate Removal :", df.shape)

# -------------------------------
# Convert Numeric Columns
# -------------------------------
num_cols = [
    "Area",
    "Production",
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide",
    "Yield"
]

for col in num_cols:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# -------------------------------
# Handle Missing Values
# -------------------------------
# Fill numeric columns with median

for col in num_cols:

    df[col] = df[col].fillna(
        df[col].median()
    )

# Remove rows only if important
# categorical columns are missing

df = df.dropna(
    subset=[
        "Crop",
        "Season",
        "State"
    ]
)

# -------------------------------
# Clean Text Columns
# -------------------------------
# for col in ["Crop", "Season", "State"]:
#     df[col] = df[col].str.strip().str.lower()
    
for col in ["Crop", "Season", "State"]:
    df[col] = df[col].astype(str).str.strip().str.lower()

# -------------------------------
# Year Cleaning
# -------------------------------
df["Crop_Year"] = pd.to_numeric(
    df["Crop_Year"],
    errors="coerce"
)

df["Crop_Year"] = df["Crop_Year"].fillna(
    df["Crop_Year"].median()
)

# -------------------------------
# VERY LIGHT VALIDATION
# -------------------------------
# Remove only impossible values

df = df[
    (df["Area"] > 0) &
    (df["Annual_Rainfall"] >= 0) &
    (df["Fertilizer"] >= 0) &
    (df["Pesticide"] >= 0) &
    (df["Yield"] > 0)
]

# -------------------------------
# Remove Leakage Column
# -------------------------------
# Remove Production for realistic
# yield prediction

# df = df.drop(
#     columns=["Production"]
# )
# -------------------------------
# Outlier Removal (IQR)
# -------------------------------
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[(df[column] >= lower) & (df[column] <= upper)]

# Apply only where useful
outlier_cols = [
    "Annual_Rainfall",
    "Fertilizer",
    "Pesticide"
]

for col in num_cols:
    df = remove_outliers(df, col)
# -------------------------------
# Reset Index
# -------------------------------
df = df.reset_index(drop=True)

# -------------------------------
# Final Dataset Checks
# -------------------------------
print("\nFinal Shape :", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nDataset Info:")
print(df.info())

print("\nYield Statistics:")
print(df["Yield"].describe())

# -------------------------------
# Save Clean Dataset
# -------------------------------
df.to_csv(
    "data/clean_yield_data.csv",
    index=False
)

print("\n✅ Clean dataset saved successfully")