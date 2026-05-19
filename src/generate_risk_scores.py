import pandas as pd

# -------------------------------
# Load datasets
# -------------------------------

yield_df = pd.read_csv("data/clean_yield_data.csv")
price_df = pd.read_csv("data/crop_price_data.csv")

# -------------------------------
# Clean crop names
# -------------------------------

yield_df["Crop"] = yield_df["Crop"].str.lower().str.strip()

price_df = price_df.rename(columns={
    "CROP": "Crop",
    "CROP_PRICE": "Price"
})

price_df["Crop"] = price_df["Crop"].str.lower().str.strip()

# -------------------------------
# Yield Risk
# -------------------------------

yield_stats = yield_df.groupby("Crop")["Yield"].agg(
    ["mean", "std"]
).reset_index()

yield_stats["yield_risk"] = (
    yield_stats["std"] / yield_stats["mean"]
)

# -------------------------------
# Price Risk
# -------------------------------

price_stats = price_df.groupby("Crop")["Price"].agg(
    ["mean", "std"]
).reset_index()

price_stats["price_risk"] = (
    price_stats["std"] / price_stats["mean"]
)

yield_crops = set(yield_df["Crop"].unique())
price_crops = set(price_df["Crop"].unique())

common_crops = yield_crops & price_crops

print("Yield crops :", len(yield_crops))
print("Price crops :", len(price_crops))
print("Common crops:", len(common_crops))


# -------------------------------
# Merge Risks
# -------------------------------

risk_df = pd.merge(
    yield_stats[["Crop", "yield_risk"]],
    price_stats[["Crop", "price_risk"]],
    on="Crop",
    how="inner"
)

# -------------------------------
# Final Risk Score
# -------------------------------

risk_df["risk_score"] = (
    risk_df["yield_risk"] +
    risk_df["price_risk"]
) / 2

# -------------------------------
# Fill missing
# -------------------------------

risk_df = risk_df.fillna(risk_df["risk_score"].mean())

# -------------------------------
# Save
# -------------------------------

risk_df.to_csv("data/crop_risk_scores.csv", index=False)

print("✅ Risk scores saved!")
print(risk_df.head())