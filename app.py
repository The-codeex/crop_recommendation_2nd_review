
#  last working code  of app.py


import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel

# -------------------------------
# Background Image
# -------------------------------

def get_base64_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

def set_background(image_path):
    img_base64 = get_base64_image(image_path)
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{img_base64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        /* Dark overlay so text remains readable */
        .stApp::before {{
            content: "";
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.55);
            z-index: 0;
        }}

        /* Ensure all content sits above overlay */
        .stApp > * {{
            position: relative;
            z-index: 1;
        }}

        /* White text throughout */
        h1, h2, h3, h4, h5, h6,
        .stMarkdown, .stText, label,
        .stSelectbox label, .stNumberInput label,
        .stTextInput label, p {{
            color: #FFFFFF !important;
        }}

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            background-color: rgba(0, 0, 0, 0.4);
            border-radius: 10px;
            padding: 4px;
        }}

        .stTabs [data-baseweb="tab"] {{
            color: #FFFFFF !important;
            font-weight: 600;
        }}

        .stTabs [aria-selected="true"] {{
            background-color: rgba(255, 255, 255, 0.2) !important;
            border-radius: 8px;
        }}

        /* Input fields */
        .stSelectbox div[data-baseweb="select"] > div,
        .stTextInput input,
        .stNumberInput input {{
            background-color: rgba(255, 255, 255, 0.15) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
            border-radius: 8px;
        }}

        /* Predict button */
        .stButton > button {{
            background-color: #2D6A4F;
            color: #FFFFFF;
            font-weight: 700;
            border: none;
            border-radius: 8px;
            padding: 0.5rem 2rem;
            transition: background-color 0.2s;
        }}

        .stButton > button:hover {{
            background-color: #52B788;
            color: #FFFFFF;
        }}

        /* Dataframe */
        .stDataFrame {{
            background-color: rgba(0, 0, 0, 0.4) !important;
            border-radius: 10px;
        }}

        /* Success box */
        .stSuccess {{
            background-color: rgba(45, 106, 79, 0.7) !important;
            border-radius: 8px;
            color: #FFFFFF !important;
        }}

        /* Subheader */
        .stSubheader {{
            color: #FFFFFF !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply background — make sure back-img.jpg is in the same folder as app.py
set_background("image/back-img.jpg")


# -------------------------------
# Tabs
# -------------------------------

tab1, tab2 = st.tabs([
    "💰 Profit Based Recommendation",
    "Soil based Recommendation"
    
])


# -------------------------------
# Load trained models
# -------------------------------

yield_model   = joblib.load("models/yield_model.pkl")
le_state      = joblib.load("models/state_encoder.pkl")
le_crop       = joblib.load("models/crop_encoder.pkl")
le_season     = joblib.load("models/season_encoder.pkl")
le_Dist     = joblib.load("models/Dist_encoder.pkl")
# Yield feature engineering objects
poly = joblib.load(
    "models/poly_transformer.pkl"
)

selected_features = joblib.load(
    "models/yield_selected_features.pkl"
)




# price model
price_model = joblib.load("models/price_model.pkl")
price_le_state = joblib.load("models/price_state_encoder.pkl")
price_le_crop = joblib.load("models/price_crop_encoder.pkl")
price_le_soil = joblib.load("models/price_soil_encoder.pkl")

# Load soil model# Load soil model

soil_model = joblib.load(
    "models/soil_model.pkl"
)
soil_scaler = joblib.load(
    "models/soil_scaler.pkl"
)
soil_label_encoder = joblib.load(
    "models/soil_label_encoder.pkl"
)
soil_features = joblib.load(
    "models/soil_features.pkl"
)

# Load cost dataset
cost_df = pd.read_csv("data/common_crop_cost.csv")

crop_cost = dict(zip(cost_df["Crop"], cost_df["Cost"]))
#risk dataset
risk_df = pd.read_csv("data/crop_risk_scores.csv")

crop_risk = dict(
    zip(risk_df["Crop"], risk_df["risk_score"])
)

# price_df = pd.read_csv("data/crop_price_avg.csv")
# crop_avg_price = dict(zip(price_df["Crop"], price_df["Price"]))






# -------------------------------
# Tab 1 — Profit Based Recommendation
# -------------------------------

with tab1:
    st.title("AI-Based Crop Recommendation System")
    st.write("Predict the most profitable crop")
    

    default_state = "odisha"

    state  = st.selectbox("Select State", le_state.classes_, index=list(le_state.classes_).index(default_state))
    state = state.lower().strip()

    season = st.selectbox("Select Season", le_season.classes_)

    area = st.number_input("Area (hectares)", min_value=1.0)
    year = st.number_input("Year", min_value=2000, max_value=2035, value=2026)
    month = st.selectbox(
    "Month",
    list(range(1, 13)),
    index=5
    )
    # ✅ UPDATED INPUTS
    rainfall   = st.number_input("Annual Rainfall (mm)")
    fertilizer = st.number_input("Fertilizer usage (kg/ha)")
    pesticide  = st.number_input("Pesticide usage (kg/ha)")
    temperature = st.number_input("Temperature (Celsis)")
    
    # price model 
    soil_type = st.selectbox("Soil Type", price_le_soil.classes_)
    n_soil = st.number_input("Nitrogen (N)")
    p_soil = st.number_input("Phosphorus (P)")
    k_soil = st.number_input("Potassium (K)")
    humidity = st.number_input("Humidity (%)")
    ph = st.number_input("Soil pH")
    supply_volume = st.number_input(
    "Supply Volume (tons)",
    min_value=100.0
    )

    demand_volume = st.number_input(
    "Demand Volume (tons)",
    min_value=100.0
    )

    if st.button("Predict Best Crop"):

        state_encoded  = le_state.transform([state])[0]
        # -------------------------------
        # SAFE STATE ENCODING
        # -------------------------------



        price_states = set(price_le_state.classes_)

        fallback_state = price_le_state.classes_[0]

        if state in price_states:
            price_state_encoded = price_le_state.transform([state])[0]
        else:
            price_state_encoded = price_le_state.transform(
            [fallback_state]
                )[0]
            
        season_encoded = le_season.transform([season])[0]

        candidate_crops = list(le_crop.classes_)
        results = []
        # Pre-calc (outside loop)
        price_supported_crops = set(price_le_crop.classes_)
        # default_price = np.mean(list(crop_avg_price.values()))
        
        

        candidate_crops = [
            crop for crop in le_crop.classes_
            if crop in price_supported_crops
        ]

        for crop in candidate_crops:

            crop_encoded = le_crop.transform([crop])[0]

            # ✅ UPDATED MODEL INPUT
            # -------------------------------
            # Yield Feature Engineering
            # -------------------------------

            full_input = pd.DataFrame([{

                "State": state_encoded,

                "Crop": crop_encoded,

                "Season": season_encoded,

                "year_from_start":
                    year - 1997,

                "log_area":
                    np.log1p(area),
                
                "log_production":
                    np.log1p(area * 10),

                "log_rainfall":
                    np.log1p(rainfall),

                "fertilizer_per_hectare":
                    fertilizer,

                "pesticide_per_hectare":
                    pesticide,

                "input_intensity":
                    fertilizer + pesticide
                    
                
                }])[[
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
]]


            # -------------------------------
            # Polynomial Transformation
            # -------------------------------

            poly_input = poly.transform(full_input)

            generated_feature_names = (
                poly.get_feature_names_out(
                full_input.columns
            )
            )

            poly_df = pd.DataFrame(
                poly_input,
                columns=generated_feature_names
            )

            # Keep selected features only
            yield_input = poly_df[
            selected_features
            ]


            # -------------------------------
            # Yield Prediction
            # -------------------------------

            yield_log = yield_model.predict(
            yield_input
            )

            predicted_yield = max(
                0.1,
            np.expm1(yield_log)[0]
            )
            
            # -------------------------------
            # SAFE PRICE PREDICTION 🔥
            # -------------------------------
            # -------------------------------
            # PRICE PREDICTION (MODEL ONLY ✅)
            # -------------------------------

            # price_input = pd.DataFrame([{
            # "State": price_state_encoded,
            # "SOIL_TYPE": price_le_soil.transform([soil_type])[0],
            # "Crop": price_le_crop.transform([crop])[0],
            # "N_SOIL": n_soil,
            # "P_SOIL": p_soil,
            # "K_SOIL": k_soil,
            # "TEMPERATURE": temperature,
            # "HUMIDITY": humidity,
            # "ph": ph,
            # "RAINFALL": rainfall
            # }])

            # price_log = price_model.predict(price_input)
            # predicted_price = max(1, np.expm1(price_log)[0])
            
            # -------------------------------
            # PRICE FEATURE ENGINEERING
            # -------------------------------

            year_price = year

            

            month_sin = np.sin(
                2 * np.pi * month / 12
            )

            month_cos = np.cos(
            2 * np.pi * month / 12
            )

            npk_sum = (
            n_soil +
            p_soil +
            k_soil
            )

            temp_humidity = (
            np.log1p(temperature) *
            np.log1p(humidity)
            )

            rain_temp = (
            np.log1p(rainfall) *
            np.log1p(temperature)
            )
            
            # -------------------------------
            # Supply/Demand Feature Engineering
            # -------------------------------

            # log transform
            supply_volume_log = np.log1p(
            supply_volume
            )

            demand_volume_log = np.log1p(
            demand_volume
            )

            supply_demand_ratio = (
            supply_volume_log /
            (demand_volume_log + 1)
            )



            # -------------------------------
            # PRICE MODEL INPUT
            # -------------------------------

            price_input = pd.DataFrame([{
                "Year": year_price,
                "Month_sin": month_sin,
                "Month_cos": month_cos,
                "State": price_state_encoded,
                "SOIL_TYPE": price_le_soil.transform(
                    [soil_type]
                )[0],
                "Crop": price_le_crop.transform(
                    [crop]
                )[0],
                "N_SOIL": n_soil / 100,
                "P_SOIL": p_soil / 100,
                "K_SOIL": k_soil / 100,
                "TEMPERATURE": np.log1p(
                    temperature
                ),
                "HUMIDITY": np.log1p(
                    humidity
                ),
                "ph": ph,
                "RAINFALL": np.log1p(
                    rainfall
                ),
                "NPK_SUM": npk_sum / 100,
                "TEMP_HUMIDITY": temp_humidity,
                "RAIN_TEMP": rain_temp,
                "Supply Volume (tons)": supply_volume_log,

                "Demand Volume (tons)": demand_volume_log,

                "SUPPLY_DEMAND_RATIO": supply_demand_ratio,
            }])
            # ✅ ADD THIS HERE
            price_input = price_input[[

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
            ]]


            # -------------------------------
            # PRICE PREDICTION
            # -------------------------------

            price_log = price_model.predict(
                price_input
            )

            predicted_price = max(
                1,
            np.expm1(price_log)[0]
            )


           

            # -------------------------------
            # Profit Calculation (fixed)
            # -------------------------------
            # default_cost = cost_df["Cost"].mean()
            # cost_per_hectare = crop_cost.get(crop)

            # profit_per_hectare = ((predicted_yield *10 ) * predicted_price) - cost_per_hectare
            # profit = (profit_per_hectare * area)
            
            #per hector cost profit calculation
            # yield_quintal_per_hectare = (
            # predicted_yield * 10
            # )
            
            # total_quintal = (
            # yield_quintal_per_hectare * area
            # )
            
            # total_revenue = (
            # total_quintal * predicted_price
            # )
            # total_cost = (
            # cost_per_hectare * area
            # )
            # profit = (
            # total_revenue - total_cost
            # ) / 100000
            
            #per quintal cost profit calculation
            cost_per_quintal = crop_cost.get(crop)
            yield_quintal_per_hectare = (
            predicted_yield * 10
            )

            total_quintal = (
            yield_quintal_per_hectare * area
            )

            total_revenue = (
            predicted_price * total_quintal
            )
            if cost_per_quintal is None:
                st.write("Missing crop:", crop)
                
            total_cost = (
            cost_per_quintal * total_quintal
            )

            profit = (
                total_revenue - total_cost
            ) / 100000
            
            
            risk_score = crop_risk.get(crop, 0.5)
            final_score = profit / (1 + risk_score)
            # stability_score = 1 / (1 + risk_score)
            
            
            
           

            


            results.append({
                "Crop": crop,
                "Predicted Yield [Quintal/hec]": predicted_yield,
                "Predicted Price [(₹)/Quintal]": predicted_price,
                "Risk Score": round(risk_score, 2),
                "Cost/Quintal [(₹)]": cost_per_quintal,
                "Expected Profit (₹ Lakh/hec)": round(profit, 2),
                "Risk-Adjusted Profit Score": round(final_score, 2)
                
            })

        results_df = pd.DataFrame(results)
        
        # -------------------------------
        # Dynamic Recommendation Category
        # -------------------------------

        q75 = results_df[
            "Expected Profit (₹ Lakh/hec)"
            ].quantile(0.75)

        q50 = results_df[
            "Expected Profit (₹ Lakh/hec)"
            ].quantile(0.50)

        q25 = results_df[
            "Expected Profit (₹ Lakh/hec)"
            ].quantile(0.25)


        def get_recommendation(profit):

            if profit >= q75:
                return "Highly Profitable"

            elif profit >= q50:
                return "Profitable"

            elif profit >= q25:
                return "Moderate Risk"

            else:
                return "High Risk / Loss"


        results_df["Recommendation"] = (
            results_df[
            "Expected Profit (₹ Lakh/hec)"
            ].apply(get_recommendation)
        )

        # ✅ Sort by Risk-Adjusted Profit Score (descending)
        results_df = results_df.sort_values(
            by="Risk-Adjusted Profit Score", 
                ascending=False
                ).reset_index(drop=True)
        # -------------------------------
        # Different Recommendation Types
        # -------------------------------

        st.subheader("All Crops (Sorted by Risk-Adjusted Profit Score)")
        st.write(f"Total crops: {len(results_df)}")
        st.dataframe(results_df, height=900)
        
        # best_crop = results_df.iloc[0]["Crop"]
        # st.success(f"Recommended Crop: {best_crop}")
        
        
        
        # Plot
        import matplotlib.pyplot as plt

        # -------------------------------
        # Lowest Risk Crops
        # -------------------------------

        top10_df = results_df.sort_values(
            by="Risk Score",
            ascending=True
        ).head(15)

        fig, ax = plt.subplots()

        ax.bar(
            top10_df["Crop"],
            top10_df["Risk Score"]
            )

        ax.set_ylabel("Risk Score")
        ax.set_title("Top 15 Lowest Risk Crops")

        plt.xticks(rotation=45)

        st.pyplot(fig)
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
# ----------------------------------------------
# Soil Based Crop Recommendation
# ----------------------------------------------
# with tab2:
#     st.title("Soil Based Crop Recommendation")

#     soil_state = st.selectbox(
#         "Select State",
#         soil_state_encoder.classes_,
#         key="soil_state"
#     )

#     soil_type = st.selectbox(
#         "Select Soil Type",
#         soil_type_encoder.classes_
#     )

#     n           = st.number_input("Nitrogen (N)",      min_value=0.0, key="soil_n")
#     p           = st.number_input("Phosphorus (P)",    min_value=0.0, key="soil_p")
#     k           = st.number_input("Potassium (K)",     min_value=0.0, key="soil_k")
#     temperature = st.number_input("Temperature (°C)",  min_value=0.0, key="soil_temp")
#     humidity    = st.number_input("Humidity (%)",      min_value=0.0, key="soil_hum")
#     ph          = st.number_input("pH Value",          min_value=0.0, key="soil_ph")
#     rainfall    = st.number_input("Rainfall (mm)",     min_value=0.0, key="soil_rain")
#     crop_price  = st.number_input(
#         "Expected Crop Price (₹/Quintal)",
#         min_value=0.0,
#         value=1500.0,
#         help="Enter the average market price you expect for crops in your region"
#     )

#     if st.button("Recommend Crops (Soil Model)"):

#         state_encoded = soil_state_encoder.transform([soil_state])[0]
#         soil_encoded  = soil_type_encoder.transform([soil_type])[0]

#         n_p_ratio  = n / (p + 1)
#         n_k_ratio  = n / (k + 1)
#         p_k_ratio  = p / (k + 1)
#         npk_total  = n + p + k
#         temp_humid = temperature * humidity / 100
#         rain_humid = rainfall    * humidity / 100

#         input_data = np.array([[
#             state_encoded, soil_encoded,
#             n, p, k,
#             temperature, humidity, ph, rainfall,
#             crop_price,
#             n_p_ratio, n_k_ratio, p_k_ratio,
#             npk_total, temp_humid, rain_humid
#         ]])

#         input_scaled = soil_scaler.transform(input_data)
#         probs        = soil_model.predict_proba(input_scaled)[0]

#         top_crop = soil_crop_encoder.classes_[np.argmax(probs)]
#         top_conf = probs[np.argmax(probs)]

#         if top_conf < 0.40 and top_crop in soil_confusable_vegs:
#             # st.warning("⚠ Similar growing conditions detected for multiple crops.")
#             st.write("All of the following vegetables grow well in these conditions:")
#             cluster = []
#             for veg in sorted(soil_confusable_vegs):
#                 if veg in soil_crop_encoder.classes_:
#                     idx = np.where(soil_crop_encoder.classes_ == veg)[0][0]
#                     cluster.append((veg, probs[idx]))
#             cluster.sort(key=lambda x: -x[1])
#             for i, (veg, conf) in enumerate(cluster, 1):
#                 st.write(f"{i}. **{veg}**  —  score: {conf*100:.1f}%")
#         else:
#             st.success("✅ Top Suitable Crops Based on Soil")
#             for i, idx in enumerate(np.argsort(probs)[::-1][:5], 1):
#                 st.write(f"{i}. **{soil_crop_encoder.classes_[idx]}**  —  {probs[idx]*100:.1f}%")
        
with tab2:

    st.title("Soil Based Crop Recommendation")

    # =====================================================
    # USER INPUTS
    # =====================================================

    n = st.number_input(
        "Nitrogen (N)",
        min_value=0.0,
        value=90.0,
    )

    p = st.number_input(
        "Phosphorus (P)",
        min_value=0.0,
        value=42.0,
    )

    k = st.number_input(
        "Potassium (K)",
        min_value=0.0,
        value=43.0,
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=0.0,
        value=21.0,
    )

    humidity = st.number_input(
        "Humidity (%)",
        min_value=0.0,
        max_value=100.0,
        value=82.0,
    )

    ph = st.number_input(
        "pH Value",
        min_value=0.0,
        value=6.5,
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        value=200.0,
    )

    # =====================================================
    # PREDICTION
    # =====================================================

    if st.button("Recommend Crops"):

        # ---------------------------------------------
        # FEATURE ENGINEERING
        # ---------------------------------------------

        n_p_ratio = n / (p + 1)
        n_k_ratio = n / (k + 1)
        p_k_ratio = p / (k + 1)

        npk_total = n + p + k

        temp_humidity = temperature * humidity / 100
        rain_humidity = rainfall * humidity / 100

        ph_N = ph * n
        ph_P = ph * p
        ph_K = ph * k

        temp_rain = temperature * rainfall / 100

        nutrient_balance = (
            abs(n - p)
            + abs(p - k)
            + abs(n - k)
        )

        # ---------------------------------------------
        # INPUT ARRAY
        # ---------------------------------------------

        input_data = np.array([[

            n,
            p,
            k,
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

        # ---------------------------------------------
        # SCALING
        # ---------------------------------------------

        input_scaled = soil_scaler.transform(input_data)

        # ---------------------------------------------
        # PREDICTION
        # ---------------------------------------------

        probabilities = soil_model.predict_proba(input_scaled)[0]

        # normalize probabilities
        probabilities = probabilities / probabilities.sum()

        sorted_indices = np.argsort(probabilities)[::-1]

        top_indices = [
            idx for idx in sorted_indices
            if probabilities[idx] > 0.03
            ][:5]

        # ---------------------------------------------
        # DISPLAY RESULTS
        # ---------------------------------------------

        st.success("Top Recommended Crops")

        for rank, idx in enumerate(top_indices, start=1):

            crop_name = soil_label_encoder.classes_[idx]
            confidence = probabilities[idx] * 100

            st.write(
                f"{rank}. **{crop_name.title()}** — "
                f"{confidence:.2f}% confidence"
            )

            max_prob = probabilities[top_indices[0]]

            st.progress(float(probabilities[idx] / max_prob))

        # ---------------------------------------------
        # BEST CROP
        # ---------------------------------------------

        best_crop = soil_label_encoder.classes_[top_indices[0]]
        best_conf = probabilities[top_indices[0]] * 100

        st.info(
            f"Best Suitable Crop: "
            f"**{best_crop.title()}** "
            f"({best_conf:.2f}% confidence)"
        )
        
        
        
        
        
        
        






































