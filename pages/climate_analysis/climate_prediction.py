import streamlit as st
import pandas as pd
from scripts.model_prediction import prepare_input, predict  

# ✅ Added function to load district coordinates
def get_district_data():
    df = pd.read_csv("data/climate_data/climate_data_nepal_district_wise_monthly.csv")
    district_info = df.groupby("DISTRICT")[["LAT", "LON"]].mean().reset_index()
    return {
        row["DISTRICT"]: {"lat": row["LAT"], "lon": row["LON"]}
        for _, row in district_info.iterrows()
    }

def display():
    st.write("""
    ## 🔍 Climate Prediction
    This page allows you to make predictions using the trained climate model.
    """)

    if 'trained_model' not in st.session_state or st.session_state.trained_model is None:
        st.warning("⚠️ Please train the model first on the Model Train page.")
        return

    model = st.session_state.trained_model

    # Load district data
    district_coords = get_district_data()

    # Select inputs from the user
    selected_year = st.number_input("📅 Enter Year", min_value=2023, max_value=2100, step=1)
    selected_month_name = st.selectbox("🧭 Select Month", [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December"
    ])
    selected_district = st.selectbox("📍 Select District", list(district_coords.keys()))

    month_to_number = {
        "January": 1, "February": 2, "March": 3, "April": 4,
        "May": 5, "June": 6, "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12
    }
    selected_month = month_to_number[selected_month_name]
    selected_lat = district_coords[selected_district]['lat']
    selected_lon = district_coords[selected_district]['lon']

    # When the user clicks "Predict", prepare data and generate prediction
    if st.button("🚀 Predict"):
        input_data = prepare_input(selected_year, selected_month, selected_district)
        prediction = predict(model, input_data)

        target_cols = [
            'precipitation_total', 'relative_humidity_2m', 'air_temp_2m',
            'max_temp_2m', 'min_temp_2m', 'wind_speed_10m', 'max_wind_speed_10m', 'min_wind_speed_10m'
        ]

        result = pd.DataFrame({
            'Parameter': target_cols,
            'Predicted Value': prediction
        })

        st.success(f"📈 Predictions for {selected_month_name} {selected_year} in {selected_district}:")
        st.dataframe(result)
        st.balloons()
