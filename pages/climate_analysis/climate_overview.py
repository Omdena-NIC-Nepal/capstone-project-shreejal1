import streamlit as st
from scripts.data_utils import load_data, clean_data, sanitize_for_arrow

def display():
    st.title("🌤️ Nepal Climate Data Overview")
    
    st.subheader("🔍 Data Preview")

    # Load and preprocess the dataset if not already loaded
    if "cleaned_climatedf" not in st.session_state:
        climatedf = load_data("data/climate_data/climate_data_nepal_district_wise_monthly.csv")
        if climatedf is not None:
            cleaned_climatedf = clean_data(climatedf, method='dropna')
            cleaned_climatedf = sanitize_for_arrow(cleaned_climatedf)
            st.session_state.cleaned_climatedf = cleaned_climatedf
        else:
            st.warning("Failed to load climate data.")
            return

    cleaned_climatedf = st.session_state.cleaned_climatedf
    
    if cleaned_climatedf.empty:
        st.warning("Climate dataset is empty after cleaning.")
        return

    st.dataframe(cleaned_climatedf, height=200)

    st.divider()

    st.markdown("""
    ## 🌍 Climate Dataset Description: Nepal District-wise Monthly Data

    This dataset captures **monthly climate statistics** across various districts in **Nepal**, with records starting from **1981**. Below is a feature breakdown:

    ### 📅 Temporal Columns
    - `DATE`: Full date (e.g., "1/31/1981")
    - `YEAR`, `MONTH`: Extracted components for easier analysis

    ### 📍 Geographic Features
    - `DISTRICT`: District name
    - `LAT`, `LON`: Latitude and Longitude

    ### 🌧️ Precipitation and Pressure
    - `PRECTOT`: Total precipitation (mm)
    - `PS`: Surface pressure (kPa)

    ### 💧 Humidity & Moisture
    - `QV2M`: Specific humidity at 2m (g/kg)
    - `RH2M`: Relative humidity at 2m (%)

    ### 🌡️ Temperature Metrics
    - `T2M`, `T2MWET`: Mean and wet bulb temperature
    - `T2M_MAX`, `T2M_MIN`: Max/Min temperature at 2m
    - `T2M_RANGE`: Temp range (max - min)
    - `TS`: Surface skin temperature

    ### 🌬️ Wind Speeds
    - `WS10M`, `WS50M`: Mean wind speeds
    - `WS10M_MAX/MIN/RANGE`, `WS50M_MAX/MIN/RANGE`: Wind speed stats at both heights

    ---

    ### ✅ Use Cases:
    - Climate trend analysis
    - Extreme weather event studies
    - District-wise climate comparison
    """)

    st.divider()
