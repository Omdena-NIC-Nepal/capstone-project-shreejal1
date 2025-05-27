import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Title and description for the prediction page
st.title("Climate Data Prediction")
st.markdown("""
    In this section, we will generate predictions based on the trained model. You can input new data
    (such as temperature, precipitation, etc.) and predict the temperature (T2M) using the trained model.
""")

# Load the trained model
model_dir = "models"
model_path = os.path.join(model_dir, "random_forest_model.pkl")
scaler_path = os.path.join(model_dir, "scaler.pkl")

try:
    if not os.path.exists(model_dir):
        st.error("Models directory doesn't exist. Please train the model first.")
        st.stop()
        
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.success("Model and scaler loaded successfully!")
    
except FileNotFoundError as e:
    st.error(f"Model files not found: {e}")
    st.error("Please train the model first before making predictions.")
    st.stop()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Only show prediction UI if model loaded successfully
st.subheader("Input Climate Data")

# Input for different climate features
district = st.selectbox("Select District", options=['District1', 'District2', 'District3'])
temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, step=0.1)
precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=500.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, step=0.1)

if st.button("Predict Temperature"):
    try:
        # Prepare input data
        user_input = pd.DataFrame({
            'T2M': [temperature],
            'PRECTOT': [precipitation],
            'RH2M': [humidity],
            'WS10M': [wind_speed]
        })
        
        # Scale the input
        user_input_scaled = scaler.transform(user_input)
        
        # Make prediction
        predicted_temperature = model.predict(user_input_scaled)
        
        # Display result
        st.subheader("Prediction Result")
        st.success(f"Predicted Temperature: {predicted_temperature[0]:.2f} °C")
        
    except Exception as e:
        st.error(f"Prediction failed: {e}")