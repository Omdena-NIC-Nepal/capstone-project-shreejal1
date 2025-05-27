import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib

# Title and description for the prediction page
st.title("Climate Data Prediction")
st.markdown("""
    In this section, we will generate predictions based on the trained model. You can input new data
    (such as temperature, precipitation, etc.) and predict the temperature (T2M) using the trained model.
""")

# Load the trained model
model_path = "models/random_forest_model.pkl"
scaler_path = "models/scaler.pkl"

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.write("Model and scaler loaded successfully.")
except FileNotFoundError:
    st.error("Model or scaler file not found. Please ensure the model is trained and saved first.")

# Collect input data from the user
st.subheader("Input Climate Data")

# Input for different climate features (example: Temperature, Precipitation)
district = st.selectbox("Select District", options=['District1', 'District2', 'District3'])  # Example districts
temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, step=0.1)
precipitation = st.number_input("Precipitation (mm)", min_value=0.0, max_value=500.0, step=0.1)
humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, step=0.1)
wind_speed = st.number_input("Wind Speed (m/s)", min_value=0.0, max_value=50.0, step=0.1)

# Collecting all the inputs into a dataframe for prediction
user_input = pd.DataFrame({
    'T2M': [temperature], 
    'PRECTOT': [precipitation], 
    'RH2M': [humidity], 
    'WS10M': [wind_speed]
    # Add other features that the model uses for prediction
})

# Normalize the input data using the same scaler used for training
user_input_scaled = scaler.transform(user_input)

# Prediction button
if st.button("Predict Temperature"):
    # Generate prediction
    predicted_temperature = model.predict(user_input_scaled)

    # Display the result
    st.subheader("Predicted Temperature")
    st.write(f"The predicted temperature for the input data is: {predicted_temperature[0]:.2f} °C")