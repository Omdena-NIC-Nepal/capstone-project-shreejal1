import streamlit as st
import pandas as pd
import numpy as np
# No need to import RandomForestRegressor, StandardScaler here if loading from joblib/session_state
import joblib # Keep joblib for local saving if needed, but not for session_state
import os     # Keep os for local saving if needed, but not for session_state

# Title and description for the prediction page
st.title("Climate Data Prediction")
st.markdown("""
    In this section, we will generate predictions based on the trained model. You can input new data
    (such as temperature, precipitation, etc.) and predict the temperature (T2M) using the trained model.
""")

# --- Load model and scaler from session state ---
model = None
scaler = None
model_features = None

if 'trained_model' in st.session_state and st.session_state.trained_model is not None:
    model = st.session_state.trained_model
    st.success("Model loaded from session state successfully!")
else:
    st.warning("No trained model found in session state. Please train a model first on the Model Training page.")
    # Consider stopping the app or disabling prediction UI if no model is found
    # st.stop() # You might want to stop or just disable prediction inputs

if 'trained_scaler' in st.session_state and st.session_state.trained_scaler is not None:
    scaler = st.session_state.trained_scaler
    st.success("Scaler loaded from session state successfully!")
else:
    st.warning("No trained scaler found in session state. Please train a model first on the Model Training page.")
    # st.stop() # You might want to stop or just disable prediction inputs

if 'model_features' in st.session_state and st.session_state.model_features is not None:
    model_features = st.session_state.model_features
    st.info(f"Expecting features: {', '.join(model_features)}")
else:
    st.warning("Model features not found in session state. Please train a model first.")


# Only show prediction UI if model and scaler are available
if model and scaler and model_features:
    st.subheader("Input Climate Data")

    # Input for different climate features - ensure these match your model's X.columns
    # This is a critical point: The features you ask for here MUST match those your model was trained on
    # and in the correct order.
    # From your training script, X excludes 'DATE', 'YEAR', 'MONTH', 'T2M'.
    # So, your input features should be the remaining ones.
    # Let's assume your 'engineered_climate_data.csv' has columns like 'PRECTOT', 'RH2M', 'WS10M', etc.
    # You also had 'DISTRICT' in the other person's code, but not in your original X.
    # You need to be explicit about the features.

    # Example: If your X.columns were ['PRECTOT', 'RH2M', 'WS10M']
    # You might need to add inputs for 'YEAR', 'MONTH', 'LAT', 'LON' if they were in X.
    # For now, let's use the ones from your original prediction code, assuming they map to your model_features

    # --- Dynamically create input fields based on stored features ---
    input_data = {}
    st.write("Please provide values for the following features:")

    # Map your previous explicit inputs to the stored model_features
    # This part requires you to know what columns were in `X` during training.
    # Based on your training script: X = climate_data.drop(columns=['DATE', 'YEAR', 'MONTH', 'T2M'])
    # So, X would contain other numerical columns from your CSV.
    # For example, if your engineered_climate_data.csv contained 'Temperature', 'Precipitation', 'Humidity', 'Wind_Speed'
    # you'd need inputs for those.
    # The 'DISTRICT' input is problematic unless you also have it as a feature and encoded it.

    # Let's re-align the prediction input based on likely features in your 'engineered_climate_data.csv'
    # without 'DATE', 'YEAR', 'MONTH', 'T2M'
    # Assuming 'PRECTOT', 'RH2M', 'WS10M' are actual features, you might also have others.
    # We need to ensure the user inputs match the columns in `model_features`.

    # Example: Let's explicitly define inputs for features that were in X based on your training script
    # This assumes your X columns were primarily 'PRECTOT', 'RH2M', 'WS10M' and possibly others like 'LAT', 'LON', 'ELEV' etc.
    # If your original X had YEAR, MONTH as features, you need to add inputs for them.
    # For now, I'll match the previous user_input structure but highlight the importance of feature consistency.

    # Dummy values for features that might be in your model but not directly input by user (e.g., engineered features)
    # You need to create inputs for all features that your model was trained on!
    # For demonstration, let's assume `model_features` are 'PRECTOT', 'RH2M', 'WS10M', etc.
    # You had 'T2M' in your user_input DataFrame which is your *target*, this needs to be removed.
    # The `user_input` DataFrame should contain only the features (columns of X).

    # Example of how you might create inputs for some common features:
    # (Adjust these based on what's truly in your `X` after dropping 'DATE', 'YEAR', 'MONTH', 'T2M')
    # If you have 'LAT' and 'LON' as features:
    # latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.01)
    # longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.01)
    # If you have 'YEAR' and 'MONTH' as features:
    # year_input = st.number_input("Year", min_value=1900, max_value=2100, value=2023)
    # month_input = st.number_input("Month", min_value=1, max_value=12, value=7)

    # Let's stick to the previous inputs, but make sure they map to the *expected features*
    # based on `model_features`.
    # IMPORTANT: The column names in `user_input` DataFrame MUST match the `model_features` exactly.
    # AND in the correct order for `scaler.transform` and `model.predict` when using numpy arrays.
    # It's safer to create the DataFrame with specific columns from `model_features`.

    # Let's retrieve actual features from `st.session_state.model_features`
    input_values = {}
    for feature in model_features:
        if feature == 'PRECTOT':
            input_values[feature] = st.number_input("Total Precipitation (mm)", min_value=0.0, max_value=500.0, step=0.1, key='pred_PRECTOT')
        elif feature == 'RH2M':
            input_values[feature] = st.number_input("Relative Humidity at 2m (%)", min_value=0.0, max_value=100.0, step=0.1, key='pred_RH2M')
        elif feature == 'WS10M':
            input_values[feature] = st.number_input("Wind Speed at 10m (m/s)", min_value=0.0, max_value=50.0, step=0.1, key='pred_WS10M')
        # Add more features from your X if they exist:
        elif feature == 'T2M_MAX':
            input_values[feature] = st.number_input("Max Temp at 2m (°C)", min_value=-50.0, max_value=50.0, step=0.1, key='pred_T2M_MAX')
        elif feature == 'T2M_MIN':
            input_values[feature] = st.number_input("Min Temp at 2m (°C)", min_value=-50.0, max_value=50.0, step=0.1, key='pred_T2M_MIN')
        elif feature == 'WS50M':
            input_values[feature] = st.number_input("Wind Speed at 50m (m/s)", min_value=0.0, max_value=100.0, step=0.1, key='pred_WS50M')
        elif feature == 'LAT':
            input_values[feature] = st.number_input("Latitude", min_value=-90.0, max_value=90.0, step=0.01, key='pred_LAT')
        elif feature == 'LON':
            input_values[feature] = st.number_input("Longitude", min_value=-180.0, max_value=180.0, step=0.01, key='pred_LON')
        elif feature == 'DISTRICT':
            # This assumes 'DISTRICT' was label encoded. If not, you need to handle it.
            # You'd need access to the original LabelEncoder for consistency.
            # A simpler approach might be to use one-hot encoding during training if 'DISTRICT' is truly categorical.
            # For now, if you're not using DISTRICT, you can remove it or map it.
            # Let's assume you've decided to handle it as an input, but need to map it if label encoded.
            # If 'DISTRICT' is in `model_features`, you need its encoded value.
            # This is complex without the actual LabelEncoder from training.
            # For simplicity, if DISTRICT is a feature, ensure it's either
            # 1. Manually mapped to an int you know it means
            # 2. Or, better, use OneHotEncoding during training and have inputs for each 'DISTRICT_X'
            # For now, let's use a dummy numeric input for 'DISTRICT' if it's expected as a feature
            input_values[feature] = st.number_input("District (encoded number)", min_value=0, max_value=100, value=0, key='pred_DISTRICT')
        elif feature == 'YEAR': # If YEAR was a feature
            input_values[feature] = st.number_input("Year", min_value=1900, max_value=2100, value=2023, key='pred_YEAR')
        elif feature == 'MONTH': # If MONTH was a feature
            input_values[feature] = st.number_input("Month", min_value=1, max_value=12, value=7, key='pred_MONTH')
        else:
            # Fallback for any other feature not explicitly listed
            input_values[feature] = st.number_input(f"{feature}", key=f'pred_{feature}')


    if st.button("Predict Temperature"):
        try:
            # Create DataFrame ensuring column order matches model_features
            user_input_df = pd.DataFrame([input_values])
            # Ensure the order of columns matches the order used during training
            user_input_df = user_input_df[model_features]

            st.write("Input data for prediction:")
            st.write(user_input_df)

            # Scale the input
            user_input_scaled = scaler.transform(user_input_df)

            # Make prediction
            predicted_temperature = model.predict(user_input_scaled)

            # Display result
            st.subheader("Prediction Result")
            st.success(f"Predicted Temperature: {predicted_temperature[0]:.2f} °C")

        except Exception as e:
            st.error(f"Prediction failed: {e}")
            st.write("Please check if the input features match the features the model was trained on.")
            st.write(f"Expected features (from training): {model_features}")

else:
    st.info("Please go to the 'Climate Data Model Training' page and train a model first.")