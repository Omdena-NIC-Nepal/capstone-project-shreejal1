import streamlit as st
import pandas as pd
import numpy as np # For potential numeric operations
from sklearn.preprocessing import StandardScaler, LabelEncoder # Ensure these are imported for dummy data

def display():
    st.title("☀️ Climate Data Prediction")
    st.markdown("""
        This section allows you to predict climate-related outcomes (e.g., a climate impact score,
        predicted temperature, etc.) based on various climate indices and geographic information.
        Select your inputs and get an instant forecast.
    """)

    # --- 1. Check for Trained Model and Necessary Objects ---
    # Retrieve the trained climate model, scaler, feature names, and encoders from session state
    climate_model = st.session_state.get('climate_model')
    climate_scaler = st.session_state.get('climate_scaler')
    climate_model_features = st.session_state.get('climate_model_features')
    climate_district_encoder = st.session_state.get('climate_district_encoder')
    climate_drought_index_encoder = st.session_state.get('climate_drought_index_encoder')
    climate_heat_stress_encoder = st.session_state.get('climate_heat_stress_encoder')

    # Add other encoders if you have more categorical features used in training
    # e.g., climate_rainfall_category_encoder = st.session_state.get('climate_rainfall_category_encoder')

    # CRITICAL: Ensure all necessary components are NOT None AND that encoders are fitted
    # Checking for .classes_ ensures the encoder has been fitted.
    all_components_present = all([
        climate_model,
        climate_scaler,
        climate_model_features,
        climate_district_encoder is not None and hasattr(climate_district_encoder, 'classes_'),
        climate_drought_index_encoder is not None and hasattr(climate_drought_index_encoder, 'classes_'),
        climate_heat_stress_encoder is not None and hasattr(climate_heat_stress_encoder, 'classes_')
    ])

    if not all_components_present:
        st.warning("⚠️ **Climate Model not ready!** Please ensure you have completed the 'Overview', 'Feature Engineering', and 'Model Training' steps for Climate Analysis.")
        st.info("The model, scaler, and **all encoders** must be trained and saved in the previous steps to enable predictions.")
        return

    st.success("✅ Climate Model and necessary components loaded successfully for prediction!")

    # --- 2. User Input for Prediction ---
    st.subheader("📝 Input Parameters for Climate Prediction")

    col1, col2 = st.columns(2)
    with col1:
        # Assuming 'Year' is always a feature and was explicitly added in your FE/training
        latest_year = st.session_state.get('engineered_climate_data', pd.DataFrame()).get('YEAR', pd.Series(2020)).max() # Changed to 'YEAR' for consistency
        prediction_year = st.number_input(
            "📅 **Enter Year for Prediction:**",
            min_value=1960,
            max_value=2050, # Allow some future predictions
            value=int(latest_year) if pd.notna(latest_year) else 2020,
            step=1,
            key='climate_predict_year'
        )
    with col2:
        # District Name dropdown - now guaranteed to have classes_ if all_components_present is true
        district_options = sorted(climate_district_encoder.classes_)
        selected_district = st.selectbox(
            "📍 **Select District:**",
            district_options,
            key='climate_predict_district'
        )

    col3, col4 = st.columns(2)
    with col3:
        # Drought Index dropdown
        drought_index_options = sorted(climate_drought_index_encoder.classes_)
        selected_drought_index = st.selectbox(
            "💧 **Select Drought Index:**",
            drought_index_options,
            key='climate_predict_drought'
        )
    with col4:
        # Heat Stress Metric dropdown
        heat_stress_options = sorted(climate_heat_stress_encoder.classes_)
        selected_heat_stress = st.selectbox(
            "🔥 **Select Heat Stress Metric:**",
            heat_stress_options,
            key='climate_predict_heat_stress'
        )

    # --- Add inputs for other numerical features if your model uses them ---
    st.markdown("---")
    st.write("📈 **Other Numerical Climate Features (if used by the model):**")
    
    input_other_numerical_features = {}
    for feature in climate_model_features:
        # Only create inputs for features that are not 'Year' or encoded categoricals
        if feature not in ['Year', 'DISTRICT_Encoded', 'Drought_Index_Encoded', 'Heat_Stress_Encoded']:
            # You should refine these default values/ranges based on your actual data statistics
            if 'Precipitation' in feature: # Example feature name from dummy data
                input_other_numerical_features[feature] = st.number_input(
                    f"🌧️ **{feature.replace('_', ' ')} (mm):**",
                    value=150.0, # Default value, adjust based on your data's mean/median
                    min_value=0.0, max_value=1000.0, # Adjust min/max
                    step=1.0,
                    key=f'input_{feature}'
                )
            elif 'TemperatureAnomaly' in feature: # Example feature name from dummy data
                 input_other_numerical_features[feature] = st.number_input(
                    f"🌡️ **{feature.replace('_', ' ')} (°C):**",
                    value=0.0, # Default around 0
                    min_value=-5.0, max_value=5.0, # Adjust min/max
                    step=0.1,
                    key=f'input_{feature}'
                )
            # Add more `elif` blocks for other specific numerical features if needed
            # For simplicity, if it's not explicitly handled, we'll assign a default in the next step.
            else: # Generic numerical input for any other numeric feature
                input_other_numerical_features[feature] = st.number_input(
                    f"🔢 **{feature.replace('_', ' ')}:**",
                    value=0.0, # Fallback default
                    step=0.01,
                    key=f'input_{feature}'
                )


    # --- 3. Prepare Input Data for Prediction ---
    # Create a dictionary to hold all input values
    processed_input_data = {}

    # Encode categorical features
    try:
        # .transform expects a 1D array-like, so wrap in a list
        processed_input_data['DISTRICT_Encoded'] = climate_district_encoder.transform([selected_district])[0]
        processed_input_data['Drought_Index_Encoded'] = climate_drought_index_encoder.transform([selected_drought_index])[0]
        processed_input_data['Heat_Stress_Encoded'] = climate_heat_stress_encoder.transform([selected_heat_stress])[0]
    except ValueError as e:
        st.error(f"❌ An error occurred during encoding. Ensure selected values are in the training data: {e}")
        return

    # Add 'Year'
    processed_input_data['YEAR'] = prediction_year # Ensure this matches the feature name in training

    # Add other numerical features.
    # IMPORTANT: Ensure all features from `climate_model_features` are present.
    # If a feature from `climate_model_features` is not covered by user inputs,
    # you must provide a default value (e.g., mean/median from training data) or
    # derive it from historical data. Using 0.0 as a default is a basic fallback.
    for feature in climate_model_features:
        if feature not in processed_input_data: # If it's not 'Year' or encoded categoricals
            if feature in input_other_numerical_features:
                processed_input_data[feature] = input_other_numerical_features[feature]
            else:
                # Fallback for features that don't have explicit user input widgets
                st.warning(f"Feature '{feature}' was part of the model's training but no explicit input was provided. Setting to 0.0. This may affect prediction accuracy.")
                processed_input_data[feature] = 0.0 # Using 0.0 as a default

    # Convert processed_input_data to DataFrame, ensuring column order matches training
    # This is crucial for models like Linear Regression and Ridge, and good practice for others.
    input_df_for_scaling = pd.DataFrame([processed_input_data])[climate_model_features]

    st.write("Prepared input features (before scaling):")
    st.dataframe(input_df_for_scaling)

    # Scale the input data using the *fitted* scaler
    scaled_input = climate_scaler.transform(input_df_for_scaling)
    st.write("Prepared input features (after scaling):")
    st.dataframe(pd.DataFrame(scaled_input, columns=climate_model_features))

    # --- 4. Make Prediction ---
    if st.button("🚀 Get Climate Prediction"):
        with st.spinner("Calculating prediction..."):
            try:
                prediction_value = climate_model.predict(scaled_input)[0]
                st.success("✅ Prediction Generated!")
                st.write("### 📈 Predicted Climate Outcome")
                st.markdown(f"""
                    Based on your selections, the predicted climate outcome is:
                    ## <span style="color:blue; font-size: 2.5em;">{float(prediction_value):.2f} °C</span>
                """, unsafe_allow_html=True) # Assuming T2M is in Celsius
                st.balloons()
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.exception(e) # Show full traceback for debugging
    else:
        st.info("Enter the climate parameters and click 'Get Climate Prediction' to see the forecast.")

    st.divider()

    st.markdown("""
        ### Understanding Your Prediction:
        - The model uses patterns learned from historical data to make this prediction.
        - Predictions for future years or unseen combinations of features may have higher uncertainty.
        - Consider revisiting the 'Model Training' page to try different models if predictions aren't satisfactory.
    """)

# --- Standalone Testing Block (for running climate_prediction.py directly) ---
if __name__ == "__main__":
    # Simulate session state if running directly
    if 'climate_model' not in st.session_state:
        st.warning("Running Climate Prediction page directly. Simulating trained model and components.")
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        # Create dummy data that would represent the *engineered* climate data
        # Ensure this dummy data has the columns your training script expects and uses for encoding/features
        dummy_data = {
            'DATE': pd.to_datetime(['2000-01-01'] * 50 + ['2000-02-01'] * 50 + ['2000-03-01'] * 50),
            'T2M': np.random.rand(150) * 10 + 20, # Example target
            'DISTRICT': np.random.choice(['Kathmandu', 'Pokhara', 'Chitwan', 'Dhankuta'], 150), # Include more categories
            'Drought Index': np.random.choice(['Low', 'Moderate', 'Severe', 'Extreme Drought'], 150),
            'Heat Stress Metric': np.random.choice(['Mild', 'Moderate Heat', 'Severe', 'Extreme Heat'], 150),
            'Precipitation': np.random.rand(150) * 100, # Range like 0-100mm
            'TemperatureAnomaly': np.random.randn(150) * 2, # Range like -4 to +4
            'YEAR': pd.to_datetime(['2000-01-01'] * 50 + ['2000-02-01'] * 50 + ['2000-03-01'] * 50).dt.year,
        }
        df_dummy = pd.DataFrame(dummy_data)

        # Fit dummy encoders
        dummy_district_encoder = LabelEncoder()
        df_dummy['DISTRICT_Encoded'] = dummy_district_encoder.fit_transform(df_dummy['DISTRICT'])

        dummy_drought_index_encoder = LabelEncoder()
        df_dummy['Drought_Index_Encoded'] = dummy_drought_index_encoder.fit_transform(df_dummy['Drought Index'])

        dummy_heat_stress_encoder = LabelEncoder()
        df_dummy['Heat_Stress_Encoded'] = dummy_heat_stress_encoder.fit_transform(df_dummy['Heat Stress Metric'])

        # Define features used in training (must match your actual training features)
        # Ensure these exactly match the `feature_cols` you expect after FE in training.py
        dummy_features = [
            'YEAR', # Must be 'YEAR' if that's what your training script uses
            'Precipitation',
            'TemperatureAnomaly',
            'DISTRICT_Encoded',
            'Drought_Index_Encoded',
            'Heat_Stress_Encoded'
        ]
        dummy_target = 'T2M' # Must be 'T2M' as per your training script

        # Filter dummy_data to only include features actually used and drop NaNs
        df_dummy_for_training = df_dummy[dummy_features + [dummy_target]].dropna()

        # Fit dummy scaler
        dummy_scaler = StandardScaler()
        dummy_scaler.fit(df_dummy_for_training[dummy_features])

        # Fit dummy model
        dummy_model = RandomForestRegressor(n_estimators=10, random_state=42)
        dummy_model.fit(dummy_scaler.transform(df_dummy_for_training[dummy_features]), df_dummy_for_training[dummy_target])

        # Store in session state
        st.session_state.climate_model = dummy_model
        st.session_state.climate_scaler = dummy_scaler
        st.session_state.climate_model_features = dummy_features # Crucial: the list of feature names
        st.session_state.climate_district_encoder = dummy_district_encoder
        st.session_state.climate_drought_index_encoder = dummy_drought_index_encoder
        st.session_state.climate_heat_stress_encoder = dummy_heat_stress_encoder
        # Also simulate some engineered data being present for the year picker
        st.session_state.engineered_climate_data = df_dummy

        st.info("Dummy climate model and components loaded for direct testing of Prediction page.")

    display()