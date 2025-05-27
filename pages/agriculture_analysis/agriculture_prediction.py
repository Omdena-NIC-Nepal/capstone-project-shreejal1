import streamlit as st
import pandas as pd
import numpy as np # For potential numeric operations

def display():
    st.title("🔮 Agricultural Indicator Prediction")
    st.markdown("""
        This section allows you to get predictions for agricultural indicator values using the model
        you trained on the 'Model Training' page. Input the desired year and indicator, and the model
        will provide a forecast.
    """)

    # --- 1. Check for Trained Model and Necessary Objects ---
    # Retrieve the trained model, scaler, feature names, and label encoder from session state
    model = st.session_state.get('agriculture_model')
    scaler = st.session_state.get('agriculture_scaler')
    model_features = st.session_state.get('agriculture_model_features')
    label_encoder = st.session_state.get('agriculture_label_encoder')

    # Ensure all necessary components are available
    if model is None or scaler is None or model_features is None or label_encoder is None:
        st.warning("⚠️ **Model not ready!** Please ensure you have completed the 'Overview', 'Feature Engineering', and 'Model Training' steps for Agriculture Analysis.")
        st.info("The model, scaler, and feature set must be trained and saved in the previous steps to enable predictions.")
        return

    st.success("✅ Model and necessary components loaded successfully for prediction!")

    # --- 2. User Input for Prediction ---
    st.subheader("📝 Input for Prediction")

    # Get indicator options from the label encoder's classes
    # This ensures we only allow indicators the model was trained on
    indicator_options = sorted(label_encoder.classes_)

    col1, col2 = st.columns(2)
    with col1:
        # Get the latest year from the engineered data if available, otherwise default
        latest_year = st.session_state.get('engineered_agriculture_data', pd.DataFrame()).get('Year', pd.Series(2020)).max()
        prediction_year = st.number_input(
            "📅 **Enter Year for Prediction:**",
            min_value=1960,
            max_value=2050, # Allow some future predictions
            value=int(latest_year) if pd.notna(latest_year) else 2020,
            step=1,
            key='agri_predict_year'
        )
    with col2:
        selected_indicator_name = st.selectbox(
            "📊 **Select Agricultural Indicator:**",
            indicator_options,
            key='agri_predict_indicator'
        )

    # --- 3. Prepare Input Data for Prediction ---
    # Create a DataFrame for the input, similar to how training data was structured
    input_data = pd.DataFrame({
        'Year': [prediction_year],
        # The encoded indicator will be handled right before scaling
    })

    st.subheader("Feature Values for Prediction")

    # IMPORTANT: The input features for prediction must match the features used in training.
    # We need to manually add other engineered features if they were used for training.
    # For now, let's assume 'Year' and 'Indicator Name_Encoded' are the primary features.
    # If other features were added (e.g., '3-Year Rolling Mean', 'Yearly Z-Score'),
    # you'll need to calculate them for the *single prediction point*.
    # This is complex and usually requires a more sophisticated forecasting approach (e.g., time series models).
    # For simplicity, we'll assume the model primarily relies on Year and Indicator.
    # If your model_features contain more, you'd need to prompt for them or derive them.

    # If the model trained on specific engineered features, we need them here.
    # This is a critical point: Simple regression models can't derive past values or rolling means from a single input year.
    # For a robust prediction, you'd need to:
    # 1. Retrieve the full historical series for the selected indicator.
    # 2. Re-calculate rolling means, Z-scores, etc., up to the `prediction_year`.
    # 3. Use the latest calculated values for these features for the `prediction_year`.
    # For this demonstration, we'll simplify and only include `Year` and `Indicator Name_Encoded`.
    # If your `model_features` array contains more complex features, you'll need to adapt this section.

    # Let's assume for simplicity, the trained model uses 'Year' and 'Indicator Name_Encoded'
    # as its direct inputs after scaling. If your `model_features` is more complex (e.g., includes
    # '3-Year Rolling Mean', 'Yearly Z-Score'), you would need to calculate those based on the
    # historical data up to the prediction_year. This is a common challenge in time-series prediction.

    # For now, we will dynamically build the input based on `model_features` for robustness,
    # and fill missing engineered features with plausible defaults (e.g., 0 or mean/median).
    # This is a simplification; for a real production system, handling this would be more involved.

    processed_input_data = {}
    for feature in model_features:
        if feature == 'Year':
            processed_input_data['Year'] = prediction_year
        elif feature == 'Indicator Name_Encoded':
            try:
                processed_input_data['Indicator Name_Encoded'] = label_encoder.transform([selected_indicator_name])[0]
            except ValueError:
                st.error(f"❌ The selected indicator '{selected_indicator_name}' was not recognized by the model's encoder. Please select an indicator from the training data.")
                return
        else:
            # If the model uses other engineered features like 'Yearly Z-Score', '3-Year Rolling Mean', etc.
            # You might need to add a way to either:
            # a) Prompt the user for these values (less user-friendly)
            # b) Have a backend logic to compute them based on historical data up to the prediction year
            # For demonstration, we'll set them to a default (e.g., 0 or mean if the scaler handles it).
            # This is a known simplification for direct prediction pages.
            st.warning(f"Feature '{feature}' was part of training but cannot be dynamically calculated for a single prediction point. Setting to 0. This might affect accuracy.")
            processed_input_data[feature] = 0 # Or some other default/imputation strategy

    # Convert processed_input_data to DataFrame
    input_df_for_scaling = pd.DataFrame([processed_input_data])

    # Ensure the order of columns in input_df_for_scaling matches model_features
    # This is CRUCIAL for correct prediction
    input_df_for_scaling = input_df_for_scaling[model_features]
    st.write("Prepared input features (before scaling):")
    st.dataframe(input_df_for_scaling)

    # Scale the input data using the *fitted* scaler
    scaled_input = scaler.transform(input_df_for_scaling)
    st.write("Prepared input features (after scaling):")
    st.dataframe(pd.DataFrame(scaled_input, columns=model_features))

    # --- 4. Make Prediction ---
    if st.button("🚀 Get Prediction"):
        with st.spinner("Calculating prediction..."):
            try:
                prediction_value = model.predict(scaled_input)[0]
                st.success("✅ Prediction Generated!")
                st.write("### 📈 Predicted Agricultural Indicator Value")
                st.markdown(f"""
                    For **'{selected_indicator_name}'** in **{prediction_year}**,
                    the predicted value is:
                    ## <span style="color:green; font-size: 2.5em;">{float(prediction_value):.2f}</span>
                """, unsafe_allow_html=True)
                st.balloons()
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.exception(e) # Show full traceback for debugging
    else:
        st.info("Enter the year and indicator, then click 'Get Prediction' to see the forecast.")

    st.divider()

    st.markdown("""
        ### Next Steps:
        - Experiment with different years and indicators to see how the model responds.
        - The accuracy of predictions for future years depends heavily on the model's training data
          and the nature of the agricultural trends.
    """)

# --- Standalone Testing Block ---
if __name__ == "__main__":
    # Simulate session state if running directly
    if 'agriculture_model' not in st.session_state:
        st.warning("Running Prediction page directly. Simulating trained model and components.")
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler, LabelEncoder

        # Create a dummy model
        dummy_model = RandomForestRegressor(n_estimators=10, random_state=42)
        # Create dummy data for fitting scaler and encoder
        dummy_data = {
            'Year': np.arange(1990, 2020).tolist() * 2,
            'Indicator Name': ['Arable land (% of land area)'] * 30 + ['Cereal yield (kg per hectare)'] * 30,
            'Value': np.random.rand(60) * 1000,
            'Yearly Z-Score': np.random.randn(60),
            'Value % Change': np.random.rand(60) * 20 - 10,
            '3-Year Rolling Mean': np.random.rand(60) * 900,
            'High Growth Flag': np.random.randint(0, 2, 60),
            'Normalized Value': np.random.rand(60)
        }
        df_dummy = pd.DataFrame(dummy_data)
        # Encode indicator
        dummy_label_encoder = LabelEncoder()
        df_dummy['Indicator Name_Encoded'] = dummy_label_encoder.fit_transform(df_dummy['Indicator Name'])

        # Define features used in training (should match what your training page actually uses)
        # Assuming training used 'Year', 'Yearly Z-Score', 'Value % Change', '3-Year Rolling Mean', 'Normalized Value', 'Indicator Name_Encoded'
        dummy_features = ['Year', 'Yearly Z-Score', 'Value % Change', '3-Year Rolling Mean', 'Normalized Value', 'Indicator Name_Encoded']
        dummy_target = 'Value'

        # Filter dummy_data to only include features actually used
        df_dummy_for_training = df_dummy[dummy_features + [dummy_target]].dropna()

        # Fit dummy scaler
        dummy_scaler = StandardScaler()
        dummy_scaler.fit(df_dummy_for_training[dummy_features])

        # Fit dummy model
        dummy_model.fit(dummy_scaler.transform(df_dummy_for_training[dummy_features]), df_dummy_for_training[dummy_target])

        # Store in session state
        st.session_state.agriculture_model = dummy_model
        st.session_state.agriculture_scaler = dummy_scaler
        st.session_state.agriculture_model_features = dummy_features
        st.session_state.agriculture_label_encoder = dummy_label_encoder

        st.info("Dummy model and components loaded for direct testing of Prediction page.")

    display()