import streamlit as st
import pandas as pd
import numpy as np

# No need to import RandomForestRegressor, StandardScaler etc. here
# as we will load them from session_state as objects

def display_prediction(): # <--- THIS MUST BE THE FUNCTION NAME
    st.title("Climate Data Prediction")
    st.markdown("""
        Use this section to make predictions based on the trained climate model.
        You can input new climate conditions, and the model will predict the 'T2M' (Temperature at 2 meters).
    """)

    # --- Check if model and scaler are available in session state ---
    if 'climate_model' not in st.session_state or st.session_state.climate_model is None:
        st.error("❌ No trained climate model found. Please go to the 'Climate Analysis' -> 'Model Training' page and train a model first.")
        return

    if 'climate_scaler' not in st.session_state or st.session_state.climate_scaler is None:
        st.error("❌ No fitted scaler found. Please train a model on the 'Model Training' page.")
        return

    if 'climate_model_features' not in st.session_state or st.session_state.climate_model_features is None:
        st.error("❌ Model feature list not found. Please train a model on the 'Model Training' page.")
        return

    model = st.session_state.climate_model
    scaler = st.session_state.climate_scaler
    model_features = st.session_state.climate_model_features

    st.success("✅ Trained model, scaler, and feature list loaded from session state.")
    st.info(f"Model Type: {type(model).__name__}")
    st.info(f"Features expected by model: {model_features}")


    st.subheader("Input New Climate Data for Prediction")

    # --- Create input fields for each feature the model expects ---
    # This part requires you to know what features your model was trained on.
    # The 'climate_model_features' in session_state should help here.
    input_data = {}
    st.write("Please enter values for the following features:")

    # Create dynamic inputs based on model_features
    # You'll need to create appropriate input widgets for each type of feature
    # For simplicity, let's assume most are numbers, but consider categories too.

    # Example: Create inputs for a few common features (adjust as per your actual features)
    # You need to manually specify min/max/default values for these inputs.
    # The order of inputs matters for consistency if you're building the DF manually.

    # Identify one-hot encoded district columns from model_features
    district_cols = [f for f in model_features if f.startswith('DISTRICT_')]
    other_numeric_features = [f for f in model_features if not f.startswith('DISTRICT_')]

    # Categorical input for District (this will drive the one-hot encoding)
    # You need to get the list of all possible districts from your original data.
    # For now, let's assume you have a way to get them or hardcode some.
    if 'climate_raw_data' in st.session_state and st.session_state.climate_raw_data is not None:
        all_districts = st.session_state.climate_raw_data['DISTRICT'].unique().tolist()
        if 'DISTRICT' in all_districts: # Remove 'DISTRICT' itself if it's there
            all_districts.remove('DISTRICT') # Should not happen if FE is correct
        all_districts.sort() # Sort for better display
    else:
        st.warning("Original climate data not found in session state to get full district list. Using a placeholder.")
        # Fallback if original data is not there
        all_districts = [col.replace('DISTRICT_', '') for col in district_cols]
        if not all_districts: # If no district cols, provide a dummy
             all_districts = ['Kathmandu', 'Lalitpur', 'Bhaktapur', 'Other']


    selected_district = st.selectbox("District", all_districts)

    # For other numeric features, use number_input
    for feature in other_numeric_features:
        # Provide reasonable min/max/default values based on your data's scale
        # This is crucial for realistic predictions and to prevent errors.
        # These are just examples, you should use actual data ranges from your EDA
        default_val = 0.0
        min_val = -100.0
        max_val = 100.0

        # Try to infer better default/min/max from training data's descriptive stats
        if 'engineered_climate_data' in st.session_state and st.session_state.engineered_climate_data is not None:
            if feature in st.session_state.engineered_climate_data.columns:
                series = st.session_state.engineered_climate_data[feature]
                default_val = float(series.mean() if series.dtype != 'object' else 0.0)
                min_val = float(series.min() if series.dtype != 'object' else -100.0)
                max_val = float(series.max() if series.dtype != 'object' else 100.0)
                # Adjust for scaled data, min/max might be around -3 to 3 after scaling
                if "ANOMALY" in feature or "RANGE" in feature or "ROLLING_MEAN" in feature:
                     # For scaled features, these are often good ranges to start
                    min_val = -3.0
                    max_val = 3.0
                    default_val = 0.0 # Mean is often 0 after scaling

        input_data[feature] = st.number_input(f"Enter value for {feature}",
                                               value=float(default_val),
                                               min_value=float(min_val),
                                               max_value=float(max_val),
                                               format="%.2f",
                                               key=f"input_{feature}")


    if st.button("Predict Temperature"):
        # Create a DataFrame for the single prediction
        # Initialize with all features set to 0, then populate
        input_df = pd.DataFrame(0.0, index=[0], columns=model_features)

        # Populate numeric features
        for feature, value in input_data.items():
            if feature in input_df.columns: # Ensure feature exists in model's expected features
                input_df[feature] = value

        # Handle one-hot encoding for the selected district
        district_one_hot_col = f'DISTRICT_{selected_district}'
        if district_one_hot_col in input_df.columns:
            input_df[district_one_hot_col] = 1.0 # Set the selected district column to 1
        else:
            st.warning(f"District column '{district_one_hot_col}' not found in model features. Check Feature Engineering.")
            st.info(f"Available district columns in model features: {district_cols}")


        # Ensure the order of columns matches the training order
        try:
            input_df = input_df[model_features] # Reorder columns to match model_features
        except KeyError as e:
            st.error(f"Error: Mismatch in input features and model's expected features. Missing: {e}")
            st.info(f"Model expects: {model_features}")
            st.info(f"Input has: {input_df.columns.tolist()}")
            return


        # Scale the input data using the *fitted scaler*
        try:
            scaled_input = scaler.transform(input_df)
            prediction = model.predict(scaled_input)[0] # Predict and get the first (only) result
            st.subheader("Prediction Result")
            st.success(f"Predicted Temperature (T2M): **{prediction:.2f} °C**")
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.info("Please check the input values and ensure the model was trained correctly.")


# This block is only executed when the script is run directly (for testing purposes)
if __name__ == "__main__":
    # Simulate session state for local testing if needed
    if 'climate_model' not in st.session_state:
        st.warning("Running Prediction page directly. No model/scaler/features in session state. Attempting to load dummy components.")
        try:
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.preprocessing import StandardScaler
            # Create a dummy model, scaler, and features for testing
            dummy_model = RandomForestRegressor(n_estimators=10, random_state=42)
            dummy_scaler = StandardScaler()

            # Define dummy features (these MUST match what your FE produces)
            # Make sure these align with the 'other_numeric_features' and 'district_cols' logic above
            dummy_features = ['T2M_ANOMALY', 'T2M_ROLLING_MEAN', 'PRECTOT_ANOMALY',
                              'WS10M_RANGE', 'DAILY_TEMP_RANGE',
                              'RH2M', 'WS10M', 'T2M_MAX', 'T2M_MIN', 'PRECTOT', # other base features
                              'YEAR', 'MONTH'] # if YEAR/MONTH were included in features to model
                              # Add dummy one-hot encoded districts
            dummy_districts = ['DISTRICT_Lalitpur', 'DISTRICT_Bhaktapur']
            dummy_features.extend(dummy_districts)

            # Fit dummy scaler on some dummy data (e.g., random data with the same columns)
            np.random.seed(42)
            dummy_X = pd.DataFrame(np.random.rand(50, len(dummy_features)), columns=dummy_features)
            # Ensure district columns are 0 or 1
            for col in dummy_districts:
                dummy_X[col] = np.random.randint(0, 2, 50)

            dummy_scaler.fit(dummy_X)
            dummy_model.fit(dummy_X, np.random.rand(50)) # Fit dummy model on dummy data

            st.session_state.climate_model = dummy_model
            st.session_state.climate_scaler = dummy_scaler
            st.session_state.climate_model_features = dummy_features
            st.session_state.climate_raw_data = pd.DataFrame({'DISTRICT': ['Kathmandu', 'Lalitpur']}) # For district list
            st.info("Dummy model, scaler, and features loaded for direct testing.")
        except Exception as e:
            st.error(f"Could not create dummy components for direct testing: {e}")
            st.session_state.climate_model = None # Ensure it's None if creation failed
            st.session_state.climate_scaler = None
            st.session_state.climate_model_features = None

    display_prediction()