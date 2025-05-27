import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler # Keep LabelEncoder if you plan to use it for 'DISTRICT' in other contexts
import os # Import os for robust path handling

# The function name MUST match what app.py expects!
def display_feature_engineering():
    st.write("""
        # Climate Data Feature Engineering
        In this section, we will generate new features and preprocess the climate data to prepare it for modeling.
        Feature engineering is a crucial step in building machine learning models to improve performance.
    """)

    # Check if the cleaned dataset exists in session state
    # Ensure 'climate_raw_data' is used here, as that's what the overview page stores
    if "climate_raw_data" not in st.session_state or st.session_state.climate_raw_data is None:
        st.error("❌ Data not loaded. Please go to the 'Climate Analysis' -> 'Overview' page and load the data first.")
        return

    # Load data from session state
    climate_data_raw = st.session_state.climate_raw_data.copy() # Work on a copy to avoid modifying the original
    st.success("✅ Raw climate data loaded from session state for feature engineering.")

    st.subheader("Initial Data Overview (from Session State)")
    st.write(climate_data_raw.head())
    st.write(f"Initial shape: {climate_data_raw.shape}")
    st.write(f"Columns: {climate_data_raw.columns.tolist()}")

    st.subheader("Feature Engineering Steps")

    # Ensure necessary columns for feature engineering exist and are numeric
    numeric_cols_for_fe = ['T2M', 'PRECTOT', 'RH2M', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'T2M_MAX', 'T2M_MIN']
    for col in numeric_cols_for_fe:
        if col in climate_data_raw.columns:
            climate_data_raw[col] = pd.to_numeric(climate_data_raw[col], errors='coerce')
        else:
            st.warning(f"Warning: Column '{col}' not found in the data, skipping related feature engineering.")

    # Drop rows with NaN values introduced by coercion or already present before feature creation
    initial_rows_before_fe_dropna = climate_data_raw.shape[0]
    climate_data_raw.dropna(inplace=True)
    rows_after_fe_dropna = climate_data_raw.shape[0]
    if initial_rows_before_fe_dropna > rows_after_fe_dropna:
        st.info(f"Dropped {initial_rows_before_fe_dropna - rows_after_fe_dropna} rows with NaNs before feature creation.")
    else:
        st.info("No NaNs found before feature creation (good).")

    # --- Feature Engineering ---

    # 1. Feature 1: Temperature Anomalies (difference from overall mean temperature)
    if 'T2M' in climate_data_raw.columns:
        climate_data_raw['T2M_ANOMALY'] = climate_data_raw['T2M'] - climate_data_raw['T2M'].mean()
        st.success("Created Feature: Temperature Anomalies (T2M_ANOMALY)")
    else:
        st.warning("Cannot create T2M_ANOMALY: 'T2M' column missing.")

    # 2. Feature 2: Rolling average of temperature (over a 3-month window)
    if 'T2M' in climate_data_raw.columns:
        if 'DATE' in climate_data_raw.columns:
            try:
                # Ensure DATE is datetime and sort for correct rolling per district
                climate_data_raw['DATE'] = pd.to_datetime(climate_data_raw['DATE'])
                climate_data_raw.sort_values(by=['DISTRICT', 'DATE'], inplace=True)
                climate_data_raw['T2M_ROLLING_MEAN'] = climate_data_raw.groupby('DISTRICT')['T2M'].transform(
                    lambda x: x.rolling(window=3, min_periods=1).mean()
                )
                st.success("Created Feature: Rolling 3-month average of Temperature (T2M_ROLLING_MEAN)")
            except Exception as e:
                st.warning(f"Could not create T2M_ROLLING_MEAN (date/sort issue): {e}. Applying simple rolling mean.")
                climate_data_raw['T2M_ROLLING_MEAN'] = climate_data_raw['T2M'].rolling(window=3, min_periods=1).mean()
        else:
            st.warning("No 'DATE' column for accurate time-based rolling mean. Applying simple rolling mean.")
            climate_data_raw['T2M_ROLLING_MEAN'] = climate_data_raw['T2M'].rolling(window=3, min_periods=1).mean()
        st.success("Created Feature: Rolling average of Temperature (T2M_ROLLING_MEAN).")
    else:
        st.warning("Cannot create T2M_ROLLING_MEAN: 'T2M' column missing.")

    # 3. Feature 3: Monthly Precipitation Anomalies
    if 'PRECTOT' in climate_data_raw.columns and 'MONTH' in climate_data_raw.columns:
        climate_data_raw['PRECTOT_ANOMALY'] = climate_data_raw['PRECTOT'] - climate_data_raw.groupby('MONTH')['PRECTOT'].transform('mean')
        st.success("Created Feature: Monthly Precipitation Anomalies (PRECTOT_ANOMALY)")
    else:
        st.warning("Cannot create PRECTOT_ANOMALY: 'PRECTOT' or 'MONTH' column missing.")

    # 4. Feature 4: Wind Speed Range (difference between max and min wind speed)
    if 'WS10M_MAX' in climate_data_raw.columns and 'WS10M_MIN' in climate_data_raw.columns:
        climate_data_raw['WS10M_RANGE'] = climate_data_raw['WS10M_MAX'] - climate_data_raw['WS10M_MIN']
        st.success("Created Feature: Wind Speed Range (WS10M_RANGE)")
    else:
        st.warning("Cannot create WS10M_RANGE: 'WS10M_MAX' or 'WS10M_MIN' column missing.")

    # 5. Feature 5: Daily Temperature Range (using T2M_MAX and T2M_MIN)
    if 'T2M_MAX' in climate_data_raw.columns and 'T2M_MIN' in climate_data_raw.columns:
        climate_data_raw['DAILY_TEMP_RANGE'] = climate_data_raw['T2M_MAX'] - climate_data_raw['T2M_MIN']
        st.success("Created Feature: Daily Temperature Range (DAILY_TEMP_RANGE)")
    else:
        st.warning("Cannot create DAILY_TEMP_RANGE: 'T2M_MAX' or 'T2M_MIN' column missing.")


    # Handle any NaN values introduced by rolling averages or other features
    initial_rows_after_fe_fillna = climate_data_raw.shape[0]
    climate_data_raw.fillna(method='ffill', inplace=True)
    climate_data_raw.fillna(method='bfill', inplace=True) # Catch leading NaNs
    rows_after_final_fillna = climate_data_raw.shape[0]

    if initial_rows_after_fe_fillna == rows_after_final_fillna and climate_data_raw.isnull().sum().sum() == 0:
        st.success("All missing values (if any were created by features) have been handled.")
    else:
        st.warning("Some NaNs might still exist after final fillna. Review data.")
        st.write(climate_data_raw.isnull().sum()[climate_data_raw.isnull().sum() > 0])


    # 6. Encode categorical variables (e.g., DISTRICT)
    if 'DISTRICT' in climate_data_raw.columns:
        # Using One-Hot Encoding which is generally preferred over Label Encoding for nominal categories
        climate_data_engineered = pd.get_dummies(climate_data_raw, columns=['DISTRICT'], prefix='DISTRICT', drop_first=True)
        st.success("Encoded 'DISTRICT' column using One-Hot Encoding.")
    else:
        st.warning("'DISTRICT' column not found for encoding. Skipping.")
        climate_data_engineered = climate_data_raw.copy() # Just copy if no encoding needed


    # Prepare for scaling: Select numerical columns that are features
    # Exclude 'DATE' (if datetime), 'YEAR', 'MONTH', and the target 'T2M' from scaling
    columns_to_scale = climate_data_engineered.select_dtypes(include=np.number).columns.tolist()

    # Remove target and time-related features if they are numeric and should not be scaled
    if 'T2M' in columns_to_scale:
        columns_to_scale.remove('T2M')
    if 'YEAR' in columns_to_scale:
        columns_to_scale.remove('YEAR')
    if 'MONTH' in columns_to_scale:
        columns_to_scale.remove('MONTH')
    # If 'DATE' became numeric (unlikely but good to check)
    if 'DATE' in columns_to_scale:
        columns_to_scale.remove('DATE')

    st.info(f"Columns to be scaled: {columns_to_scale}")

    # 7. Normalize continuous features
    if columns_to_scale:
        scaler = StandardScaler()
        # Fit and transform only the selected columns
        climate_data_engineered[columns_to_scale] = scaler.fit_transform(
            climate_data_engineered[columns_to_scale]
        )
        st.success(f"Normalized features: {', '.join(columns_to_scale)}")
    else:
        st.warning("No numeric columns found for scaling.")


    # --- Save the feature-engineered data to session state ---
    # This is the key change from saving to CSV
    st.session_state.engineered_climate_data = climate_data_engineered
    st.success("✅ Feature-engineered data saved to session state ('engineered_climate_data').")

    # Display the feature-engineered data
    st.subheader("Feature-Engineered Data Overview")
    st.write(climate_data_engineered.head())
    st.write(f"Final shape: {climate_data_engineered.shape}")
    st.write(f"Final columns: {climate_data_engineered.columns.tolist()}")

    st.markdown("""
        The feature engineering process is complete. The transformed data is now
        available in Streamlit's session state for model training.
    """)

# This block is only executed when the script is run directly (for testing purposes), not when imported by app.py
if __name__ == "__main__":
    # Simulate session state for local testing if climate_raw_data is not set
    if 'climate_raw_data' not in st.session_state:
        # Load directly if testing this page in isolation
        try:
            # Assuming data is in 'your_project/data/climate_data/'
            script_dir = os.path.dirname(os.path.abspath(__file__))
            climate_data_path_for_test = os.path.join(script_dir, "..", "..", "data", "climate_data", "climate_data_nepal_district_wise_monthly.csv")
            st.session_state.climate_raw_data = pd.read_csv(climate_data_path_for_test).copy()
            st.info("Data loaded directly for testing Feature Engineering page.")
        except FileNotFoundError:
            st.error(f"Could not load data for direct testing from {climate_data_path_for_test}. Please ensure the file exists.")
            st.session_state.climate_raw_data = None # Ensure it's None if not found
    display_feature_engineering()