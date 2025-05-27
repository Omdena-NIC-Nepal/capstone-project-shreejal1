import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Title and description for the feature engineering page
st.title("Climate Data Feature Engineering")
st.markdown("""
    In this section, we will generate new features and preprocess the climate data to prepare it for modeling. 
    Feature engineering is a crucial step in building machine learning models to improve performance.
""")

# Load the climate data
climate_data_path = "data/climate_data/climate_data_nepal_district_wise_monthly.csv"
climate_data = pd.read_csv(climate_data_path)

# Display initial data overview
st.subheader("Data Overview")
st.write(climate_data.head())

# 1. Feature 1: Temperature Anomalies (difference from mean temperature)
climate_data['T2M_ANOMALY'] = climate_data['T2M'] - climate_data['T2M'].mean()

# 2. Feature 2: Rolling average of temperature (over a 3-month window)
climate_data['T2M_ROLLING_MEAN'] = climate_data['T2M'].rolling(window=3).mean()

# 3. Feature 3: Monthly Precipitation Anomalies
climate_data['PRECTOT_ANOMALY'] = climate_data['PRECTOT'] - climate_data.groupby('MONTH')['PRECTOT'].transform('mean')

# 4. Feature 4: Wind Speed Range (difference between max and min wind speed)
climate_data['WS10M_RANGE'] = climate_data['WS10M_MAX'] - climate_data['WS10M_MIN']

# Handle missing values (e.g., forward fill)
climate_data.fillna(method='ffill', inplace=True)

# 5. Encode categorical variables (e.g., DISTRICT)
climate_data = pd.get_dummies(climate_data, columns=['DISTRICT'], drop_first=True)

# 6. Normalize continuous features (e.g., T2M, PRECTOT)
scaler = StandardScaler()
climate_data[['T2M', 'PRECTOT', 'T2M_ANOMALY', 'T2M_ROLLING_MEAN', 'PRECTOT_ANOMALY', 'WS10M_RANGE']] = scaler.fit_transform(
    climate_data[['T2M', 'PRECTOT', 'T2M_ANOMALY', 'T2M_ROLLING_MEAN', 'PRECTOT_ANOMALY', 'WS10M_RANGE']]
)

# Save the feature-engineered data
engineered_data_path = "data/climate_data/engineered_climate_data.csv"
climate_data.to_csv(engineered_data_path, index=False)

# Display the feature-engineered data
st.subheader("Feature-Engineered Data")
st.write(climate_data.head())