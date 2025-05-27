import streamlit as st
import pandas as pd

# --- Import your climate analysis page scripts as modules ---
from pages.climate_analysis import climate_overview, climate_eda, climate_feature_engineering, climate_model_training, climate_prediction

# --- Import your agriculture analysis page scripts as modules ---
# UPDATED IMPORTS:
from pages.agriculture_analysis import agriculture_overview, agriculture_eda, \
    agriculture_feature_engineering, agriculture_model_training, agriculture_prediction


# Configure page settings
st.set_page_config(layout="wide", page_title="Climate Change Impact Assessment and Prediction System for Nepal")

# Title and description of the app
st.title("Climate Change Impact Assessment and Prediction System for Nepal")
st.markdown("""
    This application analyzes climate and agricultural data to assess the impact of climate change in Nepal.
    It features climate and agriculture data analysis, model training, prediction, and visualization tools.
    Use the sidebar to navigate through different sections of the app.
""")

# --- Initialize session state variables ---
# Initialize all necessary session state keys at the very beginning of app.py

# For Climate Analysis
if 'climate_raw_data' not in st.session_state:
    st.session_state.climate_raw_data = None
if 'engineered_climate_data' not in st.session_state: # Stores the DataFrame after FE
    st.session_state.engineered_climate_data = None
if 'climate_model' not in st.session_state: # Stores the trained model object
    st.session_state.climate_model = None
if 'climate_scaler' not in st.session_state: # Stores the fitted scaler object
    st.session_state.climate_scaler = None
if 'climate_model_features' not in st.session_state: # Stores the list of feature column names
    st.session_state.climate_model_features = None

# For Agriculture Analysis (add similar initializations for agri data)
if 'agriculture_raw_data' not in st.session_state:
    st.session_state.agriculture_raw_data = None
if 'engineered_agriculture_data' not in st.session_state:
    st.session_state.engineered_agriculture_data = None
if 'agriculture_model' not in st.session_state:
    st.session_state.agriculture_model = None
if 'agriculture_scaler' not in st.session_state:
    st.session_state.agriculture_scaler = None
if 'agriculture_model_features' not in st.session_state:
    st.session_state.agriculture_model_features = None


# Sidebar for navigation
st.sidebar.title("Navigation")
# Define the structure for pages and their corresponding functions
page_functions = {
    "Climate Analysis": {
        "Overview": climate_overview.display,
        "EDA": climate_eda.display,
        "Feature Engineering": climate_feature_engineering.display_feature_engineering,
        "Model Training": climate_model_training.display_model_training,
        "Prediction": climate_prediction.display_prediction
    },
    "Agriculture Analysis": {
        "Overview": agriculture_overview.display, # UPDATED
        "EDA": agriculture_eda.display,           # UPDATED
        "Feature Engineering": agriculture_feature_engineering.display, # UPDATED
        "Model Training": agriculture_model_training.display,           # UPDATED
        "Prediction": agriculture_prediction.display                    # UPDATED
    }
}

# Display the appropriate page based on the selection
selected_section = st.sidebar.radio("Select Section", list(page_functions.keys()))
selected_page_name = st.sidebar.radio("Select Page", list(page_functions[selected_section].keys()))

# --- Call the selected page's display function ---
page_function = page_functions[selected_section][selected_page_name]
page_function() # Call the function that runs the page's content