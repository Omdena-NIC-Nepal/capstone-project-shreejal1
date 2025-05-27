import streamlit as st

# Title and description of the app
st.title("Climate Change Impact Assessment and Prediction System for Nepal")
st.markdown("""
    This application analyzes climate and agricultural data to assess the impact of climate change in Nepal. 
    It features climate and agriculture data analysis, model training, prediction, and visualization tools. 
    Use the sidebar to navigate through different sections of the app.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
pages = {
    "Climate Analysis": {
        "Overview": "pages/climate_analysis/climate_overview.py",
        "EDA": "pages/climate_analysis/climate_eda.py",
        "Feature Engineering": "pages/climate_analysis/climate_feature_engineering.py",
        "Model Training": "pages/climate_analysis/climate_model_training.py",
        "Prediction": "pages/climate_analysis/climate_prediction.py"
    },
    "Agriculture Analysis": {
        "Overview": "pages/agriculture_analysis/agri_overview.py",
        "EDA": "pages/agriculture_analysis/agri_eda.py",
        "Feature Engineering": "pages/agriculture_analysis/agri_feature_engineering.py",
        "Model Training": "pages/agriculture_analysis/agri_model_training.py",
        "Prediction": "pages/agriculture_analysis/agri_prediction.py"
    }
}

# Display the appropriate page based on the selection
selected_section = st.sidebar.radio("Select Section", list(pages.keys()))
selected_page = st.sidebar.radio("Select Page", list(pages[selected_section].keys()))

# Import and display the selected page
page_file = pages[selected_section][selected_page]
with open(page_file) as file:
    code = file.read()
    exec(code)
