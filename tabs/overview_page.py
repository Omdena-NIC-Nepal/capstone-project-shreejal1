import streamlit as st

def display():
    # Display a warning message for first-time users
    st.warning("🚨 First time here? Visit the **About the project** tab to understand the background, data sources, and how to use this dashboard effectively.")

    st.header("🌍 Climate & Agriculture Impact Dashboard")

    # Updated project overview
    st.markdown("""
    Welcome to the **Climate and Agriculture Impact Dashboard for Nepal** 🇳🇵  
    Developed as part of the **Omdena Batch II Capstone Project**, this dashboard is designed to help you explore, analyze, and predict environmental and agricultural changes using real-world data.

    ### 🔍 What You Can Explore:
    - **🌦️ Climate Analysis**  
      Analyze weather patterns across Nepal with monthly climate records from 1981 onwards — including temperature, rainfall, and humidity by district.

    - **🌾 Agriculture Analysis**  
      Understand how climate trends impact agriculture by exploring crop data, rural development indicators, and more.

    - **🧠 Machine Learning Predictions**  
      Access predictive models trained on climate and agriculture data to forecast future outcomes and risks.

    ### 📘 How to Use:
    1. Choose a data category below to start your analysis.
    2. Use the sidebar tabs to explore data (EDA), engineer features, train models, and generate predictions.
    """)

    # Data category selector
    if "selected_type" not in st.session_state:
        st.session_state["selected_type"] = "Climate"  # Default option

    option = st.selectbox(
        "Choose a data category to explore:",
        ["Climate", "Agriculture"],
        index=["Climate", "Agriculture"].index(st.session_state["selected_type"])
    )

    if st.session_state["selected_type"] != option:
        st.session_state["selected_type"] = option

    # Load appropriate overview
    if option == "Climate":
        from pages.climate_analysis import climate_overview
        climate_overview.display()

    elif option == "Agriculture":
        from pages.agriculture_analysis import agri_overview
        agri_overview.display()
