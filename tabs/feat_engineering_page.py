import streamlit as st

def display():
    st.header("Feature Engineering")

    selected_type = st.session_state.get("selected_type", "Climate")

    st.info(f"🔍 Feature Engineering for: **{selected_type}** data")

    if selected_type == "Climate":
        from pages.climate_analysis import climate_feature_engineering          
        climate_feature_engineering.display()

    elif selected_type == "Agriculture":
        from pages.agriculture_analysis import agri_feature_engineering
        agri_feature_engineering.display()
