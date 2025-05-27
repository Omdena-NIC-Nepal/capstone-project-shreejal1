import streamlit as st

def display():
    st.header("Exploratory Data Analysis")

    selected_type = st.session_state.get("selected_type", "Climate Analysis")

    st.info(f"🔍 Exploratory analysis for: **{selected_type}** data")

    if selected_type == "Climate":
        from pages.climate_analysis import climate_eda
        climate_eda.display()

    elif selected_type == "Agriculture":
        from pages.agriculture_analysis import agri_eda
        agri_eda.display()
