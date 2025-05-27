import streamlit as st

def display():
    st.header("Model Prediction")

    selected_type = st.session_state.get("selected_type", "Climate")

    st.info(f"Model Prediction for: **{selected_type}** data")

    if selected_type == "Climate":
        from pages.climate_analysis import climate_prediction        
        climate_prediction.display()

    elif selected_type == "Agriculture":
        from pages.agriculture_analysis import agri_prediction
        agri_prediction.display()
