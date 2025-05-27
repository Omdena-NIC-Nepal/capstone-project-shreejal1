import streamlit as st

def display():
    st.header("Model Training")

    selected_type = st.session_state.get("selected_type", "Climate")

    st.info(f"Model Training for: **{selected_type}** data")

    if selected_type == "Climate":
        from pages.climate_analysis import climate_model_training         
        climate_model_training.display()

    elif selected_type == "Agriculture":
        from pages.agriculture_analysis import agri_model_training
        agri_model_training.display()