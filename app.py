import streamlit as st

# Page configuration
st.set_page_config(page_title="Nepal - AgriClimate Dashboard", layout="wide")

# Title
st.header("🌍 Climate and Agriculture Analysis Dashboard - Nepal - Omdena Batch II Capstone Project")

# Sidebar Navigation
selected = st.sidebar.radio(
    "Select Tab",
    [
        "About the project",
        "Overview",
        "Exploratory Analysis",
        "Feature Engineering",
        "Model Train and Evaluation",
        "Prediction",
        "NLP"
    ],
    index=1
)

# Tab Handling
if selected == "About the project":
    from tabs import about_page
    about_page.display()

elif selected == "Overview":
    from tabs import overview_page
    overview_page.display()

elif selected == "Exploratory Analysis":
    from tabs import eda_page
    eda_page.display()

elif selected == "Feature Engineering":
    from tabs import feat_engineering_page
    feat_engineering_page.display()

elif selected == "Model Train and Evaluation":
    from tabs import modeltrain_page
    modeltrain_page.display()

elif selected == "Prediction":
    from tabs import prediction_page
    prediction_page.display()

elif selected == "NLP":
    from tabs import NLP
    NLP.display()

# Footer
if selected != "About the project":
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center;">
            <p>Version 1.0 | Developed by <b>Shreejal</b></p>
            <p>
                <a href="https://www.linkedin.com/in/shreejal/" target="_blank" style="text-decoration: none;">LinkedIn</a> |
                <a href="https://github.com/shreejal1" target="_blank" style="text-decoration: none;">GitHub</a>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )