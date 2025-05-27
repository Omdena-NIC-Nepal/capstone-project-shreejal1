import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def display():
    st.write("""
    ## 🌾 Agriculture Prediction
    Use the trained agriculture model to predict indicator values based on year and type.
    """)

    # ✅ Check if the model is trained
    if 'trained_model' not in st.session_state or st.session_state.trained_model is None:
        st.warning("⚠️ Please train the agriculture model first on the Model Train page.")
        return

    model = st.session_state.trained_model


    # 🧠 Input: Year + Indicator
    year = st.number_input("📅 Year", min_value=1960, max_value=2025, value=2020)

    # Use the original dataframe to fetch indicator names
    if 'agriculture_df' not in st.session_state:
        st.warning("❌ Data not found. Please load data from the Overview page.")
        return

    agriculture_df = st.session_state.agriculture_df
    indicator_options = sorted(agriculture_df['Indicator Name'].unique())
    indicator = st.selectbox("📊 Indicator Name", indicator_options)

    # Encode indicator name to match training
    label_encoder = LabelEncoder()
    label_encoder.fit(agriculture_df['Indicator Name'])
    encoded_indicator = label_encoder.transform([indicator])[0]

    input_df = pd.DataFrame({
        'Year': [year],
        'Indicator Name': [encoded_indicator]
    })

    # 🚀 Predict
    prediction = model.predict(input_df)

    # 🎉 Display result
    st.success("✅ Prediction successful!")
    st.write("### 📈 Predicted Value")
    st.write(f"📌 {float(prediction[0]):.2f}")
    st.balloons()
