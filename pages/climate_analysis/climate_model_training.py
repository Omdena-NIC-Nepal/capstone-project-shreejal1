import streamlit as st
from sklearn.preprocessing import LabelEncoder
from scripts.model_train import train_model
from scripts.visualization import plot_actual_vs_predicted, plot_residuals
import pandas as pd

def display():
    st.write("""
        # Model Training - Climate Data
    """)

    # Check if the cleaned dataset exists
    if "cleaned_climatedf" in st.session_state:
        cleaned_climatedf = st.session_state.cleaned_climatedf
    else:
        st.error("❌ Data not loaded. Please go to the overview and load the data again.")
        return

    # Handle categorical data: Label encode 'DISTRICT' if present
    if 'DISTRICT' in cleaned_climatedf.columns:
        label_encoder = LabelEncoder()
        cleaned_climatedf['DISTRICT'] = label_encoder.fit_transform(cleaned_climatedf['DISTRICT'])
        st.success("Encoded 'DISTRICT' column into numerical values.")
    else:
        st.warning("'DISTRICT' column not found in the data.")

    # Define feature columns and target columns
    feature_cols = ['YEAR', 'MONTH', 'DISTRICT', 'LAT', 'LON']
    target_cols = [
        'PRECTOT',    # Total Precipitation
        'RH2M',       # Relative Humidity at 2m
        'T2M',        # Air Temperature at 2m
        'T2MWET',     # Wet Bulb Temperature at 2m
        'T2M_MAX',    # Maximum Air Temperature at 2m
        'T2M_MIN',    # Minimum Air Temperature at 2m
        'WS10M',      # Wind Speed at 10m
        'WS10M_MAX',  # Maximum Wind Speed at 10m
        'WS10M_MIN',  # Minimum Wind Speed at 10m
        'WS50M',      # Wind Speed at 50m
        'WS50M_MAX',  # Maximum Wind Speed at 50m
        'WS50M_MIN'   # Minimum Wind Speed at 50m
    ]

    # Check if all the necessary columns are in the DataFrame
    missing_columns = [col for col in feature_cols + target_cols if col not in cleaned_climatedf.columns]
    if missing_columns:
        st.error(f"❌ Missing columns: {', '.join(missing_columns)}")
        return

    # Initialize session state for trained model if not already available
    if 'trained_model' not in st.session_state:
        st.session_state.trained_model = None

    st.subheader("⚙️ Model Training")

    # Model selection dropdown
    model_choice = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "Linear Regression", "Ridge Regression"])

    # Train/Test split slider
    split = st.slider("Select Train/Test Split %", min_value=50, max_value=95, value=80, step=5) / 100

    if st.button("Train Model"):
        with st.spinner('🚀 Training model. Please wait...'):
            try:
                # Train the model using the selected parameters
                model, metrics = train_model(cleaned_climatedf, model_choice, split, feature_cols, target_cols)
            except Exception as e:
                st.error(f"An error occurred while training the model: {e}")
                return

        st.progress(100, text="✅ Training Complete!")
        st.success(f"✅ Model Trained Successfully!")

        # Display Model Evaluation Metrics
        st.subheader("📊 Model Evaluation Metrics")

        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**R² Score:** {metrics['R2 Score']:.4f}")
            st.write(f"**MSE:** {metrics['MSE']:.4f}")
        with col2:
            st.write(f"**RMSE:** {metrics['RMSE']:.4f}")
            st.write(f"**MAE:** {metrics['MAE']:.4f}")

        # Save the trained model in session state
        st.session_state.trained_model = model

        # Extract true and predicted values for plotting
        y_true = metrics['y_true']
        y_pred = metrics['y_pred']

        # Actual vs Predicted
        st.subheader("Actual vs Predicted Values")
        fig1 = plot_actual_vs_predicted(y_true, y_pred)
        st.pyplot(fig1)

        # Residual Analysis
        st.subheader("Residual Analysis")
        fig2 = plot_residuals(y_true, y_pred)
        st.pyplot(fig2)

        st.divider()

    else:
        st.info("Make sure to select a model and click 'Train Model' to begin training.")
