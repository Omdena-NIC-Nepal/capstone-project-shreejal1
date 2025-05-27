import streamlit as st
from sklearn.preprocessing import LabelEncoder
from scripts.model_train import train_model  # Ensure this function is properly imported
from scripts.visualization import plot_actual_vs_predicted, plot_residuals  # Ensure these functions are available
import pandas as pd

def display():
    st.write("""
        # Model Training - Agriculture & Rural Development Data
    """)

    # Load the agricultural data if not already in session state
    if "agriculture_df" not in st.session_state:
        st.warning("❌ Data not loaded. Please go to the overview and load the data again.")
        return  # Exit function if data isn't loaded

    # If the data is loaded into session state
    agriculture_df = st.session_state.agriculture_df

    # Since the country is always Nepal, no need to encode 'Country Name'
    if 'Country Name' in agriculture_df.columns:
        agriculture_df = agriculture_df.drop(columns=['Country Name'])  # Drop 'Country Name' from features
        st.success("Dropped 'Country Name' column as it is constant (Nepal).")

    # Handle categorical data: Label encode 'Indicator Name' if present
    if 'Indicator Name' in agriculture_df.columns:
        label_encoder = LabelEncoder()
        agriculture_df['Indicator Name'] = label_encoder.fit_transform(agriculture_df['Indicator Name'])
        st.success("Encoded 'Indicator Name' column into numerical values.")
    else:
        st.warning("'Indicator Name' column not found in the data.")

    # Define feature columns and target column
    feature_cols = ['Year', 'Indicator Name']  # Country Name is dropped
    target_col = ['Value']  # Target is the 'Value' of the indicator

    # Check if all the necessary columns are in the DataFrame
    missing_columns = [col for col in feature_cols + target_col if col not in agriculture_df.columns]
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
                model, metrics = train_model(agriculture_df, model_choice, split, feature_cols, target_col)
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
