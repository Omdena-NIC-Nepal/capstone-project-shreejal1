import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder, StandardScaler # Import StandardScaler

# --- Helper Functions (moved in-line from scripts for self-containment) ---

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs. Predicted Values"):
    """Plots actual values against predicted values."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_true, y_pred, alpha=0.6, color='skyblue')
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig

def plot_residuals(y_true, y_pred, title="Residuals Plot"):
    """Plots residuals (y_true - y_pred) against predicted values."""
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(y_pred, residuals, alpha=0.6, color='lightcoral')
    ax.hlines(0, xmin=y_pred.min(), xmax=y_pred.max(), colors='r', linestyles='--', lw=2, label='Zero Residuals')
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Residuals (Actual - Predicted)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig

def train_and_evaluate_model(X, y, model_type, test_size=0.2, random_state=42):
    """
    Trains a selected machine learning model and returns the model,
    predictions, true values, and evaluation metrics.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = None
    if model_type == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)
    elif model_type == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, random_state=random_state)
    elif model_type == "Linear Regression":
        model = LinearRegression(n_jobs=-1)
    elif model_type == "Ridge Regression":
        model = Ridge(random_state=random_state)
    else:
        st.error("Invalid model type selected.")
        return None, None, None, None, None

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {
        "R2 Score": r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "y_true": y_test,
        "y_pred": y_pred
    }

    return model, metrics, X_train.columns.tolist() # Also return feature names

# --- Main Streamlit Display Function ---

def display():
    st.title("🤖 Agricultural Development Model Training")
    st.markdown("""
        This section allows you to train various machine learning models to predict agricultural indicator values
        based on the engineered features. Select a model, adjust the train/test split, and evaluate its performance.
    """)

    # --- Load Engineered Data from Session State ---
    if 'engineered_agriculture_data' not in st.session_state or st.session_state.engineered_agriculture_data is None:
        st.error("❌ Engineered agricultural data not found. Please complete the 'Feature Engineering' step first.")
        return

    df_engineered = st.session_state.engineered_agriculture_data.copy()
    st.success("✅ Engineered agricultural data loaded for model training.")

    # --- Preprocessing for Model Training ---
    st.subheader("🧹 Data Preprocessing for Training")

    # Drop any rows with NaN values that might have been introduced during FE or were not handled.
    # Models generally don't work with NaNs.
    initial_rows_pre_modeling = df_engineered.shape[0]
    df_model = df_engineered.dropna().copy() # Drop all NaNs for modeling purposes
    if initial_rows_pre_modeling > df_model.shape[0]:
        st.warning(f"Dropped {initial_rows_pre_modeling - df_model.shape[0]} rows containing NaN values for model training.")
    else:
        st.info("No NaN values found in the engineered data for modeling.")

    if df_model.empty:
        st.error("❌ Data is empty after dropping NaNs. Cannot proceed with model training.")
        return

    # Check for 'Country Name' and 'Indicator Name' for encoding/dropping
    # Assuming 'Country Name' is constant 'Nepal' and can be dropped
    if 'Country Name' in df_model.columns:
        df_model = df_model[df_model['Country Name'] == 'Nepal'] # Filter for Nepal if not already done
        df_model.drop(columns=['Country Name', 'Country ISO3'], errors='ignore', inplace=True)
        st.info("Dropped 'Country Name' and 'Country ISO3' (assuming constant 'Nepal').")

    # Label encode 'Indicator Name'
    label_encoder = None
    if 'Indicator Name' in df_model.columns:
        label_encoder = LabelEncoder()
        df_model['Indicator Name_Encoded'] = label_encoder.fit_transform(df_model['Indicator Name'])
        # Drop the original 'Indicator Name' and 'Indicator Code' if exists after encoding
        df_model.drop(columns=['Indicator Name', 'Indicator Code'], errors='ignore', inplace=True)
        st.success("Encoded 'Indicator Name' into 'Indicator Name_Encoded'.")
    else:
        st.warning("No 'Indicator Name' column found for encoding. Ensure data is correct.")


    # Define target and potential features
    target_col = 'Value'
    if target_col not in df_model.columns:
        st.error(f"❌ Target column '{target_col}' not found in data. Cannot train model.")
        return

    # Automatically select relevant numerical features and the encoded indicator
    # Exclude target, original identifier columns, and any NaN columns after dropna
    features_to_exclude = [target_col, 'Country Name', 'Country ISO3', 'Indicator Name', 'Indicator Code']
    numerical_features = [col for col in df_model.columns if df_model[col].dtype in ['float64', 'int64'] and col not in features_to_exclude]

    # Add the encoded indicator back if it exists
    if 'Indicator Name_Encoded' in df_model.columns:
        numerical_features.append('Indicator Name_Encoded')

    feature_cols = numerical_features
    if not feature_cols:
        st.error("❌ No numerical features found for model training after preprocessing. Check your data and feature engineering.")
        return

    st.write(f"**Selected features for training:** {feature_cols}")
    st.write(f"**Target variable:** `{target_col}`")
    st.dataframe(df_model[feature_cols + [target_col]].head())


    X = df_model[feature_cols]
    y = df_model[target_col]

    # --- Feature Scaling ---
    st.subheader("📏 Feature Scaling")
    st.info("Applying StandardScaler to numerical features for optimal model performance.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns, index=X.index) # Convert back to DataFrame for consistency

    st.success("Features scaled using StandardScaler.")
    st.dataframe(X_scaled_df.head())

    # --- Model Training Section ---
    st.subheader("⚙️ Choose Model and Train")

    model_options = ["Random Forest", "Gradient Boosting", "Linear Regression", "Ridge Regression"]
    model_choice = st.selectbox("Select Regression Model", model_options, key='agri_model_choice')

    split_ratio = st.slider("Train/Test Split Ratio (%)", min_value=10, max_value=90, value=80, step=5, key='agri_split_ratio') / 100

    if st.button("🚀 Train Agricultural Model", key='train_agri_model_button'):
        if df_model.empty:
            st.error("Cannot train model: Data is empty after preprocessing.")
            return

        with st.spinner('Building and evaluating the model...'):
            try:
                # Use the scaled data for training
                trained_model, metrics, trained_feature_names = train_and_evaluate_model(
                    X_scaled_df, y, model_choice, test_size=(1-split_ratio)
                )

                if trained_model:
                    st.success(f"✅ **{model_choice} Model Training Complete!**")
                    st.balloons()

                    # --- Display Model Evaluation Metrics ---
                    st.subheader("📊 Model Performance Metrics")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(label="R² Score", value=f"{metrics['R2 Score']:.4f}")
                        st.metric(label="Mean Squared Error (MSE)", value=f"{metrics['MSE']:.4f}")
                    with col2:
                        st.metric(label="Root Mean Squared Error (RMSE)", value=f"{metrics['RMSE']:.4f}")
                        st.metric(label="Mean Absolute Error (MAE)", value=f"{metrics['MAE']:.4f}")

                    # --- Visualizations ---
                    st.subheader("📉 Model Performance Visualizations")

                    # Actual vs Predicted Plot
                    fig_ap = plot_actual_vs_predicted(metrics['y_true'], metrics['y_pred'],
                                                      title=f"Actual vs. Predicted Values ({model_choice})")
                    st.pyplot(fig_ap)
                    plt.close(fig_ap)

                    # Residuals Plot
                    fig_res = plot_residuals(metrics['y_true'], metrics['y_pred'],
                                             title=f"Residuals Plot ({model_choice})")
                    st.pyplot(fig_res)
                    plt.close(fig_res)

                    # --- Save to Session State ---
                    st.session_state.agriculture_model = trained_model
                    st.session_state.agriculture_scaler = scaler # Save the fitted scaler
                    st.session_state.agriculture_model_features = trained_feature_names # Save feature names
                    st.session_state.agriculture_label_encoder = label_encoder # Save label encoder if used

                    st.info("Trained model, scaler, and feature names saved to session state for prediction.")
                else:
                    st.error("Model training failed. Please check logs for errors.")

            except Exception as e:
                st.error(f"An unexpected error occurred during model training: {e}")
                st.exception(e) # Display full traceback for debugging

    else:
        st.info("Click the 'Train Agricultural Model' button to start the training process.")

    st.divider()
    st.markdown("""
        ### ✅ Model Training Complete!
        The selected model has been trained and evaluated. You can now proceed to the **Prediction**
        section to use this trained model to forecast agricultural indicators.
    """)


# --- Standalone Testing Block ---
if __name__ == "__main__":
    # Simulate session state if running directly
    if 'engineered_agriculture_data' not in st.session_state:
        st.warning("Running Model Training page directly. Loading dummy engineered agriculture data.")
        # Create a dummy DataFrame that mimics the engineered data
        dummy_engineered_data = {
            'Year': np.arange(1990, 2020).tolist() * 2,
            'Indicator Name': ['Arable land (% of land area)'] * 30 + ['Cereal yield (kg per hectare)'] * 30,
            'Country Name': ['Nepal'] * 60,
            'Value': np.random.rand(60) * 1000,
            'Yearly Z-Score': np.random.randn(60),
            'Value % Change': np.random.rand(60) * 20 - 10, # -10 to 10
            '3-Year Rolling Mean': np.random.rand(60) * 900,
            'High Growth Flag': np.random.randint(0, 2, 60),
            'Normalized Value': np.random.rand(60)
        }
        st.session_state.engineered_agriculture_data = pd.DataFrame(dummy_engineered_data)
        st.info("Dummy engineered agricultural data loaded for direct testing.")

    # Explicitly set model, scaler, and features to None if not present for clean test runs
    if 'agriculture_model' not in st.session_state:
        st.session_state.agriculture_model = None
    if 'agriculture_scaler' not in st.session_state:
        st.session_state.agriculture_scaler = None
    if 'agriculture_model_features' not in st.session_state:
        st.session_state.agriculture_model_features = None
    if 'agriculture_label_encoder' not in st.session_state:
        st.session_state.agriculture_label_encoder = None

    display()