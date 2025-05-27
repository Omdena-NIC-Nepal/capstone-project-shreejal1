import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge # Added Ridge Regression as in the reference
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error # Added MAE as in the reference
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
# import joblib # Not strictly needed if only using session_state for model persistence
# import os     # Not strictly needed if only using session_state for model persistence
import time

# --- Initialize session state for trained model and scaler ---
# Ensure these match the keys you initialize in app.py
if 'climate_model' not in st.session_state:
    st.session_state.climate_model = None
if 'climate_scaler' not in st.session_state:
    st.session_state.climate_scaler = None
if 'climate_model_features' not in st.session_state:
    st.session_state.climate_model_features = None # To store column names used for training

# The main function for this page
def display_model_training(): # <--- THIS IS THE FUNCTION NAME TO ENSURE IT MATCHES app.py
    st.title("Climate Data Model Training")
    st.markdown("""
        In this section, you can choose a machine learning model to train on the feature-engineered climate data.
        We support Random Forest, Gradient Boosting, Linear Regression, and Ridge Regression.
    """)

    # --- Load the feature-engineered climate data from session state ---
    if 'engineered_climate_data' not in st.session_state or st.session_state.engineered_climate_data is None:
        st.error("❌ Engineered data not found. Please go to the 'Climate Analysis' -> 'Feature Engineering' page and process the data first.")
        return

    cleaned_climatedf = st.session_state.engineered_climate_data.copy()
    st.success("✅ Feature-engineered data loaded from session state.")

    st.subheader("Feature-Engineered Data Overview")
    st.write(cleaned_climatedf.head())


    # Define feature columns and target columns
    # Adjust these based on what was in your engineered_climate_data after FE
    # Ensure 'T2M' is available as the target
    if 'T2M' not in cleaned_climatedf.columns:
        st.error("❌ Target column 'T2M' not found in the engineered data. Cannot train model.")
        return

    # Dynamically determine feature columns by excluding target and ID/time columns
    # Adjust this list based on what your FE process actually outputs
    # and what you want to use as features.
    # Exclude columns that are definitely not features:
    cols_to_exclude_from_features = ['DATE', 'T2M'] # Assuming 'DATE' was kept, and 'T2M' is target

    # Filter out columns that don't exist in the DataFrame
    existing_cols_to_exclude = [col for col in cols_to_exclude_from_features if col in cleaned_climatedf.columns]
    feature_cols = [col for col in cleaned_climatedf.columns if col not in existing_cols_to_exclude]

    target_col = 'T2M' # Your target variable

    # Check if we have features left
    if not feature_cols:
        st.error("❌ No feature columns found after exclusion. Cannot train model.")
        st.info(f"DataFrame columns: {cleaned_climatedf.columns.tolist()}")
        st.info(f"Columns excluded: {existing_cols_to_exclude}")
        return

    st.write(f"**Target Column:** `{target_col}`")
    st.write(f"**Feature Columns ({len(feature_cols)}):** `{', '.join(feature_cols)}`")

    # Filter the DataFrame to only include features and target
    data_for_training = cleaned_climatedf[feature_cols + [target_col]].copy()

    # Handle any remaining missing values in the data used for training (after FE)
    initial_rows_for_training = data_for_training.shape[0]
    data_for_training.dropna(inplace=True)
    rows_after_training_dropna = data_for_training.shape[0]
    if initial_rows_for_training > rows_after_training_dropna:
        st.warning(f"Dropped {initial_rows_for_training - rows_after_training_dropna} rows with NaNs before training.")
    else:
        st.info("No NaNs found in training data (good).")

    if data_for_training.empty:
        st.error("❌ Data for training is empty after handling missing values. Cannot proceed.")
        return

    X = data_for_training[feature_cols]
    y = data_for_training[target_col]

    # Store feature column names in session state for prediction page consistency
    st.session_state.climate_model_features = X.columns.tolist()

    # Model selection dropdown
    model_choice = st.selectbox("Select Model", ["Random Forest", "Gradient Boosting", "Linear Regression", "Ridge Regression"])

    # Train/Test split slider
    split = st.slider("Select Train/Test Split %", min_value=50, max_value=95, value=80, step=5) / 100

    if st.button("Train Model"):
        with st.spinner('🚀 Training model. Please wait...'):
            try:
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 - split, random_state=42)

                # Initialize StandardScaler and fit on training data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Convert scaled arrays back to DataFrame to preserve column names for model (if needed by some models)
                # Or keep as numpy arrays if model accepts
                X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns, index=X_train.index)
                X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns, index=X_test.index)


                # Initialize the selected model
                if model_choice == "Random Forest":
                    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                elif model_choice == "Gradient Boosting":
                    model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
                elif model_choice == "Linear Regression":
                    model = LinearRegression()
                elif model_choice == "Ridge Regression":
                    model = Ridge(alpha=1.0) # You can add a slider for alpha if desired

                st.info(f"Starting training for {model_choice} model...")
                model.fit(X_train_scaled_df, y_train) # Train with DataFrame if preferred
                st.success(f"{model_choice} model trained successfully!")

                # Make predictions
                y_pred = model.predict(X_test_scaled_df) # Predict with DataFrame if preferred

                # Model Evaluation
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)

                st.progress(100, text="✅ Training Complete!")
                st.balloons()

                # Display Model Evaluation Metrics
                st.subheader("📊 Model Evaluation Metrics")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**R² Score:** {r2:.4f}")
                    st.write(f"**MSE:** {mse:.4f}")
                with col2:
                    st.write(f"**RMSE:** {rmse:.4f}")
                    st.write(f"**MAE:** {mae:.4f}")

                # --- Save the trained model and scaler in session state ---
                st.session_state.climate_model = model
                st.session_state.climate_scaler = scaler # Save the fitted scaler
                st.success("Model and Scaler stored in session state for future use!")

                # Plot Actual vs Predicted
                st.subheader("Actual vs Predicted Values")
                fig1, ax1 = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=y_test, y=y_pred, color='blue', label='Predictions', ax=ax1)
                ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Prediction')
                ax1.set_title("True vs Predicted Temperature")
                ax1.set_xlabel("True Temperature (°C)")
                ax1.set_ylabel("Predicted Temperature (°C)")
                ax1.legend()
                st.pyplot(fig1)
                plt.close(fig1)

                # Plot Residual Analysis (Residuals vs Predicted)
                st.subheader("Residual Analysis")
                residuals = y_test - y_pred
                fig2, ax2 = plt.subplots(figsize=(10, 6))
                sns.scatterplot(x=y_pred, y=residuals, ax=ax2)
                ax2.axhline(y=0, color='r', linestyle='--')
                ax2.set_title("Residuals vs Predicted Values")
                ax2.set_xlabel("Predicted Temperature (°C)")
                ax2.set_ylabel("Residuals")
                st.pyplot(fig2)
                plt.close(fig2)

                # Feature importances plot (only for tree-based models)
                if model_choice in ["Random Forest", "Gradient Boosting"]:
                    st.subheader("Feature Importances")
                    if hasattr(model, 'feature_importances_'):
                        feature_importances = pd.DataFrame({
                            'Feature': X.columns, # Use X.columns here
                            'Importance': model.feature_importances_
                        }).sort_values(by='Importance', ascending=False)

                        fig3, ax3 = plt.subplots(figsize=(10, 6))
                        sns.barplot(x='Importance', y='Feature', data=feature_importances, ax=ax3)
                        ax3.set_title("Feature Importance")
                        plt.tight_layout() # Adjust layout for potentially long labels
                        st.pyplot(fig3)
                        plt.close(fig3)
                    else:
                        st.info("Feature importances are not available for the selected model type.")

                st.divider()

            except Exception as e:
                st.error(f"An error occurred during model training or evaluation: {e}")
                st.exception(e) # Display full traceback for debugging
    else:
        st.info("Make sure to select a model and click 'Train Model' to begin training.")

# This block is only executed when the script is run directly (for testing purposes)
if __name__ == "__main__":
    # Simulate session state for local testing if needed
    # For this page to run directly, it needs 'engineered_climate_data' in session state.
    # You might need to load dummy data or run the FE page first in your main app.
    if 'engineered_climate_data' not in st.session_state:
        st.warning("Running Model Training page directly. No engineered data in session state. Attempting to load dummy data.")
        try:
            # Create some dummy data or load from a known path for testing
            # Ensure this dummy data structure matches what your FE page would produce
            dummy_data = pd.DataFrame(np.random.rand(100, 10), columns=[f'feature_{i}' for i in range(8)] + ['T2M', 'DATE'])
            dummy_data['DISTRICT'] = np.random.choice(['A', 'B'], 100)
            dummy_data['YEAR'] = np.random.randint(2000, 2020, 100)
            dummy_data['MONTH'] = np.random.randint(1, 13, 100)
            st.session_state.engineered_climate_data = dummy_data
            st.info("Dummy engineered data loaded for direct testing.")
        except Exception as e:
            st.error(f"Could not load/create dummy data for direct testing: {e}")
            st.session_state.engineered_climate_data = None # Ensure it's None if creation failed

    display_model_training()