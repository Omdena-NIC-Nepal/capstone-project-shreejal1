import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder # Make sure LabelEncoder is imported
import matplotlib.pyplot as plt
import seaborn as sns
import time

# --- Initialize session state for trained model and scaler ---
if 'climate_model' not in st.session_state:
    st.session_state.climate_model = None
if 'climate_scaler' not in st.session_state:
    st.session_state.climate_scaler = None
if 'climate_model_features' not in st.session_state:
    st.session_state.climate_model_features = None

# --- NEW: Initialize session state for encoders ---
if 'climate_district_encoder' not in st.session_state:
    st.session_state.climate_district_encoder = None
if 'climate_drought_index_encoder' not in st.session_state:
    st.session_state.climate_drought_index_encoder = None
if 'climate_heat_stress_encoder' not in st.session_state:
    st.session_state.climate_heat_stress_encoder = None


# The main function for this page
def display_model_training():
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

    # --- NEW: Handle categorical encoding before defining features ---
    # Initialize encoders
    district_encoder = LabelEncoder()
    drought_index_encoder = LabelEncoder()
    heat_stress_encoder = LabelEncoder()

    # Apply encoding if columns exist
    if 'DISTRICT' in cleaned_climatedf.columns:
        cleaned_climatedf['DISTRICT_Encoded'] = district_encoder.fit_transform(cleaned_climatedf['DISTRICT'])
        st.info("Encoded 'DISTRICT' into 'DISTRICT_Encoded'.")
    else:
        st.warning("'DISTRICT' column not found for encoding. Ensure your engineered data contains it.")

    if 'Drought Index' in cleaned_climatedf.columns:
        cleaned_climatedf['Drought_Index_Encoded'] = drought_index_encoder.fit_transform(cleaned_climatedf['Drought Index'])
        st.info("Encoded 'Drought Index' into 'Drought_Index_Encoded'.")
    else:
        st.warning("'Drought Index' column not found for encoding. Ensure your engineered data contains it.")

    if 'Heat Stress Metric' in cleaned_climatedf.columns:
        cleaned_climatedf['Heat_Stress_Encoded'] = heat_stress_encoder.fit_transform(cleaned_climatedf['Heat Stress Metric'])
        st.info("Encoded 'Heat Stress Metric' into 'Heat_Stress_Encoded'.")
    else:
        st.warning("'Heat Stress Metric' column not found for encoding. Ensure your engineered data contains it.")
    # --- END NEW: Categorical encoding ---


    # Define feature columns and target columns
    if 'T2M' not in cleaned_climatedf.columns:
        st.error("❌ Target column 'T2M' not found in the engineered data. Cannot train model.")
        return

    # Dynamically determine feature columns
    # Adjust this list based on what your FE process actually outputs
    # and what you want to use as features.
    cols_to_exclude_from_features = ['DATE', 'T2M', 'DISTRICT', 'Drought Index', 'Heat Stress Metric'] # Exclude original categorical columns

    # Filter out columns that don't exist in the DataFrame
    existing_cols_to_exclude = [col for col in cols_to_exclude_from_features if col in cleaned_climatedf.columns]
    
    # Feature columns will now include the newly encoded columns
    feature_cols = [col for col in cleaned_climatedf.columns if col not in existing_cols_to_exclude and cleaned_climatedf[col].dtype in ['float64', 'int64']]

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
                model.fit(X_train_scaled_df, y_train)
                st.success(f"{model_choice} model trained successfully!")

                # Make predictions
                y_pred = model.predict(X_test_scaled_df)

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

                # --- Save the trained model, scaler, AND ENCODERS in session state ---
                st.session_state.climate_model = model
                st.session_state.climate_scaler = scaler # Save the fitted scaler
                # --- NEW: Save fitted encoders ---
                st.session_state.climate_district_encoder = district_encoder
                st.session_state.climate_drought_index_encoder = drought_index_encoder
                st.session_state.climate_heat_stress_encoder = heat_stress_encoder
                # --- END NEW ---

                st.success("Model, Scaler, and Encoders stored in session state for future use!")

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
                        plt.tight_layout()
                        st.pyplot(fig3)
                        plt.close(fig3)
                    else:
                        st.info("Feature importances are not available for the selected model type.")

                st.divider()

            except Exception as e:
                st.error(f"An error occurred during model training or evaluation: {e}")
                st.exception(e)
    else:
        st.info("Make sure to select a model and click 'Train Model' to begin training.")

# This block is only executed when the script is run directly (for testing purposes)
if __name__ == "__main__":
    # Simulate session state for local testing if needed
    if 'engineered_climate_data' not in st.session_state:
        st.warning("Running Model Training page directly. No engineered data in session state. Attempting to load dummy data.")
        try:
            dummy_data = pd.DataFrame({
                'DATE': pd.to_datetime(['2000-01-01'] * 50 + ['2000-02-01'] * 50),
                'T2M': np.random.rand(100) * 10 + 20, # Example target
                'DISTRICT': np.random.choice(['Kathmandu', 'Pokhara'], 100),
                'Drought Index': np.random.choice(['Low', 'Medium', 'High'], 100),
                'Heat Stress Metric': np.random.choice(['Mild', 'Severe'], 100),
                'Precipitation': np.random.rand(100) * 100,
                'TemperatureAnomaly': np.random.randn(100),
                # Add other dummy features that you expect from your FE
            })
            st.session_state.engineered_climate_data = dummy_data
            st.info("Dummy engineered data loaded for direct testing.")
        except Exception as e:
            st.error(f"Could not load/create dummy data for direct testing: {e}")
            st.session_state.engineered_climate_data = None

    # Explicitly set model, scaler, and features to None if not present for clean test runs
    # This also applies to the NEW encoders
    if 'climate_model' not in st.session_state: st.session_state.climate_model = None
    if 'climate_scaler' not in st.session_state: st.session_state.climate_scaler = None
    if 'climate_model_features' not in st.session_state: st.session_state.climate_model_features = None
    if 'climate_district_encoder' not in st.session_state: st.session_state.climate_district_encoder = None
    if 'climate_drought_index_encoder' not in st.session_state: st.session_state.climate_drought_index_encoder = None
    if 'climate_heat_stress_encoder' not in st.session_state: st.session_state.climate_heat_stress_encoder = None

    display_model_training() # Ensure this matches the function name in app.py