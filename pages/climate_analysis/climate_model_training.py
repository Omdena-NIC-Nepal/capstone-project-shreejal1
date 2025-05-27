import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
from sklearn.preprocessing import StandardScaler
import os # Ensure os is imported

# Title and description for the model training page
st.title("Climate Data Model Training")
st.markdown("""
    In this section, you can choose a machine learning model to train on the feature-engineered climate data.
    We support Random Forest, Gradient Boosting, and Linear Regression.
""")

# Load the feature-engineered climate data
climate_data_path = "data/climate_data/engineered_climate_data.csv"
try:
    climate_data = pd.read_csv(climate_data_path)
    st.success(f"Successfully loaded data from {climate_data_path}")
except FileNotFoundError:
    st.error(f"Error: The file '{climate_data_path}' was not found. Please ensure it exists.")
    st.stop() # Stop the app if data is not found
except Exception as e:
    st.error(f"An error occurred while loading the data: {e}")
    st.stop()


# Display initial data overview
st.subheader("Feature-Engineered Data Overview")
st.write(climate_data.head())

# Check for missing values
missing_values = climate_data.isnull().sum()
if missing_values.any():
    st.warning(f"Missing values detected in the dataset:\n{missing_values[missing_values > 0]}")

# Handle missing values: Drop rows with any missing values
original_rows = climate_data.shape[0]
climate_data = climate_data.dropna().copy() # Ensure to use .copy()
rows_after_dropna = climate_data.shape[0]
if original_rows > rows_after_dropna:
    st.info(f"Dropped {original_rows - rows_after_dropna} rows with missing values.")
else:
    st.success("No missing values were dropped.")

# Define features (X) and target (y)
columns_to_exclude = ['DATE', 'YEAR', 'MONTH', 'T2M']
existing_columns_to_exclude = [col for col in columns_to_exclude if col in climate_data.columns]

try:
    X = climate_data.drop(columns=existing_columns_to_exclude)
    if 'T2M' in climate_data.columns:
        y = climate_data['T2M']
    else:
        st.error("Error: 'T2M' (target column) not found in the cleaned data. Please check your CSV.")
        st.stop()
except KeyError as e:
    st.error(f"Error dropping columns or selecting target: {e}. One of the columns to exclude might not exist after cleaning.")
    st.stop()


# Check if there are missing values after handling them
if X.isnull().sum().any():
    st.error("Error: Missing values found in features (X) AFTER final cleaning. Please review preprocessing.")
    st.write(X.isnull().sum()[X.isnull().sum() > 0])
    st.stop()
if y.isnull().sum().any():
    st.error("Error: Missing values found in target (y) AFTER final cleaning. Please review preprocessing.")
    st.write(y.isnull().sum())
    st.stop()
st.success("No missing values in X or y.")

# Split the data into training and testing sets
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.success(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")
except Exception as e:
    st.error(f"Error splitting data: {e}")
    st.stop()

# Apply StandardScaler to standardize the data (important for models like Linear Regression)
# Initialize scaler here
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model selection dropdown
model_choice = st.selectbox(
    "Choose the model you want to train:",
    options=["Random Forest", "Gradient Boosting", "Linear Regression"]
)

# Add a button to start training the model
if st.button('Train Model'):
    # Create a progress bar
    progress_bar = st.progress(0)

    # Simulate loading/progress steps
    for i in range(50):
        time.sleep(0.02)
        progress_bar.progress(i + 1)

    # Initialize the selected model
    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1) # Increased n_estimators for better performance
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    elif model_choice == "Linear Regression":
        model = LinearRegression()

    st.info(f"Starting training for {model_choice} model...")
    try:
        # Actual model training (fitting the model)
        model.fit(X_train_scaled, y_train)
        st.success(f"{model_choice} model trained successfully!")
    except Exception as e:
        st.error(f"Error during model training for {model_choice}: {e}")
        st.stop() # Stop if training fails

    # Continue progress bar to 100%
    for i in range(50, 100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    # Make predictions after training
    y_pred = model.predict(X_test_scaled)

    # Model evaluation: RMSE and R² score
    # Corrected the usage of mean_squared_error to get RMSE
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Display evaluation metrics
    st.subheader("Model Evaluation")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    # Plot true vs predicted values
    st.subheader("True vs Predicted Temperature")
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='blue', label='Predictions', ax=ax1)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Prediction')
    ax1.set_title("True vs Predicted Temperature")
    ax1.set_xlabel("True Temperature (°C)")
    ax1.set_ylabel("Predicted Temperature (°C)")
    ax1.legend()
    st.pyplot(fig1)
    plt.close(fig1) # Close the figure to free up memory

    # Feature importances plot (only for Random Forest and Gradient Boosting)
    if model_choice in ["Random Forest", "Gradient Boosting"]:
        st.subheader("Feature Importances")
        if hasattr(model, 'feature_importances_'):
            feature_importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importances, ax=ax2)
            ax2.set_title("Feature Importance")
            st.pyplot(fig2)
            plt.close(fig2) # Close the figure to free up memory
        else:
            st.info("Feature importances are not available for the selected model type.")

    # --- Start of Enhanced Saving Debugging ---
    st.subheader("Saving Model and Scaler")

    model_dir = "models"
    model_path = os.path.join(model_dir, "climate_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    st.info(f"Attempting to save model to: `{os.path.abspath(model_path)}`")
    st.info(f"Attempting to save scaler to: `{os.path.abspath(scaler_path)}`")
    st.info(f"Current working directory: `{os.getcwd()}`")

    # 1. Create the directory if it doesn't exist
    if not os.path.exists(model_dir):
        st.info(f"Directory '{model_dir}' does not exist. Attempting to create it...")
        try:
            os.makedirs(model_dir)
            st.success(f"Directory '{model_dir}' created successfully.")
        except OSError as e:
            st.error(f"Error creating directory '{model_dir}': {e}")
            st.warning("This is often a permissions issue. The app might not have write access to this location.")
            st.stop() # Stop if directory creation fails, as saving will also fail
    else:
        st.info(f"Directory '{model_dir}' already exists.")

    # 2. Save the trained model
    try:
        joblib.dump(model, model_path)
        st.success(f"Trained model saved to {model_path}")
        # Verify file existence (for local testing)
        if os.path.exists(model_path):
            st.info(f"Confirmed: Model file exists at {model_path}")
        else:
            st.warning(f"Warning: Model file was reported saved, but could not be verified at {model_path}")
    except Exception as e:
        st.error(f"Error saving model to {model_path}: {e}")
        st.warning("Common causes: permissions, disk full, invalid path.")

    # 3. Save the scaler
    # Note: `scaler` was already initialized and fitted on X_train above.
    try:
        joblib.dump(scaler, scaler_path)
        st.success(f"Scaler saved to {scaler_path}")
        # Verify file existence (for local testing)
        if os.path.exists(scaler_path):
            st.info(f"Confirmed: Scaler file exists at {scaler_path}")
        else:
            st.warning(f"Warning: Scaler file was reported saved, but could not be verified at {scaler_path}")
    except Exception as e:
        st.error(f"Error saving scaler to {scaler_path}: {e}")
        st.warning("Common causes: permissions, disk full, invalid path.")

    # --- End of Enhanced Saving Debugging ---

    # Display model's performance summary
    st.markdown("""
        The model has been trained on the feature-engineered data.
        We evaluated the model using RMSE and R² score, and visualized the model's performance with
        a scatter plot comparing the true and predicted temperature values.
        The model and scaler have been saved for future use in predictions.
    """)