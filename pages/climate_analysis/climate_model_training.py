import streamlit as st
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
import os # Import os for path handling

# Title and description for the model training page
st.title("Climate Data Model Training")
st.markdown("""
    In this section, you can choose a machine learning model to train on the feature-engineered climate data.
    We support Random Forest, Gradient Boosting, and Linear Regression.
""")

# Load the feature-engineered climate data
climate_data_path = "data/climate_data/engineered_climate_data.csv"
st.info(f"Attempting to load data from: `{climate_data_path}`")
try:
    climate_data = pd.read_csv(climate_data_path)
    st.success(f"Successfully loaded data from `{climate_data_path}`")
except FileNotFoundError:
    st.error(f"Error: The file '{climate_data_path}' was not found. Please ensure it exists in the correct 'data/climate_data/' directory relative to your app.py.")
    st.stop() # Stop the app if data is not found
except Exception as e:
    st.error(f"An error occurred while loading the data: {e}")
    st.stop()

# Display initial data overview
st.subheader("Feature-Engineered Data Overview (Raw)")
st.write(climate_data.head())
st.write(f"Initial shape: {climate_data.shape}")
st.write(f"Initial columns: {climate_data.columns.tolist()}")

# Preprocessing steps
st.subheader("Data Preprocessing")

# Identify non-numeric columns before coercion
non_numeric_before_coercion = climate_data.select_dtypes(include=['object', 'string']).columns.tolist()
if non_numeric_before_coercion:
    st.info(f"Columns detected as non-numeric before coercion: {non_numeric_before_coercion}")
else:
    st.info("No non-numeric columns detected before explicit coercion (good sign).")


# Convert all columns (except 'DATE') to numeric, coercing errors
# Assume 'DATE' is a string/datetime and should not be converted to numeric directly for features
cols_to_convert_to_numeric = [col for col in climate_data.columns if col not in ['DATE']]
st.info(f"Attempting to convert these columns to numeric: {cols_to_convert_to_numeric}")

for col in cols_to_convert_to_numeric:
    # We use .copy() to avoid SettingWithCopyWarning if climate_data is a slice
    climate_data.loc[:, col] = pd.to_numeric(climate_data[col], errors='coerce')

st.info("Finished numeric coercion. Checking for new NaNs introduced by coercion.")

# Check for missing values after initial load and coercion
missing_values_after_coercion = climate_data.isnull().sum()
missing_values_after_coercion_report = missing_values_after_coercion[missing_values_after_coercion > 0]
if not missing_values_after_coercion_report.empty:
    st.warning(f"Missing values detected after numeric coercion (these will be dropped):\n{missing_values_after_coercion_report}")
else:
    st.success("No new missing values introduced by numeric coercion.")

# Handle missing values: Drop rows with any missing values
original_rows = climate_data.shape[0]
climate_data_cleaned = climate_data.dropna().copy() # Use .copy() to ensure it's a new DataFrame
rows_after_dropna = climate_data_cleaned.shape[0]

if original_rows > rows_after_dropna:
    st.info(f"Dropped {original_rows - rows_after_dropna} rows with missing values.")
else:
    st.success("No missing values were dropped (or none found after coercion).")

st.write(f"Shape after dropping NaNs: {climate_data_cleaned.shape}")

# Define features (X) and target (y)
# Ensure 'DATE', 'YEAR', 'MONTH', 'T2M' are indeed column names in your CSV
columns_to_exclude = ['DATE', 'YEAR', 'MONTH', 'T2M']
# Filter out columns that don't exist in the dataframe before attempting to drop
existing_columns_to_exclude = [col for col in columns_to_exclude if col in climate_data_cleaned.columns]

st.info(f"Columns to exclude from features (X): {existing_columns_to_exclude}")
st.info(f"All columns in cleaned data: {climate_data_cleaned.columns.tolist()}")

try:
    X = climate_data_cleaned.drop(columns=existing_columns_to_exclude)
    # Ensure T2M exists and is numeric for the target
    if 'T2M' in climate_data_cleaned.columns:
        y = climate_data_cleaned['T2M']
    else:
        st.error("Error: 'T2M' (target column) not found in the cleaned data. Please check your CSV.")
        st.stop()
except KeyError as e:
    st.error(f"Error dropping columns or selecting target: {e}. One of the columns to exclude might not exist after cleaning.")
    st.stop()


# Final check for missing values and data types in X and y *before* splitting
st.subheader("Pre-Training Data Validation (X and y)")

# Check if there are missing values in X or y
if X.isnull().sum().any():
    st.error("Error: Missing values found in features (X) AFTER final cleaning. Please review preprocessing.")
    st.write(X.isnull().sum()[X.isnull().sum() > 0])
    st.stop()
if y.isnull().sum().any():
    st.error("Error: Missing values found in target (y) AFTER final cleaning. Please review preprocessing.")
    st.write(y.isnull().sum())
    st.stop()
st.success("No missing values in X or y.")

# Check data types for X and y
st.write("Data Types of Features (X):")
st.write(X.dtypes)
non_numeric_in_X = X.select_dtypes(exclude=['number']).columns.tolist()
if non_numeric_in_X:
    st.error(f"Error: Non-numeric columns still present in features (X): {non_numeric_in_X}")
    st.stop()
else:
    st.success("All features (X) are numeric.")

st.write("Data Type of Target (y):")
st.write(y.dtype)
if not pd.api.types.is_numeric_dtype(y):
    st.error(f"Error: Target (y) is not numeric. Its type is: {y.dtype}")
    st.stop()
else:
    st.success("Target (y) is numeric.")

# Display a sample of X and y right before splitting
st.write("Sample of X (features) before splitting:")
st.write(X.head())
st.write("Sample of y (target) before splitting:")
st.write(y.head())

# Split the data into training and testing sets
try:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.success(f"Data split into training ({len(X_train)} samples) and testing ({len(X_test)} samples) sets.")
except Exception as e:
    st.error(f"Error splitting data: {e}")
    st.stop()

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
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    elif model_choice == "Linear Regression":
        model = LinearRegression()

    st.info(f"Starting training for {model_choice} model...")
    st.info(f"Shape of X_train: {X_train.shape}")
    st.info(f"Shape of y_train: {y_train.shape}")
    st.info(f"Data types of X_train columns immediately before fit:\n{X_train.dtypes}")


    try:
        # Actual model training (fitting the model)
        model.fit(X_train, y_train)
        st.success(f"{model_choice} model trained successfully!")
    except Exception as e:
        st.error(f"**CRITICAL ERROR DURING MODEL TRAINING!** The error is: {e}")
        st.error("Please examine the data types and missing values reported above.")
        st.stop() # Stop if training fails

    # Continue progress bar to 100%
    for i in range(50, 100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    # Make predictions after training
    y_pred = model.predict(X_test)

    # Model evaluation: RMSE and R² score
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

    # Feature importances plot (only for Random Forest and Gradient Boosting)
    if model_choice in ["Random Forest", "Gradient Boosting"]:
        st.subheader("Feature Importances")
        # Ensure feature importances are available
        if hasattr(model, 'feature_importances_'):
            feature_importances = pd.DataFrame({
                'Feature': X.columns,
                'Importance': model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.barplot(x='Importance', y='Feature', data=feature_importances, ax=ax2)
            ax2.set_title("Feature Importance")
            st.pyplot(fig2)
        else:
            st.info("Feature importances are not available for the selected model type.")

    # Save the trained model and scaler
    model_dir = "models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    model_path = os.path.join(model_dir, "climate_model.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")

    try:
        # Save the trained model using joblib
        joblib.dump(model, model_path)
        st.success(f"Trained model saved to {model_path}")
    except Exception as e:
        st.error(f"Error saving model: {e}")

    # Save the scaler
    try:
        scaler = StandardScaler()
        # It's better to fit the scaler on the training data only to avoid data leakage
        scaler.fit(X_train)
        joblib.dump(scaler, scaler_path)
        st.success(f"Scaler saved to {scaler_path}")
    except Exception as e:
        st.error(f"Error saving scaler: {e}")

    # Display model's performance summary
    st.markdown("""
        The model has been trained on the feature-engineered data.
        We evaluated the model using RMSE and R² score, and visualized the model's performance with
        a scatter plot comparing the true and predicted temperature values.
        The model and scaler have been saved for future use in predictions.
    """)