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

# Title and description for the model training page
st.title("Climate Data Model Training")
st.markdown("""
    In this section, you can choose a machine learning model to train on the feature-engineered climate data.
    We support Random Forest, Gradient Boosting, and Linear Regression.
""")

# Load the feature-engineered climate data
climate_data_path = "data/climate_data/engineered_climate_data.csv"
climate_data = pd.read_csv(climate_data_path)

# Display initial data overview
st.subheader("Feature-Engineered Data Overview")
st.write(climate_data.head())

# Define features (X) and target (y)
X = climate_data.drop(columns=['DATE', 'YEAR', 'MONTH', 'T2M'])  # Exclude T2M as it's the target
y = climate_data['T2M']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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
        model = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=42, n_jobs=-1)
    elif model_choice == "Gradient Boosting":
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
    elif model_choice == "Linear Regression":
        model = LinearRegression()

    # Actual model training (fitting the model)
    model.fit(X_train, y_train)

    # Continue progress bar to 100%
    for i in range(50, 100):
        time.sleep(0.01)
        progress_bar.progress(i + 1)

    # Make predictions after training
    y_pred = model.predict(X_test)

    # Model evaluation: RMSE and R² score
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    r2 = r2_score(y_test, y_pred)

    # Display evaluation metrics
    st.subheader("Model Evaluation")
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    st.write(f"R² Score: {r2:.2f}")

    # Plot true vs predicted values
    st.subheader("True vs Predicted Temperature")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred, color='blue', label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Perfect Prediction')
    plt.title("True vs Predicted Temperature")
    plt.xlabel("True Temperature (°C)")
    plt.ylabel("Predicted Temperature (°C)")
    plt.legend()
    st.pyplot(plt)

    # Feature importances plot (only for Random Forest and Gradient Boosting)
    if model_choice in ["Random Forest", "Gradient Boosting"]:
        st.subheader("Feature Importances")
        feature_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=feature_importances)
        plt.title("Feature Importance")
        st.pyplot(plt)

    # Save the trained model and scaler
    model_path = "models/climate_model.pkl"
    scaler_path = "models/scaler.pkl"

    # Save the trained model using joblib
    joblib.dump(model, model_path)

    # Save the scaler
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, scaler_path)

    # Display model's performance summary
    st.markdown("""
        The model has been trained on the feature-engineered data. 
        We evaluated the model using RMSE and R² score, and visualized the model's performance with 
        a scatter plot comparing the true and predicted temperature values. 
        The model and scaler have been saved for future use in predictions.
    """)