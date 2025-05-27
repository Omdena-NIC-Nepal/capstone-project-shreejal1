import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd


def get_model(model_choice):
    """Return the base model based on user choice."""
    if model_choice == "Random Forest":
        return RandomForestRegressor()
    elif model_choice == "Gradient Boosting":
        return GradientBoostingRegressor()
    elif model_choice == "Linear Regression":
        return LinearRegression()
    elif model_choice == "Ridge Regression":
        return Ridge()
    else:
        raise ValueError("Invalid model choice")
    
def train_model(df, model_choice, split, feature_cols, target_cols):
    """Train and return the model and evaluation metrics."""
    X = df[feature_cols]
    y = df[target_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=(1-split), random_state=42)

    base_model = get_model(model_choice)
    model = MultiOutputRegressor(base_model)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "R2 Score": r2,
        'y_true': y_test,      
        'y_pred': y_pred   
    }

    return model, metrics


def prepare_input(year, indicator_name):
    """
    Prepares the input data for prediction.
    """
    # Map indicator names to numerical values (assuming the model expects these numerical representations)
    indicator_map = {
        "Fertilizer consumption (% of fertilizer production)": 0,
        "Cereal yield (kg per hectare)": 1,
        "Irrigated area (% of total area)": 2,
        "Pesticide use (kg per hectare)": 3,
    }

    # Prepare the input data as a DataFrame with necessary features (year and indicator)
    input_data = pd.DataFrame({
        'Year': [year],
        'Indicator Name': [indicator_map[indicator_name]],  # Convert indicator name to a numerical value
    })

    return input_data