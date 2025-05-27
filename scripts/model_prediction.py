import pandas as pd

def prepare_input(year, month, district):
    """Prepares input DataFrame for prediction."""
    input_data = pd.DataFrame({
        'year': [year],
        'month': [month],
        'district': [district]
    })
    return input_data

def predict(model, input_data):
    """Generates prediction using the trained model."""
    predictions = model.predict(input_data)[0]
    return predictions



import pandas as pd

def agri_input(latitudes, longitudes, mean_elevations, mean_lengths, mean_depths, height_ranges, average_slopes, compactnesses):
    data = {
        'Latitude': latitudes,
        'Longitude': longitudes,
        'Mean Elevation': mean_elevations,
        'Mean Length': mean_lengths,
        'Mean Depth': mean_depths,
        'Height Range': height_ranges,
        'Average Slope (deg)': average_slopes,
        'Compactness': compactnesses
    }
    return pd.DataFrame(data)
def predict(model, input_data):
    """
    Predicts the output using the trained model and input data.
    """
    prediction = model.predict(input_data)
    return prediction