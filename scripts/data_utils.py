import pandas as pd
import streamlit as st
import geopandas as gpd
import folium
import pandas.api.types as ptypes

@st.cache_data
def load_data(file: str, **kwargs) -> pd.DataFrame | None:
    """
    Load data from a CSV file into a pandas DataFrame.
    Accepts additional keyword arguments (e.g., skiprows).
    """
    try:
        df = pd.read_csv(file, encoding='utf-8', **kwargs)
        return df
    except Exception as e:
        st.error(f"❌ Error loading data from {file}: {e}")
        return None


def remove_unwanted_columns(df: pd.DataFrame, columns_to_remove: list) -> pd.DataFrame:
    """
    Remove unwanted columns from the DataFrame.
    """
    return df.drop(columns=columns_to_remove, errors='ignore')


def clean_data(df: pd.DataFrame, method: str) -> pd.DataFrame:
    """
    Clean the DataFrame using the specified method.
    Options:
        - 'dropna': Remove all rows with NaN values
        - 'fillna': Fill NaNs with 0
        - 'interpolate': Interpolate missing values
    """
    if method == 'dropna':
        return df.dropna()
    elif method == 'fillna':
        return df.fillna(0)
    elif method == 'interpolate':
        return df.interpolate()
    else:
        raise ValueError("Invalid cleaning method. Use 'dropna', 'fillna', or 'interpolate'.")


def sanitize_for_arrow(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert object columns in the DataFrame to strings to avoid PyArrow conversion errors
    when using functions like `pa.Table.from_pandas()`.
    """
    df = df.copy()
    for col in df.columns:
        if ptypes.is_object_dtype(df[col]):
            df[col] = df[col].astype(str)
    return df
