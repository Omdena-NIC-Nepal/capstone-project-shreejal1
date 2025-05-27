import streamlit as st
import pandas as pd
import os # For robust path handling

# Function to ensure compatibility with Streamlit's internal serialization (e.g., to Arrow)
def sanitize_dataframe_for_streamlit(df):
    """
    Converts object-type columns in a DataFrame to string type to prevent
    potential serialization issues when storing in Streamlit's session state.
    """
    for col in df.columns:
        if df[col].dtype == 'object':  # Check for object dtype
            df[col] = df[col].astype(str)
    return df

def display():
    st.title("🌱 Agricultural Data Overview for Nepal")
    st.markdown("""
        Welcome to the Agricultural Data Overview! This section provides a first look at the key agricultural
        and rural development indicators for Nepal. Understanding this raw data is the foundation
        for analyzing climate change impacts on agriculture.
    """)

    agriculture_data_path = os.path.join("data", "agricultural_data", "agriculture-and-rural-development_npl.csv")

    st.subheader("📊 Data Loading and Preview")

    # Check if data is already in session state
    if 'agriculture_raw_data' in st.session_state and st.session_state.agriculture_raw_data is not None:
        agriculture_df = st.session_state.agriculture_raw_data
        st.success("✅ Agricultural data already loaded from session state.")
        st.info("You can click 'Load/Reload Data' if you wish to refresh the dataset from its source.")
    else:
        st.info("Agricultural data not found in session state. Attempting to load from file...")
        agriculture_df = None # Initialize as None if not in session state

    # Button to explicitly load or reload the data
    if st.button("Load/Reload Agricultural Data"):
        try:
            # skiprows=[1] is often used for World Bank data to skip metadata rows
            temp_df = pd.read_csv(agriculture_data_path, skiprows=[1])
            # Apply sanitation before storing in session state
            agriculture_df = sanitize_dataframe_for_streamlit(temp_df.copy())
            st.session_state.agriculture_raw_data = agriculture_df
            st.success(f"Successfully loaded {len(agriculture_df)} records from `{agriculture_data_path}` and stored in session state.")
        except FileNotFoundError:
            st.error(f"❌ Error: Agricultural data file not found at `{agriculture_data_path}`.")
            st.warning("Please ensure 'agriculture-and-rural-development_npl.csv' is in the 'data/agricultural_data/' directory.")
            agriculture_df = None
        except Exception as e:
            st.error(f"❌ An error occurred while loading agricultural data: {e}")
            agriculture_df = None

    if agriculture_df is not None:
        st.write("First 5 rows of the dataset:")
        st.dataframe(agriculture_df.head(), height=200)

        st.subheader("📚 Dataset Information")
        st.write(f"**Total records:** `{agriculture_df.shape[0]}`")
        st.write(f"**Number of columns:** `{agriculture_df.shape[1]}`")
        st.write("**Column names and data types:**")
        st.dataframe(agriculture_df.info(verbose=True, show_counts=True, buf=None)) # buf=None to display in Streamlit

        st.subheader("📈 Basic Statistics of Numerical Columns")
        st.write(agriculture_df.describe())

        st.subheader("🔍 Missing Values Overview")
        missing_values = agriculture_df.isnull().sum()
        missing_values = missing_values[missing_values > 0]
        if not missing_values.empty:
            st.write("Columns with missing values:")
            st.dataframe(missing_values.sort_values(ascending=False))
            st.warning("Note: Missing values will be addressed in the Feature Engineering step.")
        else:
            st.info("Great! No missing values detected in the dataset.")


        st.divider()

        st.markdown("""
        ## 📘 Understanding the Agricultural Dataset

        This dataset provides a comprehensive look at various agricultural and rural development indicators
        specifically for **Nepal**. It's designed to help us understand trends and patterns that might be
        influenced by, or in turn influence, climate change.

        ---

        ### 1. **Geographic and Temporal Context** 📍📅
        - **Country Name**: Indicates the country the data pertains to (e.g., 'Nepal').
        - **Country ISO3**: The standard three-letter ISO code for the country (e.g., 'NPL').
        - **Year**: The specific year the data was recorded, enabling time-series analysis.

        ---

        ### 2. **Indicator Details** 📊
        - **Indicator Name**: A human-readable description of the agricultural or development metric (e.g., 'Arable land (% of land area)').
        - **Indicator Code**: A unique, standardized code for each indicator, useful for programmatic access and data consistency (e.g., 'AG.LND.ARBL.ZS').

        ---

        ### 3. **Quantitative Value** 🔢
        - **Value**: The numerical measure of the indicator for the given year and country. This is the core data point for analysis.

        ---

        ### Expected Data Structure Considerations:
        The raw CSV typically follows a format common in development datasets, often featuring columns that look like:
        `#country+name`, `#country+code`, `#date+year`, `#indicator+name`, `##indicator+code`, `#indicator+value+num`.
        We'll need to parse these appropriately in subsequent steps.

        ---

        ### Potential Use Cases for this Data:
        - **Historical Trend Analysis**: Track changes in agricultural output, land use, or resource consumption over decades.
        - **Impact Assessment**: Correlate agricultural trends with climate variables to understand potential climate change impacts.
        - **Policy Evaluation**: Assess the effectiveness of agricultural policies or rural development programs.
        - **Predictive Modeling**: Develop models to forecast agricultural yields, resource demand, or other key indicators.

        Ready to dig deeper? Proceed to the **EDA** section for more detailed visualizations and insights!
        """)
    else:
        st.info("Please click the 'Load/Reload Agricultural Data' button above to begin exploring the dataset.")

# This block allows you to run the page directly for testing purposes
if __name__ == "__main__":
    # Simulate session state for local testing
    if 'agriculture_raw_data' not in st.session_state:
        st.session_state.agriculture_raw_data = None
    display()