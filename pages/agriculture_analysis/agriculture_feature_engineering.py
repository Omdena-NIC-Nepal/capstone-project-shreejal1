import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler # Using MinMaxScaler for 0-1 normalization

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
    st.title("⚙️ Agricultural Data Feature Engineering")
    st.markdown("""
        This section performs advanced feature engineering to enhance insights from agricultural development indicators.
        We'll create new variables that capture trends, changes, and normalized values, making the data
        more suitable for predictive modeling.
    """)

    # --- Load Data from Session State ---
    if 'agriculture_raw_data' not in st.session_state or st.session_state.agriculture_raw_data is None:
        st.error("❌ Agricultural raw data not found in session state. Please go to 'Agriculture Analysis' -> 'Overview' to load the data first.")
        return

    df_raw = st.session_state.agriculture_raw_data.copy()
    st.success("✅ Raw agricultural data loaded from session state for feature engineering.")

    st.subheader("🧹 Initial Data Preparation")
    st.info("Standardizing column names and converting 'Year' and 'Value' to numeric types.")

    # Standardize column names (if they are World Bank style)
    if '#country+name' in df_raw.columns:
        df_raw.rename(columns={'#country+name': 'Country Name'}, inplace=True)
    if '#country+code' in df_raw.columns:
        df_raw.rename(columns={'#country+code': 'Country ISO3'}, inplace=True)
    if '#date+year' in df_raw.columns:
        df_raw.rename(columns={'#date+year': 'Year'}, inplace=True)
    if '#indicator+name' in df_raw.columns:
        df_raw.rename(columns={'#indicator+name': 'Indicator Name'}, inplace=True)
    if '#indicator+code' in df_raw.columns:
        df_raw.rename(columns={'#indicator+code': 'Indicator Code'}, inplace=True)
    if '#indicator+value+num' in df_raw.columns:
        df_raw.rename(columns={'#indicator+value+num': 'Value'}, inplace=True)

    # Ensure core columns exist before proceeding
    required_cols = ['Year', 'Value', 'Indicator Name', 'Country Name']
    if not all(col in df_raw.columns for col in required_cols):
        st.error(f"❌ Critical columns missing for Feature Engineering. Expected: {required_cols}. Found: {df_raw.columns.tolist()}")
        st.warning("Please verify your data file and column headers.")
        return

    # Convert relevant columns to numeric
    df_raw['Value'] = pd.to_numeric(df_raw['Value'], errors='coerce')
    df_raw['Year'] = pd.to_numeric(df_raw['Year'], errors='coerce')

    # Drop rows with NaN values in essential columns before feature creation
    initial_rows = df_raw.shape[0]
    df_fe = df_raw.dropna(subset=['Year', 'Value', 'Indicator Name', 'Country Name']).copy()
    if initial_rows > df_fe.shape[0]:
        st.warning(f"Dropped {initial_rows - df_fe.shape[0]} rows due to missing essential values before feature creation.")
    else:
        st.info("No essential NaNs dropped before feature creation.")

    if df_fe.empty:
        st.error("❌ Data is empty after initial cleaning. Cannot perform feature engineering.")
        return

    st.write("Data preview after initial cleaning:")
    st.dataframe(df_fe.head())
    st.write(f"Data shape after initial cleaning: {df_fe.shape[0]} rows × {df_fe.shape[1]} columns")

    st.divider()

    st.subheader("✨ Creating New Features")

    # Feature 1: Year-wise Value Normalization (Z-score)
    if 'Value' in df_fe.columns and df_fe['Value'].dtype in ['float64', 'int64']:
        # Ensure that std() is not zero to avoid division by zero
        df_fe['Yearly Z-Score'] = df_fe.groupby('Indicator Name')['Value'].transform(
            lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
        )
        st.success("Created Feature: Yearly Z-Score per Indicator.")
    else:
        st.warning("Cannot create 'Yearly Z-Score': 'Value' column missing or not numeric.")

    # Feature 2: Percentage Change in Value from Previous Year (per indicator per country)
    if 'Value' in df_fe.columns and 'Year' in df_fe.columns and 'Country Name' in df_fe.columns:
        # Sort data before calculating percentage change for correct temporal order
        df_fe.sort_values(by=['Country Name', 'Indicator Name', 'Year'], inplace=True)
        df_fe['Value % Change'] = df_fe.groupby(['Country Name', 'Indicator Name'])['Value'].transform(
            lambda x: x.pct_change() * 100
        )
        st.success("Created Feature: Percentage Change in Value from Previous Year.")
    else:
        st.warning("Cannot create 'Value % Change': Missing 'Value', 'Year', or 'Country Name' columns.")

    # Feature 3: Rolling Mean (3-year) for smoothing
    if 'Value' in df_fe.columns and 'Year' in df_fe.columns and 'Country Name' in df_fe.columns:
        # Data is already sorted from previous step
        df_fe['3-Year Rolling Mean'] = df_fe.groupby(['Country Name', 'Indicator Name'])['Value'].transform(
            lambda x: x.rolling(window=3, min_periods=1).mean()
        )
        st.success("Created Feature: 3-Year Rolling Mean for smoothing.")
    else:
        st.warning("Cannot create '3-Year Rolling Mean': Missing 'Value', 'Year', or 'Country Name' columns.")

    # Feature 4: High Growth Flag (>10% YoY increase)
    if 'Value % Change' in df_fe.columns:
        df_fe['High Growth Flag'] = df_fe['Value % Change'].apply(lambda x: 1 if x > 10 else 0)
        st.success("Created Feature: High Growth Flag (for >10% YoY increase).")
    else:
        st.warning("Cannot create 'High Growth Flag': 'Value % Change' column missing.")

    # Feature 5: Normalized Value (0-1 scale) per Indicator
    if 'Value' in df_fe.columns and df_fe['Value'].dtype in ['float64', 'int64']:
        # Use MinMaxScaler from sklearn for robustness
        scaler_01 = MinMaxScaler()
        # Apply normalization within each indicator group
        df_fe['Normalized Value'] = df_fe.groupby('Indicator Name')['Value'].transform(
            lambda x: scaler_01.fit_transform(x.values.reshape(-1, 1)).flatten() if len(x) > 1 and (x.max() - x.min()) != 0 else 0
        )
        st.success("Created Feature: Normalized Value (0-1 scale) per Indicator.")
    else:
        st.warning("Cannot create 'Normalized Value': 'Value' column missing or not numeric.")

    # Handle any NaNs introduced by feature engineering (e.g., rolling means at the start of series)
    st.subheader("🧹 Handling New Missing Values")
    initial_rows_after_fe = df_fe.shape[0]
    df_fe.fillna(method='ffill', inplace=True)
    df_fe.fillna(method='bfill', inplace=True) # Catch leading NaNs
    if df_fe.isnull().sum().sum() == 0:
        st.success("All missing values introduced by feature engineering have been handled.")
    else:
        st.warning("Some NaNs might still exist after final fillna. Review data.")
        st.write(df_fe.isnull().sum()[df_fe.isnull().sum() > 0])


    # Display the first 10 rows of the engineered dataframe
    st.subheader("✨ Engineered Features Preview")
    st.dataframe(df_fe.head(10))
    st.write(f"Final engineered data shape: {df_fe.shape[0]} rows × {df_fe.shape[1]} columns")
    st.write("New columns created:")
    st.write([col for col in df_fe.columns if col not in df_raw.columns]) # Show only newly added columns

    st.divider()

    # --- Visualization of Engineered Features ---

    # Z-Score Distribution Visualization
    st.subheader("📊 Z-Score Distribution for Selected Indicator")
    indicator_list = df_fe['Indicator Name'].dropna().unique().tolist()
    if 'Yearly Z-Score' in df_fe.columns and indicator_list:
        selected_indicator_z = st.selectbox("Choose an Indicator for Z-Score Distribution:",
                                            sorted(indicator_list), index=0, key='z_score_indicator')
        indicator_df_z = df_fe[df_fe['Indicator Name'] == selected_indicator_z].copy()
        if not indicator_df_z['Yearly Z-Score'].dropna().empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(indicator_df_z['Yearly Z-Score'].dropna(), bins=20, kde=True, ax=ax, color='skyblue')
            ax.set_title(f"Z-Score Distribution for: {selected_indicator_z}")
            ax.set_xlabel("Z-Score")
            ax.set_ylabel("Frequency")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No valid Z-Score data for the selected indicator.")
    else:
        st.warning("Cannot plot Z-Score distribution: 'Yearly Z-Score' column missing or no indicators found.")

    st.divider()

    # Yearly Trend Visualization: Raw Value vs Rolling Mean
    st.subheader("📈 Yearly Trend: Raw Value vs Rolling Mean")
    if '3-Year Rolling Mean' in df_fe.columns and 'Value' in df_fe.columns and 'Year' in df_fe.columns:
        country_list = df_fe['Country Name'].dropna().unique().tolist()
        if country_list and indicator_list:
            selected_country_roll = st.selectbox("Select Country for Rolling Mean Trend:",
                                                 sorted(country_list), index=0, key='rolling_country')
            selected_indicator_roll = st.selectbox("Select Indicator for Rolling Mean Trend:",
                                                   sorted(indicator_list), index=0, key='rolling_indicator')

            indicator_data_roll = df_fe[(df_fe['Country Name'] == selected_country_roll) &
                                        (df_fe['Indicator Name'] == selected_indicator_roll)].copy()
            indicator_data_roll.sort_values('Year', inplace=True)

            if not indicator_data_roll.empty:
                fig, ax = plt.subplots(figsize=(10, 5))
                sns.lineplot(x='Year', y='Value', data=indicator_data_roll, marker='o', label='Raw Value', ax=ax, color='blue')
                sns.lineplot(x='Year', y='3-Year Rolling Mean', data=indicator_data_roll, marker='x', label='3-Year Rolling Mean', ax=ax, color='red', linestyle='--')
                ax.set_title(f"{selected_indicator_roll} Over Time ({selected_country_roll})")
                ax.set_xlabel("Year")
                ax.set_ylabel("Value")
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No data available for the selected country and indicator for rolling mean trend.")
        else:
            st.warning("Cannot plot Rolling Mean Trend: Country or Indicator list is empty.")
    else:
        st.warning("Cannot plot Rolling Mean Trend: Missing '3-Year Rolling Mean', 'Value', 'Year' columns.")

    st.divider()

    # Growth Flag Count Visualization Over Time
    st.subheader("📊 High Growth Flag Count Over Time")
    if 'High Growth Flag' in df_fe.columns and 'Year' in df_fe.columns:
        if country_list and indicator_list: # Reuse country and indicator lists
            selected_country_flag = st.selectbox("Select Country for Growth Flag:",
                                                 sorted(country_list), index=0, key='flag_country')
            selected_indicator_flag = st.selectbox("Select Indicator for Growth Flag:",
                                                   sorted(indicator_list), index=0, key='flag_indicator')

            flag_data_filtered = df_fe[(df_fe['Country Name'] == selected_country_flag) &
                                       (df_fe['Indicator Name'] == selected_indicator_flag)].copy()
            flag_df_agg = flag_data_filtered.groupby('Year')['High Growth Flag'].sum().reset_index()

            if not flag_df_agg.empty:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=flag_df_agg, x='Year', y='High Growth Flag', ax=ax, palette='viridis')
                ax.set_title(f"High Growth Flag Count Per Year ({selected_country_flag}, {selected_indicator_flag})")
                ax.set_xlabel("Year")
                ax.set_ylabel("Count of High Growth Events")
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("No high growth flag data for the selected country and indicator.")
        else:
            st.warning("Cannot plot Growth Flag: Country or Indicator list is empty.")
    else:
        st.warning("Cannot plot Growth Flag: Missing 'High Growth Flag' or 'Year' columns.")

    st.divider()

    # --- Store the engineered dataframe back into session state for further use ---
    # Apply sanitation before storing in session state
    df_fe_sanitized = sanitize_dataframe_for_streamlit(df_fe.copy())
    st.session_state.engineered_agriculture_data = df_fe_sanitized
    st.success("✅ Feature-engineered agricultural data saved to session state ('engineered_agriculture_data').")

    st.markdown("""
        ### ✅ Feature Engineering Complete!
        The agricultural data has been enriched with new features and is now ready for model training.
        Proceed to the **Model Training** section to build predictive models on this enhanced dataset.
    """)

# This block allows you to run the page directly for testing purposes
if __name__ == "__main__":
    # Simulate session state for local testing
    if 'agriculture_raw_data' not in st.session_state:
        st.warning("Running Feature Engineering page directly. No raw agriculture data in session state. Attempting to load dummy data.")
        # Create a dummy dataframe that mimics the structure of your agricultural data
        dummy_data = {
            'Country Name': ['Nepal'] * 50,
            'Country ISO3': ['NPL'] * 50,
            'Year': list(range(1980, 2030)),
            'Indicator Name': ['Arable land (% of land area)'] * 10 +
                              ['Fertilizer consumption (kg per hectare of arable land)'] * 10 +
                              ['Agricultural land (% of land area)'] * 10 +
                              ['Rural population (% of total population)'] * 10 +
                              ['Cereal yield (kg per hectare)'] * 10,
            'Indicator Code': ['AG.LND.ARBL.ZS'] * 10 +
                              ['AG.CON.FERT.HA.ARBL'] * 10 +
                              ['AG.LND.AGRI.ZS'] * 10 +
                              ['SP.RUR.TOTL.ZS'] * 10 +
                              ['AG.YLD.CREL.KG.HA'] * 10,
            'Value': np.random.rand(50) * 100 # Random values
        }
        # Add some NaNs for testing fillna
        dummy_data['Value'][5:7] = np.nan
        dummy_data['Year'][12:14] = np.nan
        dummy_data['Indicator Name'][20:22] = np.nan

        df_dummy = pd.DataFrame(dummy_data)
        # Manually apply the initial cleaning that overview/eda would do
        df_dummy['Year'] = pd.to_numeric(df_dummy['Year'], errors='coerce')
        df_dummy['Value'] = pd.to_numeric(df_dummy['Value'], errors='coerce')
        df_dummy.dropna(subset=['Year', 'Value', 'Indicator Name', 'Country Name'], inplace=True)

        st.session_state.agriculture_raw_data = df_dummy.copy()
        st.info("Dummy raw agricultural data loaded for direct testing of Feature Engineering page.")

    display()