import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # For numerical operations

def display():
    st.title("📈 Agricultural Data Exploratory Analysis")
    st.markdown("""
        This section provides an in-depth exploratory data analysis (EDA) of agricultural indicators for Nepal.
        We'll visualize trends, distributions, and relationships within the data to uncover key insights
        and prepare for more advanced analysis.
    """)

    # --- Load Data from Session State ---
    if 'agriculture_raw_data' not in st.session_state or st.session_state.agriculture_raw_data is None:
        st.error("❌ Agricultural data not found in session state. Please go to 'Agriculture Analysis' -> 'Overview' to load the data first.")
        return

    df_raw = st.session_state.agriculture_raw_data.copy() # Work on a copy to avoid modifying the original session state data
    st.success("✅ Agricultural data loaded successfully from session state.")

    # --- Data Preprocessing for EDA ---
    st.subheader("🧹 Data Preparation for Analysis")
    st.info("Converting 'Year' and 'Value' columns to numeric and handling missing values.")

    # Standardize column names if they are World Bank style
    # This is critical for consistency in EDA
    column_mapping = {
        'Country Name': 'Country Name',
        'Country ISO3': 'Country ISO3',
        'Year': 'Year', # Sometimes '#date+year'
        'Indicator Name': 'Indicator Name', # Sometimes '#indicator+name'
        'Indicator Code': 'Indicator Code', # Sometimes '#indicator+code'
        'Value': 'Value' # Sometimes '#indicator+value+num'
    }

    # Iterate and rename if source columns exist. Use a more robust check for variations.
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
        st.error(f"❌ Critical columns missing for EDA. Expected: {required_cols}. Found: {df_raw.columns.tolist()}")
        st.warning("Please verify your data file and column headers.")
        return

    # Convert to appropriate types
    df_raw['Year'] = pd.to_numeric(df_raw['Year'], errors='coerce')
    df_raw['Value'] = pd.to_numeric(df_raw['Value'], errors='coerce')

    # Drop rows where essential columns are NaN
    initial_rows = df_raw.shape[0]
    df_eda = df_raw.dropna(subset=['Year', 'Value', 'Indicator Name']).copy()
    if initial_rows > df_eda.shape[0]:
        st.warning(f"Dropped {initial_rows - df_eda.shape[0]} rows due to missing 'Year', 'Value', or 'Indicator Name' for EDA.")

    if df_eda.empty:
        st.error("❌ Data is empty after initial cleaning. No analysis can be performed.")
        return

    st.write("Cleaned data preview for EDA:")
    st.dataframe(df_eda.head())
    st.write(f"Cleaned data shape: {df_eda.shape[0]} rows × {df_eda.shape[1]} columns")

    st.divider()

    # --- General Data Overview ---
    st.subheader("📚 General Data Insights")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Data types after cleaning:")
        st.dataframe(df_eda.dtypes, use_container_width=True)
    with col2:
        st.write("Missing values (after dropping essential NaNs):")
        st.dataframe(df_eda.isnull().sum()[df_eda.isnull().sum() > 0], use_container_width=True)
        if df_eda.isnull().sum().sum() == 0:
            st.info("No remaining missing values in the data used for EDA.")

    st.subheader("📊 Statistical Summary of Values")
    st.dataframe(df_eda.describe())

    st.divider()

    # --- Time Series Plot Section (Single Indicator) ---
    st.subheader("📈 Trend Over Time for a Single Indicator")

    indicator_list = df_eda['Indicator Name'].dropna().unique().tolist()
    if not indicator_list:
        st.warning("No unique indicators found to plot time series.")
    else:
        selected_indicator = st.selectbox(
            "Select an Agricultural Indicator to view its trend:",
            indicator_list,
            index=0 if indicator_list else None,
            key='single_indicator_select'
        )

        if selected_indicator:
            single_indicator_data = df_eda[df_eda['Indicator Name'] == selected_indicator].copy()
            single_indicator_data.sort_values('Year', inplace=True)

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(x='Year', y='Value', data=single_indicator_data, marker='o', ax=ax)
            ax.set_title(f"Annual Trend for: {selected_indicator}")
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("Select an indicator to see its time series plot.")

    st.divider()

    # --- Comparison of Two Time Series ---
    st.subheader("🔄 Compare Trends of Two Indicators")

    if len(indicator_list) >= 2:
        col_comp1, col_comp2 = st.columns(2)
        with col_comp1:
            indicator_1 = st.selectbox(
                "Select First Indicator:", indicator_list,
                index=0, key='compare_indicator_1'
            )
        with col_comp2:
            # Ensure the default index is different from the first, if possible
            default_idx_2 = 1 if len(indicator_list) > 1 else 0
            if indicator_list[default_idx_2] == indicator_1 and len(indicator_list) > 2:
                default_idx_2 = 2
            indicator_2 = st.selectbox(
                "Select Second Indicator:", indicator_list,
                index=default_idx_2, key='compare_indicator_2'
            )

        if indicator_1 and indicator_2 and indicator_1 != indicator_2:
            data_ind1 = df_eda[df_eda['Indicator Name'] == indicator_1].copy().sort_values('Year')
            data_ind2 = df_eda[df_eda['Indicator Name'] == indicator_2].copy().sort_values('Year')

            fig, ax = plt.subplots(figsize=(12, 6))
            sns.lineplot(x='Year', y='Value', data=data_ind1, marker='o', label=indicator_1, ax=ax, color='blue')
            sns.lineplot(x='Year', y='Value', data=data_ind2, marker='x', label=indicator_2, ax=ax, color='red')
            ax.set_title(f"Comparison: {indicator_1} vs {indicator_2} Over Years")
            ax.set_xlabel("Year")
            ax.set_ylabel("Value")
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.6)
            st.pyplot(fig)
            plt.close(fig)
        elif indicator_1 == indicator_2:
            st.warning("Please select two different indicators for comparison.")
        else:
            st.info("Select two indicators to compare their trends.")
    else:
        st.info("Not enough unique indicators available for a two-way comparison.")

    st.divider()

    # --- Distribution of Values by Indicator ---
    st.subheader("📦 Distribution of Indicator Values")
    st.markdown("Examine the spread and central tendency of values for different indicators.")

    if not indicator_list:
        st.warning("No indicators available for distribution plots.")
    else:
        dist_indicator = st.selectbox(
            "Select an Indicator for Distribution Plot:",
            indicator_list,
            key='dist_indicator_select'
        )

        if dist_indicator:
            dist_data = df_eda[df_eda['Indicator Name'] == dist_indicator]['Value'].dropna()
            if not dist_data.empty:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.histplot(dist_data, kde=True, ax=ax, bins=15, color='purple')
                ax.set_title(f"Distribution of Values for: {dist_indicator}")
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                st.pyplot(fig)
                plt.close(fig)

                st.markdown(f"**Box Plot for {dist_indicator}**")
                fig_box, ax_box = plt.subplots(figsize=(10, 2))
                sns.boxplot(x=dist_data, ax=ax_box, color='lightgreen')
                ax_box.set_xlabel("Value")
                ax_box.set_title(f"Box Plot of Values for: {dist_indicator}")
                st.pyplot(fig_box)
                plt.close(fig_box)
            else:
                st.info("No data available for the selected indicator's distribution plot.")
        else:
            st.info("Select an indicator to see its distribution.")
    st.divider()

    # --- Correlation Heatmap (More complex for this data structure) ---
    st.subheader("🔗 Correlation Between Indicators (Experimental)")
    st.markdown("""
        This heatmap attempts to show correlations between *different indicators*.
        It requires pivoting the data, which might result in many missing values if indicators
        don't share common years or countries, potentially limiting its usefulness for sparse datasets.
    """)

    # Filter for Nepal data for a more meaningful correlation if the dataset spans multiple countries
    # Assuming 'Country Name' is 'Nepal' after initial filter or if you filter it here
    df_nepal = df_eda[df_eda['Country Name'] == 'Nepal'].copy()

    # Ensure there's enough data for correlation
    if not df_nepal.empty and df_nepal['Indicator Name'].nunique() > 1:
        # Pivot the data to have indicators as columns and (Year, Country) as index
        # This will create NaNs where an indicator doesn't have data for a specific year/country
        pivot_df = df_nepal.pivot_table(index='Year', columns='Indicator Name', values='Value')

        # Drop columns (indicators) that have too many missing values after pivoting
        # For example, keep only indicators with less than 50% missing data
        pivot_df_cleaned = pivot_df.dropna(axis=1, thresh=len(pivot_df)*0.5)

        if not pivot_df_cleaned.empty and pivot_df_cleaned.shape[1] > 1: # Need at least 2 columns for correlation
            st.write(f"Correlation matrix for {pivot_df_cleaned.shape[1]} indicators with sufficient data in Nepal.")
            st.dataframe(pivot_df_cleaned.head()) # Show pivoted data
            correlation_matrix = pivot_df_cleaned.corr()

            fig, ax = plt.subplots(figsize=(12, 10))
            # Use mask for upper triangle to avoid redundancy
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="viridis",
                        linewidths=.5, linecolor='black', ax=ax, mask=mask,
                        cbar_kws={'label': 'Correlation Coefficient'})
            ax.set_title("Correlation Heatmap of Agricultural Indicators (Nepal)")
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.warning("Not enough indicators with sufficient overlapping data for a meaningful correlation heatmap after cleaning.")
    else:
        st.info("To generate a correlation heatmap, ensure data for 'Nepal' is available and there are multiple unique indicators.")

    st.divider()

    st.markdown("""
        ### 🎯 Key Takeaways from Agricultural EDA:
        - **Time Series Trends**: Observe the evolution of key agricultural metrics over the years.
        - **Comparative Analysis**: Understand how different indicators relate to each other over time.
        - **Data Distributions**: Gain insights into the spread and common values of various agricultural statistics.
        - **Correlation Insights**: Explore potential relationships between different agricultural indicators.
    """)

# This block allows you to run the page directly for testing purposes
if __name__ == "__main__":
    # Simulate session state for local testing
    if 'agriculture_raw_data' not in st.session_state:
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
        # Add some NaNs for testing dropna
        dummy_data['Value'][5:7] = np.nan
        dummy_data['Year'][12:14] = np.nan
        dummy_data['Indicator Name'][20:22] = np.nan


        st.session_state.agriculture_raw_data = pd.DataFrame(dummy_data)
        st.info("Dummy agricultural data loaded for direct testing of EDA page.")
    display()