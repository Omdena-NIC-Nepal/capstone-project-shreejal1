import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os # For robust path handling, though less critical if always loading from session_state

def display():
    st.title("Climate Data Exploratory Analysis")
    st.markdown("""
        In this section, we will conduct an in-depth exploratory data analysis (EDA) on the climate data.
        This includes visualizing the relationships between different climate variables, trends, and exploring
        the data by district and season.
    """)

    # --- Load data from session state ---
    climate_data = None
    if 'climate_raw_data' in st.session_state and st.session_state.climate_raw_data is not None:
        climate_data = st.session_state.climate_raw_data.copy() # Work on a copy to avoid accidental modification
        st.success("Climate data loaded successfully from session state.")
    else:
        st.warning("Climate data not found in session state. Please go to the 'Climate Overview' page and load the data first.")
        # Stop execution of this page if data isn't available
        return

    # Ensure 'YEAR' and 'MONTH' are numeric for grouping/plotting
    # Coerce errors will turn non-numeric values into NaN, which can then be dropped or handled.
    if 'YEAR' in climate_data.columns:
        climate_data['YEAR'] = pd.to_numeric(climate_data['YEAR'], errors='coerce')
    if 'MONTH' in climate_data.columns:
        climate_data['MONTH'] = pd.to_numeric(climate_data['MONTH'], errors='coerce')

    # Drop rows with NaN values in critical columns for plotting to avoid errors
    # You might want to be more specific here depending on your data quality.
    initial_rows = climate_data.shape[0]
    climate_data_cleaned_for_eda = climate_data.dropna(subset=['T2M', 'PRECTOT', 'RH2M', 'WS10M', 'T2M_MAX', 'T2M_MIN', 'DISTRICT', 'YEAR', 'MONTH'])
    if initial_rows > climate_data_cleaned_for_eda.shape[0]:
        st.info(f"Dropped {initial_rows - climate_data_cleaned_for_eda.shape[0]} rows with NaNs for EDA purposes.")
    else:
        st.info("No rows with critical NaNs dropped for EDA.")


    # Check for missing values
    st.subheader("Missing Values (in Data used for EDA)")
    missing_values = climate_data_cleaned_for_eda.isnull().sum()
    if missing_values.sum() > 0:
        st.write(missing_values[missing_values > 0])
    else:
        st.success("No missing values in the data being used for EDA plots.")

    # Display basic statistics
    st.subheader("Basic Statistics")
    st.write(climate_data_cleaned_for_eda.describe())

    # --- Plotting Section ---

    # Correlation heatmap for climate variables
    st.subheader("Correlation Heatmap")
    # Define columns to include in correlation matrix, check if they exist
    correlation_cols = ['T2M', 'PRECTOT', 'RH2M', 'WS10M']
    available_corr_cols = [col for col in correlation_cols if col in climate_data_cleaned_for_eda.columns]

    if 'T2M_MAX' in climate_data_cleaned_for_eda.columns:
        available_corr_cols.append('T2M_MAX')
    if 'T2M_MIN' in climate_data_cleaned_for_eda.columns:
        available_corr_cols.append('T2M_MIN')

    if len(available_corr_cols) >= 2: # Need at least two columns for a correlation matrix
        try:
            corr = climate_data_cleaned_for_eda[available_corr_cols].corr()
            fig1, ax1 = plt.subplots(figsize=(10, 6))
            sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, ax=ax1)
            ax1.set_title("Correlation Between Climate Variables")
            st.pyplot(fig1)
            plt.close(fig1)
        except Exception as e:
            st.error(f"Error generating correlation heatmap: {e}")
            st.info(f"Columns attempted: {available_corr_cols}")
    else:
        st.warning("Not enough numeric columns available to generate a correlation heatmap.")


    # Scatter plot: Temperature vs. Precipitation
    st.subheader("Temperature vs. Precipitation")
    if 'T2M' in climate_data_cleaned_for_eda.columns and 'PRECTOT' in climate_data_cleaned_for_eda.columns and 'YEAR' in climate_data_cleaned_for_eda.columns:
        try:
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            sns.scatterplot(x='T2M', y='PRECTOT', data=climate_data_cleaned_for_eda, hue='YEAR', palette='viridis', ax=ax2)
            ax2.set_title("Temperature vs. Precipitation")
            ax2.set_xlabel("Temperature (°C)")
            ax2.set_ylabel("Precipitation (mm)")
            st.pyplot(fig2)
            plt.close(fig2)
        except Exception as e:
            st.error(f"Error generating Temperature vs. Precipitation scatter plot: {e}")
    else:
        st.warning("Missing 'T2M', 'PRECTOT', or 'YEAR' columns for Temperature vs. Precipitation plot.")

    # Box plot of temperature by district
    st.subheader("Temperature Distribution by District")
    if 'DISTRICT' in climate_data_cleaned_for_eda.columns and 'T2M' in climate_data_cleaned_for_eda.columns:
        try:
            fig3, ax3 = plt.subplots(figsize=(12, 8))
            sns.boxplot(x='DISTRICT', y='T2M', data=climate_data_cleaned_for_eda, ax=ax3)
            ax3.set_title("Temperature Distribution by District")
            ax3.set_xticks(range(len(climate_data_cleaned_for_eda['DISTRICT'].unique()))) # Set ticks based on unique districts
            ax3.set_xticklabels(climate_data_cleaned_for_eda['DISTRICT'].unique(), rotation=90)
            ax3.set_xlabel("District")
            ax3.set_ylabel("Temperature (°C)")
            plt.tight_layout() # Adjust layout to prevent labels from overlapping
            st.pyplot(fig3)
            plt.close(fig3)
        except Exception as e:
            st.error(f"Error generating Temperature Distribution by District box plot: {e}")
    else:
        st.warning("Missing 'DISTRICT' or 'T2M' columns for Temperature Distribution by District plot.")

    # Analyze seasonal temperature variation (assuming month-wise data)
    st.subheader("Seasonal Temperature Variation")
    if 'MONTH' in climate_data_cleaned_for_eda.columns and 'T2M' in climate_data_cleaned_for_eda.columns:
        try:
            monthly_avg_temp = climate_data_cleaned_for_eda.groupby("MONTH")['T2M'].mean().reset_index()
            fig4, ax4 = plt.subplots(figsize=(10, 6))
            sns.lineplot(x='MONTH', y='T2M', data=monthly_avg_temp, marker='o', ax=ax4)
            ax4.set_title("Monthly Average Temperature")
            ax4.set_xlabel("Month")
            ax4.set_ylabel("Average Temperature (°C)")
            st.pyplot(fig4)
            plt.close(fig4)
        except Exception as e:
            st.error(f"Error generating Seasonal Temperature Variation plot: {e}")
    else:
        st.warning("Missing 'MONTH' or 'T2M' columns for Seasonal Temperature Variation plot.")


    # Analyze seasonal precipitation variation
    st.subheader("Seasonal Precipitation Variation")
    if 'MONTH' in climate_data_cleaned_for_eda.columns and 'PRECTOT' in climate_data_cleaned_for_eda.columns:
        try:
            monthly_precip = climate_data_cleaned_for_eda.groupby("MONTH")['PRECTOT'].mean().reset_index()
            fig5, ax5 = plt.subplots(figsize=(10, 6))
            sns.lineplot(x='MONTH', y='PRECTOT', data=monthly_precip, marker='o', ax=ax5)
            ax5.set_title("Monthly Average Precipitation")
            ax5.set_xlabel("Month")
            ax5.set_ylabel("Average Precipitation (mm)")
            st.pyplot(fig5)
            plt.close(fig5)
        except Exception as e:
            st.error(f"Error generating Seasonal Precipitation Variation plot: {e}")
    else:
        st.warning("Missing 'MONTH' or 'PRECTOT' columns for Seasonal Precipitation Variation plot.")

    # Seasonal wind speed variation
    st.subheader("Seasonal Wind Speed Variation")
    if 'MONTH' in climate_data_cleaned_for_eda.columns and 'WS10M' in climate_data_cleaned_for_eda.columns:
        try:
            monthly_wind = climate_data_cleaned_for_eda.groupby("MONTH")['WS10M'].mean().reset_index()
            fig6, ax6 = plt.subplots(figsize=(10, 6))
            sns.lineplot(x='MONTH', y='WS10M', data=monthly_wind, marker='o', ax=ax6)
            ax6.set_title("Monthly Average Wind Speed")
            ax6.set_xlabel("Month")
            ax6.set_ylabel("Average Wind Speed (m/s)")
            st.pyplot(fig6)
            plt.close(fig6)
        except Exception as e:
            st.error(f"Error generating Seasonal Wind Speed Variation plot: {e}")
    else:
        st.warning("Missing 'MONTH' or 'WS10M' columns for Seasonal Wind Speed Variation plot.")


    # Display the top 5 districts with the highest average temperature
    st.subheader("Top 5 Districts with Highest Average Temperature")
    if 'DISTRICT' in climate_data_cleaned_for_eda.columns and 'T2M' in climate_data_cleaned_for_eda.columns:
        try:
            district_temp = climate_data_cleaned_for_eda.groupby("DISTRICT")['T2M'].mean().sort_values(ascending=False)
            st.write(district_temp.head(5))
        except Exception as e:
            st.error(f"Error calculating Top 5 Districts by Temperature: {e}")
    else:
        st.warning("Missing 'DISTRICT' or 'T2M' columns for Top 5 Districts calculation.")

    # Provide insights on the data
    st.markdown("""
        ### Insights from EDA:
        - The correlation heatmap shows relationships between different climate variables.
        - Scatter plots help visualize how temperature and precipitation might interact.
        - Box plots reveal the distribution of variables across different districts.
        - Line plots for monthly averages illustrate seasonal patterns.
        - The list of top districts by temperature highlights regional climate differences.
    """)

# This block is only executed when the script is run directly (for testing purposes)
if __name__ == "__main__":
    # Simulate session state for local testing if climate_raw_data is not set
    if 'climate_raw_data' not in st.session_state:
        # Load directly if testing this page in isolation
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            climate_data_path = os.path.join(script_dir, "..", "..", "data", "climate_data", "climate_data_nepal_district_wise_monthly.csv")
            st.session_state.climate_raw_data = pd.read_csv(climate_data_path).copy()
            st.info("Data loaded directly for testing EDA page.")
        except FileNotFoundError:
            st.error(f"Could not load data for direct testing from {climate_data_path}. Please ensure the file exists.")
            st.session_state.climate_raw_data = None
    display()