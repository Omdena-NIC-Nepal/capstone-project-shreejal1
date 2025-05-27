import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os # For robust path handling

def display():
    st.title("Climate Overview")
    st.markdown("""
        In this section, we will explore the climate data for Nepal. The focus is on temperature, precipitation,
        and wind patterns. This page provides an overview of the available climate data and some initial insights.
    """)

    # Define the path to your raw climate data
    # Use os.path.join and os.path.dirname for more robust path handling,
    # especially if your script might be run from different working directories.
    # Assumes 'data' folder is one level up from 'pages/climate_analysis'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    climate_data_path = os.path.join(script_dir, "..", "..", "data", "climate_data", "climate_data_nepal_district_wise_monthly.csv")

    climate_data = None # Initialize to None

    # --- Data Loading and Storing in Session State ---
    st.subheader("Data Loading")
    # Check if data is already in session state to avoid reloading
    if 'climate_raw_data' in st.session_state and st.session_state.climate_raw_data is not None:
        climate_data = st.session_state.climate_raw_data
        st.success("Climate data loaded from session state.")
        st.info("You can refresh the data by clicking the button below if the file has changed.")
    else:
        st.info("Climate data not found in session state. Loading from file...")

    if st.button("Load/Reload Climate Data"):
        try:
            climate_data = pd.read_csv(climate_data_path)
            # It's good practice to make a copy if you intend to modify it later in other pages
            st.session_state.climate_raw_data = climate_data.copy()
            st.success(f"Successfully loaded climate data from `{climate_data_path}` and stored in session state.")
        except FileNotFoundError:
            st.error(f"Error: The file `{climate_data_path}` was not found.")
            st.error("Please ensure 'climate_data_nepal_district_wise_monthly.csv' is in the 'data/climate_data/' directory.")
            climate_data = None # Reset to None if loading failed
        except Exception as e:
            st.error(f"An error occurred while loading the data: {e}")
            climate_data = None # Reset to None if loading failed

    # --- Display Data Overview and Visualizations (only if data is loaded) ---
    if climate_data is not None:
        st.subheader("Data Preview")
        st.write(climate_data.head())
        st.write(f"Dataset shape: {climate_data.shape}")

        st.subheader("Basic Statistics")
        st.write(climate_data.describe())

        # Check for necessary columns before plotting
        required_cols = ['YEAR', 'T2M', 'PRECTOT', 'WS10M']
        if not all(col in climate_data.columns for col in required_cols):
            st.warning(f"Missing one or more required columns for plotting ({', '.join(required_cols)}). Skipping plots.")
        else:
            # Plot temperature trends (average temperature over time)
            st.subheader("Average Temperature Trends")
            # Ensure 'YEAR' is suitable for plotting (e.g., numeric or datetime)
            climate_data['YEAR'] = pd.to_numeric(climate_data['YEAR'], errors='coerce')
            avg_temp = climate_data.groupby("YEAR")["T2M"].mean().dropna() # Drop NaN if year conversion failed
            if not avg_temp.empty:
                fig_temp, ax_temp = plt.subplots(figsize=(10, 6))
                sns.lineplot(x=avg_temp.index, y=avg_temp.values, marker='o', label='Average Temperature', ax=ax_temp)
                ax_temp.set_title("Average Temperature Trends Over Time")
                ax_temp.set_xlabel("Year")
                ax_temp.set_ylabel("Temperature (°C)")
                st.pyplot(fig_temp)
                plt.close(fig_temp) # Close figure to free memory
            else:
                st.info("No valid 'YEAR' data for temperature trend plot.")


            # Plot total precipitation trends over time
            st.subheader("Total Precipitation Trends")
            total_precip = climate_data.groupby("YEAR")["PRECTOT"].sum().dropna()
            if not total_precip.empty:
                fig_precip, ax_precip = plt.subplots(figsize=(10, 6))
                sns.lineplot(x=total_precip.index, y=total_precip.values, marker='o', label='Total Precipitation', ax=ax_precip)
                ax_precip.set_title("Total Precipitation Trends Over Time")
                ax_precip.set_xlabel("Year")
                ax_precip.set_ylabel("Precipitation (mm)")
                st.pyplot(fig_precip)
                plt.close(fig_precip) # Close figure to free memory
            else:
                st.info("No valid 'YEAR' data for precipitation trend plot.")

            # Wind Speed Overview (10m wind speed)
            st.subheader("Wind Speed at 10 Meters")
            avg_wind_speed = climate_data.groupby("YEAR")["WS10M"].mean().dropna()
            if not avg_wind_speed.empty:
                fig_wind, ax_wind = plt.subplots(figsize=(10, 6))
                sns.lineplot(x=avg_wind_speed.index, y=avg_wind_speed.values, marker='o', label='Average Wind Speed (10m)', ax=ax_wind)
                ax_wind.set_title("Average Wind Speed at 10 Meters Over Time")
                ax_wind.set_xlabel("Year")
                ax_wind.set_ylabel("Wind Speed (m/s)")
                st.pyplot(fig_wind)
                plt.close(fig_wind) # Close figure to free memory
            else:
                st.info("No valid 'YEAR' data for wind speed trend plot.")

        # Show any missing values or NaN data in the current state of the DataFrame
        st.subheader("Missing Values Check (Current Data)")
        missing_values = climate_data.isnull().sum()
        if missing_values.sum() > 0:
            st.write(missing_values[missing_values > 0])
            st.warning("Please note: These missing values will be handled in the Feature Engineering step.")
        else:
            st.success("No missing values found in the loaded data.")

        # Provide insights on the data
        st.markdown("""
            ### Key Observations:
            - **Temperature Trends:** The line plot shows the average temperature over the years. Look for any clear warming or cooling patterns.
            - **Precipitation Trends:** The precipitation plot highlights annual rainfall variations, which can be crucial for agriculture and water resources.
            - **Wind Speed:** Average wind speed trends provide insights into atmospheric conditions and potential for extreme weather.
            - **Missing Data:** Pay attention to the 'Missing Values Check' to understand data completeness.
            
            Proceed to the 'Feature Engineering' page to preprocess this data and create new features for model training.
        """)
    else:
        st.info("Please click 'Load/Reload Climate Data' above to start exploring the climate data.")

# This block is only executed when the script is run directly (for testing purposes), not when imported by app.py
if __name__ == "__main__":
    # Simulate session state for local testing
    if 'climate_raw_data' not in st.session_state:
        st.session_state.climate_raw_data = None
    display()