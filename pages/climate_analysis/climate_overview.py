import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Title and description for the climate overview page
st.title("Climate Overview")
st.markdown("""
    In this section, we will explore the climate data for Nepal. The focus is on temperature, precipitation,
    and wind patterns. This page provides an overview of the available climate data and some initial insights.
""")

# Load the climate data
climate_data_path = "data/climate_data/climate_data_nepal_district_wise_monthly.csv"
climate_data = pd.read_csv(climate_data_path)

# Show data preview
st.subheader("Data Preview")
st.write(climate_data.head())

# Basic summary statistics
st.subheader("Basic Statistics")
st.write(climate_data.describe())

# Plot temperature trends (average temperature over time)
st.subheader("Average Temperature Trends")
avg_temp = climate_data.groupby("YEAR")["T2M"].mean()
plt.figure(figsize=(10, 6))
sns.lineplot(x=avg_temp.index, y=avg_temp.values, marker='o', label='Average Temperature')
plt.title("Average Temperature Trends Over Time")
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
st.pyplot(plt)

# Plot total precipitation trends over time
st.subheader("Total Precipitation Trends")
total_precip = climate_data.groupby("YEAR")["PRECTOT"].sum()
plt.figure(figsize=(10, 6))
sns.lineplot(x=total_precip.index, y=total_precip.values, marker='o', label='Total Precipitation')
plt.title("Total Precipitation Trends Over Time")
plt.xlabel("Year")
plt.ylabel("Precipitation (mm)")
st.pyplot(plt)

# Wind Speed Overview (10m wind speed)
st.subheader("Wind Speed at 10 Meters")
avg_wind_speed = climate_data.groupby("YEAR")["WS10M"].mean()
plt.figure(figsize=(10, 6))
sns.lineplot(x=avg_wind_speed.index, y=avg_wind_speed.values, marker='o', label='Average Wind Speed (10m)')
plt.title("Average Wind Speed at 10 Meters Over Time")
plt.xlabel("Year")
plt.ylabel("Wind Speed (m/s)")
st.pyplot(plt)

# Show any missing values or NaN data
st.subheader("Missing Values Check")
missing_values = climate_data.isnull().sum()
st.write(missing_values[missing_values > 0])

# Provide insights on the data
st.markdown("""
    - The temperature trends reveal a pattern of warming over the years.
    - Precipitation trends help in understanding the monsoon season behavior.
    - Wind speed analysis can give insights into extreme weather events like storms.
""")
