import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Title and description for the EDA page
st.title("Climate Data Exploratory Analysis")
st.markdown("""
    In this section, we will conduct an in-depth exploratory data analysis (EDA) on the climate data.
    This includes visualizing the relationships between different climate variables, trends, and exploring
    the data by district and season.
""")

# Load the climate data
climate_data_path = "data/climate_data/climate_data_nepal_district_wise_monthly.csv"
climate_data = pd.read_csv(climate_data_path)

# Check for missing values
missing_values = climate_data.isnull().sum()
st.subheader("Missing Values")
st.write(missing_values[missing_values > 0])

# Display basic statistics
st.subheader("Basic Statistics")
st.write(climate_data.describe())

# Correlation heatmap for climate variables
st.subheader("Correlation Heatmap")
corr = climate_data[['T2M', 'PRECTOT', 'RH2M', 'WS10M', 'T2M_MAX', 'T2M_MIN']].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("Correlation Between Climate Variables")
st.pyplot(plt)

# Scatter plot: Temperature vs. Precipitation
st.subheader("Temperature vs. Precipitation")
plt.figure(figsize=(10, 6))
sns.scatterplot(x='T2M', y='PRECTOT', data=climate_data, hue='YEAR', palette='viridis')
plt.title("Temperature vs. Precipitation")
plt.xlabel("Temperature (°C)")
plt.ylabel("Precipitation (mm)")
st.pyplot(plt)

# Box plot of temperature by district
st.subheader("Temperature Distribution by District")
plt.figure(figsize=(12, 8))
sns.boxplot(x='DISTRICT', y='T2M', data=climate_data)
plt.title("Temperature Distribution by District")
plt.xticks(rotation=90)
plt.xlabel("District")
plt.ylabel("Temperature (°C)")
st.pyplot(plt)

# Analyze seasonal temperature variation (assuming month-wise data)
st.subheader("Seasonal Temperature Variation")
monthly_avg_temp = climate_data.groupby("MONTH")['T2M'].mean()
plt.figure(figsize=(10, 6))
sns.lineplot(x=monthly_avg_temp.index, y=monthly_avg_temp.values, marker='o')
plt.title("Monthly Average Temperature")
plt.xlabel("Month")
plt.ylabel("Average Temperature (°C)")
st.pyplot(plt)

# Analyze seasonal precipitation variation
st.subheader("Seasonal Precipitation Variation")
monthly_precip = climate_data.groupby("MONTH")['PRECTOT'].mean()
plt.figure(figsize=(10, 6))
sns.lineplot(x=monthly_precip.index, y=monthly_precip.values, marker='o')
plt.title("Monthly Average Precipitation")
plt.xlabel("Month")
plt.ylabel("Average Precipitation (mm)")
st.pyplot(plt)

# Seasonal wind speed variation
st.subheader("Seasonal Wind Speed Variation")
monthly_wind = climate_data.groupby("MONTH")['WS10M'].mean()
plt.figure(figsize=(10, 6))
sns.lineplot(x=monthly_wind.index, y=monthly_wind.values, marker='o')
plt.title("Monthly Average Wind Speed")
plt.xlabel("Month")
plt.ylabel("Average Wind Speed (m/s)")
st.pyplot(plt)

# Display the top 5 districts with the highest average temperature
st.subheader("Top 5 Districts with Highest Average Temperature")
district_temp = climate_data.groupby("DISTRICT")['T2M'].mean().sort_values(ascending=False)
st.write(district_temp.head(5))