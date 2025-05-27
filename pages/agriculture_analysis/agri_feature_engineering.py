import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def display():
    st.write("""
    ## Feature Engineering - Agricultural and Rural Development Data
    This section performs basic feature engineering to enhance insights from agricultural development indicators.
    """)

    if "agriculture_df" not in st.session_state:
        st.error("Agriculture data not loaded. Please load it from the overview section.")
        return

    df = st.session_state.agriculture_df.copy()

    # Convert relevant columns to numeric if not already
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # Feature 1: Year-wise Value Normalization (Z-score)
    df['Yearly Z-Score'] = df.groupby('Indicator Name')['Value'].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    # Feature 2: Percentage Change in Value from Previous Year (per indicator per country)
    df['Value % Change'] = df.sort_values(['Country Name', 'Indicator Name', 'Year']) \
                            .groupby(['Country Name', 'Indicator Name'])['Value'] \
                             .transform(lambda x: x.pct_change() * 100)

    # Feature 3: Rolling Mean (3-year) for smoothing
    df_sorted = df.sort_values(['Country Name', 'Indicator Name', 'Year'])
    df['3-Year Rolling Mean'] = df_sorted.groupby(['Country Name', 'Indicator Name'])['Value'] \
                                        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())

    # Feature 4: High Growth Flag (>10% YoY increase)
    df['High Growth Flag'] = df['Value % Change'].apply(lambda x: 1 if x > 10 else 0)

    # Feature 5: Normalized Value (0-1 scale) per Indicator
    df['Normalized Value'] = df.groupby('Indicator Name')['Value'].transform(
        lambda x: (x - x.min()) / (x.max() - x.min())
    )

    # Display the first 10 rows of the engineered dataframe
    st.subheader("Engineered Features Preview")
    st.dataframe(df.head(10))

    st.divider()

    # Z-Score Distribution Visualization
    st.subheader("Z-Score Distribution for Selected Indicator")
    indicator_list = df['Indicator Name'].dropna().unique()
    selected_indicator = st.selectbox("Choose an Indicator", sorted(indicator_list), index=0)

    indicator_df = df[df['Indicator Name'] == selected_indicator]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(indicator_df['Yearly Z-Score'].dropna(), bins=20, kde=True, ax=ax)
    ax.set_title(f"Z-Score Distribution for: {selected_indicator}")
    ax.set_xlabel("Z-Score")
    st.pyplot(fig)

    st.divider()

    # Yearly Trend Visualization: Raw Value vs Rolling Mean
    st.subheader("Yearly Trend: Rolling Mean vs Raw Value")
    country = st.selectbox("Select Country", df['Country Name'].unique())
    indicator_data = df[(df['Country Name'] == country) & (df['Indicator Name'] == selected_indicator)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(indicator_data['Year'], indicator_data['Value'], marker='o', label='Raw Value')
    ax.plot(indicator_data['Year'], indicator_data['3-Year Rolling Mean'], marker='x', label='3-Year Rolling Mean')
    ax.set_title(f"{selected_indicator} Over Time ({country})")
    ax.set_xlabel("Year")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

    st.divider()

    # Growth Flag Count Visualization Over Time
    st.subheader("Growth Flag Count Over Time")
    flag_df = indicator_data.groupby('Year')['High Growth Flag'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=flag_df, x='Year', y='High Growth Flag', ax=ax)
    ax.set_title(f"High Growth Flag Count Per Year ({country}, {selected_indicator})")
    st.pyplot(fig)

    st.divider()

    # Store the engineered dataframe back into session state for further use
    st.session_state.engineered_agriculture_df = df
