import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from scripts.visualization import plot_time_series, plot_time_series_double, plot_correlation_heatmap, other_visualize
except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.stop()

def display():
    st.write("""
    ## Agricultural and Rural Development Data Exploration
    This section provides an exploratory data analysis (EDA) of agriculture-related indicators for Nepal and potentially other countries.
    """)

    st.subheader("Data Preview")

    if "agriculture_df" not in st.session_state:
        st.error("Agriculture data not loaded. Please load it in the overview page.")
        return

    df = st.session_state.agriculture_df

    # Ensure correct types
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df['Value'] = pd.to_numeric(df['Value'], errors='coerce')

    # Warn if missing
    missing_values = df['Value'].isnull().sum()
    if missing_values > 0:
        st.warning(f"There are {missing_values} missing values in the 'Value' column. They will be ignored for analysis.")

    # Show preview
    st.dataframe(df, height=250)

    st.divider()

    st.subheader("Data Overview")
    st.write(f"Data shape: {df.shape[0]} rows × {df.shape[1]} columns")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Data types:")
        st.write(df.dtypes)
    with col2:
        st.write("Missing values:")
        st.write(df.isnull().sum())

    st.divider()

    st.subheader("Statistical Summary")
    st.dataframe(df.describe(include='all'))

    st.divider()

    # Time Series Plot Section
    st.subheader("Time Series Plot")

    if 'Year' in df.columns and 'Value' in df.columns:
        indicator_col = st.selectbox("Select Indicator", df['Indicator Name'].dropna().unique())

        indicator_data = df[df['Indicator Name'] == indicator_col].copy()
        indicator_data = indicator_data.dropna(subset=['Year', 'Value'])
        indicator_data = indicator_data.sort_values('Year')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(indicator_data['Year'], indicator_data['Value'], label=indicator_col, color='b')
        ax.set_xlabel('Year')
        ax.set_ylabel('Value')
        ax.set_title(f"Time Series for {indicator_col}")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("No valid 'Year' or 'Value' columns available for time series plot.")

    st.divider()

    # Comparison of Two Time Series
    st.subheader("Compare Two Time Series")

    indicator_list = df['Indicator Name'].dropna().unique()

    if len(indicator_list) >= 2:
        col1, col2 = st.columns(2)
        with col1:
            indicator_1 = st.selectbox("Select first indicator:", indicator_list, index=0)
        with col2:
            indicator_2 = st.selectbox("Select second indicator:", indicator_list, index=1)

        indicator_1_data = df[df['Indicator Name'] == indicator_1].copy()
        indicator_2_data = df[df['Indicator Name'] == indicator_2].copy()

        indicator_1_data = indicator_1_data.dropna(subset=['Year', 'Value']).sort_values('Year')
        indicator_2_data = indicator_2_data.dropna(subset=['Year', 'Value']).sort_values('Year')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(indicator_1_data['Year'], indicator_1_data['Value'], label=indicator_1, color='b')
        ax.plot(indicator_2_data['Year'], indicator_2_data['Value'], label=indicator_2, color='r')
        ax.set_xlabel('Year')
        ax.set_ylabel('Value')
        ax.set_title(f"Comparison of {indicator_1} and {indicator_2}")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Not enough indicators for double time series comparison.")

    st.divider()

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")

    if df['Value'].nunique() > 1:
        pivot_df = df.pivot_table(index=['Country Name', 'Year'], columns='Indicator Name', values='Value')
        correlation_matrix = pivot_df.corr()

        st.write("Correlation heatmap for the numeric indicators across different countries and years.")
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("Not enough data for correlation heatmap.")

    st.divider()

    # Advanced Visualization Section
    st.subheader("Advanced Visualization")

    colu1, colu2, colu3 = st.columns(3)

    with colu1:
        plot_type = st.selectbox("Plot Type", ["box", "dist", "scatter", "pairplot"])
        cols = st.multiselect("Columns (for pairplot)", df.columns.tolist(), default=[df.columns[0]])

    with colu2:
        x_col = st.selectbox("X-Axis", df.columns.tolist(), index=2)
    with colu3:
        y_col = st.selectbox("Y-Axis", df.columns.tolist(), index=3)

    fig = other_visualize(
        df,
        plot_type=plot_type,
        x=x_col,
        y=y_col,
        cols=cols,
        title=f"{plot_type.capitalize()} of {y_col} vs {x_col}",
    )

    st.plotly_chart(fig) if 'plotly' in str(type(fig)).lower() else st.pyplot(fig)

    st.divider()
