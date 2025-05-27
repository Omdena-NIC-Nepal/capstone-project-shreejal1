import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def display():
    st.write("""
    This section includes the feature engineering of climate data overview for Nepal.
    """)

    if "cleaned_climatedf" not in st.session_state:
        st.error("Data not loaded. Please go to the overview and load again.")
        return

    cleaned_climatedf = st.session_state.cleaned_climatedf

    # Example Feature: Temperature Range (if columns exist)
    if 'T2M_MAX' in cleaned_climatedf.columns and 'T2M_MIN' in cleaned_climatedf.columns:
        cleaned_climatedf['temp_range_2m'] = cleaned_climatedf['T2M_MAX'] - cleaned_climatedf['T2M_MIN']
        st.success("Created new feature: Temperature Range (T2M_MAX - T2M_MIN)")
        st.dataframe(cleaned_climatedf[['T2M_MAX', 'T2M_MIN', 'temp_range_2m']].head())
    else:
        st.warning("No 'T2M_MAX' and 'T2M_MIN' columns found to create Temperature Range.")

    st.divider()

    # Plotting Section
    st.subheader("📊 Visualize a Feature")

    column_to_plot = st.selectbox("Select a column to plot:", cleaned_climatedf.columns, index=3)

    fig, ax = plt.subplots()

    if pd.api.types.is_numeric_dtype(cleaned_climatedf[column_to_plot]):
        cleaned_climatedf[column_to_plot].hist(ax=ax, bins=30, color='skyblue', edgecolor='black')
        ax.set_title(f"Histogram of {column_to_plot}")
        ax.set_xlabel(column_to_plot)
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    elif pd.api.types.is_datetime64_any_dtype(cleaned_climatedf[column_to_plot]):
        # Time series line plot
        numeric_cols = cleaned_climatedf.select_dtypes(include='number').columns
        time_value = st.selectbox("Select a numeric column to plot over time:", numeric_cols)
        fig, ax = plt.subplots()
        ax.plot(cleaned_climatedf[column_to_plot], cleaned_climatedf[time_value], marker='o', linestyle='-')
        ax.set_title(f"{time_value} over {column_to_plot}")
        ax.set_xlabel("Date")
        ax.set_ylabel(time_value)
        st.pyplot(fig)

    elif pd.api.types.is_object_dtype(cleaned_climatedf[column_to_plot]):
        # Categorical bar plot
        value_counts = cleaned_climatedf[column_to_plot].value_counts().sort_values(ascending=False)
        value_counts.plot(kind='bar', ax=ax, color='orange')
        ax.set_title(f"Bar Chart of {column_to_plot}")
        ax.set_xlabel(column_to_plot)
        ax.set_ylabel("Count")
        st.pyplot(fig)

    else:
        st.warning("Selected column type not supported for plotting.")

    st.divider()
