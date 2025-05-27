import streamlit as st

try:

    from scripts.data_utils import load_data, remove_unwanted_columns, clean_data
    from scripts.visualization import plot_time_series, plot_time_series_double, plot_correlation_heatmap, other_visualize

except ImportError as e:
    st.error(f"Failed to import modules: {e}")
    st.stop()

def display():
    st.write("""
    This section includes the exploration of climate data overview for Nepal.
    """)
    
    # Data preview
    
    if "cleaned_climatedf" in st.session_state:
        cleaned_climatedf = st.session_state.cleaned_climatedf

        st.subheader("Data Preview")
        st.dataframe(cleaned_climatedf, height=200)

        st.divider()
        
        st.subheader("DataFrame Info")
        st.write(f"Data shape: {cleaned_climatedf.shape[0]} Rows, {cleaned_climatedf.shape[1]} Columns")

        # Grid layout: Three columns
        col1, col2 = st.columns(2)

        
        with col1:
            st.write("Data types:", cleaned_climatedf.dtypes)

        with col2:
            st.write("Missing values:", cleaned_climatedf.isnull().sum())



        st.divider()
        st.subheader("DataFrame Statistical Description")
        st.dataframe(cleaned_climatedf.describe())
        st.divider()     
        
        st.subheader("Time Series")

        # Choose numeric columns (excluding the datetime column)
        value_columns = cleaned_climatedf.select_dtypes(include=['number']).columns


        # Let user select which column to plot
        selected_column = st.selectbox("Select a value column to plot:", value_columns, index= 3)


        fig = plot_time_series(
            cleaned_climatedf,
            x_axis="YEAR",
            y_axis=selected_column,
            figsize=(10, 5)
        )

        st.pyplot(fig)

        st.divider()

        st.subheader("Time Series Comparison")
        st.write("Select two columns to compare their time series.")
        # Grid layout
        column1, column2 = st.columns(2)

        
        with column1:
            selected_column1 = st.selectbox( "",value_columns, index= 3)

        with column2:
            selected_column2 = st.selectbox("", value_columns, index= 8)
            

        st.write(f"Comparing {selected_column1} and {selected_column2} over time")

        plot = plot_time_series_double(
        cleaned_climatedf,
        col1=selected_column1,
        col2=selected_column2,
        x_axis="YEAR",
        figsize=(10, 5)
    )
        st.pyplot(plot)
    
    

        st.divider()

        st.subheader("Correlation Heatmap")
        st.write("This heatmap shows the correlation between different weather variables.")


        heatmap = plot_correlation_heatmap(
            cleaned_climatedf.drop(columns=["DATE", "YEAR", "MONTH", 'DISTRICT']),
            )


        st.pyplot(heatmap)

        st.divider()


        st.subheader("Advanced Climate EDA")

        colu1 , colu2, colu3 = st.columns(3)

        with colu1:
            plot_type = st.selectbox("Plot Type", ["box", "dist", "scatter", "bar", "pairplot"])
            group_by = st.selectbox("Group By (for bar)",list(cleaned_climatedf.columns), index = 4)
        with colu2:
            x_col = st.selectbox("X-Axis", cleaned_climatedf.columns, index = 5)
            cols = st.multiselect("Columns (for pairplot)", cleaned_climatedf.columns, default=[cleaned_climatedf.columns[5]] )
        with colu3:
            y_col = st.selectbox("Y-Axis", cleaned_climatedf.columns, index = 6)

        fig = other_visualize(
            cleaned_climatedf,
            plot_type=plot_type,
            x=x_col,
            y=y_col,
            group_by=group_by,
            cols=cols,
            title=f"{plot_type.capitalize()} of {y_col} vs {x_col}",
        )

        st.plotly_chart(fig) if 'plotly' in str(type(fig)).lower() else st.pyplot(fig)

    else:
        st.error("Data not loaded. Please go to the overview and load again.")