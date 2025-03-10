import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt 


def visualize_excel_data(xlsx_file):
    data = pd.read_excel(xlsx_file)
    st.write("Data from uploaded Excel file:")
    st.write(data)  

def forecast_and_visualize(item_data, item):
    st.write("Input data for forecasting:")
    st.write(item_data)
    
    # Check if item_data has enough valid rows
    if item_data.shape[0] < 2:
        st.write("Not enough data available for forecasting.")
        return
    
    item_data = item_data.rename(columns={'Date': 'ds', 'Total (₹)': 'y'})
    
    st.write("Data after column renaming:")
    st.write(item_data)
    
    # Initialize Prophet model
    m = Prophet()
    
    # Fit the model with item-specific data
    m.fit(item_data)
    
    # Create a DataFrame for future dates
    future = m.make_future_dataframe(periods=7*52, freq='W')  # Forecast for upcoming 7 days
    
    # Make predictions
    forecast = m.predict(future)
    
    # Display forecast
    st.write("Forecast for", item)
    st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7))  # Display only the last 7 days forecast
    
    # Visualize the forecast
    st.write("### Forecast Plot")
    st.write("The graph below shows the forecast for the upcoming 7 days.")
    fig = m.plot_components(forecast)
    plt.title("Forecast for " + item)
    plt.xlabel("Date")
    plt.ylabel("Total Sales (₹)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

# Streamlit web app
def main():
    st.title("Restaurant Sales Prediction")
    st.write("Upload an Excel file containing past sales data:")
    xlsx_file = st.file_uploader("Upload Excel (xlsx)", type=['xlsx'])

    if xlsx_file is not None:
        st.write("Selected Excel file:")
        st.write(xlsx_file.name)
        
        # Visualize data from uploaded Excel file
        visualize_excel_data(xlsx_file)
        
        # Read data from uploaded Excel file
        data = pd.read_excel(xlsx_file)
        
        # Assuming 'Item' column contains the names of items sold
        items = data['Item'].unique()

        selected_item = st.selectbox("Select an item for forecasting", items)

        # Filter data for the selected item
        item_data = data[data['Item'] == selected_item]
        
        # Perform forecasting and visualization
        forecast_and_visualize(item_data, selected_item)

if __name__ == "__main__":
    main()
