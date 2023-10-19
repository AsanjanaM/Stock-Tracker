import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import yfinance as yf
from datetime import date, timedelta

# Create a function to calculate and display stock prices and predictions
def display_stock_data():
    st.subheader("Today's Stock Prices:")
    st.line_chart(stock_data['Close'])

    st.subheader("Predicted Stock Prices:")
    predicted_data = pd.DataFrame({'Date': prediction_dates, 'Close': predictions})
    st.line_chart(predicted_data.set_index('Date'))

    st.subheader("Open and Close Values:")
    open_close_data = pd.concat([stock_data['Open'], stock_data['Close']], axis=1)
    open_close_data.columns = ['Open', 'Close']
    st.line_chart(open_close_data)

    st.subheader("Open, Close:")
    open_close_date_data = stock_data[['Open', 'Close']]
    st.dataframe(open_close_date_data, use_container_width=True)

    current_price = yf.Ticker(option).history(period='1d')['Close'].iloc[-1]
    st.subheader("Current Stock Price:")
    st.write(current_price)

    previous_close = stock_data['Close'].iloc[-1]
    if current_price > previous_close:
        st.markdown(f"Price Up! Current Price: {current_price} (Previous Close: {previous_close})")
    elif current_price < previous_close:
        st.markdown(f"Price Down! Current Price: {current_price} (Previous Close: {previous_close})")
    else:
        st.markdown(f"Price Unchanged! Current Price: {current_price} (Previous Close: {previous_close})")

# Set the page title
st.title("Stock Price Prediction")

# Get user input for the stock symbol
option = st.selectbox('Select one symbol', ('AAPL', 'MSFT', 'SPY', 'WMT', 'GME', 'MU', 'NFLX', 'BNOX', 'TSLA', 'LRCX'))

# Calculate the start date from 2010
today = date.today()
before = today - timedelta(days=8395)
start_date = st.date_input('Start date', before)
end_date = st.date_input('End date', today)

# Initialize variables
stock_data = None
predictions = None
prediction_dates = None

if st.button('Submit'):
    # Fetch and process stock data
    stock_data = yf.download(option, start=start_date, end=end_date)
    stock_data['Date'] = stock_data.index
    stock_data['Date'] = stock_data['Date'].apply(lambda x: x.toordinal())
    X = stock_data[['Date']]
    y = stock_data['Close']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    prediction_days = 45
    prediction_dates = pd.date_range(start=stock_data.index[-1], periods=prediction_days + 1, closed='right')
    prediction_dates = prediction_dates[1:]
    prediction_dates_ord = prediction_dates.to_series().apply(lambda x: x.toordinal())
    predictions = model.predict(prediction_dates_ord.values.reshape(-1, 1))

    # Display stock data
    display_stock_data()
