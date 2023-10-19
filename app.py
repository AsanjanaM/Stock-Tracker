import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split  # Import train_test_split from scikit-learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import yfinance as yf
from datetime import date  

# Set the page title
st.title("Stock Price Prediction App")

# Get user input for the stock symbol
option = st.sidebar.selectbox('Select one symbol', ('AAPL', 'MSFT', 'SPY', 'WMT', 'GME', 'MU', 'NFLX', 'BNOX','TSLA','LRCX'))

# Calculate the start date from 2010
#start_date = st.sidebar.date_input('Start Date', date(2010, 1, 1))

# Calculate today's date
end_date = date.today()


import datetime

today = datetime.date.today()
before = today - datetime.timedelta(days=8395)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)


stock_data = yf.download(option, start='2010-01-01', end='2023-10-06')
# Calculate today's date
end_date = date.today()



# Fetch the stock data using yfinance
stock_data = yf.download(option, start=start_date, end=end_date)

# Prepare the data for training
stock_data['Date'] = stock_data.index
stock_data['Date'] = stock_data['Date'].apply(lambda x: x.toordinal())
X = stock_data[['Date']]
y = stock_data['Close']

# Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
prediction_days = 45  # Set the number of days for prediction
prediction_dates = pd.date_range(start=stock_data.index[-1], periods=prediction_days + 1, closed='right')
prediction_dates = prediction_dates[1:]  # Exclude the last date from the range
prediction_dates_ord = prediction_dates.to_series().apply(lambda x: x.toordinal())
predictions = model.predict(prediction_dates_ord.values.reshape(-1, 1))

# Display the actual stock prices
st.subheader("Today's Stock Prices:")
st.line_chart(stock_data['Close'])

# Display the predicted stock prices
st.subheader("Predicted Stock Prices:")
predicted_data = pd.DataFrame({'Date': prediction_dates, 'Close': predictions})
st.line_chart(predicted_data.set_index('Date'))

# Display the actual and predicted stock prices together
st.subheader("Today's vs. Predicted Stock Prices:")
combined_data = pd.concat([stock_data['Close'], predicted_data.set_index('Date')['Close']], axis=1)
combined_data.columns = ['Today', 'Predicted']
st.line_chart(combined_data)

# Add the 'Open' column to the stock_data DataFrame
stock_data['Open'] = stock_data['Open']

# Display the open and close values
st.subheader("Open and Close Values:")
open_close_data = pd.concat([stock_data['Open'], stock_data['Close']], axis=1)
open_close_data.columns = ['Open', 'Close']
st.line_chart(open_close_data)

# Display the open, close, and date values in a DataFrame
st.subheader("Open, Close:")
open_close_date_data = stock_data[['Open', 'Close']]
st.dataframe(open_close_date_data, use_container_width=True)


# Fetch the current stock price using yfinance
current_price = yf.Ticker(option).history(period='1d')['Close'].iloc[-1]

# Display the current stock price
st.subheader("Current Stock Price:")
st.write(current_price)




# Fetch the current stock price using yfinance
current_price = yf.Ticker(option).history(period='1d')['Close'].iloc[-1]

# Display the current stock price
st.subheader("Current Stock Price:")
st.write(current_price)

# Check if the current price is up or down compared to the previous closing price
previous_close = stock_data['Close'].iloc[-1]
if current_price > previous_close:
    st.markdown(f"Price Up! Current Price: {current_price} (Previous Close: {previous_close})")
elif current_price < previous_close:
    st.markdown(f"Price Down! Current Price: {current_price} (Previous Close: {previous_close})")
else:
    st.markdown(f"Price Unchanged! Current Price: {current_price} (Previous Close: {previous_close})")
