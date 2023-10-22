import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import yfinance as yf

# Set the page title
st.title("Stock Price Comparison App")

# Get user input for the stock symbol and date range
option = st.sidebar.selectbox('Select one symbol', ('AAPL', 'MSFT', 'SPY', 'WMT', 'GME', 'MU', 'NFLX', 'BNOX'))
start_date = st.sidebar.date_input('Start date')
end_date = st.sidebar.date_input('End date')

# Fetch the stock data using yfinance
stock_data = yf.download(option, start=start_date, end=end_date)

# Calculate the previous close
previous_close = stock_data['Close'].shift(1)

# Check if there is data in the DataFrame
if not stock_data.empty:
    # Calculate the current price if there is data
    current_price = stock_data['Close'].iloc[-1]

    # Calculate the predicted price (replace this with your prediction logic)
    prediction_price = current_price + 5
else:
    # Handle the case when there's no data
    current_price = None
    prediction_price = None

# Display the open, close, and previous close values
st.subheader("Stock Price Comparison")

# Plotting the data using Matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(12, 6))

# Plot the close price
ax.plot(stock_data.index, stock_data['Close'], label='Close Price', color='yellow', marker='o')

# Plot the open price
ax.plot(stock_data.index, stock_data['Open'], label='Open Price', color='orange', marker='o')

# Plot the previous close price
ax.plot(stock_data.index, previous_close, label='Previous Close', color='purple', marker='o')

# Plot the current price if available
if current_price is not None:
    ax.plot(stock_data.index[-1], current_price, 'bo', label='Current Price', color='red')

# Plot the predicted price if available
if prediction_price is not None:
    ax.plot(stock_data.index[-1], prediction_price, 'go', label='Predicted Price', color='green')

# Add labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.set_title(f'Stock Price Comparison for {option}')
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

# Display legend
plt.legend()

# Display the graph
st.pyplot(fig)



fig, ax = plt.subplots(figsize=(12, 6))

# Plot the open price in green
ax.plot(stock_data.index, stock_data['Open'], label='Open Price', color='g', marker='o')

# Plot the close price in blue
ax.plot(stock_data.index, stock_data['Close'], label='Close Price', color='b', marker='o')

# Add labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.set_title(f'Open and Close Price Comparison for {option}')
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

# Display legend
plt.legend()

# Display the graph
st.pyplot(fig)


fig, ax = plt.subplots(figsize=(12, 6))

# Plot the actual price with colors
ax.plot(stock_data.index, stock_data['Close'], label='Actual Price', color='blue', marker='o')

# Add labels and title
ax.set_xlabel('Date')
ax.set_ylabel('Price (USD)')
ax.set_title(f'Actual Price Comparison for {option}')
ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

# Display legend
plt.legend()

# Display the graph
st.pyplot(fig)

# Fetch the stock data using yfinance
stock_data = yf.download(option, start='2010-01-01', end='2023-10-06')

# Prepare the data for training
stock_data['Date'] = stock_data.index
stock_data['Date'] = stock_data['Date'].apply(lambda x: x.toordinal())
X = stock_data[['Date']]
y = stock_data['Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
prediction_days = 45  # Set the number of days for prediction
prediction_dates = pd.date_range(start=stock_data.index[-1], periods=prediction_days + 1, closed='right')
prediction_dates = prediction_dates[1:]  # Exclude the last date from the range
prediction_dates_ord = prediction_dates.to_series().apply(lambda x: x.toordinal())
predictions = model.predict(prediction_dates_ord.values.reshape(-1, 1))

# Display the predicted stock prices
st.subheader("Predicted Stock Prices:")
predicted_data = pd.DataFrame({'Date': prediction_dates, 'Close': predictions})
st.line_chart(predicted_data.set_index('Date'))

# Display the actual and predicted stock prices together
st.subheader("Actual vs. Predicted Stock Prices:")
combined_data = pd.concat([stock_data['Close'], predicted_data.set_index('Date')['Close']], axis=1)
combined_data.columns = ['Actual', 'Predicted']
st.line_chart(combined_data)

# Add the 'Open' column to the stock_data DataFrame
stock_data['Open'] = stock_data['Open']

# Fetch the current stock price using yfinance
current_price = yf.Ticker(option).history(period='1d')['Close'].iloc[-1]

# Calculate the previous close
previous_close = stock_data['Close'].iloc[-1]

# Fetch the open price for today
open_price = stock_data['Open'].iloc[-1]

# Fetch the close price for today
close_price = stock_data['Close'].iloc[-1]

# Fetch the current stock price using yfinance
current_price = yf.Ticker(option).history(period='1d')['Close'].iloc[0]

# Display the open, close, and previous close values
st.subheader("stock price comparision")
open_close_previous_data = stock_data[['Open', 'Close']]
previous_close = stock_data['Close'].iloc[-1]
st.write(f"Open: {stock_data['Open'].iloc[-1]:.2f}")
st.write(f"Close: {stock_data['Close'].iloc[-1]:.2f}")
st.write(f"Today's Price: {current_price:.2f}")
st.write(f"Predicted Price: {predictions[-1]:.2f}")

# Determine if the price went up, down, or stayed the same
if current_price > previous_close:
    st.markdown(f"Price Up! Current Price: {current_price} (Previous Close: {previous_close})")
elif current_price < previous_close:
    st.markdown(f"Price Down! Current Price: {current_price} (Previous Close: {previous_close})")
else:
    st.markdown(f"Price Unchanged! Current Price: {current_price} (Previous Close: {previous_close})")

# Determine if the predicted price is higher, lower, or the same as today's price
if predictions[-1] > current_price:
    st.markdown(f"Predicted Price is Higher by ${predictions[-1] - current_price:.2f} compared to Today's Price")
elif predictions[-1] < current_price:
    st.markdown(f"Predicted Price is Lower by ${current_price - predictions[-1]:.2f} compared to Today's Price")
else:
    st.markdown("Predicted Price is the Same as Today's Price")




