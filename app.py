import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import yfinance as yf


# Change the font style and color for the title
st.markdown("<h1 style='color: #D8BFD8; font-family:Gabriola, Times, serif;'>STOCK PRICE PREDICITION</h1>", unsafe_allow_html=True)

# Get user input for the stock symbol and date range
import streamlit as st

# Add custom CSS styles in one line
st.markdown(
    """
    <style>
    .st-cj, .st-cc, .st-db {
        font-family: "Ink Free", Times, serif;
        color: #FFC0CB
    }
    .st-df, .st-cf, .st-eb {
        font-family: "Ink Free", Times, serif;
        color: #FA8072
    }
    </style>
    """,
    unsafe_allow_html=True
)

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
#st.subheader("Stock Price Comparison")

# Plotting the data using Matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

fig, ax = plt.subplots(figsize=(12, 6))

# Plot the close price
ax.plot(stock_data.index, stock_data['Close'], label='Close Price', color='#FFA07A', marker='o')

# Plot the open price
ax.plot(stock_data.index, stock_data['Open'], label='Open Price', color='#FFDAB9', marker='o')

# Plot the previous close price
ax.plot(stock_data.index, previous_close, label='todays price', color='#98FB98', marker='o')

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
stock_data = yf.Ticker(option).history(period="1d")
if not stock_data.empty:
        # Original open, close, and previous close values
        open_price = stock_data['Open'].iloc[-1]
        close_price = stock_data['Close'].iloc[-1]
        # Calculate the current price
        current_price = stock_data['Close'].iloc[0]
        st.markdown(f'<span style="color: #FFFACD;">Open Price: ${open_price:.2f}</span>', unsafe_allow_html=True)
st.markdown(f'<span style="color: #F0E68C;">Close Price: ${close_price:.2f}</span>', unsafe_allow_html=True)
st.markdown(f'<span style="color: #DEB887;">Today\'s Price: ${current_price:.2f}</span>', unsafe_allow_html=True)



#coding for the 2nd graph open and close 
# Fetch the stock data using yfinance
stock_data = yf.download(option, start=start_date, end=end_date)

# Check if there is data in the DataFrame
if not stock_data.empty:
    # Create a figure and plot the data
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot the close price
    ax.plot(stock_data.index, stock_data['Close'], label='Close Price', color='blue', marker='o')

    # Plot the open price
    ax.plot(stock_data.index, stock_data['Open'], label='Open Price', color='green', marker='o')

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
st.markdown(f'<span style="color: #DEB887; font-size: 16px;"><b>Open Price:</b> ${open_price:.2f}</span>', unsafe_allow_html=True)
st.markdown(f'<span style="color: #F4A460; font-size: 16px;"><b>Close Price:</b> ${close_price:.2f}</span>', unsafe_allow_html=True)



 #coding for the 3rd graph for actual price 
# Create a figure and plot the actual price graph
fig, ax = plt.subplots(figsize=(12, 6))

# Plot the actual price with a specific color (e.g., green)
ax.plot(stock_data.index, stock_data['Close'], label='Actual Price', color='#DAA520', marker='o')

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

# Assuming you have the actual price in a variable named actual_price
actual_price = stock_data['Close'].iloc[-1]

# Define a color for the text (e.g., green)
text_color = '#DAA520'
# Display the actual price with color
st.markdown(f'<span style="color: {text_color};">Actual Price: ${actual_price:.2f}</span>', unsafe_allow_html=True)


#graph to display predicit value 
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

 # displayes the currect ,open ,close,previous value 
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

# Determine if the price went up, down, or stayed the same
price_change_message = ""
price_change_color = "da70d6"  # Default color for unchanged price

if current_price > previous_close:
    price_change_message = f"Price Up! Current Price: {current_price} (Previous Close: {previous_close})"
    price_change_color = "green"
elif current_price < previous_close:
    price_change_message = f"Price Down! Current Price: {current_price} (Previous Close: {previous_close})"
    price_change_color = "red"
else:
    price_change_message = f"Price Unchanged! Current Price: {current_price} (Previous Close: {previous_close})"

# Define the style for the boxes
box_style = "border-radius: 15px; padding: 10px; margin: 10px;"

# Display the open, close, and previous close values with curved edges and gaps
#st.subheader("Stock Price Comparison")

def create_colored_box(label, value, bg_color):
    style = f"{box_style} background-color: {bg_color};"
    return f"<div style='{style}'><b>{label}:</b> ${value:.2f}</div>"

st.markdown(create_colored_box("Open Price", open_price, "#d8f5g7h"), unsafe_allow_html=True)
st.markdown(create_colored_box("Close Price", close_price, "#h7d6f4"), unsafe_allow_html=True)
st.markdown(create_colored_box("Today's Price", current_price, price_change_color), unsafe_allow_html=True)
st.markdown(create_colored_box("Predicted Price", predictions[-1], "#j8c0s3"), unsafe_allow_html=True)

# Display the price change message with the appropriate color and style
style = f"{box_style} background-color: {price_change_color};"
st.markdown(f"<div style='{style}'><b>{price_change_message}</b></div>", unsafe_allow_html=True)

# Determine if the predicted price is higher, lower, or the same as today's price
prediction_message = ""
if predictions[-1] > current_price:
    prediction_message = f"Predicted Price is Higher by ${predictions[-1] - current_price:.2f} compared to Today's Price"
elif predictions[-1] < current_price:
    prediction_message = f"Predicted Price is Lower by ${current_price - predictions[-1]:.2f} compared to Today's Price"
else:
    prediction_message = "Predicted Price is the Same as Today's Price"

# Display the prediction message with curved edges and gaps
st.markdown(f"<div style='{box_style} background-color: #white;'><b>{prediction_message}</b></div>", unsafe_allow_html=True)

  



