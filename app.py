import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Change the font style and color for the title
st.markdown("<h1 style='color: #D8BFD8; font-family:Gabriola, Times, serif;'>STOCK PRICE PREDICITION</h1>", unsafe_allow_html=True)

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

option = st.selectbox('Select one symbol', ( 'MSFT', 'MU','SPY', 'WMT', 'GME',  'NFLX', 'BNOX','AAPL',))
start_date = st.date_input('Start date')
end_date = st.date_input('End date')
# Fetch the stock data using yfinance
stock_data = yf.download(option, start=start_date, end=end_date)



# Create a submit button
if st.button("Submit"):
    #st.markdown("<style>.css-2trqyj{color: #FF69B4; font-size: 20px;}</style>", unsafe_allow_html=True)



    # Fetch the stock data using yfinance after the Submit button is clicked
    stock_data = yf.download(option, start=start_date, end=end_date)

    # Display the stock data if it's available
    if not stock_data.empty:
        # 1st graph coding
        fig3, ax3 = plt.subplots(figsize=(12, 6))
        current_price = yf.Ticker(option).history(period='1d')['Close'].iloc[0]
        ax3.plot(stock_data.index, [current_price] * len(stock_data), label="Today's Price", color="#FF1493", marker='o')
        ax3.plot(stock_data.index, stock_data['Open'], label="Open Price", color="#FF69B4", marker='o')
        ax3.plot(stock_data.index, stock_data['Close'], label="Close Price", color="#C71585", marker='o')
        ax3.set_xlabel('Date')
        ax3.set_ylabel('Price (USD)')
        ax3.set_title(f"Price Comparison for {option}")
        ax3.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        handles, labels = ax3.get_legend_handles_labels()
        ax3.legend(handles, labels, loc="best")
        st.pyplot(fig3)

        # Fetch the open and close prices
        open_price = stock_data['Open'].iloc[-1]
        close_price = stock_data['Close'].iloc[-1]
        current_price = yf.Ticker(option).history(period='1d')['Close'].iloc[0]

        st.markdown(f'<span style="color: #FFFACD;">Open Price: ${open_price:.2f}</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="color: #F0E68C;">Close Price: ${close_price:.2f}</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="color: #DEB887;">Today\'s Price: ${current_price:.2f}</span>', unsafe_allow_html=True)

        # 2nd graph coding (Open and Close prices)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(stock_data.index, stock_data['Close'], label='Close Price', color='#66CDAA', marker='o')
        ax.plot(stock_data.index, stock_data['Open'], label='Open Price', color='#8FBC8B', marker='o')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.set_title(f'Open and Close Price Comparison for {option}')
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.legend()
        st.pyplot(fig)

        st.markdown(f'<span style="color: #DEB887; font-size: 16px;"><b>Open Price:</b> ${open_price:.2f}</span>', unsafe_allow_html=True)
        st.markdown(f'<span style="color: #F4A460; font-size: 16px;"><b>Close Price:</b> ${close_price:.2f}</span>', unsafe_allow_html=True)

        # 3rd graph coding (Actual Price)
        actual_price = stock_data['Close'].iloc[-1]

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(stock_data.index, stock_data['Close'], label="Today's Price", color='#800000', marker='o')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price (USD)')
        ax.set_title(f'Today\'s Price Comparison for {option}')
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        plt.legend()
        st.pyplot(fig)
        st.markdown(f'<span style="color: #F0E68C;">Todays Price: ${actual_price:.2f}</span>', unsafe_allow_html=True)


        # Prepare the data for the prediction model
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
        prediction_days = 45
        prediction_dates = pd.date_range(start=stock_data.index[-1], periods=prediction_days + 1, closed='right')[1:]
        prediction_dates_ord = prediction_dates.to_series().apply(lambda x: x.toordinal())
        predictions = model.predict(prediction_dates_ord.values.reshape(-1, 1))

        # Plot the predicted price
        fig2, ax2 = plt.subplots(figsize=(12, 6))
        ax2.plot([stock_data.index[-1], predictions[-1]], [predictions[-1], predictions[-1]], label="Predicted Price", color="#FFFF00", linestyle='--', marker='o')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Price (USD)')
        ax2.set_title(f"Predicted Price for {option}")
        ax2.legend()
        st.pyplot(fig2)

        st.markdown(f'<span style="color: #F0E68C; font-size: 16px;"><b>Predicted Price:</b> ${predictions[-1]:.2f}</span>', unsafe_allow_html=True)

        # Add the 'Open' column to the stock_data DataFrame
        stock_data['Open'] = stock_data['Open']

        # Fetch the current stock price using yfinance
        current_price = yf.Ticker(option).history(period='1d')['Close'].iloc[-1]

        # Determine if the predicted price is higher, lower, or the same as today's price
        prediction_message = ""
        prediction_color = "#E9967A"

        if predictions[-1] > current_price:
            prediction_message = f"Predicted Price is Higher by ${predictions[-1] - current_price:.2f} compared to Today's Price"
            prediction_color = "green"
            prediction_symbol = "↑"
        elif predictions[-1] < current_price:
            prediction_message = f"Predicted Price is Lower by ${current_price - predictions[-1]:.2f} compared to Today's Price"
            prediction_color = "red"
            prediction_symbol = "↓"
        else:
            prediction_message = "Predicted Price is the Same as Today's Price"
            prediction_symbol = "→"

        # Display the prediction message
        st.markdown("<h1 style='color:#FFE4E1; font-family:Gabriola, Times, serif;'>PREDICTION PRICE</h1>", unsafe_allow_html=True)

        def create_colored_box(label, value, bg_color):
            style = f"border-radius: 15px; padding: 10px; margin: 10px; background-color: {bg_color};"
            return f"<div style='{style}'><b>{label}:</b> ${value:.2f}</div>"

        style = f"border-radius: 15px; padding: 10px; margin: 10px; background-color: {prediction_color};"
        st.markdown(f"<div style='{style}'><b>{prediction_message} {prediction_symbol}</b></div>", unsafe_allow_html=True)
        st.markdown(create_colored_box("Predicted Price", predictions[-1], "#696969"), unsafe_allow_html=True)
        st.markdown(create_colored_box("Today's Price", current_price, "#A9A9A9"), unsafe_allow_html=True)


