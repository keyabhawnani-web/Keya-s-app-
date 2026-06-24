import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime

st.set_page_config(page_title="Stock Forecast with ARIMA", layout="wide")

st.title("📈 Stock Price Forecast using ARIMA")

ticker = st.text_input(
    "Enter Stock Ticker",
    value="AAPL"
)

if st.button("Run Forecast"):

    try:
        # Download last 5 years of data
        end_date = datetime.today()
        start_date = end_date - pd.DateOffset(years=5)

        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True
        )

        if data.empty:
            st.error("No data found for ticker.")
            st.stop()

        st.subheader("Historical Data")
        st.dataframe(data.tail())

        # Closing prices
        close_prices = data["Close"]

        # Historical Chart
        st.subheader("Historical Closing Price")

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(close_prices.index,
                close_prices,
                label="Close Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.set_title(f"{ticker} Stock Price (Last 5 Years)")
        ax.legend()

        st.pyplot(fig)

        # ARIMA Model
        st.subheader("ARIMA Forecast")

        model = ARIMA(close_prices, order=(5,1,0))
        model_fit = model.fit()

        # Forecast until June 2027
        forecast_date = pd.Timestamp("2027-06-30")

        months_needed = (
            (forecast_date.year - close_prices.index[-1].year) * 12
            + forecast_date.month
            - close_prices.index[-1].month
        )

        if months_needed < 1:
            months_needed = 1

        forecast = model_fit.forecast(steps=months_needed * 22)

        predicted_price = forecast.iloc[-1]

        st.success(
            f"Predicted {ticker} Price for June 2027: "
            f"${predicted_price:.2f}"
        )

        # Forecast Chart
        forecast_index = pd.date_range(
            start=close_prices.index[-1],
            periods=len(forecast)+1,
            freq="B"
        )[1:]

        fig2, ax2 = plt.subplots(figsize=(10, 5))

        ax2.plot(
            close_prices.index[-250:],
            close_prices[-250:],
            label="Historical"
        )

        ax2.plot(
            forecast_index,
            forecast,
            label="Forecast"
        )

        ax2.set_title(
            f"{ticker} Forecast using ARIMA"
        )

        ax2.legend()

        st.pyplot(fig2)

    except Exception as e:
        st.error(f"Error: {e}")
