from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def check_stationarity(timeseries):
    """Check if the time series is stationary."""
    result = adfuller(timeseries)
    return result[1] <= 0.05  # Return True if stationary

def predict_stock(request):
    if request.method == 'POST':
        ticker = request.POST['ticker'].strip().upper()  # Normalize input
        days = int(request.POST['days'])

        # Basic validation
        if not ticker or not ticker.isalnum():
            return render(request, 'predict.html', {
                'error': 'Please enter a valid stock ticker.'
            })

        # Fetch historical data
        try:
            data = yf.download(ticker, period='1y')
            if data.empty:
                raise ValueError("No data found for the given ticker.")
        except Exception as e:
            return render(request, 'predict.html', {
                'error': str(e)
            })

        # Prepare data for ARIMA
        close_prices = data['Close'].asfreq('B')  # Ensure business day frequency
        close_prices = close_prices.fillna(method='ffill')  # Forward fill missing values

        # Check for stationarity
        if not check_stationarity(close_prices):
            close_prices = close_prices.diff().dropna()  # Difference the series if not stationary

        # Fit the ARIMA model
        p = 1  # AR term
        d = 1  # Differencing term
        q = 1  # MA term
        model = ARIMA(data['Close'], order=(p, d, q))
        model_fit = model.fit()

        # Make predictions for the next 'days'
        forecast = model_fit.forecast(steps=days)
        if forecast.empty:
            return render(request, 'predict.html', {
                'error': 'Forecasting failed. Please try again.'
            })

        # Get the last predicted price
        predicted_price = forecast.iloc[-1]

        # Plot historical prices and forecast
        plt.figure(figsize=(10, 5))
        plt.plot(data['Close'], label='Historical Prices', color='blue')
        forecast_index = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days, freq='B')
        plt.plot(forecast_index, forecast, label='Forecast', color='orange')
        plt.axvline(x=data.index[-1], color='red', linestyle='--', label='Prediction Start')
        plt.title(f'{ticker} Stock Price History and Forecast')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # Plot ACF and PACF
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        plot_acf(close_prices, ax=ax[0])
        plot_pacf(close_prices, ax=ax[1])
        plt.suptitle('ACF and PACF Plots')
        
        # Save ACF and PACF plot to a BytesIO object
        buf_acf_pacf = io.BytesIO()
        plt.savefig(buf_acf_pacf, format='png')
        buf_acf_pacf.seek(0)
        acf_pacf_plot_url = base64.b64encode(buf_acf_pacf.read()).decode('utf-8')
        plt.close()

        # Render the results with the plots
        return render(request, 'results.html', {
            'predicted_price': predicted_price,
            'ticker': ticker,
            'plot_url': plot_url,
            'acf_pacf_plot_url': acf_pacf_plot_url  # Include ACF and PACF plot
        })

    return render(request, 'predict.html')
