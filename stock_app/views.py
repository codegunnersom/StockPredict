from django.shortcuts import render
import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import yfinance as yf
from prophet import Prophet  # Import Prophet

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

        # Fetch the current price
        try:
            current_data = yf.Ticker(ticker)
            current_price = current_data.history(period='1d')['Close'].iloc[-1]  # Get the latest closing price
        except Exception as e:
            current_price = None  # Handle the case where current price cannot be fetched
            print(f"Could not fetch current price: {e}")

        # Prepare data for Prophet
        df = data[['Close']].reset_index()  # Reset index to have a column for dates
        df.columns = ['ds', 'y']  # Rename columns to fit Prophet's requirements
        
        # Remove timezone information from the 'ds' column
        df['ds'] = df['ds'].dt.tz_localize(None)

        # Fit the Prophet model
        model = Prophet()
        model.fit(df)

        # Create a dataframe to hold future dates
        future = model.make_future_dataframe(periods=days, freq='B')  # 'B' for business days

        # Forecasting
        forecast = model.predict(future)

        # Get the last predicted price
        predicted_price = forecast['yhat'].iloc[-1]

        # Plot historical prices and forecast
        plt.figure(figsize=(10, 5))
        plt.plot(df['ds'], df['y'], label='Historical Prices', color='blue')
        plt.plot(forecast['ds'], forecast['yhat'], label='Forecast', color='orange')
        plt.axvline(x=df['ds'].iloc[-1], color='red', linestyle='--', label='Prediction Start')
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

        # Render the results with the plots
        return render(request, 'results.html', {
            'predicted_price': predicted_price,
            'current_price': current_price,  # Pass the current price here
            'ticker': ticker,
            'plot_url': plot_url,
            'current_year': datetime.datetime.now().year, 
        })

    return render(request, 'predict.html')