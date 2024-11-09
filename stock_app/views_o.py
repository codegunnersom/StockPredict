from django.shortcuts import render

# Create your views here.

# stock_app/views.py
from django.shortcuts import render
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import yfinance as yf
from datetime import datetime, timedelta

def predict_stock(request):
    if request.method == 'POST':
        ticker = request.POST['ticker']
        days = int(request.POST['days'])

        # Fetch historical data
        data = yf.download(ticker, period='1y')
        
        # Prepare data for prediction
        data['Date'] = data.index
        data['Prediction'] = data['Close'].shift(-days)
        
        # Drop the last 'days' rows where prediction is NaN
        X = data[['Close']][:-days]
        y = data['Prediction'][:-days]

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions
        last_price = data['Close'].values[-1]
        predicted_price = model.predict([[last_price]])[0]

        # Render the results
        return render(request, 'results.html', {'predicted_price': predicted_price, 'ticker': ticker})

    return render(request, 'predict.html')

# stock_app/views.py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
import io
import base64
import yfinance as yf
from sklearn.linear_model import LinearRegression
from django.shortcuts import render

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

        # Prepare data for prediction
        data['Date'] = data.index
        data['Prediction'] = data['Close'].shift(-days)
        
        # Drop the last 'days' rows where prediction is NaN
        X = data[['Close']][:-days]
        y = data['Prediction'][:-days]

        # Train model
        model = LinearRegression()
        model.fit(X, y)

        # Make predictions
        last_price = data['Close'].values[-1]

        # Debugging: Print the shape and value of last_price
        print(f"Last price: {last_price}, Type: {type(last_price)}, Shape: {np.array([[last_price]]).shape}")

        # Ensure last_price is a float
        if isinstance(last_price, (np.ndarray, pd.Series)):
            last_price = last_price.flatten()[0]  # Convert to scalar

        # Ensure last_price is a 2D array with shape (1, 1)
        predicted_price = model.predict(np.array([[last_price]]))[0]

        # Plot historical prices
        plt.figure(figsize=(10, 5))
        plt.plot(data['Close'], label='Historical Prices', color='blue')
        plt.axvline(x=data.index[-days], color='red', linestyle='--', label='Prediction Start')
        plt.title(f'{ticker} Stock Price History')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()

        # Save the plot to a BytesIO object
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plot_url = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()

        # Render the results with the plot
        return render(request, 'results.html', {
            'predicted_price': predicted_price,
            'ticker': ticker,
            'plot_url': plot_url
        })

    return render(request, 'predict.html')