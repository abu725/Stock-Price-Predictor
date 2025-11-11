# Stock Price Predictor using Linear Regression

import yfinance as yf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Step 1: Download historical stock data (Example: Apple)
stock_symbol = 'AAPL'  # You can change this to any stock symbol (e.g., 'GOOG', 'TSLA', 'MSFT')
data = yf.download(stock_symbol, start='2020-01-01', end='2025-01-01')

# Step 2: Prepare data
data = data[['Close']]
data['Prediction'] = data[['Close']].shift(-30)  # Predict 30 days into the future

# Step 3: Create feature and target datasets
X = data.drop(['Prediction'], axis=1)[:-30]
y = data['Prediction'][:-30]

# Step 4: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Test the model
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy*100:.2f}%")

# Step 7: Predict future prices
future = data.drop(['Prediction'], axis=1)[-30:]
predictions = model.predict(future)

# Step 8: Visualize
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['Close'], label='Actual Prices')
plt.plot(future.index, predictions, label='Predicted Prices', color='red')
plt.title(f"{stock_symbol} Stock Price Prediction (Linear Regression)")
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()
