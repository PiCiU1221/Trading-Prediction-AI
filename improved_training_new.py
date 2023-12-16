import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Read the data from the CSV file
df = pd.read_csv('btc_bars (next_close_with_scalling).csv')

# Extract the numerical columns for regression
numerical_columns = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap', 'SMA_20',
                     'EMA_50', 'RSI', '%K', 'BB_upper', 'BB_lower', 'ATR', 'OBV', 'Daily_Return', 'Cumulative_Return']

# Convert timestamp column to pandas datetime type
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract the numerical columns for scaling
numerical_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap', 'SMA_20',
                     'EMA_50', 'RSI', '%K', 'BB_upper', 'BB_lower', 'ATR', 'OBV', 'Daily_Return', 'Cumulative_Return', 'Next_Close']

# Split the data into features (X) and target variable (y)
X = df[numerical_columns[1:]]  # Exclude the timestamp column
y = df['Next_Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

num_iterations = 10

# Load the best model from disk
best_model = joblib.load('best_model.pkl')

# Remove the "Next_Close" column from X_test
X_test = X_test.drop(columns=['Next_Close'])

y_test_pred = best_model.predict(X_test)
best_mse = mean_squared_error(y_test, y_test_pred)

for i in range(num_iterations):
    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Evaluate the model using mean squared error
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print("Iteration:", i+1)
    print("Train MSE:", train_mse)
    print("Test MSE:", test_mse)
    print()

    # Check if the current model has a better MSE than the previous best model
    if test_mse < best_mse:
        print("New best model found! Saving...")
        best_mse = test_mse
        best_model = model
        joblib.dump(best_model, 'best_model.pkl')
