import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

# Read the data from the CSV file
df = pd.read_csv('btc_bars (one_minute_indicators).csv')

# Extract the numerical columns for regression
numerical_columns = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap', 'SMA_20',
                     'EMA_50', 'RSI', '%K', 'BB_upper', 'BB_lower', 'ATR', 'OBV', 'Daily_Return', 'Cumulative_Return']

# Convert timestamp column to pandas datetime type
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract the numerical columns for scaling
numerical_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap', 'SMA_20',
                     'EMA_50', 'RSI', '%K', 'BB_upper', 'BB_lower', 'ATR', 'OBV', 'Daily_Return', 'Cumulative_Return']

# Split the data into features (X) and target variable (y)
X = df[numerical_columns[1:-1]]  # Exclude the timestamp and Next_Close columns
y = df['Next_Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

num_iterations = 10

# Load the best model from disk
best_model = joblib.load('best_model_1minute.pkl')

predictions = []

for i in range(num_iterations):
    # Create a linear regression model
    model = LinearRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the testing data
    y_test_pred = model.predict(X_test)

    # Store the predictions in memory
    predictions.append(y_test_pred)

    # Evaluate the model using mean squared error
    test_mse = mean_squared_error(y_test, y_test_pred)

    print("Iteration:", i+1)
    print("Test MSE:", test_mse)
    print()

    # Check if the current model has a better MSE than the previous best model
    if test_mse < best_mse:
        print("New best model found! Saving...")
        best_mse = test_mse
        best_model = model
        joblib.dump(best_model, 'best_model_1minute.pkl')

# Save the predictions to a CSV file
predictions_df = pd.DataFrame(predictions).transpose()
predictions_df.columns = [f'Prediction_{i+1}' for i in range(num_iterations)]
predictions_df.to_csv('predictions.csv', index=False)
