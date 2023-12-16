import pandas as pd
import numpy as np
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# Load the saved model
model = load_model('best_model.h5')

# Read the data from the CSV file
df = pd.read_csv('btc_bars (next_close_with_scalling).csv')

# Convert timestamp column to pandas datetime type
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract the numerical columns for scaling
numerical_columns = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap', 'SMA_20',
                     'EMA_50', 'RSI', '%K', 'BB_upper', 'BB_lower', 'ATR', 'OBV', 'Daily_Return',
                     'Cumulative_Return', 'Next_Close']

# Split the data into features (X) and target variable (y)
X = df[numerical_columns[:-1]]  # Exclude the target variable
y = df['Next_Close']

# Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape the data into LSTM input shape
timesteps = 6  # Number of past time steps to consider
num_features = X_scaled.shape[1]
num_samples = len(X_scaled) - timesteps + 1

X_reshaped = np.zeros((num_samples, timesteps, num_features))
for i in range(num_samples):
    X_reshaped[i] = X_scaled[i:i+timesteps]

# Reshape the target variable to match the input shape
y_reshaped = y[timesteps-1:].values.reshape(-1, 1)

# Make predictions
y_pred = model.predict(X_reshaped)

# Inverse transform the scaled predictions and target variable
y_pred_inv = scaler.inverse_transform(y_pred)
y_inv = scaler.inverse_transform(y_reshaped)

# Calculate the evaluation metrics
mse = mean_squared_error(y_inv, y_pred_inv)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_inv, y_pred_inv)

print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)

# Visualize the predictions
plt.plot(y_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.xlabel('Time')
plt.ylabel('Next_Close')
plt.legend()
plt.show()
