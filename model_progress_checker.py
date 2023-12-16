import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from datetime import datetime

# Read the data from the CSV file
df = pd.read_csv('btc_bars (next_close_with_scalling).csv')

df_without_scaling = pd.read_csv('btc_bars.csv')
df_without_scaling = df_without_scaling.iloc[:-1]

# Choose highest and lowest values from df_without_scaling
highest_without_scaling = df_without_scaling['Next_Close'].max()
lowest_without_scaling = df_without_scaling['Next_Close'].min()

# Convert timestamp column to pandas datetime type
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract the numerical columns
numerical_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap', 'SMA_20',
                     'EMA_50', 'RSI', '%K', 'BB_upper', 'BB_lower', 'ATR', 'OBV', 'Daily_Return', 'Cumulative_Return', 'Next_Close']

# Split the data into features (X) and target variable (y)
X = df[numerical_columns[1:]]  # Exclude the timestamp column
y = df['Next_Close']

# Reshape the data into LSTM input shape
timesteps = 6  # Number of past time steps to consider
num_features = X.shape[1]
num_samples = len(X) - timesteps + 1

X_reshaped = np.zeros((num_samples, timesteps, num_features))
for i in range(num_samples):
    X_reshaped[i] = X.values[i:i+timesteps]

# Load the first model
model1 = keras.models.load_model('best_model.h5')

# Load the second model
model2 = keras.models.load_model('best_model2.h5')

# Generate random indices for data selection
num_samples = X_reshaped.shape[0]
num_predictions = 1000  # Number of random predictions to generate
random_indices = random.sample(range(num_samples), num_predictions)

# Select random data for prediction
X_random = X_reshaped[random_indices]

# Make predictions using the first model
predictions1 = model1.predict(X_random)

# Make predictions using the second model
predictions2 = model2.predict(X_random)

model1_score = 0
model2_score = 0

# Print the results
for i in range(num_predictions):
    actual = y.iloc[random_indices[i]] * \
        (highest_without_scaling - lowest_without_scaling) + lowest_without_scaling

    # Extract the single prediction values
    predicted1 = predictions1[i][0] * \
        (highest_without_scaling - lowest_without_scaling) + lowest_without_scaling
    predicted2 = predictions2[i][0] * \
        (highest_without_scaling - lowest_without_scaling) + lowest_without_scaling

    actual = round(actual, 2)
    predicted1 = round(predicted1, 2)
    predicted2 = round(predicted2, 2)
    difference1 = round(abs(actual - predicted1), 2)
    difference2 = round(abs(actual - predicted2), 2)

    # Get the corresponding timestamp
    timestamp = df['timestamp'].iloc[random_indices[i]]
    formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M")

    # Print the results
    print("Sample", i+1)
    print("Timestamp:", formatted_timestamp)
    print("Actual:", actual)
    print("Difference (Model 1):", difference1)
    print("Difference (Model 2):", difference2)
    print()
    if (difference1 < difference2):
        model1_score += 1
    else:
        model2_score += 1

print("Model 1 / Model 2")
print(model1_score, " / ", model2_score)
