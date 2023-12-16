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

'''
# Choose highest and lowest values from df
highest_df = df['Next_Close'].max()
lowest_df = df['Next_Close'].min()
'''

# Choose highest and lowest values from df_scaling
highest_without_scaling = df_without_scaling['Next_Close'].max()
lowest_without_scaling = df_without_scaling['Next_Close'].min()

'''
# Print the results
print("Highest Next_Close in df:", highest_df)
print("Lowest Next_Close in df:", lowest_df)
print("Highest Next_Close in df_without_scaling:", highest_without_scaling)
print("Lowest Next_Close in df_without_scaling:", lowest_without_scaling)
'''

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

# Load the best model
best_model_path = 'best_model.h5'
best_model = keras.models.load_model(best_model_path)

# Generate random indices for data selection
num_samples = X_reshaped.shape[0]
num_predictions = 10  # Number of random predictions to generate
random_indices = random.sample(range(num_samples), num_predictions)

# Select random data for prediction
X_random = X_reshaped[random_indices]

# Make predictions
predictions = best_model.predict(X_random)

# Load the scaler used for scaling the target variable
scaler = MinMaxScaler()  # or StandardScaler() if that was used for scaling
# Reshape y to a 2D array before fitting the scaler
scaler.fit(y.values.reshape(-1, 1))

# Inverse scaling of predictions
predictions_unscaled = scaler.inverse_transform(predictions)

# Print the results
for i in range(num_predictions):
    actual = y.iloc[random_indices[i]] * \
        (highest_without_scaling - lowest_without_scaling) + lowest_without_scaling

    # Extract the single prediction value
    predicted = predictions_unscaled[i][0] * \
        (highest_without_scaling - lowest_without_scaling) + lowest_without_scaling

    actual = round(actual, 2)
    predicted = round(predicted, 2)

    # Get the corresponding timestamp
    timestamp = df['timestamp'].iloc[random_indices[i]]
    formatted_timestamp = timestamp.strftime("%Y-%m-%d %H:%M")

    # Print the results
    print("Sample", i+1)
    print("Timestamp:", formatted_timestamp)
    print("Actual Next_Close:", actual)
    print("Predicted Next_Close:", predicted)
    print()
