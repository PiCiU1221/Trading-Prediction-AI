import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

# Read the data from the CSV file
df = pd.read_csv('btc_bars (next_close_with_scalling).csv')

# Convert timestamp column to pandas datetime type
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Extract the numerical columns for scaling
numerical_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap', 'SMA_20',
                     'EMA_50', 'RSI', '%K', 'BB_upper', 'BB_lower', 'ATR', 'OBV', 'Daily_Return', 'Cumulative_Return', 'Next_Close']

# Split the data into features (X) and target variable (y)
X = df[numerical_columns[1:]]  # Exclude the timestamp column
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

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=64, input_shape=(timesteps, num_features)))
model.add(Dense(units=1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

num_epochs = 50
batch_size = 32

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_reshaped, y[timesteps-1:], test_size=0.2, shuffle=False)

# Define early stopping callback
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)

# Initialize the best loss with a high value
best_loss = float('inf')

# Create an empty list to store the training history
history = {'loss': [], 'val_loss': []}

# Train the model
try:
    i = 0
    while True:
        i += 1
        print(f"Traing: {i}")

        # Train the model
        history_epoch = model.fit(X_train, y_train, epochs=num_epochs,
                                  batch_size=batch_size, validation_data=(
                                      X_test, y_test),
                                  callbacks=[early_stopping])

        # Update the training history
        history['loss'].extend(history_epoch.history['loss'])
        history['val_loss'].extend(history_epoch.history['val_loss'])

        # Get the validation loss for the current epoch
        val_loss = history_epoch.history['val_loss'][0]

        # Check if the current model has a better loss than the best model
        if val_loss < best_loss:
            print("New best model found! Saving...")
            model.save('best_model.h5')  # Save the new best model
            best_loss = val_loss

        print("Validation Loss: {:.4f}".format(val_loss))
        print()

except KeyboardInterrupt:
    # Training interrupted, display the loss curves
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
