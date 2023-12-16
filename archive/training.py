from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense

df = pd.read_csv('btc_bars (next_close_with_scalling).csv')

# Split the data into features (X) and target variable (y)
X = df.drop(columns=['Next_Close'])
y = df['Next_Close']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

timesteps = 6  # Number of past time steps to consider

num_features = df.shape[1] - 1

# Create the LSTM model
model = Sequential()
model.add(LSTM(units=64, input_shape=(timesteps, num_features)))
model.add(Dense(units=1))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit(X_train, y_train, epochs=10,
                    batch_size=32, validation_data=(X_test, y_test))
