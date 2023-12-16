from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
import pandas as pd

# No keys required for crypto data
client = CryptoHistoricalDataClient()

# Creating datetime objects for start and end dates
start_date = datetime(2020, 1, 1)
end_date = datetime(2023, 5, 29)

# Creating request object
request_params = CryptoBarsRequest(
    symbol_or_symbols=["BTC/USD"],
    timeframe=TimeFrame.Hour,
    start=start_date,
    end=end_date
)

'''
# If the dataframe doesnt exist
# Retrieve daily bars for Bitcoin in a DataFrame and printing it
btc_bars = client.get_crypto_bars(request_params, feed='us')

# Convert to dataframe
df = btc_bars.df
'''

# Load data from the CSV file into a DataFrame
df = pd.read_csv('btc_bars.csv')

# Calculate the 20-day moving average
df['MA_20'] = df['close'].rolling(window=20).mean()

# Calculate the 50-day moving average
df['MA_50'] = df['close'].rolling(window=50).mean()

# Calculate the RSI with a period of 14
rsi = RSIIndicator(close=df['close'], window=14)
df['RSI'] = rsi.rsi()

# Calculate Bollinger Bands with a period of 20 and standard deviations of 2
bb = BollingerBands(close=df['close'], window=20, window_dev=2)
df['BB_upper'] = bb.bollinger_hband()
df['BB_lower'] = bb.bollinger_lband()

# Print the dataframe for testing
print(df)

# Specify the file path for saving the CSV file
file_path = 'btc_bars.csv'

# Save the dataframe
if df.to_csv(file_path, index=True) is None:
    print("DataFrame saved successfully.")
else:
    print("Error occurred while saving DataFrame.")