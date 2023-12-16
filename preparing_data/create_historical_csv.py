from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
from datetime import datetime

################################################################
# CHANGABLE VARIABLES

# Specify the name for the CSV file
file_name = 'btc_bars_minute.csv'

# Specify the symbol
symbol = "BTC/USD"

# Set the start and end dates
start_date = datetime(2021, 1, 1)
end_date = datetime(2023, 5, 29)

# Specify the timeframe
timeframeVariable = TimeFrame.Minute

################################################################


# No keys required for crypto data
client = CryptoHistoricalDataClient()

# Creating request object
request_params = CryptoBarsRequest(
    symbol_or_symbols=["BTC/USD"],
    timeframe=timeframeVariable,
    start=start_date,
    end=end_date
)

# Retrieve daily bars for Bitcoin in a DataFrame and printing it
btc_bars = client.get_crypto_bars(request_params, feed='us')

# Convert to dataframe
df = btc_bars.df

# Print the dataframe for testing
print(df)

# Save the dataframe
if df.to_csv(file_name, index=True) is None:
    print("DataFrame saved successfully.")
else:
    print("Error occurred while saving DataFrame.")
