from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from ta.others import DailyReturnIndicator, CumulativeReturnIndicator
import pandas as pd

file_name = 'btc_bars_minute.csv'

# Load data from the CSV file into a DataFrame
df = pd.read_csv(file_name)

# Calculate the 5-minute simple moving average
sma_5 = SMAIndicator(close=df['close'], window=5)
df['SMA_5'] = sma_5.sma_indicator()

# Calculate the 10-minute exponential moving average
ema_10 = EMAIndicator(close=df['close'], window=10)
df['EMA_10'] = ema_10.ema_indicator()

# Calculate the RSI with a period of 14
rsi = RSIIndicator(close=df['close'], window=14)
df['RSI'] = rsi.rsi()

# Calculate the Stochastic Oscillator with default parameters
stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
df['%K'] = stoch.stoch()

# Calculate Bollinger Bands with a period of 20 and standard deviations of 2
bb = BollingerBands(close=df['close'], window=20, window_dev=2)
df['BB_upper'] = bb.bollinger_hband()
df['BB_lower'] = bb.bollinger_lband()

# Calculate Average True Range with a period of 14
atr = AverageTrueRange(
    high=df['high'], low=df['low'], close=df['close'], window=14)
df['ATR'] = atr.average_true_range()

# Calculate On-Balance Volume
obv = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume'])
df['OBV'] = obv.on_balance_volume()

# Calculate Daily Return
daily_return = DailyReturnIndicator(close=df['close'])
df['Daily_Return'] = daily_return.daily_return()

# Calculate Cumulative Return
cumulative_return = CumulativeReturnIndicator(close=df['close'])
df['Cumulative_Return'] = cumulative_return.cumulative_return()

# Add additional indicators here

# Delete the first 60 rows (since each hour contains 60 minutes)
df = df.iloc[10000:]
# Print the dataframe for testing
print(df)

# Save the dataframe
if df.to_csv('btc_bars (one_minute_indicators).csv', index=False) is None:
    print("DataFrame saved successfully.")
else:
    print("Error occurred while saving DataFrame.")
