from ta.trend import SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator
from ta.others import DailyReturnIndicator, CumulativeReturnIndicator
import pandas as pd

file_name = 'btc_bars.csv'

# Load data from the CSV file into a DataFrame
df = pd.read_csv(file_name)

# Calculate the 20-hour simple moving average
sma_20 = SMAIndicator(close=df['close'], window=20)
df['SMA_20'] = sma_20.sma_indicator()

# Calculate the 50-hour exponential moving average
ema_50 = EMAIndicator(close=df['close'], window=50)
df['EMA_50'] = ema_50.ema_indicator()

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

# Delete the first 1000 rows
df = df.iloc[1000:]

# Print the dataframe for testing
print(df)

# Save the dataframe
if df.to_csv('btc_bars (one_hour_indicators).csv', index=False) is None:
    print("DataFrame saved successfully.")
else:
    print("Error occurred while saving DataFrame.")
