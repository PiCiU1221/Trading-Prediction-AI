from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('btc_bars.csv')

# Extract the features (input) and target variable (output)
features = df.drop(columns=['Next_Close'])
target = df['Next_Close']


# Select the numerical columns for scaling
numerical_columns = ['open', 'high', 'low', 'close', 'volume', 'trade_count', 'vwap', 'SMA_20',
                     'EMA_50', 'RSI', '%K', 'BB_upper', 'BB_lower', 'ATR', 'OBV', 'Daily_Return', 'Cumulative_Return', 'Next_Close']

# Create a MinMaxScaler
scaler = MinMaxScaler()  # or StandardScaler() for standardization

# Fit and transform the data using the scaler
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# The numerical columns are now scaled or normalized within a specific range

# Print the first few rows of the dataframe
print(df.head())

# Save the dataframe
if df.to_csv('btc_bars (next_close_with_scalling).csv', index=False) is None:
    print("DataFrame saved successfully.")
else:
    print("Error occurred while saving DataFrame.")
