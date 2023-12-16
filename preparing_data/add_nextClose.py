import pandas as pd

df = pd.read_csv('btc_bars.csv')

# Add 'Next_Close' column
df['Next_Close'] = df['close'].shift(-1)

# Select all without the last row
df = df.iloc[:-1]

# Save the dataframe
if df.to_csv('btc_bars (next_close).csv', index=False) is None:
    print("DataFrame saved successfully.")
else:
    print("Error occurred while saving DataFrame.")
