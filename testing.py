import pandas as pd
import random

# Load data from the CSV file into a DataFrame
talib = pd.read_csv('btc_bars (gpt_new).csv')
gpt = pd.read_csv('btc_bars (gpt).csv')

# Dropping columns
# df = df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1)

# Print columns to delete
# print(df.columns)

# Set the start and end dates of the range
start_date = '2023-01-01 20:00:00+00:00'
end_date = '2023-05-28 20:00:00+00:00'

# Filter the data within the specified date range
filtered_df = talib[(talib['timestamp'] >= start_date)
                    & (talib['timestamp'] <= end_date)]
filtered_df2 = gpt[(gpt['timestamp'] >= start_date)
                   & (gpt['timestamp'] <= end_date)]

# Set a specific random seed value
random_seed = 42

# Check the number of rows in the filtered DataFrame
num_rows = min(len(filtered_df), len(filtered_df2))

# Get 10 random samples from the filtered data
random_samples = filtered_df.sample(
    n=min(10, num_rows), random_state=random_seed)
random_samples2 = filtered_df2.sample(
    n=min(10, num_rows), random_state=random_seed)

# Iterate through the samples simultaneously
for idx, (row1, row2) in enumerate(zip(random_samples.iterrows(), random_samples2.iterrows())):
    index1, data1 = row1
    index2, data2 = row2

    timestamp1 = data1['timestamp']
    macd_value1 = data1['MACD']

    timestamp2 = data2['timestamp']
    macd_value2 = data2['MACD']

    # Display the results
    print(f"Sample {idx+1}:")
    print(f"GPT_new - MACD value at {timestamp1}: {macd_value1}")
    print(f"GPT - MACD value at {timestamp2}: {macd_value2}")
    print()

# Print the dataframe for testing
# print(filtered_df)

# Save the dataframe
'''
if df.to_csv('btc_bars.csv', index=False) is None:
    print("DataFrame saved successfully.")
else:
    print("Error occurred while saving DataFrame.")
'''
