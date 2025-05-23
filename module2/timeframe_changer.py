import os
import pandas as pd

# Path to the specific CSV file
file_path = '/Users/architmittal/Downloads/sample/module2/data_files/SBIN_20250415.csv'
data_folder = os.path.dirname(file_path)
file_name = os.path.basename(file_path)

print(f"Processing file: {file_name}")

# Read the 1-minute timeframe data
df = pd.read_csv(file_path)

# Look for datetime column or create it if missing
datetime_col = None
for col in df.columns:
    if 'date' in col.lower() or 'time' in col.lower():
        datetime_col = col
        break

if not datetime_col:
    # Assuming first column is datetime
    datetime_col = df.columns[0]
    print(f"Using column '{datetime_col}' as datetime")

# Parse the datetime column
df[datetime_col] = pd.to_datetime(df[datetime_col])

# Set datetime as index
df.set_index(datetime_col, inplace=True)

# Get column names in standard OHLCV format
if len(df.columns) >= 5:
    column_mapping = {}
    ohlcv = ['open', 'high', 'low', 'close', 'volume']
    
    for i, col in enumerate(ohlcv):
        if i < len(df.columns):
            column_mapping[df.columns[i]] = col
    
    # Rename columns to standard names
    df.rename(columns=column_mapping, inplace=True)

# Resample to 5-minute timeframe (OHLCV)
agg_dict = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
}
df_5min = df.resample('5T').agg(agg_dict).dropna()

# Reset index for the output file
df_5min = df_5min.reset_index()

# Save to new file
output_path = os.path.join(data_folder, '5min_' + file_name)
df_5min.to_csv(output_path, index=False)

print(f"Converted from 1-minute to 5-minute timeframe")
print(f"Original data shape: {df.shape}")
print(f"5-minute data shape: {df_5min.shape}")
print(f"5-minute data saved to {output_path}")