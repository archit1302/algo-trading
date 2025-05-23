
# pandas = panel data
# same as excel sheet
# 100x more powerful than excel sheet

# import pandas as pd

# sma_value = 5

# df = pd.read_csv('SBIN_20250415.csv')

# df['avg_prices'] = df[['Open', 'High', 'Low', 'Close']].mean(axis=1)
# df['sma_close'] = df['Close'].rolling(window=sma_value).mean()

# df.to_csv('sample_file.csv', index=False)

import os
import pandas as pd
import numpy as np
from datetime import datetime

def calculate_indicators(df, ema_period=14, rsi_period=14):
    """
    Calculate technical indicators: EMA, RSI, and VWAP
    
    Args:
        df: DataFrame with OHLCV data
        ema_period: Period for EMA calculation
        rsi_period: Period for RSI calculation
        
    Returns:
        DataFrame with added indicator columns
    """
    # Ensure the DataFrame has the necessary columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        print("Warning: DataFrame is missing required columns. Attempting to map columns...")
        # Try to map columns based on position (common OHLCV format)
        if len(df.columns) >= 6:  # Datetime + OHLCV
            # Rename columns to standard names
            df.columns = ['Datetime'] + required_cols + list(df.columns[6:])
    
    # Calculate EMA
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    
    # Handle zero division
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate VWAP (Volume Weighted Average Price)
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['VP'] = df['Typical_Price'] * df['Volume']
    
    # Group by date for VWAP calculation
    df['Date'] = pd.to_datetime(df['Datetime']).dt.date
    df['Cumulative_VP'] = df.groupby('Date')['VP'].cumsum()
    df['Cumulative_Volume'] = df.groupby('Date')['Volume'].cumsum()
    df['VWAP'] = df['Cumulative_VP'] / df['Cumulative_Volume']
    
    # Clean up temporary columns
    df = df.drop(['VP', 'Typical_Price', 'Cumulative_VP', 'Cumulative_Volume'], axis=1)
    
    return df

def merge_and_process_files(directory_path):
    """
    Merge all CSV files in the directory and calculate indicators
    
    Args:
        directory_path: Path to the directory with data files
        
    Returns:
        Processed DataFrame with all data and indicators
    """
    # Check if directory exists
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' not found!")
        return None
    
    # List all CSV files
    csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
    
    if not csv_files:
        print("No CSV files found in the directory.")
        return None
    
    print(f"Found {len(csv_files)} CSV files.")
    
    # Initialize a list to store individual DataFrames
    dfs = []
    
    # Process each file
    for file in csv_files:
        file_path = os.path.join(directory_path, file)
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            
            # Add a column for the stock symbol (extracted from filename)
            stock_symbol = file.split('_')[0]
            df['Symbol'] = stock_symbol
            
            # Add the DataFrame to our list
            dfs.append(df)
            print(f"Processed: {file} - {len(df)} rows")
            
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")
    
    if not dfs:
        print("No valid data found.")
        return None
    
    # Merge all DataFrames
    merged_df = pd.concat(dfs, ignore_index=True)
    
    # Convert datetime to a proper datetime format
    if 'Datetime' not in merged_df.columns:
        # Try to find the datetime column
        date_cols = [col for col in merged_df.columns if 'date' in col.lower() or 'time' in col.lower()]
        if date_cols:
            merged_df.rename(columns={date_cols[0]: 'Datetime'}, inplace=True)
        else:
            # If there's no obvious datetime column, use the first column
            merged_df.rename(columns={merged_df.columns[0]: 'Datetime'}, inplace=True)
    
    # Convert to proper datetime
    merged_df['Datetime'] = pd.to_datetime(merged_df['Datetime'])
    
    # Sort by symbol and datetime
    merged_df.sort_values(by=['Symbol', 'Datetime'], inplace=True)
    
    # Calculate indicators
    print("\nCalculating technical indicators...")
    merged_df = calculate_indicators(merged_df)
    
    return merged_df

def save_processed_data(df, output_path):
    """
    Save the processed DataFrame to CSV
    
    Args:
        df: Processed DataFrame
        output_path: Path to save the output CSV file
    """
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    
    # Print statistics
    print("\nData Summary:")
    print(f"Total rows: {len(df)}")
    print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
    print(f"Symbols included: {', '.join(df['Symbol'].unique())}")
    print("\nIndicator columns added: EMA, RSI, VWAP")
    
    # Preview
    print("\nFirst few rows of processed data:")
    print(df.head().to_string())

def main():
    # Directory containing the data files
    directory_path = "/Users/architmittal/Downloads/sample/module2/data_files"
    
    # Output file path
    output_dir = os.path.dirname(directory_path)
    output_path = os.path.join(output_dir, "merged_data_with_indicators.csv")
    
    # Process the files
    print(f"Processing files from: {directory_path}")
    merged_df = merge_and_process_files(directory_path)
    
    if merged_df is not None:
        # Save the output
        save_processed_data(merged_df, output_path)

if __name__ == "__main__":
    main()


# Assignment
# practice 5 more functions of pandas from the official documentation
# http://pandas.pydata.org/docs/reference/api/pandas.factorize.html
# try to play around with the data 
# read the file and add the 1 minute %tage return column to it
# try to make the data into 5min, 15 min, 30 min, 1 hour, 1 day timeframe



# read one file from data_files folder and turn it into a 5-minute timeframe