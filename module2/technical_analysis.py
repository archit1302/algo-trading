import pandas as pd
import numpy as np

def add_sma(df, column='close', window=20):
    """
    Add Simple Moving Average to DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price data
    column : str
        Column name for calculation (default: 'close')
    window : int
        Window period for calculation (default: 20)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added SMA column
    """
    # Ensure column is lowercase
    column = column.lower()
    
    # Create column name
    col_name = f'sma_{window}'
    
    # Calculate SMA
    df[col_name] = df[column].rolling(window=window).mean()
    
    return df

def add_ema(df, column='close', window=20):
    """
    Add Exponential Moving Average to DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price data
    column : str
        Column name for calculation (default: 'close')
    window : int
        Window period for calculation (default: 20)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added EMA column
    """
    # Ensure column is lowercase
    column = column.lower()
    
    # Create column name
    col_name = f'ema_{window}'
    
    # Calculate EMA
    df[col_name] = df[column].ewm(span=window, adjust=False).mean()
    
    return df

def add_rsi(df, column='close', window=14):
    """
    Add Relative Strength Index to DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price data
    column : str
        Column name for calculation (default: 'close')
    window : int
        Window period for calculation (default: 14)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added RSI column
    """
    # Ensure column is lowercase
    column = column.lower()
    
    # Create column name
    col_name = f'rsi_{window}'
    
    # Calculate RSI
    delta = df[column].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Handle potential zero division
    rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)
    df[col_name] = 100 - (100 / (1 + rs))
    
    return df

def add_bollinger_bands(df, column='close', window=20, num_std=2):
    """
    Add Bollinger Bands to DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price data
    column : str
        Column name for calculation (default: 'close')
    window : int
        Window period for SMA calculation (default: 20)
    num_std : int or float
        Number of standard deviations for bands (default: 2)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added Bollinger Bands columns
    """
    # Ensure column is lowercase
    column = column.lower()
    
    # Calculate middle band (SMA)
    df = add_sma(df, column, window)
    middle_band = f'sma_{window}'
    
    # Calculate standard deviation
    std = df[column].rolling(window=window).std()
    
    # Calculate upper and lower bands
    df[f'bb_upper_{window}'] = df[middle_band] + (std * num_std)
    df[f'bb_lower_{window}'] = df[middle_band] - (std * num_std)
    df[f'bb_width_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[middle_band]
    
    return df

def add_vwap(df, datetime_col='datetime'):
    """
    Add Volume Weighted Average Price to DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLCV data
    datetime_col : str
        Name of datetime column (default: 'datetime')
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added VWAP column
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Create date column for grouping if datetime column exists
    if datetime_col in df.columns:
        df['date'] = pd.to_datetime(df[datetime_col]).dt.date
    
    # Calculate typical price
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate VWAP components
    df['vp'] = df['typical_price'] * df['volume']
    
    # Group by date for VWAP calculation
    if 'date' in df.columns:
        df['cum_vp'] = df.groupby('date')['vp'].cumsum()
        df['cum_vol'] = df.groupby('date')['volume'].cumsum()
    else:
        df['cum_vp'] = df['vp'].cumsum()
        df['cum_vol'] = df['volume'].cumsum()
    
    # Calculate VWAP
    df['vwap'] = df['cum_vp'] / df['cum_vol']
    
    # Drop intermediate columns
    df.drop(['typical_price', 'vp', 'cum_vp', 'cum_vol'], axis=1, inplace=True)
    
    # Drop date column if it was created in this function
    if 'date' in df.columns and datetime_col in df.columns:
        df.drop('date', axis=1, inplace=True)
    
    return df

def add_supertrend(df, period=10, multiplier=3):
    """
    Add SuperTrend indicator to DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLC data
    period : int
        ATR period (default: 10)
    multiplier : int or float
        ATR multiplier (default: 3)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added SuperTrend columns
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Calculate True Range
    df['tr'] = np.maximum.reduce([
        (df['high'] - df['low']),
        abs(df['high'] - df['close'].shift()),
        abs(df['low'] - df['close'].shift())
    ])
    
    # Calculate ATR
    df['atr'] = df['tr'].rolling(period).mean()
    
    # Calculate SuperTrend
    df['basic_upper'] = (df['high'] + df['low']) / 2 + (multiplier * df['atr'])
    df['basic_lower'] = (df['high'] + df['low']) / 2 - (multiplier * df['atr'])
    
    # Initial values
    df['supertrend_upper'] = df['basic_upper'].copy()
    df['supertrend_lower'] = df['basic_lower'].copy()
    df['supertrend'] = df['close'].copy()
    df['st_direction'] = 1  # 1 for uptrend, -1 for downtrend
    
    # Calculate SuperTrend values using a loop
    for i in range(1, len(df)):
        # Upper band
        if df['basic_upper'].iloc[i] < df['supertrend_upper'].iloc[i-1] or df['close'].iloc[i-1] > df['supertrend_upper'].iloc[i-1]:
            df.loc[df.index[i], 'supertrend_upper'] = df['basic_upper'].iloc[i]
        else:
            df.loc[df.index[i], 'supertrend_upper'] = df['supertrend_upper'].iloc[i-1]
        
        # Lower band
        if df['basic_lower'].iloc[i] > df['supertrend_lower'].iloc[i-1] or df['close'].iloc[i-1] < df['supertrend_lower'].iloc[i-1]:
            df.loc[df.index[i], 'supertrend_lower'] = df['basic_lower'].iloc[i]
        else:
            df.loc[df.index[i], 'supertrend_lower'] = df['supertrend_lower'].iloc[i-1]
        
        # SuperTrend value and direction
        if df['st_direction'].iloc[i-1] == 1 and df['close'].iloc[i] < df['supertrend_lower'].iloc[i]:
            df.loc[df.index[i], 'st_direction'] = -1
        elif df['st_direction'].iloc[i-1] == -1 and df['close'].iloc[i] > df['supertrend_upper'].iloc[i]:
            df.loc[df.index[i], 'st_direction'] = 1
        else:
            df.loc[df.index[i], 'st_direction'] = df['st_direction'].iloc[i-1]
        
        # Set SuperTrend value based on direction
        if df['st_direction'].iloc[i] == 1:
            df.loc[df.index[i], 'supertrend'] = df['supertrend_lower'].iloc[i]
        else:
            df.loc[df.index[i], 'supertrend'] = df['supertrend_upper'].iloc[i]
    
    # Create column name with parameters
    col_name = f'supertrend_{period}_{multiplier}'
    df[col_name] = df['supertrend']
    
    # Create signal column (1: buy, -1: sell, 0: neutral)
    df[f'{col_name}_signal'] = 0
    df.loc[df['st_direction'] == 1, f'{col_name}_signal'] = 1
    df.loc[df['st_direction'] == -1, f'{col_name}_signal'] = -1
    
    # Drop intermediate columns
    df.drop(['tr', 'atr', 'basic_upper', 'basic_lower', 'supertrend_upper', 
             'supertrend_lower', 'supertrend', 'st_direction'], axis=1, inplace=True)
    
    return df

def identify_support_resistance(df, window=5, threshold=0.01):
    """
    Identify support and resistance levels
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price data
    window : int
        Window for identifying local minima/maxima (default: 5)
    threshold : float
        Minimum percent difference between levels (default: 0.01)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added support/resistance columns
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    # Initialize columns
    df['is_support'] = 0
    df['is_resistance'] = 0
    
    # Function to check if a point is a local minimum or maximum
    def is_local_minima(p1, p2, p3, p4, p5):
        return p3 < p1 and p3 < p2 and p3 < p4 and p3 < p5
    
    def is_local_maxima(p1, p2, p3, p4, p5):
        return p3 > p1 and p3 > p2 and p3 > p4 and p3 > p5
    
    # Identify local minima and maxima
    for i in range(window, len(df) - window):
        # Get prices for comparison
        p1, p2, p3, p4, p5 = df['low'].iloc[i-2:i+3]
        
        # Check for support
        if is_local_minima(p1, p2, p3, p4, p5):
            df.loc[df.index[i], 'is_support'] = 1
        
        # Get prices for comparison
        p1, p2, p3, p4, p5 = df['high'].iloc[i-2:i+3]
        
        # Check for resistance
        if is_local_maxima(p1, p2, p3, p4, p5):
            df.loc[df.index[i], 'is_resistance'] = 1
    
    # Extract support and resistance levels
    supports = df[df['is_support'] == 1]['low'].tolist()
    resistances = df[df['is_resistance'] == 1]['high'].tolist()
    
    # Filter levels to remove closely spaced ones
    def filter_levels(levels, threshold):
        if not levels:
            return []
        
        # Sort levels
        levels = sorted(levels)
        
        # Filter by minimum percent difference
        filtered = [levels[0]]
        for level in levels[1:]:
            if (level - filtered[-1]) / filtered[-1] >= threshold:
                filtered.append(level)
        
        return filtered
    
    # Filter and sort levels
    support_levels = filter_levels(supports, threshold)
    resistance_levels = filter_levels(resistances, threshold)
    
    # Add the filtered levels to the DataFrame
    df['support_levels'] = str(support_levels)
    df['resistance_levels'] = str(resistance_levels)
    
    # Add current price position relative to support/resistance
    df['sr_position'] = 0  # 0: neutral, 1: near support, -1: near resistance
    
    # Calculate distance to nearest support and resistance
    for i in range(len(df)):
        price = df['close'].iloc[i]
        
        # Find nearest support
        nearest_support = None
        min_support_dist = float('inf')
        for s in support_levels:
            dist = (price - s) / price
            if dist > 0 and dist < min_support_dist:
                min_support_dist = dist
                nearest_support = s
        
        # Find nearest resistance
        nearest_resistance = None
        min_resistance_dist = float('inf')
        for r in resistance_levels:
            dist = (r - price) / price
            if dist > 0 and dist < min_resistance_dist:
                min_resistance_dist = dist
                nearest_resistance = r
        
        # Determine if price is near support or resistance
        if nearest_support and min_support_dist < threshold:
            df.loc[df.index[i], 'sr_position'] = 1
        elif nearest_resistance and min_resistance_dist < threshold:
            df.loc[df.index[i], 'sr_position'] = -1
        
        # Add nearest levels to the DataFrame
        df.loc[df.index[i], 'nearest_support'] = nearest_support if nearest_support else None
        df.loc[df.index[i], 'nearest_resistance'] = nearest_resistance if nearest_resistance else None
    
    return df

def add_all_indicators(df, column='close', datetime_col=None):
    """
    Add all technical indicators to the DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLCV data
    column : str
        Column name for price calculations (default: 'close')
    datetime_col : str
        Name of datetime column (default: None)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with all technical indicators
    """
    # Make a copy of the DataFrame
    df = df.copy()
    
    # Ensure column names are lowercase
    df.columns = [col.lower() for col in df.columns]
    
    # Standardize datetime column name if provided
    if datetime_col:
        datetime_col = datetime_col.lower()
    
    # Add moving averages
    df = add_sma(df, column, 20)
    df = add_sma(df, column, 50)
    df = add_sma(df, column, 200)
    
    df = add_ema(df, column, 9)
    df = add_ema(df, column, 20)
    df = add_ema(df, column, 50)
    
    # Add momentum indicators
    df = add_rsi(df, column, 14)
    
    # Add volatility indicators
    df = add_bollinger_bands(df, column, 20, 2)
    
    # Add volume indicators if volume column exists
    if 'volume' in df.columns:
        df = add_vwap(df, datetime_col)
    
    # Add trend indicators
    df = add_supertrend(df, 10, 3)
    
    # Add support and resistance
    df = identify_support_resistance(df, 5, 0.01)
    
    return df

# Plotting function removed to simplify code

def apply_technical_analysis(filepath, save_to=None):
    """
    Apply all technical indicators to a data file
    
    Parameters:
    -----------
    filepath : str
        Path to CSV file with OHLCV data
    save_to : str
        Path to save processed data (default: None)
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with all technical indicators
    """
    # Read the data
    df = pd.read_csv(filepath)
    
    # Check if datetime column exists
    datetime_col = None
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            datetime_col = col
            break
    
    # Format datetime column if it exists
    if datetime_col:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df.set_index(datetime_col, inplace=True)
    
    # Standardize column names
    standard_columns = {
        'open': ['open', 'o', 'open price'],
        'high': ['high', 'h', 'high price'],
        'low': ['low', 'l', 'low price'],
        'close': ['close', 'c', 'close price'],
        'volume': ['volume', 'vol', 'v']
    }
    
    rename_dict = {}
    for std_name, variations in standard_columns.items():
        for col in df.columns:
            if col.lower() in variations:
                rename_dict[col] = std_name
    
    # Rename columns if matches found
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
    
    # Apply all indicators
    df_indicators = add_all_indicators(df, 'close')
    
    # Save processed data if path provided
    if save_to:
        df_indicators.to_csv(save_to)
    
    return df_indicators

# Example usage
if __name__ == "__main__":
    # Define the path to your data file
    data_file = "/Users/architmittal/Downloads/sample/module2/data_files/SBIN_20250415.csv"
    
    # Process the file
    df = apply_technical_analysis(
        data_file, 
        save_to="/Users/architmittal/Downloads/sample/module2/SBIN_indicators.csv"
    )
    
    print(f"Processed data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Indicators added: {', '.join([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])}")


# Assignment 
# 1. Add a function to calculate the Average True Range (ATR) and add it to the DataFrame.
# 2. You can create your own favourite data indicator 

