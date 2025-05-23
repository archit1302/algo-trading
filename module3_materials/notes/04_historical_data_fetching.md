# Historical Data Fetching with Upstox API v3

This note covers how to fetch historical candlestick data using the Upstox API v3, including different timeframes, date ranges, and data processing techniques.

## Historical Data Overview

Historical data in trading refers to past price and volume information for financial instruments. This data is essential for:

- Technical analysis and indicator calculations
- Strategy backtesting
- Market research and analysis
- Pattern recognition

## Upstox Historical Data API

The Upstox API v3 provides historical candlestick data through the `/historical-candle` endpoint.

### API Endpoint Structure

```
GET https://api.upstox.com/v3/historical-candle/{instrument_key}/{unit}/{interval}
```

Parameters:
- `instrument_key`: Unique identifier for the instrument
- `unit`: Time unit (minute, hour, day, week, month)
- `interval`: Interval value (1, 2, 3, 5, 10, 15, 30, etc.)

Query Parameters:
- `from_date`: Start date in YYYY-MM-DD format
- `to_date`: End date in YYYY-MM-DD format

## Supported Timeframes

### Available Units and Intervals

| Unit | Supported Intervals | Use Cases |
|------|-------------------|-----------|
| minute | 1, 2, 3, 5, 10, 15, 30 | Intraday trading, scalping |
| hour | 1, 2, 4 | Short-term analysis |
| day | 1 | Daily analysis, swing trading |
| week | 1 | Weekly analysis |
| month | 1 | Long-term analysis |

### Data Limitations

- **Minute data**: Limited to 30 days of history
- **Hour data**: Limited to 90 days of history  
- **Daily data**: Available for several years
- **Weekly/Monthly**: Available for extended periods

## Basic Data Fetching

### Simple Historical Data Request

```python
import requests
import json
from datetime import datetime, timedelta

def fetch_historical_data(access_token, instrument_key, interval=1, unit="day", 
                         from_date=None, to_date=None):
    """
    Fetch historical data from Upstox API
    
    Args:
        access_token (str): Valid access token
        instrument_key (str): Instrument key
        interval (int): Candle interval
        unit (str): Time unit
        from_date (str): Start date (YYYY-MM-DD)
        to_date (str): End date (YYYY-MM-DD)
    
    Returns:
        dict: API response
    """
    
    # Set default dates if not provided
    if to_date is None:
        to_date = datetime.now().strftime("%Y-%m-%d")
    
    if from_date is None:
        # Default to 1 month back for daily data
        from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    
    # Build URL
    url = f"https://api.upstox.com/v3/historical-candle/{instrument_key}/{unit}/{interval}"
    
    # Headers
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}"
    }
    
    # Query parameters
    params = {
        "from_date": from_date,
        "to_date": to_date
    }
    
    try:
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        
        return response.json()
        
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None

# Usage example
data = fetch_historical_data(
    access_token="your_token",
    instrument_key="NSE_EQ|INE062A01020|SBIN",
    interval=1,
    unit="day",
    from_date="2024-01-01",
    to_date="2024-12-31"
)

if data and data.get('status') == 'success':
    candles = data['data']['candles']
    print(f"Fetched {len(candles)} candles")
```

## Data Processing with Pandas

### Converting to DataFrame

```python
import pandas as pd

def candles_to_dataframe(candles_data):
    """
    Convert Upstox candles data to pandas DataFrame
    
    Args:
        candles_data (list): List of candle arrays
    
    Returns:
        pandas.DataFrame: OHLCV DataFrame
    """
    if not candles_data:
        return pd.DataFrame()
    
    # Create DataFrame from candles
    df = pd.DataFrame(candles_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'
    ])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Convert to numeric types
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'oi']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    
    # Sort by timestamp (oldest first)
    df.sort_index(inplace=True)
    
    return df

# Usage
response = fetch_historical_data(access_token, instrument_key)
if response and response.get('status') == 'success':
    candles = response['data']['candles']
    df = candles_to_dataframe(candles)
    
    print(f"Data shape: {df.shape}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")
    print(df.head())
```

## Advanced Data Fetching

### Handling Rate Limits and Retries

```python
import time

def fetch_with_retry(access_token, instrument_key, interval=1, unit="day", 
                    from_date=None, to_date=None, max_retries=3):
    """
    Fetch historical data with retry logic for rate limiting
    """
    
    for attempt in range(max_retries):
        try:
            response = fetch_historical_data(
                access_token, instrument_key, interval, unit, from_date, to_date
            )
            
            if response:
                return response
                
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Rate limited
                retry_after = int(e.response.headers.get('Retry-After', 60))
                print(f"Rate limited. Waiting {retry_after} seconds...")
                time.sleep(retry_after)
            else:
                print(f"HTTP error (attempt {attempt + 1}): {e}")
                
        except Exception as e:
            print(f"Error (attempt {attempt + 1}): {e}")
            
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)  # Exponential backoff
    
    return None
```

### Batch Data Fetching

```python
def fetch_multiple_symbols(access_token, symbol_configs, delay=1):
    """
    Fetch historical data for multiple symbols
    
    Args:
        access_token (str): Access token
        symbol_configs (list): List of config dictionaries
        delay (float): Delay between requests (seconds)
    
    Returns:
        dict: Symbol -> DataFrame mapping
    """
    results = {}
    
    for config in symbol_configs:
        symbol = config['symbol']
        instrument_key = config['instrument_key']
        
        print(f"Fetching data for {symbol}...")
        
        try:
            response = fetch_with_retry(
                access_token,
                instrument_key,
                config.get('interval', 1),
                config.get('unit', 'day'),
                config.get('from_date'),
                config.get('to_date')
            )
            
            if response and response.get('status') == 'success':
                candles = response['data']['candles']
                df = candles_to_dataframe(candles)
                results[symbol] = df
                print(f"  ✓ Fetched {len(df)} records")
            else:
                print(f"  ✗ Failed to fetch data")
                results[symbol] = pd.DataFrame()
                
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results[symbol] = pd.DataFrame()
        
        # Add delay to respect rate limits
        time.sleep(delay)
    
    return results

# Usage example
symbols_config = [
    {
        'symbol': 'SBIN',
        'instrument_key': 'NSE_EQ|INE062A01020|SBIN',
        'interval': 1,
        'unit': 'day',
        'from_date': '2024-01-01'
    },
    {
        'symbol': 'RELIANCE',
        'instrument_key': 'NSE_EQ|INE002A01018|RELIANCE',
        'interval': 5,
        'unit': 'minute',
        'from_date': '2024-05-01'
    }
]

data_dict = fetch_multiple_symbols(access_token, symbols_config)
```

## Data Validation and Cleaning

### Basic Data Validation

```python
def validate_ohlc_data(df):
    """
    Validate OHLC data for consistency
    
    Args:
        df (DataFrame): OHLC DataFrame
    
    Returns:
        tuple: (is_valid, issues)
    """
    issues = []
    
    if df.empty:
        return False, ["DataFrame is empty"]
    
    # Check for required columns
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check OHLC relationships
    invalid_high = (df['high'] < df['open']) | (df['high'] < df['close']) | \
                   (df['high'] < df['low'])
    if invalid_high.any():
        issues.append(f"Invalid high prices in {invalid_high.sum()} rows")
    
    invalid_low = (df['low'] > df['open']) | (df['low'] > df['close']) | \
                  (df['low'] > df['high'])
    if invalid_low.any():
        issues.append(f"Invalid low prices in {invalid_low.sum()} rows")
    
    # Check for negative values
    negative_prices = (df[['open', 'high', 'low', 'close']] < 0).any(axis=1)
    if negative_prices.any():
        issues.append(f"Negative prices in {negative_prices.sum()} rows")
    
    # Check for zero volume (might be valid for some instruments)
    zero_volume = (df['volume'] == 0).sum()
    if zero_volume > 0:
        issues.append(f"Zero volume in {zero_volume} rows (might be normal)")
    
    return len(issues) == 0, issues

# Usage
is_valid, issues = validate_ohlc_data(df)
if not is_valid:
    print("Data validation issues:")
    for issue in issues:
        print(f"  - {issue}")
```

### Data Cleaning

```python
def clean_ohlc_data(df):
    """
    Clean OHLC data by handling common issues
    
    Args:
        df (DataFrame): Raw OHLC DataFrame
    
    Returns:
        DataFrame: Cleaned DataFrame
    """
    df_clean = df.copy()
    
    # Remove rows with any NaN values in OHLC columns
    ohlc_cols = ['open', 'high', 'low', 'close']
    df_clean = df_clean.dropna(subset=ohlc_cols)
    
    # Remove rows with invalid OHLC relationships
    valid_high = (df_clean['high'] >= df_clean['open']) & \
                 (df_clean['high'] >= df_clean['close']) & \
                 (df_clean['high'] >= df_clean['low'])
    
    valid_low = (df_clean['low'] <= df_clean['open']) & \
                (df_clean['low'] <= df_clean['close']) & \
                (df_clean['low'] <= df_clean['high'])
    
    df_clean = df_clean[valid_high & valid_low]
    
    # Remove rows with negative or zero prices
    positive_prices = (df_clean[ohlc_cols] > 0).all(axis=1)
    df_clean = df_clean[positive_prices]
    
    print(f"Data cleaning: {len(df)} -> {len(df_clean)} rows")
    
    return df_clean
```

## Saving and Loading Data

### Saving to CSV

```python
def save_ohlc_to_csv(df, symbol, timeframe, output_dir="data"):
    """
    Save OHLC data to CSV file with standardized naming
    
    Args:
        df (DataFrame): OHLC DataFrame
        symbol (str): Symbol name
        timeframe (str): Timeframe (e.g., "1day", "5minute")
        output_dir (str): Output directory
    
    Returns:
        str: File path
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename with date range
    start_date = df.index.min().strftime("%Y%m%d")
    end_date = df.index.max().strftime("%Y%m%d")
    
    filename = f"{symbol}_{timeframe}_{start_date}_to_{end_date}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Save with timestamp as first column
    df.to_csv(filepath)
    
    print(f"Saved {len(df)} records to {filepath}")
    return filepath
```

### Loading from CSV

```python
def load_ohlc_from_csv(filepath):
    """
    Load OHLC data from CSV file
    
    Args:
        filepath (str): Path to CSV file
    
    Returns:
        DataFrame: OHLC DataFrame
    """
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        
        # Ensure numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        available_cols = [col for col in numeric_cols if col in df.columns]
        df[available_cols] = df[available_cols].apply(pd.to_numeric)
        
        print(f"Loaded {len(df)} records from {filepath}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()
```

## Working with Different Timeframes

### Timeframe Conversion Examples

```python
# Daily data for long-term analysis
daily_data = fetch_historical_data(
    access_token, instrument_key,
    interval=1, unit="day",
    from_date="2023-01-01"
)

# Hourly data for short-term analysis
hourly_data = fetch_historical_data(
    access_token, instrument_key,
    interval=1, unit="hour",
    from_date="2024-05-01"
)

# 5-minute data for intraday analysis
intraday_data = fetch_historical_data(
    access_token, instrument_key,
    interval=5, unit="minute",
    from_date="2024-05-20"
)
```

## Best Practices

1. **Respect Rate Limits**: Implement delays between requests
2. **Error Handling**: Always handle API errors gracefully
3. **Data Validation**: Validate OHLC data after fetching
4. **Caching**: Save data locally to reduce API calls
5. **Incremental Updates**: Fetch only new data when possible
6. **Timeframe Awareness**: Choose appropriate timeframes for your use case
7. **Data Quality**: Clean and validate data before analysis

## Common Use Cases

### Strategy Backtesting
```python
# Fetch daily data for backtesting
backtest_data = fetch_historical_data(
    access_token, instrument_key,
    interval=1, unit="day",
    from_date="2022-01-01",
    to_date="2024-01-01"
)
```

### Technical Indicator Calculation
```python
# Fetch minute data for real-time indicators
indicator_data = fetch_historical_data(
    access_token, instrument_key,
    interval=1, unit="minute",
    from_date=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
)
```

## Key Takeaways

- Upstox API v3 provides comprehensive historical data access
- Different timeframes have different data availability limits
- Always implement proper error handling and rate limiting
- Validate and clean data before analysis
- Use pandas for efficient data processing and analysis
- Save data locally to reduce API calls and improve performance
