"""
Solution for Assignment 4: Historical Data Fetching
This script demonstrates fetching historical candlestick data using Upstox API v3.
"""

import os
import sys
import json
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class UpstoxHistoricalDataFetcher:
    """
    Historical data fetcher for Upstox API v3
    """
    
    def __init__(self, access_token):
        """
        Initialize the historical data fetcher
        
        Args:
            access_token (str): Valid Upstox access token
        """
        self.access_token = access_token
        self.base_url = "https://api.upstox.com/v3"
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        
        # Create data directory
        self.data_dir = "historical_data"
        os.makedirs(self.data_dir, exist_ok=True)
    
    def fetch_historical_data(self, instrument_key, interval=1, unit="day", 
                             from_date=None, to_date=None, retry_count=3):
        """
        Fetch historical candlestick data from Upstox API
        
        Args:
            instrument_key (str): Instrument key
            interval (int): Candle interval (1, 2, 3, 5, 10, etc.)
            unit (str): Candle unit (minute, hour, day, week, month)
            from_date (str): Start date in YYYY-MM-DD format
            to_date (str): End date in YYYY-MM-DD format
            retry_count (int): Number of retries on failure
            
        Returns:
            dict: API response or None if failed
        """
        # Validate parameters
        valid_units = ["minute", "hour", "day", "week", "month"]
        if unit not in valid_units:
            raise ValueError(f"Invalid unit: {unit}. Must be one of {valid_units}")
        
        # Set default dates if not provided
        if to_date is None:
            to_date = datetime.now().strftime("%Y-%m-%d")
            
        if from_date is None:
            # Default to different ranges based on unit
            if unit == "minute":
                from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            elif unit == "hour":
                from_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            else:
                from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        # Build URL
        endpoint = f"/historical-candle/{instrument_key}/{unit}/{interval}"
        url = f"{self.base_url}{endpoint}"
        
        # Query parameters
        params = {
            "to_date": to_date,
            "from_date": from_date
        }
        
        print(f"📡 Fetching {interval}{unit} data for {instrument_key}")
        print(f"   📅 Date range: {from_date} to {to_date}")
        
        # Make request with retries
        for attempt in range(retry_count):
            try:
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    print(f"⏱️  Rate limited. Waiting for {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                data = response.json()
                
                if data.get('status') == 'success':
                    candles = data.get('data', {}).get('candles', [])
                    print(f"✅ Fetched {len(candles)} candles")
                    return data
                else:
                    error_msg = data.get('error', {}).get('message', 'Unknown error')
                    print(f"❌ API error: {error_msg}")
                    return None
                
            except requests.RequestException as e:
                print(f"⚠️  Request failed (attempt {attempt+1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    print(f"⏳ Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Failed after {retry_count} attempts")
                    return None
        
        return None
    
    def convert_to_dataframe(self, api_response):
        """
        Convert API response to pandas DataFrame
        
        Args:
            api_response (dict): API response from fetch_historical_data
            
        Returns:
            pandas.DataFrame: OHLCV DataFrame
        """
        if not api_response or api_response.get('status') != 'success':
            return pd.DataFrame()
        
        candles = api_response.get('data', {}).get('candles', [])
        
        if not candles:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'oi']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
        
        # Sort by timestamp (oldest first)
        df.sort_index(inplace=True)
        
        return df
    
    def validate_ohlc_data(self, df):
        """
        Validate OHLC data integrity
        
        Args:
            df (DataFrame): OHLC DataFrame
            
        Returns:
            tuple: (is_valid, issues_list)
        """
        issues = []
        
        if df.empty:
            return False, ["DataFrame is empty"]
        
        # Check OHLC relationships
        invalid_high = (df['high'] < df['open']) | (df['high'] < df['close']) | (df['high'] < df['low'])
        if invalid_high.any():
            issues.append(f"Invalid high prices in {invalid_high.sum()} rows")
        
        invalid_low = (df['low'] > df['open']) | (df['low'] > df['close']) | (df['low'] > df['high'])
        if invalid_low.any():
            issues.append(f"Invalid low prices in {invalid_low.sum()} rows")
        
        # Check for negative values
        negative_prices = (df[['open', 'high', 'low', 'close']] <= 0).any(axis=1)
        if negative_prices.any():
            issues.append(f"Non-positive prices in {negative_prices.sum()} rows")
        
        # Check for missing data
        missing_data = df[['open', 'high', 'low', 'close']].isna().any(axis=1)
        if missing_data.any():
            issues.append(f"Missing OHLC data in {missing_data.sum()} rows")
        
        return len(issues) == 0, issues
    
    def save_to_csv(self, df, symbol, timeframe, output_dir=None):
        """
        Save DataFrame to CSV file with standardized naming
        
        Args:
            df (DataFrame): OHLC DataFrame
            symbol (str): Symbol name
            timeframe (str): Timeframe string (e.g., "1day", "5minute")
            output_dir (str): Output directory
            
        Returns:
            str: Path to saved file or None if failed
        """
        if df.empty:
            print("⚠️  No data to save")
            return None
        
        # Use provided output directory or default
        if output_dir is None:
            output_dir = self.data_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Clean symbol for filename
        clean_symbol = symbol.replace('|', '_').replace(':', '_')
        
        # Get date range for filename
        start_date = df.index.min().strftime("%Y%m%d")
        end_date = df.index.max().strftime("%Y%m%d")
        
        # Create filename
        filename = f"{clean_symbol}_{timeframe}_{start_date}_to_{end_date}.csv"
        filepath = os.path.join(output_dir, filename)
        
        try:
            # Save to CSV
            df.to_csv(filepath)
            print(f"💾 Saved {len(df)} records to {filepath}")
            return filepath
        
        except Exception as e:
            print(f"❌ Error saving file: {e}")
            return None
    
    def fetch_and_save(self, instrument_key, symbol, interval=1, unit="day", 
                      from_date=None, to_date=None, validate_data=True):
        """
        Fetch data and save to CSV in one operation
        
        Args:
            instrument_key (str): Instrument key
            symbol (str): Symbol name for filename
            interval (int): Candle interval
            unit (str): Time unit
            from_date (str): Start date
            to_date (str): End date
            validate_data (bool): Whether to validate data
            
        Returns:
            tuple: (DataFrame, filepath)
        """
        # Fetch data
        response = self.fetch_historical_data(
            instrument_key, interval, unit, from_date, to_date
        )
        
        if not response:
            return pd.DataFrame(), None
        
        # Convert to DataFrame
        df = self.convert_to_dataframe(response)
        
        if df.empty:
            print("⚠️  No data returned from API")
            return df, None
        
        # Validate data if requested
        if validate_data:
            is_valid, issues = self.validate_ohlc_data(df)
            if not is_valid:
                print("⚠️  Data validation issues:")
                for issue in issues:
                    print(f"   • {issue}")
        
        # Save to CSV
        timeframe = f"{interval}{unit}"
        filepath = self.save_to_csv(df, symbol, timeframe)
        
        return df, filepath
    
    def fetch_multiple_timeframes(self, instrument_key, symbol, timeframes):
        """
        Fetch multiple timeframes for a single symbol
        
        Args:
            instrument_key (str): Instrument key
            symbol (str): Symbol name
            timeframes (list): List of timeframe dictionaries
            
        Returns:
            dict: Timeframe -> DataFrame mapping
        """
        results = {}
        
        print(f"\n📊 Fetching multiple timeframes for {symbol}")
        print("=" * 50)
        
        for tf in timeframes:
            interval = tf['interval']
            unit = tf['unit']
            timeframe_key = f"{interval}{unit}"
            
            try:
                df, filepath = self.fetch_and_save(
                    instrument_key, symbol, interval, unit,
                    tf.get('from_date'), tf.get('to_date')
                )
                
                results[timeframe_key] = {
                    'data': df,
                    'filepath': filepath,
                    'records': len(df)
                }
                
                # Add delay between requests
                time.sleep(1)
                
            except Exception as e:
                print(f"❌ Error fetching {timeframe_key}: {e}")
                results[timeframe_key] = {
                    'data': pd.DataFrame(),
                    'filepath': None,
                    'error': str(e)
                }
        
        return results
    
    def create_data_summary(self, results):
        """
        Create a summary of fetched data
        
        Args:
            results (dict): Results from fetch operations
            
        Returns:
            DataFrame: Summary DataFrame
        """
        summary_data = []
        
        for symbol, timeframe_data in results.items():
            if isinstance(timeframe_data, dict):
                for timeframe, data in timeframe_data.items():
                    if isinstance(data, dict) and 'data' in data:
                        df = data['data']
                        if not df.empty:
                            summary_data.append({
                                'symbol': symbol,
                                'timeframe': timeframe,
                                'records': len(df),
                                'start_date': df.index.min().strftime("%Y-%m-%d"),
                                'end_date': df.index.max().strftime("%Y-%m-%d"),
                                'filepath': data.get('filepath', '')
                            })
        
        return pd.DataFrame(summary_data)

def demonstrate_historical_data_fetching():
    """
    Demonstrate historical data fetching capabilities
    """
    print("Historical Data Fetching - Assignment 4 Solution")
    print("=" * 55)
    
    # Check for access token
    access_token = os.getenv("UPSTOX_ACCESS_TOKEN")
    
    if not access_token:
        # Try to load from token file
        token_file = "upstox_token.json"
        if os.path.exists(token_file):
            try:
                with open(token_file, 'r') as f:
                    token_data = json.load(f)
                    access_token = token_data.get('access_token')
            except Exception as e:
                print(f"❌ Error loading token: {e}")
    
    if not access_token:
        print("❌ No access token found!")
        print("   Please set UPSTOX_ACCESS_TOKEN environment variable")
        print("   Or run the authentication script first")
        return
    
    # Initialize fetcher
    fetcher = UpstoxHistoricalDataFetcher(access_token)
    
    # Define test symbols and their instrument keys
    test_symbols = [
        {
            'symbol': 'SBIN',
            'instrument_key': 'NSE_EQ|INE062A01020|SBIN',
            'timeframes': [
                {'interval': 1, 'unit': 'day', 'from_date': '2024-01-01'},
                {'interval': 1, 'unit': 'hour', 'from_date': '2024-05-01'},
                {'interval': 5, 'unit': 'minute', 'from_date': '2024-05-20'}
            ]
        },
        {
            'symbol': 'RELIANCE',
            'instrument_key': 'NSE_EQ|INE002A01018|RELIANCE',
            'timeframes': [
                {'interval': 1, 'unit': 'day', 'from_date': '2024-01-01'},
                {'interval': 15, 'unit': 'minute', 'from_date': '2024-05-20'}
            ]
        }
    ]
    
    # Fetch data for all symbols
    all_results = {}
    
    for symbol_config in test_symbols:
        symbol = symbol_config['symbol']
        instrument_key = symbol_config['instrument_key']
        timeframes = symbol_config['timeframes']
        
        print(f"\n🔄 Processing {symbol}...")
        results = fetcher.fetch_multiple_timeframes(instrument_key, symbol, timeframes)
        all_results[symbol] = results
        
        # Show results for this symbol
        print(f"\n📋 Results for {symbol}:")
        for timeframe, data in results.items():
            if 'data' in data and not data['data'].empty:
                print(f"   ✅ {timeframe}: {data['records']} records")
            else:
                print(f"   ❌ {timeframe}: Failed")
    
    # Create and display summary
    print(f"\n📊 Data Fetching Summary")
    print("=" * 30)
    
    summary_df = fetcher.create_data_summary(all_results)
    if not summary_df.empty:
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_file = f"data_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        summary_df.to_csv(summary_file, index=False)
        print(f"\n💾 Summary saved to {summary_file}")
    else:
        print("No data was successfully fetched.")
    
    return all_results

def main():
    """
    Main function to run the historical data fetching demonstration
    """
    results = demonstrate_historical_data_fetching()
    
    print(f"\n📚 Key Learnings:")
    print("   • Historical data is limited by timeframe (minutes: 30 days, hours: 90 days)")
    print("   • Always validate OHLC data after fetching")
    print("   • Implement retry logic for robust data fetching")
    print("   • Save data locally to reduce API calls")
    print("   • Use appropriate delays between requests")
    print("   • Monitor rate limits and handle 429 responses")
    
    print(f"\n🔗 Next Steps:")
    print("   • Implement incremental data updates")
    print("   • Add data cleaning and preprocessing")
    print("   • Create automated data fetching schedules")
    print("   • Build technical indicators on the fetched data")

if __name__ == "__main__":
    main()
