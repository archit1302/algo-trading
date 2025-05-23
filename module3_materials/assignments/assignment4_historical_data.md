# Assignment 4: Historical Data Fetching with Upstox API v3

In this assignment, you will implement functionality to fetch historical candlestick data using the Upstox API v3. You'll create a module for downloading, processing, and saving historical market data for different timeframes.

## Objectives

1. Create a module for fetching historical candlestick data from Upstox API v3
2. Implement different timeframe support (minutes, hours, days)
3. Process and save data in a useful format
4. Handle API rate limiting and errors

## Tasks

### 1. Set Up Project Structure

Create a project structure:
```
upstox-historical-data/
├── .env
├── config.py
├── upstox_auth.py
├── instrument_mapper.py
├── historical_downloader.py
└── main.py
```

### 2. Implement Historical Data Downloader

Create `historical_downloader.py`:

```python
"""
Module for downloading historical data from Upstox API v3
"""
import os
import json
import time
import pandas as pd
from datetime import datetime, timedelta
import requests

class HistoricalDataDownloader:
    """
    Class for downloading historical data from Upstox API v3
    """
    
    def __init__(self, access_token, base_url="https://api.upstox.com/v3"):
        """
        Initialize the historical data downloader
        
        Args:
            access_token (str): Access token for Upstox API
            base_url (str): Base URL for the Upstox API
        """
        self.access_token = access_token
        self.base_url = base_url
        self.headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        
        # Create data directory if it doesn't exist
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        os.makedirs(self.data_dir, exist_ok=True)
    
    def get_historical_data(self, instrument_key, interval=1, unit="minute", 
                           from_date=None, to_date=None, retry_count=3):
        """
        Get historical candlestick data
        
        Args:
            instrument_key (str): Instrument key
            interval (int): Candle interval (1, 2, 3, 5, 10, etc.)
            unit (str): Candle unit (minute, hour, day, week, month)
            from_date (str): Start date in YYYY-MM-DD format
            to_date (str): End date in YYYY-MM-DD format
            retry_count (int): Number of retries on failure
            
        Returns:
            dict: API response
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
                # For minutes, default to 1 month back (Upstox limit)
                from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            elif unit == "hour":
                # For hours, default to 3 months back (Upstox limit)
                from_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            else:
                # For day/week/month, default to 1 year back
                from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        # Build URL
        endpoint = f"/historical-candle/{instrument_key}/{unit}/{interval}"
        url = f"{self.base_url}{endpoint}"
        
        # Add query parameters
        params = {
            "to_date": to_date,
            "from_date": from_date
        }
        
        # Make the request with retries
        for attempt in range(retry_count):
            try:
                response = requests.get(url, headers=self.headers, params=params)
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 60))
                    print(f"Rate limited. Waiting for {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                response.raise_for_status()
                return response.json()
                
            except requests.RequestException as e:
                print(f"Request failed (attempt {attempt+1}/{retry_count}): {e}")
                if attempt < retry_count - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    def get_candles_as_dataframe(self, instrument_key, interval=1, unit="minute", 
                                from_date=None, to_date=None):
        """
        Get historical candles as pandas DataFrame
        
        Args:
            instrument_key (str): Instrument key
            interval (int): Candle interval
            unit (str): Candle unit
            from_date (str): Start date
            to_date (str): End date
            
        Returns:
            pandas.DataFrame: DataFrame with OHLC data
        """
        response = self.get_historical_data(
            instrument_key, interval, unit, from_date, to_date
        )
        
        if response.get('status') != 'success':
            error_msg = response.get('error', {}).get('message', 'Unknown error')
            raise ValueError(f"API request failed: {error_msg}")
        
        candles = response.get('data', {}).get('candles', [])
        
        if not candles:
            return pd.DataFrame()
        
        # Convert candles to DataFrame
        df = pd.DataFrame(candles, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Convert numeric columns
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'oi']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
        
        return df
    
    def save_to_csv(self, df, instrument_key, interval, unit, output_dir=None):
        """
        Save DataFrame to CSV file
        
        Args:
            df (pandas.DataFrame): DataFrame to save
            instrument_key (str): Instrument key
            interval (int): Candle interval
            unit (str): Candle unit
            output_dir (str): Output directory
            
        Returns:
            str: Path to the saved file
        """
        if df.empty:
            print("No data to save")
            return None
        
        # Clean instrument key for filename
        clean_key = instrument_key.replace('|', '_').replace(':', '_')
        
        # Format timeframe for filename
        timeframe = f"{interval}{unit}"
        
        # Get date range for filename
        start_date = df.index.min().strftime("%Y%m%d")
        end_date = df.index.max().strftime("%Y%m%d")
        
        # Create filename
        filename = f"{clean_key}_{timeframe}_{start_date}_to_{end_date}.csv"
        
        # Use provided output directory or default
        if output_dir is None:
            output_dir = self.data_dir
            
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        # Save to CSV
        df.to_csv(filepath)
        print(f"Saved data to {filepath}")
        
        return filepath
    
    def download_and_save(self, instrument_key, interval=1, unit="minute", 
                          from_date=None, to_date=None, output_dir=None):
        """
        Download data and save to CSV in one step
        
        Args:
            instrument_key (str): Instrument key
            interval (int): Candle interval
            unit (str): Candle unit
            from_date (str): Start date
            to_date (str): End date
            output_dir (str): Output directory
            
        Returns:
            str: Path to the saved file
        """
        try:
            df = self.get_candles_as_dataframe(
                instrument_key, interval, unit, from_date, to_date
            )
            
            return self.save_to_csv(df, instrument_key, interval, unit, output_dir)
            
        except Exception as e:
            print(f"Error downloading data: {e}")
            return None
```

### 3. Create Main Script

Create `main.py` to test the historical data downloader:

```python
"""
Script to download historical data from Upstox API v3
"""
import os
import sys
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Import your modules
from upstox_auth import UpstoxAuth
from instrument_mapper import InstrumentMapper
from historical_downloader import HistoricalDataDownloader

# Load environment variables
load_dotenv()

def main():
    """Main function to download historical data"""
    
    # Check for saved access token
    token_file = "token.json"
    access_token = None
    
    if os.path.exists(token_file):
        try:
            with open(token_file, 'r') as f:
                token_data = json.load(f)
                access_token = token_data.get('access_token')
        except Exception as e:
            print(f"Error loading token: {e}")
    
    if not access_token:
        print("No access token found. Please run the authentication script first.")
        sys.exit(1)
    
    # Create instrument mapper and historical data downloader
    mapper = InstrumentMapper()
    downloader = HistoricalDataDownloader(access_token)
    
    # Example usage
    symbol_examples = [
        {"exchange": "NSE", "symbol": "SBIN"},
        {"exchange": "NSE", "symbol": "RELIANCE"},
        {"exchange": "NSE", "symbol": "INFY"}
    ]
    
    # Download data for each symbol
    for example in symbol_examples:
        exchange = example["exchange"]
        symbol = example["symbol"]
        
        print(f"\nProcessing {exchange}:{symbol}")
        
        # Get instrument key from mapper
        instrument_key = mapper.get_instrument_key(exchange, symbol)
        
        if not instrument_key:
            print(f"Could not find instrument key for {exchange}:{symbol}")
            continue
        
        print(f"Found instrument key: {instrument_key}")
        
        # Download daily data for the past year
        print(f"Downloading daily data for {symbol}...")
        daily_file = downloader.download_and_save(
            instrument_key=instrument_key,
            interval=1,
            unit="day",
            from_date=(datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d"),
            to_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Download hourly data for the past month
        print(f"Downloading hourly data for {symbol}...")
        hourly_file = downloader.download_and_save(
            instrument_key=instrument_key,
            interval=1,
            unit="hour",
            from_date=(datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d"),
            to_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        # Download 5-minute data for the past week
        print(f"Downloading 5-minute data for {symbol}...")
        five_min_file = downloader.download_and_save(
            instrument_key=instrument_key,
            interval=5,
            unit="minute",
            from_date=(datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            to_date=datetime.now().strftime("%Y-%m-%d")
        )
        
        print(f"\nCompleted download for {symbol}")
        print(f"Daily data: {daily_file}")
        print(f"Hourly data: {hourly_file}")
        print(f"5-minute data: {five_min_file}")

if __name__ == "__main__":
    main()
```

### 4. Create Upstox Auth Module

Create a simplified `upstox_auth.py` for authentication:

```python
"""
Module for Upstox API authentication
"""
import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class UpstoxAuth:
    """
    Class for handling Upstox API authentication
    """
    
    def __init__(self):
        """Initialize authentication module"""
        self.api_key = os.getenv("UPSTOX_API_KEY")
        self.api_secret = os.getenv("UPSTOX_API_SECRET")
        self.redirect_uri = os.getenv("UPSTOX_REDIRECT_URI")
        self.auth_url = "https://api.upstox.com/v2/login/authorization/dialog"
        self.token_url = "https://api.upstox.com/v2/login/authorization/token"
        self.token_file = "token.json"
        
        # Load saved token if available
        self.access_token = None
        self.refresh_token = None
        self.load_token()
    
    def load_token(self):
        """Load token from file"""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'r') as f:
                    token_data = json.load(f)
                    self.access_token = token_data.get('access_token')
                    self.refresh_token = token_data.get('refresh_token')
            except Exception as e:
                print(f"Error loading token: {e}")
    
    def get_auth_url(self):
        """Get the authorization URL"""
        return f"{self.auth_url}?response_type=code&client_id={self.api_key}&redirect_uri={self.redirect_uri}"
    
    def exchange_code_for_token(self, auth_code):
        """
        Exchange authorization code for access token
        
        Args:
            auth_code (str): Authorization code from callback
            
        Returns:
            tuple: (success, result)
        """
        payload = {
            'code': auth_code,
            'client_id': self.api_key,
            'client_secret': self.api_secret,
            'redirect_uri': self.redirect_uri,
            'grant_type': 'authorization_code'
        }
        
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        try:
            response = requests.post(self.token_url, data=payload, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            self.access_token = data.get('access_token')
            self.refresh_token = data.get('refresh_token')
            
            # Save tokens to file
            with open(self.token_file, 'w') as f:
                json.dump({
                    'access_token': self.access_token,
                    'refresh_token': self.refresh_token
                }, f)
            
            return True, data
        
        except Exception as e:
            return False, str(e)
    
    def get_headers(self):
        """Get headers for authenticated API calls"""
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}" if self.access_token else None
        }
```

### 5. Running the Code

1. Ensure you've completed the authentication assignment and have a valid access token stored in `token.json`.

2. Run the script:
   ```bash
   python main.py
   ```

3. Check the output directory for downloaded CSV files of different timeframes.

### 6. Additional Tasks

1. Expand the script to handle more complex date ranges and symbols
2. Add error handling for API rate limits and other issues
3. Implement a command-line interface for selecting symbols and timeframes

## Submission Guidelines

Create a ZIP file containing all the code files you've created, ensuring that you:
1. Have not included your actual API credentials in the files (use `.env` for local storage)
2. Have not included any downloaded data files (add `*.csv` to `.gitignore`)
3. Include a brief README.md explaining how to run your code and the functionality implemented

## Evaluation Criteria

1. Correct implementation of historical data fetching from Upstox API v3
2. Proper handling of different timeframes and date ranges
3. Error handling, including rate limiting and retries
4. Code organization and documentation

## Helpful Resources

- [Upstox Historical Candle Data V3 API Documentation](https://upstox.com/developer/api-documentation/v3/get-historical-candle-data)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Python Requests Library](https://docs.python-requests.org/)
