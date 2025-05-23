# Assignment 5: Batch Processing with Upstox API

In this assignment, you will implement batch processing functionality to fetch historical data for multiple symbols and timeframes using the Upstox API v3. You'll create a configurable system that can download data for a list of symbols in parallel or sequentially.

## Objectives

1. Create a batch processing system for downloading data for multiple symbols
2. Support multiple timeframes for each symbol
3. Implement configurable downloads via a config file
4. Add proper error handling and logging

## Tasks

### 1. Set Up Project Structure

Create a project structure:
```
upstox-batch-processing/
├── .env
├── config.py
├── batch_downloader.py
├── symbol_mapper.py
├── download_historical_data.py
├── configs/
│   ├── nifty50.py
│   └── custom_symbols.py
└── run_download.py
```

### 2. Create Configuration Module

Create a `config.py` file for global settings:

```python
"""
Configuration settings for the batch downloader
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
API_KEY = os.getenv("UPSTOX_API_KEY")
API_SECRET = os.getenv("UPSTOX_API_SECRET")
REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI")
API_BASE_URL = "https://api.upstox.com/v3"

# Data Storage
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# Batch Processing Configuration
MAX_CONCURRENT_DOWNLOADS = 5  # Maximum number of concurrent downloads
RETRY_COUNT = 3               # Number of retries for failed downloads
RETRY_DELAY = 5               # Delay between retries in seconds
REQUEST_DELAY = 0.5           # Delay between API requests in seconds

# Default date ranges (will be used if not specified in the config)
DEFAULT_FROM_DATE = None      # None means auto-calculated based on timeframe
DEFAULT_TO_DATE = None        # None means today
```

### 3. Create a Sample Symbol Configuration

Create a sample configuration in `configs/nifty50.py`:

```python
"""
Configuration for downloading Nifty 50 symbols
"""

# List of Nifty 50 symbols (as of 2023)
SYMBOLS = [
    "ADANIPORTS", "ASIANPAINT", "AXISBANK", "BAJAJ-AUTO", "BAJFINANCE",
    "BAJAJFINSV", "BPCL", "BHARTIARTL", "BRITANNIA", "CIPLA",
    "COALINDIA", "DIVISLAB", "DRREDDY", "EICHERMOT", "GRASIM",
    "HCLTECH", "HDFCBANK", "HDFCLIFE", "HEROMOTOCO", "HINDALCO",
    "HINDUNILVR", "HDFC", "ICICIBANK", "ITC", "INDUSINDBK",
    "INFY", "JSWSTEEL", "KOTAKBANK", "LT", "M&M",
    "MARUTI", "NTPC", "NESTLEIND", "ONGC", "POWERGRID",
    "RELIANCE", "SBILIFE", "SBIN", "SUNPHARMA", "TCS",
    "TATACONSUM", "TATAMOTORS", "TATASTEEL", "TECHM", "TITAN",
    "UPL", "ULTRACEMCO", "WIPRO"
]

# Exchange for these symbols
EXCHANGE = "NSE"

# Timeframes to download for each symbol
# Format: (interval, unit)
TIMEFRAMES = [
    (5, "minute"),    # 5-minute candles
    (15, "minute"),   # 15-minute candles
    (30, "minute"),   # 30-minute candles
    (1, "hour"),      # 1-hour candles
    (1, "day"),       # Daily candles
]

# Date range
FROM_DATE = None  # Use default from config.py
TO_DATE = None    # Use default from config.py

# Whether to download in parallel
USE_PARALLEL = True
MAX_WORKERS = 5  # Maximum number of parallel downloads
```

### 4. Create Symbol Mapper Module

Create a simplified `symbol_mapper.py`:

```python
"""
Module for mapping between ticker symbols and Upstox instrument keys
"""
import os
import json
import requests
import gzip
import shutil
from datetime import datetime

class SymbolMapper:
    """
    Class for mapping between ticker symbols and Upstox instrument keys
    """
    
    def __init__(self):
        """Initialize the symbol mapper"""
        # Local paths
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
        self.instruments_dir = os.path.join(self.data_dir, "instruments")
        os.makedirs(self.instruments_dir, exist_ok=True)
        
        # URLs for instrument data
        self.instrument_urls = {
            "NSE": "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
            "BSE": "https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz",
            "MCX": "https://assets.upstox.com/market-quote/instruments/exchange/MCX.json.gz",
        }
        
        # Mapping dictionaries
        self.symbols_to_keys = {}
        self.keys_to_symbols = {}
        
        # Load instrument data
        self.ensure_fresh_data()
    
    def download_instruments(self, exchange):
        """
        Download instrument data for a specific exchange
        
        Args:
            exchange (str): Exchange code (NSE, BSE, MCX)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if exchange not in self.instrument_urls:
            print(f"Unknown exchange: {exchange}")
            return False
        
        url = self.instrument_urls[exchange]
        gz_filename = f"{exchange}.json.gz"
        json_filename = f"{exchange}_instruments.json"
        
        gz_path = os.path.join(self.instruments_dir, gz_filename)
        json_path = os.path.join(self.instruments_dir, json_filename)
        
        try:
            # Download the gzipped file
            print(f"Downloading {exchange} instruments...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(gz_path, 'wb') as f_out:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f_out.write(chunk)
            
            # Extract the gzipped file
            with gzip.open(gz_path, 'rb') as f_in:
                with open(json_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove the gzipped file
            os.remove(gz_path)
            
            print(f"Successfully downloaded {exchange} instruments")
            return True
            
        except Exception as e:
            print(f"Error downloading {exchange} instruments: {e}")
            return False
    
    def load_instruments(self, exchange):
        """
        Load instrument data for a specific exchange
        
        Args:
            exchange (str): Exchange code (NSE, BSE, MCX)
            
        Returns:
            bool: True if successful, False otherwise
        """
        json_filename = f"{exchange}_instruments.json"
        json_path = os.path.join(self.instruments_dir, json_filename)
        
        if not os.path.exists(json_path):
            print(f"Instrument file not found for {exchange}. Downloading...")
            if not self.download_instruments(exchange):
                return False
        
        try:
            with open(json_path, 'r') as f:
                instruments = json.load(f)
            
            # Build mapping dictionaries
            for instrument in instruments:
                if instrument.get('instrument_type') == 'EQ':  # Focus on equities
                    symbol = instrument.get('trading_symbol')
                    instrument_key = instrument.get('instrument_key')
                    
                    if symbol and instrument_key:
                        key = f"{exchange}:{symbol}"
                        self.symbols_to_keys[key] = instrument_key
                        self.keys_to_symbols[instrument_key] = key
            
            print(f"Loaded {len(instruments)} instruments for {exchange}")
            return True
            
        except Exception as e:
            print(f"Error loading {exchange} instruments: {e}")
            return False
    
    def ensure_fresh_data(self, max_age_days=1):
        """
        Ensure instrument data is up-to-date
        
        Args:
            max_age_days (int): Maximum age of instrument data in days
            
        Returns:
            bool: True if successful, False otherwise
        """
        success = True
        
        for exchange in self.instrument_urls.keys():
            json_filename = f"{exchange}_instruments.json"
            json_path = os.path.join(self.instruments_dir, json_filename)
            
            need_download = True
            
            # Check if file exists and is recent
            if os.path.exists(json_path):
                file_time = os.path.getmtime(json_path)
                file_age_days = (datetime.now() - datetime.fromtimestamp(file_time)).days
                
                if file_age_days < max_age_days:
                    need_download = False
            
            # Download if needed
            if need_download:
                if not self.download_instruments(exchange):
                    success = False
            
            # Load the data
            if not self.load_instruments(exchange):
                success = False
        
        return success
    
    def get_instrument_key(self, exchange, symbol):
        """
        Get instrument key from exchange and symbol
        
        Args:
            exchange (str): Exchange code (NSE, BSE, MCX)
            symbol (str): Trading symbol
            
        Returns:
            str: Instrument key or None if not found
        """
        key = f"{exchange}:{symbol}"
        return self.symbols_to_keys.get(key)
    
    def get_symbol(self, instrument_key):
        """
        Get symbol from instrument key
        
        Args:
            instrument_key (str): Instrument key
            
        Returns:
            tuple: (exchange, symbol) or (None, None) if not found
        """
        key = self.keys_to_symbols.get(instrument_key)
        
        if key:
            exchange, symbol = key.split(':')
            return exchange, symbol
        
        return None, None
```

### 5. Create Historical Data Downloader

Create `download_historical_data.py`:

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
import logging
from config import API_BASE_URL, RETRY_COUNT, RETRY_DELAY, REQUEST_DELAY, DATA_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HistoricalDataDownloader:
    """
    Class for downloading historical data from Upstox API v3
    """
    
    def __init__(self, config_module, symbol_mapper):
        """
        Initialize historical data downloader
        
        Args:
            config_module: Module with configuration settings
            symbol_mapper: SymbolMapper instance
        """
        self.config = config_module
        self.symbol_mapper = symbol_mapper
        self.access_token = getattr(config_module, 'ACCESS_TOKEN', None)
        self.data_dir = DATA_DIR
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
    
    def set_access_token(self, access_token):
        """Set API access token"""
        self.access_token = access_token
    
    def get_headers(self):
        """Get API request headers"""
        if not self.access_token:
            raise ValueError("Access token not set")
            
        return {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}"
        }
    
    def get_historical_data(self, instrument_key, interval=1, unit="minute", 
                            from_date=None, to_date=None):
        """
        Get historical candlestick data
        
        Args:
            instrument_key (str): Instrument key
            interval (int): Candle interval
            unit (str): Candle unit
            from_date (str): Start date in YYYY-MM-DD format
            to_date (str): End date in YYYY-MM-DD format
            
        Returns:
            dict: API response
        """
        # Set default dates if not provided
        if not to_date:
            to_date = datetime.now().strftime("%Y-%m-%d")
        
        if not from_date:
            # Default ranges based on unit and interval
            if unit == "minute":
                # For minutes, default to appropriate range based on interval
                days_back = 30 if interval <= 15 else 90
                from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
            elif unit == "hour":
                # For hours, default to 3 months back
                from_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
            else:
                # For day/week/month, default to 1 year back
                from_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        
        # Build URL
        endpoint = f"/historical-candle/{instrument_key}/{unit}/{interval}"
        url = f"{API_BASE_URL}{endpoint}"
        
        # Add query parameters
        params = {
            "to_date": to_date,
            "from_date": from_date
        }
        
        headers = self.get_headers()
        
        # Make request with retries
        for attempt in range(RETRY_COUNT):
            try:
                logger.info(f"Requesting data for {instrument_key}, {interval}{unit}, attempt {attempt+1}/{RETRY_COUNT}")
                
                response = requests.get(url, headers=headers, params=params)
                
                # Handle rate limiting
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', RETRY_DELAY))
                    logger.warning(f"Rate limited. Waiting for {retry_after} seconds...")
                    time.sleep(retry_after)
                    continue
                
                # Handle other errors
                if response.status_code != 200:
                    logger.warning(f"Error {response.status_code}: {response.text}")
                    if attempt < RETRY_COUNT - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    else:
                        response.raise_for_status()
                
                # Parse response
                data = response.json()
                
                # Add delay before next request
                time.sleep(REQUEST_DELAY)
                
                return data
                
            except Exception as e:
                logger.error(f"Request failed: {e}")
                if attempt < RETRY_COUNT - 1:
                    time.sleep(RETRY_DELAY * (2 ** attempt))  # Exponential backoff
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
        
        if not response or response.get('status') != 'success':
            error_msg = response.get('error', {}).get('message', 'Unknown error') if response else "No response"
            logger.error(f"API request failed: {error_msg}")
            return pd.DataFrame()
        
        candles = response.get('data', {}).get('candles', [])
        
        if not candles:
            logger.warning(f"No candles returned for {instrument_key}")
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
    
    def save_to_csv(self, df, exchange, symbol, interval, unit):
        """
        Save DataFrame to CSV file
        
        Args:
            df (pandas.DataFrame): DataFrame to save
            exchange (str): Exchange code
            symbol (str): Trading symbol
            interval (int): Candle interval
            unit (str): Candle unit
            
        Returns:
            str: Path to the saved file
        """
        if df.empty:
            logger.warning(f"No data to save for {exchange}:{symbol} {interval}{unit}")
            return None
        
        # Create timeframe string
        timeframe = f"{interval}{unit}"
        
        # Get date range
        start_date = df.index.min().strftime("%Y%m%d")
        end_date = df.index.max().strftime("%Y%m%d")
        
        # Create directory for the symbol
        symbol_dir = os.path.join(self.data_dir, exchange, symbol)
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Create filename
        filename = f"{exchange}_{symbol}_{timeframe}_{start_date}_to_{end_date}.csv"
        filepath = os.path.join(symbol_dir, filename)
        
        # Save to CSV
        df.to_csv(filepath)
        logger.info(f"Saved {len(df)} candles to {filepath}")
        
        return filepath
    
    def download_for_symbol(self, exchange, symbol, timeframes, from_date=None, to_date=None):
        """
        Download data for a specific symbol for multiple timeframes
        
        Args:
            exchange (str): Exchange code
            symbol (str): Trading symbol
            timeframes (list): List of (interval, unit) tuples
            from_date (str): Start date in YYYY-MM-DD format
            to_date (str): End date in YYYY-MM-DD format
            
        Returns:
            dict: Dictionary with timeframe keys and file paths as values
        """
        logger.info(f"Processing {exchange}:{symbol}")
        
        # Get instrument key
        instrument_key = self.symbol_mapper.get_instrument_key(exchange, symbol)
        
        if not instrument_key:
            logger.error(f"Could not find instrument key for {exchange}:{symbol}")
            return {}
        
        results = {}
        
        # Process each timeframe
        for interval, unit in timeframes:
            try:
                logger.info(f"Downloading {interval}{unit} data for {symbol}...")
                
                # Get data as DataFrame
                df = self.get_candles_as_dataframe(
                    instrument_key, interval, unit, from_date, to_date
                )
                
                # Save to CSV
                if not df.empty:
                    file_path = self.save_to_csv(df, exchange, symbol, interval, unit)
                    results[f"{interval}{unit}"] = file_path
                
            except Exception as e:
                logger.error(f"Error downloading {interval}{unit} data for {symbol}: {e}")
        
        return results
    
    def run(self):
        """
        Run the download process using configuration settings
        
        Returns:
            str: Path to the output directory
        """
        # Get configuration settings
        symbols = getattr(self.config, 'SYMBOLS', [])
        exchange = getattr(self.config, 'EXCHANGE', 'NSE')
        timeframes = getattr(self.config, 'TIMEFRAMES', [(5, "minute")])
        from_date = getattr(self.config, 'FROM_DATE', None)
        to_date = getattr(self.config, 'TO_DATE', None)
        use_parallel = getattr(self.config, 'USE_PARALLEL', False)
        max_workers = getattr(self.config, 'MAX_WORKERS', 5)
        
        if not symbols:
            logger.error("No symbols specified in configuration")
            return None
        
        logger.info(f"Starting batch download for {len(symbols)} symbols from {exchange}")
        logger.info(f"Timeframes: {timeframes}")
        logger.info(f"Date range: {from_date or 'auto'} to {to_date or 'today'}")
        
        results = {}
        
        if use_parallel and max_workers > 1:
            from concurrent.futures import ThreadPoolExecutor
            
            logger.info(f"Using parallel processing with {max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all download tasks
                futures = {
                    executor.submit(
                        self.download_for_symbol, 
                        exchange, 
                        symbol, 
                        timeframes, 
                        from_date, 
                        to_date
                    ): symbol for symbol in symbols
                }
                
                # Process results as they complete
                for future in futures:
                    symbol = futures[future]
                    try:
                        result = future.result()
                        results[symbol] = result
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {e}")
        else:
            logger.info("Using sequential processing")
            
            # Process symbols sequentially
            for symbol in symbols:
                try:
                    result = self.download_for_symbol(
                        exchange, 
                        symbol, 
                        timeframes, 
                        from_date, 
                        to_date
                    )
                    results[symbol] = result
                except Exception as e:
                    logger.error(f"Error processing {symbol}: {e}")
        
        # Log summary
        successful = sum(1 for symbol, files in results.items() if files)
        logger.info(f"Download complete. Successfully processed {successful}/{len(symbols)} symbols.")
        
        return self.data_dir
```

### 6. Create Batch Downloader

Create `batch_downloader.py`:

```python
"""
Module for coordinating batch downloads of historical data
"""
import os
import importlib
import logging
import json
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_download.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchDownloader:
    """
    Class for coordinating batch downloads of historical data
    """
    
    def __init__(self, token_file="token.json"):
        """
        Initialize the batch downloader
        
        Args:
            token_file (str): Path to the token file
        """
        self.token_file = token_file
        self.access_token = None
        self.load_token()
    
    def load_token(self):
        """Load access token from file"""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'r') as f:
                    token_data = json.load(f)
                    self.access_token = token_data.get('access_token')
                logger.info("Access token loaded successfully")
            except Exception as e:
                logger.error(f"Error loading access token: {e}")
    
    def run_batch(self, config_name):
        """
        Run a batch download using a specific configuration
        
        Args:
            config_name (str): Name of the config module
            
        Returns:
            str: Path to the output directory or None on failure
        """
        if not self.access_token:
            logger.error("No access token available. Authentication required.")
            return None
        
        try:
            # Import the config module
            logger.info(f"Loading configuration from {config_name}")
            config_module = importlib.import_module(f"configs.{config_name}")
            
            # Set the access token in the config module
            setattr(config_module, 'ACCESS_TOKEN', self.access_token)
            
            # Import required modules
            from symbol_mapper import SymbolMapper
            from download_historical_data import HistoricalDataDownloader
            
            # Create symbol mapper
            symbol_mapper = SymbolMapper()
            
            # Create and run downloader
            downloader = HistoricalDataDownloader(config_module, symbol_mapper)
            output_dir = downloader.run()
            
            return output_dir
            
        except ImportError:
            logger.error(f"Config module not found: configs.{config_name}")
            return None
        except Exception as e:
            logger.error(f"Error running batch download: {e}")
            return None
```

### 7. Create Run Script

Create `run_download.py`:

```python
"""
Script to run batch downloads of historical data
"""
import sys
import logging
from batch_downloader import BatchDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to run batch download
    """
    # Default to 'nifty50' if no config name is provided
    config_name = 'nifty50'
    
    # If a config name is provided as a command-line argument, use that instead
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        
    logger.info(f"Using config: {config_name}")
    
    # Run download process
    downloader = BatchDownloader()
    output_dir = downloader.run_batch(config_name)
    
    if output_dir:
        logger.info(f"Successfully downloaded data to {output_dir}")
    else:
        logger.error("Download failed. Check logs for details.")

if __name__ == "__main__":
    main()
```

### 8. Create a Custom Config

Create `configs/custom_symbols.py`:

```python
"""
Configuration for downloading a custom list of symbols
"""

# List of symbols to download
SYMBOLS = [
    "SBIN",      # State Bank of India
    "RELIANCE",  # Reliance Industries
    "TCS",       # Tata Consultancy Services
    "INFY",      # Infosys
    "HDFCBANK",  # HDFC Bank
]

# Exchange for these symbols
EXCHANGE = "NSE"

# Timeframes to download for each symbol
# Format: (interval, unit)
TIMEFRAMES = [
    (5, "minute"),    # 5-minute candles
    (15, "minute"),   # 15-minute candles
    (30, "minute"),   # 30-minute candles
    (1, "hour"),      # 1-hour candles
]

# Date range (last 30 days)
from datetime import datetime, timedelta
TO_DATE = datetime.now().strftime("%Y-%m-%d")  # Today
FROM_DATE = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")  # 30 days ago

# Whether to download in parallel
USE_PARALLEL = True
MAX_WORKERS = 3  # Maximum number of parallel downloads
```

### 9. Running the Code

1. Ensure you have set up authentication and have a valid `token.json` file.

2. Run the batch download:
   ```bash
   python run_download.py nifty50  # Download all Nifty 50 stocks
   ```

   Or for a custom list:
   ```bash
   python run_download.py custom_symbols  # Download the custom symbol list
   ```

3. Check the `data` directory for downloaded files, organized by exchange and symbol.

## Submission Guidelines

Create a ZIP file containing all the code files you've created, ensuring that you:
1. Have not included your actual API credentials in the files (use `.env` for local storage)
2. Have not included any downloaded data files (add `data/` to `.gitignore`)
3. Include a brief README.md explaining how to run your code and the functionality implemented

## Evaluation Criteria

1. Correct implementation of batch processing for multiple symbols and timeframes
2. Efficient handling of parallel vs. sequential downloads
3. Proper error handling, logging, and rate limit management
4. Code organization and documentation
5. Configuration flexibility

## Helpful Resources

- [Upstox Historical Candle Data V3 API Documentation](https://upstox.com/developer/api-documentation/v3/get-historical-candle-data)
- [ThreadPoolExecutor Documentation](https://docs.python.org/3/library/concurrent.futures.html#threadpoolexecutor)
- [Python logging Documentation](https://docs.python.org/3/library/logging.html)
