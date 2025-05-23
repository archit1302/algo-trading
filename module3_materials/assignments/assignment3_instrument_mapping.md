# Assignment 3: Upstox Instrument Mapping

In this assignment, you will implement instrument mapping functionality for the Upstox API v3. You'll create a module that helps translate between different symbol formats and maintain an up-to-date instruments database.

## Objectives

1. Download and parse Upstox instrument JSON files
2. Create a mapping utility class for finding instrument details
3. Implement search functionality for finding symbols
4. Handle different exchanges and instrument types

## Tasks

### 1. Set Up Project Structure

Create a project structure:
```
upstox-instruments/
├── .env
├── config.py
├── upstox_auth.py
├── instrument_mapper.py
├── instrument_downloader.py
└── main.py
```

### 2. Create Configuration Module

Create a `config.py` file:

```python
"""
Configuration settings for Upstox API and instrument mapping
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Credentials
API_KEY = os.getenv("UPSTOX_API_KEY")
API_SECRET = os.getenv("UPSTOX_API_SECRET")
REDIRECT_URI = os.getenv("UPSTOX_REDIRECT_URI")

# API Endpoints
AUTH_URL = "https://api.upstox.com/v2/login/authorization/dialog"
TOKEN_URL = "https://api.upstox.com/v2/login/authorization/token"
API_BASE_URL = "https://api.upstox.com/v3"

# Instrument Files
INSTRUMENT_URLS = {
    "NSE": "https://assets.upstox.com/market-quote/instruments/exchange/NSE.json.gz",
    "BSE": "https://assets.upstox.com/market-quote/instruments/exchange/BSE.json.gz",
    "MCX": "https://assets.upstox.com/market-quote/instruments/exchange/MCX.json.gz",
}

# Local File Paths
INSTRUMENTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "instruments")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
INSTRUMENT_FILE_FMT = "{exchange}_instruments.json"

# Ensure directories exist
os.makedirs(INSTRUMENTS_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
```

### 3. Implement Instrument Downloader

Create `instrument_downloader.py`:

```python
"""
Module for downloading and managing Upstox instrument files
"""
import os
import json
import gzip
import shutil
import requests
from datetime import datetime
from config import INSTRUMENT_URLS, INSTRUMENTS_DIR, INSTRUMENT_FILE_FMT

class InstrumentDownloader:
    """
    Class for downloading and managing Upstox instrument files
    """
    
    def __init__(self):
        """Initialize the instrument downloader"""
        self.instrument_urls = INSTRUMENT_URLS
        self.instruments_dir = INSTRUMENTS_DIR
    
    def download_instruments(self, exchange="NSE"):
        """
        Download instrument file for a specific exchange
        
        Args:
            exchange (str): Exchange code (NSE, BSE, MCX)
            
        Returns:
            bool: True if successful, False otherwise
        """
        if exchange not in self.instrument_urls:
            print(f"Error: Unknown exchange {exchange}")
            return False
        
        url = self.instrument_urls[exchange]
        gz_file_path = os.path.join(self.instruments_dir, f"{exchange}.json.gz")
        json_file_path = os.path.join(self.instruments_dir, INSTRUMENT_FILE_FMT.format(exchange=exchange))
        
        print(f"Downloading instrument file for {exchange}...")
        try:
            # Download .gz file
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(gz_file_path, 'wb') as f_out:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f_out.write(chunk)
            
            # Extract .gz file
            with gzip.open(gz_file_path, 'rb') as f_in:
                with open(json_file_path, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            # Remove .gz file
            os.remove(gz_file_path)
            
            print(f"Successfully downloaded and extracted {exchange} instruments")
            return True
        
        except Exception as e:
            print(f"Error downloading instrument file for {exchange}: {e}")
            return False
    
    def download_all_instruments(self):
        """
        Download instrument files for all supported exchanges
        
        Returns:
            dict: Dictionary with exchange codes as keys and success status as values
        """
        results = {}
        for exchange in self.instrument_urls.keys():
            results[exchange] = self.download_instruments(exchange)
        
        return results
    
    def get_instrument_file_path(self, exchange="NSE"):
        """
        Get the path to the local instrument file for an exchange
        
        Args:
            exchange (str): Exchange code
            
        Returns:
            str: Path to the instrument file
        """
        return os.path.join(self.instruments_dir, INSTRUMENT_FILE_FMT.format(exchange=exchange))
    
    def get_instrument_file_age(self, exchange="NSE"):
        """
        Get the age of the instrument file in days
        
        Args:
            exchange (str): Exchange code
            
        Returns:
            float: Age in days, or None if file doesn't exist
        """
        file_path = self.get_instrument_file_path(exchange)
        
        if not os.path.exists(file_path):
            return None
        
        file_time = os.path.getmtime(file_path)
        file_date = datetime.fromtimestamp(file_time)
        age = (datetime.now() - file_date).total_seconds() / (60 * 60 * 24)  # Convert to days
        
        return age
```

### 4. Implement Instrument Mapper

Create `instrument_mapper.py`:

```python
"""
Module for mapping between different instrument identifiers in Upstox
"""
import os
import json
from instrument_downloader import InstrumentDownloader

class InstrumentMapper:
    """
    Class for mapping between different instrument identifiers in Upstox
    """
    
    def __init__(self, auto_download=True):
        """
        Initialize instrument mapper
        
        Args:
            auto_download (bool): Auto-download instrument files if needed
        """
        self.downloader = InstrumentDownloader()
        self.instruments = {}
        
        # Indices for faster lookups
        self.symbol_to_key = {}  # trading_symbol -> instrument_key
        self.name_to_keys = {}   # name -> [instrument_keys]
        self.isin_to_keys = {}   # isin -> [instrument_keys]
        
        if auto_download:
            self.ensure_fresh_data()
    
    def ensure_fresh_data(self, max_age_days=1):
        """
        Ensure instrument data is up-to-date
        
        Args:
            max_age_days (float): Maximum age of instrument files in days
            
        Returns:
            bool: True if data is fresh or was refreshed, False otherwise
        """
        for exchange in self.downloader.instrument_urls.keys():
            age = self.downloader.get_instrument_file_age(exchange)
            
            # Download if file doesn't exist or is too old
            if age is None or age > max_age_days:
                success = self.downloader.download_instruments(exchange)
                if not success:
                    return False
        
        return self.load_all_instruments()
    
    def load_instruments(self, exchange="NSE"):
        """
        Load instruments for a specific exchange
        
        Args:
            exchange (str): Exchange code
            
        Returns:
            bool: True if successful, False otherwise
        """
        file_path = self.downloader.get_instrument_file_path(exchange)
        
        if not os.path.exists(file_path):
            print(f"Instrument file for {exchange} not found. Downloading...")
            success = self.downloader.download_instruments(exchange)
            if not success:
                return False
        
        try:
            with open(file_path, 'r') as f:
                instruments = json.load(f)
            
            self.instruments[exchange] = instruments
            
            # Build indices for this exchange
            for instrument in instruments:
                instrument_key = instrument.get('instrument_key')
                trading_symbol = instrument.get('trading_symbol')
                name = instrument.get('name')
                isin = instrument.get('isin')
                
                if instrument_key and trading_symbol:
                    self.symbol_to_key[f"{exchange}:{trading_symbol}"] = instrument_key
                
                if instrument_key and name:
                    if name not in self.name_to_keys:
                        self.name_to_keys[name] = []
                    self.name_to_keys[name].append(instrument_key)
                
                if instrument_key and isin:
                    if isin not in self.isin_to_keys:
                        self.isin_to_keys[isin] = []
                    self.isin_to_keys[isin].append(instrument_key)
            
            print(f"Loaded {len(instruments)} instruments for {exchange}")
            return True
        
        except Exception as e:
            print(f"Error loading {exchange} instruments: {e}")
            return False
    
    def load_all_instruments(self):
        """
        Load instruments for all supported exchanges
        
        Returns:
            bool: True if all were loaded successfully, False otherwise
        """
        success = True
        for exchange in self.downloader.instrument_urls.keys():
            if not self.load_instruments(exchange):
                success = False
        
        return success
    
    def get_instrument_by_key(self, instrument_key):
        """
        Get instrument details by instrument key
        
        Args:
            instrument_key (str): Instrument key (e.g., NSE_EQ|INE123456789)
            
        Returns:
            dict: Instrument details or None if not found
        """
        if not instrument_key:
            return None
            
        exchange_segment = instrument_key.split('|')[0].split('_')[0] if '|' in instrument_key else None
        
        if not exchange_segment or exchange_segment not in self.instruments:
            return None
        
        for instrument in self.instruments[exchange_segment]:
            if instrument.get('instrument_key') == instrument_key:
                return instrument
        
        return None
    
    def get_instrument_by_symbol(self, exchange, symbol):
        """
        Get instrument details by exchange and trading symbol
        
        Args:
            exchange (str): Exchange code (NSE, BSE, MCX)
            symbol (str): Trading symbol
            
        Returns:
            dict: Instrument details or None if not found
        """
        key = f"{exchange}:{symbol}"
        instrument_key = self.symbol_to_key.get(key)
        
        if instrument_key:
            return self.get_instrument_by_key(instrument_key)
        
        return None
    
    def search_instruments(self, query, exchange=None, instrument_type=None):
        """
        Search for instruments by name or symbol
        
        Args:
            query (str): Search query
            exchange (str, optional): Filter by exchange
            instrument_type (str, optional): Filter by instrument type
            
        Returns:
            list: List of matching instruments
        """
        query = query.upper()
        results = []
        
        exchanges_to_search = [exchange] if exchange else self.instruments.keys()
        
        for exch in exchanges_to_search:
            if exch not in self.instruments:
                continue
                
            for instrument in self.instruments[exch]:
                # Skip if instrument type doesn't match filter
                if instrument_type and instrument.get('instrument_type') != instrument_type:
                    continue
                
                # Match by trading symbol or name
                trading_symbol = instrument.get('trading_symbol', '').upper()
                name = instrument.get('name', '').upper()
                
                if query in trading_symbol or query in name:
                    results.append(instrument)
        
        return results

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
        return self.symbol_to_key.get(key)
```

### 5. Create Main Test Module

Create `main.py` to test the instrument mapping functionality:

```python
"""
Test script for Upstox instrument mapping functionality
"""
from instrument_mapper import InstrumentMapper

def main():
    print("Initializing Instrument Mapper...")
    mapper = InstrumentMapper()
    
    # Test searching for instruments
    def test_search(query, exchange=None):
        print(f"\nSearching for '{query}' {f'in {exchange}' if exchange else ''}...")
        results = mapper.search_instruments(query, exchange)
        
        if results:
            print(f"Found {len(results)} results:")
            for idx, instrument in enumerate(results[:5], 1):
                print(f"{idx}. {instrument.get('name')} ({instrument.get('trading_symbol')}) - {instrument.get('instrument_key')}")
            
            if len(results) > 5:
                print(f"...and {len(results) - 5} more results")
        else:
            print("No results found")
    
    # Test fetching instrument by symbol
    def test_get_by_symbol(exchange, symbol):
        print(f"\nFetching details for {exchange}:{symbol}...")
        instrument = mapper.get_instrument_by_symbol(exchange, symbol)
        
        if instrument:
            print(f"Found: {instrument.get('name')} ({instrument.get('trading_symbol')})")
            print(f"Instrument Key: {instrument.get('instrument_key')}")
            print(f"Exchange Token: {instrument.get('exchange_token')}")
            print(f"Lot Size: {instrument.get('lot_size')}")
            print(f"Tick Size: {instrument.get('tick_size')}")
        else:
            print(f"No instrument found for {exchange}:{symbol}")
    
    # Run tests
    test_search("RELIANCE")
    test_search("SBIN", "NSE")
    test_search("INFY", "BSE")
    
    test_get_by_symbol("NSE", "SBIN")
    test_get_by_symbol("NSE", "RELIANCE")
    test_get_by_symbol("NSE", "NONEXISTENT")

if __name__ == "__main__":
    main()
```

### 6. Running the Code

1. Ensure you have set up your `.env` file with your Upstox API credentials.

2. Run the main script to test the instrument mapping functionality:
   ```bash
   python main.py
   ```

3. Observe the output to see the instrument details and search results.

## Submission Guidelines

Create a ZIP file containing all the code files you've created, ensuring that you:
1. Have not included your actual API credentials in the files (use `.env` for local storage)
2. Have followed the project structure as described
3. Include a brief README.md explaining how to run your code and the functionality implemented

## Evaluation Criteria

1. Correct implementation of instrument downloader and mapper
2. Efficient search and lookup functionality
3. Proper error handling and data refresh logic
4. Code organization and documentation

## Helpful Resources

- [Upstox Instruments Documentation](https://upstox.com/developer/api-documentation/instruments)
- [JSON and GZip handling in Python](https://docs.python.org/3/library/gzip.html)
- [Python Requests Library](https://docs.python-requests.org/)
