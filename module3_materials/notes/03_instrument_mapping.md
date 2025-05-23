# Instrument Mapping with Upstox API v3

This note covers instrument mapping concepts and how to work with Upstox instrument files to map exchange symbols to instrument keys.

## What is Instrument Mapping?

Instrument mapping is the process of converting human-readable symbols (like "SBIN", "RELIANCE") to the unique instrument keys required by the Upstox API. Every tradable instrument on Upstox has a unique instrument key that must be used when making API calls.

## Upstox Instrument Key Format

Upstox instrument keys follow a specific format:
```
NSE_EQ|INE062A01020|SBIN
```

Where:
- `NSE_EQ` = Exchange segment (NSE Equity)
- `INE062A01020` = ISIN (International Securities Identification Number)
- `SBIN` = Trading symbol

## Working with Instrument Files

Upstox provides instrument files that contain mappings between symbols and instrument keys.

### Downloading Instrument Files

```python
import requests
import json

def download_instrument_file():
    """
    Download the complete instrument file from Upstox
    """
    url = "https://api.upstox.com/v3/master"
    headers = {
        "Accept": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Save to file
        with open("instruments.json", "w") as f:
            json.dump(response.json(), f, indent=2)
            
        print("Instrument file downloaded successfully")
        return response.json()
        
    except Exception as e:
        print(f"Error downloading instrument file: {e}")
        return None
```

### Loading and Processing Instrument Data

```python
import pandas as pd

def load_instruments_dataframe():
    """
    Load instruments into a pandas DataFrame for easy searching
    """
    try:
        with open("instruments.json", "r") as f:
            data = json.load(f)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Create useful columns for searching
        df['search_symbol'] = df['tradingsymbol'].str.upper()
        df['exchange_symbol'] = df['exchange'] + ":" + df['tradingsymbol']
        
        return df
        
    except Exception as e:
        print(f"Error loading instruments: {e}")
        return pd.DataFrame()
```

## Symbol Search and Mapping

### Basic Symbol Search

```python
def find_instrument_by_symbol(df, symbol, exchange="NSE"):
    """
    Find instrument key by symbol and exchange
    
    Args:
        df (DataFrame): Instruments DataFrame
        symbol (str): Trading symbol (e.g., "SBIN")
        exchange (str): Exchange name (e.g., "NSE")
    
    Returns:
        str: Instrument key or None if not found
    """
    # Filter by exchange and symbol
    result = df[
        (df['exchange'] == exchange) & 
        (df['search_symbol'] == symbol.upper())
    ]
    
    if not result.empty:
        return result.iloc[0]['instrument_token']
    
    return None

# Usage example
instruments_df = load_instruments_dataframe()
sbin_key = find_instrument_by_symbol(instruments_df, "SBIN", "NSE")
print(f"SBIN instrument key: {sbin_key}")
```

### Advanced Search with Fuzzy Matching

```python
def search_instruments(df, query, exchange=None, limit=10):
    """
    Search instruments with fuzzy matching
    
    Args:
        df (DataFrame): Instruments DataFrame
        query (str): Search query
        exchange (str): Optional exchange filter
        limit (int): Maximum results to return
    
    Returns:
        DataFrame: Matching instruments
    """
    # Convert query to uppercase for case-insensitive search
    query = query.upper()
    
    # Start with exchange filter if provided
    if exchange:
        df_filtered = df[df['exchange'] == exchange]
    else:
        df_filtered = df
    
    # Search in multiple fields
    mask = (
        df_filtered['search_symbol'].str.contains(query, na=False) |
        df_filtered['name'].str.upper().str.contains(query, na=False)
    )
    
    results = df_filtered[mask].head(limit)
    
    return results[['exchange', 'tradingsymbol', 'name', 'instrument_token']]

# Usage example
search_results = search_instruments(instruments_df, "BANK", "NSE", 5)
print(search_results)
```

## Creating an Instrument Mapper Class

```python
class InstrumentMapper:
    """
    A comprehensive instrument mapping utility
    """
    
    def __init__(self):
        self.instruments_df = None
        self.symbol_to_key = {}
        self.key_to_symbol = {}
        
    def load_instruments(self, file_path="instruments.json"):
        """Load instruments from file"""
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            
            self.instruments_df = pd.DataFrame(data)
            self._build_lookup_tables()
            
            print(f"Loaded {len(self.instruments_df)} instruments")
            return True
            
        except Exception as e:
            print(f"Error loading instruments: {e}")
            return False
    
    def _build_lookup_tables(self):
        """Build lookup tables for fast access"""
        for _, row in self.instruments_df.iterrows():
            exchange_symbol = f"{row['exchange']}:{row['tradingsymbol']}"
            instrument_key = row['instrument_token']
            
            # Build bidirectional mapping
            self.symbol_to_key[exchange_symbol] = instrument_key
            self.key_to_symbol[instrument_key] = exchange_symbol
    
    def get_instrument_key(self, exchange, symbol):
        """Get instrument key for a symbol"""
        exchange_symbol = f"{exchange}:{symbol.upper()}"
        return self.symbol_to_key.get(exchange_symbol)
    
    def get_symbol_from_key(self, instrument_key):
        """Get symbol from instrument key"""
        return self.key_to_symbol.get(instrument_key)
    
    def search_symbols(self, query, exchange=None):
        """Search for symbols matching query"""
        if self.instruments_df is None:
            return []
        
        return search_instruments(self.instruments_df, query, exchange)
```

## Working with Different Exchanges

### Supported Exchanges

Common exchanges available through Upstox:

- **NSE** - National Stock Exchange (Equity)
- **BSE** - Bombay Stock Exchange (Equity)
- **NSE_FO** - NSE Futures & Options
- **BSE_FO** - BSE Futures & Options
- **MCX_FO** - Multi Commodity Exchange
- **CDS_FO** - Currency Derivatives

### Exchange-Specific Mapping

```python
def get_nse_equity_symbols():
    """Get all NSE equity symbols"""
    df = load_instruments_dataframe()
    nse_equity = df[df['exchange'] == 'NSE']
    return nse_equity[['tradingsymbol', 'name', 'instrument_token']]

def get_nifty50_instruments():
    """Get Nifty 50 constituent instrument keys"""
    nifty50_symbols = [
        "RELIANCE", "HDFCBANK", "INFY", "ICICIBANK", "TCS",
        "KOTAKBANK", "HINDUNILVR", "SBIN", "BHARTIARTL", "ITC",
        "ASIANPAINT", "LT", "AXISBANK", "DMART", "MARUTI",
        "SUNPHARMA", "TITAN", "ULTRACEMCO", "NESTLEIND", "BAJFINANCE",
        "HCLTECH", "WIPRO", "NTPC", "POWERGRID", "TECHM",
        "TATAMOTORS", "COALINDIA", "TATASTEEL", "BAJAJFINSV", "HDFCLIFE",
        "INDUSINDBK", "ADANIENT", "SBILIFE", "JSWSTEEL", "GRASIM",
        "BRITANNIA", "CIPLA", "DRREDDY", "DIVISLAB", "EICHERMOT",
        "HEROMOTOCO", "APOLLOHOSP", "ONGC", "HINDALCO", "BAJAJ-AUTO",
        "TATACONSUM", "UPL", "BPCL", "SHREECEM", "LTIM"
    ]
    
    mapper = InstrumentMapper()
    mapper.load_instruments()
    
    nifty50_keys = {}
    for symbol in nifty50_symbols:
        key = mapper.get_instrument_key("NSE", symbol)
        if key:
            nifty50_keys[symbol] = key
    
    return nifty50_keys
```

## Error Handling and Validation

### Symbol Validation

```python
def validate_symbol(mapper, exchange, symbol):
    """
    Validate if a symbol exists in the given exchange
    
    Returns:
        tuple: (is_valid, instrument_key, error_message)
    """
    try:
        instrument_key = mapper.get_instrument_key(exchange, symbol)
        
        if instrument_key:
            return True, instrument_key, None
        else:
            return False, None, f"Symbol {symbol} not found in {exchange}"
            
    except Exception as e:
        return False, None, f"Error validating symbol: {e}"

# Usage
mapper = InstrumentMapper()
mapper.load_instruments()

is_valid, key, error = validate_symbol(mapper, "NSE", "SBIN")
if is_valid:
    print(f"Valid symbol. Instrument key: {key}")
else:
    print(f"Invalid symbol: {error}")
```

### Batch Symbol Processing

```python
def process_symbol_list(mapper, symbol_list):
    """
    Process a list of exchange:symbol pairs
    
    Args:
        mapper (InstrumentMapper): Mapper instance
        symbol_list (list): List of "EXCHANGE:SYMBOL" strings
    
    Returns:
        dict: Results with valid and invalid symbols
    """
    results = {
        'valid': {},
        'invalid': []
    }
    
    for symbol_str in symbol_list:
        try:
            if ':' in symbol_str:
                exchange, symbol = symbol_str.split(':', 1)
            else:
                exchange, symbol = 'NSE', symbol_str
            
            instrument_key = mapper.get_instrument_key(exchange, symbol)
            
            if instrument_key:
                results['valid'][symbol_str] = instrument_key
            else:
                results['invalid'].append(symbol_str)
                
        except Exception as e:
            results['invalid'].append(f"{symbol_str} (Error: {e})")
    
    return results

# Usage
symbols = ["NSE:SBIN", "NSE:RELIANCE", "BSE:INVALID", "INFY"]
results = process_symbol_list(mapper, symbols)

print("Valid symbols:")
for symbol, key in results['valid'].items():
    print(f"  {symbol}: {key}")

print("Invalid symbols:")
for symbol in results['invalid']:
    print(f"  {symbol}")
```

## Best Practices

1. **Cache Instrument Data**: Download and cache instrument files locally to reduce API calls
2. **Regular Updates**: Update instrument files periodically as new instruments are added
3. **Error Handling**: Always validate symbols before making API calls
4. **Exchange Specification**: Always specify the exchange to avoid ambiguity
5. **Case Sensitivity**: Handle case-insensitive symbol matching
6. **Fallback Search**: Implement fuzzy search for user-friendly symbol lookup

## Key Takeaways

- Instrument mapping is essential for converting symbols to API-compatible keys
- Upstox provides comprehensive instrument files with all tradable instruments
- Use pandas DataFrames for efficient searching and filtering
- Implement proper error handling and validation
- Cache instrument data locally for better performance
- Build lookup tables for fast symbol-to-key conversion
