# Assignment 7: Building a Market Scanner with Upstox Data

In this assignment, you will create a market scanner that analyzes historical data from multiple symbols to identify trading opportunities based on technical indicators. You'll implement functionality to scan for symbols above their 10 EMA on 15-minute timeframe data.

## Objectives

1. Create a framework for scanning market data across multiple symbols
2. Implement technical indicators (EMAs and other relevant indicators)
3. Design flexible scanner criteria based on indicator values
4. Create reports and visualizations for identified opportunities

## Tasks

### 1. Set Up Project Structure

Create a project structure:
```
upstox-scanner/
├── .env
├── config.py
├── indicators.py
├── scanner.py
├── utils.py
├── visualization.py
└── main.py
```

### 2. Create Configuration Module

Create a `config.py` file:

```python
"""
Configuration settings for the market scanner
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Data directories
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Create directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Timeframe for scanning
SCAN_TIMEFRAME = "15minute"

# File pattern for locating data files
FILE_PATTERN = f"*_{SCAN_TIMEFRAME}_*.csv"

# Technical indicator parameters
EMA_SHORT_PERIOD = 10  # 10 EMA period
EMA_MEDIUM_PERIOD = 20  # 20 EMA period
EMA_LONG_PERIOD = 50   # 50 EMA period
RSI_PERIOD = 14        # RSI period
VOLUME_MA_PERIOD = 20  # Volume moving average period

# Scanner thresholds
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
VOLUME_THRESHOLD = 1.5  # Current volume > 1.5 * average volume
```

### 3. Create Indicators Module

Create `indicators.py`:

```python
"""
Module for calculating technical indicators for market scanning
"""
import pandas as pd
import numpy as np
import talib

def add_ema(df, period=10, column='close', result_column=None):
    """
    Add EMA to DataFrame
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLC data
        period (int): EMA period
        column (str): Column to calculate EMA on
        result_column (str): Column name for result, defaults to f"ema_{period}"
        
    Returns:
        pandas.DataFrame: DataFrame with added EMA column
    """
    if result_column is None:
        result_column = f"ema_{period}"
    
    if len(df) >= period:
        df[result_column] = talib.EMA(df[column].values, timeperiod=period)
    else:
        df[result_column] = np.nan
    
    return df

def add_sma(df, period=10, column='close', result_column=None):
    """
    Add SMA to DataFrame
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLC data
        period (int): SMA period
        column (str): Column to calculate SMA on
        result_column (str): Column name for result, defaults to f"sma_{period}"
        
    Returns:
        pandas.DataFrame: DataFrame with added SMA column
    """
    if result_column is None:
        result_column = f"sma_{period}"
    
    if len(df) >= period:
        df[result_column] = talib.SMA(df[column].values, timeperiod=period)
    else:
        df[result_column] = np.nan
    
    return df

def add_rsi(df, period=14, column='close', result_column=None):
    """
    Add RSI to DataFrame
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLC data
        period (int): RSI period
        column (str): Column to calculate RSI on
        result_column (str): Column name for result, defaults to f"rsi_{period}"
        
    Returns:
        pandas.DataFrame: DataFrame with added RSI column
    """
    if result_column is None:
        result_column = f"rsi_{period}"
    
    if len(df) >= period:
        df[result_column] = talib.RSI(df[column].values, timeperiod=period)
    else:
        df[result_column] = np.nan
    
    return df

def add_macd(df, fast_period=12, slow_period=26, signal_period=9, column='close'):
    """
    Add MACD to DataFrame
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLC data
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal EMA period
        column (str): Column to calculate MACD on
        
    Returns:
        pandas.DataFrame: DataFrame with added MACD columns
    """
    if len(df) >= slow_period:
        macd, signal, hist = talib.MACD(
            df[column].values,
            fastperiod=fast_period,
            slowperiod=slow_period,
            signalperiod=signal_period
        )
        
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
    else:
        df['macd'] = np.nan
        df['macd_signal'] = np.nan
        df['macd_hist'] = np.nan
    
    return df

def add_bollinger_bands(df, period=20, stddev=2, column='close'):
    """
    Add Bollinger Bands to DataFrame
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLC data
        period (int): Bollinger Bands period
        stddev (int): Number of standard deviations
        column (str): Column to calculate Bollinger Bands on
        
    Returns:
        pandas.DataFrame: DataFrame with added Bollinger Bands columns
    """
    if len(df) >= period:
        upper, middle, lower = talib.BBANDS(
            df[column].values,
            timeperiod=period,
            nbdevup=stddev,
            nbdevdn=stddev,
            matype=0  # Simple Moving Average
        )
        
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
    else:
        df['bb_upper'] = np.nan
        df['bb_middle'] = np.nan
        df['bb_lower'] = np.nan
    
    return df

def add_atr(df, period=14):
    """
    Add Average True Range to DataFrame
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLC data
        period (int): ATR period
        
    Returns:
        pandas.DataFrame: DataFrame with added ATR column
    """
    if len(df) >= period:
        df['atr'] = talib.ATR(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            timeperiod=period
        )
    else:
        df['atr'] = np.nan
    
    return df

def add_all_indicators(df, config=None):
    """
    Add all indicators to DataFrame using configuration
    
    Args:
        df (pandas.DataFrame): DataFrame with OHLC data
        config (module): Configuration module with indicator parameters
        
    Returns:
        pandas.DataFrame: DataFrame with all indicators
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Use default values if config not provided
    ema_short = getattr(config, 'EMA_SHORT_PERIOD', 10) if config else 10
    ema_medium = getattr(config, 'EMA_MEDIUM_PERIOD', 20) if config else 20
    ema_long = getattr(config, 'EMA_LONG_PERIOD', 50) if config else 50
    rsi_period = getattr(config, 'RSI_PERIOD', 14) if config else 14
    volume_ma_period = getattr(config, 'VOLUME_MA_PERIOD', 20) if config else 20
    
    # Add EMAs
    df = add_ema(df, period=ema_short)
    df = add_ema(df, period=ema_medium)
    df = add_ema(df, period=ema_long)
    
    # Add RSI
    df = add_rsi(df, period=rsi_period)
    
    # Add volume SMA
    df = add_sma(df, period=volume_ma_period, column='volume', result_column='volume_sma')
    
    # Add MACD
    df = add_macd(df)
    
    # Add Bollinger Bands
    df = add_bollinger_bands(df)
    
    # Add ATR
    df = add_atr(df)
    
    return df
```

### 4. Create Scanner Module

Create `scanner.py`:

```python
"""
Module for scanning market data for trading opportunities
"""
import os
import glob
import pandas as pd
import numpy as np
import logging
from concurrent.futures import ThreadPoolExecutor
from indicators import add_all_indicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MarketScanner:
    """
    Class for scanning market data for trading opportunities
    """
    
    def __init__(self, data_dir, results_dir, config):
        """
        Initialize the market scanner
        
        Args:
            data_dir (str): Directory containing data files
            results_dir (str): Directory for saving results
            config (module): Configuration module
        """
        self.data_dir = data_dir
        self.results_dir = results_dir
        self.config = config
        self.results = {}
        
        # Create results directory if it doesn't exist
        os.makedirs(results_dir, exist_ok=True)
    
    def find_data_files(self, timeframe=None, pattern=None):
        """
        Find data files in the data directory
        
        Args:
            timeframe (str): Filter files by timeframe
            pattern (str): File pattern to match
            
        Returns:
            list: List of file paths
        """
        if pattern is None:
            pattern = self.config.FILE_PATTERN
            
        if timeframe:
            pattern = pattern.replace('*', f'*_{timeframe}_*')
            
        search_pattern = os.path.join(self.data_dir, '**', pattern)
        files = glob.glob(search_pattern, recursive=True)
        
        logger.info(f"Found {len(files)} data files matching pattern {pattern}")
        return files
    
    def load_and_prepare_data(self, filepath):
        """
        Load and prepare data from a file
        
        Args:
            filepath (str): Path to the data file
            
        Returns:
            tuple: (symbol, DataFrame with indicators)
        """
        try:
            # Extract symbol from filename
            filename = os.path.basename(filepath)
            parts = filename.split('_')
            
            if len(parts) < 2:
                logger.warning(f"Could not extract symbol from filename: {filename}")
                return None, None
            
            exchange = parts[0]
            symbol = parts[1]
            
            # Read data
            df = pd.read_csv(filepath)
            
            # Set timestamp as index if available
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            elif 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            # Convert columns to correct types
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col])
            
            # Add technical indicators
            df = add_all_indicators(df, self.config)
            
            return f"{exchange}:{symbol}", df
            
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            return None, None
    
    def scan_for_above_ema(self, df, ema_period=10, lookback=1):
        """
        Scan for price above EMA
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            ema_period (int): EMA period to check against
            lookback (int): Number of bars to look back
            
        Returns:
            bool: True if price is above EMA, False otherwise
        """
        if df is None or df.empty:
            return False
            
        ema_col = f"ema_{ema_period}"
        
        if ema_col not in df.columns:
            logger.warning(f"EMA column {ema_col} not found in DataFrame")
            return False
        
        # Get the last N rows
        last_rows = df.iloc[-lookback:].copy()
        
        # Check if close is above EMA in all rows
        return (last_rows['close'] > last_rows[ema_col]).all()
    
    def scan_for_below_ema(self, df, ema_period=10, lookback=1):
        """
        Scan for price below EMA
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            ema_period (int): EMA period to check against
            lookback (int): Number of bars to look back
            
        Returns:
            bool: True if price is below EMA, False otherwise
        """
        if df is None or df.empty:
            return False
            
        ema_col = f"ema_{ema_period}"
        
        if ema_col not in df.columns:
            logger.warning(f"EMA column {ema_col} not found in DataFrame")
            return False
        
        # Get the last N rows
        last_rows = df.iloc[-lookback:].copy()
        
        # Check if close is below EMA in all rows
        return (last_rows['close'] < last_rows[ema_col]).all()
    
    def scan_for_ema_crossover(self, df, fast_period=10, slow_period=20, lookback=5):
        """
        Scan for EMA crossover (fast crosses above slow)
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            fast_period (int): Fast EMA period
            slow_period (int): Slow EMA period
            lookback (int): Number of bars to look back
            
        Returns:
            bool: True if fast EMA crossed above slow EMA, False otherwise
        """
        if df is None or df.empty or len(df) < lookback:
            return False
            
        fast_col = f"ema_{fast_period}"
        slow_col = f"ema_{slow_period}"
        
        if fast_col not in df.columns or slow_col not in df.columns:
            logger.warning(f"EMA columns {fast_col} or {slow_col} not found in DataFrame")
            return False
        
        # Get the last N+1 rows to check for crossover
        last_rows = df.iloc[-(lookback+1):].copy()
        
        # Check for crossover: fast was below slow, now fast is above slow
        was_below = last_rows[fast_col].iloc[0] < last_rows[slow_col].iloc[0]
        now_above = last_rows[fast_col].iloc[-1] > last_rows[slow_col].iloc[-1]
        
        return was_below and now_above
    
    def scan_for_high_volume(self, df, ma_period=20, threshold=1.5, lookback=1):
        """
        Scan for high volume (above average)
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            ma_period (int): Volume moving average period
            threshold (float): Volume must be this multiple of the average
            lookback (int): Number of bars to look back
            
        Returns:
            bool: True if volume is high, False otherwise
        """
        if df is None or df.empty:
            return False
        
        vol_ma_col = 'volume_sma'
        
        if vol_ma_col not in df.columns:
            logger.warning(f"Volume MA column {vol_ma_col} not found in DataFrame")
            return False
        
        # Get the last N rows
        last_rows = df.iloc[-lookback:].copy()
        
        # Check if volume is above threshold * volume_ma in any of the last N bars
        return (last_rows['volume'] > threshold * last_rows[vol_ma_col]).any()
    
    def scan_for_oversold_rsi(self, df, period=14, threshold=30, lookback=3):
        """
        Scan for oversold RSI
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            period (int): RSI period
            threshold (float): RSI threshold (oversold below this value)
            lookback (int): Number of bars to look back
            
        Returns:
            bool: True if RSI is oversold, False otherwise
        """
        if df is None or df.empty:
            return False
        
        rsi_col = f"rsi_{period}"
        
        if rsi_col not in df.columns:
            logger.warning(f"RSI column {rsi_col} not found in DataFrame")
            return False
        
        # Get the last N rows
        last_rows = df.iloc[-lookback:].copy()
        
        # Check if RSI is below threshold in any of the last N bars
        return (last_rows[rsi_col] < threshold).any()
    
    def scan_for_overbought_rsi(self, df, period=14, threshold=70, lookback=3):
        """
        Scan for overbought RSI
        
        Args:
            df (pandas.DataFrame): DataFrame with indicators
            period (int): RSI period
            threshold (float): RSI threshold (overbought above this value)
            lookback (int): Number of bars to look back
            
        Returns:
            bool: True if RSI is overbought, False otherwise
        """
        if df is None or df.empty:
            return False
        
        rsi_col = f"rsi_{period}"
        
        if rsi_col not in df.columns:
            logger.warning(f"RSI column {rsi_col} not found in DataFrame")
            return False
        
        # Get the last N rows
        last_rows = df.iloc[-lookback:].copy()
        
        # Check if RSI is above threshold in any of the last N bars
        return (last_rows[rsi_col] > threshold).any()
    
    def run_scan_on_file(self, filepath, scan_functions=None):
        """
        Run all scan functions on a single file
        
        Args:
            filepath (str): Path to the data file
            scan_functions (dict): Dictionary of scan functions to run
            
        Returns:
            dict: Results for this file
        """
        # Default scan functions if not provided
        if scan_functions is None:
            scan_functions = {
                'above_ema_10': lambda df: self.scan_for_above_ema(df, 10),
                'above_ema_20': lambda df: self.scan_for_above_ema(df, 20),
                'below_ema_10': lambda df: self.scan_for_below_ema(df, 10),
                'below_ema_20': lambda df: self.scan_for_below_ema(df, 20),
                'ema_crossover': lambda df: self.scan_for_ema_crossover(df, 10, 20),
                'high_volume': lambda df: self.scan_for_high_volume(df),
                'oversold_rsi': lambda df: self.scan_for_oversold_rsi(df),
                'overbought_rsi': lambda df: self.scan_for_overbought_rsi(df)
            }
        
        # Load and prepare data
        symbol, df = self.load_and_prepare_data(filepath)
        
        if symbol is None or df is None or df.empty:
            logger.warning(f"Could not process {filepath}")
            return {}
        
        # Run all scan functions
        results = {
            'symbol': symbol,
            'filepath': filepath,
            'last_date': str(df.index[-1]) if not df.empty else None,
            'scans': {}
        }
        
        for name, func in scan_functions.items():
            try:
                results['scans'][name] = func(df)
            except Exception as e:
                logger.error(f"Error running scan {name} on {symbol}: {e}")
                results['scans'][name] = None
        
        return {symbol: results}
    
    def run_scan(self, timeframe=None, parallel=True, max_workers=4):
        """
        Run the market scanner on all data files
        
        Args:
            timeframe (str): Timeframe to filter files by
            parallel (bool): Whether to process files in parallel
            max_workers (int): Maximum number of workers for parallel processing
            
        Returns:
            dict: Scan results
        """
        # Find data files
        files = self.find_data_files(timeframe)
        
        if not files:
            logger.warning(f"No data files found for timeframe {timeframe}")
            return {}
        
        logger.info(f"Running market scanner on {len(files)} files")
        
        # Initialize results
        self.results = {}
        
        if parallel and len(files) > 1:
            logger.info(f"Processing files in parallel with {max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(self.run_scan_on_file, f): f 
                    for f in files
                }
                
                # Process results as they complete
                for future in futures:
                    try:
                        result = future.result()
                        self.results.update(result)
                    except Exception as e:
                        logger.error(f"Error processing {futures[future]}: {e}")
        else:
            logger.info("Processing files sequentially")
            
            for f in files:
                try:
                    result = self.run_scan_on_file(f)
                    self.results.update(result)
                except Exception as e:
                    logger.error(f"Error processing {f}: {e}")
        
        return self.results
    
    def filter_results(self, condition):
        """
        Filter results based on a condition
        
        Args:
            condition (callable): Function that takes a result dict and returns bool
            
        Returns:
            dict: Filtered results
        """
        return {k: v for k, v in self.results.items() if condition(v)}
    
    def get_symbols_above_ema(self, ema_period=10):
        """
        Get symbols with price above EMA
        
        Args:
            ema_period (int): EMA period
            
        Returns:
            dict: Filtered results
        """
        scan_name = f"above_ema_{ema_period}"
        
        return self.filter_results(
            lambda r: r.get('scans', {}).get(scan_name, False)
        )
    
    def get_symbols_with_ema_crossover(self):
        """
        Get symbols with EMA crossover
        
        Returns:
            dict: Filtered results
        """
        return self.filter_results(
            lambda r: r.get('scans', {}).get('ema_crossover', False)
        )
    
    def save_results_to_csv(self, results=None, filename=None):
        """
        Save scan results to a CSV file
        
        Args:
            results (dict): Results to save, uses self.results if None
            filename (str): Output filename
            
        Returns:
            str: Path to the saved file
        """
        if results is None:
            results = self.results
            
        if not results:
            logger.warning("No results to save")
            return None
        
        # Create a DataFrame from the results
        rows = []
        
        for symbol, result in results.items():
            scans = result.get('scans', {})
            row = {
                'symbol': result.get('symbol', symbol),
                'last_date': result.get('last_date')
            }
            
            # Add scan results
            for scan_name, scan_result in scans.items():
                row[scan_name] = scan_result
                
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Create output filename if not provided
        if filename is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filename = f"scan_results_{timestamp}.csv"
            
        filepath = os.path.join(self.results_dir, filename)
        
        # Save to CSV
        df.to_csv(filepath, index=False)
        logger.info(f"Saved results to {filepath}")
        
        return filepath
```

### 5. Create Utils Module

Create `utils.py`:

```python
"""
Utility functions for market scanner
"""
import os
import pandas as pd
import glob

def find_latest_data_file(data_dir, symbol, timeframe=None):
    """
    Find the latest data file for a symbol
    
    Args:
        data_dir (str): Data directory
        symbol (str): Symbol to find
        timeframe (str): Timeframe to filter by
        
    Returns:
        str: Path to the latest file, or None if not found
    """
    # Create search pattern
    pattern = f"*{symbol}*"
    if timeframe:
        pattern = f"*{symbol}*{timeframe}*"
    
    # Search for files
    files = glob.glob(os.path.join(data_dir, '**', pattern), recursive=True)
    
    if not files:
        return None
    
    # Sort by modification time (newest first)
    return sorted(files, key=os.path.getmtime, reverse=True)[0]

def extract_symbol_from_filename(filename):
    """
    Extract symbol from filename
    
    Args:
        filename (str): Filename to parse
        
    Returns:
        str: Symbol or None if not found
    """
    parts = os.path.basename(filename).split('_')
    
    if len(parts) < 2:
        return None
    
    return parts[1]

def load_results_from_csv(filepath):
    """
    Load scanner results from CSV file
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pandas.DataFrame: DataFrame with results
    """
    return pd.read_csv(filepath)

def bool_columns_to_actual_bool(df):
    """
    Convert string 'True'/'False' values to actual bool
    
    Args:
        df (pandas.DataFrame): DataFrame with results
        
    Returns:
        pandas.DataFrame: DataFrame with bool columns
    """
    # Find columns that contain only True, False or NaN values
    bool_columns = []
    
    for col in df.columns:
        unique_values = set(df[col].dropna().astype(str).unique())
        if unique_values.issubset({'True', 'False'}):
            bool_columns.append(col)
    
    # Convert those columns to bool
    for col in bool_columns:
        df[col] = df[col].astype(str).map({'True': True, 'False': False})
    
    return df
```

### 6. Create Visualization Module

Create `visualization.py`:

```python
"""
Module for visualizing market scanner results
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_results_from_csv, bool_columns_to_actual_bool

def create_results_heatmap(results_df, output_path=None):
    """
    Create a heatmap of scanner results
    
    Args:
        results_df (pandas.DataFrame): DataFrame with scanner results
        output_path (str): Path to save the plot
        
    Returns:
        str: Path to the saved plot
    """
    # Convert string bool to actual bool
    df = bool_columns_to_actual_bool(results_df)
    
    # Find boolean columns for the heatmap
    bool_columns = [col for col in df.columns if df[col].dtype == bool]
    
    if not bool_columns:
        raise ValueError("No boolean columns found in results DataFrame")
    
    # Prepare data for heatmap
    heatmap_data = df[['symbol'] + bool_columns].set_index('symbol')
    
    # Create figure
    plt.figure(figsize=(12, max(8, len(df) * 0.3)))
    
    # Create heatmap
    sns.heatmap(
        heatmap_data,
        cmap=sns.color_palette("YlGnBu", 2),
        linewidths=0.5,
        linecolor='gray',
        cbar=False,
    )
    
    plt.title("Market Scanner Results", fontsize=16)
    plt.tight_layout()
    
    # Save plot if output_path is provided
    if output_path:
        plt.savefig(output_path)
        return output_path
    else:
        plt.show()
        return None

def plot_symbol_with_indicators(data_df, symbol=None, output_path=None):
    """
    Plot a symbol with technical indicators
    
    Args:
        data_df (pandas.DataFrame): DataFrame with OHLC data and indicators
        symbol (str): Symbol name for the title
        output_path (str): Path to save the plot
        
    Returns:
        str: Path to the saved plot
    """
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [3, 1, 1]}, sharex=True)
    
    # Plot price and EMAs
    ax1.plot(data_df.index, data_df['close'], label='Close', linewidth=2)
    
    # Plot EMAs if available
    if 'ema_10' in data_df.columns:
        ax1.plot(data_df.index, data_df['ema_10'], label='EMA 10', linewidth=1.5)
    if 'ema_20' in data_df.columns:
        ax1.plot(data_df.index, data_df['ema_20'], label='EMA 20', linewidth=1.5)
    if 'ema_50' in data_df.columns:
        ax1.plot(data_df.index, data_df['ema_50'], label='EMA 50', linewidth=1.5)
    
    # Add title and legend
    title = f"Price Chart with Indicators{f' - {symbol}' if symbol else ''}"
    ax1.set_title(title, fontsize=16)
    ax1.set_ylabel('Price')
    ax1.legend(loc='upper left')
    ax1.grid(True)
    
    # Plot volume
    ax2.bar(data_df.index, data_df['volume'], label='Volume', alpha=0.7)
    if 'volume_sma' in data_df.columns:
        ax2.plot(data_df.index, data_df['volume_sma'], label='Volume MA', color='red', linewidth=1.5)
    ax2.set_ylabel('Volume')
    ax2.legend(loc='upper left')
    ax2.grid(True)
    
    # Plot RSI if available
    if 'rsi_14' in data_df.columns:
        ax3.plot(data_df.index, data_df['rsi_14'], label='RSI(14)', color='purple', linewidth=1.5)
        ax3.axhline(70, linestyle='--', color='red', alpha=0.5)
        ax3.axhline(30, linestyle='--', color='green', alpha=0.5)
        ax3.set_ylabel('RSI')
        ax3.set_ylim(0, 100)
        ax3.legend(loc='upper left')
        ax3.grid(True)
    
    plt.tight_layout()
    
    # Save plot if output_path is provided
    if output_path:
        plt.savefig(output_path)
        return output_path
    else:
        plt.show()
        return None
```

### 7. Create Main Script

Create `main.py`:

```python
"""
Main script for running the market scanner
"""
import os
import sys
import logging
import argparse
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Import scanner modules
import config
from scanner import MarketScanner
from utils import find_latest_data_file, load_results_from_csv
from visualization import create_results_heatmap, plot_symbol_with_indicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scanner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Market scanner for finding trading opportunities')
    
    parser.add_argument(
        '-t', '--timeframe',
        default=config.SCAN_TIMEFRAME,
        help=f'Timeframe to scan (default: {config.SCAN_TIMEFRAME})'
    )
    parser.add_argument(
        '-p', '--parallel',
        action='store_true',
        help='Process files in parallel'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=4,
        help='Maximum number of worker threads for parallel processing'
    )
    parser.add_argument(
        '-o', '--output',
        help='Output filename for results (default: auto-generated)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Create visualization plot of results'
    )
    parser.add_argument(
        '--plot-symbols',
        nargs='+',
        help='Plot specific symbols with indicators'
    )
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command-line arguments
    args = parse_args()
    
    # Create output filename if not provided
    if args.output:
        output_filename = args.output
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"scan_results_{timestamp}.csv"
    
    logger.info(f"Starting market scanner for timeframe {args.timeframe}")
    
    # Create scanner
    scanner = MarketScanner(config.DATA_DIR, config.RESULTS_DIR, config)
    
    # Run scan
    results = scanner.run_scan(args.timeframe, args.parallel, args.workers)
    
    if not results:
        logger.warning("No results found")
        return 1
    
    # Save results to CSV
    output_file = scanner.save_results_to_csv(filename=output_filename)
    
    if not output_file:
        logger.error("Failed to save results")
        return 1
    
    # Get symbols above EMA 10
    above_ema10 = scanner.get_symbols_above_ema(10)
    
    logger.info(f"Found {len(above_ema10)} symbols above 10 EMA")
    for symbol in above_ema10.keys():
        logger.info(f"  {symbol}")
    
    # Create visualization if requested
    if args.plot:
        try:
            # Load results from CSV
            results_df = load_results_from_csv(output_file)
            
            # Create heatmap
            plot_file = os.path.join(
                config.RESULTS_DIR, 
                f"heatmap_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            create_results_heatmap(results_df, plot_file)
            logger.info(f"Created heatmap: {plot_file}")
            
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
    
    # Plot specific symbols if requested
    if args.plot_symbols:
        for symbol in args.plot_symbols:
            try:
                # Find latest data file for this symbol
                data_file = find_latest_data_file(config.DATA_DIR, symbol, args.timeframe)
                
                if not data_file:
                    logger.warning(f"No data file found for {symbol}")
                    continue
                
                # Load data
                data_df = pd.read_csv(data_file)
                if 'timestamp' in data_df.columns:
                    data_df['timestamp'] = pd.to_datetime(data_df['timestamp'])
                    data_df.set_index('timestamp', inplace=True)
                elif 'date' in data_df.columns:
                    data_df['date'] = pd.to_datetime(data_df['date'])
                    data_df.set_index('date', inplace=True)
                
                # Create plot
                plot_file = os.path.join(
                    config.RESULTS_DIR, 
                    f"{symbol}_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                plot_symbol_with_indicators(data_df, symbol, plot_file)
                logger.info(f"Created plot for {symbol}: {plot_file}")
                
            except Exception as e:
                logger.error(f"Error plotting {symbol}: {e}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### 8. Running the Code

1. Make sure you have historical data files from previous assignments in the `data` directory.

2. Install the required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn talib
   ```

3. Run the scanner:
   ```bash
   python main.py --timeframe 15minute --parallel --plot
   ```

4. To plot specific symbols:
   ```bash
   python main.py --plot-symbols SBIN RELIANCE TCS
   ```

5. Check the `results` directory for the output CSV file and visualizations.

### 9. Additional Tasks

1. Implement more complex scanning criteria (e.g., multiple indicator combinations)
2. Create a web interface for viewing scanner results
3. Set up automated scanning at regular intervals
4. Add support for real-time alerts based on scan results

## Submission Guidelines

Create a ZIP file containing all the code files you've created, ensuring that you:
1. Have not included data files (add `data/` to `.gitignore`)
2. Have not included result files (add `results/` to `.gitignore`)
3. Have followed the project structure as described
4. Include a brief README.md explaining how to run your code and the functionality implemented

## Evaluation Criteria

1. Correct implementation of technical indicators
2. Efficient scanning across multiple symbols
3. Quality of visualizations
4. Code organization and documentation
5. User-friendly command-line interface

## Helpful Resources

- [TA-Lib Documentation](https://mrjbq7.github.io/ta-lib/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Matplotlib Documentation](https://matplotlib.org/stable/index.html)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [Python argparse Tutorial](https://docs.python.org/3/howto/argparse.html)
