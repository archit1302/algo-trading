# Assignment 6: Data Resampling with Upstox API Data

In this assignment, you will implement data resampling functionality to convert historical data from one timeframe to another. This is crucial for technical analysis across multiple timeframes and for strategy backtesting.

## Objectives

1. Create a module for resampling time series data from lower to higher timeframes
2. Implement multiple resampling techniques
3. Process multiple files and symbols in batch
4. Handle edge cases such as missing data and timezone issues

## Tasks

### 1. Set Up Project Structure

Create a project structure:
```
upstox-resampling/
├── .env
├── config.py
├── data_resampler.py
├── utils.py
├── batch_resampler.py
└── main.py
```

### 2. Create Configuration Module

Create a `config.py` file:

```python
"""
Configuration settings for data resampling
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Data directories
INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "input")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "output")

# Ensure directories exist
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default resampling parameters
DEFAULT_INPUT_TIMEFRAME = "5minute"  # Format: {interval}{unit}
DEFAULT_OUTPUT_TIMEFRAME = "1hour"   # Format: {interval}{unit}

# File pattern for globbing
FILE_PATTERN = "NSE_*.csv"  # Example: NSE_SBIN_5minute_20230101_to_20230131.csv

# Minimum number of bars required for resampling
MIN_BARS_REQUIRED = 5  # Require at least 5 bars to form a resampled bar

# Timezone for timestamps (default is IST for Indian markets)
TIMEZONE = "Asia/Kolkata"
```

### 3. Create Utilities Module

Create a `utils.py` file with helper functions:

```python
"""
Utility functions for data resampling
"""
import os
import re
import pandas as pd
from datetime import datetime
import pytz
from config import TIMEZONE

def parse_timeframe(timeframe):
    """
    Parse a timeframe string into interval and unit
    
    Args:
        timeframe (str): Timeframe string like "5minute" or "1hour"
        
    Returns:
        tuple: (interval, unit)
    """
    # Match numbers followed by a string
    match = re.match(r"(\d+)(\D+)", timeframe)
    
    if not match:
        raise ValueError(f"Invalid timeframe format: {timeframe}")
    
    interval = int(match.group(1))
    unit = match.group(2)
    
    # Normalize units
    if unit in ["m", "min", "minute", "minutes"]:
        unit = "minute"
    elif unit in ["h", "hr", "hour", "hours"]:
        unit = "hour"
    elif unit in ["d", "day", "days"]:
        unit = "day"
    elif unit in ["w", "week", "weeks"]:
        unit = "week"
    elif unit in ["mo", "month", "months"]:
        unit = "month"
    else:
        raise ValueError(f"Unknown time unit: {unit}")
    
    return interval, unit

def timeframe_to_pandas_freq(interval, unit):
    """
    Convert interval and unit to pandas frequency string
    
    Args:
        interval (int): Number of units
        unit (str): Unit of time
        
    Returns:
        str: Pandas frequency string
    """
    if unit == "minute":
        return f"{interval}min"
    elif unit == "hour":
        return f"{interval}H"
    elif unit == "day":
        return f"{interval}D"
    elif unit == "week":
        return f"{interval}W"
    elif unit == "month":
        return f"{interval}M"
    else:
        raise ValueError(f"Unsupported unit for pandas frequency: {unit}")

def ensure_datetime_tz(df, tz=TIMEZONE):
    """
    Ensure the DataFrame index is a timezone-aware datetime
    
    Args:
        df (pandas.DataFrame): DataFrame with datetime index
        tz (str): Timezone string
        
    Returns:
        pandas.DataFrame: DataFrame with timezone-aware index
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # If index is not datetime, convert it
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Add timezone if not already present
    if df.index.tz is None:
        df.index = df.index.tz_localize(tz)
    # Convert timezone if different
    elif str(df.index.tz) != tz:
        df.index = df.index.tz_convert(tz)
    
    return df

def extract_metadata_from_filename(filename):
    """
    Extract metadata from filename
    
    Args:
        filename (str): Filename like "NSE_SBIN_5minute_20230101_to_20230131.csv"
        
    Returns:
        dict: Metadata
    """
    # Extract parts from filename
    parts = os.path.basename(filename).split('_')
    
    if len(parts) < 5:
        raise ValueError(f"Filename format not recognized: {filename}")
    
    # Extract exchange, symbol, and timeframe
    exchange = parts[0]
    symbol = parts[1]
    timeframe = parts[2]
    
    # Extract date range
    start_date = parts[3]
    end_date = parts[-1].split('.')[0]  # Remove extension
    
    return {
        "exchange": exchange,
        "symbol": symbol,
        "timeframe": timeframe,
        "start_date": start_date,
        "end_date": end_date
    }

def generate_output_filename(input_filename, output_timeframe):
    """
    Generate output filename based on input file and new timeframe
    
    Args:
        input_filename (str): Input filename
        output_timeframe (str): Output timeframe (e.g., "1hour")
        
    Returns:
        str: Output filename
    """
    metadata = extract_metadata_from_filename(input_filename)
    
    # Create new filename with output timeframe
    base = os.path.basename(input_filename)
    parts = base.split('_')
    parts[2] = output_timeframe  # Replace timeframe
    
    return '_'.join(parts)
```

### 4. Create Data Resampler Class

Create `data_resampler.py`:

```python
"""
Module for resampling time series data from one timeframe to another
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from utils import parse_timeframe, timeframe_to_pandas_freq, ensure_datetime_tz, generate_output_filename
from config import MIN_BARS_REQUIRED, OUTPUT_DIR, TIMEZONE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataResampler:
    """
    Class for resampling OHLCV data from one timeframe to another
    """
    
    def __init__(self, output_dir=OUTPUT_DIR):
        """
        Initialize the data resampler
        
        Args:
            output_dir (str): Directory for saving resampled data
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def read_data(self, filepath):
        """
        Read data from CSV file
        
        Args:
            filepath (str): Path to CSV file
            
        Returns:
            pandas.DataFrame: DataFrame with OHLCV data
        """
        try:
            df = pd.read_csv(filepath)
            
            # Check required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            # Set the timestamp column as index
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
            elif 'date' in df.columns:
                df.set_index('date', inplace=True)
            
            # Convert index to datetime
            df.index = pd.to_datetime(df.index)
            
            # Apply timezone
            df = ensure_datetime_tz(df, TIMEZONE)
            
            return df
            
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {e}")
            raise
    
    def get_output_path(self, input_filepath, output_timeframe):
        """
        Get the path for the output file
        
        Args:
            input_filepath (str): Path to input file
            output_timeframe (str): Output timeframe
            
        Returns:
            str: Output file path
        """
        filename = generate_output_filename(input_filepath, output_timeframe)
        return os.path.join(self.output_dir, filename)
    
    def resample_ohlcv(self, df, output_freq, min_bars=MIN_BARS_REQUIRED):
        """
        Resample OHLCV data to a new frequency
        
        Args:
            df (pandas.DataFrame): DataFrame with OHLCV data
            output_freq (str): Pandas frequency string for output
            min_bars (int): Minimum number of bars required for resampling
            
        Returns:
            pandas.DataFrame: Resampled DataFrame
        """
        if df.empty:
            logger.warning("Empty DataFrame, nothing to resample")
            return df
        
        # Ensure DataFrame has timezone-aware index
        df = ensure_datetime_tz(df)
        
        # Check if we have enough bars
        if len(df) < min_bars:
            logger.warning(f"Not enough bars for resampling: {len(df)} < {min_bars}")
            return pd.DataFrame()
        
        # Create resampler object
        resampler = df.resample(output_freq)
        
        # Resample OHLCV data
        resampled = pd.DataFrame({
            'open': resampler['open'].first(),
            'high': resampler['high'].max(),
            'low': resampler['low'].min(),
            'close': resampler['close'].last(),
            'volume': resampler['volume'].sum()
        })
        
        # Add open interest if available
        if 'oi' in df.columns:
            resampled['oi'] = resampler['oi'].last()
        
        # Filter out rows with NaN values
        resampled = resampled.dropna()
        
        # Check if we have any data after resampling
        if resampled.empty:
            logger.warning("No data left after resampling")
            return resampled
        
        # Convert index back to string format for better compatibility
        resampled.index = resampled.index.strftime('%Y-%m-%d %H:%M:%S%z')
        
        return resampled
    
    def resample_file(self, input_filepath, output_timeframe):
        """
        Resample data from an input file to a new timeframe
        
        Args:
            input_filepath (str): Path to input CSV file
            output_timeframe (str): Output timeframe (e.g., "1hour")
            
        Returns:
            str: Path to output file, or None if resampling failed
        """
        try:
            logger.info(f"Resampling {input_filepath} to {output_timeframe}")
            
            # Read input data
            df = self.read_data(input_filepath)
            
            if df.empty:
                logger.warning(f"No data in input file {input_filepath}")
                return None
            
            # Parse output timeframe
            interval, unit = parse_timeframe(output_timeframe)
            output_freq = timeframe_to_pandas_freq(interval, unit)
            
            # Resample data
            resampled = self.resample_ohlcv(df, output_freq)
            
            if resampled.empty:
                logger.warning(f"Resampling produced no data for {input_filepath}")
                return None
            
            # Get output path
            output_path = self.get_output_path(input_filepath, output_timeframe)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Save resampled data
            resampled.to_csv(output_path)
            logger.info(f"Saved resampled data to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error resampling {input_filepath}: {e}")
            return None
```

### 5. Create Batch Resampler

Create `batch_resampler.py`:

```python
"""
Module for batch resampling of multiple files
"""
import os
import glob
import logging
from concurrent.futures import ThreadPoolExecutor
from data_resampler import DataResampler
from config import INPUT_DIR, FILE_PATTERN, DEFAULT_INPUT_TIMEFRAME, DEFAULT_OUTPUT_TIMEFRAME

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BatchResampler:
    """
    Class for batch resampling of multiple files
    """
    
    def __init__(self, input_dir=INPUT_DIR, output_dir=None):
        """
        Initialize batch resampler
        
        Args:
            input_dir (str): Directory containing input files
            output_dir (str): Directory for saving resampled files
        """
        self.input_dir = input_dir
        self.resampler = DataResampler(output_dir)
    
    def find_files(self, pattern=FILE_PATTERN):
        """
        Find files matching the pattern in input directory
        
        Args:
            pattern (str): Glob pattern for file search
            
        Returns:
            list: List of matching file paths
        """
        search_pattern = os.path.join(self.input_dir, pattern)
        files = glob.glob(search_pattern)
        logger.info(f"Found {len(files)} files matching pattern {pattern}")
        return files
    
    def filter_files_by_timeframe(self, files, timeframe):
        """
        Filter files by timeframe
        
        Args:
            files (list): List of file paths
            timeframe (str): Timeframe to filter by
            
        Returns:
            list: Filtered list of file paths
        """
        filtered = [f for f in files if f"_{timeframe}_" in os.path.basename(f)]
        logger.info(f"Filtered to {len(filtered)} files with timeframe {timeframe}")
        return filtered
    
    def resample_all(self, files, output_timeframe=DEFAULT_OUTPUT_TIMEFRAME, parallel=True, max_workers=4):
        """
        Resample all files to the specified timeframe
        
        Args:
            files (list): List of file paths
            output_timeframe (str): Output timeframe
            parallel (bool): Whether to process in parallel
            max_workers (int): Maximum number of workers for parallel processing
            
        Returns:
            dict: Dictionary with input file paths as keys and output file paths as values
        """
        if not files:
            logger.warning("No files to resample")
            return {}
        
        results = {}
        
        if parallel and len(files) > 1:
            logger.info(f"Processing {len(files)} files in parallel with {max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                futures = {
                    executor.submit(self.resampler.resample_file, f, output_timeframe): f 
                    for f in files
                }
                
                # Process results as they complete
                for future in futures:
                    input_file = futures[future]
                    try:
                        output_file = future.result()
                        results[input_file] = output_file
                    except Exception as e:
                        logger.error(f"Error processing {input_file}: {e}")
        else:
            logger.info(f"Processing {len(files)} files sequentially")
            
            for input_file in files:
                try:
                    output_file = self.resampler.resample_file(input_file, output_timeframe)
                    results[input_file] = output_file
                except Exception as e:
                    logger.error(f"Error processing {input_file}: {e}")
        
        # Count successful conversions
        successful = sum(1 for output in results.values() if output is not None)
        logger.info(f"Completed resampling: {successful} successful, {len(files) - successful} failed")
        
        return results
    
    def run(self, input_timeframe=DEFAULT_INPUT_TIMEFRAME, 
            output_timeframe=DEFAULT_OUTPUT_TIMEFRAME, 
            parallel=True, max_workers=4):
        """
        Run the batch resampling process
        
        Args:
            input_timeframe (str): Input timeframe to filter by
            output_timeframe (str): Output timeframe
            parallel (bool): Whether to process in parallel
            max_workers (int): Maximum number of workers for parallel processing
            
        Returns:
            dict: Dictionary with input file paths as keys and output file paths as values
        """
        # Find all files in input directory
        all_files = self.find_files()
        
        # Filter files by input timeframe if specified
        if input_timeframe:
            files_to_process = self.filter_files_by_timeframe(all_files, input_timeframe)
        else:
            files_to_process = all_files
        
        # Resample all files
        return self.resample_all(files_to_process, output_timeframe, parallel, max_workers)
```

### 6. Create Main Script

Create `main.py`:

```python
"""
Main script for data resampling
"""
import os
import sys
import logging
import argparse
from batch_resampler import BatchResampler
from config import INPUT_DIR, DEFAULT_INPUT_TIMEFRAME, DEFAULT_OUTPUT_TIMEFRAME

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("resampling.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description='Resample OHLCV data from one timeframe to another')
    
    parser.add_argument(
        '-i', '--input-dir', 
        default=INPUT_DIR,
        help='Directory containing input CSV files'
    )
    parser.add_argument(
        '-f', '--input-timeframe', 
        default=DEFAULT_INPUT_TIMEFRAME,
        help='Input timeframe to filter files by (e.g., "5minute")'
    )
    parser.add_argument(
        '-t', '--output-timeframe', 
        default=DEFAULT_OUTPUT_TIMEFRAME,
        help='Output timeframe to resample to (e.g., "1hour")'
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
    
    return parser.parse_args()

def main():
    """Main function"""
    # Parse command-line arguments
    args = parse_args()
    
    logger.info(f"Starting data resampling from {args.input_timeframe} to {args.output_timeframe}")
    logger.info(f"Input directory: {args.input_dir}")
    logger.info(f"Parallel processing: {args.parallel} with {args.workers} workers")
    
    # Initialize batch resampler
    resampler = BatchResampler(args.input_dir)
    
    # Run batch resampling
    results = resampler.run(
        args.input_timeframe,
        args.output_timeframe,
        args.parallel,
        args.workers
    )
    
    # Log summary
    total = len(results)
    successful = sum(1 for output in results.values() if output is not None)
    
    if total > 0:
        logger.info(f"Resampling completed: {successful} of {total} files were successful ({successful / total:.1%})")
    else:
        logger.warning("No files were processed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
```

### 7. Running the Code

1. Place your input data files (e.g., those downloaded with the batch downloader from the previous assignment) in the `data/input` directory.

2. Run the resampler:
   ```bash
   python main.py --input-timeframe 5minute --output-timeframe 1hour --parallel
   ```

3. Check the `data/output` directory for the resampled files.

### 8. Additional Tasks

1. Implement validation for the resampled data (e.g., check that volume is properly summed)
2. Add support for custom aggregation functions
3. Create a visualization to compare the original and resampled data

## Submission Guidelines

Create a ZIP file containing all the code files you've created, ensuring that you:
1. Have not included any data files (add `data/` to `.gitignore`)
2. Have followed the project structure as described
3. Include a brief README.md explaining how to run your code and the functionality implemented

## Evaluation Criteria

1. Correct implementation of data resampling
2. Handling of edge cases (e.g., timezone issues, missing data)
3. Efficiency of batch processing
4. Code organization and documentation
5. User-friendly command-line interface

## Helpful Resources

- [Pandas Resampling Documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.resample.html)
- [Python datetime Documentation](https://docs.python.org/3/library/datetime.html)
- [Python argparse Tutorial](https://docs.python.org/3/howto/argparse.html)
