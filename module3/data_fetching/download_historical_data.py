"""
Historical Data Downloader using Upstox API

This script downloads historical market data based on configuration settings
and saves it to files for later analysis.
"""

import os
import requests
import pandas as pd
import datetime
import json
import time
from typing import List, Dict, Any, Tuple, Optional
import logging
from urllib.parse import quote

# Import local modules
from symbol_mapper import SymbolMapper
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class HistoricalDataDownloader:
    """
    Downloads historical market data from Upstox API based on provided configuration.
    """
    
    def __init__(self, config_module=None, symbol_mapper=None):
        """
        Initialize the downloader with configuration and symbol mapper.
        
        Args:
            config_module: Module containing configuration settings
            symbol_mapper: SymbolMapper instance for symbol to instrument key mapping
        """
        self.config = config_module or config
        self.symbol_mapper = symbol_mapper or SymbolMapper()
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config.OUTPUT_DIRECTORY, exist_ok=True)
        
    @staticmethod
    def list_available_timeframes(output_dir=None):
        """
        List all available timeframes in the output directory.
        
        Args:
            output_dir: Directory to scan for timeframes, defaults to config.OUTPUT_DIRECTORY
            
        Returns:
            List of available timeframes
        """
        if output_dir is None:
            output_dir = config.OUTPUT_DIRECTORY
            
        if not os.path.exists(output_dir):
            logger.warning(f"Output directory {output_dir} does not exist")
            return []
            
        # List all subdirectories in the output directory (these are timeframes)
        timeframes = [d for d in os.listdir(output_dir) 
                     if os.path.isdir(os.path.join(output_dir, d))]
        
        return sorted(timeframes)
        
    @staticmethod
    def list_symbols_for_timeframe(timeframe, output_dir=None):
        """
        List all symbols available for a specific timeframe.
        
        Args:
            timeframe: Timeframe to list symbols for
            output_dir: Base directory, defaults to config.OUTPUT_DIRECTORY
            
        Returns:
            List of symbols for the specified timeframe
        """
        if output_dir is None:
            output_dir = config.OUTPUT_DIRECTORY
            
        timeframe_dir = os.path.join(output_dir, timeframe)
        if not os.path.exists(timeframe_dir):
            logger.warning(f"Timeframe directory {timeframe_dir} does not exist")
            return []
            
        # List all subdirectories in the timeframe directory (these are symbols)
        symbols = [d for d in os.listdir(timeframe_dir) 
                  if os.path.isdir(os.path.join(timeframe_dir, d))]
        
        return sorted(symbols)
        
    def validate_config(self) -> Tuple[bool, str]:
        """
        Validate the configuration settings.
        
        Returns:
            Tuple[bool, str]: (is_valid, error_message)
        """
        # Check symbol
        if not self.config.SYMBOL:
            return False, "Symbol must be specified"
        
        # Check unit and interval
        valid_units = ["minutes", "hours", "days", "weeks", "months"]
        if self.config.UNIT not in valid_units:
            return False, f"Invalid unit: {self.config.UNIT}. Must be one of {valid_units}"
        
        # Validate interval based on unit
        interval = int(self.config.INTERVAL)
        if self.config.UNIT == "minutes" and (interval < 1 or interval > 300):
            return False, f"Invalid interval for minutes: {interval}. Must be between 1 and 300."
        elif self.config.UNIT == "hours" and (interval < 1 or interval > 5):
            return False, f"Invalid interval for hours: {interval}. Must be between 1 and 5."
        elif self.config.UNIT in ["days", "weeks", "months"] and interval != 1:
            return False, f"Invalid interval for {self.config.UNIT}: {interval}. Must be 1."
        
        # Validate dates
        try:
            start_date = datetime.datetime.strptime(self.config.START_DATE, "%Y-%m-%d")
            end_date = datetime.datetime.strptime(self.config.END_DATE, "%Y-%m-%d")
            
            if start_date > end_date:
                return False, f"Start date ({self.config.START_DATE}) must be before end date ({self.config.END_DATE})"
        except ValueError as e:
            return False, f"Invalid date format: {str(e)}"
            
        return True, ""
    
    def get_instrument_key(self, symbol: str) -> Optional[str]:
        """
        Get the instrument key for the given symbol.
        
        Args:
            symbol: Trading symbol to look up
            
        Returns:
            Instrument key or None if not found
        """
        instrument_key = self.symbol_mapper.get_instrument_key(symbol)
        if not instrument_key:
            matching_symbols = self.symbol_mapper.search_symbols(symbol)
            if matching_symbols:
                # If exact match not found but similar symbols exist, use the first one
                logger.warning(f"Exact match for {symbol} not found. Using {matching_symbols[0]} instead.")
                instrument_key = self.symbol_mapper.get_instrument_key(matching_symbols[0])
            else:
                logger.error(f"No matching symbol found for {symbol}")
                
        return instrument_key
    
    def download_historical_data(self) -> Optional[pd.DataFrame]:
        """
        Download historical data from Upstox API.
        
        Returns:
            DataFrame with historical data or None if download failed
        """
        # Validate configuration
        is_valid, error_message = self.validate_config()
        if not is_valid:
            logger.error(f"Invalid configuration: {error_message}")
            return None
            
        # Get instrument key
        instrument_key = self.get_instrument_key(self.config.SYMBOL)
        if not instrument_key:
            logger.error(f"Could not find instrument key for symbol {self.config.SYMBOL}")
            return None
            
        logger.info(f"Using instrument key {instrument_key} for symbol {self.config.SYMBOL}")
        
        # Prepare API URL
        # URL encode the instrument key
        encoded_instrument_key = quote(instrument_key)
        
        url = (
            f"{self.config.API_BASE_URL}/{encoded_instrument_key}"
            f"/{self.config.UNIT}/{self.config.INTERVAL}"
            f"/{self.config.END_DATE}/{self.config.START_DATE}"
        )
        
        headers = {
            'Accept': 'application/json'
        }
        
        if hasattr(self.config, 'API_KEY') and self.config.API_KEY:
            headers['Authorization'] = f"Bearer {self.config.API_KEY}"
            
        logger.info(f"Making API request to {url}")
        
        try:
            response = requests.get(url, headers=headers)
            
            # Check if request was successful
            if response.status_code != 200:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
            # Parse response
            response_data = response.json()
            
            if response_data.get('status') != 'success':
                logger.error(f"API returned error status: {response_data}")
                return None
                
            candles = response_data.get('data', {}).get('candles', [])
            
            if not candles:
                logger.warning("API returned no candle data")
                return None
                
            logger.info(f"Received {len(candles)} candles from API")
            
            # Convert to DataFrame
            df = pd.DataFrame(
                candles,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi']
            )
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error downloading historical data: {str(e)}")
            return None
    
    def save_data(self, df: pd.DataFrame) -> str:
        """
        Save the downloaded data to a file.
        
        Args:
            df: DataFrame containing the historical data
            
        Returns:
            Path to the saved file
        """
        if df is None or df.empty:
            logger.error("No data to save")
            return ""
        
        # Sort the dataframe by timestamp
        df = df.sort_values(by='timestamp')
            
        # Create timeframe-specific directory structure
        symbol = self.config.SYMBOL
        unit = self.config.UNIT
        interval = self.config.INTERVAL
        timeframe = f"{interval}{unit}"
        
        # Create directory structure: OUTPUT_DIRECTORY/timeframe/symbol/
        timeframe_dir = os.path.join(self.config.OUTPUT_DIRECTORY, timeframe)
        symbol_dir = os.path.join(timeframe_dir, symbol)
        
        # Ensure directories exist
        os.makedirs(symbol_dir, exist_ok=True)
        
        # Create filename with date range
        start_date = self.config.START_DATE.replace('-', '')
        end_date = self.config.END_DATE.replace('-', '')
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        filename = f"{symbol}_{timeframe}_{start_date}_to_{end_date}_{timestamp}"
        
        if self.config.OUTPUT_FORMAT.lower() == 'csv':
            filepath = os.path.join(symbol_dir, f"{filename}.csv")
            df.to_csv(filepath, index=False)
        else:
            filepath = os.path.join(symbol_dir, f"{filename}.json")
            df.to_json(filepath, orient='records', date_format='iso')
            
        logger.info(f"Data saved to {filepath}")
        return filepath
        
    def run(self) -> str:
        """
        Run the full download process.
        
        Returns:
            Path to the saved file or empty string if failed
        """
        logger.info(f"Starting download for {self.config.SYMBOL}")
        
        # Download data
        data = self.download_historical_data()
        
        # Save data
        filepath = self.save_data(data)
        
        return filepath


def list_available_data(output_dir=None):
    """
    List all available market data organized by timeframe and symbol.
    
    Args:
        output_dir: Base directory to scan, defaults to config.OUTPUT_DIRECTORY
    """
    if output_dir is None:
        output_dir = config.OUTPUT_DIRECTORY
        
    if not os.path.exists(output_dir):
        print(f"Output directory {output_dir} does not exist")
        return
        
    timeframes = HistoricalDataDownloader.list_available_timeframes(output_dir)
    
    if not timeframes:
        print(f"No timeframes found in {output_dir}")
        return
        
    print("\nAvailable Market Data:")
    print("=====================")
    
    for tf in timeframes:
        print(f"\nTimeframe: {tf}")
        print("-" * (len(tf) + 11))
        
        symbols = HistoricalDataDownloader.list_symbols_for_timeframe(tf, output_dir)
        
        if not symbols:
            print(f"  No symbols found for timeframe {tf}")
            continue
            
        for symbol in symbols:
            symbol_dir = os.path.join(output_dir, tf, symbol)
            files = [f for f in os.listdir(symbol_dir) if not f.startswith('.')]
            
            print(f"  Symbol: {symbol} ({len(files)} files)")
            
            # List a few sample files
            for i, file in enumerate(sorted(files)[:3]):
                print(f"    - {file}")
                
            if len(files) > 3:
                print(f"    ... and {len(files)-3} more files")
    
    print("\nTo access data for a specific timeframe and symbol:")
    print("import os")
    print("timeframe = '<desired_timeframe>'  # e.g., '15minutes'")
    print("symbol = '<desired_symbol>'        # e.g., 'SBIN'")
    print(f"data_dir = os.path.join('{output_dir}', timeframe, symbol)")
    print("print(os.listdir(data_dir))  # List all available files")


if __name__ == "__main__":
    import sys
    
    # Check for command-line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--list":
        # List available data
        list_available_data()
    else:
        try:
            # Create downloader
            downloader = HistoricalDataDownloader()
            
            # Run download process
            output_file = downloader.run()
            
            if output_file:
                print(f"Successfully downloaded data to {output_file}")
            else:
                print("Download failed. Check logs for details.")
                
        except Exception as e:
            logger.exception(f"Unhandled exception: {str(e)}")
            print(f"Error: {str(e)}")
            
        print("\nTo list all available downloaded data, run:")
        print("python3 download_historical_data.py --list")
