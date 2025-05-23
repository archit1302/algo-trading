"""
Simple script to download historical data using a specific config.
"""

import sys
import os
import importlib
import logging

from symbol_mapper import SymbolMapper
from download_historical_data import HistoricalDataDownloader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_with_config(config_name: str) -> str:
    """
    Download historical data using a specific config file.
    
    Args:
        config_name: Name of the config module to use (without .py extension)
        
    Returns:
        Path to the output file or empty string on failure
    """
    try:
        # Import the specified config module
        config_module = importlib.import_module(config_name)
        
        # Create symbol mapper and downloader
        symbol_mapper = SymbolMapper()
        downloader = HistoricalDataDownloader(
            config_module=config_module,
            symbol_mapper=symbol_mapper
        )
        
        # Run the download process
        output_file = downloader.run()
        return output_file
        
    except Exception as e:
        logger.exception(f"Error running downloader with config {config_name}: {str(e)}")
        return ""

if __name__ == "__main__":
    # Default to 'config' if no config name is provided
    config_name = 'config'
    
    # If a config name is provided as a command-line argument, use that instead
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        
    print(f"Using config: {config_name}")
    
    # Run download process
    output_file = download_with_config(config_name)
    
    if output_file:
        print(f"Successfully downloaded data to {output_file}")
    else:
        print("Download failed. Check logs for details.")



# Assignment 

# 1. Download historical data for 50 symbols using only config and run_download.py file automatedly
# 2. Batch download nifty 50 symbols historical data for 5 min, 15 min, 30 min timeframe 
# 3. Try to resample data from 5 minutes to 1 hour timeframe for all nifty 50 symbols
# 4. Create a scanner to identify which symbols are above 10 EMA on 15 min timframe using historical data files

