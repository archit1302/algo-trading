"""
Batch download of historical data for multiple symbols.

This script demonstrates how to download historical data for multiple symbols
in a batch process using the HistoricalDataDownloader.
"""

import os
import time
import logging
import importlib
from typing import List, Dict, Tuple

from symbol_mapper import SymbolMapper
from download_historical_data import HistoricalDataDownloader
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def download_multiple_symbols(symbols: List[str], config_module=None) -> Dict[str, str]:
    """
    Download historical data for multiple symbols.
    
    Args:
        symbols: List of symbols to download data for
        config_module: Config module to use for download settings
        
    Returns:
        Dictionary mapping symbols to output filepaths
    """
    if config_module is None:
        config_module = config
        
    # Create symbol mapper once for efficiency
    symbol_mapper = SymbolMapper()
    
    results = {}
    
    for symbol in symbols:
        try:
            logger.info(f"Processing symbol: {symbol}")
            
            # Create a new configuration for this symbol
            custom_config = importlib.import_module('config')
            custom_config.SYMBOL = symbol
            
            # Create downloader with the custom config
            downloader = HistoricalDataDownloader(
                config_module=custom_config,
                symbol_mapper=symbol_mapper
            )
            
            # Run the download process
            output_file = downloader.run()
            
            if output_file:
                results[symbol] = output_file
                logger.info(f"Successfully downloaded data for {symbol}")
            else:
                logger.error(f"Failed to download data for {symbol}")
                results[symbol] = ""
                
            # Add a small delay to avoid API rate limiting
            time.sleep(1)
            
        except Exception as e:
            logger.exception(f"Error processing {symbol}: {str(e)}")
            results[symbol] = ""
            
    return results

def download_from_list_file(file_path: str, config_module=None) -> Dict[str, str]:
    """
    Download historical data for symbols listed in a text file.
    
    Args:
        file_path: Path to a text file with one symbol per line
        config_module: Config module to use for download settings
        
    Returns:
        Dictionary mapping symbols to output filepaths
    """
    try:
        with open(file_path, 'r') as f:
            symbols = [line.strip() for line in f if line.strip()]
            
        return download_multiple_symbols(symbols, config_module)
    
    except Exception as e:
        logger.exception(f"Error reading symbol list file: {str(e)}")
        return {}

if __name__ == "__main__":
    # Example: Download data for multiple symbols
    symbols_to_download = ["SBIN", "RELIANCE", "TCS", "INFY"]
    
    results = download_from_list_file("symbols.txt")
    # print(f"Starting batch download for {len(symbols_to_download)} symbols")
    # results = download_multiple_symbols(symbols_to_download)
    
    # Print summary
    success_count = sum(1 for filepath in results.values() if filepath)
    print(f"Download summary: {success_count}/{len(results)} symbols successful")
    
    for symbol, filepath in results.items():
        status = "SUCCESS" if filepath else "FAILED"
        print(f"  {symbol}: {status}")
    
    # Example: To download from a list file, uncomment below:
    # results = download_from_list_file("symbols.txt")
