"""
Symbol Mapper Module

This module provides functionality to map trading symbols to instrument keys
using the NSE.csv.gz file containing exchange listing data.
"""

import pandas as pd
import gzip
import os
from typing import Dict, Optional, List
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SymbolMapper:
    """
    A class that provides mapping between trading symbols and instrument keys.
    """
    
    def __init__(self, data_file_path: str = None):
        """
        Initialize the SymbolMapper with the path to the exchange data file.
        
        Args:
            data_file_path: Path to the NSE.csv.gz file. If None, uses default path.
        """
        if data_file_path is None:
            # Default path is relative to this file's directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_file_path = os.path.join(current_dir, "NSE.csv.gz")
        else:
            self.data_file_path = data_file_path
            
        self.symbol_to_instrument_key = {}
        self.instrument_key_to_symbol = {}
        self.df = None
        
        # Load the data on initialization
        self._load_data()
    
    def _load_data(self):
        """
        Load data from the NSE.csv.gz file and create the mapping dictionaries.
        """
        try:
            logger.info(f"Loading data from {self.data_file_path}")
            
            # Read the gzipped CSV file
            with gzip.open(self.data_file_path, 'rt') as f:
                self.df = pd.read_csv(f)
            
            logger.info(f"Loaded {len(self.df)} instruments")
            
            # Create mappings
            self.symbol_to_instrument_key = dict(zip(self.df['tradingsymbol'], self.df['instrument_key']))
            self.instrument_key_to_symbol = dict(zip(self.df['instrument_key'], self.df['tradingsymbol']))
            
            logger.info("Symbol mappings created successfully")
        
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_instrument_key(self, symbol: str) -> Optional[str]:
        """
        Get the instrument key for a given trading symbol.
        
        Args:
            symbol: The trading symbol to look up
            
        Returns:
            The instrument key or None if not found
        """
        return self.symbol_to_instrument_key.get(symbol)
    
    def get_symbol(self, instrument_key: str) -> Optional[str]:
        """
        Get the trading symbol for a given instrument key.
        
        Args:
            instrument_key: The instrument key to look up
            
        Returns:
            The trading symbol or None if not found
        """
        return self.instrument_key_to_symbol.get(instrument_key)
    
    def search_symbols(self, pattern: str) -> List[str]:
        """
        Search for symbols containing the given pattern.
        
        Args:
            pattern: The pattern to search for in trading symbols
            
        Returns:
            A list of matching trading symbols
        """
        if self.df is None:
            logger.error("Data not loaded")
            return []
        
        matches = self.df[self.df['tradingsymbol'].str.contains(pattern, case=False)]
        return matches['tradingsymbol'].tolist()
    
    def get_instrument_details(self, symbol: str = None, instrument_key: str = None) -> Dict:
        """
        Get detailed information about an instrument by either symbol or instrument key.
        
        Args:
            symbol: The trading symbol (optional)
            instrument_key: The instrument key (optional)
            
        Returns:
            A dictionary with instrument details or empty dict if not found
        """
        if self.df is None:
            logger.error("Data not loaded")
            return {}
        
        if symbol:
            matches = self.df[self.df['tradingsymbol'] == symbol]
        elif instrument_key:
            matches = self.df[self.df['instrument_key'] == instrument_key]
        else:
            logger.error("Either symbol or instrument_key must be provided")
            return {}
        
        if len(matches) == 0:
            return {}
        
        # Return the first match as a dictionary
        return matches.iloc[0].to_dict()
    
    def reload_data(self):
        """
        Reload the data from the file, refreshing all mappings.
        """
        self._load_data()


# Example usage
if __name__ == "__main__":
    # Create a symbol mapper
    mapper = SymbolMapper()
    
    # Example: Search for NIFTY symbols
    nifty_symbols = mapper.search_symbols("NIFTY")
    print(f"Found {len(nifty_symbols)} NIFTY symbols")
    print(nifty_symbols[:5])  # Print first 5
    
    # Example: Get instrument key for a symbol
    if nifty_symbols:
        test_symbol = nifty_symbols[0]
        instrument_key = mapper.get_instrument_key(test_symbol)
        print(f"Symbol: {test_symbol}, Instrument Key: {instrument_key}")
        
        # Get details
        details = mapper.get_instrument_details(symbol=test_symbol)
        print(f"Details: {details}")
