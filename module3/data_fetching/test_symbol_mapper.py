"""
Test script for the symbol mapper module to demonstrate its functionality.
"""

from symbol_mapper import SymbolMapper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_symbol_mapper():
    """Run tests on the SymbolMapper class."""
    try:
        print("Initializing the SymbolMapper...")
        mapper = SymbolMapper()
        
        # Test case 1: Search for common stock symbols
        print("\nTest 1: Search for common stock symbols")
        test_symbols = ["NIFTY", "RELIANCE", "SBIN"]
        for symbol_pattern in test_symbols:
            matching_symbols = mapper.search_symbols(symbol_pattern)
            print(f"Found {len(matching_symbols)} matches for '{symbol_pattern}':")
            for i, symbol in enumerate(matching_symbols[:5], 1):  # Show only first 5
                print(f"  {i}. {symbol}")
                # Get the instrument key for this symbol
                instrument_key = mapper.get_instrument_key(symbol)
                print(f"     Instrument Key: {instrument_key}")
            
            if len(matching_symbols) > 5:
                print(f"     ... and {len(matching_symbols) - 5} more")
        
        # Test case 2: Get instrument details
        print("\nTest 2: Get instrument details")
        if matching_symbols:
            test_symbol = matching_symbols[0]
            details = mapper.get_instrument_details(symbol=test_symbol)
            print(f"Details for {test_symbol}:")
            for key, value in details.items():
                print(f"  {key}: {value}")
        
        # Test case 3: Bidirectional mapping
        print("\nTest 3: Bidirectional mapping")
        if matching_symbols:
            test_symbol = matching_symbols[0]
            instrument_key = mapper.get_instrument_key(test_symbol)
            mapped_symbol = mapper.get_symbol(instrument_key)
            print(f"Symbol: {test_symbol} → Instrument Key: {instrument_key} → Symbol: {mapped_symbol}")
            print(f"Mapping integrity check: {'PASSED' if test_symbol == mapped_symbol else 'FAILED'}")
        
        print("\nAll tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
        raise

if __name__ == "__main__":
    test_symbol_mapper()
