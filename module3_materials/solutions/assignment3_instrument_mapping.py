"""
Solution for Assignment 3: Instrument Mapping
This script demonstrates working with Upstox instrument files and mapping symbols to instrument keys.
"""

import os
import sys
import json
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class UpstoxInstrumentMapper:
    """
    Comprehensive instrument mapping utility for Upstox API v3
    """
    
    def __init__(self, access_token=None):
        """
        Initialize the instrument mapper
        
        Args:
            access_token (str): Optional access token for authenticated requests
        """
        self.access_token = access_token
        self.base_url = "https://api.upstox.com/v3"
        self.instruments_file = "upstox_instruments.json"
        self.instruments_df = None
        self.symbol_to_key = {}
        self.key_to_symbol = {}
        
        # Load instruments if file exists
        if os.path.exists(self.instruments_file):
            self.load_instruments_from_file()
    
    def download_instruments(self, save_to_file=True):
        """
        Download complete instrument list from Upstox API
        
        Args:
            save_to_file (bool): Whether to save the data to file
        
        Returns:
            bool: True if successful
        """
        print("📡 Downloading instrument data from Upstox API...")
        
        url = f"{self.base_url}/master"
        headers = {"Accept": "application/json"}
        
        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            instruments_data = response.json()
            
            if isinstance(instruments_data, list) and len(instruments_data) > 0:
                print(f"✅ Downloaded {len(instruments_data)} instruments")
                
                if save_to_file:
                    with open(self.instruments_file, 'w') as f:
                        json.dump(instruments_data, f, indent=2)
                    print(f"💾 Saved instruments to {self.instruments_file}")
                
                # Process the data
                self.process_instruments_data(instruments_data)
                return True
            else:
                print("❌ Invalid instrument data received")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Error downloading instruments: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def load_instruments_from_file(self):
        """Load instruments from saved file"""
        try:
            with open(self.instruments_file, 'r') as f:
                instruments_data = json.load(f)
            
            print(f"📂 Loaded {len(instruments_data)} instruments from file")
            self.process_instruments_data(instruments_data)
            return True
            
        except Exception as e:
            print(f"❌ Error loading instruments from file: {e}")
            return False
    
    def process_instruments_data(self, instruments_data):
        """
        Process raw instruments data into usable formats
        
        Args:
            instruments_data (list): Raw instruments data from API
        """
        print("🔄 Processing instrument data...")
        
        # Convert to DataFrame
        self.instruments_df = pd.DataFrame(instruments_data)
        
        # Add helpful columns
        self.instruments_df['search_symbol'] = self.instruments_df['tradingsymbol'].str.upper()
        self.instruments_df['exchange_symbol'] = (
            self.instruments_df['exchange'] + ":" + self.instruments_df['tradingsymbol']
        )
        
        # Build lookup dictionaries
        self._build_lookup_tables()
        
        print(f"✅ Processed {len(self.instruments_df)} instruments")
        self._show_data_summary()
    
    def _build_lookup_tables(self):
        """Build fast lookup tables for symbol-to-key conversion"""
        self.symbol_to_key = {}
        self.key_to_symbol = {}
        
        for _, row in self.instruments_df.iterrows():
            # Build exchange:symbol key
            exchange_symbol = f"{row['exchange']}:{row['tradingsymbol']}"
            instrument_token = row['instrument_token']
            
            # Bidirectional mapping
            self.symbol_to_key[exchange_symbol.upper()] = instrument_token
            self.key_to_symbol[instrument_token] = exchange_symbol
        
        print(f"🔗 Built lookup tables for {len(self.symbol_to_key)} symbols")
    
    def _show_data_summary(self):
        """Display summary of loaded instrument data"""
        if self.instruments_df is None or self.instruments_df.empty:
            return
        
        print("\n📊 Instrument Data Summary:")
        
        # Exchange breakdown
        exchange_counts = self.instruments_df['exchange'].value_counts()
        print(f"   📈 Exchanges:")
        for exchange, count in exchange_counts.head(10).items():
            print(f"      {exchange}: {count:,} instruments")
        
        # Instrument types
        if 'instrument_type' in self.instruments_df.columns:
            type_counts = self.instruments_df['instrument_type'].value_counts()
            print(f"\n   🔧 Instrument Types:")
            for inst_type, count in type_counts.head(5).items():
                print(f"      {inst_type}: {count:,} instruments")
    
    def get_instrument_key(self, exchange, symbol):
        """
        Get instrument key for a given exchange and symbol
        
        Args:
            exchange (str): Exchange name (e.g., 'NSE')
            symbol (str): Trading symbol (e.g., 'SBIN')
        
        Returns:
            str: Instrument key or None if not found
        """
        exchange_symbol = f"{exchange.upper()}:{symbol.upper()}"
        return self.symbol_to_key.get(exchange_symbol)
    
    def get_symbol_from_key(self, instrument_key):
        """
        Get exchange:symbol from instrument key
        
        Args:
            instrument_key (str): Instrument key
        
        Returns:
            str: Exchange:Symbol or None if not found
        """
        return self.key_to_symbol.get(instrument_key)
    
    def search_instruments(self, query, exchange=None, limit=10):
        """
        Search for instruments using partial matching
        
        Args:
            query (str): Search query
            exchange (str): Optional exchange filter
            limit (int): Maximum results to return
        
        Returns:
            DataFrame: Matching instruments
        """
        if self.instruments_df is None:
            print("❌ No instrument data loaded")
            return pd.DataFrame()
        
        query = query.upper()
        df = self.instruments_df.copy()
        
        # Apply exchange filter
        if exchange:
            df = df[df['exchange'].str.upper() == exchange.upper()]
        
        # Search in symbol and name
        mask = (
            df['search_symbol'].str.contains(query, na=False) |
            df['name'].str.upper().str.contains(query, na=False)
        )
        
        results = df[mask].head(limit)
        
        # Return useful columns
        if not results.empty:
            return results[['exchange', 'tradingsymbol', 'name', 'instrument_token', 'instrument_type']]
        else:
            return pd.DataFrame()
    
    def get_nse_equity_symbols(self, limit=None):
        """
        Get all NSE equity symbols
        
        Args:
            limit (int): Optional limit on results
        
        Returns:
            DataFrame: NSE equity instruments
        """
        if self.instruments_df is None:
            print("❌ No instrument data loaded")
            return pd.DataFrame()
        
        nse_equity = self.instruments_df[
            (self.instruments_df['exchange'] == 'NSE_EQ') |
            (self.instruments_df['exchange'] == 'NSE')
        ]
        
        if limit:
            nse_equity = nse_equity.head(limit)
        
        return nse_equity[['tradingsymbol', 'name', 'instrument_token']].copy()
    
    def get_popular_symbols(self):
        """
        Get instrument keys for popular Indian stocks
        
        Returns:
            dict: Symbol -> instrument_key mapping
        """
        popular_symbols = [
            ('NSE', 'RELIANCE'),
            ('NSE', 'HDFCBANK'),
            ('NSE', 'INFY'),
            ('NSE', 'ICICIBANK'),
            ('NSE', 'TCS'),
            ('NSE', 'KOTAKBANK'),
            ('NSE', 'HINDUNILVR'),
            ('NSE', 'SBIN'),
            ('NSE', 'BHARTIARTL'),
            ('NSE', 'ITC'),
            ('NSE', 'ASIANPAINT'),
            ('NSE', 'LT'),
            ('NSE', 'AXISBANK'),
            ('NSE', 'MARUTI'),
            ('NSE', 'SUNPHARMA')
        ]
        
        results = {}
        not_found = []
        
        for exchange, symbol in popular_symbols:
            instrument_key = self.get_instrument_key(exchange, symbol)
            if instrument_key:
                results[symbol] = instrument_key
            else:
                not_found.append(f"{exchange}:{symbol}")
        
        if not_found:
            print(f"⚠️  Could not find: {', '.join(not_found)}")
        
        return results
    
    def validate_symbol_list(self, symbol_list):
        """
        Validate a list of symbols and return valid/invalid breakdown
        
        Args:
            symbol_list (list): List of symbols in format ['EXCHANGE:SYMBOL'] or ['SYMBOL']
        
        Returns:
            dict: Results with valid and invalid symbols
        """
        results = {
            'valid': {},
            'invalid': [],
            'summary': {}
        }
        
        for symbol_str in symbol_list:
            if ':' in symbol_str:
                exchange, symbol = symbol_str.split(':', 1)
            else:
                exchange, symbol = 'NSE', symbol_str
            
            instrument_key = self.get_instrument_key(exchange, symbol)
            
            if instrument_key:
                results['valid'][symbol_str] = {
                    'instrument_key': instrument_key,
                    'exchange': exchange,
                    'symbol': symbol
                }
            else:
                results['invalid'].append(symbol_str)
        
        # Add summary
        results['summary'] = {
            'total': len(symbol_list),
            'valid': len(results['valid']),
            'invalid': len(results['invalid']),
            'success_rate': len(results['valid']) / len(symbol_list) * 100 if symbol_list else 0
        }
        
        return results
    
    def export_filtered_instruments(self, exchange=None, instrument_type=None, output_file=None):
        """
        Export filtered instruments to CSV
        
        Args:
            exchange (str): Optional exchange filter
            instrument_type (str): Optional instrument type filter
            output_file (str): Output file path
        
        Returns:
            str: Path to exported file
        """
        if self.instruments_df is None:
            print("❌ No instrument data loaded")
            return None
        
        df = self.instruments_df.copy()
        
        # Apply filters
        if exchange:
            df = df[df['exchange'].str.upper() == exchange.upper()]
        
        if instrument_type:
            df = df[df['instrument_type'].str.upper() == instrument_type.upper()]
        
        # Generate filename if not provided
        if output_file is None:
            filters = []
            if exchange:
                filters.append(exchange.lower())
            if instrument_type:
                filters.append(instrument_type.lower())
            
            filter_str = "_".join(filters) if filters else "all"
            output_file = f"instruments_{filter_str}_{datetime.now().strftime('%Y%m%d')}.csv"
        
        try:
            # Select useful columns for export
            export_columns = ['exchange', 'tradingsymbol', 'name', 'instrument_token', 'instrument_type']
            available_columns = [col for col in export_columns if col in df.columns]
            
            df[available_columns].to_csv(output_file, index=False)
            print(f"💾 Exported {len(df)} instruments to {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Error exporting instruments: {e}")
            return None
    
    def interactive_symbol_lookup(self):
        """
        Interactive symbol lookup utility
        """
        print("\n🔍 Interactive Symbol Lookup")
        print("=" * 35)
        print("Commands:")
        print("  - Enter symbol name to search (e.g., 'SBIN', 'RELIANCE')")
        print("  - Enter 'NSE:SYMBOL' for specific exchange")
        print("  - Enter 'help' for more commands")
        print("  - Enter 'quit' to exit")
        
        while True:
            try:
                query = input("\n🔎 Search: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                elif query.lower() == 'help':
                    self._show_help()
                    continue
                elif not query:
                    continue
                
                # Handle specific exchange:symbol format
                if ':' in query:
                    exchange, symbol = query.split(':', 1)
                    instrument_key = self.get_instrument_key(exchange, symbol)
                    
                    if instrument_key:
                        print(f"✅ Found: {exchange}:{symbol}")
                        print(f"   🔑 Instrument Key: {instrument_key}")
                    else:
                        print(f"❌ Not found: {exchange}:{symbol}")
                else:
                    # Search for matching instruments
                    results = self.search_instruments(query, limit=5)
                    
                    if not results.empty:
                        print(f"✅ Found {len(results)} matches:")
                        for idx, row in results.iterrows():
                            print(f"   {row['exchange']}:{row['tradingsymbol']} - {row['name']}")
                            print(f"      🔑 Key: {row['instrument_token']}")
                    else:
                        print(f"❌ No matches found for: {query}")
            
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n👋 Goodbye!")
    
    def _show_help(self):
        """Show help information"""
        print("\n📚 Help - Available Commands:")
        print("   • Search by symbol: SBIN, RELIANCE, HDFC")
        print("   • Specific exchange: NSE:SBIN, BSE:SBIN")
        print("   • Partial matching: BANK (finds all bank stocks)")
        print("   • Exchange filter: Search 'BANK' then filter by exchange")
        print("   • quit/exit - Exit the lookup tool")

def demonstrate_mapping_features():
    """
    Demonstrate various mapping features
    """
    print("\n🎯 Demonstrating Instrument Mapping Features")
    print("=" * 50)
    
    # Initialize mapper
    mapper = UpstoxInstrumentMapper()
    
    # Download or load instruments
    if not mapper.instruments_df is not None:
        print("📡 Downloading fresh instrument data...")
        if not mapper.download_instruments():
            print("❌ Failed to download instruments")
            return
    
    # Demonstrate symbol lookup
    print("\n1️⃣ Individual Symbol Lookup:")
    test_symbols = [('NSE', 'SBIN'), ('NSE', 'RELIANCE'), ('BSE', 'SBIN')]
    
    for exchange, symbol in test_symbols:
        key = mapper.get_instrument_key(exchange, symbol)
        print(f"   {exchange}:{symbol} -> {key}")
    
    # Demonstrate search functionality
    print("\n2️⃣ Search Functionality:")
    search_results = mapper.search_instruments("BANK", limit=3)
    if not search_results.empty:
        print("   Search results for 'BANK':")
        for _, row in search_results.iterrows():
            print(f"      {row['exchange']}:{row['tradingsymbol']} - {row['name']}")
    
    # Demonstrate popular symbols
    print("\n3️⃣ Popular Symbols:")
    popular = mapper.get_popular_symbols()
    for symbol, key in list(popular.items())[:5]:
        print(f"   {symbol}: {key}")
    
    # Demonstrate validation
    print("\n4️⃣ Symbol Validation:")
    test_list = ['SBIN', 'RELIANCE', 'INVALID', 'NSE:INFY']
    validation_results = mapper.validate_symbol_list(test_list)
    
    print(f"   Valid symbols: {len(validation_results['valid'])}")
    print(f"   Invalid symbols: {len(validation_results['invalid'])}")
    print(f"   Success rate: {validation_results['summary']['success_rate']:.1f}%")
    
    return mapper

def main():
    """
    Main function to demonstrate instrument mapping
    """
    print("Upstox Instrument Mapping - Assignment 3 Solution")
    print("=" * 55)
    
    # Run demonstration
    mapper = demonstrate_mapping_features()
    
    if mapper and mapper.instruments_df is not None:
        # Offer interactive lookup
        print("\n🎮 Interactive Features:")
        print("1. Interactive symbol lookup")
        print("2. Export filtered instruments")
        print("3. Exit")
        
        choice = input("\nChoose an option (1-3): ").strip()
        
        if choice == "1":
            mapper.interactive_symbol_lookup()
        elif choice == "2":
            print("\nExport options:")
            exchange = input("Exchange (or press Enter for all): ").strip() or None
            output_file = mapper.export_filtered_instruments(exchange=exchange)
            if output_file:
                print(f"✅ Exported to {output_file}")
        
    print("\n📚 Key Learnings:")
    print("   • Instrument mapping is essential for API calls")
    print("   • Cache instrument data locally for better performance")
    print("   • Always validate symbols before making API requests")
    print("   • Use search functionality for user-friendly symbol lookup")
    print("   • Different exchanges may have the same symbol")

if __name__ == "__main__":
    main()
