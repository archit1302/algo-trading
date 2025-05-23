"""
Assignment 5: Batch Processing Solution
Fetches historical data for multiple instruments using Upstox API v3
"""

import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class UpstoxBatchProcessor:
    def __init__(self):
        self.access_token = os.getenv('UPSTOX_ACCESS_TOKEN')
        self.base_url = "https://api.upstox.com/v2"
        self.headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }
        
    def get_instrument_key(self, symbol, exchange='NSE_EQ'):
        """Get instrument key for a symbol"""
        try:
            url = f"{self.base_url}/market-quote/quotes"
            params = {'symbol': f'{exchange}:{symbol}'}
            
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    return f'{exchange}:{symbol}'
            return None
        except Exception as e:
            print(f"Error getting instrument key for {symbol}: {e}")
            return None
    
    def fetch_historical_data(self, instrument_key, from_date, to_date, interval='1day'):
        """Fetch historical data for a single instrument"""
        try:
            url = f"{self.base_url}/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
            
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and data.get('data', {}).get('candles'):
                    candles = data['data']['candles']
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp')
                    return df
            return None
        except Exception as e:
            print(f"Error fetching data for {instrument_key}: {e}")
            return None
    
    def batch_fetch_data(self, symbols, from_date, to_date, interval='1day', delay=0.5):
        """Fetch historical data for multiple symbols with rate limiting"""
        results = {}
        total_symbols = len(symbols)
        
        print(f"Starting batch fetch for {total_symbols} symbols...")
        print(f"Date range: {from_date} to {to_date}")
        print(f"Interval: {interval}")
        print("-" * 50)
        
        for i, symbol in enumerate(symbols, 1):
            print(f"Processing {i}/{total_symbols}: {symbol}")
            
            # Get instrument key
            instrument_key = self.get_instrument_key(symbol)
            if not instrument_key:
                print(f"  ❌ Could not get instrument key for {symbol}")
                results[symbol] = None
                continue
            
            # Fetch historical data
            df = self.fetch_historical_data(instrument_key, from_date, to_date, interval)
            if df is not None and not df.empty:
                results[symbol] = df
                print(f"  ✅ Fetched {len(df)} records")
            else:
                results[symbol] = None
                print(f"  ❌ No data available")
            
            # Rate limiting
            if i < total_symbols:
                time.sleep(delay)
        
        print("-" * 50)
        successful_fetches = sum(1 for v in results.values() if v is not None)
        print(f"Batch processing complete: {successful_fetches}/{total_symbols} successful")
        
        return results
    
    def save_batch_data(self, batch_results, output_dir='batch_data'):
        """Save batch results to CSV files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_files = []
        for symbol, df in batch_results.items():
            if df is not None and not df.empty:
                filename = f"{symbol}_historical_data.csv"
                filepath = os.path.join(output_dir, filename)
                df.to_csv(filepath, index=False)
                saved_files.append(filepath)
                print(f"Saved {symbol} data to {filepath}")
        
        return saved_files
    
    def create_summary_report(self, batch_results):
        """Create a summary report of the batch processing"""
        summary_data = []
        
        for symbol, df in batch_results.items():
            if df is not None and not df.empty:
                summary_data.append({
                    'Symbol': symbol,
                    'Records': len(df),
                    'Start_Date': df['timestamp'].min().strftime('%Y-%m-%d'),
                    'End_Date': df['timestamp'].max().strftime('%Y-%m-%d'),
                    'Avg_Close': round(df['close'].mean(), 2),
                    'Max_High': df['high'].max(),
                    'Min_Low': df['low'].min(),
                    'Total_Volume': df['volume'].sum(),
                    'Status': 'Success'
                })
            else:
                summary_data.append({
                    'Symbol': symbol,
                    'Records': 0,
                    'Start_Date': 'N/A',
                    'End_Date': 'N/A',
                    'Avg_Close': 'N/A',
                    'Max_High': 'N/A',
                    'Min_Low': 'N/A',
                    'Total_Volume': 'N/A',
                    'Status': 'Failed'
                })
        
        summary_df = pd.DataFrame(summary_data)
        return summary_df

def main():
    # Initialize batch processor
    processor = UpstoxBatchProcessor()
    
    # Define symbols to fetch
    symbols = ['SBIN', 'RELIANCE', 'TCS', 'INFY', 'HDFC', 'ICICIBANK', 'HDFCBANK', 'ITC', 'LT', 'WIPRO']
    
    # Define date range (last 30 days)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    from_date = start_date.strftime('%Y-%m-%d')
    to_date = end_date.strftime('%Y-%m-%d')
    
    # Perform batch fetch
    batch_results = processor.batch_fetch_data(
        symbols=symbols,
        from_date=from_date,
        to_date=to_date,
        interval='1day',
        delay=0.5  # 500ms delay between requests
    )
    
    # Save data to files
    saved_files = processor.save_batch_data(batch_results)
    print(f"\nSaved {len(saved_files)} files")
    
    # Create and save summary report
    summary_df = processor.create_summary_report(batch_results)
    summary_file = 'batch_data/batch_summary_report.csv'
    summary_df.to_csv(summary_file, index=False)
    
    print(f"\nBatch Summary Report:")
    print("-" * 80)
    print(summary_df.to_string(index=False))
    print(f"\nSummary report saved to: {summary_file}")
    
    # Calculate success rate
    success_count = len([r for r in batch_results.values() if r is not None])
    success_rate = (success_count / len(symbols)) * 100
    print(f"\nOverall Success Rate: {success_rate:.1f}% ({success_count}/{len(symbols)})")

if __name__ == "__main__":
    main()
