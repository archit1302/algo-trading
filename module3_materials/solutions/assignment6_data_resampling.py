"""
Assignment 6: Data Resampling Solution
Demonstrates different resampling techniques using Upstox historical data
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import matplotlib.pyplot as plt

# Load environment variables
load_dotenv()

class UpstoxDataResampler:
    def __init__(self):
        self.access_token = os.getenv('UPSTOX_ACCESS_TOKEN')
        self.base_url = "https://api.upstox.com/v2"
        self.headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }
    
    def fetch_historical_data(self, symbol, days=90, interval='1day', exchange='NSE_EQ'):
        """Fetch historical data for resampling"""
        try:
            instrument_key = f'{exchange}:{symbol}'
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/historical-candle/{instrument_key}/{interval}/{to_date}/{from_date}"
            
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success' and data.get('data', {}).get('candles'):
                    candles = data['data']['candles']
                    df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'oi'])
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
                    df = df.sort_values('timestamp').set_index('timestamp')
                    return df
            return None
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None
    
    def resample_to_weekly(self, df):
        """Resample daily data to weekly"""
        if df is None or df.empty:
            return None
        
        weekly_data = df.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'oi': 'last'
        }).dropna()
        
        return weekly_data
    
    def resample_to_monthly(self, df):
        """Resample daily data to monthly"""
        if df is None or df.empty:
            return None
        
        monthly_data = df.resample('M').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'oi': 'last'
        }).dropna()
        
        return monthly_data
    
    def calculate_moving_averages(self, df, windows=[5, 10, 20, 50]):
        """Calculate moving averages"""
        if df is None or df.empty:
            return df
        
        result_df = df.copy()
        
        for window in windows:
            if len(df) >= window:
                result_df[f'MA_{window}'] = df['close'].rolling(window=window).mean()
                result_df[f'Volume_MA_{window}'] = df['volume'].rolling(window=window).mean()
        
        return result_df
    
    def calculate_technical_indicators(self, df):
        """Calculate basic technical indicators"""
        if df is None or df.empty:
            return df
        
        result_df = df.copy()
        
        # Daily returns
        result_df['daily_return'] = df['close'].pct_change()
        
        # Volatility (rolling 20-day)
        result_df['volatility_20d'] = df['close'].pct_change().rolling(20).std()
        
        # Price change
        result_df['price_change'] = df['close'] - df['open']
        result_df['price_change_pct'] = (result_df['price_change'] / df['open']) * 100
        
        # High-Low spread
        result_df['hl_spread'] = df['high'] - df['low']
        result_df['hl_spread_pct'] = (result_df['hl_spread'] / df['close']) * 100
        
        # Volume ratios
        result_df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        
        return result_df
    
    def resample_and_analyze(self, symbol, days=90):
        """Complete resampling and analysis workflow"""
        print(f"Analyzing {symbol} with {days} days of data...")
        print("-" * 50)
        
        # Fetch data
        daily_data = self.fetch_historical_data(symbol, days)
        if daily_data is None:
            print(f"❌ Could not fetch data for {symbol}")
            return None
        
        print(f"✅ Fetched {len(daily_data)} daily records")
        
        # Add technical indicators to daily data
        daily_enhanced = self.calculate_technical_indicators(daily_data)
        daily_with_ma = self.calculate_moving_averages(daily_enhanced)
        
        # Resample to different timeframes
        weekly_data = self.resample_to_weekly(daily_data)
        monthly_data = self.resample_to_monthly(daily_data)
        
        results = {
            'daily': daily_with_ma,
            'weekly': weekly_data,
            'monthly': monthly_data
        }
        
        # Print summary statistics
        self.print_resampling_summary(symbol, results)
        
        return results
    
    def print_resampling_summary(self, symbol, results):
        """Print summary of resampled data"""
        print(f"\nResampling Summary for {symbol}:")
        print("=" * 60)
        
        timeframes = ['daily', 'weekly', 'monthly']
        
        for timeframe in timeframes:
            data = results[timeframe]
            if data is not None and not data.empty:
                print(f"\n{timeframe.upper()} Data:")
                print(f"  Records: {len(data)}")
                print(f"  Date Range: {data.index.min().strftime('%Y-%m-%d')} to {data.index.max().strftime('%Y-%m-%d')}")
                print(f"  Avg Close: ₹{data['close'].mean():.2f}")
                print(f"  Price Range: ₹{data['low'].min():.2f} - ₹{data['high'].max():.2f}")
                print(f"  Avg Volume: {data['volume'].mean():,.0f}")
                
                if timeframe == 'daily' and 'daily_return' in data.columns:
                    avg_return = data['daily_return'].mean() * 100
                    volatility = data['daily_return'].std() * 100
                    print(f"  Avg Daily Return: {avg_return:.2f}%")
                    print(f"  Daily Volatility: {volatility:.2f}%")
    
    def save_resampled_data(self, symbol, results, output_dir='resampled_data'):
        """Save resampled data to CSV files"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_files = []
        
        for timeframe, data in results.items():
            if data is not None and not data.empty:
                filename = f"{symbol}_{timeframe}_data.csv"
                filepath = os.path.join(output_dir, filename)
                data.to_csv(filepath)
                saved_files.append(filepath)
                print(f"Saved {timeframe} data to {filepath}")
        
        return saved_files
    
    def create_comparison_chart(self, symbol, results):
        """Create comparison chart of different timeframes"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'{symbol} - Multi-Timeframe Analysis', fontsize=16)
            
            # Daily price with moving averages
            ax1 = axes[0, 0]
            daily_data = results['daily']
            ax1.plot(daily_data.index, daily_data['close'], label='Close Price', linewidth=1)
            if 'MA_20' in daily_data.columns:
                ax1.plot(daily_data.index, daily_data['MA_20'], label='MA 20', alpha=0.7)
            if 'MA_50' in daily_data.columns:
                ax1.plot(daily_data.index, daily_data['MA_50'], label='MA 50', alpha=0.7)
            ax1.set_title('Daily Price with Moving Averages')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Volume comparison
            ax2 = axes[0, 1]
            ax2.bar(daily_data.index, daily_data['volume'], alpha=0.6, width=1)
            if 'Volume_MA_20' in daily_data.columns:
                ax2.plot(daily_data.index, daily_data['Volume_MA_20'], color='red', label='Volume MA 20')
                ax2.legend()
            ax2.set_title('Daily Volume')
            ax2.grid(True, alpha=0.3)
            
            # Weekly vs Monthly comparison
            ax3 = axes[1, 0]
            weekly_data = results['weekly']
            monthly_data = results['monthly']
            
            ax3.plot(daily_data.index, daily_data['close'], label='Daily', alpha=0.5, linewidth=1)
            if weekly_data is not None:
                ax3.plot(weekly_data.index, weekly_data['close'], label='Weekly', linewidth=2, marker='o')
            if monthly_data is not None:
                ax3.plot(monthly_data.index, monthly_data['close'], label='Monthly', linewidth=3, marker='s')
            ax3.set_title('Multi-Timeframe Price Comparison')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Daily returns distribution
            ax4 = axes[1, 1]
            if 'daily_return' in daily_data.columns:
                returns = daily_data['daily_return'].dropna() * 100
                ax4.hist(returns, bins=30, alpha=0.7, edgecolor='black')
                ax4.axvline(returns.mean(), color='red', linestyle='--', label=f'Mean: {returns.mean():.2f}%')
                ax4.set_title('Daily Returns Distribution')
                ax4.set_xlabel('Return (%)')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save chart
            chart_path = f'resampled_data/{symbol}_analysis_chart.png'
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Chart saved to {chart_path}")
            
        except Exception as e:
            print(f"Error creating chart: {e}")

def main():
    # Initialize resampler
    resampler = UpstoxDataResampler()
    
    # Analyze multiple symbols
    symbols = ['RELIANCE', 'TCS', 'SBIN']
    
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        results = resampler.resample_and_analyze(symbol, days=90)
        if results:
            all_results[symbol] = results
            
            # Save data
            saved_files = resampler.save_resampled_data(symbol, results)
            
            # Create chart
            resampler.create_comparison_chart(symbol, results)
        
        print("\n" + "="*60)
    
    # Summary across all symbols
    print(f"\nProcessed {len(all_results)} symbols successfully")
    print("Data saved in 'resampled_data' directory")
    
    # Create comparative analysis
    if len(all_results) > 1:
        print("\nComparative Analysis:")
        print("-" * 40)
        
        for symbol, results in all_results.items():
            daily_data = results['daily']
            if 'daily_return' in daily_data.columns:
                avg_return = daily_data['daily_return'].mean() * 100
                volatility = daily_data['daily_return'].std() * 100
                print(f"{symbol:10} | Return: {avg_return:6.2f}% | Volatility: {volatility:6.2f}%")

if __name__ == "__main__":
    main()
