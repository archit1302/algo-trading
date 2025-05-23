"""
Assignment 7: Market Scanner Solution
Scans multiple instruments and identifies trading opportunities using Upstox API v3
"""

import os
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class UpstoxMarketScanner:
    def __init__(self):
        self.access_token = os.getenv('UPSTOX_ACCESS_TOKEN')
        self.base_url = "https://api.upstox.com/v2"
        self.headers = {
            'Authorization': f'Bearer {self.access_token}',
            'Accept': 'application/json'
        }
        
        # Define scanning criteria
        self.scan_criteria = {
            'volume_spike': {'threshold': 2.0, 'description': 'Volume > 2x average'},
            'price_breakout': {'threshold': 0.05, 'description': 'Price breakout > 5%'},
            'high_momentum': {'threshold': 0.03, 'description': 'Daily return > 3%'},
            'low_volatility': {'threshold': 0.02, 'description': 'Volatility < 2%'},
            'oversold': {'rsi_threshold': 30, 'description': 'RSI < 30 (Oversold)'},
            'overbought': {'rsi_threshold': 70, 'description': 'RSI > 70 (Overbought)'}
        }
    
    def get_current_quote(self, symbol, exchange='NSE_EQ'):
        """Get current market quote"""
        try:
            url = f"{self.base_url}/market-quote/quotes"
            params = {'symbol': f'{exchange}:{symbol}'}
            
            response = requests.get(url, headers=self.headers, params=params)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'success':
                    quote_data = data.get('data', {}).get(f'{exchange}:{symbol}', {})
                    return quote_data
            return None
        except Exception as e:
            print(f"Error getting quote for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol, days=30, exchange='NSE_EQ'):
        """Get historical data for analysis"""
        try:
            instrument_key = f'{exchange}:{symbol}'
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            from_date = start_date.strftime('%Y-%m-%d')
            to_date = end_date.strftime('%Y-%m-%d')
            
            url = f"{self.base_url}/historical-candle/{instrument_key}/1day/{to_date}/{from_date}"
            
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
            print(f"Error getting historical data for {symbol}: {e}")
            return None
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI (Relative Strength Index)"""
        if len(prices) < period + 1:
            return None
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else None
    
    def calculate_technical_indicators(self, df):
        """Calculate technical indicators for scanning"""
        if df is None or len(df) < 20:
            return {}
        
        current_price = df['close'].iloc[-1]
        previous_close = df['close'].iloc[-2] if len(df) > 1 else current_price
        
        # Basic metrics
        daily_return = (current_price - previous_close) / previous_close
        volume_avg_20 = df['volume'].tail(20).mean()
        current_volume = df['volume'].iloc[-1]
        volume_ratio = current_volume / volume_avg_20 if volume_avg_20 > 0 else 0
        
        # Price levels
        high_20 = df['high'].tail(20).max()
        low_20 = df['low'].tail(20).min()
        price_position = (current_price - low_20) / (high_20 - low_20) if high_20 != low_20 else 0.5
        
        # Volatility
        returns = df['close'].pct_change().tail(20)
        volatility = returns.std()
        
        # Moving averages
        ma_5 = df['close'].tail(5).mean()
        ma_20 = df['close'].tail(20).mean()
        
        # RSI
        rsi = self.calculate_rsi(df['close'])
        
        return {
            'current_price': current_price,
            'daily_return': daily_return,
            'volume_ratio': volume_ratio,
            'price_position': price_position,
            'volatility': volatility,
            'ma_5': ma_5,
            'ma_20': ma_20,
            'rsi': rsi,
            'high_20d': high_20,
            'low_20d': low_20
        }
    
    def scan_symbol(self, symbol):
        """Scan a single symbol for opportunities"""
        # Get current quote
        quote = self.get_current_quote(symbol)
        if not quote:
            return None
        
        # Get historical data
        hist_data = self.get_historical_data(symbol)
        if hist_data is None or hist_data.empty:
            return None
        
        # Calculate indicators
        indicators = self.calculate_technical_indicators(hist_data)
        if not indicators:
            return None
        
        # Identify signals
        signals = self.identify_signals(indicators)
        
        # Compile scan result
        scan_result = {
            'symbol': symbol,
            'current_price': indicators.get('current_price', 0),
            'daily_return_pct': indicators.get('daily_return', 0) * 100,
            'volume_ratio': indicators.get('volume_ratio', 0),
            'rsi': indicators.get('rsi'),
            'signals': signals,
            'signal_count': len(signals),
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return scan_result
    
    def identify_signals(self, indicators):
        """Identify trading signals based on criteria"""
        signals = []
        
        # Volume spike
        if indicators.get('volume_ratio', 0) > self.scan_criteria['volume_spike']['threshold']:
            signals.append({
                'type': 'volume_spike',
                'description': f"Volume spike: {indicators['volume_ratio']:.1f}x average",
                'strength': min(indicators['volume_ratio'] / 2, 5)  # Cap at 5
            })
        
        # Price momentum
        daily_return = indicators.get('daily_return', 0)
        if abs(daily_return) > self.scan_criteria['high_momentum']['threshold']:
            direction = 'bullish' if daily_return > 0 else 'bearish'
            signals.append({
                'type': 'high_momentum',
                'description': f"High momentum ({direction}): {daily_return*100:.2f}%",
                'strength': min(abs(daily_return) * 100 / 3, 5)
            })
        
        # Price breakout
        price_position = indicators.get('price_position', 0.5)
        if price_position > 0.95:
            signals.append({
                'type': 'price_breakout',
                'description': f"Near 20-day high: {price_position*100:.1f}% of range",
                'strength': 4
            })
        elif price_position < 0.05:
            signals.append({
                'type': 'price_breakdown',
                'description': f"Near 20-day low: {price_position*100:.1f}% of range",
                'strength': 4
            })
        
        # RSI signals
        rsi = indicators.get('rsi')
        if rsi is not None:
            if rsi < self.scan_criteria['oversold']['rsi_threshold']:
                signals.append({
                    'type': 'oversold',
                    'description': f"Oversold condition: RSI {rsi:.1f}",
                    'strength': (30 - rsi) / 10
                })
            elif rsi > self.scan_criteria['overbought']['rsi_threshold']:
                signals.append({
                    'type': 'overbought',
                    'description': f"Overbought condition: RSI {rsi:.1f}",
                    'strength': (rsi - 70) / 10
                })
        
        # Low volatility (consolidation)
        volatility = indicators.get('volatility', 0)
        if volatility < self.scan_criteria['low_volatility']['threshold']:
            signals.append({
                'type': 'low_volatility',
                'description': f"Low volatility: {volatility*100:.2f}% (Potential breakout setup)",
                'strength': 2
            })
        
        # Moving average signals
        ma_5 = indicators.get('ma_5', 0)
        ma_20 = indicators.get('ma_20', 0)
        current_price = indicators.get('current_price', 0)
        
        if ma_5 > ma_20 and current_price > ma_5:
            signals.append({
                'type': 'bullish_ma',
                'description': "Bullish MA alignment (5 > 20, Price > MA5)",
                'strength': 3
            })
        elif ma_5 < ma_20 and current_price < ma_5:
            signals.append({
                'type': 'bearish_ma',
                'description': "Bearish MA alignment (5 < 20, Price < MA5)",
                'strength': 3
            })
        
        return signals
    
    def scan_market(self, symbols, delay=0.5):
        """Scan multiple symbols"""
        print(f"Starting market scan for {len(symbols)} symbols...")
        print("-" * 60)
        
        scan_results = []
        
        for i, symbol in enumerate(symbols, 1):
            print(f"Scanning {i}/{len(symbols)}: {symbol}")
            
            result = self.scan_symbol(symbol)
            if result:
                scan_results.append(result)
                signal_count = result['signal_count']
                print(f"  ✅ Found {signal_count} signal(s)")
                
                # Print signals
                for signal in result['signals']:
                    strength_stars = "★" * int(signal['strength'])
                    print(f"    {strength_stars} {signal['description']}")
            else:
                print(f"  ❌ Failed to scan")
            
            # Rate limiting
            if i < len(symbols):
                time.sleep(delay)
        
        return scan_results
    
    def filter_results(self, scan_results, min_signals=1, signal_types=None):
        """Filter scan results based on criteria"""
        filtered_results = []
        
        for result in scan_results:
            # Minimum signals filter
            if result['signal_count'] < min_signals:
                continue
            
            # Signal type filter
            if signal_types:
                result_signal_types = [s['type'] for s in result['signals']]
                if not any(st in result_signal_types for st in signal_types):
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def create_scan_report(self, scan_results, output_file='market_scan_report.csv'):
        """Create detailed scan report"""
        if not scan_results:
            print("No scan results to report")
            return
        
        # Prepare data for DataFrame
        report_data = []
        
        for result in scan_results:
            # Calculate total signal strength
            total_strength = sum(s['strength'] for s in result['signals'])
            
            # Get signal types
            signal_types = [s['type'] for s in result['signals']]
            signal_descriptions = [s['description'] for s in result['signals']]
            
            report_data.append({
                'Symbol': result['symbol'],
                'Current_Price': result['current_price'],
                'Daily_Return_%': round(result['daily_return_pct'], 2),
                'Volume_Ratio': round(result['volume_ratio'], 2),
                'RSI': round(result['rsi'], 1) if result['rsi'] else 'N/A',
                'Signal_Count': result['signal_count'],
                'Total_Strength': round(total_strength, 1),
                'Signal_Types': ', '.join(signal_types),
                'Top_Signal': signal_descriptions[0] if signal_descriptions else 'None',
                'Last_Updated': result['last_updated']
            })
        
        # Create DataFrame and sort by total strength
        df = pd.DataFrame(report_data)
        df = df.sort_values('Total_Strength', ascending=False)
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        
        # Print summary
        print(f"\nMarket Scan Report")
        print("=" * 80)
        print(df.to_string(index=False))
        print(f"\nReport saved to: {output_file}")
        
        return df
    
    def print_top_opportunities(self, scan_results, top_n=5):
        """Print top opportunities"""
        if not scan_results:
            return
        
        # Sort by signal count and total strength
        sorted_results = sorted(
            scan_results,
            key=lambda x: (x['signal_count'], sum(s['strength'] for s in x['signals'])),
            reverse=True
        )
        
        print(f"\nTop {min(top_n, len(sorted_results))} Opportunities:")
        print("=" * 60)
        
        for i, result in enumerate(sorted_results[:top_n], 1):
            total_strength = sum(s['strength'] for s in result['signals'])
            print(f"\n{i}. {result['symbol']} (Price: ₹{result['current_price']:.2f})")
            print(f"   Signals: {result['signal_count']} | Strength: {total_strength:.1f}")
            print(f"   Daily Return: {result['daily_return_pct']:.2f}%")
            
            for signal in result['signals']:
                strength_indicator = "★" * int(signal['strength'])
                print(f"   {strength_indicator} {signal['description']}")

def main():
    # Initialize scanner
    scanner = UpstoxMarketScanner()
    
    # Define symbols to scan (mix of large cap, mid cap stocks)
    symbols = [
        'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
        'SBIN', 'HDFC', 'ITC', 'LT', 'KOTAKBANK',
        'BAJFINANCE', 'MARUTI', 'ASIANPAINT', 'WIPRO', 'ONGC',
        'TATASTEEL', 'HINDALCO', 'JSWSTEEL', 'NTPC', 'POWERGRID'
    ]
    
    # Perform market scan
    scan_results = scanner.scan_market(symbols, delay=0.5)
    
    print(f"\nScan completed. Analyzed {len(scan_results)} symbols successfully.")
    
    # Filter results (symbols with at least 2 signals)
    filtered_results = scanner.filter_results(scan_results, min_signals=2)
    print(f"Found {len(filtered_results)} symbols with significant signals.")
    
    # Create comprehensive report
    report_df = scanner.create_scan_report(scan_results)
    
    # Show top opportunities
    scanner.print_top_opportunities(scan_results, top_n=5)
    
    # Additional analysis
    if scan_results:
        print(f"\nScan Statistics:")
        print("-" * 40)
        
        total_signals = sum(r['signal_count'] for r in scan_results)
        symbols_with_signals = len([r for r in scan_results if r['signal_count'] > 0])
        
        print(f"Total signals found: {total_signals}")
        print(f"Symbols with signals: {symbols_with_signals}/{len(scan_results)}")
        print(f"Average signals per symbol: {total_signals/len(scan_results):.1f}")
        
        # Signal type breakdown
        signal_types = {}
        for result in scan_results:
            for signal in result['signals']:
                signal_type = signal['type']
                signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
        
        if signal_types:
            print(f"\nSignal Type Breakdown:")
            for signal_type, count in sorted(signal_types.items(), key=lambda x: x[1], reverse=True):
                print(f"  {signal_type}: {count}")

if __name__ == "__main__":
    main()
