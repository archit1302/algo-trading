"""
Assignment 5 Solution: File Handling for Market Data
Complete solution for reading and writing financial data from CSV files

Author: GitHub Copilot
Module: 1 - Python Fundamentals
Assignment: 5 - File Handling
"""

import csv
import os
from datetime import datetime
import json

class StockDataProcessor:
    """Class to handle stock data file operations"""
    
    def __init__(self):
        self.data_directory = "data"
        self.ensure_data_directory()
    
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_directory):
            os.makedirs(self.data_directory)
            print(f"Created {self.data_directory} directory")
    
    def read_stock_data(self, filename):
        """
        Read stock data from CSV file
        
        Args:
            filename (str): Name of the CSV file
        
        Returns:
            list: List of dictionaries containing stock data
        """
        filepath = os.path.join(self.data_directory, filename)
        stock_data = []
        
        try:
            with open(filepath, 'r', newline='', encoding='utf-8') as file:
                # Use DictReader to automatically handle headers
                reader = csv.DictReader(file)
                
                for row in reader:
                    # Convert numeric fields
                    try:
                        processed_row = {
                            'date': row['date'],
                            'symbol': row['symbol'],
                            'open': float(row['open']),
                            'high': float(row['high']),
                            'low': float(row['low']),
                            'close': float(row['close']),
                            'volume': int(row['volume'])
                        }
                        stock_data.append(processed_row)
                    except (ValueError, KeyError) as e:
                        print(f"Error processing row {reader.line_num}: {e}")
                        continue
                
                print(f"Successfully read {len(stock_data)} records from {filename}")
                return stock_data
        
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found in {self.data_directory} directory")
            return []
        except Exception as e:
            print(f"Error reading file '{filename}': {e}")
            return []
    
    def write_stock_data(self, data, filename):
        """
        Write stock data to CSV file
        
        Args:
            data (list): List of dictionaries with stock data
            filename (str): Output filename
        
        Returns:
            bool: Success status
        """
        if not data:
            print("No data to write")
            return False
        
        filepath = os.path.join(self.data_directory, filename)
        
        try:
            with open(filepath, 'w', newline='', encoding='utf-8') as file:
                # Get fieldnames from first record
                fieldnames = data[0].keys()
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                
                # Write header
                writer.writeheader()
                
                # Write data rows
                writer.writerows(data)
                
                print(f"Successfully wrote {len(data)} records to {filename}")
                return True
        
        except Exception as e:
            print(f"Error writing to file '{filename}': {e}")
            return False
    
    def filter_data_by_symbol(self, data, symbol):
        """
        Filter stock data by symbol
        
        Args:
            data (list): Stock data list
            symbol (str): Stock symbol to filter
        
        Returns:
            list: Filtered data
        """
        filtered_data = [row for row in data if row['symbol'].upper() == symbol.upper()]
        print(f"Filtered {len(filtered_data)} records for symbol {symbol}")
        return filtered_data
    
    def filter_data_by_date_range(self, data, start_date, end_date):
        """
        Filter stock data by date range
        
        Args:
            data (list): Stock data list
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
        
        Returns:
            list: Filtered data
        """
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            filtered_data = []
            for row in data:
                try:
                    row_date = datetime.strptime(row['date'], '%Y-%m-%d')
                    if start_dt <= row_date <= end_dt:
                        filtered_data.append(row)
                except ValueError:
                    continue
            
            print(f"Filtered {len(filtered_data)} records between {start_date} and {end_date}")
            return filtered_data
        
        except ValueError as e:
            print(f"Error parsing dates: {e}")
            return []
    
    def calculate_daily_returns(self, data):
        """
        Calculate daily returns for stock data
        
        Args:
            data (list): Stock data sorted by date
        
        Returns:
            list: Data with daily returns added
        """
        if len(data) < 2:
            print("Need at least 2 data points to calculate returns")
            return data
        
        # Sort data by date
        sorted_data = sorted(data, key=lambda x: x['date'])
        result_data = []
        
        # First day has no return
        first_day = sorted_data[0].copy()
        first_day['daily_return'] = 0.0
        result_data.append(first_day)
        
        # Calculate returns for subsequent days
        for i in range(1, len(sorted_data)):
            current_row = sorted_data[i].copy()
            prev_close = sorted_data[i-1]['close']
            current_close = current_row['close']
            
            daily_return = ((current_close - prev_close) / prev_close) * 100
            current_row['daily_return'] = round(daily_return, 4)
            result_data.append(current_row)
        
        print(f"Calculated daily returns for {len(result_data)} records")
        return result_data
    
    def generate_summary_report(self, data, symbol):
        """
        Generate summary report for stock data
        
        Args:
            data (list): Stock data
            symbol (str): Stock symbol
        
        Returns:
            dict: Summary statistics
        """
        if not data:
            return {}
        
        # Calculate statistics
        closes = [row['close'] for row in data]
        volumes = [row['volume'] for row in data]
        highs = [row['high'] for row in data]
        lows = [row['low'] for row in data]
        
        summary = {
            'symbol': symbol,
            'total_records': len(data),
            'date_range': {
                'start': min(row['date'] for row in data),
                'end': max(row['date'] for row in data)
            },
            'price_statistics': {
                'highest': max(highs),
                'lowest': min(lows),
                'avg_close': round(sum(closes) / len(closes), 2),
                'latest_close': data[-1]['close'] if data else 0
            },
            'volume_statistics': {
                'total_volume': sum(volumes),
                'avg_volume': round(sum(volumes) / len(volumes), 0),
                'max_volume': max(volumes)
            }
        }
        
        # Calculate returns if data has daily_return field
        if 'daily_return' in data[0]:
            returns = [row['daily_return'] for row in data if row['daily_return'] != 0]
            if returns:
                summary['return_statistics'] = {
                    'avg_daily_return': round(sum(returns) / len(returns), 4),
                    'best_day': max(returns),
                    'worst_day': min(returns),
                    'positive_days': len([r for r in returns if r > 0]),
                    'negative_days': len([r for r in returns if r < 0])
                }
        
        return summary
    
    def save_summary_report(self, summary, filename):
        """
        Save summary report to JSON file
        
        Args:
            summary (dict): Summary data
            filename (str): Output filename
        """
        filepath = os.path.join(self.data_directory, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as file:
                json.dump(summary, file, indent=2)
            print(f"Summary report saved to {filename}")
        except Exception as e:
            print(f"Error saving summary report: {e}")

def create_sample_data():
    """Create sample stock data for testing"""
    sample_data = [
        {'date': '2024-01-01', 'symbol': 'SBIN', 'open': 845.00, 'high': 850.50, 'low': 840.25, 'close': 848.75, 'volume': 2500000},
        {'date': '2024-01-02', 'symbol': 'SBIN', 'open': 849.00, 'high': 855.30, 'low': 845.80, 'close': 852.40, 'volume': 2750000},
        {'date': '2024-01-03', 'symbol': 'SBIN', 'open': 851.50, 'high': 858.90, 'low': 849.20, 'close': 856.15, 'volume': 3100000},
        {'date': '2024-01-04', 'symbol': 'SBIN', 'open': 855.25, 'high': 860.75, 'low': 850.30, 'close': 853.80, 'volume': 2890000},
        {'date': '2024-01-05', 'symbol': 'SBIN', 'open': 854.10, 'high': 862.45, 'low': 851.65, 'close': 859.90, 'volume': 3250000},
        
        {'date': '2024-01-01', 'symbol': 'RELIANCE', 'open': 2445.00, 'high': 2460.30, 'low': 2438.75, 'close': 2456.20, 'volume': 1500000},
        {'date': '2024-01-02', 'symbol': 'RELIANCE', 'open': 2458.50, 'high': 2475.80, 'low': 2450.40, 'close': 2468.90, 'volume': 1650000},
        {'date': '2024-01-03', 'symbol': 'RELIANCE', 'open': 2470.25, 'high': 2485.60, 'low': 2462.35, 'close': 2478.45, 'volume': 1820000},
        {'date': '2024-01-04', 'symbol': 'RELIANCE', 'open': 2476.80, 'high': 2490.15, 'low': 2465.25, 'close': 2472.35, 'volume': 1750000},
        {'date': '2024-01-05', 'symbol': 'RELIANCE', 'open': 2474.60, 'high': 2488.90, 'low': 2470.80, 'close': 2485.25, 'volume': 1900000},
    ]
    return sample_data

def main():
    """Demonstrate file handling operations"""
    print("=== Assignment 5: File Handling for Market Data ===\n")
    
    # Initialize processor
    processor = StockDataProcessor()
    
    # Create and save sample data
    print("1. Creating sample stock data...")
    sample_data = create_sample_data()
    processor.write_stock_data(sample_data, 'sample_stock_data.csv')
    print()
    
    # Read data back
    print("2. Reading stock data from file...")
    read_data = processor.read_stock_data('sample_stock_data.csv')
    print(f"Read {len(read_data)} records")
    print()
    
    # Filter by symbol
    print("3. Filtering data by symbol...")
    sbin_data = processor.filter_data_by_symbol(read_data, 'SBIN')
    reliance_data = processor.filter_data_by_symbol(read_data, 'RELIANCE')
    print()
    
    # Filter by date range
    print("4. Filtering data by date range...")
    filtered_data = processor.filter_data_by_date_range(read_data, '2024-01-02', '2024-01-04')
    print()
    
    # Calculate daily returns
    print("5. Calculating daily returns...")
    sbin_with_returns = processor.calculate_daily_returns(sbin_data)
    
    # Display sample returns
    print("SBIN Daily Returns:")
    for row in sbin_with_returns[:3]:  # Show first 3 records
        print(f"  {row['date']}: Close ₹{row['close']:.2f}, Return {row['daily_return']:+.4f}%")
    print()
    
    # Save processed data
    print("6. Saving processed data...")
    processor.write_stock_data(sbin_with_returns, 'sbin_with_returns.csv')
    print()
    
    # Generate summary report
    print("7. Generating summary report...")
    sbin_summary = processor.generate_summary_report(sbin_with_returns, 'SBIN')
    
    print("SBIN Summary:")
    print(f"  Total Records: {sbin_summary['total_records']}")
    print(f"  Date Range: {sbin_summary['date_range']['start']} to {sbin_summary['date_range']['end']}")
    print(f"  Price Range: ₹{sbin_summary['price_statistics']['lowest']:.2f} - ₹{sbin_summary['price_statistics']['highest']:.2f}")
    print(f"  Average Close: ₹{sbin_summary['price_statistics']['avg_close']:.2f}")
    
    if 'return_statistics' in sbin_summary:
        print(f"  Average Daily Return: {sbin_summary['return_statistics']['avg_daily_return']:+.4f}%")
        print(f"  Best Day: {sbin_summary['return_statistics']['best_day']:+.4f}%")
        print(f"  Worst Day: {sbin_summary['return_statistics']['worst_day']:+.4f}%")
    print()
    
    # Save summary report
    print("8. Saving summary report...")
    processor.save_summary_report(sbin_summary, 'sbin_summary.json')
    
    # Show file operations summary
    print("\n=== File Operations Summary ===")
    print("Files created in data/ directory:")
    if os.path.exists(processor.data_directory):
        for file in os.listdir(processor.data_directory):
            filepath = os.path.join(processor.data_directory, file)
            size = os.path.getsize(filepath)
            print(f"  {file} ({size} bytes)")
    
    print("\nAll file operations completed successfully!")

if __name__ == "__main__":
    main()
