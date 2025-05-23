# Note 5: File Handling for Financial Data

## Introduction

File handling is crucial in financial programming as you'll frequently work with CSV files containing stock prices, trading records, portfolio data, and market information. This note covers reading, writing, and processing financial data files efficiently.

## Working with CSV Files

CSV (Comma Separated Values) is the most common format for financial data. Let's learn how to handle CSV files containing stock market data.

### Reading CSV Files - Basic Method

```python
# Basic file reading (without external libraries)
def read_stock_data_basic(filename):
    """
    Read stock data from CSV file using basic Python
    
    Parameters:
    filename (str): Path to CSV file
    
    Returns:
    list: List of dictionaries containing stock data
    """
    stock_data = []
    
    try:
        with open(filename, 'r') as file:
            lines = file.readlines()
            
            # Get header (first line)
            header = lines[0].strip().split(',')
            
            # Process data lines
            for line in lines[1:]:
                values = line.strip().split(',')
                
                # Create dictionary for each row
                row_data = {}
                for i, value in enumerate(values):
                    column_name = header[i].lower()
                    
                    # Convert numeric values
                    if column_name in ['open', 'high', 'low', 'close', 'volume']:
                        try:
                            row_data[column_name] = float(value)
                        except ValueError:
                            row_data[column_name] = 0.0
                    else:
                        row_data[column_name] = value
                
                stock_data.append(row_data)
    
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found")
        return []
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    
    return stock_data

# Example usage
# stock_data = read_stock_data_basic('SBIN_20250415.csv')
```

### Reading CSV Files with csv Module

```python
import csv
from datetime import datetime

def read_stock_csv(filename):
    """
    Read stock data using Python's csv module
    
    Expected CSV format:
    Date,Open,High,Low,Close,Volume
    2025-04-15,720.00,728.45,718.50,725.50,1234567
    """
    stock_data = []
    
    try:
        with open(filename, 'r') as file:
            csv_reader = csv.DictReader(file)
            
            for row in csv_reader:
                # Process each row
                processed_row = {
                    'date': row['Date'],
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                }
                stock_data.append(processed_row)
    
    except FileNotFoundError:
        print(f"File '{filename}' not found")
        return []
    except KeyError as e:
        print(f"Missing column in CSV: {e}")
        return []
    except ValueError as e:
        print(f"Invalid data format: {e}")
        return []
    
    return stock_data

def analyze_stock_file(filename):
    """
    Analyze stock data from file and return summary
    """
    data = read_stock_csv(filename)
    
    if not data:
        return None
    
    # Extract closing prices
    closes = [row['close'] for row in data]
    volumes = [row['volume'] for row in data]
    
    analysis = {
        'symbol': filename.split('_')[0],  # Extract symbol from filename
        'period_start': data[0]['date'],
        'period_end': data[-1]['date'],
        'total_days': len(data),
        'price_range': {
            'high': max(row['high'] for row in data),
            'low': min(row['low'] for row in data),
            'start': data[0]['close'],
            'end': data[-1]['close']
        },
        'volume_stats': {
            'avg_volume': sum(volumes) / len(volumes),
            'max_volume': max(volumes),
            'min_volume': min(volumes)
        },
        'return': ((data[-1]['close'] - data[0]['close']) / data[0]['close']) * 100
    }
    
    return analysis

# Example usage
analysis = analyze_stock_file('data/SBIN_20250415.csv')
if analysis:
    print(f"Stock: {analysis['symbol']}")
    print(f"Period: {analysis['period_start']} to {analysis['period_end']}")
    print(f"Total Return: {analysis['return']:.2f}%")
    print(f"Price Range: ₹{analysis['price_range']['low']} - ₹{analysis['price_range']['high']}")
```

## Writing CSV Files

### Creating Stock Data Files

```python
import csv
from datetime import datetime, timedelta

def write_stock_data(filename, stock_data):
    """
    Write stock data to CSV file
    
    Parameters:
    filename (str): Output filename
    stock_data (list): List of dictionaries with stock data
    """
    try:
        with open(filename, 'w', newline='') as file:
            fieldnames = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Write data
            for row in stock_data:
                writer.writerow({
                    'Date': row['date'],
                    'Open': f"{row['open']:.2f}",
                    'High': f"{row['high']:.2f}",
                    'Low': f"{row['low']:.2f}",
                    'Close': f"{row['close']:.2f}",
                    'Volume': row['volume']
                })
        
        print(f"Successfully wrote {len(stock_data)} records to {filename}")
    
    except Exception as e:
        print(f"Error writing file: {e}")

def generate_sample_data(symbol, start_price=100, days=30):
    """
    Generate sample stock data for testing
    """
    import random
    
    data = []
    current_price = start_price
    current_date = datetime.now() - timedelta(days=days)
    
    for i in range(days):
        # Simulate price movement
        change_percent = random.uniform(-0.05, 0.05)  # -5% to +5%
        price_change = current_price * change_percent
        
        # Calculate OHLC
        open_price = current_price
        close_price = current_price + price_change
        
        # High and low based on open and close
        high_price = max(open_price, close_price) * random.uniform(1.0, 1.02)
        low_price = min(open_price, close_price) * random.uniform(0.98, 1.0)
        
        # Volume
        volume = random.randint(1000000, 5000000)
        
        data.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'open': round(open_price, 2),
            'high': round(high_price, 2),
            'low': round(low_price, 2),
            'close': round(close_price, 2),
            'volume': volume
        })
        
        current_price = close_price
        current_date += timedelta(days=1)
    
    return data

# Generate and save sample data
sample_data = generate_sample_data('SAMPLE', 720, 10)
write_stock_data('sample_stock_data.csv', sample_data)
```

### Creating Portfolio Reports

```python
def export_portfolio_report(portfolio, filename):
    """
    Export portfolio analysis to CSV file
    """
    try:
        with open(filename, 'w', newline='') as file:
            fieldnames = ['Symbol', 'Quantity', 'Avg_Price', 'Current_Price', 
                         'Invested', 'Current_Value', 'PnL', 'PnL_Percent']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            writer.writeheader()
            
            total_invested = 0
            total_current = 0
            
            for symbol, details in portfolio.items():
                qty = details['quantity']
                avg_price = details['avg_price']
                current_price = details['current_price']
                
                invested = qty * avg_price
                current_value = qty * current_price
                pnl = current_value - invested
                pnl_percent = (pnl / invested) * 100
                
                total_invested += invested
                total_current += current_value
                
                writer.writerow({
                    'Symbol': symbol,
                    'Quantity': qty,
                    'Avg_Price': f"{avg_price:.2f}",
                    'Current_Price': f"{current_price:.2f}",
                    'Invested': f"{invested:.2f}",
                    'Current_Value': f"{current_value:.2f}",
                    'PnL': f"{pnl:.2f}",
                    'PnL_Percent': f"{pnl_percent:.2f}"
                })
            
            # Add summary row
            total_pnl = total_current - total_invested
            total_pnl_percent = (total_pnl / total_invested) * 100
            
            writer.writerow({
                'Symbol': 'TOTAL',
                'Quantity': '',
                'Avg_Price': '',
                'Current_Price': '',
                'Invested': f"{total_invested:.2f}",
                'Current_Value': f"{total_current:.2f}",
                'PnL': f"{total_pnl:.2f}",
                'PnL_Percent': f"{total_pnl_percent:.2f}"
            })
        
        print(f"Portfolio report exported to {filename}")
    
    except Exception as e:
        print(f"Error exporting portfolio: {e}")

# Example usage
portfolio = {
    "SBIN": {"quantity": 100, "avg_price": 720.00, "current_price": 725.50},
    "RELIANCE": {"quantity": 50, "avg_price": 2450.00, "current_price": 2456.75},
    "TCS": {"quantity": 25, "avg_price": 3650.00, "current_price": 3678.90}
}

export_portfolio_report(portfolio, 'portfolio_report.csv')
```

## Processing Multiple Files

### Batch Processing Stock Files

```python
import os
import glob

def process_multiple_stock_files(directory_path, file_pattern="*.csv"):
    """
    Process multiple stock files in a directory
    
    Parameters:
    directory_path (str): Path to directory containing files
    file_pattern (str): Pattern to match files (default: "*.csv")
    
    Returns:
    dict: Analysis results for each file
    """
    results = {}
    
    # Find all matching files
    file_pattern_full = os.path.join(directory_path, file_pattern)
    csv_files = glob.glob(file_pattern_full)
    
    if not csv_files:
        print(f"No files found matching pattern: {file_pattern_full}")
        return results
    
    print(f"Found {len(csv_files)} files to process")
    
    for file_path in csv_files:
        filename = os.path.basename(file_path)
        print(f"Processing: {filename}")
        
        try:
            analysis = analyze_stock_file(file_path)
            if analysis:
                results[filename] = analysis
            else:
                print(f"  - Failed to analyze {filename}")
        
        except Exception as e:
            print(f"  - Error processing {filename}: {e}")
    
    return results

def create_multi_stock_summary(results, output_filename):
    """
    Create summary report from multiple stock analyses
    """
    try:
        with open(output_filename, 'w', newline='') as file:
            fieldnames = ['Symbol', 'Period_Start', 'Period_End', 'Days', 
                         'Start_Price', 'End_Price', 'Return_Pct', 'High', 'Low', 'Avg_Volume']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            writer.writeheader()
            
            for filename, analysis in results.items():
                writer.writerow({
                    'Symbol': analysis['symbol'],
                    'Period_Start': analysis['period_start'],
                    'Period_End': analysis['period_end'],
                    'Days': analysis['total_days'],
                    'Start_Price': f"{analysis['price_range']['start']:.2f}",
                    'End_Price': f"{analysis['price_range']['end']:.2f}",
                    'Return_Pct': f"{analysis['return']:.2f}",
                    'High': f"{analysis['price_range']['high']:.2f}",
                    'Low': f"{analysis['price_range']['low']:.2f}",
                    'Avg_Volume': f"{analysis['volume_stats']['avg_volume']:.0f}"
                })
        
        print(f"Multi-stock summary saved to {output_filename}")
    
    except Exception as e:
        print(f"Error creating summary: {e}")

# Example usage
# results = process_multiple_stock_files('data/', 'SBIN_*.csv')
# create_multi_stock_summary(results, 'multi_stock_summary.csv')
```

## File Management Utilities

### File Organization Functions

```python
import os
import shutil
from datetime import datetime

def organize_stock_files(source_dir, organized_dir):
    """
    Organize stock files by symbol into subdirectories
    """
    if not os.path.exists(organized_dir):
        os.makedirs(organized_dir)
    
    files = [f for f in os.listdir(source_dir) if f.endswith('.csv')]
    
    for filename in files:
        # Extract symbol from filename (assumes format: SYMBOL_YYYYMMDD.csv)
        try:
            symbol = filename.split('_')[0]
            
            # Create symbol directory if it doesn't exist
            symbol_dir = os.path.join(organized_dir, symbol)
            if not os.path.exists(symbol_dir):
                os.makedirs(symbol_dir)
            
            # Move file
            source_path = os.path.join(source_dir, filename)
            dest_path = os.path.join(symbol_dir, filename)
            shutil.copy2(source_path, dest_path)
            
            print(f"Moved {filename} to {symbol}/ directory")
        
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def backup_data_files(source_dir, backup_dir):
    """
    Create backup of data files with timestamp
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_folder = os.path.join(backup_dir, f"backup_{timestamp}")
    
    try:
        shutil.copytree(source_dir, backup_folder)
        print(f"Backup created: {backup_folder}")
    except Exception as e:
        print(f"Backup failed: {e}")

def clean_old_files(directory, days_old=30):
    """
    Remove files older than specified days
    """
    cutoff_time = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
    removed_count = 0
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        
        if os.path.isfile(file_path):
            file_time = os.path.getmtime(file_path)
            
            if file_time < cutoff_time:
                try:
                    os.remove(file_path)
                    print(f"Removed old file: {filename}")
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {filename}: {e}")
    
    print(f"Cleaned {removed_count} old files")
```

## Error Handling and Data Validation

### Robust File Processing

```python
def validate_stock_data(data):
    """
    Validate stock data for common issues
    
    Returns:
    tuple: (is_valid, error_messages)
    """
    errors = []
    
    if not data:
        return False, ["No data provided"]
    
    required_fields = ['date', 'open', 'high', 'low', 'close', 'volume']
    
    for i, row in enumerate(data):
        row_num = i + 1
        
        # Check required fields
        for field in required_fields:
            if field not in row:
                errors.append(f"Row {row_num}: Missing field '{field}'")
        
        # Validate price relationships
        try:
            if row['high'] < row['low']:
                errors.append(f"Row {row_num}: High ({row['high']}) < Low ({row['low']})")
            
            if row['open'] < 0 or row['high'] < 0 or row['low'] < 0 or row['close'] < 0:
                errors.append(f"Row {row_num}: Negative prices found")
            
            if row['volume'] < 0:
                errors.append(f"Row {row_num}: Negative volume")
        
        except (KeyError, TypeError) as e:
            errors.append(f"Row {row_num}: Data validation error - {e}")
    
    return len(errors) == 0, errors

def safe_read_stock_file(filename):
    """
    Read stock file with comprehensive error handling and validation
    """
    print(f"Reading file: {filename}")
    
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File '{filename}' does not exist")
        return None
    
    # Check file size
    file_size = os.path.getsize(filename)
    if file_size == 0:
        print(f"Error: File '{filename}' is empty")
        return None
    
    # Read data
    try:
        data = read_stock_csv(filename)
        
        if not data:
            print("No data could be read from file")
            return None
        
        # Validate data
        is_valid, errors = validate_stock_data(data)
        
        if not is_valid:
            print("Data validation failed:")
            for error in errors[:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(errors) > 10:
                print(f"  ... and {len(errors) - 10} more errors")
            return None
        
        print(f"Successfully read and validated {len(data)} records")
        return data
    
    except Exception as e:
        print(f"Unexpected error reading file: {e}")
        return None

# Example usage
stock_data = safe_read_stock_file('data/SBIN_20250415.csv')
if stock_data:
    print("File processed successfully")
    # Continue with analysis
else:
    print("File processing failed")
```

## Best Practices for Financial File Handling

### 1. Consistent File Naming

```python
def generate_filename(symbol, date, data_type="daily"):
    """
    Generate consistent filename for stock data
    
    Examples:
    - SBIN_20250415_daily.csv
    - RELIANCE_20250415_intraday.csv
    """
    date_str = date.strftime('%Y%m%d') if hasattr(date, 'strftime') else str(date)
    return f"{symbol}_{date_str}_{data_type}.csv"

# Usage
from datetime import date
filename = generate_filename('SBIN', date.today(), 'daily')
print(filename)  # SBIN_20250524_daily.csv
```

### 2. Data Backup and Recovery

```python
def create_data_checkpoint(data, checkpoint_name):
    """
    Create a checkpoint of current data state
    """
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{checkpoint_name}_{timestamp}.csv"
    filepath = os.path.join(checkpoint_dir, filename)
    
    write_stock_data(filepath, data)
    print(f"Checkpoint created: {filepath}")
    return filepath
```

### 3. Memory-Efficient Processing

```python
def process_large_file_chunked(filename, chunk_size=1000):
    """
    Process large files in chunks to save memory
    """
    def process_chunk(chunk_data):
        # Process each chunk
        total_volume = sum(row['volume'] for row in chunk_data)
        avg_price = sum(row['close'] for row in chunk_data) / len(chunk_data)
        return {'volume': total_volume, 'avg_price': avg_price}
    
    chunk = []
    results = []
    
    try:
        with open(filename, 'r') as file:
            csv_reader = csv.DictReader(file)
            
            for row in csv_reader:
                # Convert numeric fields
                processed_row = {
                    'date': row['Date'],
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                }
                chunk.append(processed_row)
                
                # Process chunk when it reaches chunk_size
                if len(chunk) >= chunk_size:
                    result = process_chunk(chunk)
                    results.append(result)
                    chunk = []  # Clear chunk
            
            # Process remaining data
            if chunk:
                result = process_chunk(chunk)
                results.append(result)
    
    except Exception as e:
        print(f"Error processing file in chunks: {e}")
        return None
    
    return results
```

## Summary

In this note, you learned:
- How to read and write CSV files containing financial data
- Processing multiple stock data files efficiently
- Creating portfolio reports and summaries
- Organizing and managing financial data files
- Error handling and data validation techniques
- Best practices for file naming and data backup

## Next Steps

In the next note, we'll explore error handling and debugging techniques to make your financial programs more robust and reliable.

---

**Key Takeaways:**
- Always validate financial data after reading from files
- Use consistent file naming conventions
- Implement proper error handling for file operations
- Create backups before processing important data
- Process large files in chunks to manage memory usage
- Organize files systematically for easy maintenance
