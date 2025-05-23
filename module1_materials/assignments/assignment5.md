# Assignment 5: File Handling for Market Data

## Objective
Master file operations to process real market data, create automated reports, and build a data management system for trading applications.

## Prerequisites
- Complete Assignments 1-4
- Complete reading: `05_file_handling.md`

## Tasks

### Task 1: CSV Data Processing (30 points)
Work with real stock market data from CSV files:

1. **Data Reading and Analysis**
   - Read the provided `SBIN_sample_data.csv` file
   - Extract and display basic statistics:
     - Total trading days
     - Highest and lowest prices
     - Average daily volume
     - Most volatile day (highest price range)

2. **Data Cleaning**
   - Handle missing values (if any)
   - Remove duplicate entries
   - Validate data types and ranges
   - Save cleaned data to `SBIN_cleaned.csv`

3. **Data Filtering**
   - Extract high-volume trading days (volume > average)
   - Filter data for specific date ranges
   - Find days with price gaps > 2%

### Task 2: Portfolio Data Management (25 points)
Create a comprehensive portfolio management system:

1. **Portfolio Creation**
   - Create a portfolio CSV with columns: Symbol, Quantity, Purchase_Price, Purchase_Date
   - Add multiple stocks with different purchase dates
   - Calculate current values and P&L

2. **Transaction Logging**
   - Create a transaction log system that appends new trades
   - Include: Date, Symbol, Action (BUY/SELL), Quantity, Price
   - Implement proper file locking to prevent data corruption

3. **Report Generation**
   - Generate daily portfolio summary reports
   - Create monthly performance reports
   - Export data in different formats (CSV, TXT)

### Task 3: Market Data Processing (25 points)
Build an automated market data processing system:

1. **Batch File Processing**
   - Process multiple stock CSV files in a directory
   - Merge data from different sources
   - Create consolidated market reports

2. **Technical Indicators File Export**
   - Calculate moving averages, RSI, MACD for each stock
   - Export indicators to separate CSV files
   - Create summary statistics files

3. **Data Backup System**
   - Implement automatic backup of important files
   - Create timestamped backup copies
   - Compress old data files

### Task 4: Error Handling and Logging (20 points)
Implement robust error handling and logging:

1. **Exception Handling**
   - Handle file not found errors gracefully
   - Manage permission denied errors
   - Deal with corrupted data files

2. **Logging System**
   - Create application logs for all file operations
   - Log errors, warnings, and information messages
   - Implement log rotation to manage file sizes

3. **Data Validation**
   - Validate CSV file structures before processing
   - Check data integrity after file operations
   - Create validation reports

## Sample Data Structure

Create these sample CSV files for testing:

**SBIN_sample_data.csv**
```csv
Date,Open,High,Low,Close,Volume
2025-05-01,665.20,678.50,662.10,675.25,2500000
2025-05-02,675.00,682.30,673.80,678.90,1800000
2025-05-03,679.50,685.75,677.20,682.30,2100000
2025-05-04,682.00,688.90,679.50,685.75,1950000
2025-05-05,686.00,692.40,684.20,689.80,2300000
```

**portfolio.csv**
```csv
Symbol,Quantity,Purchase_Price,Purchase_Date,Current_Price
SBIN,100,650.50,2025-04-15,689.80
RELIANCE,50,2890.75,2025-04-10,2915.60
TCS,25,3450.20,2025-04-20,3475.90
INFY,75,1876.30,2025-04-18,1901.75
```

## Implementation Requirements

### File Processing Functions
```python
def read_stock_data(filename):
    """Read stock data from CSV file with error handling"""
    
def calculate_daily_returns(data):
    """Calculate daily returns from price data"""
    
def export_technical_indicators(symbol, data, indicators):
    """Export calculated indicators to CSV"""
    
def backup_data_files(source_dir, backup_dir):
    """Create timestamped backup of data files"""
    
def validate_csv_structure(filename, expected_columns):
    """Validate CSV file structure"""
    
def log_operation(operation, status, details):
    """Log file operations with timestamp"""
```

### Portfolio Management Functions
```python
def add_transaction(symbol, action, quantity, price):
    """Add new transaction to log file"""
    
def calculate_portfolio_value():
    """Calculate current portfolio value from files"""
    
def generate_portfolio_report(date):
    """Generate daily portfolio report"""
    
def export_tax_report(year):
    """Generate annual tax report from transactions"""
```

## Expected Output Files

After completion, your program should create:

```
data/
├── SBIN_cleaned.csv
├── SBIN_indicators.csv
├── portfolio_summary_2025-05-24.csv
├── transaction_log.csv
├── backup/
│   ├── data_backup_2025-05-24_143022.zip
│   └── portfolio_backup_2025-05-24_143022.csv
└── logs/
    ├── application.log
    └── error.log

reports/
├── daily_summary_2025-05-24.txt
├── monthly_report_2025-05.csv
└── tax_report_2025.csv
```

## Expected Console Output
```
=== MARKET DATA PROCESSING ===
Processing SBIN_sample_data.csv...
✓ Data loaded: 5 trading days
✓ Data cleaned: 0 duplicates removed
✓ Statistics calculated:
  - Highest Price: ₹692.40
  - Lowest Price: ₹662.10
  - Average Volume: 2,130,000
  - Most Volatile Day: 2025-05-01 (₹16.40 range)

=== PORTFOLIO ANALYSIS ===
Portfolio loaded: 4 stocks
Current Value: ₹4,18,967.25
Total Investment: ₹4,12,345.50
Profit/Loss: ₹6,621.75 (+1.61%)

✓ Portfolio report generated: portfolio_summary_2025-05-24.csv
✓ Transaction logged: BUY SBIN 100 @ 689.80

=== TECHNICAL INDICATORS ===
✓ SBIN indicators exported: SBIN_indicators.csv
  - 5-day SMA, RSI, MACD calculated
✓ Backup created: data_backup_2025-05-24_143022.zip

=== ERROR HANDLING TEST ===
Testing file not found...
✗ Error handled: FileNotFoundError for missing_file.csv
✓ Logged to error.log

All operations completed successfully!
```

## Submission Guidelines
1. Create a file named `assignment5_solution.py`
2. Include all required functions with proper error handling
3. Create sample data files for testing
4. Implement comprehensive logging
5. Add detailed comments explaining file operations

## Evaluation Criteria
- Correct file reading/writing operations (30%)
- Proper error handling and validation (25%)
- Data processing accuracy (20%)
- Code organization and documentation (15%)
- Logging and backup implementation (10%)

## Bonus Challenges (25 extra points)
1. Implement real-time data monitoring with file watching
2. Create a web dashboard that displays file-based data
3. Add support for different file formats (JSON, XML)
4. Implement data encryption for sensitive files
5. Create automated email reports using file data
