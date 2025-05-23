# Pandas Fundamentals for Financial Data

## Introduction
Pandas is the cornerstone of financial data analysis in Python. This comprehensive guide will teach you to manipulate, analyze, and process financial data using pandas DataFrames and Series.

## Table of Contents
1. [Pandas Basics](#pandas-basics)
2. [DataFrames for Stock Data](#dataframes-for-stock-data)
3. [Data Import and Export](#data-import-and-export)
4. [Data Cleaning and Preprocessing](#data-cleaning-and-preprocessing)
5. [Financial Calculations](#financial-calculations)
6. [Time Series Operations](#time-series-operations)
7. [Practical Examples](#practical-examples)

## Pandas Basics

### What is Pandas?
Pandas is a powerful data manipulation library that provides:
- **DataFrame**: 2D labeled data structure (like Excel spreadsheet)
- **Series**: 1D labeled array (like a single column)
- **Efficient data operations**: Filtering, grouping, merging
- **Time series functionality**: Perfect for financial data

### Installation and Import
```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Display all columns and rows
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
```

### Creating DataFrames

#### From Dictionary (Basic Stock Data)
```python
# Create sample stock data
stock_data = {
    'Date': ['2025-05-20', '2025-05-21', '2025-05-22', '2025-05-23', '2025-05-24'],
    'Open': [675.20, 678.50, 682.30, 679.80, 685.40],
    'High': [680.75, 685.20, 688.90, 687.50, 692.60],
    'Low': [672.30, 676.80, 679.50, 677.20, 683.10],
    'Close': [678.50, 682.30, 679.80, 685.40, 690.25],
    'Volume': [2500000, 1800000, 2100000, 1950000, 2300000]
}

df = pd.DataFrame(stock_data)
print("Sample Stock DataFrame:")
print(df)
```

#### From Lists (Portfolio Data)
```python
# Create portfolio data
symbols = ['SBIN', 'RELIANCE', 'TCS', 'INFY', 'HDFC']
prices = [675.25, 2915.60, 3475.90, 1901.75, 1678.45]
quantities = [100, 50, 25, 75, 60]

portfolio_df = pd.DataFrame({
    'Symbol': symbols,
    'Price': prices,
    'Quantity': quantities,
    'Value': [p * q for p, q in zip(prices, quantities)]
})

print("Portfolio DataFrame:")
print(portfolio_df)
```

## DataFrames for Stock Data

### Essential DataFrame Operations

#### Viewing Data
```python
# Load sample data (assuming we have SBIN.csv)
df = pd.read_csv('data/SBIN.csv', parse_dates=['Date'], index_col='Date')

# Basic information
print("DataFrame Info:")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Data types:\n{df.dtypes}")

# View data
print("First 5 rows:")
print(df.head())

print("Last 5 rows:")
print(df.tail())

print("Statistical summary:")
print(df.describe())
```

#### Selecting Data
```python
# Select single column (returns Series)
close_prices = df['Close']
print(f"Close prices type: {type(close_prices)}")

# Select multiple columns (returns DataFrame)
ohlc_data = df[['Open', 'High', 'Low', 'Close']]
print("OHLC Data:")
print(ohlc_data.head())

# Select rows by index
latest_data = df.iloc[-5:]  # Last 5 rows
print("Latest 5 trading days:")
print(latest_data)

# Select rows by date range
recent_data = df.loc['2025-05-01':'2025-05-24']
print("May 2025 data:")
print(recent_data)
```

#### Filtering Data
```python
# Filter by conditions
high_volume_days = df[df['Volume'] > 2000000]
print("High volume trading days:")
print(high_volume_days)

# Multiple conditions
volatile_days = df[(df['High'] - df['Low']) > 15]
print("Volatile trading days (range > ₹15):")
print(volatile_days)

# Filter by date
recent_bullish = df[(df['Close'] > df['Open']) & (df.index >= '2025-05-01')]
print("Recent bullish days:")
print(recent_bullish)
```

## Data Import and Export

### Reading Financial Data

#### CSV Files
```python
# Read CSV with proper date handling
df = pd.read_csv(
    'data/SBIN.csv',
    parse_dates=['Date'],  # Convert Date column to datetime
    index_col='Date',      # Set Date as index
    dtype={                # Specify data types
        'Open': 'float64',
        'High': 'float64', 
        'Low': 'float64',
        'Close': 'float64',
        'Volume': 'int64'
    }
)

# Read with custom date format
df = pd.read_csv(
    'data/custom_format.csv',
    parse_dates=['Date'],
    date_parser=lambda x: pd.to_datetime(x, format='%d-%m-%Y')
)
```

#### Excel Files
```python
# Read Excel file
df = pd.read_excel('data/portfolio.xlsx', sheet_name='Holdings')

# Read multiple sheets
excel_data = pd.read_excel('data/portfolio.xlsx', sheet_name=None)
holdings = excel_data['Holdings']
transactions = excel_data['Transactions']
```

#### API Data (Yahoo Finance)
```python
import yfinance as yf

# Download stock data
ticker = yf.Ticker("SBIN.NS")
df = ticker.history(period="1y", interval="1d")

print("Downloaded SBIN data:")
print(df.head())
```

### Exporting Data

#### Save to CSV
```python
# Save processed data
df.to_csv('data/processed/SBIN_processed.csv')

# Save without index
df.to_csv('data/processed/SBIN_no_index.csv', index=False)

# Save specific columns
df[['Close', 'Volume']].to_csv('data/processed/SBIN_close_volume.csv')
```

#### Save to Excel
```python
# Save to Excel with formatting
with pd.ExcelWriter('data/reports/portfolio_report.xlsx') as writer:
    portfolio_df.to_excel(writer, sheet_name='Holdings', index=False)
    df.to_excel(writer, sheet_name='Price_Data')
```

## Data Cleaning and Preprocessing

### Handling Missing Data

#### Detecting Missing Values
```python
# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Visualize missing data pattern
print("Missing data percentage:")
print((df.isnull().sum() / len(df)) * 100)
```

#### Dealing with Missing Values
```python
# Forward fill (use last known value)
df_filled = df.fillna(method='ffill')

# Backward fill
df_filled = df.fillna(method='bfill')

# Fill with specific value
df_filled = df.fillna(0)

# Fill with mean for numeric columns
df_numeric = df.select_dtypes(include=[np.number])
df_filled = df.fillna(df_numeric.mean())

# Drop rows with missing values
df_clean = df.dropna()

# Drop columns with too many missing values
df_clean = df.dropna(axis=1, thresh=len(df) * 0.8)  # Keep columns with 80%+ data
```

### Data Validation

#### Remove Outliers
```python
# Remove price outliers using IQR method
def remove_outliers(df, column, factor=1.5):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Remove volume outliers
df_clean = remove_outliers(df, 'Volume')
print(f"Removed {len(df) - len(df_clean)} outlier rows")
```

#### Data Validation
```python
# Validate OHLC relationships
invalid_ohlc = df[
    (df['High'] < df['Low']) |  # High should be >= Low
    (df['High'] < df['Open']) | # High should be >= Open
    (df['High'] < df['Close']) | # High should be >= Close
    (df['Low'] > df['Open']) |  # Low should be <= Open
    (df['Low'] > df['Close'])   # Low should be <= Close
]

if len(invalid_ohlc) > 0:
    print(f"Found {len(invalid_ohlc)} rows with invalid OHLC data")
    print(invalid_ohlc)
```

## Financial Calculations

### Basic Price Calculations

#### Daily Returns
```python
# Simple returns
df['Daily_Return'] = df['Close'].pct_change()

# Log returns (more appropriate for compounding)
df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))

# Price change in rupees
df['Price_Change'] = df['Close'] - df['Close'].shift(1)

print("Returns calculation:")
print(df[['Close', 'Daily_Return', 'Log_Return', 'Price_Change']].head(10))
```

#### Moving Averages
```python
# Simple moving averages
df['SMA_10'] = df['Close'].rolling(window=10).mean()
df['SMA_20'] = df['Close'].rolling(window=20).mean()
df['SMA_50'] = df['Close'].rolling(window=50).mean()

# Exponential moving average
df['EMA_12'] = df['Close'].ewm(span=12).mean()

print("Moving averages:")
print(df[['Close', 'SMA_10', 'SMA_20', 'EMA_12']].tail())
```

#### Volatility Calculations
```python
# Rolling volatility (20-day)
df['Volatility_20'] = df['Daily_Return'].rolling(window=20).std() * np.sqrt(252)

# True Range (for ATR calculation)
df['High_Low'] = df['High'] - df['Low']
df['High_Close'] = abs(df['High'] - df['Close'].shift(1))
df['Low_Close'] = abs(df['Low'] - df['Close'].shift(1))
df['True_Range'] = df[['High_Low', 'High_Close', 'Low_Close']].max(axis=1)

# Average True Range
df['ATR'] = df['True_Range'].rolling(window=14).mean()

print("Volatility metrics:")
print(df[['Volatility_20', 'ATR']].tail())
```

### Portfolio Calculations

#### Portfolio Value Tracking
```python
# Create portfolio tracking DataFrame
portfolio_data = {
    'Symbol': ['SBIN', 'RELIANCE', 'TCS'],
    'Quantity': [100, 50, 25],
    'Purchase_Price': [650.50, 2890.75, 3450.20],
    'Current_Price': [675.25, 2915.60, 3475.90]
}

portfolio = pd.DataFrame(portfolio_data)

# Calculate portfolio metrics
portfolio['Investment'] = portfolio['Quantity'] * portfolio['Purchase_Price']
portfolio['Current_Value'] = portfolio['Quantity'] * portfolio['Current_Price']
portfolio['P&L'] = portfolio['Current_Value'] - portfolio['Investment']
portfolio['P&L_Percent'] = (portfolio['P&L'] / portfolio['Investment']) * 100
portfolio['Weight'] = portfolio['Current_Value'] / portfolio['Current_Value'].sum()

print("Portfolio Analysis:")
print(portfolio)

# Portfolio summary
total_investment = portfolio['Investment'].sum()
total_value = portfolio['Current_Value'].sum()
total_pnl = portfolio['P&L'].sum()
total_return = (total_pnl / total_investment) * 100

print(f"\nPortfolio Summary:")
print(f"Total Investment: ₹{total_investment:,.2f}")
print(f"Current Value: ₹{total_value:,.2f}")
print(f"Total P&L: ₹{total_pnl:,.2f} ({total_return:.2f}%)")
```

## Time Series Operations

### Date and Time Handling

#### Working with DateTime Index
```python
# Ensure proper datetime index
df.index = pd.to_datetime(df.index)

# Extract date components
df['Year'] = df.index.year
df['Month'] = df.index.month
df['Day'] = df.index.day
df['Weekday'] = df.index.day_name()

# Check trading day patterns
print("Trading volume by weekday:")
weekday_volume = df.groupby('Weekday')['Volume'].mean()
print(weekday_volume.sort_values(ascending=False))
```

#### Resampling Data
```python
# Convert daily data to weekly
weekly_df = df.resample('W').agg({
    'Open': 'first',   # First day's opening
    'High': 'max',     # Highest high
    'Low': 'min',      # Lowest low
    'Close': 'last',   # Last day's closing
    'Volume': 'sum'    # Total volume
})

print("Weekly OHLC data:")
print(weekly_df.head())

# Monthly data
monthly_df = df.resample('M').agg({
    'Open': 'first',
    'High': 'max', 
    'Low': 'min',
    'Close': 'last',
    'Volume': 'mean'
})

print("Monthly data:")
print(monthly_df.head())
```

#### Time-based Filtering
```python
# Filter by specific time periods
q1_2025 = df.loc['2025-01':'2025-03']
may_2025 = df.loc['2025-05']

# Last N trading days
last_30_days = df.tail(30)

# Business days only (exclude weekends)
business_days = df[df.index.dayofweek < 5]

print(f"Q1 2025 data points: {len(q1_2025)}")
print(f"May 2025 data points: {len(may_2025)}")
```

## Practical Examples

### Example 1: Stock Screener
```python
def screen_stocks(symbols, criteria):
    """Screen stocks based on technical criteria"""
    results = []
    
    for symbol in symbols:
        # Load data (in practice, you'd load from files/API)
        df = load_stock_data(symbol)
        
        # Calculate indicators
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['RSI'] = calculate_rsi(df['Close'])
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Apply criteria
        meets_criteria = True
        for criterion, value in criteria.items():
            if criterion == 'price_above_sma20' and latest['Close'] <= latest['SMA_20']:
                meets_criteria = False
            elif criterion == 'sma20_above_sma50' and latest['SMA_20'] <= latest['SMA_50']:
                meets_criteria = False
            elif criterion == 'rsi_oversold' and latest['RSI'] >= value:
                meets_criteria = False
        
        if meets_criteria:
            results.append({
                'Symbol': symbol,
                'Price': latest['Close'],
                'SMA_20': latest['SMA_20'],
                'SMA_50': latest['SMA_50'],
                'RSI': latest['RSI']
            })
    
    return pd.DataFrame(results)

# Screen for bullish stocks
criteria = {
    'price_above_sma20': True,
    'sma20_above_sma50': True,
    'rsi_oversold': 70  # RSI below 70
}

screened_stocks = screen_stocks(['SBIN', 'RELIANCE', 'TCS'], criteria)
print("Bullish stock candidates:")
print(screened_stocks)
```

### Example 2: Performance Analysis
```python
def analyze_performance(df):
    """Comprehensive performance analysis"""
    # Calculate returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # Performance metrics
    total_return = (df['Close'].iloc[-1] / df['Close'].iloc[0] - 1) * 100
    daily_returns = df['Daily_Return'].dropna()
    
    metrics = {
        'Total Return (%)': total_return,
        'Annual Return (%)': (1 + total_return/100) ** (252/len(df)) - 1 * 100,
        'Volatility (%)': daily_returns.std() * np.sqrt(252) * 100,
        'Sharpe Ratio': (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)),
        'Max Drawdown (%)': calculate_max_drawdown(df['Close']),
        'Win Rate (%)': (daily_returns > 0).sum() / len(daily_returns) * 100,
        'Best Day (%)': daily_returns.max() * 100,
        'Worst Day (%)': daily_returns.min() * 100
    }
    
    return pd.Series(metrics)

def calculate_max_drawdown(prices):
    """Calculate maximum drawdown"""
    peak = prices.expanding().max()
    drawdown = (prices - peak) / peak
    return drawdown.min() * 100

# Analyze stock performance
performance = analyze_performance(df)
print("Performance Analysis:")
for metric, value in performance.items():
    print(f"{metric}: {value:.2f}")
```

## Best Practices

### Performance Optimization
```python
# Use vectorized operations instead of loops
# ❌ Slow
def calculate_returns_slow(df):
    returns = []
    for i in range(1, len(df)):
        ret = (df['Close'].iloc[i] / df['Close'].iloc[i-1]) - 1
        returns.append(ret)
    return returns

# ✅ Fast
def calculate_returns_fast(df):
    return df['Close'].pct_change()

# Use .loc for conditional selection
# ❌ Avoid chained assignment
df[df['Volume'] > 1000000]['New_Column'] = 'High Volume'

# ✅ Use .loc
df.loc[df['Volume'] > 1000000, 'New_Column'] = 'High Volume'
```

### Memory Management
```python
# Check memory usage
print("Memory usage by column:")
print(df.memory_usage(deep=True))

# Optimize data types
df['Volume'] = df['Volume'].astype('int32')  # Instead of int64
df['Close'] = df['Close'].astype('float32')  # Instead of float64

# Use categories for repeated strings
df['Sector'] = df['Sector'].astype('category')
```

### Error Handling
```python
def safe_calculation(df, column, operation):
    """Safely perform calculations with error handling"""
    try:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if operation == 'sma':
            return df[column].rolling(20).mean()
        elif operation == 'returns':
            return df[column].pct_change()
        else:
            raise ValueError(f"Unknown operation: {operation}")
            
    except Exception as e:
        print(f"Error in calculation: {str(e)}")
        return pd.Series(index=df.index, dtype=float)

# Use the safe function
df['SMA_20'] = safe_calculation(df, 'Close', 'sma')
```

## Summary

### Key Takeaways
1. **DataFrames are powerful**: Perfect for financial time series data
2. **Index matters**: Use datetime index for time series operations
3. **Vectorization is fast**: Avoid loops, use pandas built-in functions
4. **Data quality is crucial**: Always validate and clean your data
5. **Memory efficiency**: Choose appropriate data types

### Next Steps
- Practice with real market data
- Learn advanced pandas features (groupby, pivot tables)
- Combine pandas with visualization libraries
- Explore pandas performance optimization
- Study financial data patterns and anomalies

### Common Pandas Functions for Finance
```python
# Essential functions reference
df.head()                    # View first N rows
df.tail()                    # View last N rows  
df.info()                    # DataFrame information
df.describe()                # Statistical summary
df.pct_change()             # Percentage change
df.rolling(window)          # Rolling window operations
df.shift(periods)           # Shift data by periods
df.resample(rule)           # Time-based grouping
df.fillna(method)           # Handle missing values
df.dropna()                 # Remove missing values
df.groupby(column)          # Group by column values
df.merge(other)             # Join DataFrames
df.pivot_table()            # Create pivot tables
```

Ready to dive deeper into technical analysis? 📊📈

Next: [Technical Indicators Implementation →](02_technical_indicators.md)
