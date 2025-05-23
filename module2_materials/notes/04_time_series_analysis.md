# Module 2.4: Time Series Analysis for Financial Data

## Introduction

Time series analysis is fundamental to financial data analysis as stock prices, trading volumes, and economic indicators are all time-dependent. This module covers working with datetime data, resampling, rolling calculations, and temporal pattern analysis.

## Learning Objectives

By the end of this lesson, you will be able to:
- Handle datetime data effectively in pandas
- Perform time-based indexing and slicing
- Resample data to different time frequencies
- Calculate rolling statistics and moving windows
- Identify seasonal patterns and trends
- Handle missing data in time series

## 1. DateTime Fundamentals

### 1.1 Working with Dates and Times

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Creating datetime objects
date1 = datetime(2023, 1, 15)
date2 = pd.Timestamp('2023-01-15')
date3 = pd.to_datetime('2023-01-15')

print(f"Python datetime: {date1}")
print(f"Pandas Timestamp: {date2}")
print(f"Pandas to_datetime: {date3}")

# Date ranges
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
print(f"Trading days in 2023: {len(dates)}")

# Business days only (excludes weekends)
business_days = pd.date_range(start='2023-01-01', end='2023-12-31', freq='B')
print(f"Business days in 2023: {len(business_days)}")
```

### 1.2 DateTime Parsing and Formatting

```python
# Parse various date formats
date_strings = [
    '2023-01-15',
    '01/15/2023',
    '15-Jan-2023',
    '2023-01-15 09:30:00'
]

parsed_dates = [pd.to_datetime(date) for date in date_strings]
for i, date in enumerate(parsed_dates):
    print(f"Original: {date_strings[i]} -> Parsed: {date}")

# Custom date parsing
custom_format = pd.to_datetime('15012023', format='%d%m%Y')
print(f"Custom format: {custom_format}")

# Extract components
sample_date = pd.Timestamp('2023-03-15 14:30:00')
print(f"Year: {sample_date.year}")
print(f"Month: {sample_date.month}")
print(f"Day: {sample_date.day}")
print(f"Hour: {sample_date.hour}")
print(f"Day of week: {sample_date.dayofweek}")  # Monday=0, Sunday=6
print(f"Day name: {sample_date.day_name()}")
```

## 2. Time Series Data Structures

### 2.1 Creating Time Series DataFrames

```python
# Generate sample stock data with proper datetime index
dates = pd.date_range('2023-01-01', periods=252, freq='B')  # Business days
np.random.seed(42)

# Simulate stock price data
initial_price = 100
returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
prices = [initial_price]

for ret in returns[1:]:
    prices.append(prices[-1] * (1 + ret))

stock_data = pd.DataFrame({
    'Open': prices,
    'High': [p * (1 + np.random.uniform(0, 0.02)) for p in prices],
    'Low': [p * (1 - np.random.uniform(0, 0.02)) for p in prices],
    'Close': [p * (1 + np.random.uniform(-0.01, 0.01)) for p in prices],
    'Volume': np.random.randint(1000000, 10000000, len(dates))
}, index=dates)

print(stock_data.head())
print(f"\nData shape: {stock_data.shape}")
print(f"Date range: {stock_data.index.min()} to {stock_data.index.max()}")
```

### 2.2 Time-Based Indexing and Slicing

```python
# Various ways to select time periods
print("=== Time-Based Selection ===")

# Select specific date
jan_15 = stock_data.loc['2023-01-15']
print(f"Data for Jan 15: {jan_15['Close']:.2f}")

# Select date range
q1_data = stock_data.loc['2023-01-01':'2023-03-31']
print(f"Q1 data shape: {q1_data.shape}")

# Select by month
march_data = stock_data.loc['2023-03']
print(f"March data shape: {march_data.shape}")

# Select last N days
last_30_days = stock_data.tail(30)
print(f"Last 30 days average price: {last_30_days['Close'].mean():.2f}")

# Boolean indexing with dates
recent_data = stock_data[stock_data.index > '2023-06-01']
print(f"Data after June 1: {recent_data.shape}")
```

## 3. Resampling and Frequency Conversion

### 3.1 Downsampling (Higher to Lower Frequency)

```python
# Convert daily data to weekly
weekly_data = stock_data.resample('W').agg({
    'Open': 'first',    # First trading day of week
    'High': 'max',      # Highest price of week
    'Low': 'min',       # Lowest price of week
    'Close': 'last',    # Last trading day of week
    'Volume': 'sum'     # Total volume for week
})

print("Weekly OHLCV Data:")
print(weekly_data.head())

# Convert to monthly data
monthly_data = stock_data.resample('M').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

print("\nMonthly OHLCV Data:")
print(monthly_data.head())

# Custom resampling with multiple statistics
custom_resample = stock_data['Close'].resample('M').agg([
    'first', 'last', 'min', 'max', 'mean', 'std'
])
print("\nMonthly Close Price Statistics:")
print(custom_resample)
```

### 3.2 Upsampling (Lower to Higher Frequency)

```python
# Create hourly data from daily (for demonstration)
# Note: This is mainly for illustration - real intraday data would come from APIs

# Forward fill method
hourly_ffill = monthly_data.resample('D').ffill()
print(f"Upsampled daily data (forward fill): {hourly_ffill.shape}")

# Interpolation method
hourly_interp = monthly_data.resample('D').interpolate()
print(f"Upsampled daily data (interpolation): {hourly_interp.shape}")

# Compare original vs upsampled
print(f"Original monthly data points: {len(monthly_data)}")
print(f"Upsampled to daily: {len(hourly_ffill)}")
```

## 4. Rolling Windows and Moving Statistics

### 4.1 Simple Moving Averages

```python
# Calculate various moving averages
stock_data['SMA_5'] = stock_data['Close'].rolling(window=5).mean()
stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()

# Plot price with moving averages
plt.figure(figsize=(12, 6))
plt.plot(stock_data.index, stock_data['Close'], label='Close Price', linewidth=1)
plt.plot(stock_data.index, stock_data['SMA_5'], label='SMA 5', linewidth=2)
plt.plot(stock_data.index, stock_data['SMA_20'], label='SMA 20', linewidth=2)
plt.plot(stock_data.index, stock_data['SMA_50'], label='SMA 50', linewidth=2)
plt.title('Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Moving averages calculated:")
print(stock_data[['Close', 'SMA_5', 'SMA_20', 'SMA_50']].tail())
```

### 4.2 Exponential Moving Average (EMA)

```python
# Calculate exponential moving averages
stock_data['EMA_12'] = stock_data['Close'].ewm(span=12).mean()
stock_data['EMA_26'] = stock_data['Close'].ewm(span=26).mean()

# MACD calculation
stock_data['MACD'] = stock_data['EMA_12'] - stock_data['EMA_26']
stock_data['MACD_Signal'] = stock_data['MACD'].ewm(span=9).mean()
stock_data['MACD_Histogram'] = stock_data['MACD'] - stock_data['MACD_Signal']

print("MACD Indicators:")
print(stock_data[['Close', 'EMA_12', 'EMA_26', 'MACD', 'MACD_Signal']].tail())
```

### 4.3 Rolling Statistics

```python
# Calculate rolling statistics
window = 20

stock_data['Rolling_Mean'] = stock_data['Close'].rolling(window=window).mean()
stock_data['Rolling_Std'] = stock_data['Close'].rolling(window=window).std()
stock_data['Rolling_Min'] = stock_data['Close'].rolling(window=window).min()
stock_data['Rolling_Max'] = stock_data['Close'].rolling(window=window).max()

# Bollinger Bands
stock_data['BB_Upper'] = stock_data['Rolling_Mean'] + (stock_data['Rolling_Std'] * 2)
stock_data['BB_Lower'] = stock_data['Rolling_Mean'] - (stock_data['Rolling_Std'] * 2)

# Rolling correlations (example with volume)
stock_data['Price_Volume_Corr'] = stock_data['Close'].rolling(window=30).corr(
    stock_data['Volume']
)

print("Rolling Statistics:")
print(stock_data[['Close', 'Rolling_Mean', 'Rolling_Std', 'BB_Upper', 'BB_Lower']].tail())
```

## 5. Returns and Performance Analysis

### 5.1 Calculate Returns

```python
# Different types of returns
stock_data['Daily_Return'] = stock_data['Close'].pct_change()
stock_data['Log_Return'] = np.log(stock_data['Close'] / stock_data['Close'].shift(1))

# Multi-period returns
stock_data['Weekly_Return'] = stock_data['Close'].pct_change(periods=5)  # 5 trading days
stock_data['Monthly_Return'] = stock_data['Close'].pct_change(periods=21)  # ~21 trading days

# Cumulative returns
stock_data['Cumulative_Return'] = (1 + stock_data['Daily_Return']).cumprod() - 1

print("Returns Analysis:")
print(stock_data[['Close', 'Daily_Return', 'Log_Return', 'Cumulative_Return']].tail())

# Returns statistics
returns_stats = {
    'Mean Daily Return': stock_data['Daily_Return'].mean(),
    'Daily Volatility': stock_data['Daily_Return'].std(),
    'Annualized Return': stock_data['Daily_Return'].mean() * 252,
    'Annualized Volatility': stock_data['Daily_Return'].std() * np.sqrt(252),
    'Sharpe Ratio': (stock_data['Daily_Return'].mean() * 252) / (stock_data['Daily_Return'].std() * np.sqrt(252)),
    'Total Return': stock_data['Cumulative_Return'].iloc[-1]
}

print("\nPerformance Metrics:")
for metric, value in returns_stats.items():
    print(f"{metric}: {value:.4f}")
```

### 5.2 Risk Metrics

```python
# Value at Risk (VaR) calculation
confidence_level = 0.05
VaR_95 = np.percentile(stock_data['Daily_Return'].dropna(), confidence_level * 100)
VaR_99 = np.percentile(stock_data['Daily_Return'].dropna(), 1)

# Maximum Drawdown
def calculate_max_drawdown(returns):
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()

max_drawdown = calculate_max_drawdown(stock_data['Daily_Return'].dropna())

print(f"Value at Risk (95%): {VaR_95:.4f}")
print(f"Value at Risk (99%): {VaR_99:.4f}")
print(f"Maximum Drawdown: {max_drawdown:.4f}")
```

## 6. Seasonal Analysis

### 6.1 Day-of-Week Effects

```python
# Add time components
stock_data['Year'] = stock_data.index.year
stock_data['Month'] = stock_data.index.month
stock_data['DayOfWeek'] = stock_data.index.dayofweek
stock_data['DayName'] = stock_data.index.day_name()

# Analyze day-of-week returns
dow_returns = stock_data.groupby('DayOfWeek')['Daily_Return'].agg(['mean', 'std', 'count'])
dow_returns.index = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

print("Day-of-Week Return Analysis:")
print(dow_returns)

# Plot day-of-week effects
plt.figure(figsize=(10, 6))
plt.bar(dow_returns.index, dow_returns['mean'])
plt.title('Average Daily Returns by Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Return')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()
```

### 6.2 Monthly Seasonality

```python
# Monthly return analysis
monthly_returns = stock_data.groupby('Month')['Daily_Return'].agg(['mean', 'std', 'count'])
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_returns.index = month_names

print("Monthly Return Analysis:")
print(monthly_returns)

# Plot monthly seasonality
plt.figure(figsize=(12, 6))
plt.bar(monthly_returns.index, monthly_returns['mean'])
plt.title('Average Daily Returns by Month')
plt.xlabel('Month')
plt.ylabel('Average Return')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.show()
```

## 7. Handling Missing Data

### 7.1 Identifying Missing Data

```python
# Introduce some missing data for demonstration
sample_data = stock_data[['Close', 'Volume']].copy()
sample_data.loc['2023-03-15':'2023-03-17', 'Close'] = np.nan
sample_data.loc['2023-06-20', 'Volume'] = np.nan

print("Missing data summary:")
print(sample_data.isnull().sum())
print(f"\nMissing data percentage:")
print((sample_data.isnull().sum() / len(sample_data)) * 100)
```

### 7.2 Handling Missing Data

```python
# Different strategies for handling missing data

# Forward fill (carry forward last known value)
sample_data['Close_ffill'] = sample_data['Close'].fillna(method='ffill')

# Backward fill
sample_data['Close_bfill'] = sample_data['Close'].fillna(method='bfill')

# Linear interpolation
sample_data['Close_interp'] = sample_data['Close'].interpolate()

# Fill with mean
sample_data['Volume_mean'] = sample_data['Volume'].fillna(sample_data['Volume'].mean())

# Drop rows with missing data
clean_data = sample_data.dropna()

print("Original vs. cleaned data shapes:")
print(f"Original: {sample_data.shape}")
print(f"After dropna: {clean_data.shape}")

# Show filled values around missing data
print("\nAround missing dates:")
print(sample_data.loc['2023-03-13':'2023-03-20', ['Close', 'Close_ffill', 'Close_interp']])
```

## 8. Time Zone Handling

### 8.1 Working with Time Zones

```python
# Create timezone-aware data
utc_dates = pd.date_range('2023-01-01', periods=10, freq='H', tz='UTC')
ny_dates = utc_dates.tz_convert('America/New_York')
london_dates = utc_dates.tz_convert('Europe/London')

timezone_data = pd.DataFrame({
    'UTC': utc_dates,
    'New_York': ny_dates,
    'London': london_dates
})

print("Timezone comparison:")
print(timezone_data.head())

# Market hours analysis
def is_market_hours(timestamp, tz='America/New_York'):
    """Check if timestamp is during market hours (9:30 AM - 4:00 PM ET)"""
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize('UTC').tz_convert(tz)
    elif timestamp.tz != tz:
        timestamp = timestamp.tz_convert(tz)
    
    return (timestamp.hour >= 9 and timestamp.minute >= 30) and timestamp.hour < 16

# Apply to sample data
sample_timestamps = pd.date_range('2023-01-01 08:00:00', 
                                 periods=24, freq='H', tz='America/New_York')
market_hours = [is_market_hours(ts) for ts in sample_timestamps]

market_analysis = pd.DataFrame({
    'Timestamp': sample_timestamps,
    'Is_Market_Hours': market_hours
})

print("\nMarket hours analysis:")
print(market_analysis)
```

## 9. Advanced Time Series Operations

### 9.1 Lag and Lead Operations

```python
# Create lagged variables
stock_data['Close_Lag1'] = stock_data['Close'].shift(1)  # Previous day
stock_data['Close_Lag5'] = stock_data['Close'].shift(5)  # 5 days ago
stock_data['Close_Lead1'] = stock_data['Close'].shift(-1)  # Next day

# Calculate price changes
stock_data['Price_Change_1D'] = stock_data['Close'] - stock_data['Close_Lag1']
stock_data['Price_Change_5D'] = stock_data['Close'] - stock_data['Close_Lag5']

print("Lag and lead analysis:")
print(stock_data[['Close', 'Close_Lag1', 'Close_Lag5', 'Price_Change_1D']].tail())
```

### 9.2 Window Functions

```python
# Rolling window functions
rolling_window = 20

# Rank within window (percentile rank)
stock_data['Rolling_Rank'] = stock_data['Close'].rolling(window=rolling_window).rank(pct=True)

# Rolling quantiles
stock_data['Rolling_Q25'] = stock_data['Close'].rolling(window=rolling_window).quantile(0.25)
stock_data['Rolling_Q75'] = stock_data['Close'].rolling(window=rolling_window).quantile(0.75)

# Custom rolling function
def rolling_sharpe(returns, window=20):
    return returns.rolling(window=window).mean() / returns.rolling(window=window).std()

stock_data['Rolling_Sharpe'] = rolling_sharpe(stock_data['Daily_Return']) * np.sqrt(252)

print("Rolling window analysis:")
print(stock_data[['Close', 'Rolling_Rank', 'Rolling_Q25', 'Rolling_Q75', 'Rolling_Sharpe']].tail())
```

## Practice Exercises

1. **Date Range Analysis**: Create a function that calculates returns for any given date range
2. **Seasonal Patterns**: Analyze if there are consistent patterns around earnings seasons
3. **Rolling Correlations**: Calculate rolling correlations between two stocks
4. **Volatility Clustering**: Identify periods of high and low volatility
5. **Missing Data Strategy**: Implement a robust missing data handling strategy for market data

## Key Takeaways

- Proper datetime handling is crucial for financial analysis
- Resampling allows analysis at different time frequencies
- Rolling statistics capture temporal patterns and trends
- Time zone awareness is important for global markets
- Missing data handling must preserve temporal integrity
- Returns analysis requires careful consideration of compounding effects

## Next Steps

In the next module, we'll learn about strategy development and how to combine time series analysis with trading signals.
