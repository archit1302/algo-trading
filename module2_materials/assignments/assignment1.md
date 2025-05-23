# Module 2 Assignment 1: Pandas Fundamentals for Financial Data

## Objective
Master pandas fundamentals by working with real financial data structures, including data manipulation, cleaning, and basic analysis.

## Instructions
Complete the following tasks using pandas. Create a Python script that demonstrates each concept with financial data examples.

## Prerequisites
- Complete Module 2.1 notes (Pandas Fundamentals)
- Install required libraries: pandas, numpy, matplotlib

## Tasks

### Task 1: Data Loading and Exploration (25 points)

Create a sample dataset representing stock price data for multiple companies:

**Requirements:**
1. Create a DataFrame with the following columns:
   - Date (datetime index)
   - Symbol (stock ticker)
   - Open, High, Low, Close prices
   - Volume
   - Market Cap

2. Include data for at least 3 different stocks over 6 months (approximately 126 trading days)

3. Demonstrate the following pandas operations:
   - `df.head()`, `df.tail()`, `df.info()`, `df.describe()`
   - Check data types and memory usage
   - Display unique symbols and date range

**Expected Output:**
```python
# Sample output structure
print("Dataset Overview:")
print(f"Shape: {df.shape}")
print(f"Symbols: {df['Symbol'].unique()}")
print(f"Date Range: {df.index.min()} to {df.index.max()}")
```

### Task 2: Data Indexing and Selection (25 points)

**Requirements:**
1. Set the Date column as the index
2. Demonstrate various selection methods:
   - Select data for a specific stock
   - Select data for a date range
   - Select specific columns for analysis
   - Use boolean indexing to find high-volume days

3. Create multi-level indexing with Date and Symbol
4. Demonstrate `.loc[]`, `.iloc[]`, and boolean indexing

**Examples to implement:**
```python
# Select Apple stock data
apple_data = df[df['Symbol'] == 'AAPL']

# Select last month's data
recent_data = df.loc['2023-11-01':'2023-11-30']

# Find days with volume > 10 million
high_volume_days = df[df['Volume'] > 10000000]
```

### Task 3: Data Aggregation and Grouping (25 points)

**Requirements:**
1. Group data by Symbol and calculate:
   - Average daily return for each stock
   - Total trading volume
   - Price volatility (standard deviation of returns)
   - Maximum and minimum prices

2. Create monthly aggregations:
   - Monthly average prices
   - Monthly total volume
   - Monthly high and low prices

3. Calculate cross-stock statistics:
   - Average market cap by month
   - Correlation matrix between stock returns

**Expected calculations:**
```python
# Daily returns by stock
daily_returns = df.groupby('Symbol')['Close'].pct_change()

# Monthly aggregation
monthly_stats = df.groupby(['Symbol', pd.Grouper(freq='M')]).agg({
    'Close': ['mean', 'max', 'min'],
    'Volume': 'sum',
    'Market_Cap': 'mean'
})
```

### Task 4: Time Series Operations (25 points)

**Requirements:**
1. Calculate technical indicators using pandas:
   - Simple Moving Average (20-day and 50-day)
   - Daily returns and cumulative returns
   - Rolling volatility (20-day window)

2. Demonstrate time series functionality:
   - Resampling to weekly and monthly data
   - Shift operations for lagged analysis
   - Date-based filtering and slicing

3. Handle missing data:
   - Introduce some missing values intentionally
   - Demonstrate forward fill, backward fill, and interpolation methods
   - Calculate data completeness statistics

**Technical indicators to implement:**
```python
# Moving averages
df['SMA_20'] = df.groupby('Symbol')['Close'].rolling(20).mean()
df['SMA_50'] = df.groupby('Symbol')['Close'].rolling(50).mean()

# Returns analysis
df['Daily_Return'] = df.groupby('Symbol')['Close'].pct_change()
df['Cumulative_Return'] = df.groupby('Symbol')['Daily_Return'].cumsum()

# Volatility
df['Rolling_Vol'] = df.groupby('Symbol')['Daily_Return'].rolling(20).std()
```

## Bonus Tasks (Additional 10 points each)

### Bonus 1: Portfolio Analysis
Create a portfolio consisting of equal weights of all stocks and calculate:
- Portfolio daily returns
- Portfolio cumulative performance
- Portfolio volatility
- Comparison with individual stock performance

### Bonus 2: Data Export and Import
- Save your DataFrame to CSV, Excel, and JSON formats
- Read the data back and verify integrity
- Export specific subsets (e.g., only AAPL data, only 2023 data)

### Bonus 3: Performance Optimization
- Compare performance of different pandas operations
- Use `.apply()`, `.map()`, and vectorized operations
- Measure execution time for large datasets

## Deliverables

Submit a Python script (`assignment1_solution.py`) that includes:

1. **Data Creation Section**: Code to generate the sample dataset
2. **Analysis Section**: Implementation of all required tasks
3. **Results Section**: Print statements showing key findings
4. **Documentation**: Comments explaining each step

## Sample Output Format

Your script should produce output similar to:

```
=== PANDAS FUNDAMENTALS ASSIGNMENT ===

1. Data Overview:
   Shape: (378, 6)
   Symbols: ['AAPL', 'GOOGL', 'MSFT']
   Date Range: 2023-06-01 to 2023-11-30

2. Stock Statistics:
   AAPL - Avg Return: 0.15%, Volatility: 1.2%
   GOOGL - Avg Return: 0.08%, Volatility: 1.5%
   MSFT - Avg Return: 0.12%, Volatility: 1.1%

3. Monthly Performance:
   [Monthly aggregation results]

4. Technical Analysis:
   [Moving averages and indicators]

5. Data Quality:
   Missing values: 5 (1.3%)
   Completeness: 98.7%
```

## Evaluation Criteria

- **Correctness (40%)**: All calculations and operations work correctly
- **Code Quality (30%)**: Clean, readable, well-commented code
- **Completeness (20%)**: All required tasks completed
- **Insights (10%)**: Quality of analysis and interpretation

## Tips for Success

1. **Start with small datasets** to test your logic before scaling up
2. **Use meaningful variable names** for financial data
3. **Add comments** explaining financial concepts
4. **Handle edge cases** like weekends, holidays, missing data
5. **Verify calculations** with manual spot checks

## Common Financial Data Patterns

When creating sample data, consider realistic patterns:
- Stock prices generally trend upward over time
- Volume varies significantly day-to-day
- Market cap correlates with stock price
- Returns should be normally distributed around small positive mean
- Volatility clusters (high volatility periods)

## Resources

- Pandas documentation: https://pandas.pydata.org/docs/
- Financial data patterns: Study real stock data for realistic simulation
- Module 2.1 notes for detailed explanations and examples

Good luck with your assignment!
