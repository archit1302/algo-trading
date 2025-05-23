# Assignment 3: Control Flow for Trading Logic

## Objective
Implement trading decision logic using conditional statements and loops to automate investment strategies.

## Prerequisites
- Complete Assignments 1 & 2
- Complete reading: `03_control_flow.md`

## Tasks

### Task 1: Stock Screening with Conditionals (30 points)
Create a stock screening system that categorizes stocks based on various criteria:

```python
# Sample stock data
stocks = [
    {"symbol": "SBIN", "price": 675.25, "pe_ratio": 12.5, "market_cap": 550000},
    {"symbol": "RELIANCE", "price": 2915.60, "pe_ratio": 18.7, "market_cap": 1975000},
    {"symbol": "TCS", "price": 3475.90, "pe_ratio": 25.3, "market_cap": 1265000},
    {"symbol": "INFY", "price": 1901.75, "pe_ratio": 22.1, "market_cap": 789000},
    {"symbol": "HDFC", "price": 1678.45, "pe_ratio": 15.8, "market_cap": 912000}
]
```

Implement screening logic:
1. **Value Stocks**: PE ratio < 15 and Market Cap > 500000 Cr
2. **Growth Stocks**: PE ratio > 20 and Price > 2000
3. **Large Cap**: Market Cap > 1000000 Cr
4. **Mid Cap**: Market Cap between 250000 and 1000000 Cr
5. **Small Cap**: Market Cap < 250000 Cr

For each stock, determine and print its category.

### Task 2: Trading Signal Generator (25 points)
Create a simple moving average crossover signal system:

```python
# 5-day price data for SBIN
sbin_prices = [665.20, 670.50, 675.25, 678.90, 682.30]
```

Implement logic to:
1. Calculate 3-day simple moving average
2. Generate trading signals:
   - **BUY**: Current price > 3-day SMA and previous day was below SMA
   - **SELL**: Current price < 3-day SMA and previous day was above SMA
   - **HOLD**: No crossover detected

3. Print daily analysis with signals

### Task 3: Portfolio Risk Assessment (20 points)
Implement risk categorization based on portfolio composition:

```python
portfolio_allocation = {
    "equity": 70,      # percentage
    "debt": 20,
    "gold": 5,
    "cash": 5
}
```

Risk categories:
- **Conservative**: Equity < 40%
- **Moderate**: Equity 40-70%
- **Aggressive**: Equity > 70%

Additional checks:
- Warn if cash > 15% (too much idle money)
- Warn if any single asset > 80% (concentration risk)
- Suggest rebalancing if needed

### Task 4: Batch Processing with Loops (25 points)
Process multiple stocks and generate a comprehensive report:

```python
stock_data = [
    {"symbol": "SBIN", "prices": [665.20, 670.50, 675.25, 678.90, 682.30]},
    {"symbol": "RELIANCE", "prices": [2890.75, 2895.60, 2915.60, 2920.40, 2935.80]},
    {"symbol": "TCS", "prices": [3450.20, 3465.90, 3475.90, 3480.25, 3495.60]}
]
```

For each stock, calculate:
1. Daily returns (percentage change)
2. Volatility (standard deviation of returns)
3. Total return over the period
4. Risk-adjusted return (return/volatility)

Use nested loops to process all data and rank stocks.

## Expected Output Format

```
=== STOCK SCREENING RESULTS ===
SBIN: Value Stock, Large Cap
RELIANCE: Large Cap
TCS: Growth Stock, Large Cap
INFY: Mid Cap
HDFC: Mid Cap

=== TRADING SIGNALS - SBIN ===
Day 1: Price=665.20, SMA=N/A, Signal=HOLD
Day 2: Price=670.50, SMA=N/A, Signal=HOLD
Day 3: Price=675.25, SMA=670.32, Signal=BUY
Day 4: Price=678.90, SMA=674.88, Signal=HOLD
Day 5: Price=682.30, SMA=678.82, Signal=HOLD

=== PORTFOLIO RISK ASSESSMENT ===
Risk Category: AGGRESSIVE
Warnings:
- No concentration risk detected
- Cash allocation optimal
Recommendation: Consider reducing equity allocation for better risk management

=== BATCH STOCK ANALYSIS ===
SBIN:
  Daily Returns: [0.80%, 0.71%, 0.54%, 0.50%]
  Volatility: 0.13%
  Total Return: 2.57%
  Risk-Adjusted Return: 19.77

RELIANCE:
  Daily Returns: [0.17%, 0.69%, 0.16%, 0.52%]
  Volatility: 0.25%
  Total Return: 1.56%
  Risk-Adjusted Return: 6.24

Stock Rankings (by Risk-Adjusted Return):
1. SBIN (19.77)
2. TCS (12.45)
3. RELIANCE (6.24)
```

## Submission Guidelines
1. Create a file named `assignment3_solution.py`
2. Use appropriate conditional statements (if-elif-else)
3. Implement efficient loops (for, while)
4. Include error handling for edge cases
5. Add meaningful comments explaining the logic

## Evaluation Criteria
- Correct conditional logic (35%)
- Proper loop implementation (30%)
- Accurate calculations (25%)
- Code structure and readability (10%)

## Bonus Challenges (15 extra points)
1. Implement a stop-loss system that triggers alerts
2. Create a momentum indicator using nested conditions
3. Add support for multiple timeframes in signal generation
4. Implement portfolio optimization using loops and conditions
