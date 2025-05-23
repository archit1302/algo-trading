# Assignment 4: Functions for Financial Calculations

## Objective
Create reusable functions for common financial calculations and trading analysis to build a personal finance toolkit.

## Prerequisites
- Complete Assignments 1, 2 & 3
- Complete reading: `04_functions.md`

## Tasks

### Task 1: Basic Financial Functions (25 points)
Create the following utility functions:

1. **Simple Interest Calculator**
```python
def calculate_simple_interest(principal, rate, time):
    """
    Calculate simple interest
    Args:
        principal (float): Initial investment amount
        rate (float): Annual interest rate (as percentage)
        time (float): Time period in years
    Returns:
        tuple: (interest_amount, total_amount)
    """
```

2. **Compound Interest Calculator**
```python
def calculate_compound_interest(principal, rate, time, frequency=1):
    """
    Calculate compound interest
    Args:
        principal (float): Initial investment amount
        rate (float): Annual interest rate (as percentage)
        time (float): Time period in years
        frequency (int): Compounding frequency per year
    Returns:
        tuple: (interest_amount, total_amount)
    """
```

3. **SIP Returns Calculator**
```python
def calculate_sip_returns(monthly_investment, annual_return, years):
    """
    Calculate SIP (Systematic Investment Plan) returns
    Args:
        monthly_investment (float): Monthly investment amount
        annual_return (float): Expected annual return (as percentage)
        years (int): Investment period in years
    Returns:
        dict: {total_invested, final_value, total_returns, return_percentage}
    """
```

### Task 2: Stock Analysis Functions (25 points)
Create functions for technical analysis:

1. **Moving Average Calculator**
```python
def calculate_moving_average(prices, period):
    """
    Calculate simple moving average
    Args:
        prices (list): List of stock prices
        period (int): Moving average period
    Returns:
        list: Moving averages
    """
```

2. **RSI Calculator**
```python
def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index
    Args:
        prices (list): List of closing prices
        period (int): RSI period (default 14)
    Returns:
        list: RSI values
    """
```

3. **Support and Resistance Finder**
```python
def find_support_resistance(prices, window=5):
    """
    Find support and resistance levels
    Args:
        prices (list): List of stock prices
        window (int): Window size for peak/trough detection
    Returns:
        dict: {support_levels, resistance_levels}
    """
```

### Task 3: Portfolio Management Functions (25 points)
Create portfolio analysis functions:

1. **Portfolio Diversification Score**
```python
def calculate_diversification_score(portfolio):
    """
    Calculate portfolio diversification score
    Args:
        portfolio (dict): {sector: allocation_percentage}
    Returns:
        tuple: (diversification_score, recommendations)
    """
```

2. **Portfolio Risk Calculator**
```python
def calculate_portfolio_risk(stocks_data):
    """
    Calculate portfolio risk metrics
    Args:
        stocks_data (list): List of dicts with stock info
    Returns:
        dict: {portfolio_beta, sharpe_ratio, max_drawdown}
    """
```

3. **Rebalancing Calculator**
```python
def calculate_rebalancing(current_portfolio, target_allocation):
    """
    Calculate required trades for portfolio rebalancing
    Args:
        current_portfolio (dict): Current holdings
        target_allocation (dict): Target allocation percentages
    Returns:
        dict: Required buy/sell actions
    """
```

### Task 4: Advanced Trading Functions (25 points)
Create sophisticated trading tools:

1. **Position Sizing Calculator**
```python
def calculate_position_size(account_balance, risk_percentage, entry_price, stop_loss):
    """
    Calculate optimal position size based on risk management
    Args:
        account_balance (float): Total account balance
        risk_percentage (float): Maximum risk per trade (as percentage)
        entry_price (float): Entry price for the trade
        stop_loss (float): Stop loss price
    Returns:
        dict: {position_size, risk_amount, profit_target}
    """
```

2. **Options Strategy Analyzer**
```python
def analyze_covered_call(stock_price, strike_price, premium, expiry_days):
    """
    Analyze covered call strategy
    Args:
        stock_price (float): Current stock price
        strike_price (float): Call option strike price
        premium (float): Premium received
        expiry_days (int): Days to expiration
    Returns:
        dict: {max_profit, break_even, annualized_return}
    """
```

3. **Backtesting Function**
```python
def backtest_strategy(prices, buy_signals, sell_signals, initial_capital=100000):
    """
    Backtest a trading strategy
    Args:
        prices (list): Historical prices
        buy_signals (list): Boolean list for buy signals
        sell_signals (list): Boolean list for sell signals
        initial_capital (float): Starting capital
    Returns:
        dict: {final_value, total_return, win_rate, max_drawdown}
    """
```

## Implementation Example
Test your functions with this sample data:

```python
# Test data
sample_prices = [675.25, 678.90, 682.30, 679.50, 685.75, 688.20, 692.40, 689.80, 695.30, 698.50]

# Test function calls
sip_result = calculate_sip_returns(5000, 12, 10)
ma_5 = calculate_moving_average(sample_prices, 5)
position = calculate_position_size(100000, 2, 685.75, 665.00)

print(f"SIP Returns: {sip_result}")
print(f"5-day MA: {ma_5}")
print(f"Position Size: {position}")
```

## Expected Output Format
```
=== FINANCIAL CALCULATIONS ===
Simple Interest (₹50,000 @ 8% for 3 years):
Interest: ₹12,000.00
Total Amount: ₹62,000.00

Compound Interest (₹50,000 @ 8% for 3 years, quarterly):
Interest: ₹13,449.09
Total Amount: ₹63,449.09

SIP Returns (₹5,000/month @ 12% for 10 years):
Total Invested: ₹6,00,000
Final Value: ₹11,61,695
Total Returns: ₹5,61,695
Return Percentage: 93.6%

=== TECHNICAL ANALYSIS ===
Moving Average (5-day): [680.04, 681.21, 683.04, 685.13, 688.84]
RSI: [52.3, 58.7, 61.2, 59.8, 64.1]

Support Levels: [675.25, 679.50]
Resistance Levels: [692.40, 698.50]

=== PORTFOLIO ANALYSIS ===
Diversification Score: 7.2/10
Risk Level: Moderate
Recommended Actions: Increase allocation in defensive sectors

=== TRADING TOOLS ===
Position Size Analysis:
Account Balance: ₹1,00,000
Risk Amount: ₹2,000 (2%)
Recommended Shares: 96
Stop Loss: ₹665.00
Profit Target: ₹726.25 (6% gain)

Backtesting Results:
Total Return: 15.7%
Win Rate: 68%
Max Drawdown: -8.2%
Sharpe Ratio: 1.34
```

## Submission Guidelines
1. Create a file named `assignment4_solution.py`
2. Include proper docstrings for all functions
3. Add input validation and error handling
4. Create a main section to test all functions
5. Use meaningful variable names and comments

## Evaluation Criteria
- Function correctness and logic (40%)
- Proper use of parameters and return values (25%)
- Documentation and docstrings (15%)
- Error handling and edge cases (10%)
- Code organization and testing (10%)

## Bonus Challenges (20 extra points)
1. Create a function decorator for performance timing
2. Implement memoization for expensive calculations
3. Add support for different currencies and tax calculations
4. Create a complete trading bot using all your functions
