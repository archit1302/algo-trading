# Note 4: Functions for Financial Calculations

## Introduction

Functions are reusable blocks of code that perform specific tasks. In financial programming, functions help you create modular, testable, and maintainable code for calculations like returns, risk metrics, position sizing, and technical indicators.

## Creating Basic Functions

### Function Syntax

```python
def function_name(parameters):
    """
    Docstring explaining what the function does
    """
    # Function body
    return result
```

### Simple Financial Functions

```python
def calculate_return(buy_price, sell_price):
    """
    Calculate percentage return on investment
    
    Parameters:
    buy_price (float): Purchase price
    sell_price (float): Selling price
    
    Returns:
    float: Percentage return
    """
    return ((sell_price - buy_price) / buy_price) * 100

# Usage
profit_percent = calculate_return(720.00, 725.50)
print(f"Return: {profit_percent:.2f}%")  # Output: Return: 0.76%
```

### Functions with Multiple Parameters

```python
def calculate_position_size(account_balance, risk_percent, stop_loss_percent):
    """
    Calculate position size based on risk management rules
    
    Parameters:
    account_balance (float): Total account value
    risk_percent (float): Risk per trade as decimal (0.02 = 2%)
    stop_loss_percent (float): Stop loss as decimal (0.05 = 5%)
    
    Returns:
    float: Maximum position size in currency
    """
    risk_amount = account_balance * risk_percent
    position_size = risk_amount / stop_loss_percent
    return position_size

# Usage
position = calculate_position_size(100000, 0.02, 0.05)
print(f"Maximum position size: ₹{position:,.2f}")  # Output: ₹40,000.00
```

## Default Parameters

Functions can have default values for parameters, making them more flexible.

```python
def calculate_sma(prices, period=20):
    """
    Calculate Simple Moving Average
    
    Parameters:
    prices (list): List of prices
    period (int): Number of periods (default: 20)
    
    Returns:
    float: Simple moving average
    """
    if len(prices) < period:
        return None
    
    recent_prices = prices[-period:]
    return sum(recent_prices) / period

# Usage with default period
sbin_prices = [720, 722, 718, 725, 730, 728, 732, 729, 735, 731]
sma_20 = calculate_sma(sbin_prices)  # Uses default period of 20
sma_5 = calculate_sma(sbin_prices, 5)  # Uses custom period of 5

print(f"SMA-20: ₹{sma_20 or 'Not enough data'}")
print(f"SMA-5: ₹{sma_5:.2f}")
```

## Return Values

Functions can return different types of values or multiple values.

### Single Return Value

```python
def calculate_portfolio_value(holdings):
    """
    Calculate total portfolio value
    
    Parameters:
    holdings (dict): Dictionary of stock holdings
    
    Returns:
    float: Total portfolio value
    """
    total_value = 0
    
    for stock, details in holdings.items():
        quantity = details["quantity"]
        current_price = details["current_price"]
        total_value += quantity * current_price
    
    return total_value
```

### Multiple Return Values

```python
def analyze_portfolio(holdings):
    """
    Analyze portfolio and return multiple metrics
    
    Parameters:
    holdings (dict): Portfolio holdings
    
    Returns:
    tuple: (total_value, total_pnl, best_performer, worst_performer)
    """
    total_invested = 0
    total_current = 0
    stock_performance = {}
    
    for stock, details in holdings.items():
        quantity = details["quantity"]
        avg_price = details["avg_price"]
        current_price = details["current_price"]
        
        invested = quantity * avg_price
        current_value = quantity * current_price
        pnl_percent = ((current_value - invested) / invested) * 100
        
        total_invested += invested
        total_current += current_value
        stock_performance[stock] = pnl_percent
    
    total_pnl = total_current - total_invested
    best_performer = max(stock_performance, key=stock_performance.get)
    worst_performer = min(stock_performance, key=stock_performance.get)
    
    return total_current, total_pnl, best_performer, worst_performer

# Usage
portfolio = {
    "SBIN": {"quantity": 100, "avg_price": 720.00, "current_price": 725.50},
    "RELIANCE": {"quantity": 50, "avg_price": 2450.00, "current_price": 2456.75},
    "TCS": {"quantity": 25, "avg_price": 3650.00, "current_price": 3678.90}
}

value, pnl, best, worst = analyze_portfolio(portfolio)
print(f"Portfolio Value: ₹{value:,.2f}")
print(f"Total P&L: ₹{pnl:,.2f}")
print(f"Best Performer: {best}")
print(f"Worst Performer: {worst}")
```

### Returning Dictionary for Complex Data

```python
def get_stock_metrics(prices):
    """
    Calculate various metrics for a stock
    
    Parameters:
    prices (list): List of historical prices
    
    Returns:
    dict: Dictionary containing various metrics
    """
    if len(prices) < 2:
        return {"error": "Insufficient data"}
    
    # Calculate metrics
    current_price = prices[-1]
    previous_price = prices[-2]
    highest_price = max(prices)
    lowest_price = min(prices)
    
    daily_return = ((current_price - previous_price) / previous_price) * 100
    total_return = ((current_price - prices[0]) / prices[0]) * 100
    
    # Calculate volatility (simplified)
    returns = []
    for i in range(1, len(prices)):
        ret = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
        returns.append(ret)
    
    avg_return = sum(returns) / len(returns)
    variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
    volatility = variance ** 0.5
    
    return {
        "current_price": current_price,
        "daily_return": daily_return,
        "total_return": total_return,
        "high": highest_price,
        "low": lowest_price,
        "volatility": volatility,
        "trend": "Bullish" if daily_return > 0 else "Bearish"
    }

# Usage
sbin_prices = [720, 722, 718, 725, 730, 728, 732, 729, 735, 731]
metrics = get_stock_metrics(sbin_prices)

if "error" not in metrics:
    print("SBIN Metrics:")
    print(f"Current Price: ₹{metrics['current_price']}")
    print(f"Daily Return: {metrics['daily_return']:+.2f}%")
    print(f"Total Return: {metrics['total_return']:+.2f}%")
    print(f"Volatility: {metrics['volatility']:.2f}%")
    print(f"Trend: {metrics['trend']}")
```

## Advanced Financial Functions

### Risk Management Functions

```python
def calculate_kelly_criterion(win_rate, avg_win, avg_loss):
    """
    Calculate optimal position size using Kelly Criterion
    
    Parameters:
    win_rate (float): Probability of winning (0-1)
    avg_win (float): Average winning amount
    avg_loss (float): Average losing amount
    
    Returns:
    float: Optimal fraction of capital to risk
    """
    if avg_loss == 0:
        return 0
    
    win_loss_ratio = avg_win / avg_loss
    kelly_fraction = win_rate - ((1 - win_rate) / win_loss_ratio)
    
    # Cap at 25% for safety
    return min(max(kelly_fraction, 0), 0.25)

def position_size_with_kelly(account_balance, kelly_fraction, current_price):
    """
    Calculate position size using Kelly criterion
    """
    capital_to_risk = account_balance * kelly_fraction
    shares = int(capital_to_risk / current_price)
    return shares

# Usage
optimal_fraction = calculate_kelly_criterion(0.6, 150, 100)  # 60% win rate
shares = position_size_with_kelly(100000, optimal_fraction, 725.50)
print(f"Kelly Fraction: {optimal_fraction:.3f}")
print(f"Recommended Position: {shares} shares")
```

### Technical Indicator Functions

```python
def calculate_rsi(prices, period=14):
    """
    Calculate Relative Strength Index (RSI)
    
    Parameters:
    prices (list): List of closing prices
    period (int): RSI period (default: 14)
    
    Returns:
    float: RSI value (0-100)
    """
    if len(prices) < period + 1:
        return None
    
    # Calculate price changes
    changes = []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        changes.append(change)
    
    # Separate gains and losses
    gains = [change if change > 0 else 0 for change in changes]
    losses = [-change if change < 0 else 0 for change in changes]
    
    # Calculate average gains and losses
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def get_rsi_signal(rsi):
    """
    Generate trading signal based on RSI
    """
    if rsi is None:
        return "NO_DATA"
    elif rsi <= 30:
        return "OVERSOLD"
    elif rsi >= 70:
        return "OVERBOUGHT"
    else:
        return "NEUTRAL"

# Usage
prices = [720, 722, 718, 725, 730, 728, 732, 729, 735, 731, 740, 738, 742, 745, 743]
rsi = calculate_rsi(prices)
signal = get_rsi_signal(rsi)

print(f"RSI: {rsi:.2f}")
print(f"Signal: {signal}")
```

### Portfolio Management Functions

```python
def rebalance_portfolio(current_holdings, target_allocation, total_value):
    """
    Calculate trades needed to rebalance portfolio
    
    Parameters:
    current_holdings (dict): Current stock holdings
    target_allocation (dict): Target allocation percentages
    total_value (float): Total portfolio value
    
    Returns:
    dict: Required trades for each stock
    """
    trades = {}
    
    for stock, target_percent in target_allocation.items():
        target_value = total_value * target_percent
        current_price = current_holdings[stock]["current_price"]
        current_quantity = current_holdings[stock]["quantity"]
        current_value = current_quantity * current_price
        
        value_difference = target_value - current_value
        shares_to_trade = int(value_difference / current_price)
        
        if shares_to_trade != 0:
            action = "BUY" if shares_to_trade > 0 else "SELL"
            trades[stock] = {
                "action": action,
                "quantity": abs(shares_to_trade),
                "value": abs(value_difference)
            }
    
    return trades

# Usage
current_portfolio = {
    "SBIN": {"quantity": 100, "current_price": 725.50},
    "RELIANCE": {"quantity": 30, "current_price": 2456.75},
    "TCS": {"quantity": 20, "current_price": 3678.90}
}

target_weights = {
    "SBIN": 0.4,      # 40%
    "RELIANCE": 0.35,  # 35%
    "TCS": 0.25       # 25%
}

portfolio_value = calculate_portfolio_value(current_portfolio)
rebalance_trades = rebalance_portfolio(current_portfolio, target_weights, portfolio_value)

print("Rebalancing Required:")
for stock, trade in rebalance_trades.items():
    print(f"{stock}: {trade['action']} {trade['quantity']} shares (₹{trade['value']:,.2f})")
```

## Function Documentation and Best Practices

### Comprehensive Documentation

```python
def backtest_strategy(prices, entry_signal_func, exit_signal_func, initial_capital=100000):
    """
    Backtest a trading strategy on historical data
    
    Parameters:
    -----------
    prices : list of float
        Historical price data in chronological order
    entry_signal_func : function
        Function that returns True for entry signals
    exit_signal_func : function
        Function that returns True for exit signals
    initial_capital : float, optional
        Starting capital amount (default: 100,000)
    
    Returns:
    --------
    dict
        Backtest results containing:
        - total_return: Overall return percentage
        - trades: Number of completed trades
        - win_rate: Percentage of winning trades
        - max_drawdown: Maximum portfolio decline
        - final_value: Final portfolio value
    
    Example:
    --------
    >>> def simple_entry(price, sma): return price > sma
    >>> def simple_exit(price, sma): return price < sma
    >>> results = backtest_strategy(prices, simple_entry, simple_exit)
    >>> print(f"Total Return: {results['total_return']:.2f}%")
    """
    # Implementation would go here
    pass
```

### Error Handling in Functions

```python
def safe_calculate_return(buy_price, sell_price):
    """
    Calculate return with comprehensive error handling
    """
    try:
        # Input validation
        if not isinstance(buy_price, (int, float)) or not isinstance(sell_price, (int, float)):
            raise TypeError("Prices must be numeric")
        
        if buy_price <= 0 or sell_price <= 0:
            raise ValueError("Prices must be positive")
        
        return_pct = ((sell_price - buy_price) / buy_price) * 100
        return return_pct
    
    except TypeError as e:
        print(f"Type Error: {e}")
        return None
    except ValueError as e:
        print(f"Value Error: {e}")
        return None
    except ZeroDivisionError:
        print("Error: Buy price cannot be zero")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

# Usage with error handling
returns = [
    safe_calculate_return(720, 725.50),    # Valid
    safe_calculate_return("720", 725.50),  # Type error
    safe_calculate_return(-720, 725.50),   # Value error
    safe_calculate_return(0, 725.50)       # Zero division
]

for i, ret in enumerate(returns):
    if ret is not None:
        print(f"Trade {i+1}: {ret:.2f}% return")
    else:
        print(f"Trade {i+1}: Invalid calculation")
```

## Lambda Functions for Quick Calculations

Lambda functions are short, anonymous functions useful for simple calculations.

```python
# Quick calculators using lambda
calculate_commission = lambda value, rate=0.001: value * rate
calculate_tax = lambda profit, rate=0.15: max(0, profit * rate) if profit > 0 else 0
is_profitable = lambda buy, sell: sell > buy

# Usage
trade_value = 72550
commission = calculate_commission(trade_value)
profit = 550
tax = calculate_tax(profit)
profitable = is_profitable(720, 725.50)

print(f"Commission: ₹{commission:.2f}")
print(f"Tax: ₹{tax:.2f}")
print(f"Profitable: {profitable}")

# Using lambda with lists
prices = [720, 722, 718, 725, 730]
above_average = list(filter(lambda x: x > sum(prices)/len(prices), prices))
print(f"Prices above average: {above_average}")
```

## Organizing Functions in Modules

Create a separate file for your trading functions:

```python
# trading_utils.py (separate file)
def calculate_all_metrics(prices):
    """
    Calculate comprehensive metrics for a stock
    """
    return {
        "sma_20": calculate_sma(prices, 20),
        "rsi": calculate_rsi(prices),
        "volatility": calculate_volatility(prices),
        "trend": determine_trend(prices)
    }

def portfolio_summary(holdings):
    """
    Generate complete portfolio summary
    """
    return {
        "total_value": calculate_portfolio_value(holdings),
        "pnl": calculate_total_pnl(holdings),
        "allocation": calculate_allocation(holdings),
        "risk_metrics": calculate_risk_metrics(holdings)
    }

# In your main script:
# from trading_utils import calculate_all_metrics, portfolio_summary
```

## Best Practices

1. **Single Responsibility**: Each function should do one thing well
2. **Clear Names**: Use descriptive function names
3. **Documentation**: Always include docstrings
4. **Error Handling**: Validate inputs and handle errors gracefully
5. **Return Consistency**: Always return the same data type
6. **Testing**: Test functions with various inputs

```python
# Good example: Well-designed function
def calculate_sharpe_ratio(returns, risk_free_rate=0.06):
    """
    Calculate Sharpe ratio for a series of returns
    
    Parameters:
    returns (list): List of periodic returns (as decimals)
    risk_free_rate (float): Annual risk-free rate (default: 6%)
    
    Returns:
    float: Sharpe ratio, or None if insufficient data
    """
    if not returns or len(returns) < 2:
        return None
    
    try:
        # Convert annual risk-free rate to period rate
        periods_per_year = 252  # Trading days
        period_risk_free = risk_free_rate / periods_per_year
        
        # Calculate excess returns
        excess_returns = [r - period_risk_free for r in returns]
        
        # Calculate mean and standard deviation
        mean_excess = sum(excess_returns) / len(excess_returns)
        
        variance = sum((r - mean_excess) ** 2 for r in excess_returns) / (len(excess_returns) - 1)
        std_dev = variance ** 0.5
        
        if std_dev == 0:
            return 0
        
        # Annualize the Sharpe ratio
        sharpe_ratio = (mean_excess / std_dev) * (periods_per_year ** 0.5)
        return sharpe_ratio
    
    except Exception as e:
        print(f"Error calculating Sharpe ratio: {e}")
        return None

# Example usage
daily_returns = [0.01, -0.005, 0.02, 0.008, -0.012, 0.015]
sharpe = calculate_sharpe_ratio(daily_returns)
print(f"Sharpe Ratio: {sharpe:.3f}" if sharpe else "Unable to calculate Sharpe ratio")
```

## Summary

In this note, you learned:
- How to create and use functions for financial calculations
- Working with parameters, default values, and return values
- Building complex financial functions for risk management and analysis
- Error handling and validation in functions
- Best practices for function design and documentation

## Next Steps

In the next note, we'll explore file handling - how to read and write financial data from CSV files, which is essential for working with historical stock data.

---

**Key Takeaways:**
- Functions make code reusable and organized
- Always validate inputs and handle errors
- Use clear names and comprehensive documentation
- Return consistent data types
- Test functions thoroughly with different scenarios
