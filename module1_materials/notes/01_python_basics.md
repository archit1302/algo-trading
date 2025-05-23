# Note 1: Python Basics & Syntax

## Introduction to Python for Financial Data Processing

Python is one of the most popular programming languages for financial analysis and algorithmic trading. This note covers the fundamental concepts you need to start working with financial data.

## Why Python for Finance?

- **Simple Syntax**: Easy to learn and read
- **Rich Libraries**: Pandas, NumPy, Matplotlib for financial analysis
- **Community Support**: Large community of traders and analysts
- **Industry Standard**: Used by major financial institutions

## Variables and Data Types

### Variables
Variables store data values. In Python, you don't need to declare variable types.

```python
# Stock information
stock_symbol = "SBIN"          # String
current_price = 725.50         # Float
shares_owned = 100             # Integer
is_profitable = True           # Boolean

print(f"Stock: {stock_symbol}")
print(f"Price: ₹{current_price}")
print(f"Shares: {shares_owned}")
print(f"Profitable: {is_profitable}")
```

### Common Data Types in Finance

1. **Strings (str)**: Stock symbols, company names
2. **Integers (int)**: Share quantities, number of trades
3. **Floats (float)**: Prices, percentages, ratios
4. **Booleans (bool)**: Buy/sell signals, profitable trades

```python
# Financial data examples
ticker = "RELIANCE"           # Company symbol
quantity = 50                 # Number of shares
buy_price = 2456.75          # Purchase price
sell_price = 2500.00         # Selling price
profit = sell_price - buy_price  # Calculate profit
is_long_position = True      # Position type
```

## Basic Operations

### Arithmetic Operations
```python
# Price calculations
buy_price = 100.00
sell_price = 110.00

# Calculate returns
absolute_return = sell_price - buy_price
percentage_return = (sell_price - buy_price) / buy_price * 100

print(f"Absolute Return: ₹{absolute_return}")
print(f"Percentage Return: {percentage_return:.2f}%")

# Position sizing
capital = 100000              # Total capital
risk_per_trade = 0.02        # 2% risk per trade
stop_loss = 0.05             # 5% stop loss

position_size = (capital * risk_per_trade) / stop_loss
print(f"Position Size: ₹{position_size}")
```

### Comparison Operations
```python
# Price comparisons
current_price = 725.50
support_level = 720.00
resistance_level = 730.00

# Check price levels
above_support = current_price > support_level
below_resistance = current_price < resistance_level
at_resistance = current_price >= resistance_level

print(f"Above Support: {above_support}")
print(f"Below Resistance: {below_resistance}")
print(f"At/Above Resistance: {at_resistance}")
```

## Input and Output

### Getting User Input
```python
# Get stock information from user
stock_symbol = input("Enter stock symbol: ")
shares = int(input("Enter number of shares: "))
price = float(input("Enter current price: "))

# Calculate portfolio value
portfolio_value = shares * price
print(f"Portfolio value for {stock_symbol}: ₹{portfolio_value}")
```

### Formatted Output
```python
# Professional formatting
stock = "SBIN"
price = 725.50
change = 12.25
change_percent = 1.72

# Using f-strings (recommended)
print(f"{stock}: ₹{price:.2f} (+₹{change:.2f}, +{change_percent:.2f}%)")

# Using format method
print("{}: ₹{:.2f} (+₹{:.2f}, +{:.2f}%)".format(stock, price, change, change_percent))

# Output: SBIN: ₹725.50 (+₹12.25, +1.72%)
```

## Comments and Documentation

### Single Line Comments
```python
# This is a single line comment
current_price = 725.50  # Current stock price in INR
```

### Multi-line Comments
```python
"""
This is a multi-line comment used for documentation.
It explains the purpose of the code block below.

This function calculates the position size based on risk management rules.
"""

def calculate_position_size(capital, risk_percent, stop_loss_percent):
    return (capital * risk_percent) / stop_loss_percent
```

## Best Practices for Financial Programming

1. **Use Descriptive Names**: `current_price` instead of `cp`
2. **Add Comments**: Explain financial logic and calculations
3. **Format Numbers**: Use appropriate decimal places for currency
4. **Validate Input**: Check for negative prices or quantities
5. **Use Constants**: Define fixed values like commission rates

```python
# Good example
COMMISSION_RATE = 0.001  # 0.1% commission
BROKERAGE_FEE = 20       # Fixed brokerage fee in INR

def calculate_net_profit(buy_price, sell_price, quantity):
    """
    Calculate net profit after deducting all charges
    """
    gross_profit = (sell_price - buy_price) * quantity
    commission = gross_profit * COMMISSION_RATE
    total_brokerage = BROKERAGE_FEE * 2  # Buy + Sell
    
    net_profit = gross_profit - commission - total_brokerage
    return net_profit
```

## Common Errors and Solutions

### 1. Type Errors
```python
# Error: mixing strings and numbers
# price = "725.50" + 10  # This will cause an error

# Solution: convert types
price = float("725.50") + 10
print(price)  # Output: 735.5
```

### 2. Division by Zero
```python
# Error: division by zero in percentage calculations
# percentage = profit / 0

# Solution: check for zero before division
cost_price = 0
profit = 100

if cost_price != 0:
    percentage = (profit / cost_price) * 100
    print(f"Return: {percentage}%")
else:
    print("Cannot calculate percentage: cost price is zero")
```

## Practice Exercises

1. **Basic Calculations**: Calculate portfolio value for multiple stocks
2. **Price Analysis**: Determine if a stock is above/below moving average
3. **Risk Management**: Calculate position size based on account size
4. **Return Analysis**: Calculate absolute and percentage returns

## Summary

In this note, you learned:
- How to create and use variables for financial data
- Basic arithmetic and comparison operations
- Input/output operations for interactive programs
- Best practices for financial programming
- Common errors and how to avoid them

## Next Steps

In the next note, we'll explore Python data structures (lists, dictionaries) and how to use them for organizing financial data like stock portfolios, price histories, and trading records.

---

**Key Takeaways:**
- Variables store financial data (prices, quantities, symbols)
- Use appropriate data types for different kinds of financial information
- Always format currency values properly
- Add comments to explain financial logic
- Validate input data to avoid errors
