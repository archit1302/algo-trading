# Note 2: Data Structures for Financial Data

## Introduction

Data structures are ways to organize and store data efficiently. In financial programming, we work with various types of data like stock prices, portfolios, trading signals, and historical data. Understanding the right data structure for each use case is crucial.

## Lists - Sequential Financial Data

Lists are ordered collections that can store multiple items. They're perfect for storing sequential data like price histories, trading signals, or stock symbols.

### Creating Lists

```python
# Stock symbols
nifty_50_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR"]

# Daily closing prices for SBIN
sbin_prices = [720.50, 725.30, 718.75, 722.10, 728.45]

# Trading signals (1 = Buy, 0 = Hold, -1 = Sell)
trading_signals = [1, 1, 0, -1, 0, 1, -1]

# Mixed data types (generally avoid this)
stock_info = ["SBIN", 725.50, 100, True]  # symbol, price, quantity, profitable
```

### Accessing List Elements

```python
stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "HINDUNILVR"]
prices = [2456.75, 3678.90, 1542.30, 1789.45, 2234.80]

# Access by index (starts from 0)
first_stock = stocks[0]        # "RELIANCE"
first_price = prices[0]        # 2456.75

# Access last element
last_stock = stocks[-1]        # "HINDUNILVR"
last_price = prices[-1]        # 2234.80

# Access range (slicing)
top_3_stocks = stocks[0:3]     # ["RELIANCE", "TCS", "HDFCBANK"]
last_2_prices = prices[-2:]    # [1789.45, 2234.80]
```

### List Operations for Financial Data

```python
# Portfolio of stocks
portfolio = ["RELIANCE", "TCS", "SBIN"]

# Add new stock to portfolio
portfolio.append("INFY")
print(portfolio)  # ["RELIANCE", "TCS", "SBIN", "INFY"]

# Remove stock from portfolio
portfolio.remove("SBIN")
print(portfolio)  # ["RELIANCE", "TCS", "INFY"]

# Check if stock is in portfolio
if "TCS" in portfolio:
    print("TCS is in the portfolio")

# Get portfolio size
portfolio_size = len(portfolio)
print(f"Portfolio contains {portfolio_size} stocks")

# Sort stocks alphabetically
sorted_portfolio = sorted(portfolio)
print(sorted_portfolio)  # ["INFY", "RELIANCE", "TCS"]
```

### Price Analysis with Lists

```python
# SBIN daily closing prices for a week
sbin_week_prices = [720.50, 725.30, 718.75, 722.10, 728.45]

# Calculate statistics
highest_price = max(sbin_week_prices)
lowest_price = min(sbin_week_prices)
average_price = sum(sbin_week_prices) / len(sbin_week_prices)

print(f"Highest Price: ₹{highest_price}")
print(f"Lowest Price: ₹{lowest_price}")
print(f"Average Price: ₹{average_price:.2f}")

# Calculate daily returns
daily_returns = []
for i in range(1, len(sbin_week_prices)):
    return_pct = ((sbin_week_prices[i] - sbin_week_prices[i-1]) / sbin_week_prices[i-1]) * 100
    daily_returns.append(return_pct)

print(f"Daily Returns: {[f'{r:.2f}%' for r in daily_returns]}")
```

## Dictionaries - Structured Financial Data

Dictionaries store data in key-value pairs. They're perfect for storing structured financial information like stock details, portfolio holdings, or trading parameters.

### Creating Dictionaries

```python
# Single stock information
sbin_info = {
    "symbol": "SBIN",
    "name": "State Bank of India",
    "sector": "Banking",
    "current_price": 725.50,
    "market_cap": 647000,  # in crores
    "pe_ratio": 12.5,
    "dividend_yield": 2.8
}

# Portfolio holdings
portfolio = {
    "RELIANCE": {"quantity": 50, "avg_price": 2450.00},
    "TCS": {"quantity": 25, "avg_price": 3650.00},
    "SBIN": {"quantity": 100, "avg_price": 720.00}
}

# Trading parameters
trading_config = {
    "risk_per_trade": 0.02,    # 2%
    "max_portfolio_risk": 0.06, # 6%
    "stop_loss": 0.05,         # 5%
    "target": 0.10,            # 10%
    "commission": 0.001        # 0.1%
}
```

### Accessing Dictionary Values

```python
stock = {
    "symbol": "SBIN",
    "current_price": 725.50,
    "quantity": 100,
    "avg_price": 720.00
}

# Access values by key
symbol = stock["symbol"]              # "SBIN"
current_price = stock["current_price"] # 725.50

# Safe access with get() method
pe_ratio = stock.get("pe_ratio", "N/A")  # Returns "N/A" if key doesn't exist

# Check if key exists
if "dividend_yield" in stock:
    print(f"Dividend Yield: {stock['dividend_yield']}%")
else:
    print("Dividend yield data not available")
```

### Dictionary Operations

```python
# Portfolio management
portfolio = {
    "RELIANCE": {"quantity": 50, "avg_price": 2450.00},
    "TCS": {"quantity": 25, "avg_price": 3650.00}
}

# Add new stock
portfolio["SBIN"] = {"quantity": 100, "avg_price": 720.00}

# Update existing stock
portfolio["RELIANCE"]["quantity"] += 25  # Bought 25 more shares

# Remove stock
del portfolio["TCS"]

# Get all stock symbols
stock_symbols = list(portfolio.keys())
print(f"Portfolio stocks: {stock_symbols}")

# Iterate through portfolio
for symbol, details in portfolio.items():
    quantity = details["quantity"]
    avg_price = details["avg_price"]
    value = quantity * avg_price
    print(f"{symbol}: {quantity} shares @ ₹{avg_price} = ₹{value}")
```

### Portfolio Analysis with Dictionaries

```python
# Complete portfolio with current prices
portfolio = {
    "RELIANCE": {
        "quantity": 50,
        "avg_price": 2450.00,
        "current_price": 2456.75
    },
    "TCS": {
        "quantity": 25,
        "avg_price": 3650.00,
        "current_price": 3678.90
    },
    "SBIN": {
        "quantity": 100,
        "avg_price": 720.00,
        "current_price": 725.50
    }
}

# Calculate portfolio statistics
total_invested = 0
current_value = 0
total_pnl = 0

for symbol, details in portfolio.items():
    quantity = details["quantity"]
    avg_price = details["avg_price"]
    current_price = details["current_price"]
    
    invested = quantity * avg_price
    current = quantity * current_price
    pnl = current - invested
    pnl_percent = (pnl / invested) * 100
    
    total_invested += invested
    current_value += current
    total_pnl += pnl
    
    print(f"{symbol}:")
    print(f"  Invested: ₹{invested:,.2f}")
    print(f"  Current: ₹{current:,.2f}")
    print(f"  P&L: ₹{pnl:,.2f} ({pnl_percent:+.2f}%)")
    print()

total_pnl_percent = (total_pnl / total_invested) * 100
print(f"Portfolio Summary:")
print(f"Total Invested: ₹{total_invested:,.2f}")
print(f"Current Value: ₹{current_value:,.2f}")
print(f"Total P&L: ₹{total_pnl:,.2f} ({total_pnl_percent:+.2f}%)")
```

## Tuples - Immutable Financial Data

Tuples are ordered collections that cannot be changed after creation. They're useful for storing fixed data like coordinates, OHLC data points, or configuration settings.

### Creating and Using Tuples

```python
# OHLC data point (Open, High, Low, Close)
ohlc_today = (720.00, 728.45, 718.50, 725.50)

# Stock with sector information
stock_sector = ("SBIN", "Banking")

# Trading levels (support, resistance)
key_levels = (715.00, 730.00)

# Access tuple elements
open_price = ohlc_today[0]    # 720.00
high_price = ohlc_today[1]    # 728.45
low_price = ohlc_today[2]     # 718.50
close_price = ohlc_today[3]   # 725.50

# Unpack tuple values
support, resistance = key_levels
print(f"Support: ₹{support}, Resistance: ₹{resistance}")
```

### Multiple OHLC Data

```python
# Week's OHLC data for SBIN
week_ohlc = [
    (720.00, 728.45, 718.50, 725.50),  # Monday
    (725.50, 730.20, 722.80, 728.90),  # Tuesday
    (728.90, 732.15, 726.40, 729.75),  # Wednesday
    (729.75, 734.20, 728.10, 731.60),  # Thursday
    (731.60, 735.80, 730.45, 733.25)   # Friday
]

# Calculate weekly statistics
weekly_high = max(day[1] for day in week_ohlc)  # Highest of all highs
weekly_low = min(day[2] for day in week_ohlc)   # Lowest of all lows
weekly_open = week_ohlc[0][0]                   # Monday's open
weekly_close = week_ohlc[-1][3]                 # Friday's close

print(f"Weekly Range: ₹{weekly_low} - ₹{weekly_high}")
print(f"Weekly Return: {((weekly_close - weekly_open) / weekly_open * 100):+.2f}%")
```

## Nested Data Structures

Combining different data structures for complex financial data.

```python
# Market data with multiple stocks
market_data = {
    "SBIN": {
        "info": {
            "name": "State Bank of India",
            "sector": "Banking",
            "market_cap": 647000
        },
        "ohlc": (720.00, 728.45, 718.50, 725.50),
        "volume": 12548963,
        "signals": ["RSI_Oversold", "Support_Hold"]
    },
    "RELIANCE": {
        "info": {
            "name": "Reliance Industries",
            "sector": "Oil & Gas",
            "market_cap": 1654000
        },
        "ohlc": (2450.00, 2465.80, 2445.20, 2456.75),
        "volume": 8965247,
        "signals": ["Bullish_Crossover", "Volume_High"]
    }
}

# Access nested data
sbin_high = market_data["SBIN"]["ohlc"][1]
reliance_sector = market_data["RELIANCE"]["info"]["sector"]
sbin_signals = market_data["SBIN"]["signals"]

print(f"SBIN High: ₹{sbin_high}")
print(f"Reliance Sector: {reliance_sector}")
print(f"SBIN Signals: {sbin_signals}")
```

## Best Practices

1. **Choose the Right Structure**:
   - Lists: Sequential data (prices, signals)
   - Dictionaries: Structured data (stock info, portfolio)
   - Tuples: Fixed data (OHLC, coordinates)

2. **Use Descriptive Keys**: `"current_price"` instead of `"cp"`

3. **Consistent Data Types**: Keep similar data types in lists

4. **Handle Missing Data**: Use `.get()` for dictionaries

```python
# Good example: Well-structured portfolio data
portfolio = {
    "stocks": {
        "SBIN": {
            "quantity": 100,
            "avg_price": 720.00,
            "current_price": 725.50,
            "sector": "Banking"
        }
    },
    "cash": 50000.00,
    "total_value": 122550.00,
    "last_updated": "2025-05-24"
}

# Access with error handling
def get_stock_value(portfolio, symbol):
    stock = portfolio["stocks"].get(symbol)
    if stock:
        return stock["quantity"] * stock["current_price"]
    else:
        return 0

sbin_value = get_stock_value(portfolio, "SBIN")
print(f"SBIN holding value: ₹{sbin_value}")
```

## Common Use Cases in Trading

1. **Watchlist Management**: Lists of stock symbols
2. **Portfolio Tracking**: Dictionary of holdings
3. **Price History**: Lists of historical prices
4. **Trading Signals**: Lists of buy/sell indicators
5. **Risk Parameters**: Dictionary of risk management settings

## Summary

In this note, you learned:
- How to use lists for sequential financial data
- Dictionary usage for structured stock and portfolio information
- Tuples for fixed financial data points
- Nested structures for complex market data
- Best practices for organizing financial data

## Next Steps

In the next note, we'll explore control flow (if-else statements and loops) to implement trading logic and automate financial calculations.

---

**Key Takeaways:**
- Lists are perfect for price histories and stock symbols
- Dictionaries excel at storing structured financial information
- Tuples are ideal for fixed data like OHLC points
- Nested structures handle complex market data
- Always use descriptive names and handle missing data gracefully
