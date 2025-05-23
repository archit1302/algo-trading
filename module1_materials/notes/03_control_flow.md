# Note 3: Control Flow for Trading Logic

## Introduction

Control flow structures allow your program to make decisions and repeat operations automatically. In trading, these are essential for implementing buy/sell logic, risk management rules, and automated analysis of multiple stocks or time periods.

## Conditional Statements (if-else)

Conditional statements help you implement trading logic based on market conditions.

### Basic if Statement

```python
# Simple trading signal
current_price = 725.50
moving_average = 720.00

if current_price > moving_average:
    print("Price above MA - Bullish signal")
```

### if-else Statement

```python
# Buy/Sell decision
current_price = 725.50
support_level = 720.00
resistance_level = 730.00

if current_price <= support_level:
    signal = "BUY"
    print(f"Price at support (₹{current_price}) - {signal} signal")
else:
    signal = "WAIT"
    print(f"Price above support (₹{current_price}) - {signal}")
```

### if-elif-else Statement

```python
# Multi-level trading strategy
def get_trading_signal(price, sma_20, sma_50):
    """
    Generate trading signals based on moving averages
    """
    if price > sma_20 > sma_50:
        return "STRONG BUY"
    elif price > sma_20 and sma_20 < sma_50:
        return "WEAK BUY"
    elif price < sma_20 > sma_50:
        return "WEAK SELL"
    elif price < sma_20 < sma_50:
        return "STRONG SELL"
    else:
        return "HOLD"

# Example usage
current_price = 725.50
sma_20 = 722.30
sma_50 = 718.75

signal = get_trading_signal(current_price, sma_20, sma_50)
print(f"Trading Signal: {signal}")
```

### Complex Trading Conditions

```python
# Risk management with multiple conditions
def should_enter_trade(price, volume, rsi, account_balance, position_size):
    """
    Determine if trade entry conditions are met
    """
    # Define thresholds
    MIN_VOLUME = 1000000        # Minimum daily volume
    RSI_OVERSOLD = 30          # RSI oversold level
    RSI_OVERBOUGHT = 70        # RSI overbought level
    MAX_POSITION_PCT = 0.10    # Maximum 10% of portfolio per trade
    
    # Calculate position value
    position_value = position_size * price
    max_position_value = account_balance * MAX_POSITION_PCT
    
    # Entry conditions
    if (volume >= MIN_VOLUME and 
        RSI_OVERSOLD <= rsi <= RSI_OVERBOUGHT and 
        position_value <= max_position_value and
        account_balance > position_value):
        
        return True, "All conditions met - ENTER TRADE"
    
    # Check specific failures
    reasons = []
    if volume < MIN_VOLUME:
        reasons.append(f"Low volume: {volume:,} < {MIN_VOLUME:,}")
    if rsi <= RSI_OVERSOLD:
        reasons.append(f"RSI oversold: {rsi}")
    if rsi >= RSI_OVERBOUGHT:
        reasons.append(f"RSI overbought: {rsi}")
    if position_value > max_position_value:
        reasons.append(f"Position too large: ₹{position_value:,.2f} > ₹{max_position_value:,.2f}")
    if account_balance <= position_value:
        reasons.append("Insufficient funds")
    
    return False, "; ".join(reasons)

# Example usage
can_trade, reason = should_enter_trade(
    price=725.50,
    volume=1500000,
    rsi=45,
    account_balance=100000,
    position_size=100
)

print(f"Can Trade: {can_trade}")
print(f"Reason: {reason}")
```

## Loops for Automation

Loops allow you to automate repetitive tasks like analyzing multiple stocks or processing historical data.

### for Loop with Lists

```python
# Analyze multiple stocks
nifty_stocks = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "SBIN"]
current_prices = [2456.75, 3678.90, 1542.30, 1789.45, 725.50]

print("Stock Analysis:")
print("-" * 40)

for i in range(len(nifty_stocks)):
    stock = nifty_stocks[i]
    price = current_prices[i]
    
    # Simple momentum check (price > 1000 = strong stock)
    if price > 2000:
        momentum = "STRONG"
    elif price > 1000:
        momentum = "MODERATE"
    else:
        momentum = "WEAK"
    
    print(f"{stock:10} | ₹{price:8.2f} | {momentum}")
```

### for Loop with Dictionary

```python
# Portfolio analysis
portfolio = {
    "RELIANCE": {"quantity": 50, "avg_price": 2450.00, "current_price": 2456.75},
    "TCS": {"quantity": 25, "avg_price": 3650.00, "current_price": 3678.90},
    "SBIN": {"quantity": 100, "avg_price": 720.00, "current_price": 725.50},
    "INFY": {"quantity": 75, "avg_price": 1780.00, "current_price": 1789.45}
}

print("Portfolio Performance:")
print("-" * 60)
print(f"{'Stock':<10} {'Qty':<5} {'Avg Price':<10} {'Current':<10} {'P&L':<10} {'%':<8}")
print("-" * 60)

total_invested = 0
total_current = 0

for stock, details in portfolio.items():
    qty = details["quantity"]
    avg_price = details["avg_price"]
    current_price = details["current_price"]
    
    invested = qty * avg_price
    current_value = qty * current_price
    pnl = current_value - invested
    pnl_percent = (pnl / invested) * 100
    
    total_invested += invested
    total_current += current_value
    
    print(f"{stock:<10} {qty:<5} ₹{avg_price:<9.2f} ₹{current_price:<9.2f} ₹{pnl:<9.2f} {pnl_percent:<7.2f}%")

total_pnl = total_current - total_invested
total_pnl_percent = (total_pnl / total_invested) * 100

print("-" * 60)
print(f"Total: ₹{total_invested:,.2f} → ₹{total_current:,.2f} | P&L: ₹{total_pnl:,.2f} ({total_pnl_percent:+.2f}%)")
```

### while Loop for Monitoring

```python
# Price monitoring system
def monitor_price_breakout(symbol, target_price, current_price, max_checks=10):
    """
    Monitor price until it breaks target level
    """
    check_count = 0
    
    print(f"Monitoring {symbol} for breakout above ₹{target_price}")
    
    while current_price < target_price and check_count < max_checks:
        check_count += 1
        print(f"Check #{check_count}: {symbol} = ₹{current_price:.2f} (Target: ₹{target_price})")
        
        # Simulate price movement (in real scenario, you'd fetch live price)
        import random
        price_change = random.uniform(-5, 10)  # Random price movement
        current_price += price_change
        
        # Simulate delay
        import time
        time.sleep(1)
    
    if current_price >= target_price:
        print(f"🎯 BREAKOUT! {symbol} reached ₹{current_price:.2f}")
        return True
    else:
        print(f"⏰ Monitoring stopped after {max_checks} checks")
        return False

# Example usage (commented out to avoid actual delays)
# breakout_achieved = monitor_price_breakout("SBIN", 730.00, 725.50, max_checks=5)
```

### Processing Historical Data

```python
# Analyze historical price data
def analyze_price_history(prices):
    """
    Analyze historical prices for trends and patterns
    """
    if len(prices) < 2:
        return "Insufficient data"
    
    results = {
        "trend": None,
        "volatility": None,
        "support": None,
        "resistance": None,
        "avg_return": None
    }
    
    # Calculate daily returns
    returns = []
    for i in range(1, len(prices)):
        daily_return = (prices[i] - prices[i-1]) / prices[i-1] * 100
        returns.append(daily_return)
    
    # Trend analysis
    positive_days = sum(1 for r in returns if r > 0)
    total_days = len(returns)
    
    if positive_days / total_days > 0.6:
        results["trend"] = "BULLISH"
    elif positive_days / total_days < 0.4:
        results["trend"] = "BEARISH"
    else:
        results["trend"] = "SIDEWAYS"
    
    # Volatility (standard deviation of returns)
    avg_return = sum(returns) / len(returns)
    variance = sum((r - avg_return) ** 2 for r in returns) / len(returns)
    volatility = variance ** 0.5
    
    if volatility > 3:
        results["volatility"] = "HIGH"
    elif volatility > 1.5:
        results["volatility"] = "MODERATE"
    else:
        results["volatility"] = "LOW"
    
    # Support and resistance
    results["support"] = min(prices)
    results["resistance"] = max(prices)
    results["avg_return"] = avg_return
    
    return results

# Example: SBIN price history for 10 days
sbin_prices = [720.50, 725.30, 718.75, 722.10, 728.45, 724.80, 730.25, 726.90, 732.15, 729.60]

analysis = analyze_price_history(sbin_prices)
print("SBIN Price Analysis:")
print(f"Trend: {analysis['trend']}")
print(f"Volatility: {analysis['volatility']}")
print(f"Support: ₹{analysis['support']}")
print(f"Resistance: ₹{analysis['resistance']}")
print(f"Average Daily Return: {analysis['avg_return']:.2f}%")
```

## Nested Loops for Complex Analysis

```python
# Compare multiple stocks across multiple timeframes
stocks_data = {
    "SBIN": [720.50, 725.30, 718.75, 722.10, 728.45],
    "RELIANCE": [2450.00, 2456.75, 2448.20, 2465.80, 2470.25],
    "TCS": [3650.00, 3678.90, 3642.50, 3685.20, 3692.75]
}

timeframes = ["1D", "3D", "5D"]

print("Multi-Stock, Multi-Timeframe Analysis:")
print("=" * 50)

for stock, prices in stocks_data.items():
    print(f"\n{stock}:")
    
    for i, timeframe in enumerate(timeframes):
        if i == 0:  # 1 day
            period_prices = prices[-1:]
        elif i == 1:  # 3 days
            period_prices = prices[-3:]
        else:  # 5 days
            period_prices = prices
        
        if len(period_prices) > 1:
            start_price = period_prices[0]
            end_price = period_prices[-1]
            return_pct = (end_price - start_price) / start_price * 100
            
            status = "📈" if return_pct > 0 else "📉" if return_pct < 0 else "➡️"
            print(f"  {timeframe}: {return_pct:+6.2f}% {status}")
        else:
            print(f"  {timeframe}: No change")
```

## Break and Continue

Control loop execution with break and continue statements.

```python
# Find first stock meeting criteria
def find_undervalued_stock(stocks_data, max_pe_ratio=15):
    """
    Find the first stock with PE ratio below threshold
    """
    print(f"Searching for stocks with PE ratio < {max_pe_ratio}")
    
    for stock, data in stocks_data.items():
        pe_ratio = data.get("pe_ratio")
        
        if pe_ratio is None:
            print(f"{stock}: PE data not available - skipping")
            continue  # Skip to next stock
        
        print(f"{stock}: PE ratio = {pe_ratio}")
        
        if pe_ratio < max_pe_ratio:
            print(f"✅ Found: {stock} with PE {pe_ratio}")
            return stock  # Found our stock, exit function
    
    print("❌ No undervalued stocks found")
    return None

# Example data
market_data = {
    "SBIN": {"pe_ratio": 12.5, "price": 725.50},
    "RELIANCE": {"pe_ratio": 18.2, "price": 2456.75},
    "TCS": {"pe_ratio": 14.8, "price": 3678.90},
    "HDFCBANK": {"pe_ratio": None, "price": 1542.30}
}

undervalued = find_undervalued_stock(market_data, max_pe_ratio=15)
```

## Error Handling in Loops

```python
# Robust portfolio calculation with error handling
def calculate_portfolio_value(portfolio):
    """
    Calculate total portfolio value with error handling
    """
    total_value = 0
    successful_calculations = 0
    errors = []
    
    for stock, details in portfolio.items():
        try:
            quantity = details["quantity"]
            price = details["current_price"]
            
            # Validate data
            if quantity <= 0:
                raise ValueError(f"Invalid quantity: {quantity}")
            if price <= 0:
                raise ValueError(f"Invalid price: {price}")
            
            stock_value = quantity * price
            total_value += stock_value
            successful_calculations += 1
            
            print(f"✅ {stock}: {quantity} × ₹{price} = ₹{stock_value:,.2f}")
            
        except KeyError as e:
            error_msg = f"{stock}: Missing data - {e}"
            errors.append(error_msg)
            print(f"❌ {error_msg}")
            
        except ValueError as e:
            error_msg = f"{stock}: {e}"
            errors.append(error_msg)
            print(f"⚠️ {error_msg}")
        
        except Exception as e:
            error_msg = f"{stock}: Unexpected error - {e}"
            errors.append(error_msg)
            print(f"🚨 {error_msg}")
    
    print(f"\nSummary:")
    print(f"Successful calculations: {successful_calculations}")
    print(f"Errors: {len(errors)}")
    print(f"Total portfolio value: ₹{total_value:,.2f}")
    
    return total_value, errors

# Example with some problematic data
portfolio_with_errors = {
    "SBIN": {"quantity": 100, "current_price": 725.50},
    "RELIANCE": {"quantity": 50},  # Missing current_price
    "TCS": {"quantity": -25, "current_price": 3678.90},  # Invalid quantity
    "INFY": {"quantity": 75, "current_price": 1789.45}
}

total, errors = calculate_portfolio_value(portfolio_with_errors)
```

## Best Practices for Trading Logic

1. **Use Clear Conditions**: Make trading logic easy to understand
2. **Handle Edge Cases**: Check for zero values, missing data
3. **Log Decisions**: Print why trades were or weren't taken
4. **Validate Data**: Ensure prices and quantities are reasonable
5. **Use Constants**: Define thresholds as named constants

```python
# Good example: Clear, well-structured trading logic
# Trading constants
RSI_OVERSOLD = 30
RSI_OVERBOUGHT = 70
MIN_VOLUME = 1000000
MAX_POSITION_SIZE = 0.05  # 5% of portfolio

def generate_trading_signal(stock_data, portfolio_value):
    """
    Generate trading signal based on multiple criteria
    """
    # Extract data
    symbol = stock_data["symbol"]
    price = stock_data["current_price"]
    volume = stock_data["volume"]
    rsi = stock_data["rsi"]
    
    # Calculate position size
    max_investment = portfolio_value * MAX_POSITION_SIZE
    max_shares = int(max_investment / price)
    
    # Decision logic
    if volume < MIN_VOLUME:
        return "NO_TRADE", f"Volume too low: {volume:,} < {MIN_VOLUME:,}"
    
    if rsi <= RSI_OVERSOLD and max_shares > 0:
        return "BUY", f"RSI oversold ({rsi}) - Buy up to {max_shares} shares"
    
    elif rsi >= RSI_OVERBOUGHT:
        return "SELL", f"RSI overbought ({rsi}) - Consider selling"
    
    else:
        return "HOLD", f"RSI neutral ({rsi}) - No action needed"

# Example usage
stock_data = {
    "symbol": "SBIN",
    "current_price": 725.50,
    "volume": 1500000,
    "rsi": 28
}

signal, reason = generate_trading_signal(stock_data, portfolio_value=100000)
print(f"Signal: {signal}")
print(f"Reason: {reason}")
```

## Summary

In this note, you learned:
- How to implement trading logic with if-else statements
- Using loops to automate analysis of multiple stocks
- Processing historical data with for and while loops
- Error handling in trading calculations
- Best practices for clear, robust trading logic

## Next Steps

In the next note, we'll explore functions - how to create reusable code for financial calculations and trading strategies.

---

**Key Takeaways:**
- Use if-else statements for trading decisions
- Loops automate repetitive analysis tasks
- Always validate financial data before calculations
- Handle errors gracefully in trading systems
- Use clear, descriptive conditions and constants
