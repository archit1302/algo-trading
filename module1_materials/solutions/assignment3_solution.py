# Assignment 3 Solution: Control Flow for Trading Logic
# Author: Module 1 Learning Materials
# Date: May 2025

"""
This solution demonstrates control flow structures (if-else, loops) 
through comprehensive trading logic implementation.
"""

print("=" * 60)
print("    TRADING LOGIC WITH CONTROL FLOW")
print("=" * 60)

# Task 1: Stock Screening with Conditionals

print("\n=== TASK 1: STOCK SCREENING SYSTEM ===")

# Sample stock data for screening
stocks = [
    {"symbol": "SBIN", "price": 675.25, "pe_ratio": 12.5, "market_cap": 550000},
    {"symbol": "RELIANCE", "price": 2915.60, "pe_ratio": 18.7, "market_cap": 1975000},
    {"symbol": "TCS", "price": 3475.90, "pe_ratio": 25.3, "market_cap": 1265000},
    {"symbol": "INFY", "price": 1901.75, "pe_ratio": 22.1, "market_cap": 789000},
    {"symbol": "HDFC", "price": 1678.45, "pe_ratio": 15.8, "market_cap": 912000}
]

# Screening criteria
VALUE_PE_THRESHOLD = 15
VALUE_MCAP_MIN = 500000
GROWTH_PE_THRESHOLD = 20
GROWTH_PRICE_MIN = 2000
LARGE_CAP_MIN = 1000000
MID_CAP_MIN = 250000
MID_CAP_MAX = 1000000

print("Stock Screening Results:")
print("-" * 40)

for stock in stocks:
    symbol = stock["symbol"]
    price = stock["price"]
    pe_ratio = stock["pe_ratio"]
    market_cap = stock["market_cap"]
    
    categories = []
    
    # Value stock criteria
    if pe_ratio < VALUE_PE_THRESHOLD and market_cap > VALUE_MCAP_MIN:
        categories.append("Value Stock")
    
    # Growth stock criteria
    if pe_ratio > GROWTH_PE_THRESHOLD and price > GROWTH_PRICE_MIN:
        categories.append("Growth Stock")
    
    # Market cap classification
    if market_cap > LARGE_CAP_MIN:
        categories.append("Large Cap")
    elif market_cap >= MID_CAP_MIN and market_cap <= MID_CAP_MAX:
        categories.append("Mid Cap")
    else:
        categories.append("Small Cap")
    
    # Display results
    category_str = ", ".join(categories) if categories else "No Category"
    print(f"{symbol:10}: {category_str}")

# Task 2: Trading Signal Generator

print("\n=== TASK 2: TRADING SIGNAL GENERATOR ===")

# 5-day price data for SBIN
sbin_prices = [665.20, 670.50, 675.25, 678.90, 682.30]

def calculate_sma(prices, period):
    """Calculate Simple Moving Average"""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

print("Trading Signals - SBIN:")
print("-" * 30)

previous_signal = None

for i in range(len(sbin_prices)):
    current_price = sbin_prices[i]
    
    # Calculate 3-day SMA if we have enough data
    if i >= 2:  # Need at least 3 days for 3-day SMA
        sma_3 = calculate_sma(sbin_prices[:i+1], 3)
        
        # Determine signal
        if i > 2:  # Can compare with previous day
            prev_price = sbin_prices[i-1]
            prev_sma_3 = calculate_sma(sbin_prices[:i], 3)
            
            # Crossover logic
            if current_price > sma_3 and prev_price <= prev_sma_3:
                signal = "BUY"
            elif current_price < sma_3 and prev_price >= prev_sma_3:
                signal = "SELL"
            else:
                signal = "HOLD"
        else:
            signal = "HOLD"
    else:
        sma_3 = None
        signal = "HOLD"
    
    # Display daily analysis
    sma_str = f"{sma_3:.2f}" if sma_3 else "N/A"
    print(f"Day {i+1}: Price={current_price:.2f}, SMA={sma_str}, Signal={signal}")

# Task 3: Portfolio Risk Assessment

print("\n=== TASK 3: PORTFOLIO RISK ASSESSMENT ===")

# Portfolio allocation
portfolio_allocation = {
    "equity": 70,      # percentage
    "debt": 20,
    "gold": 5,
    "cash": 5
}

print("Portfolio Risk Analysis:")
print("-" * 25)

# Risk categorization
equity_percentage = portfolio_allocation["equity"]

if equity_percentage < 40:
    risk_category = "CONSERVATIVE"
elif equity_percentage <= 70:
    risk_category = "MODERATE"
else:
    risk_category = "AGGRESSIVE"

print(f"Risk Category: {risk_category}")

# Additional risk checks
warnings = []
recommendations = []

# Cash allocation check
cash_percentage = portfolio_allocation["cash"]
if cash_percentage > 15:
    warnings.append(f"High cash allocation ({cash_percentage}%) - money sitting idle")

# Concentration risk check
max_allocation = max(portfolio_allocation.values())
max_asset = max(portfolio_allocation.items(), key=lambda x: x[1])

if max_allocation > 80:
    warnings.append(f"Concentration risk in {max_asset[0]} ({max_allocation}%)")

# Generate recommendations
if risk_category == "AGGRESSIVE":
    recommendations.append("Consider reducing equity allocation for better risk management")
elif risk_category == "CONSERVATIVE":
    recommendations.append("Consider increasing equity allocation for better returns")

# Display warnings and recommendations
if warnings:
    print("Warnings:")
    for warning in warnings:
        print(f"- {warning}")
else:
    print("- No concentration risk detected")

if recommendations:
    print("Recommendations:")
    for rec in recommendations:
        print(f"- {rec}")
else:
    print("- Portfolio allocation optimal")

# Task 4: Batch Processing with Loops

print("\n=== TASK 4: BATCH STOCK ANALYSIS ===")

# Stock data with price histories
stock_data = [
    {"symbol": "SBIN", "prices": [665.20, 670.50, 675.25, 678.90, 682.30]},
    {"symbol": "RELIANCE", "prices": [2890.75, 2895.60, 2915.60, 2920.40, 2935.80]},
    {"symbol": "TCS", "prices": [3450.20, 3465.90, 3475.90, 3480.25, 3495.60]}
]

def calculate_daily_returns(prices):
    """Calculate daily returns as percentages"""
    returns = []
    for i in range(1, len(prices)):
        daily_return = ((prices[i] - prices[i-1]) / prices[i-1]) * 100
        returns.append(daily_return)
    return returns

def calculate_volatility(returns):
    """Calculate volatility (standard deviation of returns)"""
    if len(returns) < 2:
        return 0
    
    mean_return = sum(returns) / len(returns)
    variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
    return variance ** 0.5

def calculate_total_return(prices):
    """Calculate total return over the period"""
    return ((prices[-1] - prices[0]) / prices[0]) * 100

# Analyze each stock
stock_analysis = []

for stock in stock_data:
    symbol = stock["symbol"]
    prices = stock["prices"]
    
    # Calculate metrics
    daily_returns = calculate_daily_returns(prices)
    volatility = calculate_volatility(daily_returns)
    total_return = calculate_total_return(prices)
    
    # Risk-adjusted return (return per unit of risk)
    risk_adjusted_return = total_return / volatility if volatility > 0 else 0
    
    analysis = {
        "symbol": symbol,
        "daily_returns": daily_returns,
        "volatility": volatility,
        "total_return": total_return,
        "risk_adjusted_return": risk_adjusted_return
    }
    
    stock_analysis.append(analysis)

# Display analysis results
for analysis in stock_analysis:
    symbol = analysis["symbol"]
    daily_returns = analysis["daily_returns"]
    volatility = analysis["volatility"]
    total_return = analysis["total_return"]
    risk_adjusted_return = analysis["risk_adjusted_return"]
    
    print(f"\n{symbol} Analysis:")
    returns_str = [f"{r:.2f}%" for r in daily_returns]
    print(f"  Daily Returns: {returns_str}")
    print(f"  Volatility: {volatility:.2f}%")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Risk-Adjusted Return: {risk_adjusted_return:.2f}")

# Rank stocks by risk-adjusted return
stock_analysis.sort(key=lambda x: x["risk_adjusted_return"], reverse=True)

print(f"\nStock Rankings (by Risk-Adjusted Return):")
for i, analysis in enumerate(stock_analysis, 1):
    symbol = analysis["symbol"]
    rar = analysis["risk_adjusted_return"]
    print(f"{i}. {symbol} ({rar:.2f})")

# Bonus Challenges

print("\n=== BONUS CHALLENGES ===")

# Bonus 1: Stop-loss system
print("1. Stop-Loss Alert System:")

def check_stop_loss(current_price, purchase_price, stop_loss_percent=5):
    """Check if stop-loss is triggered"""
    stop_loss_price = purchase_price * (1 - stop_loss_percent/100)
    if current_price <= stop_loss_price:
        loss_percent = ((current_price - purchase_price) / purchase_price) * 100
        return True, loss_percent, stop_loss_price
    return False, 0, stop_loss_price

# Test stop-loss for sample positions
test_positions = [
    {"symbol": "SBIN", "purchase": 720.00, "current": 680.00},
    {"symbol": "TCS", "purchase": 3400.00, "current": 3475.90}
]

for position in test_positions:
    symbol = position["symbol"]
    purchase = position["purchase"]
    current = position["current"]
    
    triggered, loss_pct, stop_price = check_stop_loss(current, purchase)
    
    if triggered:
        print(f"🚨 STOP-LOSS TRIGGERED: {symbol}")
        print(f"   Purchase: ₹{purchase:.2f}, Current: ₹{current:.2f}")
        print(f"   Loss: {loss_pct:.2f}%, Stop Price: ₹{stop_price:.2f}")
    else:
        print(f"✅ {symbol}: Stop-loss not triggered (Stop: ₹{stop_price:.2f})")

# Bonus 2: Momentum indicator
print("\n2. Momentum Indicator:")

def calculate_momentum(prices, period=3):
    """Calculate price momentum"""
    if len(prices) < period + 1:
        return None
    
    current_price = prices[-1]
    past_price = prices[-(period + 1)]
    momentum = ((current_price - past_price) / past_price) * 100
    
    if momentum > 2:
        return "STRONG_BULLISH"
    elif momentum > 0:
        return "BULLISH"
    elif momentum > -2:
        return "BEARISH"
    else:
        return "STRONG_BEARISH"

for stock in stock_data:
    symbol = stock["symbol"]
    prices = stock["prices"]
    momentum_signal = calculate_momentum(prices)
    
    if momentum_signal:
        print(f"{symbol}: {momentum_signal}")

print("\n" + "=" * 60)
print("Control Flow Assignment Completed Successfully! 🎉")
print("=" * 60)

"""
KEY LEARNING POINTS:

1. CONDITIONAL STATEMENTS:
   - Used if-elif-else for multi-condition logic
   - Implemented complex screening criteria
   - Boolean logic for risk assessment

2. LOOPS:
   - For loops for iterating through data
   - Nested loops for complex analysis
   - List comprehensions for efficient processing

3. FUNCTIONS IN LOOPS:
   - Created reusable calculation functions
   - Modular code design for better maintainability

4. REAL-WORLD APPLICATIONS:
   - Stock screening systems
   - Trading signal generation
   - Risk management alerts
   - Performance analysis

5. BEST PRACTICES:
   - Clear variable names and constants
   - Modular function design
   - Comprehensive error handling
   - Professional output formatting

This solution demonstrates how control flow structures enable
sophisticated trading logic and automated decision-making systems.
"""
