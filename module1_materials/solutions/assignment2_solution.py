# Assignment 2 Solution: Data Structures for Portfolio Management
# Author: Module 1 Learning Materials
# Date: May 2025

"""
This solution demonstrates Python data structures (lists, dictionaries, tuples)
through comprehensive portfolio management examples.
"""

print("=" * 60)
print("    PORTFOLIO MANAGEMENT WITH DATA STRUCTURES")
print("=" * 60)

# Task 1: Portfolio with Lists

print("\n=== TASK 1: PORTFOLIO ANALYSIS WITH LISTS ===")

# Create portfolio data using lists
stock_symbols = ["SBIN", "RELIANCE", "TCS", "INFY", "HDFC"]
purchase_prices = [650.50, 2890.75, 3450.20, 1876.30, 1654.80]
current_prices = [675.25, 2915.60, 3475.90, 1901.75, 1678.45]
quantities = [100, 50, 25, 75, 60]

# Calculate portfolio metrics
total_stocks = len(stock_symbols)

# Find most expensive and cheapest stocks
max_price_index = current_prices.index(max(current_prices))
min_price_index = current_prices.index(min(current_prices))

most_expensive_stock = stock_symbols[max_price_index]
most_expensive_price = current_prices[max_price_index]

cheapest_stock = stock_symbols[min_price_index]  
cheapest_price = current_prices[min_price_index]

# Calculate total investment and current value
total_investment = sum(purchase_prices[i] * quantities[i] for i in range(len(stock_symbols)))
current_value = sum(current_prices[i] * quantities[i] for i in range(len(stock_symbols)))
overall_profit = current_value - total_investment

# Display results
print(f"Total Stocks: {total_stocks}")
print(f"Most Expensive: {most_expensive_stock} at ₹{most_expensive_price:.2f}")
print(f"Cheapest: {cheapest_stock} at ₹{cheapest_price:.2f}")
print(f"Total Investment: ₹{total_investment:,.2f}")
print(f"Current Value: ₹{current_value:,.2f}")
print(f"Overall Profit: ₹{overall_profit:,.2f}")

# Task 2: Stock Information Dictionary

print("\n=== TASK 2: STOCK DICTIONARY OPERATIONS ===")

# Create SBIN stock dictionary
sbin_stock = {
    "symbol": "SBIN",
    "company_name": "State Bank of India",
    "sector": "Banking",
    "purchase_price": 650.50,
    "current_price": 675.25,
    "quantity": 100,
    "purchase_date": "2025-01-15"
}

# Add new key
sbin_stock["market_cap"] = "5,50,000 Cr"

# Update current price
sbin_stock["current_price"] = 680.75

# Calculate profit/loss for SBIN
sbin_investment = sbin_stock["purchase_price"] * sbin_stock["quantity"]
sbin_current_value = sbin_stock["current_price"] * sbin_stock["quantity"]
sbin_profit_loss = sbin_current_value - sbin_investment
sbin_profit_percentage = (sbin_profit_loss / sbin_investment) * 100

print(f"Symbol: {sbin_stock['symbol']}")
print(f"Company: {sbin_stock['company_name']}")
print(f"Sector: {sbin_stock['sector']}")
print(f"Purchase Price: ₹{sbin_stock['purchase_price']:.2f}")
print(f"Current Price: ₹{sbin_stock['current_price']:.2f}")
print(f"Quantity: {sbin_stock['quantity']}")
print(f"Market Cap: {sbin_stock['market_cap']}")
print(f"Profit/Loss: ₹{sbin_profit_loss:.2f} ({sbin_profit_percentage:.2f}%)")

print(f"\nAll keys: {list(sbin_stock.keys())}")
print(f"All values: {list(sbin_stock.values())}")

# Task 3: Multi-Stock Portfolio Dictionary

print("\n=== TASK 3: COMPREHENSIVE PORTFOLIO DICTIONARY ===")

# Create comprehensive portfolio using nested dictionaries
portfolio = {
    "SBIN": {
        "company_name": "State Bank of India",
        "purchase_price": 650.50,
        "current_price": 675.25,
        "quantity": 100
    },
    "RELIANCE": {
        "company_name": "Reliance Industries",
        "purchase_price": 2890.75,
        "current_price": 2915.60,
        "quantity": 50
    },
    "TCS": {
        "company_name": "Tata Consultancy Services",
        "purchase_price": 3450.20,
        "current_price": 3475.90,
        "quantity": 25
    },
    "INFY": {
        "company_name": "Infosys Limited",
        "purchase_price": 1876.30,
        "current_price": 1901.75,
        "quantity": 75
    },
    "HDFC": {
        "company_name": "HDFC Bank Limited",
        "purchase_price": 1654.80,
        "current_price": 1678.45,
        "quantity": 60
    }
}

# Calculate portfolio metrics
total_portfolio_value = 0
portfolio_performance = {}
profitable_stocks = 0
loss_making_stocks = 0

for symbol, stock_data in portfolio.items():
    # Calculate individual stock metrics
    investment = stock_data["purchase_price"] * stock_data["quantity"]
    current_val = stock_data["current_price"] * stock_data["quantity"]
    profit_loss = current_val - investment
    profit_percentage = (profit_loss / investment) * 100
    
    # Add to portfolio totals
    total_portfolio_value += current_val
    
    # Store performance data
    portfolio_performance[symbol] = {
        "profit_loss": profit_loss,
        "profit_percentage": profit_percentage
    }
    
    # Count profitable vs loss-making stocks
    if profit_loss > 0:
        profitable_stocks += 1
    else:
        loss_making_stocks += 1

# Find best and worst performing stocks
best_performer = max(portfolio_performance.items(), key=lambda x: x[1]["profit_percentage"])
worst_performer = min(portfolio_performance.items(), key=lambda x: x[1]["profit_percentage"])

print(f"Total Portfolio Value: ₹{total_portfolio_value:,.2f}")
print(f"Best Performer: {best_performer[0]} (+{best_performer[1]['profit_percentage']:.2f}%)")
print(f"Worst Performer: {worst_performer[0]} ({worst_performer[1]['profit_percentage']:+.2f}%)")
print(f"Profitable Stocks: {profitable_stocks}")
print(f"Loss-making Stocks: {loss_making_stocks}")

# Task 4: Tuples for Stock Data

print("\n=== TASK 4: OHLC DATA WITH TUPLES ===")

# Create tuples for daily OHLC data (Open, High, Low, Close)
day1_ohlc = (670.50, 680.75, 665.20, 675.25)
day2_ohlc = (675.00, 682.30, 673.80, 678.90)

def analyze_ohlc_day(ohlc_data, day_number):
    """Analyze a single day's OHLC data"""
    open_price, high_price, low_price, close_price = ohlc_data
    
    # Calculate trading range
    trading_range = high_price - low_price
    
    # Determine if day was bullish
    is_bullish = close_price > open_price
    
    return {
        "day": day_number,
        "range": trading_range,
        "bullish": is_bullish,
        "close": close_price
    }

# Analyze both days
day1_analysis = analyze_ohlc_day(day1_ohlc, 1)
day2_analysis = analyze_ohlc_day(day2_ohlc, 2)

# Calculate average closing price
closing_prices = [day1_analysis["close"], day2_analysis["close"]]
average_closing = sum(closing_prices) / len(closing_prices)

print(f"Day 1: Range = ₹{day1_analysis['range']:.2f}, Bullish: {'Yes' if day1_analysis['bullish'] else 'No'}")
print(f"Day 2: Range = ₹{day2_analysis['range']:.2f}, Bullish: {'Yes' if day2_analysis['bullish'] else 'No'}")
print(f"Average Closing: ₹{average_closing:.2f}")

# Task 5: Mixed Data Structures

print("\n=== TASK 5: MIXED DATA STRUCTURES ===")

# Create weekly data using list of tuples
weekly_data = {
    "SBIN": [
        ("2025-05-19", 670.50, 680.75, 665.20, 675.25, 2500000),  # (Date, O, H, L, C, Volume)
        ("2025-05-20", 675.00, 682.30, 673.80, 678.90, 1800000),
        ("2025-05-21", 679.50, 685.75, 677.20, 682.30, 2100000),
        ("2025-05-22", 682.00, 688.90, 679.50, 685.75, 1950000),
        ("2025-05-23", 686.00, 692.40, 684.20, 689.80, 2300000)
    ]
}

# Calculate weekly performance metrics
symbol = "SBIN"
week_data = weekly_data[symbol]

# Extract closing prices
weekly_closes = [day[4] for day in week_data]  # Index 4 is close price

# Calculate weekly performance
week_start_price = weekly_closes[0]
week_end_price = weekly_closes[-1]
weekly_return = ((week_end_price - week_start_price) / week_start_price) * 100

# Calculate average volume
weekly_volumes = [day[5] for day in week_data]  # Index 5 is volume
average_volume = sum(weekly_volumes) / len(weekly_volumes)

# Find highest and lowest prices of the week
all_highs = [day[2] for day in week_data]  # Index 2 is high
all_lows = [day[3] for day in week_data]   # Index 3 is low
week_high = max(all_highs)
week_low = min(all_lows)

print(f"Weekly Analysis for {symbol}:")
print(f"Week Start Price: ₹{week_start_price:.2f}")
print(f"Week End Price: ₹{week_end_price:.2f}")
print(f"Weekly Return: {weekly_return:+.2f}%")
print(f"Week High: ₹{week_high:.2f}")
print(f"Week Low: ₹{week_low:.2f}")
print(f"Average Volume: {average_volume:,.0f}")

print("\n" + "=" * 60)
print("    BONUS CHALLENGES")
print("=" * 60)

# Bonus Challenge 1: Sort stocks by performance
print("\n=== BONUS 1: STOCKS SORTED BY PERFORMANCE ===")

# Create a list of tuples for sorting
performance_list = [(symbol, data["profit_percentage"]) for symbol, data in portfolio_performance.items()]

# Sort by performance (descending order)
performance_list.sort(key=lambda x: x[1], reverse=True)

print("Stocks ranked by performance:")
for i, (symbol, performance) in enumerate(performance_list, 1):
    print(f"{i}. {symbol}: {performance:+.2f}%")

# Bonus Challenge 2: Add new stock function
print("\n=== BONUS 2: ADD NEW STOCK FUNCTION ===")

def add_new_stock(portfolio_dict, symbol, company_name, purchase_price, current_price, quantity):
    """Add a new stock to the portfolio"""
    portfolio_dict[symbol] = {
        "company_name": company_name,
        "purchase_price": purchase_price,
        "current_price": current_price,
        "quantity": quantity
    }
    print(f"✅ Added {symbol} ({company_name}) to portfolio")
    return portfolio_dict

# Add a new stock
portfolio = add_new_stock(
    portfolio, 
    "ICICIBANK", 
    "ICICI Bank Limited", 
    1234.50, 
    1245.20, 
    80
)

print(f"Portfolio now contains {len(portfolio)} stocks")

# Bonus Challenge 3: Portfolio rebalancing
print("\n=== BONUS 3: PORTFOLIO REBALANCING ===")

def calculate_portfolio_weights(portfolio_dict):
    """Calculate current portfolio weights"""
    total_value = 0
    stock_values = {}
    
    # Calculate total portfolio value and individual stock values
    for symbol, data in portfolio_dict.items():
        stock_value = data["current_price"] * data["quantity"]
        stock_values[symbol] = stock_value
        total_value += stock_value
    
    # Calculate weights as percentages
    weights = {symbol: (value / total_value) * 100 for symbol, value in stock_values.items()}
    
    return weights, total_value

weights, total_val = calculate_portfolio_weights(portfolio)

print(f"Current Portfolio Allocation (Total: ₹{total_val:,.2f}):")
for symbol, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
    print(f"  {symbol}: {weight:.1f}%")

# Check for concentration risk
max_allocation = max(weights.values())
if max_allocation > 30:
    print(f"⚠️ Concentration Risk: {max(weights, key=weights.get)} has {max_allocation:.1f}% allocation")
else:
    print("✅ Well diversified portfolio - no single stock > 30%")

print("\n" + "=" * 60)
print("    LEARNING SUMMARY")
print("=" * 60)

print("""
KEY CONCEPTS DEMONSTRATED:

1. LISTS:
   ✓ Creating and accessing lists
   ✓ List methods: index(), max(), min()
   ✓ List comprehensions for calculations
   ✓ Iterating through lists with enumerate()

2. DICTIONARIES:
   ✓ Creating nested dictionaries
   ✓ Adding and updating key-value pairs
   ✓ Dictionary methods: keys(), values(), items()
   ✓ Complex data lookups and calculations

3. TUPLES:
   ✓ Immutable data storage (OHLC data)
   ✓ Tuple unpacking for clean code
   ✓ Using tuples in lists for structured data

4. MIXED STRUCTURES:
   ✓ Lists of tuples for time-series data
   ✓ Dictionaries containing lists and tuples
   ✓ Complex data manipulations

5. PRACTICAL APPLICATIONS:
   ✓ Portfolio management and analysis
   ✓ Performance calculations and rankings
   ✓ Risk assessment and diversification
   ✓ Financial data processing

This solution showcases how data structures form the foundation
of financial applications and algorithmic trading systems.
""")

print("Assignment 2 completed successfully! 🎉📊")
