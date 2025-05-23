# Assignment 1 Solution: Python Basics for Financial Data
# Author: Module 1 Learning Materials
# Date: May 2025

"""
This solution demonstrates Python fundamentals through stock investment calculations.
It covers variables, data types, basic arithmetic, and string operations.
"""

print("=" * 50)
print("    STOCK INVESTMENT CALCULATOR")
print("=" * 50)

# Task 1: Variables and Basic Calculations

# Define stock investment variables
stock_name = "SBIN"
purchase_price = 650.50  # Price at which stock was bought
current_price = 675.25   # Current market price
quantity = 100           # Number of shares

# Calculate investment metrics
total_investment = purchase_price * quantity
current_value = current_price * quantity
profit_loss_amount = current_value - total_investment
profit_loss_percentage = (profit_loss_amount / total_investment) * 100

# Display results with proper formatting
print(f"Stock: {stock_name}")
print(f"Purchase Price: ₹{purchase_price:.2f}")
print(f"Current Price: ₹{current_price:.2f}")
print(f"Quantity: {quantity} shares")
print()
print(f"Total Investment: ₹{total_investment:,.2f}")
print(f"Current Value: ₹{current_value:,.2f}")
print(f"Profit/Loss: ₹{profit_loss_amount:,.2f}")
print(f"Profit/Loss %: {profit_loss_percentage:.2f}%")

print("\n" + "=" * 50)
print("    DATA TYPES DEMONSTRATION")
print("=" * 50)

# Task 2: Data Type Operations

# Create variables of different data types
company_name = "SBIN"          # String
shares_count = 100             # Integer  
stock_price = 675.25           # Float
is_profitable = profit_loss_amount > 0  # Boolean

# Display variable types
print(f"Company Name: {company_name} (type: {type(company_name)})")
print(f"Shares: {shares_count} (type: {type(shares_count)})")
print(f"Price: {stock_price} (type: {type(stock_price)})")
print(f"Profitable: {is_profitable} (type: {type(is_profitable)})")

# Type conversion demonstration
price_as_integer = int(stock_price)
print(f"\nPrice as integer: {price_as_integer}")

print("\n" + "=" * 50)
print("    STRING OPERATIONS")
print("=" * 50)

# Task 3: String Operations

# Create stock ticker string
full_ticker = "SBIN.NS"

# Extract stock symbol (before the dot)
stock_symbol = full_ticker.split('.')[0]

# Alternative method using string slicing
# stock_symbol = full_ticker[:4]

# Create formatted string
formatted_string = f"Stock: {stock_symbol}, Price: ₹{current_price:.2f}"

# String case conversions
uppercase_name = stock_symbol.upper()
lowercase_name = stock_symbol.lower()

# Display string operations results
print(f"Full Ticker: {full_ticker}")
print(f"Stock Symbol: {stock_symbol}")
print(f"Formatted: {formatted_string}")
print(f"Uppercase: {uppercase_name}")
print(f"Lowercase: {lowercase_name}")

print("\n" + "=" * 50)
print("    BONUS CHALLENGE")
print("=" * 50)

# Bonus Challenge: Break-even calculation
daily_movement = 0.50  # Stock moves ₹0.50 per day

if profit_loss_amount < 0:  # If currently in loss
    loss_amount = abs(profit_loss_amount)
    days_to_breakeven = loss_amount / (daily_movement * quantity)
    print(f"Current Loss: ₹{loss_amount:.2f}")
    print(f"Days to break even (at ₹{daily_movement}/day): {days_to_breakeven:.1f} days")
else:
    print(f"Already profitable by ₹{profit_loss_amount:.2f}")
    print("No need to calculate break-even time!")

print("\n" + "=" * 50)
print("    CALCULATION SUMMARY")
print("=" * 50)

# Summary with all key metrics
print(f"Investment Analysis for {stock_name}:")
print(f"• Invested Amount: ₹{total_investment:,.2f}")
print(f"• Current Value: ₹{current_value:,.2f}")
print(f"• Gain/Loss: ₹{profit_loss_amount:+,.2f} ({profit_loss_percentage:+.2f}%)")
print(f"• Investment Status: {'Profitable' if is_profitable else 'Loss-making'}")

# Additional insights
if profit_loss_percentage > 5:
    print("💰 Excellent returns! Consider taking some profits.")
elif profit_loss_percentage > 2:
    print("📈 Good performance! Hold for further gains.")
elif profit_loss_percentage > 0:
    print("✅ Positive returns. Continue monitoring.")
else:
    print("⚠️ In loss. Consider reviewing your strategy.")

print("\nProgram completed successfully! 🎉")

"""
KEY LEARNING POINTS:

1. VARIABLES: 
   - Used meaningful names like 'purchase_price' instead of 'pp'
   - Proper data types for each piece of information

2. CALCULATIONS:
   - Basic arithmetic operations (+, -, *, /)
   - Percentage calculations using formula: (change/original) * 100

3. STRING OPERATIONS:
   - String formatting with f-strings for clean output
   - String methods: split(), upper(), lower()
   - String slicing for extracting parts

4. DATA TYPES:
   - String for text data (stock names)
   - Integer for whole numbers (quantity)
   - Float for decimal numbers (prices)
   - Boolean for true/false conditions

5. BEST PRACTICES:
   - Clear variable names
   - Proper comments explaining logic
   - Formatted output for readability
   - Error handling considerations

This solution demonstrates real-world application of Python basics
in financial contexts, preparing students for more advanced concepts.
"""
