# Assignment 1: Python Basics for Financial Data

## Objective
Learn Python fundamentals by working with basic financial calculations and stock data.

## Prerequisites
- Complete reading: `01_python_basics.md`
- Basic understanding of variables and data types

## Tasks

### Task 1: Variables and Basic Calculations (20 points)
Create a Python script that:
1. Defines variables for a stock investment scenario:
   - `stock_name = "SBIN"`
   - `purchase_price = 650.50`
   - `current_price = 675.25`
   - `quantity = 100`

2. Calculate and print:
   - Total investment amount
   - Current portfolio value
   - Profit/Loss amount
   - Profit/Loss percentage

### Task 2: Data Type Operations (15 points)
1. Create variables of different data types:
   - String: Company name
   - Integer: Number of shares
   - Float: Stock price
   - Boolean: Is the stock profitable?

2. Print the type of each variable using `type()` function
3. Demonstrate type conversion by converting the stock price to an integer

### Task 3: String Operations (15 points)
1. Create a stock ticker string: `"SBIN.NS"`
2. Extract just the stock symbol (before the dot)
3. Create a formatted string that displays: "Stock: SBIN, Price: ₹675.25"
4. Convert the stock name to uppercase and lowercase

## Expected Output Format
```
=== STOCK INVESTMENT CALCULATOR ===
Stock: SBIN
Purchase Price: ₹650.50
Current Price: ₹675.25
Quantity: 100 shares

Total Investment: ₹65,050.00
Current Value: ₹67,525.00
Profit/Loss: ₹2,475.00
Profit/Loss %: 3.81%

=== DATA TYPES ===
Company Name: SBIN (type: <class 'str'>)
Shares: 100 (type: <class 'int'>)
Price: 675.25 (type: <class 'float'>)
Profitable: True (type: <class 'bool'>)

Price as integer: 675

=== STRING OPERATIONS ===
Full Ticker: SBIN.NS
Stock Symbol: SBIN
Formatted: Stock: SBIN, Price: ₹675.25
Uppercase: SBIN
Lowercase: sbin
```

## Submission Guidelines
1. Create a file named `assignment1_solution.py`
2. Include comments explaining each calculation
3. Use meaningful variable names
4. Format output clearly with proper labels

## Evaluation Criteria
- Correct variable declarations (20%)
- Accurate calculations (40%)
- Proper string operations (20%)
- Code readability and comments (20%)

## Bonus Challenge (5 extra points)
Calculate the number of days it would take to break even if the stock moves by ₹0.50 per day in your favor.
