# Assignment 2: Data Structures for Portfolio Management

## Objective
Master Python data structures (lists, dictionaries, tuples) for managing financial portfolios and stock data.

## Prerequisites
- Complete Assignment 1
- Complete reading: `02_data_structures.md`

## Tasks

### Task 1: Portfolio with Lists (25 points)
Create a portfolio management system using lists:

1. Create lists for:
   - `stock_symbols = ["SBIN", "RELIANCE", "TCS", "INFY", "HDFC"]`
   - `purchase_prices = [650.50, 2890.75, 3450.20, 1876.30, 1654.80]`
   - `current_prices = [675.25, 2915.60, 3475.90, 1901.75, 1678.45]`
   - `quantities = [100, 50, 25, 75, 60]`

2. Calculate and display:
   - Total number of stocks in portfolio
   - Most expensive stock (by current price)
   - Cheapest stock (by current price)
   - Total portfolio investment
   - Current portfolio value

### Task 2: Stock Information Dictionary (25 points)
Create detailed stock information using dictionaries:

1. Create a dictionary for SBIN stock:
```python
sbin_stock = {
    "symbol": "SBIN",
    "company_name": "State Bank of India",
    "sector": "Banking",
    "purchase_price": 650.50,
    "current_price": 675.25,
    "quantity": 100,
    "purchase_date": "2025-01-15"
}
```

2. Perform operations:
   - Add a new key "market_cap" with value "5,50,000 Cr"
   - Update the current price to 680.75
   - Calculate profit/loss for this stock
   - Print all keys and values

### Task 3: Multi-Stock Portfolio Dictionary (25 points)
Create a comprehensive portfolio using nested dictionaries:

```python
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
    }
    # Add 3 more stocks
}
```

Operations:
- Calculate total portfolio value
- Find the best performing stock (highest % gain)
- Find the worst performing stock
- Count profitable vs loss-making stocks

### Task 4: Tuples for Stock Data (15 points)
Use tuples for immutable stock data:

1. Create tuples for daily OHLC data:
   - `day1_ohlc = (670.50, 680.75, 665.20, 675.25)`  # Open, High, Low, Close
   - `day2_ohlc = (675.00, 682.30, 673.80, 678.90)`

2. Operations:
   - Calculate daily trading range (High - Low)
   - Determine if the day was bullish (Close > Open)
   - Find the average closing price

### Task 5: Mixed Data Structures (10 points)
Combine all data structures:

1. Create a list of tuples for weekly data
2. Store this in a dictionary with stock symbol as key
3. Calculate weekly performance metrics

## Expected Output Format
```
=== PORTFOLIO ANALYSIS ===
Total Stocks: 5
Most Expensive: TCS at ₹3,475.90
Cheapest: SBIN at ₹675.25
Total Investment: ₹4,12,345.50
Current Value: ₹4,18,967.25
Overall Profit: ₹6,621.75

=== SBIN STOCK DETAILS ===
Symbol: SBIN
Company: State Bank of India
Sector: Banking
Purchase Price: ₹650.50
Current Price: ₹680.75
Quantity: 100
Market Cap: 5,50,000 Cr
Profit/Loss: ₹3,025.00 (4.65%)

=== BEST PERFORMERS ===
Best: RELIANCE (+0.86%)
Worst: HDFC (+1.43%)
Profitable Stocks: 5
Loss-making Stocks: 0

=== DAILY OHLC ANALYSIS ===
Day 1: Range = ₹15.55, Bullish: Yes
Day 2: Range = ₹8.50, Bullish: Yes
Average Closing: ₹677.08
```

## Submission Guidelines
1. Create a file named `assignment2_solution.py`
2. Use appropriate data structures for each task
3. Include comprehensive comments
4. Handle edge cases (empty lists, missing keys)

## Evaluation Criteria
- Correct use of lists (25%)
- Proper dictionary operations (35%)
- Appropriate tuple usage (15%)
- Code organization and comments (25%)

## Bonus Challenges (10 extra points)
1. Sort stocks by performance percentage
2. Create a function to add new stocks to the portfolio
3. Implement portfolio rebalancing logic
