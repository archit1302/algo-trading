# Solutions for Topic 6

import statistics

# 1.
def list_stats(prices):
    return min(prices), max(prices), statistics.mean(prices)

# 2.
def build_market(symbols, prices):
    return [{"symbol":s, "price":p} for s, p in zip(symbols, prices)]

# 3.
def find_stock(market, symbol):
    for stock in market:
        if stock["symbol"] == symbol:
            return stock["price"]
    return None

# 4.
def update_price(market, symbol, new_price):
    for stock in market:
        if stock["symbol"] == symbol:
            stock["price"] = new_price
            print(f"Updated {symbol} to ₹{new_price}")
            return
    print(symbol, "not found")
