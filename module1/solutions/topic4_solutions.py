# Solutions for Topic 4

# 1.
def square(x):
    return x * x

# 2.
def return_pct(buy, sell):
    return ((sell - buy) / buy) * 100

# 3.
def sma(prices):
    return sum(prices) / len(prices)

# 4.
def greet_stock(symbol):
    print(f"🔍 Analyzing {symbol}...")

# 5. Combined
if __name__ == "__main__":
    greet_stock("TCS")
    print("Square of 10:", square(10))
    print("Return % (₹1000→₹1100):", return_pct(1000, 1100))
    avg = sma([3299.75, 3300.00, 3310.25])
    print("3-period SMA:", avg)
