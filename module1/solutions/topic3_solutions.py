# Solutions for Topic 3

# 1. Watchlist Printer
symbols = ["TCS","INFY","HDFC","SBIN"]
for s in symbols:
    print("Watch:", s)

# 2. Sum Prices
prices = []
for i in range(5):
    p = float(input(f"Enter price #{i+1} (₹): "))
    prices.append(p)
total = 0
for p in prices:
    total += p
print("Total sum: ₹", total)

# 3. Max Price
max_price = prices[0]
max_sym   = symbols[0] if len(symbols)>0 else "N/A"
for sym, pr in zip(symbols, prices):
    if pr > max_price:
        max_price = pr
        max_sym   = sym
print(f"{max_sym} highest at ₹{max_price}")

# 4. Market-Open Countdown
count = 3
while count > 0:
    print("Opens in", count)
    count -= 1
print("Market OPEN!")

# 5. Clean List
mixed = [100, -5, 0, 250, 300]
for pr in mixed:
    if pr <= 0:
        continue
    print("Valid price:", pr)
