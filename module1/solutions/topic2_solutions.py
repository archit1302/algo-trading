# Solutions for Topic 2

# 1. Simple BUY/HOLD for TCS
price_TCS = float(input("Enter TCS price (₹): "))
if price_TCS < 3000:
    print("BUY TCS at ₹", price_TCS)
else:
    print("HOLD TCS at ₹", price_TCS)

# 2. RELIANCE Grade
price_REL = float(input("Enter RELIANCE price (₹): "))
if price_REL < 2500:
    print("Grade A BUY for RELIANCE")
elif price_REL < 2800:
    print("Grade B HOLD for RELIANCE")
else:
    print("Consider SELLING RELIANCE")

# 3. Market-Open Check for INFY
is_open = input("Is market open? (y/n): ").lower()
price_INFY = float(input("Enter INFY price (₹): "))
if is_open == "y":
    if price_INFY < 1200:
        print("Buy INFY at ₹", price_INFY)
    else:
        print("Wait for a better entry")
else:
    print("Market Closed")

# 4. Batch Classification
symbols   = ["TCS", "INFY", "HDFC"]
thresholds = [3000, 1200, 2600]
prices     = []
for s in symbols:
    p = float(input(f"Enter {s} price (₹): "))
    prices.append(p)

for sym, thr, pr in zip(symbols, thresholds, prices):
    action = "BUY" if pr < thr else "HOLD"
    print(f"{sym}: {action} at ₹{pr}")
