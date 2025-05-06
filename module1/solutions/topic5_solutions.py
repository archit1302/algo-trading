# Solutions for Topic 5

# 1.
def format_message(symbol, price):
    return f"BUY {symbol} at ₹{price:.2f}"

# 2.
def parse_candle(line):
    sym, o, h, l, c = line.split(",")
    return {
        "symbol": sym,
        "open": float(o),
        "high": float(h),
        "low": float(l),
        "close": float(c),
    }

# 3.
def build_watchlist(symbols):
    return ",".join(symbols)

# 4.
def tidy_input():
    raw = input("Enter symbol: ")
    return raw.strip().upper()

# Demo
if __name__=="__main__":
    print(format_message("INFY", 1200))
    print(parse_candle("TCS,3000,3050,2950,3025"))
    print(build_watchlist(["TCS","HDFC","SBIN"]))
    print("Tidied:", tidy_input())
