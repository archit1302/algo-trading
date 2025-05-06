"""
Topic 2: Conditional Programming Assignments

You will:
1. Check buy/sell signals for TCS
2. Grade RELIANCE price zones
3. Nested logic on market status
4. Batch-classify a list of prices
"""

# 1. Simple BUY/HOLD:
#    - price_TCS: float input
#    - if price_TCS < 3000: print "BUY TCS"
#      else: print "HOLD TCS"

# 2. RELIANCE Grade:
#    - price_REL: float input
#    - if < 2500 → "Grade A BUY"
#      elif 2500–2800 → "Grade B HOLD"
#      else → "Consider SELL"

# 3. Market-Open Check:
#    - is_open: input "y" or "n"
#    - if is_open=="y": then check price_INFY:
#         if price_INFY < 1200 → "Buy INFY"
#         else → "Wait"
#      else: print "Market Closed"

# 4. Batch Classification:
#    - prices = [float inputs for TCS, INFY, HDFC]
#    - loop through and:
#         if price < threshold (3000, 1200, 2600 respectively)
#         print "<symbol>: BUY" or "<symbol>: HOLD"
