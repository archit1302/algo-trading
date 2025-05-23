# Types of Errors in Python

# 1. SyntaxError: 

# for i in range(10):
#     print(i)

# 2. IndentationError:

# for i in range(10):
#     print(i)
#     # print("Hello")

# 3. NameError:

# nifty_pr

# 4. TypeError:

symbol = "NIFTY"
expiry = "20250531" # YYYYMMDD format Year-Month-Day
strike = 20000
option_type = "CE"

trading_symbol = symbol + expiry + str(strike) + option_type
# print(trading_symbol)

# 5. ValueError:

# nifty_price = int("NIFTY")
# print(nifty_price)  # This will raise a ValueError due to invalid literal for int() with base 10: 'NIFTY'

# 6. IndexError:
nifty_prices = [1200, 1300, 1400, 1500, 1600]
# print(nifty_prices[5])  # This will raise an IndexError because the index is out of range

# 7. KeyError:

nifty_prices = {"RELIANCE": 2500, "TCS": 3500, "INFY": 4500}
print(nifty_prices.get("VBL"))  # This will raise a KeyError because "VBL" is not a key in the dictionary

# Module Not Found Error:
# from module1 import strings_python
# MODULENOTFOUND ; module not found for time
# pip3 install time 

nifty_prices = {"RELIANCE": 2500, "TCS": 3500, "INFY": 4500}
try:
    print(nifty_prices["VBL"])
except KeyError as e:
    print(f"The error is handled smoothly")

# Assignments
# 1. Create a function that takes a dictionary of stock names and prices, and adds a new stock to the dictionary.
# 2. Handle the KeyError exception when trying to access a non-existent key in the dictionary.
# 3. Handle the ValueError exception when trying to convert a string to an integer.
# 4. Handle the IndexError exception when trying to access an index that is out of range in a list.
