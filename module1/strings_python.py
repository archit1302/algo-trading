
# a = "NIFTY"
# b = "Bank"

# # print(a + " " + b)

# c = a + b
# print(c)

symbol = "NIFTY"
expiry = "20250531" # YYYYMMDD format Year-Month-Day
strike = "20000"
option_type = "CE"

trading_symbol = symbol + expiry + strike + option_type
# print(trading_symbol)

trading_symbol1 = f"RELIANCEexpiry"
trading_symbol2 = f"RELIANCE{expiry}"

# print(trading_symbol2)

# date time -> datetime format

from datetime import datetime

# Example string representing time in the format HH:MM:SS
time_string = "15:30:00"
time_format = "%H:%M:%S"

# Convert the string time to a datetime object
time_obj = datetime.strptime(time_string, time_format)

print("Converted datetime object:", time_obj.time(), "Type:", type(time_obj))

time_back_to_string = time_obj.strftime(time_format)
print("Converted back to string:", time_back_to_string, "Type:", type(time_back_to_string))

# Assignments
# 1. Convert the string "2023-10-01" to a datetime object and print it
# 2. Convert the datetime object back to a string in the format "YYYY-MM-DD"
