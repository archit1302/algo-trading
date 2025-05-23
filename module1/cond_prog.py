# Check the availability of ice cream and decide what to eat
# ice_cream = input("Enter the available ice cream flavor (chocolate/mango/strawberry/none): ").lower()

# if ice_cream == "chocolate":
#     print("I'll eat chocolate ice cream.")
# elif ice_cream == "mango":
#     print("I'll eat mango ice cream.")
# elif ice_cream == "strawberry":
#     print("I'll eat strawberry ice cream.")
# else:
#     print("I won't eat any ice cream.")

# Assignments - 
# 1. Top 5 stocks and make if-else ladder like above with these stocks

# Example variables for stock prices and presence
nifty = 26000  # Example value for Nifty index
reliance_present = True
reliance_price = 1200
tcs_present = True
tcs_price = 3600
itc_present = False
itc_price = 240
wipro_present = True

# Nested if-else conditions
if nifty > 25000:
    if reliance_present and reliance_price > 1300:
        print("Buy Reliance")
    elif tcs_present and tcs_price > 3500:
        print("Buy TCS")
elif nifty < 22000:
    if itc_present and itc_price < 250:
        print("Buy ITC")
    elif not itc_present and wipro_present:
        print("Buy Wipro")
 


