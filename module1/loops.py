# For loop to print the first 10 numbers
# for i in range(1, 11):
#     print(i)

# import random

# # Example ATM strike price
# atm_strike_price = 25000

# # Generate random premiums for 10 strike prices above and below the ATM strike price
# strike_prices = [atm_strike_price + (i * 50) for i in range(-10, 11)]
# premiums = {strike: round(random.uniform(50, 500), 2) for strike in strike_prices}

# # Iterate and print the strike prices and their premiums
# print("Strike Price | Premium")
# print("----------------------")
# for strike, premium in premiums.items():
#     print(f"{strike:<12} | {premium}")

# import random

# # Example ATM strike price
# atm_strike_price = 25000

# # Customizable strike difference
# strike_difference = 10  # Change this value to customize the difference between strike prices

# # Generate random premiums for 10 strike prices above and below the ATM strike price
# strike_prices = [atm_strike_price + (i * strike_difference) for i in range(-10, 11)]
# premiums = {strike: round(random.uniform(50, 500), 2) for strike in strike_prices}

# # Iterate and print the strike prices and their premiums
# print("Strike Price | Premium")
# print("----------------------")
# for strike, premium in premiums.items():
#     print(f"{strike:<12} | {premium}")


# import random
# from datetime import datetime, timedelta

# # Define the start and end times
# start_time = datetime.strptime("09:15", "%H:%M")
# end_time = datetime.strptime("15:30", "%H:%M")

# # Initialize the current time to the start time
# current_time = start_time

# # Loop to simulate the time from 9:15 AM to 3:30 PM
# while current_time <= end_time:
#     # Generate a random Nifty price
#     nifty_price = random.randint(24000, 26000)  # Random price between 24000 and 26000

#     # Check if Nifty price goes above 25000
#     if nifty_price > 25000:
#         print(f"At {current_time.strftime('%H:%M')}, Nifty price is {nifty_price} (Above 25000)")
#     else:
#         print(f"At {current_time.strftime('%H:%M')}, Nifty price is {nifty_price} (Below 25000)")

#     # Increment the time by 1 minute
#     current_time += timedelta(minutes=1)


# import random
# from datetime import datetime, timedelta
# import time  # Import time module for sleep functionality

# # Define the start and end times
# start_time = datetime.strptime("09:15", "%H:%M")
# end_time = datetime.strptime("15:30", "%H:%M")

# # Initialize the current time to the start time
# current_time = start_time

# # Loop to simulate the time from 9:15 AM to 3:30 PM
# while current_time <= end_time:
#     # Generate a random Nifty price
#     nifty_price = random.randint(24000, 26000)  # Random price between 24000 and 26000

#     # Check if Nifty price goes above 25000
#     if nifty_price > 25000:
#         print(f"At {current_time.strftime('%H:%M')}, Nifty price is {nifty_price} (Above 25000)")
#     else:
#         print(f"At {current_time.strftime('%H:%M')}, Nifty price is {nifty_price} (Below 25000)")

#     # Increment the time by 1 minute
#     current_time += timedelta(minutes=1)

#     # Wait for 1 minute before the next iteration
#     time.sleep(5)




import random
from datetime import datetime, timedelta
import time  # Import time module for sleep functionality

# Define the start and end times
start_time = datetime.strptime("09:15", "%H:%M")
end_time = datetime.strptime("15:30", "%H:%M")

# Initialize the current time to the start time
current_time = start_time

# List of 5 Indian stocks
stocks = ["Reliance", "TCS", "Infosys", "HDFC Bank", "ITC"]

# Loop to simulate the time from 9:15 AM to 3:30 PM
while current_time <= end_time:
    print(f"Time: {current_time.strftime('%H:%M')}")

    # Nested loop to generate and print prices for 5 stocks
    for stock in stocks:
        stock_price = round(random.uniform(1000, 5000), 2)  # Generate random stock price
        print(f"{stock}: ₹{stock_price}")

    print("-" * 30)  # Separator for readability

    # Increment the time by 1 minute
    current_time += timedelta(minutes=1)

    # Wait for 1 minute before the next iteration
    time.sleep(5)


# Assignment -
# 1. Create a list of 5 Indian stocks and generate random prices for each stock every minute from 9:15 AM to 3:30 PM.
# 2. Use a while loop to iterate through the time and print the stock prices.
# 3. Use a nested loop to generate and print prices for each stock.
# 4. Use a sleep function to simulate the time delay between each iteration.



# Create a list of 5 Indian stocks and generate random prices for each stock every minute from 9:15 AM to 3:30 PM.

