

# # List -> 

# # Linear Structure -> [1,2,3,4,5,6]
# # Dictionary -> {1: "a", 2: "b", 3: "c", 4: "d", 5: "e", 6: "f"} 
# # dictionary is non-linear structure

# # 1 -> "a" -> key value pair 

# prices = {"RELIANCE": 2500, "TCS": 3500, "INFY": 4500}

# # print(prices["VBL"])

# # print(prices.get("VBL")) 

# price_list = list(prices.keys())
# # print(price_list)

# # INDEXING

# # everything in python starts from 0 

# price_list = [1200, 1300, 1400, 1500, 1600]

# print(price_list[2]) 

import time

def add_stock_to_dict(stock_dict: dict, stock_name: str, stock_price: float) -> dict:
    """
    Adds a stock name and price to the dictionary and returns the updated dictionary.
    
    Args:
        stock_dict (dict): The dictionary to update.
        stock_name (str): The name of the stock to add.
        stock_price (float): The price of the stock to add.
        
    Returns:
        dict: The updated dictionary with the new stock added.
    """
    stock_dict[stock_name] = stock_price
    return stock_dict

# Sample data
stock_names = ["RELIANCE", "TCS", "INFY", "HDFC", "ITC", "RELIANCE"]
stock_prices = [2500, 3500, 4500, 2700, 1900, 3500]

# Initialize an empty dictionary
stocks = {}

# Add stocks one by one and print the updated dictionary
for name, price in zip(stock_names, stock_prices):
    stocks = add_stock_to_dict(stocks, name, price)
    print(f"Updated Dictionary: {stocks}")
    time.sleep(3)  # Sleep for 3 seconds to simulate time delay


# Assignments
# # 1. Create a function that takes a dictionary of stock names and prices, and adds a new stock to the dictionary.
# # 2. Create a function that takes a dictionary and returns the average price of all stocks in the dictionary.
