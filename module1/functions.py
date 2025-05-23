# def calculate_atm_strike(spot_price: float, strike_difference: float) -> float:
   
#     # Calculate the nearest strike price
#     atm_strike = round(spot_price / strike_difference) * strike_difference
#     return atm_strike

# # Example usage
# if __name__ == "__main__":
#     # Get user inputs
#     spot = float(input("Enter the spot price: "))
#     diff = float(input("Enter the strike difference: "))
    
#     # Calculate ATM strike
#     atm = calculate_atm_strike(spot, diff)
#     print(f"ATM Strike Price: {atm}")



# def analyze_price_levels(spot_price: float, support: float, resistance: float) -> str:
#     """
#     Analyzes price levels and returns trading action based on conditions
    
#     Args:
#         spot_price (float): Current market price
#         support (float): Support level price
#         resistance (float): Resistance level price
        
#     Returns:
#         str: Recommended trading action
#     """
#     if spot_price >= resistance:
#         return "SELL - Price at resistance level"
#     elif spot_price <= support:
#         return "BUY - Price at support level"
#     else:
#         return "HOLD - Price in between support and resistance"

# # Example usage
# if __name__ == "__main__":
#     # Get user inputs
#     current_price = float(input("Enter current price: "))
#     support_level = float(input("Enter support level: "))
#     resistance_level = float(input("Enter resistance level: "))
    
#     # Get trading action
#     action = analyze_price_levels(current_price, support_level, resistance_level)
#     print(f"\nPrice Analysis Result:")
#     print(f"Current Price: {current_price}")
#     print(f"Recommended Action: {action}")





def analyze_market_data(stocks: dict, target_price: float) -> list:
    """
    Analyzes multiple Indian stocks and returns those meeting target price criteria
    
    Args:
        stocks (dict): Dictionary of stock names and their prices
        target_price (float): Price threshold for analysis
        
    Returns:
        list: List of stocks meeting the criteria
    """
    filtered_stocks = []
    
    for stock_name, price in stocks.items():
        if price >= target_price:
            filtered_stocks.append(f"{stock_name} (₹{price})")
            
    return filtered_stocks

def monitor_stocks(duration_minutes: int = 5):
    """
    Monitors Indian stocks for a specified duration, checking prices every minute
    
    Args:
        duration_minutes (int): Duration to monitor in minutes
    """
    import random
    import time
    from datetime import datetime
    
    # Sample Indian stocks
    stock_names = ["RELIANCE", "TCS", "HDFC", "INFY", "ITC"]
    
    for minute in range(duration_minutes):
        current_time = datetime.now().strftime("%H:%M:%S")
        print(f"\nTime: {current_time}")
        print("-" * 40)
        
        # Generate random prices for stocks
        stocks_data = {
            stock: round(random.uniform(1000, 5000), 2)
            for stock in stock_names
        }
        
        # Analyze stocks above ₹2500
        high_value_stocks = analyze_market_data(stocks_data, 1000)
        
        # Print results
        print("Current Stock Prices:")
        for stock, price in stocks_data.items():
            print(f"{stock}: ₹{price}")
            
        print("\nStocks above ₹1000:")
        for stock in high_value_stocks:
            print(stock)
            
        if minute < duration_minutes - 1:
            time.sleep(5)  # Wait for 1 minute

if __name__ == "__main__":
    print("Starting Market Monitor...")
    monitor_stocks(5)  # Monitor for 5 minutes


# Assignment -  
# 1. Create a function to analyze multiple Indian stocks and return those meeting a target price criteria.
