"""
Assignment 4 Solution: Functions for Financial Calculations
Complete solution for creating reusable financial functions

Author: GitHub Copilot
Module: 1 - Python Fundamentals
Assignment: 4 - Functions
"""

import math
from datetime import datetime, timedelta

# Function 1: Calculate Simple Interest
def calculate_simple_interest(principal, rate, time):
    """
    Calculate simple interest for investment
    
    Args:
        principal (float): Initial investment amount
        rate (float): Annual interest rate (as percentage)
        time (float): Time period in years
    
    Returns:
        dict: Contains principal, interest, and total amount
    """
    try:
        # Validate inputs
        if principal <= 0 or rate < 0 or time <= 0:
            raise ValueError("Principal and time must be positive, rate must be non-negative")
        
        # Calculate simple interest
        interest = (principal * rate * time) / 100
        total_amount = principal + interest
        
        return {
            'principal': principal,
            'rate': rate,
            'time': time,
            'interest': interest,
            'total_amount': total_amount
        }
    
    except (TypeError, ValueError) as e:
        print(f"Error in simple interest calculation: {e}")
        return None

# Function 2: Calculate Compound Interest
def calculate_compound_interest(principal, rate, time, compound_frequency=1):
    """
    Calculate compound interest for investment
    
    Args:
        principal (float): Initial investment amount
        rate (float): Annual interest rate (as percentage)
        time (float): Time period in years
        compound_frequency (int): Number of times compounded per year
    
    Returns:
        dict: Contains all calculation details
    """
    try:
        # Validate inputs
        if principal <= 0 or rate < 0 or time <= 0 or compound_frequency <= 0:
            raise ValueError("Invalid input values")
        
        # Calculate compound interest
        # A = P(1 + r/n)^(nt)
        rate_decimal = rate / 100
        amount = principal * (1 + rate_decimal/compound_frequency) ** (compound_frequency * time)
        compound_interest = amount - principal
        
        return {
            'principal': principal,
            'rate': rate,
            'time': time,
            'compound_frequency': compound_frequency,
            'compound_interest': compound_interest,
            'final_amount': amount,
            'difference_from_simple': compound_interest - ((principal * rate * time) / 100)
        }
    
    except (TypeError, ValueError) as e:
        print(f"Error in compound interest calculation: {e}")
        return None

# Function 3: Calculate Percentage Change
def calculate_percentage_change(old_value, new_value):
    """
    Calculate percentage change between two values (useful for stock price changes)
    
    Args:
        old_value (float): Original value
        new_value (float): New value
    
    Returns:
        float: Percentage change
    """
    try:
        if old_value == 0:
            raise ValueError("Old value cannot be zero")
        
        percentage_change = ((new_value - old_value) / old_value) * 100
        return round(percentage_change, 2)
    
    except (TypeError, ValueError, ZeroDivisionError) as e:
        print(f"Error in percentage change calculation: {e}")
        return None

# Function 4: Portfolio Value Calculator
def calculate_portfolio_value(stocks):
    """
    Calculate total portfolio value from list of stocks
    
    Args:
        stocks (list): List of dictionaries with stock data
                      Each dict should have: 'symbol', 'quantity', 'price'
    
    Returns:
        dict: Portfolio summary
    """
    try:
        if not stocks or not isinstance(stocks, list):
            raise ValueError("Stocks must be a non-empty list")
        
        total_value = 0
        stock_values = []
        
        for stock in stocks:
            if not all(key in stock for key in ['symbol', 'quantity', 'price']):
                raise ValueError("Each stock must have symbol, quantity, and price")
            
            stock_value = stock['quantity'] * stock['price']
            total_value += stock_value
            
            stock_values.append({
                'symbol': stock['symbol'],
                'quantity': stock['quantity'],
                'price': stock['price'],
                'value': stock_value,
                'percentage': 0  # Will calculate after total is known
            })
        
        # Calculate percentage allocation
        for stock in stock_values:
            stock['percentage'] = round((stock['value'] / total_value) * 100, 2)
        
        return {
            'total_value': total_value,
            'stock_count': len(stocks),
            'stocks': stock_values,
            'largest_holding': max(stock_values, key=lambda x: x['value']),
            'smallest_holding': min(stock_values, key=lambda x: x['value'])
        }
    
    except (TypeError, ValueError) as e:
        print(f"Error in portfolio calculation: {e}")
        return None

# Function 5: SIP Calculator
def calculate_sip_returns(monthly_investment, rate, years):
    """
    Calculate Systematic Investment Plan (SIP) returns
    
    Args:
        monthly_investment (float): Monthly investment amount
        rate (float): Expected annual return rate (as percentage)
        years (int): Investment period in years
    
    Returns:
        dict: SIP calculation results
    """
    try:
        if monthly_investment <= 0 or rate < 0 or years <= 0:
            raise ValueError("Invalid input values")
        
        # Convert annual rate to monthly rate
        monthly_rate = (rate / 12) / 100
        total_months = years * 12
        
        # SIP formula: M * [((1 + r)^n - 1) / r] * (1 + r)
        if monthly_rate == 0:
            future_value = monthly_investment * total_months
        else:
            future_value = monthly_investment * (((1 + monthly_rate) ** total_months - 1) / monthly_rate) * (1 + monthly_rate)
        
        total_invested = monthly_investment * total_months
        total_returns = future_value - total_invested
        
        return {
            'monthly_investment': monthly_investment,
            'annual_rate': rate,
            'years': years,
            'total_invested': total_invested,
            'future_value': round(future_value, 2),
            'total_returns': round(total_returns, 2),
            'return_percentage': round((total_returns / total_invested) * 100, 2)
        }
    
    except (TypeError, ValueError) as e:
        print(f"Error in SIP calculation: {e}")
        return None

# Function 6: Risk Assessment
def assess_portfolio_risk(stocks_data):
    """
    Assess portfolio risk based on stock volatility
    
    Args:
        stocks_data (list): List of dicts with 'symbol' and 'volatility' (standard deviation)
    
    Returns:
        dict: Risk assessment
    """
    try:
        if not stocks_data:
            raise ValueError("Stock data cannot be empty")
        
        volatilities = [stock['volatility'] for stock in stocks_data]
        avg_volatility = sum(volatilities) / len(volatilities)
        
        # Risk categories based on volatility
        if avg_volatility < 15:
            risk_level = "Low"
        elif avg_volatility < 25:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        return {
            'average_volatility': round(avg_volatility, 2),
            'risk_level': risk_level,
            'highest_risk_stock': max(stocks_data, key=lambda x: x['volatility']),
            'lowest_risk_stock': min(stocks_data, key=lambda x: x['volatility']),
            'recommendation': get_risk_recommendation(risk_level)
        }
    
    except (TypeError, ValueError) as e:
        print(f"Error in risk assessment: {e}")
        return None

def get_risk_recommendation(risk_level):
    """Helper function to provide risk-based recommendations"""
    recommendations = {
        "Low": "Conservative portfolio suitable for risk-averse investors",
        "Medium": "Balanced portfolio with moderate risk-reward ratio",
        "High": "Aggressive portfolio suitable for risk-tolerant investors"
    }
    return recommendations.get(risk_level, "Unable to provide recommendation")

# Example usage and testing
def main():
    """Demonstrate all functions with sample data"""
    print("=== Assignment 4: Functions for Financial Calculations ===\n")
    
    # Test 1: Simple Interest
    print("1. Simple Interest Calculation:")
    si_result = calculate_simple_interest(100000, 8.5, 2)
    if si_result:
        print(f"Principal: ₹{si_result['principal']:,.2f}")
        print(f"Interest: ₹{si_result['interest']:,.2f}")
        print(f"Total Amount: ₹{si_result['total_amount']:,.2f}")
    print()
    
    # Test 2: Compound Interest
    print("2. Compound Interest Calculation:")
    ci_result = calculate_compound_interest(100000, 8.5, 2, 4)  # Quarterly compounding
    if ci_result:
        print(f"Principal: ₹{ci_result['principal']:,.2f}")
        print(f"Compound Interest: ₹{ci_result['compound_interest']:,.2f}")
        print(f"Final Amount: ₹{ci_result['final_amount']:,.2f}")
        print(f"Advantage over Simple Interest: ₹{ci_result['difference_from_simple']:,.2f}")
    print()
    
    # Test 3: Percentage Change
    print("3. Stock Price Change:")
    old_price = 850.50
    new_price = 892.75
    change = calculate_percentage_change(old_price, new_price)
    if change is not None:
        print(f"SBIN: ₹{old_price} → ₹{new_price}")
        print(f"Change: {change:+.2f}%")
    print()
    
    # Test 4: Portfolio Value
    print("4. Portfolio Calculation:")
    portfolio = [
        {'symbol': 'SBIN', 'quantity': 100, 'price': 892.75},
        {'symbol': 'RELIANCE', 'quantity': 50, 'price': 2456.30},
        {'symbol': 'TCS', 'quantity': 25, 'price': 3234.85}
    ]
    portfolio_result = calculate_portfolio_value(portfolio)
    if portfolio_result:
        print(f"Total Portfolio Value: ₹{portfolio_result['total_value']:,.2f}")
        print("Holdings:")
        for stock in portfolio_result['stocks']:
            print(f"  {stock['symbol']}: ₹{stock['value']:,.2f} ({stock['percentage']}%)")
    print()
    
    # Test 5: SIP Calculator
    print("5. SIP Returns Calculation:")
    sip_result = calculate_sip_returns(10000, 12, 5)
    if sip_result:
        print(f"Monthly Investment: ₹{sip_result['monthly_investment']:,.2f}")
        print(f"Total Invested: ₹{sip_result['total_invested']:,.2f}")
        print(f"Future Value: ₹{sip_result['future_value']:,.2f}")
        print(f"Total Returns: ₹{sip_result['total_returns']:,.2f}")
        print(f"Return %: {sip_result['return_percentage']}%")
    print()
    
    # Test 6: Risk Assessment
    print("6. Portfolio Risk Assessment:")
    risk_data = [
        {'symbol': 'SBIN', 'volatility': 18.5},
        {'symbol': 'RELIANCE', 'volatility': 22.3},
        {'symbol': 'TCS', 'volatility': 16.8}
    ]
    risk_result = assess_portfolio_risk(risk_data)
    if risk_result:
        print(f"Average Volatility: {risk_result['average_volatility']}%")
        print(f"Risk Level: {risk_result['risk_level']}")
        print(f"Highest Risk: {risk_result['highest_risk_stock']['symbol']} ({risk_result['highest_risk_stock']['volatility']}%)")
        print(f"Recommendation: {risk_result['recommendation']}")

if __name__ == "__main__":
    main()
