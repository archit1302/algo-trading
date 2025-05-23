# Assignment 2: Simple Upstox API Authentication

In this assignment, you'll learn the simplest way to authenticate with Upstox API v3 using a direct access token approach. This is perfect for beginners who want to focus on learning the API without dealing with complex authentication flows.

## Objectives

1. Set up a simple authentication method for Upstox API
2. Create a reusable client class for making API calls
3. Test your authentication with basic API requests
4. Learn to handle authentication errors properly

## Tasks

### 1. Get Your Access Token from Upstox

**Step 1: Register on Upstox Developer Portal**
1. Go to [Upstox Developer Portal](https://developer.upstox.com/)
2. Create an account or log in
3. Create a new app to get your API credentials

**Step 2: Generate Access Token**
1. Go to your Upstox account settings
2. Navigate to the API section  
3. Generate a personal access token for development
4. Copy this token and save it securely

### 2. Set Up Your Project

Create this project structure:
```
upstox-simple-auth/
├── .env
├── upstox_client.py
├── test_auth.py
└── main.py
```

### 3. Create Environment File

Create a `.env` file to store your credentials securely:

```
# .env file
UPSTOX_ACCESS_TOKEN=your_actual_access_token_here
```

**Important**: Never put your actual token in your code files!

### 4. Create Simple Upstox Client

Create `upstox_client.py`:

```python
"""
Simple Upstox API Client for beginners
No complex OAuth flow - just use your access token directly!
"""
import os
import requests
from dotenv import load_dotenv

class SimpleUpstoxClient:
    """
    A beginner-friendly client for Upstox API v3
    """
    
    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()
        
        # Get access token from environment
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN")
        
        # Check if token is available
        if not self.access_token:
            raise ValueError("❌ Access token not found! Please set UPSTOX_ACCESS_TOKEN in your .env file")
        
        # API configuration
        self.base_url = "https://api.upstox.com/v3"
        self.headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        
        print("✅ Upstox client initialized successfully!")
    
    def test_connection(self):
        """Test if our authentication is working"""
        try:
            profile = self.get_user_profile()
            if profile:
                print("🎉 Authentication successful!")
                return True
            else:
                print("❌ Authentication failed!")
                return False
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
    
    def get_user_profile(self):
        """Get user profile information"""
        url = f"{self.base_url}/user/profile"
        
        try:
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 401:
                print("❌ Unauthorized! Your access token might be expired or invalid.")
                return None
            else:
                print(f"❌ Error getting profile: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return None
    
    def get_funds_and_margin(self):
        """Get user funds and margin information"""
        url = f"{self.base_url}/user/get-funds-and-margin"
        
        try:
            response = requests.get(url, headers=self.headers)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Error getting funds: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return None
    
    def get_historical_data(self, instrument_key, interval, to_date, from_date=None):
        """
        Get historical candle data for an instrument
        
        Args:
            instrument_key (str): Instrument identifier (e.g., "NSE_EQ|INE009A01021")
            interval (str): Time interval (e.g., "1minute", "5minute", "1day")
            to_date (str): End date in YYYY-MM-DD format
            from_date (str, optional): Start date in YYYY-MM-DD format
        """
        url = f"{self.base_url}/historical-candle/{instrument_key}/{interval}/{to_date}"
        
        # Add from_date as query parameter if provided
        params = {}
        if from_date:
            params['from'] = from_date
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"❌ Error getting historical data: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return None
```

### 5. Create Test Script

Create `test_auth.py` to verify your authentication works:

```python
"""
Test script to verify Upstox API authentication
"""
from upstox_client import SimpleUpstoxClient

def main():
    print("🚀 Testing Upstox API Authentication...")
    print("=" * 50)
    
    try:
        # Create client
        client = SimpleUpstoxClient()
        
        # Test connection
        if not client.test_connection():
            print("❌ Authentication test failed!")
            return
        
        # Get user profile
        print("\n📋 Getting user profile...")
        profile = client.get_user_profile()
        if profile and 'data' in profile:
            user_data = profile['data']
            print(f"👤 User Name: {user_data.get('user_name', 'N/A')}")
            print(f"📧 Email: {user_data.get('email', 'N/A')}")
            print(f"📱 Mobile: {user_data.get('mobile', 'N/A')}")
        
        # Get funds and margin
        print("\n💰 Getting funds and margin...")
        funds = client.get_funds_and_margin()
        if funds and 'data' in funds:
            equity_data = funds['data'].get('equity', {})
            print(f"💵 Available Margin: ₹{equity_data.get('available_margin', 'N/A')}")
            print(f"🏦 Used Margin: ₹{equity_data.get('used_margin', 'N/A')}")
        
        print("\n✅ All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")

if __name__ == "__main__":
    main()
```

### 6. Create Main Application

Create `main.py` for a complete example:

```python
"""
Main application demonstrating Upstox API usage
"""
from upstox_client import SimpleUpstoxClient
import json

def pretty_print(data, title="Data"):
    """Print data in a nice format"""
    print(f"\n{title}:")
    print("-" * len(title))
    print(json.dumps(data, indent=2))

def main():
    print("🏛️ Upstox API Demo Application")
    print("=" * 40)
    
    try:
        # Initialize client
        client = SimpleUpstoxClient()
        
        # Test authentication
        if not client.test_connection():
            return
        
        # Menu for different operations
        while True:
            print("\n📋 Choose an option:")
            print("1. Get User Profile")
            print("2. Get Funds and Margin")
            print("3. Get Historical Data (Example)")
            print("4. Exit")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                profile = client.get_user_profile()
                if profile:
                    pretty_print(profile, "User Profile")
            
            elif choice == "2":
                funds = client.get_funds_and_margin()
                if funds:
                    pretty_print(funds, "Funds and Margin")
            
            elif choice == "3":
                print("\n📈 Getting sample historical data for SBIN...")
                # Example: Get SBIN data for last day
                historical_data = client.get_historical_data(
                    instrument_key="NSE_EQ|INE062A01020",  # SBIN instrument key
                    interval="1minute",
                    to_date="2024-01-15"
                )
                if historical_data:
                    candles = historical_data.get('data', {}).get('candles', [])
                    print(f"📊 Received {len(candles)} candles")
                    if candles:
                        print("📈 Sample candle data:")
                        print(f"First candle: {candles[0]}")
            
            elif choice == "4":
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice! Please try again.")
    
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
```

### 7. Running Your Code

1. **Install required packages:**
   ```bash
   pip install requests python-dotenv
   ```

2. **Set up your .env file** with your actual access token

3. **Test authentication:**
   ```bash
   python test_auth.py
   ```

4. **Run the main application:**
   ```bash
   python main.py
   ```

## Expected Output

When you run `test_auth.py`, you should see something like:

```
🚀 Testing Upstox API Authentication...
==================================================
✅ Upstox client initialized successfully!
🎉 Authentication successful!

📋 Getting user profile...
👤 User Name: Your Name
📧 Email: your.email@example.com
📱 Mobile: +91XXXXXXXXXX

💰 Getting funds and margin...
💵 Available Margin: ₹50000.00
🏦 Used Margin: ₹0.00

✅ All tests completed successfully!
```

## Submission Guidelines

Submit a ZIP file containing:
1. All your code files (`upstox_client.py`, `test_auth.py`, `main.py`)
2. A sample `.env` file with placeholder values (not your actual token!)
3. A README.md explaining how to run your code

## Evaluation Criteria

1. **Correct implementation** of the authentication client
2. **Proper error handling** for various scenarios
3. **Security best practices** (using environment variables)
4. **Code organization** and comments
5. **Working demonstration** of API calls

## Troubleshooting

**Common Issues:**

1. **"Access token not found"** - Check your `.env` file and ensure it's in the same directory
2. **"401 Unauthorized"** - Your access token might be expired; generate a new one
3. **"Network error"** - Check your internet connection
4. **"Module not found"** - Make sure you've installed the required packages

**Getting Help:**
- Check the Upstox API documentation
- Verify your access token is valid
- Ensure your `.env` file is properly formatted

This simple approach will get you started with Upstox API without any complex authentication flows!
