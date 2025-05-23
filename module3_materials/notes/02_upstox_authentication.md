# Simple Authentication with Upstox API v3

This note covers the easiest way to authenticate with Upstox API v3 for beginners. We'll use a direct access token approach that's perfect for learning and development.

## What is Authentication?

Authentication is like showing your ID card to prove who you are. When you want to use Upstox API to get market data or place orders, you need to prove to Upstox that you have permission to access their services.

## Simple Authentication Approach

For beginners, the easiest way to authenticate with Upstox API is to:

1. **Get an API Key** from Upstox Developer Portal
2. **Generate an Access Token** manually from your Upstox account  
3. **Use the Access Token** in your Python code

This approach avoids complex OAuth flows and lets you focus on learning the API functionality.

## Step-by-Step Setup

### Step 1: Register on Upstox Developer Portal

1. Go to [Upstox Developer Portal](https://developer.upstox.com/)
2. Create an account or log in with your existing Upstox account
3. Create a new app to get your API credentials

### Step 2: Get Your Access Token

The easiest way to get an access token for development:

1. Go to your Upstox account settings
2. Navigate to the API section
3. Generate a personal access token for development
4. Copy this token and save it securely

### Step 3: Create Your First API Call

Here's a simple example that uses your access token directly:

```python
import requests

# Your access token (keep this secure!)
access_token = "YOUR_ACCESS_TOKEN_HERE"

# Upstox API base URL for v3
base_url = "https://api.upstox.com/v3"

# Headers for authentication
headers = {
    'accept': 'application/json',
    'Authorization': f'Bearer {access_token}'
}

# Example: Get user profile
def get_user_profile():
    url = f"{base_url}/user/profile"
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

# Call the function
profile_data = get_user_profile()
if profile_data:
    print("User Profile:", profile_data)
```

## Environment Variables for Security

Never put your access token directly in your code! Instead, use environment variables:

### Step 1: Create a .env file

Create a file named `.env` in your project folder:

```
UPSTOX_ACCESS_TOKEN=your_actual_access_token_here
```

### Step 2: Use python-dotenv to load it

```python
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get access token from environment
access_token = os.getenv("UPSTOX_ACCESS_TOKEN")

# Headers for authentication
headers = {
    'accept': 'application/json',
    'Authorization': f'Bearer {access_token}'
}

# Now you can make API calls safely
def get_user_profile():
    url = "https://api.upstox.com/v3/user/profile"
    response = requests.get(url, headers=headers)
    return response.json() if response.status_code == 200 else None
```

## Complete Simple Example

Here's a complete working example that follows best practices:

```python
"""
Simple Upstox API v3 Authentication Example
This example shows how to authenticate and make basic API calls
"""
import os
import requests
from dotenv import load_dotenv

class SimpleUpstoxClient:
    """
    A simple client for Upstox API v3
    """
    
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Get access token from environment
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN")
        
        if not self.access_token:
            raise ValueError("Access token not found! Please set UPSTOX_ACCESS_TOKEN in your .env file")
        
        # Set up headers
        self.headers = {
            'accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        
        # API base URL
        self.base_url = "https://api.upstox.com/v3"
    
    def get_user_profile(self):
        """Get user profile information"""
        url = f"{self.base_url}/user/profile"
        response = requests.get(url, headers=self.headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting profile: {response.status_code} - {response.text}")
            return None
    
    def get_historical_data(self, instrument_key, interval, to_date, from_date=None):
        """Get historical candle data"""
        url = f"{self.base_url}/historical-candle/{instrument_key}/{interval}/{to_date}"
        
        # Add from_date if provided
        params = {}
        if from_date:
            params['from'] = from_date
        
        response = requests.get(url, headers=self.headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error getting historical data: {response.status_code} - {response.text}")
            return None

# Example usage
if __name__ == "__main__":
    # Create client
    client = SimpleUpstoxClient()
    
    # Test authentication by getting profile
    profile = client.get_user_profile()
    if profile:
        print("Authentication successful!")
        print(f"User: {profile.get('data', {}).get('user_name', 'Unknown')}")
    else:
        print("Authentication failed!")
```

## What You Need to Remember

1. **Get your access token** from Upstox account settings
2. **Store it in a .env file** for security
3. **Use it in the Authorization header** for all API calls
4. **Never share or commit your token** to version control

## Token Validity

- Access tokens are valid for a limited time (usually 24 hours)
- When your token expires, you'll get a 401 Unauthorized error
- Simply generate a new token from your Upstox account when needed

## Next Steps

Once you have authentication working:
1. Learn about instrument mapping (finding stock symbols)
2. Fetch historical data for analysis
3. Process and analyze the data with pandas
4. Build trading strategies and scanners

This simple approach will get you started quickly without dealing with complex OAuth flows!
