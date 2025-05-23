# Assignment 1: Upstox API Basics

In this assignment, you will get started with Upstox API v3. You'll set up your development environment, register for API access, and make your first API requests.

## Objectives

1. Set up development environment for Upstox API integration
2. Register for Upstox API access
3. Make basic API requests to test connectivity

## Tasks

### 1. Setup Development Environment

1. Create a virtual environment for your project:
   ```bash
   python -m venv upstox-env
   source upstox-env/bin/activate  # On Windows: upstox-env\Scripts\activate
   ```

2. Install required packages:
   ```bash
   pip install requests python-dotenv pandas
   ```

3. Create a project structure:
   ```
   upstox-project/
   ├── .env
   ├── config.py
   ├── api_client.py
   └── main.py
   ```

### 2. Register for Upstox API Access

1. Visit [Upstox Developer Portal](https://developer.upstox.com/) and create an account if you haven't already.
2. Create a new application to obtain API Key and API Secret.
3. Set redirect URI to `http://localhost:5000/callback` (for development purposes).
4. Store your credentials in the `.env` file:
   ```
   UPSTOX_API_KEY=your_api_key
   UPSTOX_API_SECRET=your_api_secret
   UPSTOX_REDIRECT_URI=http://localhost:5000/callback
   ```

### 3. Create a Basic API Client

Create `api_client.py` with a basic structure for making API calls:

```python
import os
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class UpstoxAPIClient:
    """Basic Upstox API v3 client"""
    
    BASE_URL = "https://api.upstox.com/v3"
    
    def __init__(self):
        self.api_key = os.getenv("UPSTOX_API_KEY")
        self.api_secret = os.getenv("UPSTOX_API_SECRET")
        self.redirect_uri = os.getenv("UPSTOX_REDIRECT_URI")
        self.auth_url = f"https://api.upstox.com/v2/login/authorization/dialog"
        self.token_url = f"https://api.upstox.com/v2/login/authorization/token"
        self.access_token = None
    
    def get_auth_url(self):
        """Generate authorization URL for OAuth flow"""
        auth_url = f"{self.auth_url}?response_type=code&client_id={self.api_key}&redirect_uri={self.redirect_uri}"
        return auth_url
    
    def set_access_token(self, token):
        """Set the access token for API calls"""
        self.access_token = token
    
    def get_headers(self):
        """Get headers for API requests"""
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.access_token}" if self.access_token else None
        }
        return {k: v for k, v in headers.items() if v is not None}
```

### 4. Create Configuration Module

Create a `config.py` file for configuration settings:

```python
"""
Configuration settings for Upstox API connection
"""

# API Configuration
API_VERSION = "v3"
RETRY_COUNT = 3
RETRY_DELAY = 1  # seconds

# Market Data Defaults
DEFAULT_EXCHANGE = "NSE"
DEFAULT_SYMBOL = "SBIN"
DEFAULT_TIMEFRAME = "1minute"  # Format: {interval}{unit}
```

### 5. Create Main Entry Point

Create a `main.py` file to test API connectivity:

```python
"""
Main entry point for testing Upstox API connectivity
"""
from api_client import UpstoxAPIClient

def main():
    # Initialize API client
    client = UpstoxAPIClient()
    
    # Generate authorization URL
    auth_url = client.get_auth_url()
    
    print("=" * 80)
    print("Upstox API Setup")
    print("=" * 80)
    print(f"\nAPI Key: {client.api_key[:4]}{'*' * 8}")
    print(f"Redirect URI: {client.redirect_uri}")
    print("\nTo authorize this application, please visit:")
    print(auth_url)
    print("\nAfter authorization, you will be redirected to your redirect URI with a 'code' parameter.")
    print("You will need this code to generate an access token in future assignments.")
    
if __name__ == "__main__":
    main()
```

## Submission Guidelines

Create a ZIP file containing all the code files you've created, ensuring that you:
1. Have not included your actual API credentials in the files (use `.env` for local storage)
2. Have followed the project structure as described
3. Include a brief README.md explaining how to run your code

## Evaluation Criteria

1. Code structure and organization
2. Proper implementation of environment variable handling
3. Implementation of basic API client structure
4. Documentation and comments

## Helpful Resources

- [Upstox API Documentation](https://upstox.com/developer/api-documentation/v3)
- [OAuth 2.0 Authentication Flow](https://upstox.com/developer/api-documentation/authentication)
- [Python dotenv documentation](https://pypi.org/project/python-dotenv/)
