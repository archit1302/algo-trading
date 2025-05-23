"""
Solution for Assignment 1: Upstox API Basics
This script demonstrates basic Upstox API setup and environment configuration.
"""

import os
import sys
import json
import requests
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class UpstoxBasics:
    """
    Basic Upstox API utilities for getting started
    """
    
    def __init__(self):
        """Initialize with API credentials from environment"""
        self.api_key = os.getenv("UPSTOX_API_KEY")
        self.api_secret = os.getenv("UPSTOX_API_SECRET")
        self.redirect_uri = os.getenv("UPSTOX_REDIRECT_URI")
        self.base_url = "https://api.upstox.com/v3"
        
        # Validate credentials
        self.validate_credentials()
    
    def validate_credentials(self):
        """Validate that all required credentials are available"""
        required_vars = ['UPSTOX_API_KEY', 'UPSTOX_API_SECRET', 'UPSTOX_REDIRECT_URI']
        missing_vars = []
        
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            print("❌ Missing environment variables:")
            for var in missing_vars:
                print(f"   - {var}")
            print("\nPlease set these variables in your .env file")
            return False
        else:
            print("✅ All required environment variables are set")
            return True
    
    def check_api_connectivity(self):
        """
        Test basic API connectivity using a public endpoint
        """
        print("\n🌐 Testing API connectivity...")
        
        try:
            # Test with the master contract endpoint (public)
            url = f"{self.base_url}/master"
            
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print("✅ API connectivity successful")
                
                # Check response format
                try:
                    data = response.json()
                    if isinstance(data, list) and len(data) > 0:
                        print(f"✅ Received {len(data)} instruments in master contract")
                        
                        # Show sample instrument
                        sample = data[0]
                        print("\n📄 Sample instrument data:")
                        for key, value in sample.items():
                            print(f"   {key}: {value}")
                    
                    return True
                    
                except json.JSONDecodeError:
                    print("⚠️  API responded but data format is unexpected")
                    return False
            else:
                print(f"❌ API connectivity failed. Status code: {response.status_code}")
                return False
                
        except requests.RequestException as e:
            print(f"❌ Network error: {e}")
            return False
    
    def get_authorization_url(self):
        """
        Generate the authorization URL for OAuth 2.0 flow
        """
        print("\n🔗 Generating authorization URL...")
        
        auth_url = "https://api.upstox.com/v2/login/authorization/dialog"
        params = {
            'response_type': 'code',
            'client_id': self.api_key,
            'redirect_uri': self.redirect_uri
        }
        
        # Build URL with parameters
        param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        full_url = f"{auth_url}?{param_string}"
        
        print("🔐 Authorization URL generated:")
        print(f"   {full_url}")
        print("\n📋 Steps to get authorization:")
        print("   1. Copy the URL above")
        print("   2. Open it in your web browser")
        print("   3. Login to your Upstox account")
        print("   4. Grant permissions to your app")
        print("   5. Copy the authorization code from the redirect URL")
        
        return full_url
    
    def demonstrate_rate_limits(self):
        """
        Demonstrate understanding of rate limits
        """
        print("\n⏱️  Rate Limiting Information:")
        print("   • Upstox API v3 has rate limits to prevent abuse")
        print("   • Typical limits: 10 requests per second")
        print("   • Historical data: Limited requests per minute")
        print("   • Always implement delays between requests")
        print("   • Use exponential backoff for retries")
        print("   • Monitor 'X-RateLimit-*' headers in responses")
    
    def show_api_endpoints(self):
        """
        Display information about key API endpoints
        """
        print("\n🔗 Key Upstox API v3 Endpoints:")
        
        endpoints = {
            "Authentication": [
                "POST /v2/login/authorization/token - Get access token",
                "GET /v2/user/profile - Get user profile"
            ],
            "Market Data": [
                "GET /v3/master - Get master contracts",
                "GET /v3/historical-candle/{key}/{unit}/{interval} - Historical data",
                "GET /v3/market-quote - Get market quotes"
            ],
            "Trading": [
                "POST /v2/order/place - Place order",
                "GET /v2/order - Get orders",
                "GET /v2/portfolio/positions - Get positions"
            ]
        }
        
        for category, endpoint_list in endpoints.items():
            print(f"\n   📁 {category}:")
            for endpoint in endpoint_list:
                print(f"      • {endpoint}")
    
    def create_sample_env_file(self):
        """
        Create a sample .env file for reference
        """
        sample_content = """# Upstox API Configuration
# Get these values from https://developer.upstox.com/

UPSTOX_API_KEY=your_api_key_here
UPSTOX_API_SECRET=your_api_secret_here
UPSTOX_REDIRECT_URI=http://localhost:8080/callback

# Optional: For production use
# UPSTOX_ACCESS_TOKEN=your_access_token_here
"""
        
        env_file = ".env.sample"
        
        try:
            with open(env_file, 'w') as f:
                f.write(sample_content)
            
            print(f"\n📝 Sample environment file created: {env_file}")
            print("   Copy this to '.env' and fill in your actual credentials")
            
        except Exception as e:
            print(f"❌ Error creating sample file: {e}")
    
    def run_diagnostics(self):
        """
        Run complete diagnostic check
        """
        print("🔍 Running Upstox API Diagnostics")
        print("=" * 50)
        
        # Check credentials
        creds_ok = self.validate_credentials()
        
        if creds_ok:
            # Check API connectivity
            api_ok = self.check_api_connectivity()
            
            # Show authorization process
            self.get_authorization_url()
            
            # Show rate limiting info
            self.demonstrate_rate_limits()
            
            # Show available endpoints
            self.show_api_endpoints()
            
            print("\n" + "=" * 50)
            print("🎉 Diagnostics completed!")
            
            if api_ok:
                print("✅ System is ready for Upstox API development")
            else:
                print("⚠️  Some issues detected. Please check network connectivity")
        else:
            print("\n" + "=" * 50)
            print("❌ Please fix credential issues before proceeding")
            self.create_sample_env_file()

def main():
    """
    Main function to demonstrate Upstox API basics
    """
    print("Upstox API Basics - Assignment 1 Solution")
    print("=" * 50)
    
    # Initialize the basics utility
    upstox_basics = UpstoxBasics()
    
    # Run diagnostics
    upstox_basics.run_diagnostics()
    
    # Additional demonstrations
    print("\n📚 Additional Information:")
    print("   • Keep your API credentials secure")
    print("   • Never commit credentials to version control")
    print("   • Use environment variables for configuration")
    print("   • Test connectivity before implementing trading logic")
    print("   • Read the official documentation regularly")
    
    print("\n🔗 Useful Resources:")
    print("   • Upstox Developer Portal: https://developer.upstox.com/")
    print("   • API Documentation: https://upstox.com/developer/api-documentation/")
    print("   • Python SDK: https://github.com/upstox/upstox-python")

if __name__ == "__main__":
    main()
