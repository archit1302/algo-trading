"""
Solution for Assignment 2: Upstox Authentication
This script demonstrates direct access token authentication with Upstox API v3.
"""

import os
import sys
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class UpstoxAuthenticator:
    """
    Simple authentication handler for Upstox API v3
    """
    
    def __init__(self):
        """Initialize with API credentials"""
        self.api_key = os.getenv("UPSTOX_API_KEY")
        self.api_secret = os.getenv("UPSTOX_API_SECRET")
        self.redirect_uri = os.getenv("UPSTOX_REDIRECT_URI")
        self.access_token = os.getenv("UPSTOX_ACCESS_TOKEN")
        
        self.base_url = "https://api.upstox.com"
        self.token_file = "upstox_token.json"
        
        # Load saved token if available
        self.load_saved_token()
    
    def load_saved_token(self):
        """Load previously saved access token"""
        if os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'r') as f:
                    token_data = json.load(f)
                    self.access_token = token_data.get('access_token')
                    print("✅ Loaded saved access token")
                    return True
            except Exception as e:
                print(f"⚠️  Error loading saved token: {e}")
        return False
    
    def save_token(self, token_data):
        """Save access token to file"""
        try:
            save_data = {
                'access_token': token_data.get('access_token'),
                'token_type': token_data.get('token_type', 'Bearer'),
                'expires_in': token_data.get('expires_in'),
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.token_file, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"✅ Token saved to {self.token_file}")
            return True
            
        except Exception as e:
            print(f"❌ Error saving token: {e}")
            return False
    
    def get_authorization_url(self):
        """Generate authorization URL for OAuth flow"""
        auth_url = f"{self.base_url}/v2/login/authorization/dialog"
        
        params = {
            'response_type': 'code',
            'client_id': self.api_key,
            'redirect_uri': self.redirect_uri
        }
        
        # Build complete URL
        param_string = '&'.join([f"{k}={v}" for k, v in params.items()])
        complete_url = f"{auth_url}?{param_string}"
        
        return complete_url
    
    def exchange_auth_code(self, auth_code):
        """
        Exchange authorization code for access token
        
        Args:
            auth_code (str): Authorization code from redirect URL
        
        Returns:
            dict: Token response or None if failed
        """
        print(f"\n🔄 Exchanging authorization code for access token...")
        
        token_url = f"{self.base_url}/v2/login/authorization/token"
        
        payload = {
            'code': auth_code,
            'client_id': self.api_key,
            'client_secret': self.api_secret,
            'redirect_uri': self.redirect_uri,
            'grant_type': 'authorization_code'
        }
        
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/x-www-form-urlencoded'
        }
        
        try:
            response = requests.post(token_url, data=payload, headers=headers)
            response.raise_for_status()
            
            token_data = response.json()
            
            if 'access_token' in token_data:
                self.access_token = token_data['access_token']
                self.save_token(token_data)
                print("✅ Successfully obtained access token")
                return token_data
            else:
                print("❌ No access token in response")
                print(f"Response: {token_data}")
                return None
                
        except requests.RequestException as e:
            print(f"❌ Token exchange failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"Error details: {error_detail}")
                except:
                    print(f"Error response: {e.response.text}")
            return None
    
    def set_access_token_manually(self, token):
        """
        Set access token manually (for direct token approach)
        
        Args:
            token (str): Access token string
        """
        self.access_token = token
        
        # Save token data
        token_data = {
            'access_token': token,
            'token_type': 'Bearer',
            'manual_entry': True,
            'timestamp': datetime.now().isoformat()
        }
        
        self.save_token(token_data)
        print("✅ Access token set manually")
    
    def test_authentication(self):
        """
        Test if the current access token is valid
        
        Returns:
            bool: True if authentication is successful
        """
        if not self.access_token:
            print("❌ No access token available")
            return False
        
        print(f"\n🔍 Testing access token authentication...")
        
        # Test with user profile endpoint
        url = f"{self.base_url}/v2/user/profile"
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                user_data = response.json()
                print("✅ Authentication successful!")
                
                # Display user information
                if 'data' in user_data:
                    profile = user_data['data']
                    print(f"   👤 User: {profile.get('user_name', 'Unknown')}")
                    print(f"   📧 Email: {profile.get('email', 'Unknown')}")
                    print(f"   🏦 Broker: {profile.get('broker', 'Unknown')}")
                
                return True
                
            elif response.status_code == 401:
                print("❌ Authentication failed - Invalid or expired token")
                return False
            else:
                print(f"❌ Authentication test failed - Status: {response.status_code}")
                print(f"Response: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing authentication: {e}")
            return False
    
    def test_market_data_access(self):
        """
        Test access to market data endpoints
        
        Returns:
            bool: True if market data access is successful
        """
        if not self.access_token:
            print("❌ No access token for market data test")
            return False
        
        print(f"\n📊 Testing market data access...")
        
        # Test with a simple market quote request for SBIN
        url = f"{self.base_url}/v3/market-quote/NSE_EQ|INE062A01020|SBIN"
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.access_token}'
        }
        
        try:
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                quote_data = response.json()
                print("✅ Market data access successful!")
                
                # Display sample data
                if 'data' in quote_data:
                    data = quote_data['data']
                    if isinstance(data, dict):
                        ltp = data.get('ltp', 'N/A')
                        change = data.get('net_change', 'N/A')
                        print(f"   📈 SBIN LTP: ₹{ltp}")
                        print(f"   📊 Change: {change}")
                
                return True
                
            elif response.status_code == 401:
                print("❌ Market data access failed - Authentication issue")
                return False
            else:
                print(f"❌ Market data access failed - Status: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Error testing market data access: {e}")
            return False
    
    def interactive_setup(self):
        """
        Interactive setup process for authentication
        """
        print("🔐 Upstox Authentication Setup")
        print("=" * 40)
        
        print("\nChoose authentication method:")
        print("1. OAuth 2.0 Flow (Recommended)")
        print("2. Direct Access Token Entry")
        
        while True:
            choice = input("\nEnter your choice (1 or 2): ").strip()
            
            if choice == "1":
                self.oauth_flow_setup()
                break
            elif choice == "2":
                self.direct_token_setup()
                break
            else:
                print("❌ Invalid choice. Please enter 1 or 2.")
    
    def oauth_flow_setup(self):
        """Setup using OAuth 2.0 flow"""
        print("\n🔗 OAuth 2.0 Authentication Flow")
        print("-" * 35)
        
        # Generate authorization URL
        auth_url = self.get_authorization_url()
        
        print(f"\n📋 Step 1: Open this URL in your browser:")
        print(f"   {auth_url}")
        
        print(f"\n📋 Step 2: Login and authorize your application")
        print(f"📋 Step 3: Copy the authorization code from the redirect URL")
        
        # Get authorization code from user
        auth_code = input("\n🔑 Enter the authorization code: ").strip()
        
        if auth_code:
            # Exchange for access token
            token_data = self.exchange_auth_code(auth_code)
            
            if token_data:
                print("\n🎉 OAuth setup completed successfully!")
                return True
            else:
                print("\n❌ OAuth setup failed")
                return False
        else:
            print("\n❌ No authorization code provided")
            return False
    
    def direct_token_setup(self):
        """Setup using direct access token"""
        print("\n🔑 Direct Access Token Setup")
        print("-" * 30)
        
        print("📋 To get your access token:")
        print("   1. Login to your Upstox developer account")
        print("   2. Go to your app dashboard")
        print("   3. Generate or copy your access token")
        
        token = input("\n🔑 Enter your access token: ").strip()
        
        if token:
            self.set_access_token_manually(token)
            print("\n🎉 Direct token setup completed!")
            return True
        else:
            print("\n❌ No access token provided")
            return False
    
    def run_authentication_test(self):
        """
        Run complete authentication test suite
        """
        print("\n🧪 Running Authentication Tests")
        print("=" * 35)
        
        # Test basic authentication
        auth_success = self.test_authentication()
        
        if auth_success:
            # Test market data access
            market_success = self.test_market_data_access()
            
            print(f"\n📊 Test Results:")
            print(f"   Authentication: {'✅ PASS' if auth_success else '❌ FAIL'}")
            print(f"   Market Data: {'✅ PASS' if market_success else '❌ FAIL'}")
            
            if auth_success and market_success:
                print(f"\n🎉 All tests passed! You're ready to use Upstox API")
                return True
            else:
                print(f"\n⚠️  Some tests failed. Please check your setup.")
                return False
        else:
            print(f"\n❌ Authentication failed. Please check your token.")
            return False

def main():
    """
    Main function to demonstrate Upstox authentication
    """
    print("Upstox Authentication - Assignment 2 Solution")
    print("=" * 50)
    
    # Initialize authenticator
    auth = UpstoxAuthenticator()
    
    # Check if we already have a token
    if auth.access_token:
        print("🔍 Found existing access token, testing...")
        success = auth.run_authentication_test()
        
        if not success:
            print("\n🔄 Existing token failed, setting up new authentication...")
            auth.interactive_setup()
            auth.run_authentication_test()
    else:
        print("🚀 No access token found, starting setup...")
        auth.interactive_setup()
        auth.run_authentication_test()
    
    print("\n📚 Key Learnings:")
    print("   • Access tokens are required for all authenticated API calls")
    print("   • Tokens should be stored securely and not hardcoded")
    print("   • Always test authentication before implementing trading logic")
    print("   • Handle token expiration gracefully in production systems")

if __name__ == "__main__":
    main()
