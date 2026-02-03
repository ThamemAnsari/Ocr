"""
ONE-TIME SETUP: Run this script to generate initial OAuth tokens
python zoho_oauth_setup.py
"""

import requests
import json
from urllib.parse import urlencode

# Your credentials from Zoho API Console
CLIENT_ID = "1000.K79W0PP1ZGEUW4QV5KNHU2IHEC62XX"
CLIENT_SECRET = "a0d0132fb75462c88c35701542cd5d2acc2091a868"
REDIRECT_URI = "https://3e7064dbd961.ngrok-free.app/oauth/callback"

# Step 1: Generate authorization URL
def get_auth_url():
    params = {
        'scope': 'ZohoCreator.form.CREATE',  # ← CHANGED: Form scope instead of report
        'client_id': CLIENT_ID,
        'response_type': 'code',
        'access_type': 'offline',
        'redirect_uri': REDIRECT_URI,
        'prompt': 'consent'
    }
    
    auth_url = f"https://accounts.zoho.com/oauth/v2/auth?{urlencode(params)}"
    
    print("="*80)
    print("STEP 1: AUTHORIZE APPLICATION")
    print("="*80)
    print("\n1. Open this URL in your browser:")
    print(f"\n{auth_url}\n")
    print("2. Authorize the application")
    print("3. Copy the 'code' parameter from the redirect URL")
    print("4. IMPORTANT: Use the code immediately (expires in 60 seconds)")
    print("="*80)

# Step 2: Exchange code for tokens
def get_tokens(code):
    url = "https://accounts.zoho.com/oauth/v2/token"
    
    data = {
        'code': code,
        'client_id': CLIENT_ID,
        'client_secret': CLIENT_SECRET,
        'redirect_uri': REDIRECT_URI,
        'grant_type': 'authorization_code'
    }
    
    try:
        response = requests.post(url, data=data, timeout=30)
        
        print("\n" + "="*80)
        print("API RESPONSE")
        print("="*80)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        print("="*80 + "\n")
        
        if response.status_code == 200:
            tokens = response.json()
            
            # Check if we got a refresh token
            if 'refresh_token' not in tokens:
                print("⚠️  WARNING: No refresh_token received!")
                print("\nThis usually happens because:")
                print("1. The authorization code was already used")
                print("2. The code expired (valid for only 60 seconds)")
                print("3. Missing 'access_type=offline' parameter")
                print("\n🔄 SOLUTION: Generate a NEW authorization URL and try again\n")
                
                # Show what we did receive
                print("Tokens received:")
                for key, value in tokens.items():
                    print(f"  - {key}: {value[:50]}..." if len(str(value)) > 50 else f"  - {key}: {value}")
                
                # If we got an access token, we can still use it temporarily
                if 'access_token' in tokens:
                    print("\n✓ You have an access_token (temporary)")
                    print("But you NEED a refresh_token for long-term use!")
                    print("\n🔄 Please run this script again to get refresh_token")
                
                return False
            
            # Success - we have everything
            print("\n" + "="*80)
            print("✓ TOKENS GENERATED SUCCESSFULLY!")
            print("="*80)
            print("\nAdd these to your .env file:\n")
            print(f"ZOHO_CLIENT_ID={CLIENT_ID}")
            print(f"ZOHO_CLIENT_SECRET={CLIENT_SECRET}")
            print(f"ZOHO_REFRESH_TOKEN={tokens['refresh_token']}")
            print(f"ZOHO_ACCESS_TOKEN={tokens['access_token']}")
            print(f"\nZOHO_OWNER_NAME=your_zoho_username")
            print(f"ZOHO_APP_LINK_NAME=your_app_link_name")
            print(f"ZOHO_REPORT_LINK_NAME=Extraction_Results")
            print("="*80)
            
            # Save to file
            with open('.env.zoho', 'w') as f:
                f.write(f"ZOHO_CLIENT_ID={CLIENT_ID}\n")
                f.write(f"ZOHO_CLIENT_SECRET={CLIENT_SECRET}\n")
                f.write(f"ZOHO_REFRESH_TOKEN={tokens['refresh_token']}\n")
                f.write(f"ZOHO_ACCESS_TOKEN={tokens['access_token']}\n")
                f.write(f"\nZOHO_OWNER_NAME=your_zoho_username\n")
                f.write(f"ZOHO_APP_LINK_NAME=your_app_link_name\n")
                f.write(f"ZOHO_REPORT_LINK_NAME=Extraction_Results\n")
            
            print("\n✓ Tokens saved to .env.zoho file")
            print("Copy these values to your main .env file\n")
            
            return True
            
        else:
            print(f"\n✗ Error: {response.text}")
            print("\nCommon errors:")
            print("- 'invalid_code': Code already used or expired (get a new one)")
            print("- 'invalid_client': Check CLIENT_ID and CLIENT_SECRET")
            return False
            
    except Exception as e:
        print(f"\n✗ Exception: {str(e)}")
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("ZOHO CREATOR OAUTH SETUP")
    print("="*80 + "\n")
    
    get_auth_url()
    
    print("\nEnter the authorization code: ", end="")
    code = input().strip()
    
    if code:
        success = get_tokens(code)
        
        if not success:
            print("\n" + "="*80)
            print("RETRY INSTRUCTIONS")
            print("="*80)
            print("\n1. Run this script again: python zoho_oauth_setup.py")
            print("2. Use the NEW authorization URL it generates")
            print("3. Complete authorization QUICKLY (code expires in 60s)")
            print("4. Paste the NEW code immediately")
            print("\n" + "="*80 + "\n")
    else:
        print("No code provided. Exiting.")