"""
Quick script to test ALL Zoho tokens and find which ones work
"""
import os
import requests
from dotenv import load_dotenv

load_dotenv()

# Test all tokens
def test_token(token_num):
    client_id = os.getenv(f"ZOHO_CLIENT_ID_{token_num}")
    client_secret = os.getenv(f"ZOHO_CLIENT_SECRET_{token_num}")
    refresh_token = os.getenv(f"ZOHO_REFRESH_TOKEN_{token_num}")
    
    if not all([client_id, client_secret, refresh_token]):
        return None, f"Token{token_num}: Missing credentials"
    
    try:
        # Get access token
        token_url = "https://accounts.zoho.com/oauth/v2/token"
        params = {
            "refresh_token": refresh_token,
            "client_id": client_id,
            "client_secret": client_secret,
            "grant_type": "refresh_token"
        }
        
        token_response = requests.post(token_url, params=params, timeout=10)
        
        if token_response.status_code != 200:
            return None, f"Token{token_num}: Refresh failed - {token_response.text[:100]}"
        
        access_token = token_response.json().get("access_token")
        
        if not access_token:
            return None, f"Token{token_num}: No access token in response"
        
        # Test the token
        api_url = "https://creator.zoho.com/api/v2.1/teameverest/iatc-scholarship/report/Scholar_Fee_Request_OCR_View"
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
        params = {"from": 1, "limit": 1}
        
        test_response = requests.get(api_url, headers=headers, params=params, timeout=30)
        
        if test_response.status_code == 200:
            data = test_response.json()
            if data.get("data"):
                return access_token, f"Token{token_num}: ✅ VALID - Got {len(data['data'])} records"
            else:
                return None, f"Token{token_num}: ⚠️ Valid but no data returned"
        else:
            return None, f"Token{token_num}: ❌ API call failed - {test_response.status_code}"
            
    except Exception as e:
        return None, f"Token{token_num}: ❌ Error - {str(e)}"

print("\n" + "="*80)
print("TESTING ALL ZOHO TOKENS")
print("="*80)

working_tokens = []

for i in range(1, 19):  # Test tokens 1-18
    access_token, message = test_token(i)
    print(message)
    if access_token:
        working_tokens.append((i, access_token))

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Working tokens: {len(working_tokens)}")

if working_tokens:
    print("\n✅ Using first working token to fetch record fields...")
    token_num, access_token = working_tokens[0]
    
    api_url = "https://creator.zoho.com/api/v2.1/teameverest/iatc-scholarship/report/Scholar_Fee_Request_OCR_View"
    headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
    params = {"from": 1, "limit": 1}
    
    response = requests.get(api_url, headers=headers, params=params, timeout=30)
    data = response.json()
    
    if data.get("data"):
        record = data["data"][0]
        print("\n" + "="*80)
        print("ZOHO RECORD FIELDS")
        print("="*80)
        
        # Show all fields
        for field_name, field_value in record.items():
            print(f"\n{field_name}:")
            print(f"  Type: {type(field_value).__name__}")
            print(f"  Value: {str(field_value)[:200]}")
            
            # Check if it looks like an image field
            if isinstance(field_value, (str, list, dict)):
                value_str = str(field_value)
                if 'http' in value_str or 'download_url' in value_str or '/api/v2.1/' in value_str:
                    print(f"  ⚠️ LOOKS LIKE IMAGE FIELD!")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Total fields: {len(record.keys())}")
        print(f"\nAll field names:")
        for name in sorted(record.keys()):
            print(f"  - {name}")
else:
    print("\n❌ NO WORKING TOKENS FOUND!")
    print("You need to regenerate your Zoho refresh tokens.")
