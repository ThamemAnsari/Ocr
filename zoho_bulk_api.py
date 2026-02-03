import requests
import json
from typing import List, Dict
import os
from dotenv import load_dotenv
import time

load_dotenv()

class ZohoBulkAPI:
    def __init__(self):
        self.client_id = os.getenv("ZOHO_CLIENT_ID")
        self.client_secret = os.getenv("ZOHO_CLIENT_SECRET")
        self.refresh_token = os.getenv("ZOHO_REFRESH_TOKEN")
        self.access_token = os.getenv("ZOHO_ACCESS_TOKEN")
        
        self.owner_name = os.getenv("ZOHO_OWNER_NAME")
        self.app_name = os.getenv("ZOHO_CREATOR_APP_NAME")
        self.form_name = os.getenv("ZOHO_CREATOR_FORM_NAME")
        
        # FORM endpoint (not report)
        self.base_url = f"https://creator.zoho.com/api/v2/{self.owner_name}/{self.app_name}/form/{self.form_name}"
        
        print(f"✓ Zoho Form API URL: {self.base_url}")
    
    def refresh_access_token(self):
        """Refresh the access token using refresh token"""
        url = "https://accounts.zoho.com/oauth/v2/token"
        
        data = {
            'refresh_token': self.refresh_token,
            'client_id': self.client_id,
            'client_secret': self.client_secret,
            'grant_type': 'refresh_token'
        }
        
        try:
            response = requests.post(url, data=data, timeout=30)
            response.raise_for_status()
            
            token_data = response.json()
            self.access_token = token_data['access_token']
            
            print("✓ Access token refreshed")
            return True
        except Exception as e:
            print(f"✗ Failed to refresh token: {e}")
            return False
    
    def format_record_for_zoho_form(self, record: Dict) -> Dict:
        """
        Format record to match YOUR Zoho Creator form fields
        """
        bill_data = record.get('bill_data', {})
        bank_data = record.get('bank_data', {})
        
        # Handle JSON fields if they're strings
        if isinstance(bill_data, str):
            try:
                bill_data = json.loads(bill_data)
            except:
                bill_data = {}
        
        if isinstance(bank_data, str):
            try:
                bank_data = json.loads(bank_data)
            except:
                bank_data = {}
        
        # Map to YOUR form fields
        return {
            "Scholar_Name": record.get('student_name') or bill_data.get('student_name') or "",
            "Scholar_ID": record.get('scholar_id') or "",
            "Account_No": bank_data.get('account_number') or "",
            "Total_Extraction": record.get('tokens_used') or 0,
            "Status": record.get('status', 'completed'),
            "Amount": float(bill_data.get('amount') or 0),
            "Bank_Name": bank_data.get('bank_name') or "",
            "Holder_Name": bank_data.get('account_holder_name') or "",
            "IFSC_Code": bank_data.get('ifsc_code') or "",
            "Branch_Name": bank_data.get('branch_name') or "",
            
            # Store full bill data as JSON string
            "Bill_Data": json.dumps(bill_data) if bill_data else "",
            
            # Individual bill amounts
            "Bill1_Amount": float(bill_data.get('amount') or 0),  # Primary amount
            "Bill2_Amount": 0,  # Placeholder for future bills
            "Bill3_Amount": 0,
            "Bill4_Amount": 0,
            "Bill5_Amount": 0,
            "Bill6_Amount": 0,
        }
    
    def push_single_record(self, record: Dict) -> Dict:
        """Push a single record to Zoho Creator form"""
        try:
            formatted_record = self.format_record_for_zoho_form(record)
            
            # Zoho Form API expects data in specific format
            payload = {
                "data": formatted_record
            }
            
            headers = {
                'Authorization': f'Zoho-oauthtoken {self.access_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                self.base_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            # Handle token expiry
            if response.status_code == 401:
                print("⚠️ Token expired, refreshing...")
                if self.refresh_access_token():
                    headers['Authorization'] = f'Zoho-oauthtoken {self.access_token}'
                    response = requests.post(
                        self.base_url,
                        json=payload,
                        headers=headers,
                        timeout=30
                    )
            
            response.raise_for_status()
            result = response.json()
            
            return {
                "success": True,
                "record": formatted_record.get('Scholar_ID', 'unknown'),
                "zoho_response": result
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "record": record.get('scholar_id', 'unknown')
            }
    
    def bulk_insert(self, records: List[Dict], batch_size: int = 1) -> Dict:
        """
        Bulk insert records to Zoho Creator FORM
        
        Note: Form API typically accepts 1 record at a time
        If bulk endpoint is available, set batch_size higher
        """
        total_records = len(records)
        
        results = {
            "total_records": total_records,
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        print(f"\n{'='*80}")
        print(f"BULK TRANSFER TO ZOHO CREATOR FORM")
        print(f"{'='*80}")
        print(f"Form: {self.form_name}")
        print(f"Total Records: {total_records}")
        print(f"{'='*80}\n")
        
        for idx, record in enumerate(records, 1):
            print(f"[{idx}/{total_records}] Processing {record.get('scholar_id', 'unknown')}...", end=' ')
            
            result = self.push_single_record(record)
            
            if result['success']:
                results['successful'] += 1
                print("✓")
            else:
                results['failed'] += 1
                results['errors'].append({
                    "record": result['record'],
                    "error": result['error']
                })
                print(f"✗ {result['error']}")
            
            # Rate limiting - wait 1 second between requests
            if idx < total_records:
                time.sleep(1)
        
        print(f"\n{'='*80}")
        print(f"BULK TRANSFER COMPLETE")
        print(f"{'='*80}")
        print(f"✓ Successful: {results['successful']}/{total_records}")
        print(f"✗ Failed: {results['failed']}/{total_records}")
        
        if results['errors']:
            print(f"\nErrors:")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"  - {error['record']}: {error['error']}")
        
        print(f"{'='*80}\n")
        
        return results
    
    def test_connection(self) -> Dict:
        """Test connection with a dummy record"""
        test_record = {
            "scholar_id": "TEST_001",
            "student_name": "Test Student",
            "bank_data": {
                "account_number": "1234567890",
                "bank_name": "Test Bank",
                "account_holder_name": "Test Holder",
                "ifsc_code": "TEST0001234",
                "branch_name": "Test Branch"
            },
            "bill_data": {
                "student_name": "Test Student",
                "college_name": "Test College",
                "amount": "5000"
            },
            "status": "test",
            "tokens_used": 100
        }
        
        return self.push_single_record(test_record)