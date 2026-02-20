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
    
    # ✅ NEW: Helper method to safely convert to float
    @staticmethod
    def safe_float(value, default=0.0):
        """
        Safely convert to float, handling None/null/empty values
        Returns default (0.0) for null amounts so records can still be pushed
        """
        if value is None or value == "null" or value == "":
            return default
        
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
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
        Handles MULTIPLE bills correctly with NULL amount handling!
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
        
        # Convert bill_data to array format (handle both single object and array)
        bill_data_array = []
        if isinstance(bill_data, list):
            bill_data_array = bill_data
        elif isinstance(bill_data, dict) and bill_data:
            bill_data_array = [bill_data]
        
        # ✅ UPDATED: Format bill data with NULL amount handling
        def format_bill_text(bills):
            if not bills:
                return ""
            
            formatted_parts = []
            for idx, bill in enumerate(bills):
                amount = bill.get('amount')
                
                # ✅ Mark bills with null amounts
                if amount is None or amount == 'null':
                    amount_str = "⚠️ Amount not extracted"
                else:
                    amount_str = f"₹{self.safe_float(amount):,.2f}"
                
                prefix = f"Bill {idx + 1}: " if len(bills) > 1 else ""
                part = (
                    f"{prefix}"
                    f"Student: {bill.get('student_name', 'N/A')} | "
                    f"College: {bill.get('college_name', 'N/A')} | "
                    f"Receipt: {bill.get('receipt_number', 'N/A')} | "
                    f"Roll: {bill.get('roll_number', 'N/A')} | "
                    f"Class: {bill.get('class_course', 'N/A')} | "
                    f"Date: {bill.get('bill_date', 'N/A')} | "
                    f"Amount: {amount_str}"
                )
                formatted_parts.append(part)
            
            return " || ".join(formatted_parts)
        
        # ✅ FIXED: Extract individual bill amounts safely (handles None/null)
        bill1_amount = self.safe_float(bill_data_array[0].get('amount')) if len(bill_data_array) > 0 else 0.0
        bill2_amount = self.safe_float(bill_data_array[1].get('amount')) if len(bill_data_array) > 1 else 0.0
        bill3_amount = self.safe_float(bill_data_array[2].get('amount')) if len(bill_data_array) > 2 else 0.0
        bill4_amount = self.safe_float(bill_data_array[3].get('amount')) if len(bill_data_array) > 3 else 0.0
        bill5_amount = self.safe_float(bill_data_array[4].get('amount')) if len(bill_data_array) > 4 else 0.0
        
        # Calculate total (sum of valid amounts, nulls become 0)
        total_amount = bill1_amount + bill2_amount + bill3_amount + bill4_amount + bill5_amount
        
        # Get scholar info (prefer from first bill)
        scholar_name = (
            record.get('Scholar_Name') or 
            record.get('student_name') or 
            (bill_data_array[0].get('student_name') if bill_data_array else '') or 
            ''
        )
        
        scholar_id = (
            record.get('Scholar_ID') or 
            record.get('scholar_id') or 
            (bill_data_array[0].get('scholar_id') if bill_data_array else '') or 
            ''
        )
        
        # ✅ Add warning status if amounts are missing
        has_null_amounts = any(
            bill.get('amount') is None or bill.get('amount') == 'null'
            for bill in bill_data_array
        ) if bill_data_array else False
        
        status = record.get('status', 'completed')
        if has_null_amounts:
            status = "⚠️ Needs Review - Some amounts missing"
        
        # Map to YOUR form fields
        return {
            "Scholar_Name": scholar_name,
            "Scholar_ID": scholar_id,
            "Tracking_ID": record.get('Tracking_id', ''),
            "Account_No": bank_data.get('account_number', ''),
            "Total_Extraction": record.get('tokens_used', 0),
            "Status": status,  # ✅ Updated status with warning
            "Amount": bill1_amount,  # Primary amount (first bill)
            "Bank_Name": bank_data.get('bank_name', ''),
            "Holder_Name": bank_data.get('account_holder_name', ''),
            "IFSC_Code": bank_data.get('ifsc_code', ''),
            "Branch_Name": bank_data.get('branch_name', ''),
            
            # Store formatted bill data as readable text (with null warnings)
            "Bill_Data": format_bill_text(bill_data_array),
            
            # Individual bill amounts (supports up to 5 bills) - will be 0.0 if null
            "Bill1_Amount": bill1_amount,
            "Bill2_Amount": bill2_amount,
            "Bill3_Amount": bill3_amount,
            "Bill4_Amount": bill4_amount,
            "Bill5_Amount": bill5_amount,
            "Total_Amount": total_amount,  # ✅ Add total
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
            "errors": [],
            "warnings": []  # ✅ Add warnings tracking
        }
        
        print(f"\n{'='*80}")
        print(f"BULK TRANSFER TO ZOHO CREATOR FORM")
        print(f"{'='*80}")
        print(f"Form: {self.form_name}")
        print(f"Total Records: {total_records}")
        print(f"{'='*80}\n")
        
        for idx, record in enumerate(records, 1):
            scholar_id = record.get('scholar_id', 'unknown')
            print(f"[{idx}/{total_records}] Processing {scholar_id}...", end=' ')
            
            # ✅ Check for null amounts before pushing
            bill_data = record.get('bill_data', [])
            if isinstance(bill_data, str):
                try:
                    bill_data = json.loads(bill_data)
                except:
                    bill_data = []
            
            has_null = any(
                bill.get('amount') is None or bill.get('amount') == 'null'
                for bill in bill_data
            ) if bill_data else False
            
            result = self.push_single_record(record)
            
            if result['success']:
                results['successful'] += 1
                if has_null:
                    print("⚠️ (null amounts)")
                    results['warnings'].append({
                        "record": scholar_id,
                        "message": "Some bill amounts are null"
                    })
                else:
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
        print(f"⚠️ With null amounts: {len(results['warnings'])}/{total_records}")
        
        if results['errors']:
            print(f"\nErrors:")
            for error in results['errors'][:5]:  # Show first 5 errors
                print(f"  - {error['record']}: {error['error']}")
        
        if results['warnings']:
            print(f"\nWarnings (null amounts):")
            for warning in results['warnings'][:5]:  # Show first 5 warnings
                print(f"  - {warning['record']}: {warning['message']}")
        
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