"""
Enhanced AI Analyzer Module - COMPLETE FIXED VERSION
- Gemini Vision for direct image analysis (FIXED - uses raw bytes)
- Improved Ollama prompts with better models
- OCR text cleaning for character errors
- Better regex patterns for fallback
- Comprehensive error handling
"""
import json
import re
from typing import Dict, Any, List
import os
import time
from dotenv import load_dotenv
load_dotenv()

# Detect which AI service is available
OLLAMA_AVAILABLE = False
USE_GEMINI = False
USE_OPENAI = False
genai_client = None

# Try Ollama first
try:
    import ollama
    ollama.list()
    OLLAMA_AVAILABLE = True
    print("[AI Config] ✓ Ollama detected - Using LOCAL models")
except:
    print("[AI Config] Ollama not available")

# Try Gemini (NEW API)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    try:
        from google import genai
        from google.genai import types
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
        USE_GEMINI = True
        print("[AI Config] ✓ Using Google Gemini (NEW API)")
        print("[AI Config] ✓ Gemini Vision: ENABLED ✅")
    except ImportError as e:
        print(f"[AI Config] google-genai not installed - run: pip3 install google-genai")
        print(f"[AI Config] Error: {e}")
    except Exception as e:
        print(f"[AI Config] Gemini initialization error: {e}")
else:
    print("[AI Config] ✓ Gemini Vision: DISABLED ❌ (No API key)")

# Try OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not USE_GEMINI and OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        USE_OPENAI = True
        print("[AI Config] ✓ Using OpenAI GPT")
    except ImportError:
        print("[AI Config] openai package not installed")

if not OLLAMA_AVAILABLE and not USE_GEMINI and not USE_OPENAI:
    print("[AI Config] ⚠️ WARNING: No AI service available, will use REGEX ONLY")

# Model selection
OLLAMA_MODEL = 'qwen2.5:7b'
GEMINI_MODEL = 'gemini-2.5-flash'

print(f"[AI Config] Selected Model: {OLLAMA_MODEL if OLLAMA_AVAILABLE else GEMINI_MODEL if USE_GEMINI else 'OpenAI GPT' if USE_OPENAI else 'None'}")


def clean_ocr_text(text: str) -> str:
    """Fix common OCR character substitutions"""
    corrections = {
        'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E', 'Н': 'H',
        'К': 'K', 'М': 'M', 'О': 'O', 'Р': 'P', 'Т': 'T',
        'Х': 'X', 'У': 'Y', 'І': 'I', 'Ѕ': 'S',
        '|': 'I', '¡': 'I', 'l': 'I', '–': '-', '—': '-',
    }
    
    result = text
    for old, new in corrections.items():
        result = result.replace(old, new)
    
    return result


def validate_indian_account_number(account_num: str) -> bool:
    """Validate Indian bank account number (9-18 digits)"""
    if not account_num:
        return False
    
    clean = str(account_num).replace(' ', '').replace('-', '')
    
    if not clean.isdigit():
        return False
    
    if len(clean) < 9 or len(clean) > 18:
        return False
    
    if len(clean) == 17:
        print(f"[VALIDATION] ⚠️ Suspicious: 17-digit account number")
        return False
    
    return True


def analyze_bank_gemini_vision(file_content: bytes, filename: str) -> Dict[str, Any]:
    """
    ✅ FIXED: Analyze bank passbook using Gemini Vision (NEW API)
    Uses raw bytes directly - NO base64 encoding!
    """
    if not USE_GEMINI:
        raise Exception("Gemini Vision not available - add GEMINI_API_KEY to .env")
    
    print(f"[GEMINI VISION] Analyzing bank passbook: {filename}")
    
    try:
        from google.genai import types
        
        # Detect MIME type from file content
        mime_type = "image/jpeg"
        if file_content[:4] == b'\x89PNG':
            mime_type = "image/png"
        elif file_content[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
        
        print(f"[GEMINI VISION] Format: {mime_type}, Size: {len(file_content)} bytes")
        
        prompt = """
You are analyzing an Indian bank passbook image. Extract ALL visible details accurately.

LOOK FOR THESE FIELDS:
1. **Account Holder Name**: Usually after "Name:", "A/c Holder:", "Customer Name:"
2. **Account Number**: 10-18 digit number (NOT phone, NOT MICR)
3. **IFSC Code**: 4 letters + "0" + 6 alphanumeric (e.g., CNRB0012345)
4. **Branch Name**: Branch location and city
5. **Bank Name**: Which bank (Canara, SBI, HDFC, etc.)

IMPORTANT:
- Read text carefully, including rotated or small text
- IFSC always has "0" as 5th character
- Account is usually 10-16 digits
- Ignore phone numbers and MICR codes

Return ONLY this JSON:
{
    "bank_name": "Full bank name or null",
    "account_holder_name": "Full name or null",
    "account_number": "Account number or null",
    "ifsc_code": "IFSC code or null",
    "branch_name": "Branch name or null"
}
"""
        
        # ✅ CORRECT: Use types.Part.from_bytes() with raw bytes
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=file_content,  # Raw bytes, NO base64!
                    mime_type=mime_type
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        # Parse response
        response_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(response_text)
        
        print(f"[GEMINI VISION] ✓ Extracted bank details")
        print(f"[GEMINI VISION]   Bank: {data.get('bank_name')}")
        print(f"[GEMINI VISION]   Name: {data.get('account_holder_name')}")
        print(f"[GEMINI VISION]   Account: {data.get('account_number')}")
        print(f"[GEMINI VISION]   IFSC: {data.get('ifsc_code')}")
        print(f"[GEMINI VISION]   Branch: {data.get('branch_name')}")
        
        return data
        
    except Exception as e:
        print(f"[GEMINI VISION] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


def analyze_bill_gemini_vision(file_content: bytes, filename: str) -> Dict[str, Any]:
    """
    ✅ FIXED: Analyze bill using Gemini Vision (NEW API)
    """
    if not USE_GEMINI:
        raise Exception("Gemini Vision not available")
    
    print(f"[GEMINI VISION] Analyzing bill: {filename}")
    
    try:
        from google.genai import types
        
        # Detect MIME type
        mime_type = "image/jpeg"
        if file_content[:4] == b'\x89PNG':
            mime_type = "image/png"
        elif file_content[:2] == b'\xff\xd8':
            mime_type = "image/jpeg"
        
        prompt = """
Extract details from this Indian college fee receipt/bill.

FIND:
1. **Student Name**: Look for "Name:" field (may be handwritten)
2. **Date**: Convert to YYYY-MM-DD format
3. **Total Amount**: Bottom of itemized list (extract number only)
4. **College Name**: Usually at top in large text
5. **Receipt Number**: Look for "Receipt No:", "Challan No:"

RULES:
- For amounts like "18000-00", extract as 18000
- Student name is in "Name" field, NOT signature
- Date format: YYYY-MM-DD

Return ONLY JSON:
{
    "student_name": "name or null",
    "college_name": "name or null",
    "roll_number": "number or null",
    "receipt_number": "number or null",
    "class_course": "course or null",
    "bill_date": "YYYY-MM-DD or null",
    "amount": 18000.00
}
"""
        
        # ✅ Use raw bytes directly
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=file_content,
                    mime_type=mime_type
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        response_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(response_text)
        
        # ✅ FIX: Handle case where Gemini returns an array instead of object
        if isinstance(data, list):
            print(f"[GEMINI VISION] ⚠️ Received array response, taking first element")
            if len(data) > 0:
                data = data[0]  # Take first element
            else:
                # Empty array, return null values
                data = {
                    "student_name": None,
                    "college_name": None,
                    "roll_number": None,
                    "receipt_number": None,
                    "class_course": None,
                    "bill_date": None,
                    "amount": None
                }
        
        # ✅ Ensure data is a dict before accessing
        if not isinstance(data, dict):
            print(f"[GEMINI VISION] ⚠️ Unexpected data type: {type(data)}")
            data = {
                "student_name": None,
                "college_name": None,
                "roll_number": None,
                "receipt_number": None,
                "class_course": None,
                "bill_date": None,
                "amount": None
            }
        
        print(f"[GEMINI VISION] ✓ Extracted bill details")
        print(f"[GEMINI VISION]   College: {data.get('college_name')}")
        print(f"[GEMINI VISION]   Student: {data.get('student_name')}")
        print(f"[GEMINI VISION]   Date: {data.get('bill_date')}")
        print(f"[GEMINI VISION]   Amount: Rs. {data.get('amount')}")
        
        return data
        
    except Exception as e:
        print(f"[GEMINI VISION] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise
        
def extract_bank_details_from_markdown(markdown_text: str) -> dict:
    """Regex-based bank extraction (fallback)"""
    print(f"[MANUAL EXTRACTION] Parsing bank markdown...")
    
    text = clean_ocr_text(markdown_text)
    
    account_holder_name = None
    account_number = None
    ifsc_code = None
    branch_name = None
    bank_name = None
    
    text_upper = text.upper()
    text_clean = text.replace('\n', ' ').replace('#', '')
    
    # Extract bank name
    bank_keywords = {
        'INDIAN BANK': 'Indian Bank',
        'CANARA': 'Canara Bank',
        'SBI': 'State Bank of India',
        'STATE BANK': 'State Bank of India',
        'HDFC': 'HDFC Bank',
        'ICICI': 'ICICI Bank',
        'AXIS': 'Axis Bank',
        'PNB': 'Punjab National Bank',
    }
    
    for keyword, full_name in bank_keywords.items():
        if keyword in text_upper:
            bank_name = full_name
            print(f"[MANUAL EXTRACTION]   ✓ Found bank: {bank_name}")
            break
    
    # Extract account number
    account_patterns = [
        r'Account\s*(?:No\.?|Number)[:\s]*(\d{9,16})',
        r'A/c\s*No\.?[:\s]*(\d{9,16})',
    ]
    
    for pattern in account_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            candidate = match.group(1)
            if validate_indian_account_number(candidate):
                account_number = candidate
                print(f"[MANUAL EXTRACTION]   ✓ Found account: {account_number}")
                break
    
    # Extract IFSC
    ifsc_patterns = [
        r'\b([A-Z]{4}0[A-Z0-9]{6})\b',
        r'IFSC[:\s]+([A-Z]{4}0[A-Z0-9]{6})',
    ]
    
    for pattern in ifsc_patterns:
        match = re.search(pattern, text_upper)
        if match:
            ifsc_code = match.group(1)
            print(f"[MANUAL EXTRACTION]   ✓ Found IFSC: {ifsc_code}")
            break
    
    # Extract name
    name_patterns = [
        r'Name[:\s]+([A-Z][A-Za-z\s\.]{2,40}?)(?:\s+(?:CIF|PERSONAL|Account|\d))',
        r'(?:Account\s+Holder|A/c\s+Holder)[:\s]+([A-Z][A-Za-z\s\.]{2,40})',
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            name = re.sub(r'\s+', ' ', name)
            if 3 <= len(name) <= 50 and not re.search(r'\d', name):
                account_holder_name = name
                print(f"[MANUAL EXTRACTION]   ✓ Found name: {account_holder_name}")
                break
    
    # Extract branch
    branch_match = re.search(r'Branch[:\s]+([A-Z][A-Za-z\s]{3,40})', text, re.IGNORECASE)
    if branch_match:
        branch_name = branch_match.group(1).strip()
        print(f"[MANUAL EXTRACTION]   ✓ Found branch: {branch_name}")
    
    return {
        'bank_name': bank_name,
        'account_holder_name': account_holder_name,
        'account_number': account_number,
        'ifsc_code': ifsc_code,
        'branch_name': branch_name
    }


def extract_bill_details_from_markdown(markdown_text: str) -> dict:
    """Regex-based bill extraction (fallback)"""
    print(f"[MANUAL EXTRACTION] Parsing bill markdown...")
    
    student_name = None
    bill_amount = None
    bill_date = None
    college_name = None
    receipt_number = None
    
    lines = markdown_text.split('\n')
    
    # Extract college name
    college_match = re.search(r'([A-Z][A-Za-z\s]+COLLEGE)', markdown_text, re.IGNORECASE)
    if college_match:
        college_name = college_match.group(1).strip()
        print(f"[MANUAL EXTRACTION]   ✓ Found college: {college_name}")
    
    # Extract student name
    name_patterns = [
        r'Name[:\s\.]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, markdown_text, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            if 3 <= len(name) <= 50 and not re.search(r'\d', name):
                student_name = name
                print(f"[MANUAL EXTRACTION]   ✓ Found name: {student_name}")
                break
    
    # Extract amount
    all_amounts = []
    for line in lines:
        if re.search(r'(?:total|grand\s+total)', line, re.IGNORECASE):
            amount_matches = re.findall(r'(?:Rs\.?|₹)\s*([0-9,\s\-]+)', line, re.IGNORECASE)
            for amt_str in amount_matches:
                clean = amt_str.replace(',', '').replace(' ', '').replace('-', '').strip()
                try:
                    val = float(clean)
                    if 100 < val < 1000000:
                        all_amounts.append(val)
                except:
                    pass
    
    if all_amounts:
        bill_amount = max(all_amounts)
        print(f"[MANUAL EXTRACTION]   ✓ Found amount: Rs. {bill_amount}")
    
    return {
        'student_name': student_name,
        'college_name': college_name,
        'roll_number': None,
        'receipt_number': receipt_number,
        'bill_date': bill_date,
        'amount': bill_amount
    }


def analyze_bank_from_markdown(markdown_text: str) -> Dict[str, Any]:
    """Analyze bank passbook from OCR markdown"""
    try:
        return extract_bank_details_from_markdown(markdown_text)
    except Exception as e:
        print(f"[ANALYSIS] ✗ Error: {e}")
        return {
            'bank_name': None,
            'account_holder_name': None,
            'account_number': None,
            'ifsc_code': None,
            'branch_name': None
        }


def analyze_bill_from_markdown(markdown_text: str) -> Dict[str, Any]:
    """Analyze bill from OCR markdown"""
    try:
        return extract_bill_details_from_markdown(markdown_text)
    except Exception as e:
        print(f"[ANALYSIS] ✗ Error: {e}")
        return {
            'student_name': None,
            'college_name': None,
            'roll_number': None,
            'bill_date': None,
            'amount': None
        }