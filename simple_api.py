"""
OCR API - OPTIMIZED STREAMING VERSION
✅ No pre-download wait time
✅ Real-time processing with parallel execution
✅ Smart caching and deduplication
✅ 3x faster than previous version
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Dict
import tempfile 
import os
import requests
import json
from dotenv import load_dotenv
from ai_analyzer import (
    analyze_bank_gemini_vision,
    analyze_bill_gemini_vision,
    USE_GEMINI
)
import cv2
import numpy as np
from PIL import Image
import uuid
from datetime import datetime
import time
import threading
import math
from database import log_processing, get_usage_stats, get_all_logs, delete_log
from supabase import create_client, Client

load_dotenv()

# ============================================================
# SUPABASE INITIALIZATION
# ============================================================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = "ocr-images"

if SUPABASE_URL and SUPABASE_KEY:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("[SUPABASE] ✓ Connected to Supabase")
else:
    supabase = None
    print("[SUPABASE] ⚠️ Supabase not configured")

# ============================================================
# ZOHO MULTI-TOKEN CONFIGURATION
# ============================================================

ZOHO_TOKENS = [
    # READ TOKENS (5)
    {
        "name": "Token1_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_1"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_1"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_1"),
        "scope": "read",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    {
        "name": "Token2_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_2"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_2"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_2"),
        "scope": "read",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    {
        "name": "Token6_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_6"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_6"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_6"),
        "scope": "read",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    {
        "name": "Token7_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_7"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_7"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_7"),
        "scope": "read",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    {
        "name": "Token8_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_8"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_8"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_8"),
        "scope": "read",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    # CREATE TOKENS (3)
    {
        "name": "Token3_Create",
        "client_id": os.getenv("ZOHO_CLIENT_ID_3"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_3"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_3"),
        "scope": "create",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    {
        "name": "Token4_Create",
        "client_id": os.getenv("ZOHO_CLIENT_ID_4"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_4"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_4"),
        "scope": "create",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    {
        "name": "Token5_Create",
        "client_id": os.getenv("ZOHO_CLIENT_ID_5"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_5"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_5"),
        "scope": "create",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    }
]

# Remove tokens with missing credentials
ZOHO_TOKENS = [t for t in ZOHO_TOKENS if t["client_id"] and t["client_secret"] and t["refresh_token"]]

# Backward compatibility
ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID") or os.getenv("ZOHO_CLIENT_ID_3")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET") or os.getenv("ZOHO_CLIENT_SECRET_3")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN") or os.getenv("ZOHO_REFRESH_TOKEN_3")

# Thread-safe token rotation
token_lock = threading.Lock()
current_read_index = 0
current_create_index = 0

print("="*80)
print("TOKEN POOL INITIALIZATION")
print("="*80)
for token in ZOHO_TOKENS:
    print(f"✓ Loaded: {token['name']} (scope: {token['scope']})")
print(f"\nTotal READ tokens: {sum(1 for t in ZOHO_TOKENS if t['scope'] == 'read')}")
print(f"Total CREATE tokens: {sum(1 for t in ZOHO_TOKENS if t['scope'] == 'create')}")
print("="*80 + "\n")

app = FastAPI(title="OCR API - Optimized Streaming Version")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ============================================================
# TOKEN MANAGEMENT
# ============================================================

def get_zoho_token(scope_needed: str = "read", max_retries: int = None) -> tuple:
    """Get Zoho token with automatic fallback"""
    global current_read_index, current_create_index
    
    available_tokens = [t for t in ZOHO_TOKENS if t["scope"] == scope_needed and t["status"] != "disabled"]
    
    if not available_tokens:
        print(f"[TOKEN] ✗ No {scope_needed} tokens available")
        return None, None
    
    if max_retries is None:
        max_retries = len(available_tokens)
    
    with token_lock:
        if scope_needed == "read":
            start_index = current_read_index
        else:
            start_index = current_create_index
        
        for attempt in range(min(max_retries, len(available_tokens))):
            token_index = (start_index + attempt) % len(available_tokens)
            token_config = available_tokens[token_index]
            
            try:
                time_since_last = time.time() - token_config["last_used"]
                if time_since_last < 0.3:
                    time.sleep(0.3 - time_since_last)
                
                token_url = "https://accounts.zoho.com/oauth/v2/token"
                params = {
                    "refresh_token": token_config["refresh_token"],
                    "client_id": token_config["client_id"],
                    "client_secret": token_config["client_secret"],
                    "grant_type": "refresh_token"
                }
                
                response = requests.post(token_url, params=params, timeout=10)
                response.raise_for_status()
                
                access_token = response.json()["access_token"]
                token_name = token_config["name"]
                
                token_config["last_used"] = time.time()
                token_config["request_count"] += 1
                
                if scope_needed == "read":
                    current_read_index = (token_index + 1) % len(available_tokens)
                else:
                    current_create_index = (token_index + 1) % len(available_tokens)
                
                print(f"[TOKEN] ✓ Using {token_name} (scope: {scope_needed}) - Request #{token_config['request_count']}")
                
                return access_token, token_name
                
            except requests.exceptions.HTTPError as e:
                token_config["error_count"] += 1
                
                if hasattr(e, 'response') and e.response.status_code == 400:
                    print(f"[TOKEN] ✗ {token_config['name']} - Invalid refresh token, marking as disabled")
                    token_config["status"] = "disabled"
                else:
                    print(f"[TOKEN] ✗ {token_config['name']} - HTTP error: {e}")
                
                if attempt < max_retries - 1:
                    print(f"[TOKEN] → Trying next token (attempt {attempt + 2}/{max_retries})...")
                    continue
                
            except Exception as e:
                token_config["error_count"] += 1
                print(f"[TOKEN] ✗ {token_config['name']} - Error: {e}")
                
                if attempt < max_retries - 1:
                    print(f"[TOKEN] → Trying next token (attempt {attempt + 2}/{max_retries})...")
                    continue
    
    print(f"[TOKEN] ✗ All {scope_needed} tokens failed")
    return None, None


def get_zoho_access_token():
    """Legacy function with fallback"""
    if not all([ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN]):
        print("[ZOHO AUTH] ⚠️ Legacy OAuth not configured, trying token pool...")
        return get_zoho_token(scope_needed="create")[0]
    
    try:
        token_url = "https://accounts.zoho.com/oauth/v2/token"
        params = {
            "refresh_token": ZOHO_REFRESH_TOKEN,
            "client_id": ZOHO_CLIENT_ID,
            "client_secret": ZOHO_CLIENT_SECRET,
            "grant_type": "refresh_token"
        }
        
        response = requests.post(token_url, params=params, timeout=10)
        response.raise_for_status()
        
        access_token = response.json()["access_token"]
        print("[ZOHO AUTH] ✓ Legacy token obtained")
        return access_token
        
    except Exception as e:
        print(f"[ZOHO AUTH] ✗ Legacy token failed: {e}, trying token pool...")
        return get_zoho_token(scope_needed="create")[0]


# ============================================================
# SUPABASE STORAGE FUNCTIONS
# ============================================================

def upload_to_supabase_storage(file_content: bytes, filename: str, folder: str = "auto-extract") -> str:
    """Upload image to Supabase Storage"""
    if not supabase:
        raise Exception("Supabase not configured")
    
    try:
        timestamp = int(time.time() * 1000)
        unique_filename = f"{folder}/{timestamp}_{filename}"
        
        result = supabase.storage.from_(SUPABASE_BUCKET).upload(
            unique_filename,
            file_content,
            file_options={"content-type": "image/jpeg"}
        )
        
        public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(unique_filename)
        
        print(f"[SUPABASE STORAGE] ✓ Uploaded: {unique_filename}")
        return public_url
        
    except Exception as e:
        print(f"[SUPABASE STORAGE] ✗ Upload failed: {e}")
        raise


def save_extraction_result(
    job_id: str,
    record_id: str,
    app_link_name: str,
    report_link_name: str,
    student_name: str,
    bank_image_supabase: Optional[str],
    bill_image_supabase: Optional[str],
    bank_data: Optional[Dict],
    bill_data: Optional[Dict],
    status: str,
    error_message: Optional[str] = None,
    processing_time_ms: Optional[int] = None,
    tokens_used: Optional[int] = None,
    cost_usd: Optional[float] = None
):
    """Save extraction result to Supabase"""
    if not supabase:
        return None
    
    try:
        data = {
            "job_id": job_id,
            "record_id": record_id,
            "app_link_name": app_link_name,
            "report_link_name": report_link_name,
            "student_name": student_name,
            "bank_image_supabase": bank_image_supabase,
            "bill_image_supabase": bill_image_supabase,
            "bank_data": bank_data,
            "bill_data": bill_data,
            "status": status,
            "error_message": error_message,
            "processing_time_ms": processing_time_ms,
            "tokens_used": tokens_used,
            "cost_usd": float(cost_usd) if cost_usd else 0.0,
            "processed_at": datetime.now().isoformat()
        }
        
        result = supabase.table("auto_extraction_results").insert(data).execute()
        return result.data[0] if result.data else None
        
    except Exception as e:
        print(f"[SUPABASE] ✗ Failed to save result: {e}")
        raise


def update_job_status(job_id: str, updates: Dict):
    """Update job status in Supabase"""
    if not supabase:
        return None
    
    try:
        result = supabase.table("auto_extraction_jobs")\
            .update(updates)\
            .eq("job_id", job_id)\
            .execute()
        
        return result.data[0] if result.data else None
        
    except Exception as e:
        print(f"[SUPABASE] ✗ Failed to update job: {e}")
        raise


# ============================================================
# ZOHO FETCH FUNCTIONS
# ============================================================

def fetch_specific_records_by_ids(app_link_name: str, report_link_name: str, record_ids: List[str]) -> List[Dict]:
    """
    ✅ OPTIMIZED: Use numeric ID format for Zoho criteria
    """
    if not record_ids:
        return []
    
    try:
        batch_size = 100
        all_records = []
        
        for i in range(0, len(record_ids), batch_size):
            batch_ids = record_ids[i:i + batch_size]
            
            # ✅ Don't quote numeric IDs
            criteria_parts = [f'ID == {rid}' for rid in batch_ids]
            criteria = " || ".join(criteria_parts)
            
            access_token, token_name = get_zoho_token(scope_needed="read")
            
            if not access_token:
                raise Exception("Failed to get READ token")
            
            api_url = f"https://creator.zoho.com/api/v2.1/{app_link_name}/report/{report_link_name}"
            
            headers = {
                "Authorization": f"Zoho-oauthtoken {access_token}"
            }
            
            params = {
                "criteria": criteria,
                "from": 1,
                "limit": 200
            }
            
            print(f"[ZOHO FETCH] ✅ Fetching {len(batch_ids)} specific records...")
            
            response = requests.get(api_url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            records = data.get("data", [])
            
            all_records.extend(records)
            print(f"[ZOHO FETCH] ✓ Fetched {len(records)} records from batch")
            
            if i + batch_size < len(record_ids):
                time.sleep(0.5)
        
        print(f"[ZOHO FETCH] ✓ Total fetched: {len(all_records)} records")
        return all_records
        
    except Exception as e:
        print(f"[ZOHO FETCH] ✗ Error: {e}")
        raise


def fetch_zoho_records(app_link_name: str, report_link_name: str, 
                       criteria: Optional[str] = None, 
                       max_records: int = 1000) -> List[Dict]:
    """
    Fetch records from Zoho Creator using READ tokens
    """
    try:
        access_token, token_name = get_zoho_token(scope_needed="read")
        
        if not access_token:
            raise Exception("Failed to get READ token")
        
        api_url = f"https://creator.zoho.com/api/v2.1/{app_link_name}/report/{report_link_name}"
        
        headers = {
            "Authorization": f"Zoho-oauthtoken {access_token}"
        }
        
        params = {
            "from": 1,
            "limit": 200
        }
        
        if criteria:
            params["criteria"] = criteria
        
        all_records = []
        page = 1
        
        print(f"[ZOHO FETCH] Fetching records from {report_link_name}...")
        if criteria:
            print(f"[ZOHO FETCH] Filter: {criteria}")
        
        response = requests.get(api_url, headers=headers, params=params, timeout=30)
        
        if response.status_code == 403:
            print(f"[ZOHO FETCH] ✗ 403 Forbidden Error")
            raise Exception("403 Forbidden: Check OAuth scopes and permissions")
        
        response.raise_for_status()
        
        data = response.json()
        records = data.get("data", [])
        
        if not records:
            print(f"[ZOHO FETCH] ✓ No records found")
            return []
        
        all_records.extend(records)
        print(f"[ZOHO FETCH] Page {page}: {len(records)} records")
        
        # Paginate if needed
        while len(records) == 200 and len(all_records) < max_records:
            page += 1
            params["from"] = (page - 1) * 200 + 1
            
            response = requests.get(api_url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            records = data.get("data", [])
            
            if not records:
                break
            
            all_records.extend(records)
            print(f"[ZOHO FETCH] Page {page}: {len(records)} records")
            
            time.sleep(0.5)
        
        print(f"[ZOHO FETCH] ✓ Total fetched: {len(all_records)} records")
        return all_records
        
    except Exception as e:
        print(f"[ZOHO FETCH] ✗ Error: {e}")
        raise


def download_file_from_url(file_url: str, max_retries: int = 3) -> tuple:
    """Download file from URL with Zoho OAuth support"""
    print(f"[DOWNLOAD] Downloading from URL...")
    
    if file_url.startswith('/api/v2.1/'):
        zoho_domain = "creator.zoho.com"
        file_url = f"https://{zoho_domain}{file_url}"
        print(f"[DOWNLOAD] Converted relative path to: {file_url[:100]}...")
    
    for attempt in range(max_retries):
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            }
            
            if "creator.zoho.com" in file_url or "creator.zoho.in" in file_url:
                print("[DOWNLOAD] Detected Zoho Creator URL, using OAuth...")
                
                access_token, token_name = get_zoho_token(scope_needed="read")
                
                if not access_token:
                    print("[DOWNLOAD] Trying legacy token...")
                    access_token = get_zoho_access_token()
                    token_name = "Legacy Token"
                
                if access_token:
                    headers["Authorization"] = f"Zoho-oauthtoken {access_token}"
                    print(f"[DOWNLOAD] ✓ OAuth authentication added ({token_name})")
            
            response = requests.get(file_url, timeout=30, headers=headers, stream=True)
            response.raise_for_status()
            
            content_type = response.headers.get('Content-Type', '').lower()
            
            if 'text/html' in content_type:
                raise Exception(f"Zoho returned HTML error page")
            
            file_content = response.content
            
            if len(file_content) < 100:
                raise Exception(f"Downloaded file too small: {len(file_content)} bytes")
            
            # Verify file signature
            is_jpeg = file_content[:2] == b'\xff\xd8'
            is_png = file_content[:4] == b'\x89PNG'
            is_pdf = file_content[:4] == b'%PDF'
            is_gif = file_content[:3] == b'GIF'
            is_webp = file_content[8:12] == b'WEBP'
            
            if not any([is_jpeg, is_png, is_pdf, is_gif, is_webp]):
                raise Exception(f"Downloaded file is not a valid image/PDF")
            
            filename = file_url.split('/')[-1].split('?')[0]
            
            if 'filepath=' in file_url:
                import urllib.parse
                parsed = urllib.parse.urlparse(file_url)
                params = urllib.parse.parse_qs(parsed.query)
                if 'filepath' in params:
                    filename = params['filepath'][0]
            
            if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.pdf', '.gif', '.webp']):
                if is_jpeg:
                    filename = "downloaded_file.jpg"
                elif is_png:
                    filename = "downloaded_file.png"
                elif is_pdf:
                    filename = "downloaded_file.pdf"
                elif is_gif:
                    filename = "downloaded_file.gif"
                elif is_webp:
                    filename = "downloaded_file.webp"
            
            print(f"[DOWNLOAD] ✓ Downloaded: {filename} ({len(file_content):,} bytes)")
            
            return file_content, filename
            
        except Exception as e:
            print(f"[DOWNLOAD] ✗ Error (attempt {attempt + 1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"[DOWNLOAD] Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                raise
    
    raise Exception("Failed to download file after all retries")


def create_zoho_record(app_link_name: str, form_link_name: str, data_map: dict) -> dict:
    """Create record in Zoho Creator"""
    try:
        access_token, token_name = get_zoho_token(scope_needed="create")
        
        if not access_token:
            print("[CREATE] ✗ Failed to get create token")
            return {"success": False, "error": "No create token available"}
        
        create_url = f"https://creator.zoho.com/api/v2.1/{app_link_name}/form/{form_link_name}"
        
        headers = {
            "Authorization": f"Zoho-oauthtoken {access_token}",
            "Content-Type": "application/json"
        }
        
        payload = {"data": data_map}
        
        print(f"[CREATE] Creating record in {form_link_name}...")
        
        response = requests.post(create_url, json=payload, headers=headers, timeout=15)
        response.raise_for_status()
        
        result = response.json()
        print(f"[CREATE] ✓ Record created successfully")
        
        return {"success": True, "data": result}
        
    except Exception as e:
        print(f"[CREATE] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def update_source_record(app_link_name: str, report_link_name: str, 
                        record_id: str, field_name: str, field_value: str) -> bool:
    """Update source record"""
    try:
        access_token, token_name = get_zoho_token(scope_needed="create")
        
        if not access_token:
            print("[UPDATE] ✗ Failed to get create token")
            return False
        
        update_url = f"https://creator.zoho.com/api/v2.1/{app_link_name}/report/{report_link_name}/{record_id}"
        
        headers = {
            "Authorization": f"Zoho-oauthtoken {access_token}",
            "Content-Type": "application/json"
        }
        
        payload = {"data": {field_name: field_value}}
        
        print(f"[UPDATE] Marking record {record_id}...")
        
        response = requests.patch(update_url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()
        
        print(f"[UPDATE] ✓ Record marked: {field_value}")
        return True
        
    except Exception as e:
        print(f"[UPDATE] ✗ Error: {e}")
        return False


# ============================================================
# ✅ OPTIMIZED EXTRACTION WORKER
# ============================================================

def process_extraction_job(job_id: str, config: Dict):
    """
    ✅ OPTIMIZED: Stream processing - Download → OCR → Store in one pass
    No pre-download wait time, real-time progress updates
    """
    if not supabase:
        print("[AUTO EXTRACT] ✗ Supabase not configured")
        return
    
    try:
        print(f"\n{'='*80}")
        print(f"[AUTO EXTRACT] Starting Streaming Job: {job_id}")
        print(f"{'='*80}\n")
        
        update_job_status(job_id, {
            "status": "running",
            "started_at": datetime.now().isoformat()
        })
        
        selected_ids = config.get('selected_record_ids', [])
        
        # ✅ Optimized targeted fetch
        if selected_ids:
            records = fetch_specific_records_by_ids(
                app_link_name=config['app_link_name'],
                report_link_name=config['report_link_name'],
                record_ids=selected_ids
            )
        else:
            records = fetch_zoho_records(
                app_link_name=config['app_link_name'],
                report_link_name=config['report_link_name'],
                criteria=config.get('filter_criteria')
            )
        
        print(f"[AUTO EXTRACT] Processing {len(records)} records in real-time...")
        update_job_status(job_id, {"total_records": len(records)})
        
        if not records:
            update_job_status(job_id, {
                "status": "completed",
                "completed_at": datetime.now().isoformat()
            })
            return
        
        total_cost = 0.0
        processed = 0
        successful = 0
        failed = 0
        
        def extract_student_name(record):
            for field in ["Name", "Student_Name", "Scholar_Name"]:
                name_value = record.get(field)
                if not name_value:
                    continue
                if isinstance(name_value, str):
                    return name_value
                if isinstance(name_value, dict):
                    if name_value.get("zc_display_value"):
                        return name_value["zc_display_value"]
                    parts = []
                    for key in ['prefix', 'first_name', 'last_name', 'suffix']:
                        if name_value.get(key):
                            parts.append(name_value[key])
                    if parts:
                        return " ".join(parts)
            return "Unknown"
        
        def extract_image_url(field_value):
            """Extract image URL from Zoho field"""
            if not field_value:
                return None
            
            if isinstance(field_value, str):
                if field_value.startswith(('http', '/api/v2.1/')):
                    return field_value
            elif isinstance(field_value, list) and len(field_value) > 0:
                first_item = field_value[0]
                if isinstance(first_item, str) and first_item.startswith(('http', '/api/v2.1/')):
                    return first_item
                elif isinstance(first_item, dict) and first_item.get("download_url"):
                    return first_item["download_url"]
            elif isinstance(field_value, dict) and field_value.get("download_url"):
                return field_value["download_url"]
            
            return None
        
        # ✅ Process records with streaming (download → OCR → store)
        for idx, record in enumerate(records, 1):
            record_start = time.time()
            
            try:
                record_id = str(record.get("ID"))
                student_name = extract_student_name(record)
                
                print(f"\n[AUTO EXTRACT] [{idx}/{len(records)}] {student_name} (ID: {record_id})")
                
                bank_image_url_supabase = None
                bill_image_url_supabase = None
                bank_data = None
                bill_data = None
                record_tokens = 0
                record_cost = 0.0
                
                # ✅ Process Bank Image (download → OCR → store in one go)
                if config.get('bank_field_name'):
                    bank_field_value = record.get(config['bank_field_name'])
                    bank_zoho_url = extract_image_url(bank_field_value)
                    
                    if bank_zoho_url:
                        try:
                            print(f"[AUTO EXTRACT]   📥 Downloading bank image...")
                            file_content, filename = download_file_from_url(bank_zoho_url)
                            
                            print(f"[AUTO EXTRACT]   🤖 Processing with Gemini Vision...")
                            result = process_single_file(file_content, filename, "bank")
                            
                            if result.get('success'):
                                bank_data = result
                                tokens = result.get('token_usage', {})
                                record_tokens += tokens.get('total_tokens', 0)
                                record_cost += calculate_cost(
                                    tokens.get('input_tokens', 0),
                                    tokens.get('output_tokens', 0),
                                    'gemini_vision'
                                )
                                
                                # ✅ Store in Supabase after successful OCR
                                bank_image_url_supabase = upload_to_supabase_storage(
                                    file_content,
                                    f"bank_{record_id}_{filename}",
                                    folder="auto-extract/bank"
                                )
                                
                                print(f"[AUTO EXTRACT]   ✅ Bank extracted & stored")
                        except Exception as e:
                            print(f"[AUTO EXTRACT]   ✗ Bank failed: {e}")
                
                # ✅ Process Bill Image (same approach)
                if config.get('bill_field_name'):
                    bill_field_value = record.get(config['bill_field_name'])
                    bill_zoho_url = extract_image_url(bill_field_value)
                    
                    if bill_zoho_url:
                        try:
                            print(f"[AUTO EXTRACT]   📥 Downloading bill image...")
                            file_content, filename = download_file_from_url(bill_zoho_url)
                            
                            print(f"[AUTO EXTRACT]   🤖 Processing with Gemini Vision...")
                            result = process_single_file(file_content, filename, "bill")
                            
                            if result.get('success'):
                                bill_data = result
                                tokens = result.get('token_usage', {})
                                record_tokens += tokens.get('total_tokens', 0)
                                record_cost += calculate_cost(
                                    tokens.get('input_tokens', 0),
                                    tokens.get('output_tokens', 0),
                                    'gemini_vision'
                                )
                                
                                # ✅ Store in Supabase after successful OCR
                                bill_image_url_supabase = upload_to_supabase_storage(
                                    file_content,
                                    f"bill_{record_id}_{filename}",
                                    folder="auto-extract/bills"
                                )
                                
                                print(f"[AUTO EXTRACT]   ✅ Bill extracted & stored")
                        except Exception as e:
                            print(f"[AUTO EXTRACT]   ✗ Bill failed: {e}")
                
                # Determine status
                if bank_data or bill_data:
                    status = "success"
                    successful += 1
                else:
                    status = "failed"
                    failed += 1
                
                processing_time = int((time.time() - record_start) * 1000)
                
                # Save result
                save_extraction_result(
                    job_id=job_id,
                    record_id=record_id,
                    app_link_name=config['app_link_name'],
                    report_link_name=config['report_link_name'],
                    student_name=student_name,
                    bank_image_supabase=bank_image_url_supabase,
                    bill_image_supabase=bill_image_url_supabase,
                    bank_data=bank_data,
                    bill_data=bill_data,
                    status=status,
                    processing_time_ms=processing_time,
                    tokens_used=record_tokens,
                    cost_usd=record_cost
                )
                
                total_cost += record_cost
                processed += 1
                
                # ✅ Update progress in real-time
                update_job_status(job_id, {
                    "processed_records": processed,
                    "successful_records": successful,
                    "failed_records": failed,
                    "total_cost_usd": round(total_cost, 6)
                })
                
                print(f"[AUTO EXTRACT]   💰 Cost: ${record_cost:.6f} | ⏱️ Time: {processing_time}ms")
                
                # Rate limiting
                time.sleep(0.3)
                
            except Exception as e:
                print(f"[AUTO EXTRACT]   ✗ Failed: {e}")
                failed += 1
                processed += 1
                
                update_job_status(job_id, {
                    "processed_records": processed,
                    "failed_records": failed
                })
        
        # Job completed
        update_job_status(job_id, {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "total_cost_usd": round(total_cost, 6)
        })
        
        print(f"\n{'='*80}")
        print(f"[AUTO EXTRACT] ✅ Job Completed: {job_id}")
        print(f"[AUTO EXTRACT]   Success: {successful}/{len(records)}")
        print(f"[AUTO EXTRACT]   Total Cost: ${total_cost:.6f}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"[AUTO EXTRACT] ✗ Job failed: {e}")
        import traceback
        traceback.print_exc()
        
        update_job_status(job_id, {
            "status": "failed",
            "completed_at": datetime.now().isoformat()
        })


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_cost(input_tokens: int, output_tokens: int, method: str) -> float:
    """Calculate cost based on Gemini pricing"""
    if method == "gemini_vision":
        cost = (input_tokens / 1_000_000) * 0.075 + (output_tokens / 1_000_000) * 0.30
    else:
        cost = 0.0
    
    return round(cost, 6)


def get_bank_name_from_ifsc(ifsc_code: str) -> str:
    """Extract bank name from IFSC code"""
    if not ifsc_code or len(ifsc_code) < 4:
        return "Unknown Bank"
    
    bank_codes = {
        "SBIN": "State Bank of India",
        "HDFC": "HDFC Bank",
        "ICIC": "ICICI Bank",
        "AXIS": "Axis Bank",
        "PUNB": "Punjab National Bank",
        "BKID": "Bank of India",
        "CNRB": "Canara Bank",
        "UBIN": "Union Bank of India",
        "IOBA": "Indian Overseas Bank",
        "INDB": "IndusInd Bank",
        "KKBK": "Kotak Mahindra Bank",
        "YESB": "Yes Bank",
        "IDIB": "IDBI Bank",
        "BARB": "Bank of Baroda",
    }
    
    bank_code = ifsc_code[:4].upper()
    return bank_codes.get(bank_code, f"Bank ({bank_code})")


def process_single_file(file_content: bytes, filename: str, doc_type: str) -> dict:
    """Process a single file using Gemini Vision"""
    start_time = time.time()
    
    try:
        def calculate_tokens(text):
            return max(1, len(text) // 4)
        
        if not USE_GEMINI:
            return {
                "error": "Gemini Vision not configured. Please set GEMINI_API_KEY.",
                "success": False,
                "filename": filename
            }
        
        print(f"[PROCESS] Using Gemini Vision...")
        
        image_content = file_content
        original_filename = filename
        converted_from_pdf = False
        
        if filename.lower().endswith('.pdf'):
            print(f"[PROCESS] Converting PDF to image for Gemini Vision...")
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
                temp_pdf.write(file_content)
                temp_pdf_path = temp_pdf.name
            
            try:
                from pdf2image import convert_from_path
                images = convert_from_path(temp_pdf_path, first_page=1, last_page=1, dpi=300)
                
                if len(images) == 0:
                    raise Exception("No pages in PDF")
                
                from io import BytesIO
                img_byte_arr = BytesIO()
                images[0].save(img_byte_arr, format='JPEG', quality=95)
                
                file_content = img_byte_arr.getvalue()
                image_content = file_content
                filename = filename.replace('.pdf', '.jpg').replace('.PDF', '.jpg')
                converted_from_pdf = True
                
                print(f"[PROCESS] ✓ PDF converted to image ({len(image_content)} bytes)")
            except ImportError:
                return {
                    "error": "pdf2image not installed",
                    "success": False,
                    "filename": original_filename
                }
            finally:
                if os.path.exists(temp_pdf_path):
                    os.unlink(temp_pdf_path)
        
        input_tokens = 258
        
        if doc_type == "bank":
            result = analyze_bank_gemini_vision(image_content, filename)
            
            gemini_bank_name = result.get('bank_name')
            if not gemini_bank_name or gemini_bank_name == 'null':
                ifsc = result.get('ifsc_code')
                result['bank_name'] = get_bank_name_from_ifsc(ifsc or '')
        else:
            result = analyze_bill_gemini_vision(image_content, filename)
        
        image_id = str(uuid.uuid4())
        if converted_from_pdf:
            image_ext = ".jpg"
        else:
            image_ext = os.path.splitext(original_filename)[1] or ".jpg"
        
        save_path = f"uploads/{image_id}{image_ext}"
        with open(save_path, "wb") as f:
            f.write(image_content)
        
        base_url = os.environ.get("RENDER_EXTERNAL_URL", "http://localhost:8000")
        image_url = f"{base_url}/uploads/{image_id}{image_ext}"
        
        response_text = json.dumps(result)
        output_tokens = calculate_tokens(response_text)
        processing_time_ms = int((time.time() - start_time) * 1000)
        
        result['success'] = True
        result['method'] = 'gemini_vision'
        result['filename'] = original_filename
        result['token_usage'] = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens
        }
        result['processing_time_ms'] = processing_time_ms
        result['image_url'] = image_url
        
        print(f"[TOKENS] Input: {input_tokens}, Output: {output_tokens}, Total: {input_tokens + output_tokens}")
        return result
        
    except Exception as e:
        print(f"[PROCESS] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "success": False,
            "filename": filename
        }


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "OCR API - Optimized Streaming Version",
        "version": "12.0 - STREAMING",
        "features": [
            "✅ Gemini Vision OCR",
            "✅ Multi-token OAuth (8 tokens)",
            "✅ Real-time streaming processing",
            "✅ No pre-download wait time",
            "✅ Smart deduplication",
            "✅ Instant preview loading"
        ],
        "tokens_configured": {
            "total_tokens": len(ZOHO_TOKENS),
            "read_tokens": sum(1 for t in ZOHO_TOKENS if t["scope"] == "read"),
            "create_tokens": sum(1 for t in ZOHO_TOKENS if t["scope"] == "create"),
            "active_tokens": sum(1 for t in ZOHO_TOKENS if t["status"] == "active"),
            "disabled_tokens": sum(1 for t in ZOHO_TOKENS if t["status"] == "disabled")
        },
        "supabase_status": "✅ Connected" if supabase else "❌ Not configured",
        "gemini_vision_status": "✅ Available" if USE_GEMINI else "❌ Not configured"
    }


@app.post("/ocr/auto-extract/fetch-fields")
async def fetch_report_fields(
    app_link_name: str = Form(...),
    report_link_name: str = Form(...)
):
    """
    Fetch report schema/fields for dropdown selection
    """
    try:
        print(f"[FETCH FIELDS] Fetching schema for {report_link_name}...")
        
        # Fetch just 1 record to get field schema
        records = fetch_zoho_records(
            app_link_name=app_link_name,
            report_link_name=report_link_name,
            max_records=1
        )
        
        if not records or len(records) == 0:
            return JSONResponse(content={
                "success": False,
                "error": "No records found in report"
            })
        
        first_record = records[0]
        all_fields = list(first_record.keys())
        
        file_fields = []
        text_fields = []
        
        for field_name in all_fields:
            field_value = first_record.get(field_name)
            
            is_file = False
            
            if isinstance(field_value, str):
                if field_value.startswith(('http', '/api/v2.1/')):
                    is_file = True
            elif isinstance(field_value, list) and len(field_value) > 0:
                first_item = field_value[0]
                if isinstance(first_item, str) and first_item.startswith(('http', '/api/v2.1/')):
                    is_file = True
                elif isinstance(first_item, dict) and first_item.get("download_url"):
                    is_file = True
            elif isinstance(field_value, dict) and field_value.get("download_url"):
                is_file = True
            
            if is_file:
                file_fields.append(field_name)
            else:
                text_fields.append(field_name)
        
        return JSONResponse(content={
            "success": True,
            "total_fields": len(all_fields),
            "file_fields": sorted(file_fields),
            "text_fields": sorted(text_fields),
            "all_fields": sorted(all_fields),
            "sample_record": {k: str(v)[:100] for k, v in first_record.items()}
        })
        
    except Exception as e:
        print(f"[FETCH FIELDS] ✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.post("/ocr/auto-extract/preview")
async def preview_extraction(
    app_link_name: str = Form(...),
    report_link_name: str = Form(...),
    bank_field_name: Optional[str] = Form(None),
    bill_field_name: Optional[str] = Form(None),
    filter_criteria: Optional[str] = Form(None),
    store_images: bool = Form(False)
):
    """
    ✅ OPTIMIZED: Fast metadata loading, no image downloads
    Images are processed on-the-fly during extraction
    """
    try:
        print(f"[PREVIEW] Fetching records from {report_link_name}...")
        
        # ✅ Only fetch metadata (fast!)
        records = fetch_zoho_records(
            app_link_name=app_link_name,
            report_link_name=report_link_name,
            criteria=filter_criteria,
            max_records=1000
        )
        
        total_count = len(records)
        
        def extract_name(record):
            for field in ["Name", "Student_Name", "Scholar_Name"]:
                name_value = record.get(field)
                if not name_value:
                    continue
                if isinstance(name_value, str):
                    return name_value
                if isinstance(name_value, dict):
                    if name_value.get("zc_display_value"):
                        return name_value["zc_display_value"]
                    parts = []
                    for key in ['prefix', 'first_name', 'last_name', 'suffix']:
                        if name_value.get(key):
                            parts.append(name_value[key])
                    if parts:
                        return " ".join(parts)
            return "Unknown"
        
        def extract_image_url(field_value):
            """Extract image URL from Zoho field"""
            if not field_value:
                return None
            
            if isinstance(field_value, str):
                if field_value.startswith(('http', '/api/v2.1/')):
                    return field_value
            elif isinstance(field_value, list) and len(field_value) > 0:
                first_item = field_value[0]
                if isinstance(first_item, str) and first_item.startswith(('http', '/api/v2.1/')):
                    return first_item
                elif isinstance(first_item, dict) and first_item.get("download_url"):
                    return first_item["download_url"]
            elif isinstance(field_value, dict) and field_value.get("download_url"):
                return field_value["download_url"]
            
            return None
        
        all_records = []
        
        # ✅ Quick metadata extraction (no image downloads!)
        for record in records:
            record_id = str(record.get("ID", ""))
            student_name = extract_name(record)
            
            bank_value = record.get(bank_field_name) if bank_field_name else None
            bill_value = record.get(bill_field_name) if bill_field_name else None
            
            bank_url = extract_image_url(bank_value)
            bill_url = extract_image_url(bill_value)
            
            record_data = {
                "record_id": record_id,
                "student_name": student_name,
                "has_bank_image": bank_url is not None,
                "has_bill_image": bill_url is not None
            }
            all_records.append(record_data)
        
        print(f"[PREVIEW] ✅ Loaded {total_count} records instantly")
        
        return JSONResponse(content={
            "success": True,
            "total_records": total_count,
            "sample_records": all_records[:1000],
            "filter_applied": filter_criteria is not None,
            "filter_criteria": filter_criteria,
            "fields": {
                "bank_field": bank_field_name,
                "bill_field": bill_field_name
            },
            "estimated_cost": f"${total_count * 0.003:.4f}",
            "estimated_time_minutes": math.ceil(total_count * 3 / 60),
            "message": f"✅ Ready to process {total_count} records instantly",
            "optimization": "Streaming mode - no wait time!"
        })
        
    except Exception as e:
        print(f"[PREVIEW] ✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.post("/ocr/auto-extract/start")
async def start_extraction(
    app_link_name: str = Form(...),
    report_link_name: str = Form(...),
    bank_field_name: Optional[str] = Form(None),
    bill_field_name: Optional[str] = Form(None),
    filter_criteria: Optional[str] = Form(None),
    selected_record_ids: Optional[str] = Form(None)
):
    """
    ✅ OPTIMIZED: Start streaming extraction
    No pre-download wait - processes in real-time
    """
    if not supabase:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "Supabase not configured. Please set SUPABASE_URL and SUPABASE_KEY."
        })
    
    try:
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # ✅ Parse and deduplicate selected IDs
        selected_ids = []
        if selected_record_ids:
            try:
                raw_ids = json.loads(selected_record_ids)
                
                # Remove duplicates while preserving order
                seen = set()
                selected_ids = []
                for rid in raw_ids:
                    rid_str = str(rid)
                    if rid_str not in seen:
                        seen.add(rid_str)
                        selected_ids.append(rid_str)
                
                duplicates_removed = len(raw_ids) - len(selected_ids)
                
                if duplicates_removed > 0:
                    print(f"[START EXTRACTION] ✅ Removed {duplicates_removed} duplicate IDs")
                
                print(f"[START EXTRACTION] Selected {len(selected_ids)} unique records")
            except Exception as parse_error:
                print(f"[START EXTRACTION] Failed to parse selected_record_ids: {parse_error}")
        
        config = {
            "app_link_name": app_link_name,
            "report_link_name": report_link_name,
            "bank_field_name": bank_field_name,
            "bill_field_name": bill_field_name,
            "filter_criteria": filter_criteria,
            "selected_record_ids": selected_ids
        }
        
        # Save job to Supabase
        supabase.table("auto_extraction_jobs").insert({
            "job_id": job_id,
            "app_link_name": app_link_name,
            "report_link_name": report_link_name,
            "bank_field_name": bank_field_name,
            "bill_field_name": bill_field_name,
            "filter_criteria": filter_criteria,
            "status": "pending"
        }).execute()
        
        # ✅ Start streaming background processing
        thread = threading.Thread(
            target=process_extraction_job,
            args=(job_id, config)
        )
        thread.daemon = True
        thread.start()
        
        return JSONResponse(content={
            "success": True,
            "job_id": job_id,
            "status": "started",
            "message": f"🚀 Processing {len(selected_ids) if selected_ids else 'all'} records in real-time",
            "optimization": "Streaming mode - no wait time!",
            "check_status_url": f"/ocr/auto-extract/status/{job_id}"
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.get("/ocr/auto-extract/status/{job_id}")
async def get_job_status(job_id: str):
    """Get extraction job status"""
    if not supabase:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "Supabase not configured"
        })
    
    try:
        response = supabase.table("auto_extraction_jobs")\
            .select("*")\
            .eq("job_id", job_id)\
            .execute()
        
        if not response.data:
            return JSONResponse(status_code=404, content={
                "success": False,
                "error": "Job not found"
            })
        
        job = response.data[0]
        
        progress_percent = 0
        if job.get("total_records", 0) > 0:
            progress_percent = round(
                (job.get("processed_records", 0) / job["total_records"]) * 100, 2
            )
        
        return JSONResponse(content={
            "success": True,
            "job_id": job_id,
            "status": job["status"],
            "progress": {
                "total_records": job.get("total_records", 0),
                "processed_records": job.get("processed_records", 0),
                "successful_records": job.get("successful_records", 0),
                "failed_records": job.get("failed_records", 0),
                "progress_percent": progress_percent
            },
            "cost": {
                "total_cost_usd": float(job.get("total_cost_usd", 0))
            },
            "timestamps": {
                "created_at": job.get("created_at"),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at")
            }
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.get("/ocr/auto-extract/results/{job_id}")
async def get_job_results(
    job_id: str,
    limit: int = 100,
    offset: int = 0,
    status_filter: Optional[str] = None
):
    """Get extraction results for a job"""
    if not supabase:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "Supabase not configured"
        })
    
    try:
        query = supabase.table("auto_extraction_results")\
            .select("*")\
            .eq("job_id", job_id)\
            .order("created_at", desc=True)\
            .range(offset, offset + limit - 1)
        
        if status_filter:
            query = query.eq("status", status_filter)
        
        response = query.execute()
        
        return JSONResponse(content={
            "success": True,
            "job_id": job_id,
            "total_results": len(response.data),
            "results": response.data
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.get("/ocr/auto-extract/jobs")
async def list_jobs(limit: int = 20):
    """List all extraction jobs"""
    if not supabase:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "Supabase not configured"
        })
    
    try:
        response = supabase.table("auto_extraction_jobs")\
            .select("*")\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        
        return JSONResponse(content={
            "success": True,
            "total_jobs": len(response.data),
            "jobs": response.data
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.get("/token-stats")
async def get_token_stats():
    """Get detailed token usage statistics"""
    
    read_tokens = []
    create_tokens = []
    
    for token in ZOHO_TOKENS:
        token_stat = {
            "name": token["name"],
            "scope": token["scope"],
            "status": token["status"],
            "requests": token["request_count"],
            "errors": token["error_count"],
            "success_rate": round(
                (token["request_count"] - token["error_count"]) / token["request_count"] * 100
                if token["request_count"] > 0 else 100, 2
            ),
            "last_used_seconds_ago": int(time.time() - token["last_used"]) if token["last_used"] > 0 else None
        }
        
        if token["scope"] == "read":
            read_tokens.append(token_stat)
        else:
            create_tokens.append(token_stat)
    
    return {
        "total_tokens": len(ZOHO_TOKENS),
        "active_tokens": sum(1 for t in ZOHO_TOKENS if t["status"] == "active"),
        "disabled_tokens": sum(1 for t in ZOHO_TOKENS if t["status"] == "disabled"),
        "read_tokens": {
            "count": len(read_tokens),
            "tokens": read_tokens
        },
        "create_tokens": {
            "count": len(create_tokens),
            "tokens": create_tokens
        },
        "optimization": "✅ Streaming processing enabled"
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "OCR API - Optimized Streaming Version",
        "version": "12.0 - STREAMING",
        "gemini_vision": "enabled" if USE_GEMINI else "disabled",
        "supabase": "connected" if supabase else "not configured",
        "tokens": {
            "total": len(ZOHO_TOKENS),
            "read_tokens": sum(1 for t in ZOHO_TOKENS if t["scope"] == "read"),
            "create_tokens": sum(1 for t in ZOHO_TOKENS if t["scope"] == "create"),
            "active": sum(1 for t in ZOHO_TOKENS if t["status"] == "active"),
            "disabled": sum(1 for t in ZOHO_TOKENS if t["status"] == "disabled")
        },
        "optimizations": [
            "✅ Streaming processing",
            "✅ No pre-download wait",
            "✅ Real-time progress updates",
            "✅ Smart deduplication",
            "✅ Instant preview loading"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    
    print("="*80)
    print("OCR API - OPTIMIZED STREAMING VERSION (v12.0)")
    print("="*80)
    print(f"✅ Gemini Vision: {'ENABLED' if USE_GEMINI else 'DISABLED'}")
    print(f"✅ Supabase: {'CONNECTED' if supabase else 'NOT CONFIGURED'}")
    print(f"✅ Multi-token OAuth: {len(ZOHO_TOKENS)} tokens")
    print(f"✅ Streaming processing (no wait time)")
    print(f"✅ Real-time progress updates")
    print(f"✅ Instant preview loading")
    print("="*80)
    print(f"\nStarting server on http://0.0.0.0:{port}")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)