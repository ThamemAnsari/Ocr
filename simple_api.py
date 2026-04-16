"""
OCR API - COMPLETE VERSION with Auto-Extract + Multi-Bill Image Support
✅ Gemini Vision OCR
✅ Auto-extraction with job tracking
✅ Multi-token OAuth support
✅ Supabase integration
✅ Multi-bill extraction from single JPG/PNG images
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, status, Request, Response, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Dict, Any
import httpx
import secrets
import requests
from pydantic import BaseModel
from zoho_bulk_api import ZohoBulkAPI
import tempfile
import os
import requests
import json
from dotenv import load_dotenv
from ai_analyzer import (
    analyze_bank_gemini_vision,
    analyze_bill_gemini_vision,
    analyze_bill_multi_page,
    analyze_bill_multi_bills_from_image,
    analyze_barcode_gemini_vision,
    USE_GEMINI
)
import cv2
import numpy as np
from PIL import Image
import uuid
from datetime import datetime, timedelta
import time
import threading
import math
from functools import wraps
from database import log_processing, get_usage_stats, get_all_logs, delete_log
from enum import Enum
from typing import Literal


class SyncMode(str, Enum):
    INSERT = "insert"
    UPDATE = "update"
    AUTO = "auto"


class ZohoConfig(BaseModel):
    owner_name: str
    app_name: str
    form_name: str
    report_name: Optional[str] = None


class ZohoSyncRequest(BaseModel):
    config: ZohoConfig
    record_ids: List[Any]
    sync_mode: Literal['insert', 'update', 'auto'] = 'auto'
    tag: Optional[str] = None
    tag_color: Optional[str] = 'blue'
    field_mapping: Optional[Dict[str, str]] = None


class CreatorWebhookPayload(BaseModel):
    app_link_name: str
    report_link_name: str
    record_ids: Optional[Any] = None
    bank_field_name: Optional[str] = None
    bill_field_name: Optional[str] = None
    auto_push: bool = False
    target_app_name: Optional[str] = "iatc-scholarship"
    target_form_name: Optional[str] = "OCR_Extraction_From"
    target_owner_name: Optional[str] = "teameverest"


load_dotenv()

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

# ============================================================
# FASTAPI APP INITIALIZATION
# ============================================================

app = FastAPI(title="OCR API - Complete with Auto-Extract + Multi-Bill Image")

# from auto_startup import register_startup_hook
# register_startup_hook(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
zoho_bulk = ZohoBulkAPI()

# ============================================================
# SUPABASE INITIALIZATION
# ============================================================
try:
    from supabase import create_client, Client

    SUPABASE_URL = os.getenv("SUPABASE_URL") or os.getenv("VITE_SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY") or os.getenv("VITE_SUPABASE_ANON_KEY")
    SUPABASE_BUCKET = "ocr-images"

    if SUPABASE_URL and SUPABASE_KEY:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("[SUPABASE] ✓ Connected")
    else:
        supabase = None
        print("[SUPABASE] ⚠️ Not configured")
except ImportError:
    supabase = None
    print("[SUPABASE] ⚠️ supabase-py not installed")

# ============================================================
# ZOHO MULTI-TOKEN CONFIGURATION
# ============================================================

ZOHO_TOKENS = [
    {
        "name": "Token1_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_1"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_1"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_1"),
        "scope": "read", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
    {
        "name": "Token2_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_2"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_2"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_2"),
        "scope": "read", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
    {
        "name": "Token3_Create",
        "client_id": os.getenv("ZOHO_CLIENT_ID_3"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_3"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_3"),
        "scope": "create", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
    {
        "name": "Token4_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_4"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_4"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_4"),
        "scope": "read", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
    {
        "name": "Token5_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_5"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_5"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_5"),
        "scope": "read", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
    {
        "name": "Token6_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_6"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_6"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_6"),
        "scope": "read", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
    {
        "name": "Token7_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_7"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_7"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_7"),
        "scope": "read", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
    {
        "name": "Token8_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_8"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_8"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_8"),
        "scope": "read", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
    {
        "name": "Token9_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_9"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_9"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_9"),
        "scope": "read", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
    {
        "name": "Token10_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_10"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_10"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_10"),
        "scope": "read", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
    {
        "name": "Token11_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_11"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_11"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_11"),
        "scope": "read", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
    {
        "name": "Token12_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_12"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_12"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_12"),
        "scope": "read", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
    {
        "name": "Token13_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_13"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_13"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_13"),
        "scope": "read", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
    {
        "name": "Token14_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_14"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_14"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_14"),
        "scope": "read", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
    {
        "name": "Token15_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_15"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_15"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_15"),
        "scope": "read", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
    {
        "name": "Token16_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_16"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_16"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_16"),
        "scope": "read", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
    {
        "name": "Token17_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_17"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_17"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_17"),
        "scope": "read", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
    {
        "name": "Token18_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_18"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_18"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_18"),
        "scope": "read", "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
]

WORKDRIVE_TOKENS = [
    {
        "name": "Workdrive_Token1",
        "client_id": os.getenv("WORKDRIVE_CLIENT_ID_1"),
        "client_secret": os.getenv("WORKDRIVE_CLIENT_SECRET_1"),
        "refresh_token": os.getenv("WORKDRIVE_REFRESH_TOKEN_1"),
        "request_count": 0, "error_count": 0, "last_used": 0, "status": "active"
    },
]

WORKDRIVE_TOKENS = [t for t in WORKDRIVE_TOKENS if t["client_id"] and t["client_secret"] and t["refresh_token"]]
ZOHO_TOKENS = [t for t in ZOHO_TOKENS if t["client_id"] and t["client_secret"] and t["refresh_token"]]

read_token_count = sum(1 for t in ZOHO_TOKENS if t["scope"] == "read")
create_token_count = sum(1 for t in ZOHO_TOKENS if t["scope"] == "create")

print(f"\n{'='*80}")
print(f"[TOKENS] Token Pool Initialized")
print(f"{'='*80}")
print(f"✅ Total Tokens Loaded: {len(ZOHO_TOKENS)}")
print(f"📖 Read Tokens: {read_token_count}")
print(f"✍️  Create Tokens: {create_token_count}")
print(f"{'='*80}\n")

ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")

token_lock = threading.Lock()
current_read_index = 0
current_create_index = 0

# ============================================================
# TOKEN CACHING
# ============================================================

token_cache = {}
token_cache_lock = threading.Lock()

# ============================================================
# TOKEN MANAGEMENT
# ============================================================

def get_workdrive_token() -> tuple:
    token_name = "workdrive_token"
    with token_cache_lock:
        if token_name in token_cache:
            cached = token_cache[token_name]
            remaining_time = cached["expires_at"] - time.time()
            if remaining_time > 300:
                return cached["access_token"], token_name

    if not WORKDRIVE_TOKENS:
        return None, None

    token_config = WORKDRIVE_TOKENS[0]
    try:
        token_url = "https://accounts.zoho.com/oauth/v2/token"
        params = {
            "refresh_token": token_config["refresh_token"],
            "client_id": token_config["client_id"],
            "client_secret": token_config["client_secret"],
            "grant_type": "refresh_token"
        }
        response = requests.post(token_url, params=params, timeout=10)
        response.raise_for_status()
        response_data = response.json()
        access_token = response_data["access_token"]
        expires_in = response_data.get("expires_in", 3600)
        with token_cache_lock:
            token_cache[token_name] = {"access_token": access_token, "expires_at": time.time() + expires_in}
        return access_token, token_name
    except Exception as e:
        print(f"[WORKDRIVE] ✗ Token generation failed: {e}")
        return None, None


def fetch_workdrive_files(folder_id: str) -> List[Dict]:
    try:
        access_token, token_name = get_workdrive_token()
        if not access_token:
            raise Exception("Failed to get Workdrive token")

        api_url = f"https://workdrive.zoho.com/api/v1/files/{folder_id}/files"
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}", "Accept": "application/vnd.api+json"}

        response = requests.get(api_url, headers=headers, timeout=30)
        response.raise_for_status()

        data = response.json()
        items = data.get("data", [])

        image_files = []
        for item in items:
            item_type = item.get("type")
            attributes = item.get("attributes", {})
            name = attributes.get("name", "")
            item_id = item.get("id")
            size = attributes.get("size", 0)

            if item_type == "files":
                is_supported = name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg', '.pdf'))
                if is_supported:
                    image_files.append({
                        "file_id": item_id, "filename": name, "size": size,
                        "created_time": attributes.get("created_time"),
                        "modified_time": attributes.get("modified_time"),
                        "extension": name.split(".")[-1].lower() if "." in name else "unknown"
                    })

        return image_files

    except Exception as e:
        print(f"[WORKDRIVE] ✗ Error: {e}")
        import traceback; traceback.print_exc()
        raise


def download_workdrive_file(file_id: str) -> tuple:
    try:
        access_token, token_name = get_workdrive_token()
        if not access_token:
            raise Exception("Failed to get Workdrive token")

        api_url = f"https://workdrive.zoho.com/api/v1/download/{file_id}"
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}", "Accept": "application/vnd.api+json"}

        response = requests.get(api_url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()

        file_content = response.content
        filename = file_id

        if 'Content-Disposition' in response.headers:
            import re
            cd = response.headers['Content-Disposition']
            matches = re.findall('filename="?([^"]+)"?', cd)
            if matches:
                filename = matches[0]

        if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.pdf', '.gif', '.webp']):
            filename = f"{file_id}.jpg"

        return file_content, filename

    except Exception as e:
        print(f"[WORKDRIVE] ✗ Download failed: {e}")
        raise


# ============================================================
# TOKEN MANAGEMENT
# ============================================================

def get_zoho_token(scope_needed: str = "read", max_retries: int = None) -> tuple:
    global current_read_index, current_create_index

    available_tokens = [t for t in ZOHO_TOKENS if t["scope"] == scope_needed and t["status"] != "disabled"]

    if not available_tokens:
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
            token_name = token_config["name"]

            with token_cache_lock:
                if token_name in token_cache:
                    cached = token_cache[token_name]
                    remaining_time = cached["expires_at"] - time.time()
                    if remaining_time > 300:
                        return cached["access_token"], token_name

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
                response_data = response.json()

                if "error" in response_data:
                    token_config["error_count"] += 1
                    token_config["status"] = "invalid"
                    continue

                if "access_token" not in response_data:
                    token_config["error_count"] += 1
                    continue

                access_token = response_data["access_token"]
                expires_in = response_data.get("expires_in", 3600)

                with token_cache_lock:
                    token_cache[token_name] = {"access_token": access_token, "expires_at": time.time() + expires_in}

                token_config["last_used"] = time.time()
                token_config["request_count"] += 1

                if scope_needed == "read":
                    current_read_index = token_index
                else:
                    current_create_index = token_index

                return access_token, token_name

            except Exception as e:
                token_config["error_count"] += 1
                if attempt < max_retries - 1:
                    continue

    return None, None


def get_zoho_access_token():
    if not all([ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN]):
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
        return response.json()["access_token"]
    except Exception as e:
        return get_zoho_token(scope_needed="create")[0]


def retry_on_network_error(max_retries=3, delay=2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    error_msg = str(e).lower()
                    is_network_error = any([
                        "errno 35" in error_msg,
                        "resource temporarily unavailable" in error_msg,
                        "connection reset" in error_msg,
                        "broken pipe" in error_msg,
                        "timeout" in error_msg
                    ])
                    if is_network_error and attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                        continue
                    else:
                        raise
            raise last_error
        return wrapper
    return decorator


# ============================================================
# SUPABASE FUNCTIONS
# ============================================================

@retry_on_network_error(max_retries=3, delay=2)
def upload_to_supabase_storage(file_content: bytes, filename: str, folder: str = "auto-extract") -> str:
    if not supabase:
        raise Exception("Supabase not configured")
    timestamp = int(time.time() * 1000)
    unique_filename = f"{folder}/{timestamp}_{filename}"
    supabase.storage.from_(SUPABASE_BUCKET).upload(unique_filename, file_content, file_options={"content-type": "image/jpeg"})
    return supabase.storage.from_(SUPABASE_BUCKET).get_public_url(unique_filename)


@retry_on_network_error(max_retries=3, delay=2)
def save_extraction_result(
    job_id, record_id, app_link_name, report_link_name, student_name,
    bank_image_supabase, bill_image_supabase, bank_data, bill_data, status,
    error_message=None, processing_time_ms=None, tokens_used=None,
    cost_usd=None, scholar_id=None, tracking_id=None, email=None
):
    if not supabase:
        return None

    data = {
        "job_id": job_id,
        "record_id": record_id,
        "Record_id": record_id,
        "app_link_name": app_link_name,
        "report_link_name": report_link_name,
        "status": status,
        "student_name": student_name,
        "email": email,
        "scholar_id": scholar_id,
        "tracking_id": tracking_id,
        "bank_image_supabase": bank_image_supabase,
        "bill_image_supabase": bill_image_supabase,
        "bank_data": bank_data,
        "bill_data": bill_data,
        "error_message": error_message,
        "push_status": None,
        "processing_time_ms": processing_time_ms,
        "tokens_used": tokens_used,
        "cost_usd": float(cost_usd) if cost_usd else 0.0,
        "processed_at": datetime.now().isoformat()
    }

    try:
        result = supabase.table("auto_extraction_results").insert(data).execute()
        if result.data:
            row = result.data[0]
            print(f"[SAVE] ✅ Saved to Supabase (row_id: {row.get('id')})")
            return row
        return None
    except Exception as save_error:
        print(f"[SAVE] ✗ Error: {save_error}")
        raise


@retry_on_network_error(max_retries=3, delay=2)
def update_job_status(job_id: str, data: Dict[str, Any]):
    if not supabase:
        return None
    return supabase.table("auto_extraction_jobs").update(data).eq("job_id", job_id).execute()


# ============================================================
# ZOHO FETCH FUNCTIONS
# ============================================================

def extract_all_image_urls(field_value, max_images=5):
    if not field_value:
        return []
    urls = []
    if isinstance(field_value, str):
        if field_value.startswith(('http', '/api/v2.1/')):
            urls.append(field_value)
    elif isinstance(field_value, list):
        for item in field_value[:max_images]:
            if isinstance(item, str) and item.startswith(('http', '/api/v2.1/')):
                urls.append(item)
            elif isinstance(item, dict):
                # Check multiple possible keys for image URLs
                url = item.get("download_url") or item.get("value") or item.get("url")
                if url and isinstance(url, str) and url.startswith(('http', '/api/v2.1/')):
                    urls.append(url)
    elif isinstance(field_value, dict):
        # Check multiple possible keys for image URLs
        url = field_value.get("download_url") or field_value.get("value") or field_value.get("url")
        if url and isinstance(url, str) and url.startswith(('http', '/api/v2.1/')):
            urls.append(url)
    return urls[:max_images]


def fetch_zoho_records(app_link_name: str, report_link_name: str,
                       criteria: Optional[str] = None, max_records: int = None) -> List[Dict]:
    try:
        access_token, token_name = get_zoho_token(scope_needed="read")
        if not access_token:
            raise Exception("Failed to get READ token")

        api_url = f"https://creator.zoho.com/api/v2.1/{app_link_name}/report/{report_link_name}"
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}

        if criteria:
            params = {"criteria": criteria, "from": 1, "limit": 200}
            response = requests.get(api_url, headers=headers, params=params, timeout=30)
            if response.status_code == 400:
                try:
                    err_data = response.json()
                    if err_data.get("code") == 9220: # No records exist in this report
                        return []
                except:
                    pass

            if response.status_code != 200:
                print(f"[ZOHO] ✗ Response Status: {response.status_code}")
                print(f"[ZOHO] ✗ Response Body: {response.text}")
            response.raise_for_status()
            data = response.json()
            records = data.get("data", [])
            seen_ids = set()
            unique_records = []
            for record in records:
                record_id = str(record.get("ID", ""))
                if record_id and record_id not in seen_ids:
                    seen_ids.add(record_id)
                    unique_records.append(record)
            return unique_records
        else:
            all_records = []
            seen_ids = set()
            page = 1
            max_id = None

            while True:
                if max_id is None:
                    params = {"from": 1, "limit": 200}
                else:
                    params = {"criteria": f"ID < {max_id}", "from": 1, "limit": 200}

                response = requests.get(api_url, headers=headers, params=params, timeout=30)
                if response.status_code == 400:
                    try:
                        err_data = response.json()
                        if err_data.get("code") == 9220: # No records exist in this report
                            break
                    except:
                        pass

                if response.status_code != 200:
                    print(f"[ZOHO] ✗ Response Status: {response.status_code}")
                    print(f"[ZOHO] ✗ Response Body: {response.text}")
                response.raise_for_status()
                data = response.json()
                records = data.get("data", [])

                if not records:
                    break

                batch_ids = [int(r.get("ID", 0)) for r in records if r.get("ID")]
                if batch_ids:
                    max_id = min(batch_ids)

                new_records = 0
                for record in records:
                    record_id = str(record.get("ID", ""))
                    if record_id and record_id not in seen_ids:
                        seen_ids.add(record_id)
                        all_records.append(record)
                        new_records += 1

                if new_records == 0:
                    break
                if max_records and len(all_records) >= max_records:
                    all_records = all_records[:max_records]
                    break
                if len(records) < 200:
                    break

                page += 1
                time.sleep(0.5)

            return all_records

    except Exception as e:
        print(f"[ZOHO] ✗ Error: {e}")
        raise


def fetch_specific_records_by_ids(app_link_name: str, report_link_name: str, record_ids: List[str]) -> List[Dict]:
    if not record_ids:
        return []
    try:
        all_records = []
        batch_size = 100
        for i in range(0, len(record_ids), batch_size):
            batch_ids = record_ids[i:i + batch_size]
            criteria_parts = [f'ID == {rid}' for rid in batch_ids]
            criteria = " || ".join(criteria_parts)
            access_token, token_name = get_zoho_token(scope_needed="read")
            if not access_token:
                raise Exception("Failed to get READ token")
            api_url = f"https://creator.zoho.com/api/v2.1/{app_link_name}/report/{report_link_name}"
            headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
            params = {"criteria": criteria, "from": 1, "limit": 200}
            response = requests.get(api_url, headers=headers, params=params, timeout=30)

            if response.status_code == 400:

                try:

                    err_data = response.json()

                    if err_data.get("code") == 9220: # No records exist in this report

                        break

                except:

                    pass


            response.raise_for_status()
            data = response.json()
            all_records.extend(data.get("data", []))
            if i + batch_size < len(record_ids):
                time.sleep(0.5)
        return all_records
    except Exception as e:
        print(f"[ZOHO] ✗ Error: {e}")
        raise


def check_active_jobs_for_records(record_ids: List[str]) -> Optional[str]:
    if not supabase or not record_ids:
        return None
    try:
        response = supabase.table("auto_extraction_jobs").select("job_id, status").in_("status", ["pending", "running"]).execute()
        if not response.data:
            return None
        for job in response.data:
            job_id = job["job_id"]
            results = supabase.table("auto_extraction_results").select("record_id").eq("job_id", job_id).execute()
            if results.data:
                active_record_ids = {str(r["record_id"]) for r in results.data}
                if set(record_ids) & active_record_ids:
                    return job_id
        return None
    except Exception as e:
        print(f"[DUPLICATE CHECK] Error: {e}")
        return None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_cost(input_tokens: int, output_tokens: int, method: str) -> float:
    if method == "gemini_vision":
        cost = (input_tokens / 1_000_000) * 0.075 + (output_tokens / 1_000_000) * 0.30
    else:
        cost = 0.0
    return round(cost, 6)


def get_bank_name_from_ifsc(ifsc_code: str) -> str:
    if not ifsc_code or len(ifsc_code) < 4:
        return "Unknown Bank"
    bank_codes = {
        "SBIN": "State Bank of India", "HDFC": "HDFC Bank", "ICIC": "ICICI Bank",
        "AXIS": "Axis Bank", "PUNB": "Punjab National Bank", "BKID": "Bank of India",
        "CNRB": "Canara Bank", "UBIN": "Union Bank of India", "IOBA": "Indian Overseas Bank",
        "INDB": "IndusInd Bank", "KKBK": "Kotak Mahindra Bank", "YESB": "Yes Bank",
        "IDIB": "IDBI Bank", "BARB": "Bank of Baroda",
    }
    return bank_codes.get(ifsc_code[:4].upper(), f"Bank ({ifsc_code[:4].upper()})")


def validate_file_format(file_content: bytes, filename: str) -> dict:
    result = {"valid": False, "format": "Unknown", "size": len(file_content), "message": ""}
    if len(file_content) < 10:
        result["message"] = "File is too small"
        return result
    if file_content[:4] == b'\x89PNG':
        result["valid"] = True
        result["format"] = "PNG"
    elif file_content[:2] == b'\xff\xd8':
        result["valid"] = True
        result["format"] = "JPEG"
    elif file_content[:4] == b'%PDF':
        result["valid"] = True
        result["format"] = "PDF"
    result["message"] = f"Valid {result['format']} file" if result["valid"] else "Unknown format"
    return result


def download_file_from_url(file_url: str) -> tuple:
    print(f"[DOWNLOAD] Downloading...")
    if file_url.startswith('/api/v2.1/'):
        file_url = f"https://creator.zoho.com{file_url}"

    try:
        headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

        if "creator.zoho.com" in file_url or "creator.zoho.in" in file_url:
            access_token, token_name = get_zoho_token(scope_needed="read")
            if not access_token:
                access_token = get_zoho_access_token()
            if access_token:
                headers["Authorization"] = f"Zoho-oauthtoken {access_token}"

        response = requests.get(file_url, timeout=30, headers=headers, stream=True)
        response.raise_for_status()
        file_content = response.content

        if file_content[:4] == b'%PDF':
            extension = '.pdf'
        elif file_content[:2] == b'\xff\xd8':
            extension = '.jpg'
        elif file_content[:4] == b'\x89PNG':
            extension = '.png'
        elif len(file_content) > 12 and file_content[8:12] == b'WEBP':
            extension = '.webp'
        else:
            content_type = response.headers.get('Content-Type', '').lower()
            if 'pdf' in content_type:
                extension = '.pdf'
            elif 'jpeg' in content_type or 'jpg' in content_type:
                extension = '.jpg'
            elif 'png' in content_type:
                extension = '.png'
            else:
                url_lower = file_url.lower()
                if '.pdf' in url_lower:
                    extension = '.pdf'
                elif any(ext in url_lower for ext in ['.jpg', '.jpeg']):
                    extension = '.jpg'
                elif '.png' in url_lower:
                    extension = '.png'
                else:
                    extension = '.jpg'

        filename = file_url.split('/')[-1].split('?')[0]
        if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.pdf', '.webp']):
            filename = f"downloaded_file{extension}"
        else:
            base_name = filename.rsplit('.', 1)[0]
            filename = f"{base_name}{extension}"

        print(f"[DOWNLOAD] ✓ {filename} ({len(file_content):,} bytes)")
        return file_content, filename

    except Exception as e:
        print(f"[DOWNLOAD] ✗ Error: {str(e)}")
        raise


def convert_filters_to_zoho_criteria(filters: list) -> str:
    if not filters:
        return None
    criteria_parts = []
    for filter_item in filters:
        field = filter_item.get("field", "")
        operator = filter_item.get("operator", "")
        value = filter_item.get("value", "")
        if not field or not operator:
            continue
        if operator == "equals":
            criteria_parts.append(f'{field} == "{value}"')
        elif operator == "not equals":
            criteria_parts.append(f'{field} != "{value}"')
        elif operator in ["is not empty", "is_not_empty"]:
            criteria_parts.append(f'{field} != ""')
        elif operator in ["is empty", "is_empty"]:
            criteria_parts.append(f'{field} == ""')
        elif operator == "contains":
            criteria_parts.append(f'{field}.contains("{value}")')
        elif operator == "does not contain":
            criteria_parts.append(f'!{field}.contains("{value}")')
    return " && ".join(criteria_parts) if criteria_parts else None


def process_single_file(file_content: bytes, filename: str, doc_type: str) -> dict:
    start_time = time.time()
    try:
        def calculate_tokens(text):
            return max(1, len(text) // 4)

        if not USE_GEMINI:
            return {"error": "Gemini Vision not configured", "success": False, "filename": filename}

        image_content = file_content
        original_filename = filename
        converted_from_pdf = False

        if filename.lower().endswith('.pdf'):
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
                filename = filename.replace('.pdf', '.jpg')
                converted_from_pdf = True
            except ImportError:
                return {"error": "pdf2image not installed", "success": False, "filename": original_filename}
            finally:
                if os.path.exists(temp_pdf_path):
                    os.unlink(temp_pdf_path)

        input_tokens = 258

        if doc_type == "bank":
            result = analyze_bank_gemini_vision(image_content, filename)
            if not result.get('bank_name') or result.get('bank_name') == 'null':
                result['bank_name'] = get_bank_name_from_ifsc(result.get('ifsc_code') or '')
        else:
            result = analyze_bill_gemini_vision(image_content, filename)

        image_id = str(uuid.uuid4())
        image_ext = ".jpg" if converted_from_pdf else (os.path.splitext(original_filename)[1] or ".jpg")
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
        result['token_usage'] = {'input_tokens': input_tokens, 'output_tokens': output_tokens, 'total_tokens': input_tokens + output_tokens}
        result['processing_time_ms'] = processing_time_ms
        result['image_url'] = image_url

        return result

    except Exception as e:
        print(f"[PROCESS] ✗ Error: {e}")
        import traceback; traceback.print_exc()
        return {"error": str(e), "success": False, "filename": filename}


# ============================================================
# MAIN EXTRACTION JOB PROCESSOR
# ============================================================

def process_extraction_job(job_id: str, config: Dict):
    """
    Background job processor with flexible field extraction.
    ✅ JPG/PNG with multiple bills → analyze_bill_multi_bills_from_image()
    ✅ PDF with multiple pages   → analyze_bill_multi_page()
    ✅ Falls back gracefully if multi-bill detection fails
    """
    if not supabase:
        print("[AUTO EXTRACT] ✗ Supabase not configured")
        return

    try:
        print(f"\n{'='*80}")
        print(f"[AUTO EXTRACT] Starting Job: {job_id}")
        print(f"{'='*80}\n")

        update_job_status(job_id, {"status": "running", "started_at": datetime.now().isoformat()})

        selected_ids = config.get('selected_record_ids', [])

        print(f"[DEBUG] selected_ids = {selected_ids}")
        print(f"[DEBUG] filter_criteria = {config.get('filter_criteria')}")

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
                criteria=config.get('filter_criteria') if config.get('filter_criteria') else None,
                max_records=None
            )
            
            # Filter out records that already have OCR_Status filled
            # (only when no specific IDs are selected)
            if not selected_ids:
                original_count = len(records)
                records = [r for r in records if not r.get('OCR_Status') or r.get('OCR_Status') == '']
                filtered_count = original_count - len(records)
                if filtered_count > 0:
                    print(f"[AUTO EXTRACT] Filtered out {filtered_count} already processed records")

        print(f"[DEBUG] records fetched = {len(records)}")
        print(f"[AUTO EXTRACT] Processing {len(records)} records...")
        update_job_status(job_id, {"total_records": len(records)})

        if not records:
            update_job_status(job_id, {"status": "completed", "completed_at": datetime.now().isoformat()})
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

        def extract_field_value(record, field_names):
            for field_name in field_names:
                field_value = record.get(field_name)
                if not field_value:
                    continue
                if isinstance(field_value, str):
                    return (field_value.strip() if field_value.strip() else None), True
                if isinstance(field_value, dict):
                    display_value = field_value.get("zc_display_value")
                    if display_value:
                        return display_value.strip(), True
                    id_value = field_value.get("ID")
                    if id_value:
                        return str(id_value), True
            return None, False

        first_record = records[0] if records else {}
        first_keys = set(first_record.keys())

        has_scholar_id = any(k in first_keys for k in ["Actual_Scholar_ID", "Scholar_ID", "ScholarID"])
        has_tracking_id = any(k in first_keys for k in ["Tracking_ID", "Tracking_Id", "TrackingID"])
        has_email = any(k in first_keys for k in ["Email", "email", "student_email"])

        print(f"[AUTO EXTRACT] Field detection:")
        print(f"  Scholar ID:  {has_scholar_id}")
        print(f"  Tracking ID: {has_tracking_id}")
        print(f"  Email:       {has_email}\n")

        for idx, record in enumerate(records, 1):
            record_start = time.time()
            record_id = str(record.get("ID"))
            student_name = extract_student_name(record)

            scholar_id = None
            tracking_id = None
            email = None

            if has_scholar_id:
                scholar_id, _ = extract_field_value(record, ["Actual_Scholar_ID", "Scholar_ID", "ScholarID"])
            if has_tracking_id:
                tracking_id, _ = extract_field_value(record, ["Tracking_ID", "Tracking_Id", "TrackingID"])
            if has_email:
                email, _ = extract_field_value(record, ["Email", "email", "student_email"])

            print(f"\n[AUTO EXTRACT] [{idx}/{len(records)}] {student_name} (ID: {record_id})")
            if scholar_id:
                print(f"[AUTO EXTRACT]   📋 Scholar ID: {scholar_id}")
            if tracking_id:
                print(f"[AUTO EXTRACT]   📋 Tracking ID: {tracking_id}")
            if email:
                print(f"[AUTO EXTRACT]   📧 Email: {email}")

            bank_image_url_supabase = None
            bill_images_supabase    = []
            bank_data               = None
            bill_data_array         = []
            record_tokens           = 0
            record_cost             = 0.0
            error_msg               = None

            try:
                # ── Process Bank Image ────────────────────────────────
                if config.get('bank_field_name'):
                    bank_field_value = record.get(config['bank_field_name'])
                    bank_zoho_urls = extract_all_image_urls(bank_field_value, max_images=1)

                    if bank_zoho_urls:
                        try:
                            print(f"[AUTO EXTRACT]   📥 Downloading bank image...")
                            file_content, filename = download_file_from_url(bank_zoho_urls[0])
                            print(f"[AUTO EXTRACT]   🤖 Processing with Gemini...")
                            result = process_single_file(file_content, filename, "bank")

                            if result.get('success'):
                                bank_data = result
                                tokens = result.get('token_usage', {})
                                record_tokens += tokens.get('total_tokens', 0)
                                record_cost += calculate_cost(tokens.get('input_tokens', 0), tokens.get('output_tokens', 0), 'gemini_vision')
                                print(f"[AUTO EXTRACT]   ✅ Bank extracted")
                            else:
                                error_msg = f"Bank OCR failed: {result.get('error')}"
                                print(f"[AUTO EXTRACT]   ✗ {error_msg}")

                        except Exception as e:
                            error_msg = f"Bank processing error: {str(e)}"
                            print(f"[AUTO EXTRACT]   ✗ {error_msg}")

                # ── Process Bill Images (multi-bill images + multi-page PDFs) ────
                if config.get('bill_field_name'):
                    bill_field_value = record.get(config['bill_field_name'])
                    bill_zoho_urls = extract_all_image_urls(bill_field_value, max_images=5)

                    if bill_zoho_urls:
                        print(f"[AUTO EXTRACT]   📥 Found {len(bill_zoho_urls)} bill file(s) in field")

                        for file_idx, bill_zoho_url in enumerate(bill_zoho_urls, 1):
                            try:
                                print(f"[AUTO EXTRACT]   📥 Downloading bill file {file_idx}...")
                                file_content, filename = download_file_from_url(bill_zoho_url)

                                # ── Detect file type ──────────────────
                                is_pdf = (
                                    file_content[:4] == b'%PDF'
                                    or file_content[:5] == b'\x0a%PDF'
                                )

                                if is_pdf:
                                    # ── Multi-page PDF → one bill per page ──
                                    print(f"[AUTO EXTRACT]   📄 PDF detected for file {file_idx} "
                                          f"– running multi-page bill extraction")

                                    try:
                                        page_bills = analyze_bill_multi_page(file_content, filename)
                                    except Exception as mp_err:
                                        print(f"[AUTO EXTRACT]   ✗ Multi-page extraction failed: {mp_err}")
                                        if not error_msg:
                                            error_msg = f"Bill file {file_idx}: multi-page extraction failed"
                                        page_bills = []

                                    for bill_result in page_bills:
                                        wrapped = {
                                            **bill_result,
                                            "success": True,
                                            "method": "gemini_vision",
                                            "filename": filename,
                                            "source_type": "pdf_page",
                                            "token_usage": {
                                                "input_tokens": 258,
                                                "output_tokens": 80,
                                                "total_tokens": 338,
                                            },
                                        }
                                        bill_data_array.append(wrapped)
                                        record_tokens += 338
                                        record_cost += calculate_cost(258, 80, "gemini_vision")

                                    if page_bills:
                                        print(f"[AUTO EXTRACT]   ✅ File {file_idx}: "
                                              f"{len(page_bills)} bill(s) extracted from PDF")
                                    else:
                                        print(f"[AUTO EXTRACT]   ⚠️ File {file_idx}: no bills in PDF")
                                        if not error_msg:
                                            error_msg = f"Bill file {file_idx}: no bills found in PDF"

                                else:
                                    # ── Single image → may contain MULTIPLE bills ──
                                    print(f"[AUTO EXTRACT]   🖼️ Image detected for file {file_idx} "
                                          f"– scanning for multiple bills in one photo...")

                                    try:
                                        image_bills = analyze_bill_multi_bills_from_image(
                                            file_content, filename
                                        )
                                    except Exception as mb_err:
                                        print(f"[AUTO EXTRACT]   ✗ Multi-bill image extraction failed: {mb_err}")
                                        print(f"[AUTO EXTRACT]   ↩️  Falling back to single-bill extraction...")
                                        single = process_single_file(file_content, filename, "bill")
                                        image_bills = [single] if single.get("success") else []

                                    for bill_result in image_bills:
                                        # Normalise amount
                                        raw_amount = bill_result.get("amount")
                                        if raw_amount in (None, "null", ""):
                                            bill_result["amount"] = None
                                        else:
                                            try:
                                                bill_result["amount"] = float(raw_amount)
                                            except (ValueError, TypeError):
                                                bill_result["amount"] = None

                                        wrapped = {
                                            **bill_result,
                                            "success": True,
                                            "method": "gemini_vision",
                                            "filename": filename,
                                            "source_type": "multi_bill_image",
                                            "token_usage": {
                                                "input_tokens": 300,
                                                "output_tokens": 100,
                                                "total_tokens": 400,
                                            },
                                        }
                                        bill_data_array.append(wrapped)
                                        record_tokens += 400
                                        record_cost += calculate_cost(300, 100, "gemini_vision")

                                    if image_bills:
                                        print(f"[AUTO EXTRACT]   ✅ File {file_idx}: "
                                              f"{len(image_bills)} bill(s) extracted from image")
                                    else:
                                        print(f"[AUTO EXTRACT]   ⚠️ File {file_idx}: no bills detected")
                                        if not error_msg:
                                            error_msg = f"Bill file {file_idx}: no bills detected in image"

                            except Exception as e:
                                print(f"[AUTO EXTRACT]   ✗ Bill file {file_idx} error: {str(e)}")
                                if not error_msg:
                                    error_msg = f"Bill file {file_idx} processing error"

                        # ── Summary log ───────────────────────────────
                        print(f"[AUTO EXTRACT]   📊 Total bills for this record: {len(bill_data_array)}")
                        for i, b in enumerate(bill_data_array, 1):
                            src = b.get("source_type", "?")
                            page = b.get("page_number", "?")
                            bidx = b.get("bill_index", "?")
                            print(f"[AUTO EXTRACT]      Bill {i} "
                                  f"[{src} | page={page} | idx={bidx}]: "
                                  f"Amount=₹{b.get('amount')} | "
                                  f"College={b.get('college_name')}")

                # ── Determine record status ───────────────────────────
                if bank_data or bill_data_array:
                    status = "success"
                    successful += 1
                else:
                    status = "failed"
                    failed += 1
                    if not error_msg:
                        error_msg = "No images found or OCR failed"

            except Exception as record_error:
                status    = "failed"
                failed   += 1
                error_msg = f"Record error: {str(record_error)}"
                print(f"[AUTO EXTRACT]   ✗ {error_msg}")

            processing_time = int((time.time() - record_start) * 1000)

            if supabase:
                try:
                    save_extraction_result(
                        job_id=job_id,
                        record_id=record_id,
                        app_link_name=config['app_link_name'],
                        report_link_name=config['report_link_name'],
                        student_name=student_name,
                        scholar_id=scholar_id,
                        tracking_id=tracking_id,
                        email=email,
                        bank_image_supabase=bank_image_url_supabase,
                        bill_image_supabase=bill_images_supabase if bill_images_supabase else None,
                        bank_data=bank_data,
                        bill_data=bill_data_array if bill_data_array else None,
                        status=status,
                        error_message=error_msg,
                        processing_time_ms=processing_time,
                        tokens_used=record_tokens,
                        cost_usd=record_cost
                    )
                except Exception as save_error:
                    print(f"[AUTO EXTRACT]   ✗ Save failed: {save_error}")

            total_cost += record_cost
            processed += 1

            update_job_status(job_id, {
                "processed_records":  processed,
                "successful_records": successful,
                "failed_records":     failed,
                "total_cost_usd":     round(total_cost, 6)
            })

            print(f"[AUTO EXTRACT]   💰 ${record_cost:.6f} | ⏱️ {processing_time}ms | {status}")
            time.sleep(0.3)

        update_job_status(job_id, {
            "status":        "completed",
            "completed_at":  datetime.now().isoformat(),
            "total_cost_usd": round(total_cost, 6)
        })

        print(f"\n{'='*80}")
        print(f"[AUTO EXTRACT] ✅ Completed: {job_id}")
        print(f"[AUTO EXTRACT]   Success: {successful}/{len(records)}")
        print(f"[AUTO EXTRACT]   Failed:  {failed}/{len(records)}")
        print(f"[AUTO EXTRACT]   Cost:    ${total_cost:.6f}")
        print(f"{'='*80}\n")
        
        # ── Auto Push logic for webhooks ───────────────────────────
        if config.get("auto_push") and successful > 0:
            print(f"[AUTO EXTRACT] 🚀 Auto-pushing {successful} records to Zoho Creator...")
            try:
                # We need the supabase IDs (uuid) that were successful, not the Zoho record IDs
                success_supabase_ids = []
                for rec_id in selected_ids:
                    # Query supabase for successful records from this job specifically
                    res = supabase.table('auto_extraction_results').select('id, status').eq('job_id', job_id).eq('record_id', rec_id).execute()
                    if res.data and res.data[0]['status'] == 'success':
                        success_supabase_ids.append(res.data[0]['id'])
                
                if success_supabase_ids:
                    payload = {
                        "config": {
                            "owner_name": config.get("target_owner_name", "teameverest"),
                            "app_name": config.get("target_app_name", "iatc-scholarship"),
                            "form_name": config.get("target_form_name", "OCR_Extraction_From")
                        },
                        "record_ids": success_supabase_ids,
                        "sync_mode": "auto"
                    }
                    import threading
                    def fire_and_forget():
                        try:
                            # Direct HTTP request works universally
                            requests.post("http://127.0.0.1:8000/zoho/sync-records", json=payload, timeout=60)
                        except Exception as e:
                            print(f"[AUTO PUSH] Failed: {e}")
                    threading.Thread(target=fire_and_forget, daemon=True).start()
            except Exception as auto_push_error:
                print(f"[AUTO EXTRACT] ✗ Auto-push failed: {auto_push_error}")

    except Exception as e:
        print(f"[AUTO EXTRACT] ✗ Job failed: {e}")
        import traceback; traceback.print_exc()
        update_job_status(job_id, {"status": "failed", "completed_at": datetime.now().isoformat()})


# ============================================================
# FORMAT RECORD FOR ZOHO - UPDATED FOR MULTI-BILL IMAGES
# ============================================================

def format_record_for_zoho_creator(record: Dict) -> Dict:
    """
    Format extraction result for Zoho Creator form.
    ✅ Handles bills from multi-page PDFs  (source_type = "pdf_page")
    ✅ Handles bills from multi-bill images (source_type = "multi_bill_image")
    ✅ Handles legacy single-bill results  (no source_type)
    """
    bill_data = record.get('bill_data', {})
    bank_data = record.get('bank_data', {})

    if isinstance(bill_data, str):
        try:
            bill_data = json.loads(bill_data)
        except Exception:
            bill_data = {}

    if isinstance(bank_data, str):
        try:
            bank_data = json.loads(bank_data)
        except Exception:
            bank_data = {}

    bill_data_array = []
    if isinstance(bill_data, list):
        bill_data_array = bill_data
    elif isinstance(bill_data, dict) and bill_data:
        bill_data_array = [bill_data]

    def safe_float(value, default=0.0):
        if value in (None, "null", "", "None"):
            return default
        try:
            return float(value)
        except Exception:
            return default

    def get_bill_amount(bill):
        return safe_float(bill.get('amount'))

    def slot_bills(bills):
        """Slot bills into 5 positions, honouring bill_index if present."""
        slots = [None] * 5
        for bill in bills[:5]:
            idx = bill.get('bill_index')
            if idx is not None:
                pos = int(idx) - 1
                if 0 <= pos < 5:
                    slots[pos] = bill
                    continue
            for i in range(5):
                if slots[i] is None:
                    slots[i] = bill
                    break
        return slots

    slots = slot_bills(bill_data_array)

    bill1_amount = get_bill_amount(slots[0]) if slots[0] else 0.0
    bill2_amount = get_bill_amount(slots[1]) if slots[1] else 0.0
    bill3_amount = get_bill_amount(slots[2]) if slots[2] else 0.0
    bill4_amount = get_bill_amount(slots[3]) if slots[3] else 0.0
    bill5_amount = get_bill_amount(slots[4]) if slots[4] else 0.0
    
    # Extended to 8 bills
    bill6_amount = get_bill_amount(bill_data_array[5]) if len(bill_data_array) > 5 else 0.0
    bill7_amount = get_bill_amount(bill_data_array[6]) if len(bill_data_array) > 6 else 0.0
    bill8_amount = get_bill_amount(bill_data_array[7]) if len(bill_data_array) > 7 else 0.0
    
    total_amount = bill1_amount + bill2_amount + bill3_amount + bill4_amount + bill5_amount + bill6_amount + bill7_amount + bill8_amount

    scholar_name = (
        record.get('Scholar_Name') or
        record.get('student_name') or
        (slots[0].get('student_name') if slots[0] else '') or ''
    )

    scholar_id = (
        record.get('Scholar_ID') or
        record.get('scholar_id') or ''
    )
    
    tracking_id = (
        record.get('Tracking_ID') or
        record.get('tracking_id') or
        record.get('record_id') or ''
    )

    def format_bill_text(bill, label):
        if not bill:
            return None
        amount = bill.get('amount')
        if amount in (None, 'null', ''):
            amount_str = "⚠️ Not extracted"
        else:
            amount_str = f"₹{safe_float(amount):,.2f}"
        src = bill.get('source_type', '')
        src_tag = f" [{src}]" if src else ""
        return (
            f"{label}: "
            f"Student: {bill.get('student_name', 'N/A')} | "
            f"College: {bill.get('college_name', 'N/A')} | "
            f"Receipt: {bill.get('receipt_number', 'N/A')} | "
            f"Amount: {amount_str}{src_tag}"
        )

    bill_texts = []
    for i, slot in enumerate(slots, 1):
        text = format_bill_text(slot, f"Bill {i}")
        if text:
            bill_texts.append(text)

    bill_data_str = " || ".join(bill_texts) if bill_texts else ""

    has_null_amounts = any(
        b.get('amount') in (None, 'null', '') for b in bill_data_array
    ) if bill_data_array else False

    status = record.get('status', 'completed')
    if has_null_amounts:
        status = "⚠️ Review - Missing amounts"

    return {
        "Scholar_Name": scholar_name,
        "Scholar_ID": scholar_id,
        "Tracking_ID": tracking_id,
        "Account_No": bank_data.get('account_number', ''),
        "Bank_Name": bank_data.get('bank_name', ''),
        "Holder_Name": bank_data.get('account_holder_name', ''),
        "IFSC_Code": bank_data.get('ifsc_code', ''),
        "Branch_Name": bank_data.get('branch_name', ''),
        "Bill_Data": bill_data_str,
        "Bill1_Amount": bill1_amount,
        "Bill2_Amount": bill2_amount,
        "Bill3_Amount": bill4_amount,
        "Bill4_Amount": bill4_amount,
        "Bill5_Amount": bill5_amount,
        "Bill6_Amount": bill6_amount,
        "Bill7_Amount": bill7_amount,
        "Bill8_Amount": bill8_amount,
        "Total_Amount": total_amount,
        "Status": status,
        "Tokens_Used": record.get('tokens_used', 0),
        "Cost_USD": float(record.get('cost_usd', 0)),
    }


# ============================================================
# WORKDRIVE BARCODE EXTRACTION
# ============================================================

@app.post("/barcode/workdrive/list-files")
async def list_workdrive_files(folder_id: str = Form(...)):
    try:
        files = fetch_workdrive_files(folder_id)
        return JSONResponse(content={"success": True, "total_files": len(files), "files": files})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/barcode/local/start")
async def start_local_barcode_extraction(request: Request):
    if not supabase:
        return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured"})

    try:
        form = await request.form()
        uploaded_files = form.getlist("files")

        if not uploaded_files:
            return JSONResponse(status_code=400, content={"success": False, "error": "No files provided"})

        job_id = f"barcode_local_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        temp_files = []

        for uploaded_file in uploaded_files:
            file_content = await uploaded_file.read()
            temp_file_path = f"temp_uploads/{job_id}_{uploaded_file.filename}"
            os.makedirs("temp_uploads", exist_ok=True)
            with open(temp_file_path, "wb") as f:
                f.write(file_content)
            temp_files.append({
                "file_id": f"local_{uuid.uuid4().hex}",
                "filename": uploaded_file.filename,
                "temp_path": temp_file_path,
                "size": len(file_content)
            })

        supabase.table("barcode_extraction_jobs").insert({
            "job_id": job_id, "source": "local",
            "total_files": len(temp_files), "status": "pending"
        }).execute()

        config = {"job_id": job_id, "source": "local", "temp_files": temp_files}
        thread = threading.Thread(target=process_local_barcode_extraction_job, args=(job_id, config))
        thread.daemon = True
        thread.start()

        return JSONResponse(content={
            "success": True, "job_id": job_id, "status": "started",
            "message": f"🚀 Processing {len(temp_files)} files",
            "check_status_url": f"/barcode/workdrive/status/{job_id}"
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


def process_local_barcode_extraction_job(job_id: str, config: Dict):
    if not supabase:
        return

    try:
        supabase.table("barcode_extraction_jobs").update({
            "status": "running", "started_at": datetime.now().isoformat()
        }).eq("job_id", job_id).execute()

        temp_files = config['temp_files']
        total_cost = 0.0
        processed = successful = failed = total_barcodes_extracted = 0

        for idx, file_info in enumerate(temp_files, 1):
            record_start = time.time()
            file_id = file_info['file_id']
            filename = file_info['filename']
            temp_path = file_info['temp_path']

            print(f"\n[LOCAL BARCODE] [{idx}/{len(temp_files)}] Processing: {filename}")

            try:
                with open(temp_path, "rb") as f:
                    file_content = f.read()

                result = analyze_barcode_gemini_vision(file_content, filename)

                if result.get('success'):
                    supabase_url = None
                    tokens = result.get('token_usage', {})
                    cost = calculate_cost(tokens.get('input_tokens', 0), tokens.get('output_tokens', 0), 'gemini_vision')
                    total_cost += cost

                    all_barcodes = result.get('all_barcodes', [])
                    total_found = len(all_barcodes)
                    is_multipage = result.get('is_multipage', False)
                    pages_processed = result.get('total_pages_processed', 1)
                    total_barcodes_extracted += total_found

                    if total_found > 0:
                        saved_count = 0
                        for barcode_idx, barcode in enumerate(all_barcodes, 1):
                            try:
                                if barcode_idx > 1:
                                    time.sleep(0.1)
                                supabase.table("barcode_extraction_results").insert({
                                    "job_id": job_id, "file_id": file_id, "filename": filename,
                                    "barcode_number": barcode_idx, "total_barcodes_in_file": total_found,
                                    "barcode_data": barcode.get('data'), "barcode_type": barcode.get('type'),
                                    "page_number": barcode.get('page', 1), "is_primary": barcode_idx == 1,
                                    "image_url": supabase_url, "status": "success",
                                    "tokens_used": tokens.get('total_tokens', 0),
                                    "cost_usd": cost / total_found if total_found > 0 else 0,
                                    "processing_time_ms": int((time.time() - record_start) * 1000),
                                    "is_multipage": is_multipage, "pages_processed": pages_processed, "source": "local"
                                }).execute()
                                saved_count += 1
                            except Exception as save_error:
                                print(f"[LOCAL BARCODE]   ❌ Failed barcode {barcode_idx}: {save_error}")

                        if saved_count == total_found:
                            successful += 1
                        else:
                            failed += 1
                    else:
                        failed += 1
                        supabase.table("barcode_extraction_results").insert({
                            "job_id": job_id, "file_id": file_id, "filename": filename,
                            "status": "no_barcode", "error_message": "No barcodes detected", "source": "local"
                        }).execute()
                else:
                    failed += 1
                    supabase.table("barcode_extraction_results").insert({
                        "job_id": job_id, "file_id": file_id, "filename": filename,
                        "status": "failed", "error_message": result.get('error'), "source": "local"
                    }).execute()

            except Exception as file_error:
                failed += 1
                try:
                    supabase.table("barcode_extraction_results").insert({
                        "job_id": job_id, "file_id": file_id, "filename": filename,
                        "status": "failed", "error_message": str(file_error), "source": "local"
                    }).execute()
                except:
                    pass
            finally:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except:
                    pass

            processed += 1
            try:
                supabase.table("barcode_extraction_jobs").update({
                    "processed_files": processed, "successful_files": successful,
                    "failed_files": failed, "total_cost_usd": round(total_cost, 6),
                    "total_barcodes_extracted": total_barcodes_extracted
                }).eq("job_id", job_id).execute()
            except:
                pass

        supabase.table("barcode_extraction_jobs").update({
            "status": "completed", "completed_at": datetime.now().isoformat(),
            "total_cost_usd": round(total_cost, 6), "total_barcodes_extracted": total_barcodes_extracted
        }).eq("job_id", job_id).execute()

    except Exception as e:
        print(f"[LOCAL BARCODE] ✗ Job failed: {e}")
        try:
            supabase.table("barcode_extraction_jobs").update({
                "status": "failed", "completed_at": datetime.now().isoformat()
            }).eq("job_id", job_id).execute()
        except:
            pass


@app.post("/barcode/workdrive/preview")
async def preview_workdrive_extraction(folder_id: str = Form(...), selected_file_ids: Optional[str] = Form(None)):
    try:
        files = fetch_workdrive_files(folder_id)
        if selected_file_ids:
            selected_ids = json.loads(selected_file_ids)
            files = [f for f in files if f["file_id"] in selected_ids]
        return JSONResponse(content={
            "success": True, "total_files": len(files),
            "estimated_cost": f"${len(files) * 0.002:.4f}",
            "estimated_time_minutes": math.ceil(len(files) * 2 / 60),
            "files": [{"file_id": f["file_id"], "filename": f["filename"], "size": f["size"]} for f in files]
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/barcode/workdrive/start")
async def start_workdrive_barcode_extraction(folder_id: str = Form(...), selected_file_ids: str = Form(...)):
    if not supabase:
        return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured"})
    try:
        selected_ids = json.loads(selected_file_ids)
        if not selected_ids:
            return JSONResponse(status_code=400, content={"success": False, "error": "No files selected"})

        job_id = f"barcode_job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        supabase.table("barcode_extraction_jobs").insert({
            "job_id": job_id, "folder_id": folder_id,
            "total_files": len(selected_ids), "status": "pending"
        }).execute()

        config = {"folder_id": folder_id, "selected_file_ids": selected_ids}
        thread = threading.Thread(target=process_barcode_extraction_job, args=(job_id, config))
        thread.daemon = True
        thread.start()

        return JSONResponse(content={
            "success": True, "job_id": job_id, "status": "started",
            "message": f"🚀 Processing {len(selected_ids)} files",
            "check_status_url": f"/barcode/workdrive/status/{job_id}"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


def process_barcode_extraction_job(job_id: str, config: Dict):
    if not supabase:
        return
    try:
        supabase.table("barcode_extraction_jobs").update({
            "status": "running", "started_at": datetime.now().isoformat()
        }).eq("job_id", job_id).execute()

        selected_ids = config['selected_file_ids']
        total_cost = 0.0
        processed = successful = failed = total_barcodes_extracted = 0

        for idx, file_id in enumerate(selected_ids, 1):
            record_start = time.time()
            print(f"\n[BARCODE] [{idx}/{len(selected_ids)}] Processing file: {file_id}")

            try:
                file_content, filename = download_workdrive_file(file_id)
                result = analyze_barcode_gemini_vision(file_content, filename)

                if result.get('success'):
                    supabase_url = None
                    try:
                        supabase_url = upload_to_supabase_storage(file_content, f"barcode_{file_id}_{filename}", folder="barcode-extractions")
                    except Exception as upload_error:
                        print(f"[BARCODE]   ⚠️ Upload failed: {upload_error}")

                    tokens = result.get('token_usage', {})
                    cost = calculate_cost(tokens.get('input_tokens', 0), tokens.get('output_tokens', 0), 'gemini_vision')
                    total_cost += cost

                    all_barcodes = result.get('all_barcodes', [])
                    total_found = len(all_barcodes)
                    is_multipage = result.get('is_multipage', False)
                    pages_processed = result.get('total_pages_processed', 1)
                    total_barcodes_extracted += total_found

                    if total_found > 0:
                        for barcode_idx, barcode in enumerate(all_barcodes, 1):
                            try:
                                supabase.table("barcode_extraction_results").insert({
                                    "job_id": job_id, "file_id": file_id, "filename": filename,
                                    "barcode_number": barcode_idx, "total_barcodes_in_file": total_found,
                                    "barcode_data": barcode.get('data'), "barcode_type": barcode.get('type'),
                                    "page_number": barcode.get('page', 1), "is_primary": barcode_idx == 1,
                                    "image_url": supabase_url, "status": "success",
                                    "tokens_used": tokens.get('total_tokens', 0),
                                    "cost_usd": cost / total_found if total_found > 0 else 0,
                                    "processing_time_ms": int((time.time() - record_start) * 1000),
                                    "is_multipage": is_multipage, "pages_processed": pages_processed
                                }).execute()
                            except Exception as save_error:
                                print(f"[BARCODE]   ⚠️ Failed to save barcode {barcode_idx}: {save_error}")
                        successful += 1
                    else:
                        failed += 1
                        supabase.table("barcode_extraction_results").insert({
                            "job_id": job_id, "file_id": file_id, "filename": filename,
                            "status": "no_barcode", "error_message": "No barcodes detected",
                            "processing_time_ms": int((time.time() - record_start) * 1000)
                        }).execute()
                else:
                    failed += 1
                    supabase.table("barcode_extraction_results").insert({
                        "job_id": job_id, "file_id": file_id, "filename": filename,
                        "status": "failed", "error_message": result.get('error'),
                        "processing_time_ms": int((time.time() - record_start) * 1000)
                    }).execute()

            except Exception as file_error:
                failed += 1
                try:
                    supabase.table("barcode_extraction_results").insert({
                        "job_id": job_id, "file_id": file_id,
                        "status": "failed", "error_message": str(file_error)
                    }).execute()
                except:
                    pass

            processed += 1
            try:
                supabase.table("barcode_extraction_jobs").update({
                    "processed_files": processed, "successful_files": successful,
                    "failed_files": failed, "total_cost_usd": round(total_cost, 6),
                    "total_barcodes_extracted": total_barcodes_extracted
                }).eq("job_id", job_id).execute()
            except:
                pass

            time.sleep(0.5)

        supabase.table("barcode_extraction_jobs").update({
            "status": "completed", "completed_at": datetime.now().isoformat(),
            "total_cost_usd": round(total_cost, 6), "total_barcodes_extracted": total_barcodes_extracted
        }).eq("job_id", job_id).execute()

    except Exception as e:
        print(f"[BARCODE] ✗ Job failed: {e}")
        try:
            supabase.table("barcode_extraction_jobs").update({
                "status": "failed", "completed_at": datetime.now().isoformat()
            }).eq("job_id", job_id).execute()
        except:
            pass


@app.get("/barcode/workdrive/status/{job_id}")
async def get_barcode_job_status(job_id: str):
    if not supabase:
        return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured"})
    try:
        response = supabase.table("barcode_extraction_jobs").select("*").eq("job_id", job_id).execute()
        if not response.data:
            return JSONResponse(status_code=404, content={"success": False, "error": "Job not found"})
        job = response.data[0]
        progress_percent = 0
        if job.get("total_files", 0) > 0:
            progress_percent = round((job.get("processed_files", 0) / job["total_files"]) * 100, 2)
        return JSONResponse(content={
            "success": True, "job_id": job_id, "status": job["status"],
            "progress": {
                "total_files": job.get("total_files", 0),
                "processed_files": job.get("processed_files", 0),
                "successful_files": job.get("successful_files", 0),
                "failed_files": job.get("failed_files", 0),
                "progress_percent": progress_percent
            },
            "cost": {"total_cost_usd": float(job.get("total_cost_usd", 0))},
            "timestamps": {
                "created_at": job.get("created_at"),
                "started_at": job.get("started_at"),
                "completed_at": job.get("completed_at")
            }
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/barcode/workdrive/results/{job_id}")
async def get_barcode_job_results(job_id: str, limit: int = 100):
    if not supabase:
        return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured"})
    try:
        response = supabase.table("barcode_extraction_results").select("*").eq("job_id", job_id).order("created_at", desc=True).limit(limit).execute()
        return JSONResponse(content={"success": True, "job_id": job_id, "total_results": len(response.data), "results": response.data})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ============================================================
# AUTHENTICATION ENDPOINTS
# ============================================================

sessions = {}

class LoginRequest(BaseModel):
    username: str
    password: str

class User(BaseModel):
    id: str
    username: str
    email: str
    name: str
    groups: List[str] = []
    is_admin: bool = False
    user_type: Optional[str] = None
    avatar: Optional[str] = None

class LoginResponse(BaseModel):
    success: bool
    message: str
    user: Optional[User] = None

AUTHENTIK_URL = os.getenv("AUTHENTIK_URL", "https://authentik.teameverest.ngo")
AUTHENTIK_API_TOKEN = os.getenv("AUTHENTIK_API_TOKEN", "KPu2Ow7RVjIZFFZ5DdXSu1LqaI5oxFBPxsnORgQHykECbiCYoReoBE5vEx2U")

def create_session_token() -> str:
    return secrets.token_urlsafe(32)

async def find_authentik_user(username: str) -> Optional[Dict]:
    if not AUTHENTIK_API_TOKEN:
        return None
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                f"{AUTHENTIK_URL}/api/v3/core/users/",
                headers={"Authorization": f"Bearer {AUTHENTIK_API_TOKEN}"},
                params={"username": username},
                timeout=10.0
            )
            response.raise_for_status()
            data = response.json()
            if data.get("results") and len(data["results"]) > 0:
                return data["results"][0]
            return None
        except Exception as e:
            print(f"Error finding user: {e}")
            return None

def get_user_type(groups: List[Dict]) -> Optional[str]:
    group_names = [g.get("name", "") for g in groups]
    if "ECR Student" in group_names:
        return "ecr_student"
    elif "ICM Student" in group_names:
        return "icm_student"
    elif "IATC Admin" in group_names:
        return "admin"
    return None

async def require_auth(request: Request) -> User:
    session_token = request.cookies.get("session_token")
    if not session_token or session_token not in sessions:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    session = sessions[session_token]
    if datetime.now() > session["expires_at"]:
        del sessions[session_token]
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired"
        )
    return User(**session["user"])

@app.post("/auth/login", response_model=LoginResponse)
async def login(login_data: LoginRequest, response: Response):
    try:
        username = login_data.username
        password = login_data.password

        if not username or '@' in username:
            return LoginResponse(success=False, message="Please login with your username, not email address")
        if not password:
            return LoginResponse(success=False, message="Password is required")

        print(f"\n🔐 LOGIN ATTEMPT for user: {username}")

        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            flow_response = await client.post(f"{AUTHENTIK_URL}/api/v3/flows/executor/default-authentication-flow/", json={})
            cookies = flow_response.cookies

            id_response = await client.post(
                f"{AUTHENTIK_URL}/api/v3/flows/executor/default-authentication-flow/",
                json={"uid_field": username}, cookies=cookies
            )
            if id_response.cookies:
                cookies.update(id_response.cookies)

            try:
                id_data = id_response.json()
            except:
                return LoginResponse(success=False, message="Invalid username. Please check and try again.")

            if id_data.get("component") != "ak-stage-password":
                return LoginResponse(success=False, message="Invalid username. Please check and try again.")

            password_response = await client.post(
                f"{AUTHENTIK_URL}/api/v3/flows/executor/default-authentication-flow/",
                json={"password": password}, cookies=cookies
            )

            try:
                password_data = password_response.json()
            except:
                return LoginResponse(success=False, message="Authentication failed")

            component = password_data.get("component")

            if component == "xak-flow-redirect":
                authentik_user = await find_authentik_user(username)
                if not authentik_user:
                    return LoginResponse(success=False, message="User not found in system")

                user_groups = authentik_user.get("groups_obj", authentik_user.get("groups", []))
                user_type = get_user_type(user_groups)
                is_admin = any(g.get("name") == "IATC Admin" for g in user_groups)
                group_names = [g.get("name") for g in user_groups]

                user = User(
                    id=str(authentik_user.get("pk")),
                    username=authentik_user.get("username"),
                    email=authentik_user.get("email", ""),
                    name=authentik_user.get("name", authentik_user.get("username")),
                    groups=group_names, is_admin=is_admin,
                    user_type=user_type, avatar=authentik_user.get("avatar", "")
                )

                session_token = create_session_token()
                sessions[session_token] = {
                    "user": user.model_dump(), "created_at": datetime.now(),
                    "expires_at": datetime.now() + timedelta(hours=24),
                    "login_time": datetime.now().isoformat(), "auth_method": "password"
                }

                response.set_cookie(key="session_token", value=session_token, httponly=True, secure=False, samesite="lax", max_age=86400)
                return LoginResponse(success=True, message="Authentication successful", user=user)

            elif component in ("ak-stage-identification", "ak-stage-password"):
                return LoginResponse(success=False, message="Invalid username or password.")
            else:
                return LoginResponse(success=False, message="Authentication failed. Please try again.")

    except httpx.ConnectError:
        return LoginResponse(success=False, message="Authentication service unavailable")
    except Exception as e:
        import traceback; traceback.print_exc()
        return LoginResponse(success=False, message="Authentication failed")


@app.get("/auth/me", response_model=User)
async def get_me(current_user: User = Depends(require_auth)):
    return current_user


@app.post("/auth/logout")
async def logout(request: Request, response: Response):
    session_token = request.cookies.get("session_token")
    if session_token and session_token in sessions:
        del sessions[session_token]
    response.delete_cookie("session_token")
    return {"success": True, "message": "Logged out successfully"}


@app.get("/auth/refresh")
async def refresh_token(request: Request):
    session_token = request.cookies.get("session_token")
    if not session_token or session_token not in sessions:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    sessions[session_token]["expires_at"] = datetime.now() + timedelta(hours=24)
    return {"success": True, "message": "Session refreshed"}


@app.get("/auth/sessions")
async def get_active_sessions(current_user: User = Depends(require_auth)):
    if not current_user.is_admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return {
        "success": True, "active_sessions": len(sessions),
        "sessions": [
            {"username": s["user"]["username"], "created_at": s["created_at"].isoformat(), "expires_at": s["expires_at"].isoformat()}
            for s in sessions.values()
        ]
    }


# ============================================================
# BANK FOLDER EXTRACTION
# ============================================================

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.pdf', '.bmp'}


def scan_folder_for_images(folder_path: str, recursive: bool = False) -> List[Dict]:
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Path is not a folder: {folder_path}")

    files = []
    walk = os.walk(folder_path) if recursive else [(folder_path, [], os.listdir(folder_path))]

    for root, dirs, filenames in walk:
        for fname in filenames:
            if fname.startswith('.'):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, fname)
                if os.path.isfile(full_path):
                    files.append({
                        "file_id": f"local_{uuid.uuid4().hex}",
                        "filename": fname, "filepath": full_path,
                        "size": os.path.getsize(full_path),
                        "rel_path": os.path.relpath(full_path, folder_path)
                    })

    files.sort(key=lambda x: x["rel_path"])
    return files


@app.post("/bank/folder/preview")
async def preview_bank_folder(folder_path: str = Form(...), recursive: str = Form("false")):
    try:
        is_recursive = recursive.lower() in ("true", "1", "yes")
        files = scan_folder_for_images(folder_path, recursive=is_recursive)
        return JSONResponse(content={
            "success": True, "folder_path": folder_path, "recursive": is_recursive,
            "total_files": len(files),
            "estimated_cost": f"${len(files) * 0.002:.4f}",
            "estimated_time_minutes": max(1, math.ceil(len(files) * 3 / 60)),
            "files": [{"filename": f["filename"], "rel_path": f["rel_path"], "size_kb": round(f["size"] / 1024, 1)} for f in files]
        })
    except (FileNotFoundError, NotADirectoryError) as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/bank/folder/start")
async def start_bank_folder_extraction(folder_path: str = Form(...), recursive: str = Form("false")):
    if not supabase:
        return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured"})

    try:
        is_recursive = recursive.lower() in ("true", "1", "yes")
        files = scan_folder_for_images(folder_path, recursive=is_recursive)

        if not files:
            return JSONResponse(status_code=400, content={"success": False, "error": f"No supported files found in: {folder_path}"})

        job_id = f"bank_folder_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        supabase.table("auto_extraction_jobs").insert({
            "job_id": job_id, "app_link_name": "local_folder",
            "report_link_name": folder_path, "bank_field_name": "local_file",
            "bill_field_name": None, "status": "pending",
            "total_records": len(files), "folder_path": folder_path
        }).execute()

        thread = threading.Thread(target=process_bank_folder_job, args=(job_id, files, folder_path))
        thread.daemon = True
        thread.start()

        return JSONResponse(content={
            "success": True, "job_id": job_id, "status": "started",
            "folder_path": folder_path, "total_files": len(files),
            "message": f"🚀 Processing {len(files)} file(s) from folder",
            "check_status_url": f"/bank/folder/status/{job_id}",
            "results_url": f"/bank/folder/results/{job_id}"
        })

    except (FileNotFoundError, NotADirectoryError) as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


def process_bank_folder_job(job_id: str, files: List[Dict], folder_path: str):
    if not supabase:
        return

    try:
        supabase.table("auto_extraction_jobs").update({
            "status": "running", "started_at": datetime.now().isoformat()
        }).eq("job_id", job_id).execute()

        processed = successful = failed = 0
        total_cost = 0.0

        for idx, file_info in enumerate(files, 1):
            file_id = file_info["file_id"]
            filename = file_info["filename"]
            filepath = file_info["filepath"]
            rel_path = file_info["rel_path"]
            t0 = time.time()

            print(f"[BANK FOLDER] [{idx}/{len(files)}] {rel_path}")

            try:
                with open(filepath, "rb") as f:
                    file_content = f.read()

                validation = validate_file_format(file_content, filename)
                if not validation["valid"]:
                    raise ValueError(f"Invalid format: {validation['message']}")

                result = process_single_file(file_content, filename, "bank")

                if not result.get("success"):
                    raise RuntimeError(result.get("error", "OCR failed"))

                tokens = result.get("token_usage", {})
                cost = calculate_cost(tokens.get("input_tokens", 0), tokens.get("output_tokens", 0), "gemini_vision")
                total_cost += cost
                processing_ms = int((time.time() - t0) * 1000)

                supabase.table("auto_extraction_results").insert({
                    "job_id": job_id, "record_id": file_id,
                    "app_link_name": "local_folder", "report_link_name": folder_path,
                    "student_name": filename, "bank_data": result, "bill_data": None,
                    "status": "success", "processing_time_ms": processing_ms,
                    "tokens_used": tokens.get("total_tokens", 0), "cost_usd": cost,
                    "processed_at": datetime.now().isoformat(), "rel_path": rel_path
                }).execute()

                successful += 1
                print(f"[BANK FOLDER]   ✅ {result.get('bank_name')} | A/C: {result.get('account_number')} | ${cost:.6f}")

            except Exception as err:
                failed += 1
                print(f"[BANK FOLDER]   ✗ {err}")
                try:
                    supabase.table("auto_extraction_results").insert({
                        "job_id": job_id, "record_id": file_id,
                        "app_link_name": "local_folder", "report_link_name": folder_path,
                        "student_name": filename, "status": "failed",
                        "error_message": str(err), "processed_at": datetime.now().isoformat(), "rel_path": rel_path
                    }).execute()
                except:
                    pass

            processed += 1
            try:
                supabase.table("auto_extraction_jobs").update({
                    "processed_records": processed, "successful_records": successful,
                    "failed_records": failed, "total_cost_usd": round(total_cost, 6)
                }).eq("job_id", job_id).execute()
            except:
                pass

            time.sleep(0.3)

        supabase.table("auto_extraction_jobs").update({
            "status": "completed", "completed_at": datetime.now().isoformat(),
            "total_cost_usd": round(total_cost, 6)
        }).eq("job_id", job_id).execute()

    except Exception as e:
        print(f"[BANK FOLDER] ✗ Job crashed: {e}")
        try:
            supabase.table("auto_extraction_jobs").update({
                "status": "failed", "completed_at": datetime.now().isoformat()
            }).eq("job_id", job_id).execute()
        except:
            pass


@app.get("/bank/folder/status/{job_id}")
async def get_bank_folder_status(job_id: str):
    if not supabase:
        return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured"})
    try:
        resp = supabase.table("auto_extraction_jobs").select("*").eq("job_id", job_id).execute()
        if not resp.data:
            return JSONResponse(status_code=404, content={"success": False, "error": "Job not found"})
        job = resp.data[0]
        total = job.get("total_records", 0)
        done = job.get("processed_records", 0)
        pct = round((done / total) * 100, 2) if total else 0
        return JSONResponse(content={
            "success": True, "job_id": job_id, "status": job["status"],
            "folder_path": job.get("folder_path"),
            "progress": {
                "total_files": total, "processed_files": done,
                "successful_files": job.get("successful_records", 0),
                "failed_files": job.get("failed_records", 0), "progress_percent": pct
            },
            "cost": {"total_cost_usd": float(job.get("total_cost_usd", 0))},
            "timestamps": {"created_at": job.get("created_at"), "started_at": job.get("started_at"), "completed_at": job.get("completed_at")}
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/bank/folder/results/{job_id}")
async def get_bank_folder_results(job_id: str, limit: int = 500, status_filter: Optional[str] = None):
    if not supabase:
        return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured"})
    try:
        query = supabase.table("auto_extraction_results").select(
            "id, record_id, student_name, rel_path, status, bank_data, error_message, cost_usd, processing_time_ms, processed_at"
        ).eq("job_id", job_id).order("processed_at", desc=False).limit(limit)
        if status_filter:
            query = query.eq("status", status_filter)
        resp = query.execute()

        flat_results = []
        for row in resp.data:
            bank = row.get("bank_data") or {}
            flat_results.append({
                "filename": row.get("student_name"), "rel_path": row.get("rel_path"),
                "status": row.get("status"),
                "bank_name": bank.get("bank_name"), "account_number": bank.get("account_number"),
                "account_holder": bank.get("account_holder_name"), "ifsc_code": bank.get("ifsc_code"),
                "branch_name": bank.get("branch_name"), "error_message": row.get("error_message"),
                "cost_usd": row.get("cost_usd"), "processing_time_ms": row.get("processing_time_ms")
            })

        return JSONResponse(content={"success": True, "job_id": job_id, "total_results": len(flat_results), "results": flat_results})

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "OCR API - Complete with Auto-Extract + Auth + Multi-Bill Image",
        "version": "8.0",
        "features": [
            "✅ Gemini Vision OCR",
            "✅ Auto-extraction with job tracking",
            "✅ Multi-token OAuth support",
            "✅ Supabase integration",
            "✅ PDF support",
            "✅ Bank & Bill extraction",
            "✅ Batch processing",
            "✅ Authentik authentication",
            "✅ Multi-bill extraction from single JPG/PNG images",
        ],
        "supabase_status": "✅ Connected" if supabase else "❌ Not configured",
        "gemini_vision_status": "✅ Available" if USE_GEMINI else "❌ Not configured",
        "authentication_status": "✅ Enabled" if AUTHENTIK_URL and AUTHENTIK_API_TOKEN else "❌ Not configured",
        "tokens_configured": len(ZOHO_TOKENS),
    }


@app.post("/ocr/bank")
async def process_bank_passbook(
    request: Request,
    files: List[UploadFile] = File(default=[]),
    file: Optional[UploadFile] = File(None),
    File_upload_Bank: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    File_upload_Bank_url: Optional[str] = Form(None),
    scholarship_id: Optional[str] = Form(None),
    student_name: Optional[str] = Form(None)
):
    user_info = "user"

    try:
        files_to_process = []

        if files:
            for uploaded_file in files:
                file_content = await uploaded_file.read()
                files_to_process.append((file_content, uploaded_file.filename))
        elif file or File_upload_Bank:
            uploaded_file = file or File_upload_Bank
            file_content = await uploaded_file.read()
            files_to_process.append((file_content, uploaded_file.filename))
        elif file_url or File_upload_Bank_url:
            url = file_url or File_upload_Bank_url
            file_content, filename = download_file_from_url(url)
            files_to_process.append((file_content, filename))
        else:
            return JSONResponse(status_code=400, content={"success": False, "error": "No file provided"})

        results = []
        total_tokens = {"input": 0, "output": 0, "total": 0}

        for file_content, filename in files_to_process:
            validation = validate_file_format(file_content, filename)
            if not validation['valid']:
                results.append({"filename": filename, "success": False, "error": f"Invalid file: {validation['message']}"})
                continue

            result = process_single_file(file_content, filename, "bank")

            token_usage = result.get('token_usage', {})
            input_tok = token_usage.get('input_tokens', 0)
            output_tok = token_usage.get('output_tokens', 0)
            cost = calculate_cost(input_tok, output_tok, result.get('method', 'unknown'))

            log_processing(
                doc_type="bank", filename=filename, method=result.get('method', 'unknown'),
                input_tokens=input_tok, output_tokens=output_tok,
                total_tokens=token_usage.get('total_tokens', 0), cost_usd=cost,
                success=result.get('success', False), error_message=result.get('error'),
                student_name=student_name, scholarship_id=scholarship_id,
                extracted_data=result if result.get('success') else None,
                processing_time_ms=result.get('processing_time_ms'), image_url=result.get('image_url')
            )

            if result.get('success'):
                total_tokens["input"] += input_tok
                total_tokens["output"] += output_tok
                total_tokens["total"] += token_usage.get('total_tokens', 0)

            results.append({"filename": filename, **result})

        if len(results) == 1:
            return JSONResponse(content=results[0])
        else:
            return JSONResponse(content={"success": True, "total_files": len(results), "results": results, "total_token_usage": total_tokens})

    except Exception as e:
        log_processing(
            doc_type="bank", filename="unknown", method="error",
            input_tokens=0, output_tokens=0, total_tokens=0, cost_usd=0.0,
            success=False, error_message=str(e), student_name=student_name, scholarship_id=scholarship_id
        )
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/ocr/bill")
async def process_bill(
    request: Request,
    files: List[UploadFile] = File(default=[]),
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    scholarship_id: Optional[str] = Form(None),
    student_name: Optional[str] = Form(None)
):
    try:
        files_to_process = []

        if files:
            for uploaded_file in files:
                file_content = await uploaded_file.read()
                files_to_process.append((file_content, uploaded_file.filename))
        elif file:
            file_content = await file.read()
            files_to_process.append((file_content, file.filename))
        elif file_url:
            file_content, filename = download_file_from_url(file_url)
            files_to_process.append((file_content, filename))
        else:
            return JSONResponse(status_code=400, content={"success": False, "error": "No file provided"})

        results = []
        total_tokens = {"input": 0, "output": 0, "total": 0}

        for file_content, filename in files_to_process:
            validation = validate_file_format(file_content, filename)
            if not validation['valid']:
                results.append({"filename": filename, "success": False, "error": f"Invalid file: {validation['message']}"})
                continue

            # ✅ Use multi-bill extraction for images
            is_pdf = file_content[:4] == b'%PDF' or file_content[:5] == b'\x0a%PDF'

            if is_pdf:
                result = process_single_file(file_content, filename, "bill")
                results.append({"filename": filename, **result})
            else:
                # Try multi-bill extraction first
                try:
                    image_bills = analyze_bill_multi_bills_from_image(file_content, filename)
                    if len(image_bills) > 1:
                        # Multiple bills found in one image
                        for b in image_bills:
                            results.append({
                                "filename": filename,
                                "success": True,
                                "method": "gemini_vision",
                                "bill_index": b.get("bill_index"),
                                **b
                            })
                    else:
                        # Single bill - return as normal
                        result = image_bills[0] if image_bills else {}
                        result["success"] = True
                        result["method"] = "gemini_vision"
                        results.append({"filename": filename, **result})
                except Exception:
                    # Fallback to single extraction
                    result = process_single_file(file_content, filename, "bill")
                    results.append({"filename": filename, **result})

            token_usage = results[-1].get('token_usage', {})
            input_tok = token_usage.get('input_tokens', 0)
            output_tok = token_usage.get('output_tokens', 0)
            cost = calculate_cost(input_tok, output_tok, 'gemini_vision')

            log_processing(
                doc_type="bill", filename=filename, method='gemini_vision',
                input_tokens=input_tok, output_tokens=output_tok,
                total_tokens=token_usage.get('total_tokens', 0), cost_usd=cost,
                success=True, student_name=student_name, scholarship_id=scholarship_id
            )

        if len(results) == 1:
            return JSONResponse(content=results[0])
        else:
            return JSONResponse(content={"success": True, "total_files": len(results), "results": results, "total_token_usage": total_tokens})

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ============================================================
# ZOHO SYNC ENDPOINTS
# ============================================================

@app.post("/zoho/sync-records")
async def sync_records_to_zoho(request: ZohoSyncRequest, today_only: bool = False):
    try:
        if not supabase:
            return JSONResponse(status_code=400, content={"success": False, "error": "Supabase not configured"})

        record_ids_str = [str(rid) for rid in request.record_ids]
        records_to_sync = []
        supabase_ids = []
        failed_extraction_records = []
        
        print(f"\n{'='*60}")
        print(f"SYNC TO ZOHO - Starting")
        print(f"{'='*60}")
        print(f"Requested records: {len(record_ids_str)}")
        print(f"{'='*60}\n")

        for record_id in record_ids_str:
            try:
                query = supabase.table('auto_extraction_results').select('*').eq('id', record_id)
                
                response = query.execute()
                if response.data and len(response.data) > 0:
                    record = response.data[0]
                    
                    print(f"  Record {record_id}: status={record.get('status')}, push_status={record.get('push_status')}, processed_at={record.get('processed_at')}")
                    
                    # Check if extraction was successful
                    if record.get('status') != 'success':
                        print(f"    ❌ Skipped - extraction status is '{record.get('status')}'")
                        failed_extraction_records.append({
                            'id': record.get('id'),
                            'scholar_id': record.get('scholar_id'),
                            'error': record.get('error_message', 'Extraction failed')
                        })
                        continue
                    
                    # Only include successfully extracted records that haven't been synced
                    if record.get('push_status') is None:
                        print(f"    ✅ Added to sync queue")
                        records_to_sync.append(record)
                        supabase_ids.append(record.get('id'))
                    else:
                        print(f"    ⏭️  Skipped - already synced (push_status={record.get('push_status')})")
                else:
                    print(f"  Record {record_id}: ❌ Not found")
            except Exception as fetch_error:
                print(f"❌ Error fetching record {record_id}: {fetch_error}")

        if not records_to_sync:
            error_details = {
                "success": False,
                "error": "No valid records found",
                "details": {
                    "total_requested": len(record_ids_str),
                    "failed_extractions": len(failed_extraction_records),
                    "failed_extraction_ids": [r['id'] for r in failed_extraction_records]
                }
            }
            print(f"❌ No valid records to sync:")
            print(f"   Requested: {len(record_ids_str)} records")
            print(f"   Failed extractions: {len(failed_extraction_records)}")
            print(f"   Valid to sync: 0")
            return JSONResponse(status_code=400, content=error_details)

        # Get access token using Token3 (create scope)
        print(f"🔑 Attempting to get Zoho token (create scope)...")
        access_token, token_name = get_zoho_token(scope_needed="create")
        
        print(f"🔍 Token result: token_name={token_name}, access_token={'None' if not access_token else access_token[:20]+'...'}")
        
        if not access_token:
            error_msg = f"Failed to get Zoho access token from {token_name}"
            print(f"❌ {error_msg}")
            return JSONResponse(status_code=400, content={"success": False, "error": error_msg})
        
        print(f"✅ Using Zoho token: {token_name} ({access_token[:20]}...)")

        form_url = f"https://creator.zoho.com/api/v2/{request.config.owner_name}/{request.config.app_name}/form/{request.config.form_name}"
        headers = {'Authorization': f'Zoho-oauthtoken {access_token}', 'Content-Type': 'application/json'}
        
        print(f"📤 Zoho Form URL: {form_url}")
        print(f"{'='*60}\n")

        results = {"total_records": len(records_to_sync), "successful": 0, "inserted": 0, "updated": 0, "failed": 0, "errors": [], "sync_details": []}
        successfully_synced_ids = []
        
        # Track retry attempts
        retry_queue = []

        for idx, record in enumerate(records_to_sync, 1):
            record_id = record.get('id')
            zoho_record_id = record.get('record_id')
            scholar_id = record.get('scholar_id', 'unknown')

            print(f"[{idx}/{len(records_to_sync)}] Processing {scholar_id}...", end=' ')

            try:
                if request.field_mapping:
                    # ✅ Apply dynamic mapping if provided by ZohoConfigModal
                    zoho_record = {}
                    
                    def resolve_path(obj, path):
                        keys = path.replace('[', '.').replace(']', '').split('.')
                        val = obj
                        for key in keys:
                            if val is None: return None
                            if isinstance(val, list) and key.isdigit():
                                try: val = val[int(key)]
                                except IndexError: return None
                            elif isinstance(val, dict):
                                val = val.get(key)
                            else:
                                return None
                        return val
                        
                    for src_field, zoho_field in request.field_mapping.items():
                        if not zoho_field: continue
                        val = resolve_path(record, src_field)
                        if val is not None:
                            zoho_record[zoho_field] = val
                            
                    # Optional dynamic Scholar ID for lookup
                    scholar_id_field_to_check = None
                    for src_field, zoho_field in request.field_mapping.items():
                        if src_field == 'scholar_id':
                            scholar_id_field_to_check = zoho_field
                            break
                else:
                    # ⚠️ Legacy fallback
                    zoho_record = format_record_for_zoho_creator(record)
                    if not zoho_record:
                        raise Exception("format_record_for_zoho_creator returned None - check record data")
                    scholar_id_field_to_check = 'Scholar_ID'

                if request.tag:
                    zoho_record['tag'] = request.tag
                    zoho_record['tag_color'] = request.tag_color

                sync_action = request.sync_mode
                existing_id = None

                if request.sync_mode == 'auto':
                    # Determine report Link Name (use explicit report_name if mapped, otherwise All_form_name)
                    report_link_name = request.config.report_name if hasattr(request.config, 'report_name') and request.config.report_name else f"All_{request.config.form_name}"
                    
                    # We check for update using Scholar_ID
                    if scholar_id_field_to_check and zoho_record.get(scholar_id_field_to_check):
                        scholar_id_val = zoho_record.get(scholar_id_field_to_check)
                        check_url = f"https://creator.zoho.com/api/v2/{request.config.owner_name}/{request.config.app_name}/report/{report_link_name}"
                        check_response = requests.get(check_url, headers=headers, params={'criteria': f'{scholar_id_field_to_check} == "{scholar_id_val}"', 'max_records': 1}, timeout=10)
                        if check_response.status_code == 200:
                            data = check_response.json()
                            if data.get('data') and len(data['data']) > 0:
                                sync_action = 'update'
                                existing_id = data['data'][0].get('ID')
                            else:
                                sync_action = 'insert'
                        else:
                            sync_action = 'insert'

                if sync_action == 'insert':
                    response = requests.post(form_url, json={"data": zoho_record}, headers=headers, timeout=30)
                    response.raise_for_status()
                    results['successful'] += 1
                    results['inserted'] += 1
                    successfully_synced_ids.append(record_id)
                    print("✅ INSERTED")
                    results['sync_details'].append({"record_id": record_id, "scholar_id": scholar_id, "action": "INSERT", "status": "success"})

                elif sync_action == 'update' and existing_id:
                    report_link_name = request.config.report_name if hasattr(request.config, 'report_name') and request.config.report_name else f"All_{request.config.form_name}"
                    update_url = f"https://creator.zoho.com/api/v2/{request.config.owner_name}/{request.config.app_name}/report/{report_link_name}/{existing_id}"
                    response = requests.put(update_url, json={"data": zoho_record}, headers=headers, timeout=30)
                    response.raise_for_status()
                    results['successful'] += 1
                    results['updated'] += 1
                    successfully_synced_ids.append(record_id)
                    print("🔄 UPDATED")
                    results['sync_details'].append({"record_id": record_id, "scholar_id": scholar_id, "action": "UPDATE", "zoho_id": existing_id, "status": "success"})
                else:
                    results['failed'] += 1
                    print("⚠️ NOT FOUND")
                    results['sync_details'].append({"record_id": record_id, "scholar_id": scholar_id, "status": "not_found"})

            except Exception as record_error:
                error_msg = str(record_error)
                
                # Check if it's a 401 error (token expired)
                if '401' in error_msg:
                    print(f"❌ Token expired, refreshing...")
                    # Get a new token
                    access_token, token_name = get_zoho_token(scope_needed="create")
                    if access_token:
                        headers['Authorization'] = f'Zoho-oauthtoken {access_token}'
                        print(f"✅ Token refreshed ({token_name}), adding to retry queue...")
                        # Add to retry queue instead of continuing
                        retry_queue.append(record)
                        continue
                
                results['failed'] += 1
                print(f"❌ {error_msg}")
                results['errors'].append({"record_id": record_id, "scholar_id": scholar_id, "error": error_msg})
                results['sync_details'].append({"record_id": record_id, "scholar_id": scholar_id, "status": "failed", "error": error_msg})

            time.sleep(0.5)
        
        # Retry failed records with refreshed token
        if retry_queue:
            print(f"\n🔄 Retrying {len(retry_queue)} records with refreshed token...")
            for idx, record in enumerate(retry_queue, 1):
                record_id = record.get('id')
                scholar_id = record.get('scholar_id', 'unknown')
                
                print(f"[RETRY {idx}/{len(retry_queue)}] Processing {scholar_id}...", end=' ')
                
                try:
                    zoho_record = format_record_for_zoho_creator(record)
                    
                    if not zoho_record:
                        raise Exception("format_record_for_zoho_creator returned None")
                    
                    if request.tag:
                        zoho_record['tag'] = request.tag
                        zoho_record['tag_color'] = request.tag_color
                    
                    # Always insert on retry (simplified)
                    response = requests.post(form_url, json={"data": zoho_record}, headers=headers, timeout=30)
                    response.raise_for_status()
                    results['successful'] += 1
                    results['inserted'] += 1
                    successfully_synced_ids.append(record_id)
                    print("✅ INSERTED")
                    results['sync_details'].append({"record_id": record_id, "scholar_id": scholar_id, "action": "INSERT", "status": "success"})
                    
                except Exception as retry_error:
                    results['failed'] += 1
                    error_msg = str(retry_error)
                    print(f"❌ {error_msg}")
                    results['errors'].append({"record_id": record_id, "scholar_id": scholar_id, "error": f"Retry failed: {error_msg}"})
                    results['sync_details'].append({"record_id": record_id, "scholar_id": scholar_id, "status": "failed", "error": error_msg})
                
                time.sleep(0.5)

        for supabase_id in successfully_synced_ids:
            try:
                update_data = {"push_status": "synced", "synced_at": datetime.now().isoformat(), "sync_mode": request.sync_mode}
                if request.tag:
                    update_data["tag"] = request.tag
                    update_data["tag_color"] = request.tag_color
                supabase.table('auto_extraction_results').update(update_data).eq('id', supabase_id).execute()
            except Exception as update_error:
                print(f"⚠️ Failed to update sync status for {supabase_id}: {update_error}")

        # Add failed extraction records to response
        response_data = {
            "success": results['successful'] > 0,
            "message": f"✅ Synced {results['successful']}/{len(records_to_sync)} records to Creator",
            "details": results
        }
        
        if failed_extraction_records:
            response_data["failed_extractions"] = failed_extraction_records
            response_data["message"] += f" | ⚠️ {len(failed_extraction_records)} records need re-extraction"
            response_data["warning"] = "Some records failed extraction. Please re-run extraction for these records before pushing."
        
        return JSONResponse(content=response_data)

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/ocr/cross-report/async")
async def cross_report_ocr_async(
    source_record_id: str = Form(...),
    source_report_name: str = Form(...),
    app_link_name: str = Form(...),
    scholar_id: str = Form(...),
    scholar_name: str = Form(...),
    tracking_id: Optional[str] = Form(None),
    dest_form_name: str = Form(...),
    bank_passbook_url: Optional[str] = Form(None),
    bill_urls: Optional[str] = Form(None)
):
    try:
        bill_url_list = []
        if bill_urls:
            try:
                bill_url_list = json.loads(bill_urls)
            except:
                pass

        if not bank_passbook_url and not bill_url_list:
            return JSONResponse(status_code=400, content={"success": False, "error": "No files provided"})

        config = {
            "source_record_id": source_record_id, "source_report_name": source_report_name,
            "app_link_name": app_link_name, "scholar_id": scholar_id, "scholar_name": scholar_name,
            "tracking_id": tracking_id, "dest_form_name": dest_form_name,
            "bank_passbook_url": bank_passbook_url, "bill_url_list": bill_url_list
        }

        thread = threading.Thread(target=process_cross_report_background, args=(config,))
        thread.daemon = True
        thread.start()

        return JSONResponse(content={"success": True, "status": "processing", "message": f"OCR started for {scholar_name}", "record_id": source_record_id})

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


def process_cross_report_background(config: Dict):
    scholar_name = config['scholar_name']
    scholar_id = config['scholar_id']

    bank_data = None
    bill_data_list = []
    total_amount = 0.0

    if config.get('bank_passbook_url'):
        try:
            file_content, filename = download_file_from_url(config['bank_passbook_url'])
            result = process_single_file(file_content, filename, "bank")
            if result.get('success'):
                bank_data = result
        except Exception as e:
            print(f"[CROSS-REPORT BG] ✗ Bank error: {e}")

    for idx, bill_url in enumerate(config.get('bill_url_list', []), 1):
        try:
            file_content, filename = download_file_from_url(bill_url)
            result = process_single_file(file_content, filename, "bill")
            if result.get('success'):
                bill_data_list.append(result)
                try:
                    total_amount += float(result.get('amount') or 0)
                except:
                    pass
        except Exception as e:
            print(f"[CROSS-REPORT BG] ✗ Bill {idx} error: {e}")

    try:
        access_token, _ = get_zoho_token(scope_needed="create")
        if not access_token:
            return

        owner, app = config['app_link_name'].split('/', 1)
        form_url = f"https://creator.zoho.com/api/v2/{owner}/{app}/form/{config['dest_form_name']}"

        bill_text = ""
        for i, bill in enumerate(bill_data_list, 1):
            bill_text += f"Bill {i}: Student: {bill.get('student_name')} | College: {bill.get('college_name')} | Amount: ₹{bill.get('amount', 0)}\n\n"

        data = {
            "Scholar_ID": scholar_id, "Scholar_Name": scholar_name,
            "Tracking_ID": config.get('tracking_id'),
            "Bank_Name": bank_data.get('bank_name', '') if bank_data else '',
            "Holder_Name": bank_data.get('account_holder_name', '') if bank_data else '',
            "Account_No": bank_data.get('account_number', '') if bank_data else '',
            "IFSC_Code": bank_data.get('ifsc_code', '') if bank_data else '',
            "Branch_Name": bank_data.get('branch_name', '') if bank_data else '',
            "Bill_Data": bill_text,
            "Bill1_Amount": bill_data_list[0].get('amount', 0) if len(bill_data_list) > 0 else 0,
            "Bill2_Amount": bill_data_list[1].get('amount', 0) if len(bill_data_list) > 1 else 0,
            "Bill3_Amount": bill_data_list[2].get('amount', 0) if len(bill_data_list) > 2 else 0,
            "Bill4_Amount": bill_data_list[3].get('amount', 0) if len(bill_data_list) > 3 else 0,
            "Bill5_Amount": bill_data_list[4].get('amount', 0) if len(bill_data_list) > 4 else 0,
            "Amount": total_amount, "status": "OCR Completed"
        }

        response = requests.post(form_url, json={"data": data}, headers={"Authorization": f"Zoho-oauthtoken {access_token}"}, timeout=30)
        print(f"[CROSS-REPORT BG] ✅ Record saved: {response.status_code}")

    except Exception as e:
        print(f"[CROSS-REPORT BG] ✗ Save error: {e}")


# ============================================================
# AUTO-EXTRACT ENDPOINTS
# ============================================================

@app.post("/ocr/auto-extract/fetch-fields")
async def fetch_report_fields(app_link_name: str = Form(...), report_link_name: str = Form(...)):
    try:
        records = fetch_zoho_records(app_link_name=app_link_name, report_link_name=report_link_name, max_records=1)

        if not records:
            return JSONResponse(content={"success": False, "error": "No records found in report"})

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
            "success": True, "total_fields": len(all_fields),
            "file_fields": sorted(file_fields), "text_fields": sorted(text_fields),
            "all_fields": sorted(all_fields)
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


def get_already_extracted_record_ids(app_link_name: str, report_link_name: str, include_failed: bool = False, include_synced: bool = False) -> set:
    if not supabase:
        return set()

    try:
        response = supabase.table("auto_extraction_results").select("id, record_id, status, push_status, scholar_id, tracking_id, created_at").eq("app_link_name", app_link_name).eq("report_link_name", report_link_name).execute()

        if not response.data:
            return set()

        all_results = response.data
        status_counts = {}
        push_status_counts = {}

        for r in all_results:
            status = r.get("status", "unknown")
            status_counts[status] = status_counts.get(status, 0) + 1
            push_status = r.get("push_status", "not_synced")
            push_status_counts[push_status] = push_status_counts.get(push_status, 0) + 1

        already_extracted = set()

        successful_results = [r for r in all_results if r.get("status") == "success"]
        successful_ids = {str(r["record_id"]) for r in successful_results if r.get("record_id")}
        already_extracted.update(successful_ids)

        if not include_synced:
            synced_ids = {
                str(r["record_id"])
                for r in all_results
                if r.get("push_status") == "synced" and r.get("record_id")
            }
            already_extracted.update(synced_ids)

        if include_failed:
            failed_ids = {
                str(r["record_id"])
                for r in all_results
                if r.get("status") in ("failed", "error") and r.get("record_id")
            }
            already_extracted.update(failed_ids)

        return already_extracted

    except Exception as e:
        print(f"[FILTER] ✗ Error: {e}")
        return set()


def fetch_zoho_records_with_dedup(app_link_name: str, report_link_name: str, criteria: Optional[str] = None,
                                   max_records: int = None, exclude_already_extracted: bool = True,
                                   include_failed_retries: bool = False, include_synced_for_retry: bool = False) -> tuple:
    try:
        access_token, token_name = get_zoho_token(scope_needed="read")
        if not access_token:
            raise Exception("Failed to get READ token")

        api_url = f"https://creator.zoho.com/api/v2.1/{app_link_name}/report/{report_link_name}"
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}

        all_records = []
        seen_ids = set()
        page = 1
        max_id = None

        while True:
            if criteria:
                params = {"criteria": criteria, "from": 1, "limit": 200}
            else:
                params = {"from": 1, "limit": 200} if max_id is None else {"criteria": f"ID < {max_id}", "from": 1, "limit": 200}

            response = requests.get(api_url, headers=headers, params=params, timeout=30)


            if response.status_code == 400:


                try:


                    err_data = response.json()


                    if err_data.get("code") == 9220: # No records exist in this report


                        break


                except:


                    pass



            response.raise_for_status()

            data = response.json()
            records = data.get("data", [])

            if not records:
                break

            batch_ids = [int(r.get("ID", 0)) for r in records if r.get("ID")]
            if batch_ids:
                max_id = min(batch_ids)

            for record in records:
                record_id = str(record.get("ID", ""))
                if record_id and record_id not in seen_ids:
                    seen_ids.add(record_id)
                    all_records.append(record)

            if len(records) < 200 or (max_records and len(all_records) >= max_records):
                break

            page += 1
            time.sleep(0.5)

        stats = {
            "total_fetched": len(all_records), "excluded_successful": 0,
            "excluded_synced": 0, "excluded_failed": 0, "total_excluded": 0,
            "new_count": len(all_records), "final_records": len(all_records), "excluded_ids": []
        }

        if exclude_already_extracted:
            already_extracted = get_already_extracted_record_ids(
                app_link_name, report_link_name,
                include_failed=include_failed_retries, include_synced=include_synced_for_retry
            )

            if already_extracted:
                original_count = len(all_records)
                all_records = [r for r in all_records if str(r.get("ID", "")) not in already_extracted]
                excluded_count = original_count - len(all_records)
                stats["total_excluded"] = excluded_count
                stats["new_count"] = len(all_records)
                stats["final_records"] = len(all_records)
                stats["excluded_ids"] = list(already_extracted)[:10]

        if max_records and len(all_records) > max_records:
            all_records = all_records[:max_records]

        return all_records, stats

    except Exception as e:
        print(f"[ZOHO FETCH] ✗ Error: {e}")
        raise


@app.post("/ocr/auto-extract/preview")
async def preview_extraction(
    app_link_name: str = Form(...),
    report_link_name: str = Form(...),
    bank_field_name: Optional[str] = Form(None),
    bill_field_name: Optional[str] = Form(None),
    filter_criteria: Optional[str] = Form(None),
    store_images: str = Form("false"),
    fetch_all: str = Form("false"),
    exclude_already_extracted: str = Form("true"),
    include_failed_retries: str = Form("false"),
    include_synced_for_retry: str = Form("false"),
    max_records_limit: int = Form(1000),
    selected_fields: Optional[str] = Form(None)
):
    try:
        exclude_extracted = exclude_already_extracted.lower() in ('true', '1', 'yes')
        include_failed = include_failed_retries.lower() in ('true', '1', 'yes')
        include_synced = include_synced_for_retry.lower() in ('true', '1', 'yes')

        zoho_criteria = None
        if filter_criteria:
            try:
                filters = json.loads(filter_criteria)
                zoho_criteria = convert_filters_to_zoho_criteria(filters)
            except (json.JSONDecodeError, TypeError):
                # Raw string criteria — pass directly to Zoho as-is
                if isinstance(filter_criteria, str) and filter_criteria.strip():
                    zoho_criteria = filter_criteria

        records, fetch_stats = fetch_zoho_records_with_dedup(
            app_link_name=app_link_name, report_link_name=report_link_name,
            criteria=zoho_criteria, max_records=max_records_limit,
            exclude_already_extracted=exclude_extracted,
            include_failed_retries=include_failed, include_synced_for_retry=include_synced
        )

        fields_to_fetch = []
        if selected_fields:
            try:
                fields_to_fetch = json.loads(selected_fields)
            except:
                pass

        def extract_image_url(field_value):
            if not field_value:
                return None
            if isinstance(field_value, str) and field_value.startswith(('http', '/api/v2.1/')):
                return field_value
            elif isinstance(field_value, list) and len(field_value) > 0:
                first = field_value[0]
                if isinstance(first, str) and first.startswith(('http', '/api/v2.1/')):
                    return first
                elif isinstance(first, dict) and first.get("download_url"):
                    return first["download_url"]
            elif isinstance(field_value, dict) and field_value.get("download_url"):
                return field_value["download_url"]
            return None

        def extract_name(record):
            for field in ["Name", "Student_Name", "Scholar_Name"]:
                name_value = record.get(field)
                if isinstance(name_value, str):
                    return name_value
                if isinstance(name_value, dict) and name_value.get("zc_display_value"):
                    return name_value["zc_display_value"]
            return "Unknown"

        preview_records = []
        for record in records:
            record_id = str(record.get("ID", ""))
            student_name = extract_name(record)

            bank_value = record.get(bank_field_name) if bank_field_name else None
            bill_value = record.get(bill_field_name) if bill_field_name else None

            preview_record = {
                "record_id": record_id, "student_name": student_name,
                "has_bank_image": extract_image_url(bank_value) is not None,
                "has_bill_image": extract_image_url(bill_value) is not None
            }

            if fields_to_fetch:
                for field in fields_to_fetch:
                    value = record.get(field)
                    if isinstance(value, dict):
                        preview_record[field] = value.get("zc_display_value", str(value))
                    else:
                        preview_record[field] = value

            preview_records.append(preview_record)

        return JSONResponse(content={
            "success": True, "total_records": len(preview_records),
            "total_fetched_from_zoho": fetch_stats["total_fetched"],
            "total_excluded": fetch_stats["total_excluded"],
            "new_records_count": fetch_stats["new_count"],
            "sample_excluded_ids": fetch_stats["excluded_ids"],
            "selected_fields": fields_to_fetch,
            "sample_records": preview_records,
            "estimated_cost": f"${len(preview_records) * 0.003:.4f}",
            "estimated_time_minutes": math.ceil(len(preview_records) * 3 / 60),
            "deduplication_applied": exclude_extracted,
            "include_failed_for_retry": include_failed,
            "include_synced_for_retry": include_synced
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/ocr/webhook/trigger")
async def creator_webhook_trigger(payload: CreatorWebhookPayload):
    if not supabase:
        return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured."})

    try:
        selected_ids = []
        if payload.record_ids:
            if isinstance(payload.record_ids, list):
                selected_ids = [str(rid) for rid in payload.record_ids]
            elif isinstance(payload.record_ids, str):
                try:
                    raw_ids = json.loads(payload.record_ids)
                    if isinstance(raw_ids, list):
                        selected_ids = [str(rid) for rid in raw_ids]
                    else:
                        selected_ids = [str(payload.record_ids)]
                except:
                    selected_ids = [str(payload.record_ids)]

        already_extracted = get_already_extracted_record_ids(payload.app_link_name, payload.report_link_name)

        if selected_ids:
            selected_ids = [rid for rid in selected_ids if rid not in already_extracted]
            if len(selected_ids) == 0:
                return JSONResponse(status_code=400, content={"success": False, "error": "All records previously extracted"})

            active_job = check_active_jobs_for_records(selected_ids)
            if active_job:
                return JSONResponse(status_code=409, content={"success": False, "error": f"Records already active in job: {active_job}"})

        job_id = f"job_webhook_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        config = {
            "app_link_name": payload.app_link_name,
            "report_link_name": payload.report_link_name,
            "bank_field_name": payload.bank_field_name,
            "bill_field_name": payload.bill_field_name,
            "filter_criteria": None,
            "selected_record_ids": selected_ids,
            "selected_fields": None,
            "auto_push": payload.auto_push,
            "target_app_name": payload.target_app_name,
            "target_form_name": payload.target_form_name,
            "target_owner_name": payload.target_owner_name
        }

        supabase.table("auto_extraction_jobs").insert({
            "job_id": job_id,
            "app_link_name": payload.app_link_name,
            "report_link_name": payload.report_link_name,
            "bank_field_name": payload.bank_field_name,
            "bill_field_name": payload.bill_field_name,
            "filter_criteria": None,
            "selected_fields": None,
            "status": "pending"
        }).execute()

        thread = threading.Thread(target=process_extraction_job, args=(job_id, config))
        thread.daemon = True
        thread.start()

        print(f"[WEBHOOK] Triggered background job {job_id} for {len(selected_ids)} records")

        return JSONResponse(content={
            "success": True, 
            "job_id": job_id, 
            "status": "started",
            "message": f"Webhook triggered extraction for {len(selected_ids)} records",
            "check_status_url": f"/ocr/auto-extract/status/{job_id}"
        })

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/ocr/auto-extract/start")
async def start_extraction(
    app_link_name: str = Form(...),
    report_link_name: str = Form(...),
    bank_field_name: Optional[str] = Form(None),
    bill_field_name: Optional[str] = Form(None),
    filter_criteria: Optional[str] = Form(None),
    selected_record_ids: Optional[str] = Form(None),
    selected_fields: Optional[str] = Form(None)
):
    if not supabase:
        return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured."})

    try:
        zoho_criteria = None
        if filter_criteria:
            try:
                filters = json.loads(filter_criteria)
                zoho_criteria = convert_filters_to_zoho_criteria(filters)
            except:
                if isinstance(filter_criteria, str) and filter_criteria.strip():
                    zoho_criteria = filter_criteria

        selected_ids = []
        if selected_record_ids:
            try:
                raw_ids = json.loads(selected_record_ids)
                selected_ids = list(set([str(rid) for rid in raw_ids]))
            except:
                pass

        already_extracted = get_already_extracted_record_ids(app_link_name, report_link_name)

        if selected_ids:
            selected_ids = [rid for rid in selected_ids if rid not in already_extracted]
            if len(selected_ids) == 0:
                return JSONResponse(status_code=400, content={"success": False, "error": "All selected records have already been extracted"})
        # NOTE: if selected_ids is empty (no manual selection), 
        # dedup happens inside process_extraction_job via fetch_zoho_records_with_dedup
        # which respects the filter_criteria from Zoho directly

        if selected_ids:
            active_job = check_active_jobs_for_records(selected_ids)
            if active_job:
                return JSONResponse(status_code=409, content={"success": False, "error": f"Records are already being processed in job {active_job}", "active_job_id": active_job})

        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        config = {
            "app_link_name": app_link_name, "report_link_name": report_link_name,
            "bank_field_name": bank_field_name, "bill_field_name": bill_field_name,
            "filter_criteria": zoho_criteria, "selected_record_ids": selected_ids,
            "selected_fields": selected_fields
        }

        supabase.table("auto_extraction_jobs").insert({
            "job_id": job_id, "app_link_name": app_link_name,
            "report_link_name": report_link_name, "bank_field_name": bank_field_name,
            "bill_field_name": bill_field_name, "filter_criteria": zoho_criteria,
            "selected_fields": selected_fields, "status": "pending"
        }).execute()

        thread = threading.Thread(target=process_extraction_job, args=(job_id, config))
        thread.daemon = True
        thread.start()

        return JSONResponse(content={
            "success": True, "job_id": job_id, "status": "started",
            "message": f"🚀 Processing {len(selected_ids)} records",
            "check_status_url": f"/ocr/auto-extract/status/{job_id}"
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/ocr/auto-extract/status/{job_id}")
async def get_job_status(job_id: str):
    if not supabase:
        return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured"})
    try:
        response = supabase.table("auto_extraction_jobs").select("*").eq("job_id", job_id).execute()
        if not response.data:
            return JSONResponse(status_code=404, content={"success": False, "error": "Job not found"})
        job = response.data[0]
        progress_percent = 0
        if job.get("total_records", 0) > 0:
            progress_percent = round((job.get("processed_records", 0) / job["total_records"]) * 100, 2)
        return JSONResponse(content={
            "success": True, "job_id": job_id, "status": job["status"],
            "progress": {
                "total_records": job.get("total_records", 0),
                "processed_records": job.get("processed_records", 0),
                "successful_records": job.get("successful_records", 0),
                "failed_records": job.get("failed_records", 0),
                "progress_percent": progress_percent
            },
            "cost": {"total_cost_usd": float(job.get("total_cost_usd", 0))},
            "timestamps": {"created_at": job.get("created_at"), "started_at": job.get("started_at"), "completed_at": job.get("completed_at")}
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/ocr/auto-extract/failed-records")
async def get_failed_extraction_records(app_link_name: str, report_link_name: str, today_only: bool = True):
    """Get records that failed extraction and need to be re-run"""
    if not supabase:
        return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured"})
    try:
        query = supabase.table("auto_extraction_results").select("*").eq("app_link_name", app_link_name).eq("report_link_name", report_link_name).eq("status", "failed")
        
        # Filter by today's date if enabled
        if today_only:
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            today_end = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999).isoformat()
            query = query.gte('processed_at', today_start).lte('processed_at', today_end)
        
        response = query.execute()
        failed_records = response.data
        
        return JSONResponse(content={
            "success": True,
            "total_failed": len(failed_records),
            "failed_records": failed_records,
            "message": f"Found {len(failed_records)} failed extraction records"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/ocr/auto-extract/retry-failed")
async def retry_failed_extractions(
    app_link_name: str = Form(...),
    report_link_name: str = Form(...),
    bank_field_name: Optional[str] = Form(None),
    bill_field_name: Optional[str] = Form(None),
    today_only: bool = Form(True)
):
    """Retry extraction for failed records"""
    if not supabase:
        return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured"})
    
    try:
        # Get failed records
        query = supabase.table("auto_extraction_results").select("*").eq("app_link_name", app_link_name).eq("report_link_name", report_link_name).eq("status", "failed")
        
        if today_only:
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            today_end = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999).isoformat()
            query = query.gte('processed_at', today_start).lte('processed_at', today_end)
        
        response = query.execute()
        failed_records = response.data
        
        if not failed_records:
            return JSONResponse(content={"success": False, "message": "No failed records found to retry"})
        
        # Extract record IDs from failed records
        failed_record_ids = [str(record.get('record_id')) for record in failed_records if record.get('record_id')]
        
        if not failed_record_ids:
            return JSONResponse(content={"success": False, "message": "No valid record IDs found in failed records"})
        
        # Create a new extraction job for these failed records
        job_id = f"retry_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        config = {
            "app_link_name": app_link_name,
            "report_link_name": report_link_name,
            "bank_field_name": bank_field_name,
            "bill_field_name": bill_field_name,
            "filter_criteria": None,
            "selected_record_ids": failed_record_ids,
            "selected_fields": None
        }
        
        supabase.table("auto_extraction_jobs").insert({
            "job_id": job_id,
            "app_link_name": app_link_name,
            "report_link_name": report_link_name,
            "bank_field_name": bank_field_name,
            "bill_field_name": bill_field_name,
            "status": "pending"
        }).execute()
        
        # Start processing in background
        thread = threading.Thread(target=process_extraction_job, args=(job_id, config))
        thread.daemon = True
        thread.start()
        
        return JSONResponse(content={
            "success": True,
            "job_id": job_id,
            "status": "started",
            "message": f"🔄 Retrying extraction for {len(failed_record_ids)} failed records",
            "failed_count": len(failed_record_ids),
            "check_status_url": f"/ocr/auto-extract/status/{job_id}"
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/ocr/auto-extract/results/{job_id}")
async def get_job_results(job_id: str, limit: int = 100, offset: int = 0, status_filter: Optional[str] = None):
    if not supabase:
        return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured"})
    try:
        query = supabase.table("auto_extraction_results").select("*").eq("job_id", job_id).order("created_at", desc=True).range(offset, offset + limit - 1)
        if status_filter:
            query = query.eq("status", status_filter)
        response = query.execute()
        return JSONResponse(content={"success": True, "job_id": job_id, "total_results": len(response.data), "results": response.data})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/ocr/auto-extract/jobs")
async def list_jobs(limit: int = 20):
    if not supabase:
        return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured"})
    try:
        response = supabase.table("auto_extraction_jobs").select("*").order("created_at", desc=True).limit(limit).execute()
        return JSONResponse(content={"success": True, "total_jobs": len(response.data), "jobs": response.data})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ============================================================
# STATS & MONITORING ENDPOINTS
# ============================================================

@app.get("/stats")
async def get_stats(days: int = 30):
    try:
        stats = get_usage_stats(days)
        return JSONResponse(content=stats)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/logs")
async def get_logs(limit: int = 100):
    try:
        logs = get_all_logs(limit)
        return JSONResponse(content={"logs": logs})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/logs/{log_id}")
async def remove_log(log_id: int):
    try:
        success = delete_log(log_id)
        if success:
            return {"success": True, "message": "Log deleted"}
        else:
            return JSONResponse(status_code=404, content={"success": False, "message": "Log not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/token-stats")
async def get_token_stats():
    read_tokens = []
    create_tokens = []
    for token in ZOHO_TOKENS:
        token_stat = {
            "name": token["name"], "scope": token["scope"], "status": token["status"],
            "requests": token["request_count"], "errors": token["error_count"],
            "success_rate": round((token["request_count"] - token["error_count"]) / token["request_count"] * 100 if token["request_count"] > 0 else 100, 2)
        }
        if token["scope"] == "read":
            read_tokens.append(token_stat)
        else:
            create_tokens.append(token_stat)
    return {"total_tokens": len(ZOHO_TOKENS), "active_tokens": sum(1 for t in ZOHO_TOKENS if t["status"] == "active"), "read_tokens": read_tokens, "create_tokens": create_tokens}


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "OCR API - Complete with Auto-Extract + Auth + Multi-Bill Image",
        "version": "8.0",
        "gemini_vision": "enabled" if USE_GEMINI else "disabled",
        "supabase": "connected" if supabase else "not configured",
        "authentication": "enabled" if AUTHENTIK_URL and AUTHENTIK_API_TOKEN else "not configured",
        "tokens_configured": len(ZOHO_TOKENS),
        "active_sessions": len(sessions)
    }


# ============================================================
# ZOHO SYNC ENDPOINTS
# ============================================================

@app.post("/sync/bulk-push-to-zoho")
async def bulk_push_to_zoho(limit: int = 1000, today_only: bool = True, retry_failed: bool = True):
    try:
        if not supabase:
            return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured"})
        
        # Build query
        query = supabase.table('auto_extraction_results').select('*')
        
        # Filter by today's date if enabled
        if today_only:
            today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
            today_end = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999).isoformat()
            query = query.gte('processed_at', today_start).lte('processed_at', today_end)
        
        # Only get successfully extracted records that haven't been synced
        # Failed extraction records should be re-extracted first, not pushed
        query = query.eq('status', 'success').is_('push_status', 'null')
        
        response = query.limit(limit).execute()
        records = response.data
        
        if not records:
            return JSONResponse(content={"success": False, "message": "No successfully extracted records found to push. Re-run extraction for failed records first."})
        
        result = zoho_bulk.bulk_insert(records)
        return JSONResponse(content={"success": True, "message": f"Synced {result['successful']}/{result['total_records']}", "details": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/sync/test-zoho-connection")
async def test_zoho_connection():
    try:
        result = zoho_bulk.test_connection()
        return JSONResponse(content={"success": result['success'], "result": result})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.get("/sync/test-token")
async def test_zoho_token():
    """Test if Zoho Token3 (create scope) is valid"""
    try:
        # Get Token3 (create scope)
        access_token, token_name = get_zoho_token(scope_needed="create")
        
        if not access_token:
            return JSONResponse(content={
                "success": False,
                "error": "Failed to get access token",
                "token_name": token_name
            })
        
        # Test the token by making a simple API call
        test_url = "https://creator.zoho.com/api/v2.1/data/applications"
        headers = {'Authorization': f'Zoho-oauthtoken {access_token}'}
        
        response = requests.get(test_url, headers=headers, timeout=10)
        
        return JSONResponse(content={
            "success": response.status_code == 200,
            "token_name": token_name,
            "token_preview": f"{access_token[:20]}...",
            "status_code": response.status_code,
            "response": response.json() if response.status_code == 200 else response.text,
            "message": "Token is valid ✅" if response.status_code == 200 else f"Token is invalid ❌ (HTTP {response.status_code})"
        })
        
    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })


@app.get("/sync/test-all-tokens")
async def test_all_zoho_tokens():
    """Test all configured Zoho tokens to see which ones are valid"""
    try:
        results = []
        test_url = "https://creator.zoho.com/api/v2.1/data/applications"
        
        for token_config in ZOHO_TOKENS:
            token_name = token_config['name']
            client_id = token_config['client_id']
            client_secret = token_config['client_secret']
            refresh_token = token_config['refresh_token']
            scope = token_config['scope']
            
            # Skip if credentials are missing
            if not all([client_id, client_secret, refresh_token]):
                results.append({
                    "token_name": token_name,
                    "scope": scope,
                    "status": "❌ Missing credentials",
                    "valid": False,
                    "error": "client_id, client_secret, or refresh_token not configured"
                })
                continue
            
            try:
                # Get access token using refresh token
                token_url = "https://accounts.zoho.com/oauth/v2/token"
                params = {
                    "refresh_token": refresh_token,
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "grant_type": "refresh_token"
                }
                
                token_response = requests.post(token_url, params=params, timeout=10)
                
                if token_response.status_code != 200:
                    results.append({
                        "token_name": token_name,
                        "scope": scope,
                        "status": f"❌ Refresh failed (HTTP {token_response.status_code})",
                        "valid": False,
                        "error": token_response.text
                    })
                    continue
                
                access_token = token_response.json().get("access_token")
                
                if not access_token:
                    results.append({
                        "token_name": token_name,
                        "scope": scope,
                        "status": "❌ No access token in response",
                        "valid": False,
                        "error": "access_token not found in response"
                    })
                    continue
                
                # Test the access token
                headers = {'Authorization': f'Zoho-oauthtoken {access_token}'}
                test_response = requests.get(test_url, headers=headers, timeout=10)
                
                if test_response.status_code == 200:
                    results.append({
                        "token_name": token_name,
                        "scope": scope,
                        "status": "✅ Valid",
                        "valid": True,
                        "token_preview": f"{access_token[:20]}...",
                        "full_token": access_token
                    })
                else:
                    results.append({
                        "token_name": token_name,
                        "scope": scope,
                        "status": f"❌ Invalid (HTTP {test_response.status_code})",
                        "valid": False,
                        "error": test_response.text[:200]
                    })
                    
            except Exception as token_error:
                results.append({
                    "token_name": token_name,
                    "scope": scope,
                    "status": "❌ Error",
                    "valid": False,
                    "error": str(token_error)
                })
        
        # Summary
        valid_count = sum(1 for r in results if r.get('valid'))
        invalid_count = len(results) - valid_count
        
        return JSONResponse(content={
            "success": True,
            "summary": {
                "total_tokens": len(results),
                "valid": valid_count,
                "invalid": invalid_count
            },
            "tokens": results
        })
        
    except Exception as e:
        import traceback
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        })



@app.get("/debug/today-count")
async def debug_today_count():
    """Debug endpoint to check today's record count"""
    try:
        if not supabase:
            return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured"})
        
        # Get today's date range
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        today_end = datetime.now().replace(hour=23, minute=59, second=59, microsecond=999999).isoformat()
        
        # Query 1: All records
        all_response = supabase.table('auto_extraction_results').select('*', count='exact').execute()
        total_count = all_response.count
        
        # Query 2: Today's records by processed_at
        today_processed = supabase.table('auto_extraction_results').select('*', count='exact').gte('processed_at', today_start).lte('processed_at', today_end).execute()
        
        # Query 3: Today's records by created_at
        today_created = supabase.table('auto_extraction_results').select('*', count='exact').gte('created_at', today_start).lte('created_at', today_end).execute()
        
        # Query 4: Today's records (either field)
        today_either = supabase.table('auto_extraction_results').select('*').or_(f'processed_at.gte.{today_start},created_at.gte.{today_start}').execute()
        today_either_filtered = [r for r in today_either.data if 
            (r.get('processed_at') and r['processed_at'] >= today_start and r['processed_at'] <= today_end) or
            (r.get('created_at') and r['created_at'] >= today_start and r['created_at'] <= today_end)
        ]
        
        # Query 5: Check push status for today's records
        need_to_push = [r for r in today_either_filtered if not r.get('push_status') or r.get('push_status') != 'synced']
        pushed = [r for r in today_either_filtered if r.get('push_status') == 'synced']
        
        # Sample records
        sample_records = today_either_filtered[:5] if today_either_filtered else []
        sample_data = [{
            'id': r.get('id'),
            'record_id': r.get('record_id'),
            'processed_at': r.get('processed_at'),
            'created_at': r.get('created_at'),
            'push_status': r.get('push_status'),
            'status': r.get('status')
        } for r in sample_records]
        
        return JSONResponse(content={
            "success": True,
            "today_range": {
                "start": today_start,
                "end": today_end
            },
            "counts": {
                "total_all_records": total_count,
                "today_by_processed_at": today_processed.count,
                "today_by_created_at": today_created.count,
                "today_either_field": len(today_either_filtered),
                "today_need_to_push": len(need_to_push),
                "today_pushed": len(pushed)
            },
            "sample_records": sample_data
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/sync/bulk-push-to-zoho-selected")
async def bulk_push_selected_to_zoho(request: dict):
    try:
        records = request.get('records', [])
        if not records:
            return JSONResponse(content={"success": False, "error": "No records provided"})
        if not isinstance(records, list):
            return JSONResponse(status_code=400, content={"success": False, "error": "Records must be an array"})
        result = zoho_bulk.bulk_insert(records)
        return JSONResponse(content={"success": True, "message": f"Pushed {result['successful']}/{result['total_records']} records", "details": result})
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


class ZohoConfigRequest(BaseModel):
    owner_name: str
    app_name: str
    form_name: str


class DynamicPushRequest(BaseModel):
    config: ZohoConfig
    field_mapping: Dict[str, str] = {}
    record_ids: List[Any]


@app.post("/zoho/get-form-fields")
async def get_zoho_form_fields(request: ZohoConfigRequest):
    try:
        access_token = os.getenv("ZOHO_ACCESS_TOKEN")
        if not access_token:
            return JSONResponse(status_code=400, content={"success": False, "error": "Zoho access token not configured"})

        report_url = f"https://creator.zoho.com/api/v2/{request.owner_name}/{request.app_name}/report/All_{request.form_name}"
        headers = {'Authorization': f'Zoho-oauthtoken {access_token}'}
        response = requests.get(report_url, headers=headers, params={'max_records': 1}, timeout=30)

        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                fields = [f for f in list(data['data'][0].keys()) if not f.startswith('zc_') and f not in ['ID', 'Added_User', 'Modified_User']]
            else:
                fields = []
        else:
            fields = ["Scholar_Name", "Scholar_ID", "Tracking_ID", "Account_Number", "Bank_Name", "Account_Holder_Name", "IFSC_Code", "Branch_Name", "Bill_Data", "Bill1_Amount", "Bill2_Amount", "Bill3_Amount", "Bill4_Amount", "Bill5_Amount", "Total_Amount", "Tokens_Used", "Status"]

        return JSONResponse(content={"success": True, "fields": fields, "message": f"Found {len(fields)} fields"})

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/zoho/dynamic-push")
async def dynamic_push_to_zoho(request: DynamicPushRequest):
    try:
        if not supabase:
            return JSONResponse(status_code=400, content={"success": False, "error": "Supabase not configured"})

        record_ids_str = [str(rid) for rid in request.record_ids]
        records = []
        for record_id in record_ids_str:
            try:
                response = supabase.table('auto_extraction_results').select('*').eq('id', record_id).execute()
                if response.data and len(response.data) > 0:
                    records.append(response.data[0])
            except:
                pass

        if not records:
            return JSONResponse(status_code=400, content={"success": False, "error": "No valid records found"})

        result = zoho_bulk.bulk_insert(records)
        return JSONResponse(content={"success": result['successful'] > 0, "details": result})

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ============================================================
# BARCODE DEBUG ENDPOINTS
# ============================================================

@app.post("/barcode/test-extraction")
async def test_barcode_extraction(file_url: str = Form(...)):
    debug_log = []
    try:
        file_content, filename = download_file_from_url(file_url)
        debug_log.append(f"Downloaded: {filename} ({len(file_content)} bytes)")

        from ai_analyzer import validate_and_optimize_image_single
        optimized_content, mime_type = validate_and_optimize_image_single(file_content, filename)
        debug_log.append(f"Image validated: {mime_type}")

        result = analyze_barcode_gemini_vision(file_content, filename)
        debug_log.append(f"Gemini response: success={result.get('success')}")

        if not result.get('success'):
            return JSONResponse(content={"success": False, "stage": "gemini_analysis", "error": result.get('error'), "debug_log": debug_log})

        all_barcodes = result.get('all_barcodes', [])
        debug_log.append(f"Extracted {len(all_barcodes)} barcodes")

        return JSONResponse(content={"success": True, "stage": "completed", "result": result, "debug_log": debug_log, "message": "Extraction pipeline working perfectly!"})

    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e), "debug_log": debug_log})


@app.get("/barcode/list-recent-results/{limit}")
async def list_recent_results(limit: int = 10):
    try:
        response = supabase.table("barcode_extraction_results").select("*").order("created_at", desc=True).limit(limit).execute()
        return JSONResponse(content={"success": True, "total_results": len(response.data), "results": response.data})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    print("="*80)
    print("OCR API - COMPLETE WITH AUTO-EXTRACT + AUTH + MULTI-BILL IMAGE")
    print("="*80)
    print(f"✅ Gemini Vision: {'ENABLED' if USE_GEMINI else 'DISABLED'}")
    print(f"✅ Supabase: {'CONNECTED' if supabase else 'NOT CONFIGURED'}")
    print(f"✅ Multi-token OAuth: {len(ZOHO_TOKENS)} tokens")
    print(f"✅ Authentication: {'ENABLED' if AUTHENTIK_URL and AUTHENTIK_API_TOKEN else 'DISABLED'}")
    print(f"✅ Multi-bill image extraction: ENABLED")
    print("="*80)
    print(f"\nStarting server on http://0.0.0.0:{port}")
    print("="*80 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port)