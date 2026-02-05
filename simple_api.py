"""
OCR API - COMPLETE VERSION with Auto-Extract + Authentication
✅ Gemini Vision OCR
✅ Auto-extraction with job tracking
✅ Multi-token OAuth support
✅ Supabase integration
✅ Authentik authentication
"""

from fastapi import FastAPI, File, UploadFile, Form, Depends, HTTPException, status, Request, Response
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional, List, Dict, Any
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
import secrets
import httpx

load_dotenv()

# ============================================================
# AUTHENTICATION CONFIGURATION
# ============================================================

AUTHENTIK_URL = os.getenv("AUTHENTIK_URL", "https://authentik.teameverest.ngo")
AUTHENTIK_API_TOKEN = os.getenv("AUTHENTIK_API_TOKEN", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))

# In-memory session storage (use Redis in production)
sessions = {}

# ============================================================
# AUTHENTICATION MODELS
# ============================================================

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

# ============================================================
# FASTAPI APP INITIALIZATION
# ============================================================

app = FastAPI(title="OCR API - Complete with Auto-Extract + Auth")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "*"],  # Update this in production
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
        print("[SUPABASE] ⚠️ Not configured (optional for auto-extract)")
except ImportError:
    supabase = None
    print("[SUPABASE] ⚠️ supabase-py not installed (optional)")

# ============================================================
# ZOHO MULTI-TOKEN CONFIGURATION
# ============================================================

ZOHO_TOKENS = [
    # READ TOKENS - Primary Pool (1-2)
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
    
    # CREATE TOKEN (3)
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
    
    # EXTENDED READ TOKENS (4-8) - Dual Purpose Tokens used as READ
    {
        "name": "Token4_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_4"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_4"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_4"),
        "scope": "read",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    {
        "name": "Token5_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_5"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_5"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_5"),
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
    
    # NEW READ TOKENS (9-18)
    {
        "name": "Token9_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_9"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_9"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_9"),
        "scope": "read",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    {
        "name": "Token10_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_10"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_10"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_10"),
        "scope": "read",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    {
        "name": "Token11_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_11"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_11"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_11"),
        "scope": "read",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    {
        "name": "Token12_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_12"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_12"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_12"),
        "scope": "read",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    {
        "name": "Token13_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_13"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_13"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_13"),
        "scope": "read",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    {
        "name": "Token14_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_14"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_14"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_14"),
        "scope": "read",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    {
        "name": "Token15_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_15"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_15"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_15"),
        "scope": "read",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    {
        "name": "Token16_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_16"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_16"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_16"),
        "scope": "read",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    {
        "name": "Token17_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_17"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_17"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_17"),
        "scope": "read",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    {
        "name": "Token18_Read",
        "client_id": os.getenv("ZOHO_CLIENT_ID_18"),
        "client_secret": os.getenv("ZOHO_CLIENT_SECRET_18"),
        "refresh_token": os.getenv("ZOHO_REFRESH_TOKEN_18"),
        "scope": "read",
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
]

# Remove tokens with missing credentials
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

# Fallback to single token config
ZOHO_CLIENT_ID = os.getenv("ZOHO_CLIENT_ID")
ZOHO_CLIENT_SECRET = os.getenv("ZOHO_CLIENT_SECRET")
ZOHO_REFRESH_TOKEN = os.getenv("ZOHO_REFRESH_TOKEN")

# Thread-safe token rotation
token_lock = threading.Lock()
current_read_index = 0
current_create_index = 0

print(f"[TOKENS] Loaded {len(ZOHO_TOKENS)} tokens")

# ============================================================
# AUTHENTICATION HELPER FUNCTIONS
# ============================================================

def create_session_token() -> str:
    """Create a secure session token"""
    return secrets.token_urlsafe(32)

async def find_authentik_user(username: str) -> Optional[Dict]:
    """Find user in Authentik by username"""
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
    """Determine user type from groups"""
    group_names = [g.get("name", "") for g in groups]
    
    if "ECR Student" in group_names:
        return "ecr_student"
    elif "ICM Student" in group_names:
        return "icm_student"
    elif "IATC Admin" in group_names:
        return "admin"
    
    return None

async def get_current_user(request: Request) -> Optional[User]:
    """Dependency to get current authenticated user (optional)"""
    session_token = request.cookies.get("session_token")
    
    if not session_token or session_token not in sessions:
        return None
    
    session = sessions[session_token]
    if datetime.now() > session["expires_at"]:
        del sessions[session_token]
        return None
    
    return User(**session["user"])

async def require_auth(request: Request) -> User:
    """Dependency to require authentication"""
    user = await get_current_user(request)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    return user

# ============================================================
# TOKEN CACHING (ADD THIS NEAR THE TOP)
# ============================================================

# Cache structure: {token_config_name: {"access_token": "...", "expires_at": timestamp}}
token_cache = {}
token_cache_lock = threading.Lock()

# ============================================================
# TOKEN MANAGEMENT (REPLACE YOUR EXISTING get_zoho_token)
# ============================================================

def get_zoho_token(scope_needed: str = "read", max_retries: int = None) -> tuple:
    """Get Zoho token with automatic fallback and caching"""
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
            
            # ✅ CHECK CACHE FIRST
            with token_cache_lock:
                if token_name in token_cache:
                    cached = token_cache[token_name]
                    remaining_time = cached["expires_at"] - time.time()
                    
                    # If more than 5 minutes remaining, use cached token
                    if remaining_time > 300:
                        print(f"[TOKEN] ✓ Using cached {token_name} (expires in {int(remaining_time/60)} min)")
                        # ❌ REMOVED: Don't rotate! Keep using the same token
                        return cached["access_token"], token_name
                    else:
                        print(f"[TOKEN] ⚠️ Cached {token_name} expiring soon ({int(remaining_time)}s), refreshing...")
            
            try:
                # Rate limiting
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
                
                # Check for error response
                if "error" in response_data:
                    error_msg = response_data.get("error")
                    error_desc = response_data.get("error_description", "")
                    print(f"[TOKEN] ✗ {token_name} OAuth error: {error_msg} - {error_desc}")
                    token_config["error_count"] += 1
                    token_config["status"] = "invalid"
                    continue
                
                # Check for access token
                if "access_token" not in response_data:
                    print(f"[TOKEN] ✗ {token_name} missing access_token in response")
                    token_config["error_count"] += 1
                    continue
                
                access_token = response_data["access_token"]
                expires_in = response_data.get("expires_in", 3600)
                
                # ✅ CACHE THE TOKEN
                with token_cache_lock:
                    token_cache[token_name] = {
                        "access_token": access_token,
                        "expires_at": time.time() + expires_in
                    }
                
                token_config["last_used"] = time.time()
                token_config["request_count"] += 1
                
                # ✅ ONLY rotate on FIRST successful generation (not on cache hits)
                if scope_needed == "read":
                    current_read_index = token_index  # Stay on this token
                else:
                    current_create_index = token_index  # Stay on this token
                
                print(f"[TOKEN] ✓ Generated new {token_name} (valid for {expires_in}s, cached)")
                return access_token, token_name
                
            except Exception as e:
                token_config["error_count"] += 1
                print(f"[TOKEN] ✗ {token_name} failed: {e}")
                
                if attempt < max_retries - 1:
                    continue
    
    return None, None

def get_zoho_access_token():
    """Legacy single token auth with fallback"""
    if not all([ZOHO_CLIENT_ID, ZOHO_CLIENT_SECRET, ZOHO_REFRESH_TOKEN]):
        print("[ZOHO AUTH] Using multi-token pool...")
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
        print(f"[ZOHO AUTH] Legacy failed: {e}, trying token pool...")
        return get_zoho_token(scope_needed="create")[0]


def retry_on_network_error(max_retries=3, delay=2):
    """Decorator to retry function on network errors"""
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
                        wait_time = delay * (attempt + 1)
                        print(f"[RETRY] Network error, waiting {wait_time}s...")
                        time.sleep(wait_time)
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
    """Upload image to Supabase Storage"""
    if not supabase:
        raise Exception("Supabase not configured")
    
    timestamp = int(time.time() * 1000)
    unique_filename = f"{folder}/{timestamp}_{filename}"
    
    supabase.storage.from_(SUPABASE_BUCKET).upload(
        unique_filename,
        file_content,
        file_options={"content-type": "image/jpeg"}
    )
    
    public_url = supabase.storage.from_(SUPABASE_BUCKET).get_public_url(unique_filename)
    print(f"[SUPABASE] ✓ Uploaded: {unique_filename}")
    return public_url


@retry_on_network_error(max_retries=3, delay=2)
def save_extraction_result(
    job_id: str,
    record_id: str,
    app_link_name: str,
    report_link_name: str,
    student_name: str,
    scholar_id: Optional[str],
    tracking_id: Optional[str],
    bank_image_supabase: Optional[str],
    bill_image_supabase: Optional[List[str]],
    bank_data: Optional[Dict],
    bill_data: Optional[List[Dict]],
    status: str,
    error_message: Optional[str] = None,
    processing_time_ms: Optional[int] = None,
    tokens_used: Optional[int] = None,
    cost_usd: Optional[float] = None
):
    """Save extraction result to Supabase"""
    if not supabase:
        return None
    
    data = {
        "job_id": job_id,
        "record_id": record_id,
        "app_link_name": app_link_name,
        "report_link_name": report_link_name,
        "student_name": student_name,
        "scholar_id": scholar_id,  
        "Tracking_id": tracking_id,
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


@retry_on_network_error(max_retries=3, delay=2)
def update_job_status(job_id: str, data: Dict[str, Any]):
    """Update job status in Supabase"""
    if not supabase:
        return None
    
    result = supabase.table("auto_extraction_jobs")\
        .update(data)\
        .eq("job_id", job_id)\
        .execute()
    
    return result

# ============================================================
# ZOHO FETCH FUNCTIONS
# ============================================================

def extract_all_image_urls(field_value, max_images=5):
    """Extract ALL image URLs from Zoho field"""
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
            elif isinstance(item, dict) and item.get("download_url"):
                urls.append(item["download_url"])
    elif isinstance(field_value, dict) and field_value.get("download_url"):
        urls.append(field_value["download_url"])
    
    return urls[:max_images]


def fetch_zoho_records(app_link_name: str, report_link_name: str, 
                       criteria: Optional[str] = None, 
                       max_records: int = None) -> List[Dict]:
    """Fetch records from Zoho Creator"""
    try:
        access_token, token_name = get_zoho_token(scope_needed="read")
        
        if not access_token:
            raise Exception("Failed to get READ token")
        
        api_url = f"https://creator.zoho.com/api/v2.1/{app_link_name}/report/{report_link_name}"
        
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
        params = {"from": 1, "limit": 200}
        
        if criteria:
            params["criteria"] = criteria
        
        all_records = []
        page = 1
        
        print(f"[ZOHO] Fetching records from {report_link_name}...")
        
        response = requests.get(api_url, headers=headers, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        records = data.get("data", [])
        
        if not records:
            return []
        
        all_records.extend(records)
        print(f"[ZOHO] Page {page}: {len(records)} records")
        
        # Paginate
        while len(records) == 200:
            if max_records and len(all_records) >= max_records:
                break
            
            page += 1
            params["from"] = (page - 1) * 200 + 1
            
            response = requests.get(api_url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            records = data.get("data", [])
            
            if not records:
                break
            
            all_records.extend(records)
            print(f"[ZOHO] Page {page}: {len(records)} records")
            time.sleep(0.5)
        
        if max_records and len(all_records) > max_records:
            all_records = all_records[:max_records]
        
        print(f"[ZOHO] ✓ Total: {len(all_records)} records")
        return all_records
        
    except Exception as e:
        print(f"[ZOHO] ✗ Error: {e}")
        raise


def fetch_specific_records_by_ids(app_link_name: str, report_link_name: str, record_ids: List[str]) -> List[Dict]:
    """Fetch specific records by ID"""
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
            response.raise_for_status()
            
            data = response.json()
            records = data.get("data", [])
            all_records.extend(records)
            
            if i + batch_size < len(record_ids):
                time.sleep(0.5)
        
        print(f"[ZOHO] ✓ Fetched {len(all_records)} specific records")
        return all_records
        
    except Exception as e:
        print(f"[ZOHO] ✗ Error: {e}")
        raise


def get_already_extracted_record_ids(app_link_name: str, report_link_name: str) -> set:
    """Get IDs of already extracted records"""
    if not supabase:
        return set()
    
    try:
        response = supabase.table("auto_extraction_results")\
            .select("record_id")\
            .eq("app_link_name", app_link_name)\
            .eq("report_link_name", report_link_name)\
            .eq("status", "success")\
            .execute()
        
        if response.data:
            extracted_ids = {str(r["record_id"]) for r in response.data}
            print(f"[FILTER] Found {len(extracted_ids)} already extracted")
            return extracted_ids
        
        return set()
        
    except Exception as e:
        print(f"[FILTER] Error: {e}")
        return set()


def check_active_jobs_for_records(record_ids: List[str]) -> Optional[str]:
    """Check if records are currently being processed"""
    if not supabase or not record_ids:
        return None
    
    try:
        response = supabase.table("auto_extraction_jobs")\
            .select("job_id, status")\
            .in_("status", ["pending", "running"])\
            .execute()
        
        if not response.data:
            return None
        
        for job in response.data:
            job_id = job["job_id"]
            
            results = supabase.table("auto_extraction_results")\
                .select("record_id")\
                .eq("job_id", job_id)\
                .execute()
            
            if results.data:
                active_record_ids = {str(r["record_id"]) for r in results.data}
                overlap = set(record_ids) & active_record_ids
                
                if overlap:
                    print(f"[DUPLICATE] Found {len(overlap)} records in job {job_id}")
                    return job_id
        
        return None
        
    except Exception as e:
        print(f"[DUPLICATE CHECK] Error: {e}")
        return None

# ============================================================
# HELPER FUNCTIONS (from simple_api.py)
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


def validate_file_format(file_content: bytes, filename: str) -> dict:
    """Validate file format"""
    result = {
        "valid": False,
        "format": "Unknown",
        "size": len(file_content),
        "message": ""
    }
    
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
    """Download file from URL with proper file type detection"""
    print(f"[DOWNLOAD] Downloading...")
    
    if file_url.startswith('/api/v2.1/'):
        file_url = f"https://creator.zoho.com{file_url}"
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        
        if "creator.zoho.com" in file_url or "creator.zoho.in" in file_url:
            access_token, token_name = get_zoho_token(scope_needed="read")
            
            if not access_token:
                access_token = get_zoho_access_token()
            
            if access_token:
                headers["Authorization"] = f"Zoho-oauthtoken {access_token}"
        
        response = requests.get(file_url, timeout=30, headers=headers, stream=True)
        response.raise_for_status()
        
        file_content = response.content
        
        # ✅ FIX: Detect actual file type from content (magic bytes)
        if file_content[:4] == b'%PDF':
            # It's a PDF
            extension = '.pdf'
            print(f"[DOWNLOAD] ⚠️ Detected PDF file")
        elif file_content[:2] == b'\xff\xd8':
            # JPEG
            extension = '.jpg'
        elif file_content[:4] == b'\x89PNG':
            # PNG
            extension = '.png'
        elif len(file_content) > 12 and file_content[8:12] == b'WEBP':
            # WebP
            extension = '.webp'
        else:
            # Unknown, check URL or Content-Type header
            content_type = response.headers.get('Content-Type', '').lower()
            if 'pdf' in content_type:
                extension = '.pdf'
            elif 'jpeg' in content_type or 'jpg' in content_type:
                extension = '.jpg'
            elif 'png' in content_type:
                extension = '.png'
            else:
                # Last resort: check URL
                url_lower = file_url.lower()
                if '.pdf' in url_lower:
                    extension = '.pdf'
                elif any(ext in url_lower for ext in ['.jpg', '.jpeg']):
                    extension = '.jpg'
                elif '.png' in url_lower:
                    extension = '.png'
                else:
                    extension = '.jpg'  # Final fallback
        
        # Get filename from URL or use generic name
        filename = file_url.split('/')[-1].split('?')[0]
        
        # If filename doesn't have proper extension, add detected one
        if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.pdf', '.webp']):
            filename = f"downloaded_file{extension}"
        else:
            # Replace extension with detected one if they don't match
            base_name = filename.rsplit('.', 1)[0]
            filename = f"{base_name}{extension}"
        
        print(f"[DOWNLOAD] ✓ {filename} ({len(file_content):,} bytes)")
        return file_content, filename
        
    except Exception as e:
        print(f"[DOWNLOAD] ✗ Error: {str(e)}")
        raise

def convert_filters_to_zoho_criteria(filters: list) -> str:
    """
    Convert UI filters to Zoho Creator criteria format
    """
    if not filters:
        return None
    
    criteria_parts = []
    
    for filter_item in filters:
        field = filter_item.get("field", "")
        operator = filter_item.get("operator", "")
        value = filter_item.get("value", "")
        
        if not field or not operator:
            continue
        
        # Convert operator to Zoho format
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
    
    if criteria_parts:
        return " && ".join(criteria_parts)
    
    return None

def process_single_file(file_content: bytes, filename: str, doc_type: str) -> dict:
    """Process a single file using Gemini Vision"""
    start_time = time.time()
    
    try:
        def calculate_tokens(text):
            return max(1, len(text) // 4)
        
        if not USE_GEMINI:
            return {
                "error": "Gemini Vision not configured",
                "success": False,
                "filename": filename
            }
        
        image_content = file_content
        original_filename = filename
        converted_from_pdf = False
        
        # PDF conversion
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
        
        # Save image
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
        result['token_usage'] = {
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'total_tokens': input_tokens + output_tokens
        }
        result['processing_time_ms'] = processing_time_ms
        result['image_url'] = image_url
        
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
# EXTRACTION WORKER
# ============================================================

def process_extraction_job(job_id: str, config: Dict):
    """Background job processor"""
    if not supabase:
        print("[AUTO EXTRACT] ✗ Supabase not configured")
        return
    
    try:
        print(f"\n{'='*80}")
        print(f"[AUTO EXTRACT] Starting Job: {job_id}")
        print(f"{'='*80}\n")
        
        update_job_status(job_id, {
            "status": "running",
            "started_at": datetime.now().isoformat()
        })
        
        # Fetch records
        selected_ids = config.get('selected_record_ids', [])
        
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
        
        print(f"[AUTO EXTRACT] Processing {len(records)} records...")
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
            """Extract student name from various possible fields"""
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
        
        def extract_scholar_id(record):
            """Extract Actual_Scholar_ID from record"""
            field_value = record.get("Actual_Scholar_ID")
            
            if not field_value:
                field_value = record.get("Scholar_ID") or record.get("ScholarID")
            
            if not field_value:
                return None
            
            if isinstance(field_value, str):
                return field_value.strip() if field_value.strip() else None
            
            if isinstance(field_value, dict):
                display_value = field_value.get("zc_display_value")
                if display_value:
                    return display_value.strip()
                
                id_value = field_value.get("ID")
                if id_value:
                    return str(id_value)
            
            return None
        
        def extract_tracking_id(record):
            """Extract Tracking_ID from record"""
            field_value = record.get("Tracking_ID")
            
            if not field_value:
                field_value = record.get("TrackingID") or record.get("Tracking_Id")
            
            if not field_value:
                return None
            
            if isinstance(field_value, str):
                return field_value.strip() if field_value.strip() else None
            
            if isinstance(field_value, dict):
                display_value = field_value.get("zc_display_value")
                if display_value:
                    return display_value.strip()
                
                id_value = field_value.get("ID")
                if id_value:
                    return str(id_value)
            
            return None
        
        # Process records
        for idx, record in enumerate(records, 1):
            record_start = time.time()
            
            record_id = str(record.get("ID"))
            student_name = extract_student_name(record)
            scholar_id = extract_scholar_id(record)
            tracking_id = extract_tracking_id(record)
            
            print(f"\n[AUTO EXTRACT] [{idx}/{len(records)}] {student_name} (ID: {record_id})")
            
            if scholar_id:
                print(f"[AUTO EXTRACT]   📋 Scholar ID: {scholar_id}")
            if tracking_id:
                print(f"[AUTO EXTRACT]   📋 Tracking ID: {tracking_id}")
            
            bank_image_url_supabase = None
            bill_images_supabase = []
            bank_data = None
            bill_data_array = []
            record_tokens = 0
            record_cost = 0.0
            error_msg = None

            try:
                # Process Bank Image
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
                                record_cost += calculate_cost(
                                    tokens.get('input_tokens', 0),
                                    tokens.get('output_tokens', 0),
                                    'gemini_vision'
                                )
                                
                                if supabase:
                                    bank_image_url_supabase = upload_to_supabase_storage(
                                        file_content,
                                        f"bank_{record_id}_{filename}",
                                        folder="auto-extract/bank"
                                    )
                                
                                print(f"[AUTO EXTRACT]   ✅ Bank extracted")
                            else:
                                error_msg = f"Bank OCR failed: {result.get('error')}"
                                print(f"[AUTO EXTRACT]   ✗ {error_msg}")
                        except Exception as e:
                            error_msg = f"Bank processing error: {str(e)}"
                            print(f"[AUTO EXTRACT]   ✗ {error_msg}")
                
                # Process Bill Images (multiple)
                if config.get('bill_field_name'):
                    bill_field_value = record.get(config['bill_field_name'])
                    bill_zoho_urls = extract_all_image_urls(bill_field_value, max_images=5)
                    
                    if bill_zoho_urls:
                        print(f"[AUTO EXTRACT]   📥 Found {len(bill_zoho_urls)} bill images")
                        
                        for bill_idx, bill_zoho_url in enumerate(bill_zoho_urls, 1):
                            try:
                                print(f"[AUTO EXTRACT]   📥 Downloading bill {bill_idx}...")
                                file_content, filename = download_file_from_url(bill_zoho_url)
                                
                                print(f"[AUTO EXTRACT]   🤖 Processing bill {bill_idx}...")
                                result = process_single_file(file_content, filename, "bill")
                                
                                if result.get('success'):
                                    bill_data_array.append(result)
                                    tokens = result.get('token_usage', {})
                                    record_tokens += tokens.get('total_tokens', 0)
                                    record_cost += calculate_cost(
                                        tokens.get('input_tokens', 0),
                                        tokens.get('output_tokens', 0),
                                        'gemini_vision'
                                    )
                                    
                                    if supabase:
                                        bill_image_url = upload_to_supabase_storage(
                                            file_content,
                                            f"bill{bill_idx}_{record_id}_{filename}",
                                            folder="auto-extract/bills"
                                        )
                                        bill_images_supabase.append(bill_image_url)
                                    
                                    print(f"[AUTO EXTRACT]   ✅ Bill {bill_idx} extracted")
                                else:
                                    error_detail = result.get('error')
                                    print(f"[AUTO EXTRACT]   ✗ Bill {bill_idx} failed: {error_detail}")
                                    if not error_msg:
                                        error_msg = f"Bill {bill_idx} OCR failed"
                                    
                            except Exception as e:
                                print(f"[AUTO EXTRACT]   ✗ Bill {bill_idx} error: {str(e)}")
                                if not error_msg:
                                    error_msg = f"Bill {bill_idx} processing error"
                
                # Determine status
                if bank_data or bill_data_array:
                    status = "success"
                    successful += 1
                else:
                    status = "failed"
                    failed += 1
                    if not error_msg:
                        error_msg = "No images found or OCR failed"
                
            except Exception as record_error:
                status = "failed"
                failed += 1
                error_msg = f"Record error: {str(record_error)}"
                print(f"[AUTO EXTRACT]   ✗ {error_msg}")
            
            processing_time = int((time.time() - record_start) * 1000)
            
            # Save result
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
                        bank_image_supabase=bank_image_url_supabase,
                        bill_image_supabase=bill_images_supabase,
                        bank_data=bank_data,
                        bill_data=bill_data_array,
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
            
            # Update progress
            update_job_status(job_id, {
                "processed_records": processed,
                "successful_records": successful,
                "failed_records": failed,
                "total_cost_usd": round(total_cost, 6)
            })
            
            print(f"[AUTO EXTRACT]   💰 ${record_cost:.6f} | ⏱️ {processing_time}ms | {status}")
            time.sleep(0.3)
        
        # Job completed
        update_job_status(job_id, {
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "total_cost_usd": round(total_cost, 6)
        })
        
        print(f"\n{'='*80}")
        print(f"[AUTO EXTRACT] ✅ Completed: {job_id}")
        print(f"[AUTO EXTRACT]   Success: {successful}/{len(records)}")
        print(f"[AUTO EXTRACT]   Failed: {failed}/{len(records)}")
        print(f"[AUTO EXTRACT]   Cost: ${total_cost:.6f}")
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
# AUTHENTICATION ENDPOINTS
# ============================================================

@app.post("/auth/login", response_model=LoginResponse)
async def login(login_data: LoginRequest, response: Response):
    """Authenticate user with Authentik"""
    try:
        username = login_data.username
        password = login_data.password

        if not username or '@' in username:
            return LoginResponse(
                success=False,
                message="Please login with your username, not email address"
            )

        if not password:
            return LoginResponse(
                success=False,
                message="Password is required"
            )

        print(f"\n🔐 LOGIN ATTEMPT for user: {username}")

        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            # Step 1: Initialize authentication flow
            flow_response = await client.post(
                f"{AUTHENTIK_URL}/api/v3/flows/executor/default-authentication-flow/",
                json={}
            )
            
            print(f"📥 Step 1 Status: {flow_response.status_code}")
            cookies = flow_response.cookies
            
            # Step 2: Submit username
            id_response = await client.post(
                f"{AUTHENTIK_URL}/api/v3/flows/executor/default-authentication-flow/",
                json={"uid_field": username},
                cookies=cookies
            )
            
            print(f"📥 Step 2 Final Status: {id_response.status_code}")
            
            if id_response.cookies:
                cookies.update(id_response.cookies)
            
            try:
                id_data = id_response.json()
                print(f"📥 Step 2 Component: {id_data.get('component')}")
            except Exception as e:
                print(f"❌ JSON parse error at Step 2: {e}")
                return LoginResponse(
                    success=False,
                    message="Invalid username. Please check and try again."
                )
            
            if id_data.get("component") != "ak-stage-password":
                print(f"❌ Invalid username - got component: {id_data.get('component')}")
                return LoginResponse(
                    success=False,
                    message="Invalid username. Please check and try again."
                )
            
            # Step 3: Submit password
            password_response = await client.post(
                f"{AUTHENTIK_URL}/api/v3/flows/executor/default-authentication-flow/",
                json={"password": password},
                cookies=cookies
            )
            
            print(f"📥 Step 3 Final Status: {password_response.status_code}")
            
            try:
                password_data = password_response.json()
            except Exception as e:
                print(f"❌ JSON parse error at Step 3: {e}")
                return LoginResponse(
                    success=False,
                    message="Authentication failed"
                )
            
            component = password_data.get("component")
            print(f"🔍 Password response component: {component}")
            
            # Step 4: Handle authentication result
            if component == "xak-flow-redirect":
                # ✅ SUCCESS
                authentik_user = await find_authentik_user(username)
                
                if not authentik_user:
                    print("❌ Could not find Authentik user")
                    return LoginResponse(
                        success=False,
                        message="User not found in system"
                    )
                
                user_groups = authentik_user.get("groups_obj", authentik_user.get("groups", []))
                user_type = get_user_type(user_groups)
                is_admin = any(g.get("name") == "IATC Admin" for g in user_groups)
                group_names = [g.get("name") for g in user_groups]
                
                if is_admin:
                    print("👑 ADMIN LOGIN DETECTED")
                
                print(f"👥 User Type: {user_type}")
                print(f"👥 User Groups: {', '.join(group_names)}")
                print("✅ Login successful")
                
                user = User(
                    id=str(authentik_user.get("pk")),
                    username=authentik_user.get("username"),
                    email=authentik_user.get("email", ""),
                    name=authentik_user.get("name", authentik_user.get("username")),
                    groups=group_names,
                    is_admin=is_admin,
                    user_type=user_type,
                    avatar=authentik_user.get("avatar", "")
                )
                
                session_token = create_session_token()
                sessions[session_token] = {
                    "user": user.model_dump(),
                    "created_at": datetime.now(),
                    "expires_at": datetime.now() + timedelta(hours=24),
                    "login_time": datetime.now().isoformat(),
                    "auth_method": "password"
                }
                
                response.set_cookie(
                    key="session_token",
                    value=session_token,
                    httponly=True,
                    secure=False,
                    samesite="lax",
                    max_age=86400
                )
                
                print(f"✅ Session created for {username}")
                print("=" * 50 + "\n")
                
                return LoginResponse(
                    success=True,
                    message="Authentication successful",
                    user=user
                )
                
            elif component == "ak-stage-identification":
                print("❌ Invalid username (detected at password stage)")
                return LoginResponse(
                    success=False,
                    message="Invalid username. Please check and try again."
                )
                
            elif component == "ak-stage-password":
                print("❌ Invalid password")
                return LoginResponse(
                    success=False,
                    message="Invalid password. Please try again."
                )
                
            else:
                print(f"❌ Unexpected component: {component}")
                return LoginResponse(
                    success=False,
                    message="Authentication failed. Please try again."
                )
                
    except httpx.ConnectError:
        print("❌ Connection error to Authentik")
        return LoginResponse(
            success=False,
            message="Authentication service unavailable"
        )
    except Exception as e:
        print(f"❌ Authentication error: {str(e)}")
        import traceback
        traceback.print_exc()
        return LoginResponse(
            success=False,
            message="Authentication failed"
        )

@app.get("/auth/me", response_model=User)
async def get_me(current_user: User = Depends(require_auth)):
    """Get current user information"""
    return current_user

@app.post("/auth/logout")
async def logout(request: Request, response: Response):
    """Logout user and clear session"""
    session_token = request.cookies.get("session_token")
    
    if session_token and session_token in sessions:
        user = sessions[session_token].get("user", {})
        print(f"👋 User logged out: {user.get('username')}")
        del sessions[session_token]
    
    response.delete_cookie("session_token")
    
    return {"success": True, "message": "Logged out successfully"}

@app.get("/auth/refresh")
async def refresh_token(request: Request):
    """Refresh session expiry"""
    session_token = request.cookies.get("session_token")
    
    if not session_token or session_token not in sessions:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    session = sessions[session_token]
    session["expires_at"] = datetime.now() + timedelta(hours=24)
    
    return {"success": True, "message": "Session refreshed"}

@app.get("/auth/sessions")
async def get_active_sessions(current_user: User = Depends(require_auth)):
    """Get active session count (admin only)"""
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return {
        "success": True,
        "active_sessions": len(sessions),
        "sessions": [
            {
                "username": s["user"]["username"],
                "created_at": s["created_at"].isoformat(),
                "expires_at": s["expires_at"].isoformat()
            }
            for s in sessions.values()
        ]
    }

# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root():
    return {
        "service": "OCR API - Complete with Auto-Extract + Auth",
        "version": "7.0 - WITH AUTH",
        "features": [
            "✅ Gemini Vision OCR",
            "✅ Auto-extraction with job tracking",
            "✅ Multi-token OAuth support",
            "✅ Supabase integration",
            "✅ PDF support",
            "✅ Bank & Bill extraction",
            "✅ Batch processing",
            "✅ Authentik authentication"
        ],
        "supabase_status": "✅ Connected" if supabase else "❌ Not configured",
        "gemini_vision_status": "✅ Available" if USE_GEMINI else "❌ Not configured",
        "authentication_status": "✅ Enabled" if AUTHENTIK_URL and AUTHENTIK_API_TOKEN else "❌ Not configured",
        "tokens_configured": len(ZOHO_TOKENS),
        "endpoints": {
            "POST /auth/login": "Login with username/password",
            "GET /auth/me": "Get current user info",
            "POST /auth/logout": "Logout current session",
            "GET /auth/refresh": "Refresh session token",
            "POST /ocr/bank": "Process bank passbook",
            "POST /ocr/bill": "Process college bill",
            "POST /ocr/auto-extract/fetch-fields": "Fetch report fields",
            "POST /ocr/auto-extract/preview": "Preview records",
            "POST /ocr/auto-extract/start": "Start extraction job",
            "GET /ocr/auto-extract/status/{job_id}": "Get job status",
            "GET /ocr/auto-extract/results/{job_id}": "Get job results",
            "GET /stats": "Usage statistics",
            "GET /logs": "Processing logs"
        }
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
    """Process bank passbook - supports single or multiple files"""
    # Optional authentication - get user if logged in
    current_user = await get_current_user(request)
    user_info = current_user.username if current_user else "anonymous"
    
    try:
        print(f"\n{'='*80}")
        print(f"[BANK OCR] Processing bank passbook(s)")
        if student_name:
            print(f"[BANK OCR] Student: {student_name}")
        print(f"[BANK OCR] User: {user_info}")
        print(f"{'='*80}")
        
        # Collect files
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
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": "No file provided"
            })
        
        print(f"[BANK OCR] Processing {len(files_to_process)} file(s)")
        
        # Process all files
        results = []
        total_tokens = {"input": 0, "output": 0, "total": 0}
        
        for file_content, filename in files_to_process:
            validation = validate_file_format(file_content, filename)
            if not validation['valid']:
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": f"Invalid file: {validation['message']}"
                })
                continue
            
            result = process_single_file(file_content, filename, "bank")
            
            # Calculate cost and log
            token_usage = result.get('token_usage', {})
            input_tok = token_usage.get('input_tokens', 0)
            output_tok = token_usage.get('output_tokens', 0)
            cost = calculate_cost(input_tok, output_tok, result.get('method', 'unknown'))
            
            log_processing(
                doc_type="bank",
                filename=filename,
                method=result.get('method', 'unknown'),
                input_tokens=input_tok,
                output_tokens=output_tok,
                total_tokens=token_usage.get('total_tokens', 0),
                cost_usd=cost,
                success=result.get('success', False),
                error_message=result.get('error'),
                student_name=student_name,
                scholarship_id=scholarship_id,
                extracted_data=result if result.get('success') else None,
                processing_time_ms=result.get('processing_time_ms'),
                image_url=result.get('image_url')
            )
            
            if result.get('success'):
                total_tokens["input"] += input_tok
                total_tokens["output"] += output_tok
                total_tokens["total"] += token_usage.get('total_tokens', 0)
            
            results.append({
                "filename": filename,
                **result
            })
            
            print(f"[BANK OCR] ✓ Completed: {filename}")
            print(f"[BANK OCR]   Cost: ${cost}")
        
        print(f"{'='*80}\n")
        
        if len(results) == 1:
            return JSONResponse(content=results[0])
        else:
            return JSONResponse(content={
                "success": True,
                "total_files": len(results),
                "results": results,
                "total_token_usage": total_tokens
            })
        
    except Exception as e:
        print(f"[BANK OCR] ✗ Error: {str(e)}")
        
        log_processing(
            doc_type="bank",
            filename="unknown",
            method="error",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            success=False,
            error_message=str(e),
            student_name=student_name,
            scholarship_id=scholarship_id
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
    """Process college bill - supports single or multiple files"""
    # Optional authentication
    current_user = await get_current_user(request)
    user_info = current_user.username if current_user else "anonymous"
    
    try:
        print(f"\n{'='*80}")
        print(f"[BILL OCR] Processing bill(s)")
        if student_name:
            print(f"[BILL OCR] Student: {student_name}")
        print(f"[BILL OCR] User: {user_info}")
        print(f"{'='*80}")
        
        # Collect files
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
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": "No file provided"
            })
        
        print(f"[BILL OCR] Processing {len(files_to_process)} file(s)")
        
        results = []
        total_tokens = {"input": 0, "output": 0, "total": 0}
        
        for file_content, filename in files_to_process:
            validation = validate_file_format(file_content, filename)
            if not validation['valid']:
                results.append({
                    "filename": filename,
                    "success": False,
                    "error": f"Invalid file: {validation['message']}"
                })
                continue
            
            result = process_single_file(file_content, filename, "bill")
            
            # Calculate cost and log
            token_usage = result.get('token_usage', {})
            input_tok = token_usage.get('input_tokens', 0)
            output_tok = token_usage.get('output_tokens', 0)
            cost = calculate_cost(input_tok, output_tok, result.get('method', 'unknown'))
            
            log_processing(
                doc_type="bill",
                filename=filename,
                method=result.get('method', 'unknown'),
                input_tokens=input_tok,
                output_tokens=output_tok,
                total_tokens=token_usage.get('total_tokens', 0),
                cost_usd=cost,
                success=result.get('success', False),
                error_message=result.get('error'),
                student_name=student_name,
                scholarship_id=scholarship_id,
                extracted_data=result if result.get('success') else None,
                processing_time_ms=result.get('processing_time_ms'),
                image_url=result.get('image_url')
            )
            
            if result.get('success'):
                total_tokens["input"] += input_tok
                total_tokens["output"] += output_tok
                total_tokens["total"] += token_usage.get('total_tokens', 0)
            
            results.append({
                "filename": filename,
                **result
            })
            
            print(f"[BILL OCR] ✓ Completed: {filename}")
            print(f"[BILL OCR]   Cost: ${cost}")
        
        print(f"{'='*80}\n")
        
        if len(results) == 1:
            return JSONResponse(content=results[0])
        else:
            return JSONResponse(content={
                "success": True,
                "total_files": len(results),
                "results": results,
                "total_token_usage": total_tokens
            })
        
    except Exception as e:
        print(f"[BILL OCR] ✗ Error: {str(e)}")
        
        log_processing(
            doc_type="bill",
            filename="unknown",
            method="error",
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            success=False,
            error_message=str(e),
            student_name=student_name,
            scholarship_id=scholarship_id
        )
        
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


@app.post("/sync/bulk-push-to-zoho-selected")
async def bulk_push_selected_to_zoho(request: dict):
    """
    Bulk push selected records to Zoho Creator
    """
    try:
        records = request.get('records', [])
        if not records:
            return JSONResponse(content={
                "success": False,
                "error": "No records provided"
            })
        
        print(f"\n{'='*80}")
        print(f"PUSHING {len(records)} SELECTED RECORDS TO ZOHO")
        print(f"{'='*80}\n")
        
        # Validate records
        if not isinstance(records, list):
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Records must be an array"
                }
            )
        
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                print(f"[ERROR] Record {idx} is not a dictionary: {type(record)}")
                return JSONResponse(
                    status_code=400,
                    content={
                        "success": False,
                        "error": f"Record at index {idx} is not a valid object"
                    }
                )
        
        # Use existing bulk push functionality
        result = zoho_bulk.bulk_insert(records)
        
        return JSONResponse(content={
            "success": True,
            "message": f"Pushed {result['successful']}/{result['total_records']} records",
            "details": result
        })
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

# ============================================================
# ✅ AUTO-EXTRACT ENDPOINTS
# ============================================================

@app.post("/ocr/auto-extract/fetch-fields")
async def fetch_report_fields(
    app_link_name: str = Form(...),
    report_link_name: str = Form(...)
):
    """Fetch report schema/fields for dropdown selection"""
    try:
        print(f"[FETCH FIELDS] Fetching schema for {report_link_name}...")
        
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
            "all_fields": sorted(all_fields)
        })
        
    except Exception as e:
        print(f"[FETCH FIELDS] ✗ Error: {str(e)}")
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
    store_images: str = Form("false"),
    fetch_all: str = Form("false"),
    include_already_extracted: str = Form("false"),
    max_records_limit: int = Form(3000)
):
    """Preview records for extraction"""
    try:
        print(f"[PREVIEW] Loading records from {report_link_name}...")

        zoho_criteria = None
        if filter_criteria:
            try:
                filters = json.loads(filter_criteria)
                zoho_criteria = convert_filters_to_zoho_criteria(filters)
                print(f"[PREVIEW] 🔍 Converted filters to: {zoho_criteria}")
            except Exception as e:
                print(f"[PREVIEW] ⚠️ Filter error: {e}")
                # Fallback: if already a string, use it
                if isinstance(filter_criteria, str) and filter_criteria.strip():
                    zoho_criteria = filter_criteria
        
        fetch_all_bool = fetch_all.lower() in ('true', '1', 'yes')
        include_already_extracted_bool = include_already_extracted.lower() in ('true', '1', 'yes')

        if fetch_all_bool:
            max_records = min(max_records_limit, 10000)
            print(f"[PREVIEW] ⚠️ Fetch all enabled - limiting to {max_records} records for safety")
        else:
            max_records = 1000
        
        records = fetch_zoho_records(
            app_link_name=app_link_name,
            report_link_name=report_link_name,
            criteria=zoho_criteria,
            max_records=max_records
        )
        
        total_fetched = len(records)
        
        already_extracted = set()
        if not include_already_extracted_bool:
            already_extracted = get_already_extracted_record_ids(app_link_name, report_link_name)
            records = [r for r in records if str(r.get("ID", "")) not in already_extracted]
        
        def extract_name(record):
            for field in ["Name", "Student_Name", "Scholar_Name"]:
                name_value = record.get(field)
                if isinstance(name_value, str):
                    return name_value
                if isinstance(name_value, dict):
                    if name_value.get("zc_display_value"):
                        return name_value["zc_display_value"]
            return "Unknown"
        
        def extract_image_url(field_value):
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
        for record in records:
            record_id = str(record.get("ID", ""))
            student_name = extract_name(record)
            
            bank_value = record.get(bank_field_name) if bank_field_name else None
            bill_value = record.get(bill_field_name) if bill_field_name else None
            
            bank_url = extract_image_url(bank_value)
            bill_url = extract_image_url(bill_value)
            
            all_records.append({
                "record_id": record_id,
                "student_name": student_name,
                "has_bank_image": bank_url is not None,
                "has_bill_image": bill_url is not None
            })
        
        print(f"[PREVIEW] ✅ Loaded {len(all_records)} records")
        
        return JSONResponse(content={
            "success": True,
            "total_records": len(all_records),
            "total_fetched_from_zoho": total_fetched,
            "already_extracted_count": len(already_extracted),
            "sample_records": all_records,
            "estimated_cost": f"${len(all_records) * 0.003:.4f}",
            "estimated_time_minutes": math.ceil(len(all_records) * 3 / 60)
        })
        
    except Exception as e:
        print(f"[PREVIEW] ✗ Error: {str(e)}")
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
    """Start extraction job"""
    if not supabase:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "Supabase not configured. Auto-extract requires Supabase."
        })
    
    try:
        # ✅ All lines below 'try:' should have 8 spaces (2 levels of indentation)
        zoho_criteria = None
        if filter_criteria:
            try:
                filters = json.loads(filter_criteria)
                zoho_criteria = convert_filters_to_zoho_criteria(filters)
                print(f"[START] 🔍 Converted filters to: {zoho_criteria}")
            except Exception as e:
                print(f"[START] ⚠️ Filter error: {e}")
                if isinstance(filter_criteria, str) and filter_criteria.strip():
                    zoho_criteria = filter_criteria

        selected_ids = []
        if selected_record_ids:
            try:
                raw_ids = json.loads(selected_record_ids)
                selected_ids = list(set([str(rid) for rid in raw_ids]))
                print(f"[START] Selected {len(selected_ids)} unique records")
            except Exception as parse_error:
                print(f"[START] Failed to parse IDs: {parse_error}")
        
        already_extracted = get_already_extracted_record_ids(app_link_name, report_link_name)
        
        if selected_ids:
            selected_ids = [rid for rid in selected_ids if rid not in already_extracted]
            
            if len(selected_ids) == 0:
                return JSONResponse(status_code=400, content={
                    "success": False,
                    "error": "All selected records have already been extracted"
                })
        
        if selected_ids:
            active_job = check_active_jobs_for_records(selected_ids)
            if active_job:
                return JSONResponse(status_code=409, content={
                    "success": False,
                    "error": f"Records are already being processed in job {active_job}",
                    "active_job_id": active_job,
                    "message": "⚠️ Duplicate job detected"
                })
        
        job_id = f"job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        config = {
            "app_link_name": app_link_name,
            "report_link_name": report_link_name,
            "bank_field_name": bank_field_name,
            "bill_field_name": bill_field_name,
            "filter_criteria": zoho_criteria,
            "selected_record_ids": selected_ids
        }
        
        supabase.table("auto_extraction_jobs").insert({
            "job_id": job_id,
            "app_link_name": app_link_name,
            "report_link_name": report_link_name,
            "bank_field_name": bank_field_name,
            "bill_field_name": bill_field_name,
            "filter_criteria": zoho_criteria,  # ✅ ALSO CHANGE THIS to zoho_criteria
            "status": "pending"
        }).execute()
        
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
            "message": f"🚀 Processing {len(selected_ids)} records",
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


# ============================================================
# STATS & MONITORING ENDPOINTS
# ============================================================

@app.get("/stats")
async def get_stats(days: int = 30):
    """Get usage statistics"""
    try:
        stats = get_usage_stats(days)
        return JSONResponse(content=stats)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/logs")
async def get_logs(limit: int = 100):
    """Get processing logs"""
    try:
        logs = get_all_logs(limit)
        return JSONResponse(content={"logs": logs})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.delete("/logs/{log_id}")
async def remove_log(log_id: int):
    """Delete a processing log"""
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
    """Get token usage statistics"""
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
            )
        }
        
        if token["scope"] == "read":
            read_tokens.append(token_stat)
        else:
            create_tokens.append(token_stat)
    
    return {
        "total_tokens": len(ZOHO_TOKENS),
        "active_tokens": sum(1 for t in ZOHO_TOKENS if t["status"] == "active"),
        "read_tokens": read_tokens,
        "create_tokens": create_tokens
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "OCR API - Complete with Auto-Extract + Auth",
        "version": "7.0 - WITH AUTH",
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
async def bulk_push_to_zoho(limit: int = 1000):
    """Bulk push records from Supabase to Zoho Creator"""
    try:
        print(f"\n{'='*80}")
        print(f"BULK SYNC TO ZOHO")
        print(f"{'='*80}")
        
        if not supabase:
            return JSONResponse(status_code=500, content={
                "success": False,
                "error": "Supabase not configured"
            })
        
        response = supabase.table('auto_extraction_results').select('*').limit(limit).execute()
        records = response.data
        
        if not records:
            return JSONResponse(content={
                "success": False,
                "message": "No records found"
            })
        
        result = zoho_bulk.bulk_insert(records)
        
        return JSONResponse(content={
            "success": True,
            "message": f"Synced {result['successful']}/{result['total_records']}",
            "details": result
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.get("/sync/test-zoho-connection")
async def test_zoho_connection():
    """Test Zoho Creator connection"""
    try:
        result = zoho_bulk.test_connection()
        return JSONResponse(content={
            "success": result['success'],
            "result": result
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


# ============================================================
# DYNAMIC ZOHO CREATOR ENDPOINTS
# ============================================================

class ZohoConfigRequest(BaseModel):
    owner_name: str
    app_name: str
    form_name: str

class ZohoConfig(BaseModel):
    owner_name: str
    app_name: str
    form_name: str

class DynamicPushRequest(BaseModel):
    config: ZohoConfig
    field_mapping: Dict[str, str] = {}  # Optional - not used
    record_ids: List[Any]  # Accept both strings and integers


@app.post("/zoho/get-form-fields")
async def get_zoho_form_fields(request: ZohoConfigRequest):
    """
    Fetch available fields from a Zoho Creator form
    This allows dynamic field mapping in the frontend
    """
    try:
        # Get access token
        access_token = os.getenv("ZOHO_ACCESS_TOKEN")
        if not access_token:
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": "Zoho access token not configured"
            })
        
        # Build Zoho Creator API URL
        form_url = f"https://creator.zoho.com/api/v2/{request.owner_name}/{request.app_name}/form/{request.form_name}"
        
        # Fetch form metadata to get field names
        headers = {
            'Authorization': f'Zoho-oauthtoken {access_token}'
        }
        
        # Try to get form fields by fetching a sample record or form metadata
        # Note: Zoho Creator doesn't have a direct "get fields" endpoint,
        # so we'll try to infer from form structure or use common fields
        
        # Common Zoho Creator field approach: try to add a test record (dry-run style)
        # or use the report API to get field structure
        report_url = f"https://creator.zoho.com/api/v2/{request.owner_name}/{request.app_name}/report/All_{request.form_name}"
        
        response = requests.get(
            report_url,
            headers=headers,
            params={'max_records': 1},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract field names from the first record or metadata
            if 'data' in data and len(data['data']) > 0:
                fields = list(data['data'][0].keys())
                # Filter out system fields
                fields = [f for f in fields if not f.startswith('zc_') and f not in ['ID', 'Added_User', 'Modified_User']]
            else:
                # If no records exist, try to get from metadata
                fields = []
                if 'fields' in data:
                    fields = [f['display_name'] for f in data['fields']]
        else:
            # Fallback: Return your known fields from the existing configuration
            print(f"⚠️  Could not fetch fields from Zoho (Status: {response.status_code})")
            print(f"Response: {response.text[:200]}")
            
            # Return common fields as fallback
            fields = [
                "Scholar_Name",
                "Scholar_ID", 
                "Tracking_ID",
                "Account_Number",
                "Bank_Name",
                "Account_Holder_Name",
                "IFSC_Code",
                "Branch_Name",
                "Bill_Data",
                "Bill1_Amount",
                "Bill2_Amount",
                "Bill3_Amount",
                "Bill4_Amount",
                "Bill5_Amount",
                "Total_Amount",
                "Tokens_Used",
                "Status"
            ]
        
        return JSONResponse(content={
            "success": True,
            "fields": fields,
            "message": f"Found {len(fields)} fields"
        })
        
    except Exception as e:
        print(f"❌ Error fetching Zoho form fields: {e}")
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.post("/zoho/dynamic-push")
async def dynamic_push_to_zoho(request: DynamicPushRequest):
    """
    Push records to Zoho Creator using existing zoho_bulk_api.py
    This uses the proven field mapping from zoho_bulk_api.py
    (Tracking_ID, Bill1-5_Amount, formatted Bill_Data, etc.)
    """
    try:
        if not supabase:
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": "Supabase not configured"
            })
        
        print(f"\n{'='*80}")
        print(f"DYNAMIC PUSH TO ZOHO")
        print(f"{'='*80}")
        print(f"Records to push: {len(request.record_ids)}")
        print(f"Record IDs: {request.record_ids[:5]}...")  # Show first 5
        print(f"{'='*80}\n")
        
        # Convert record_ids to strings
        record_ids_str = [str(rid) for rid in request.record_ids]
        
        # Fetch full records from Supabase
        records = []
        for record_id in record_ids_str:
            try:
                response = supabase.table('auto_extraction_results')\
                    .select('*')\
                    .eq('id', record_id)\
                    .execute()
                
                if response.data and len(response.data) > 0:
                    records.append(response.data[0])
                else:
                    print(f"⚠️  Record {record_id} not found")
            except Exception as fetch_error:
                print(f"❌ Error fetching record {record_id}: {fetch_error}")
        
        if not records:
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": "No valid records found"
            })
        
        print(f"✅ Successfully fetched {len(records)} records from Supabase")
        
        # Use existing zoho_bulk_api.py - proven to work!
        result = zoho_bulk.bulk_insert(records)
        
        print(f"\n{'='*80}")
        print(f"PUSH RESULT")
        print(f"{'='*80}")
        print(f"✅ Successful: {result['successful']}/{result['total_records']}")
        print(f"❌ Failed: {result['failed']}/{result['total_records']}")
        print(f"{'='*80}\n")
        
        return JSONResponse(content={
            "success": result['successful'] > 0,
            "details": result
        })
        
    except Exception as e:
        print(f"❌ Dynamic push error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


def get_nested_value(obj: dict, path: str):
    """
    Get value from nested dictionary using dot notation
    Example: get_nested_value(record, "bank_data.account_number")
    Also handles array indexing: "bill_data[0].amount"
    """
    try:
        keys = path.replace('[', '.').replace(']', '').split('.')
        value = obj
        
        for key in keys:
            if key.isdigit():
                value = value[int(key)]
            else:
                value = value.get(key) if isinstance(value, dict) else None
            
            if value is None:
                return None
        
        return value
    except:
        return None


def format_value_for_zoho(value):
    """Format values appropriately for Zoho Creator"""
    if value is None:
        return ""
    
    if isinstance(value, dict):
        # Convert dict to JSON string
        return json.dumps(value)
    
    if isinstance(value, list):
        # Convert list to JSON string
        return json.dumps(value)
    
    if isinstance(value, (int, float)):
        return value
    
    return str(value)


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    
    print("="*80)
    print("OCR API - COMPLETE WITH AUTO-EXTRACT + AUTHENTICATION")
    print("="*80)
    print(f"✅ Gemini Vision: {'ENABLED' if USE_GEMINI else 'DISABLED'}")
    print(f"✅ Supabase: {'CONNECTED' if supabase else 'NOT CONFIGURED'}")
    print(f"✅ Multi-token OAuth: {len(ZOHO_TOKENS)} tokens")
    print(f"✅ Auto-extraction: {'ENABLED' if supabase else 'DISABLED (needs Supabase)'}")
    print(f"✅ Authentication: {'ENABLED' if AUTHENTIK_URL and AUTHENTIK_API_TOKEN else 'DISABLED'}")
    print("="*80)
    print(f"\nStarting server on http://0.0.0.0:{port}")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)