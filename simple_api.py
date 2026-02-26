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
import secrets
import httpx
from enum import Enum
from typing import Literal

class SyncMode(str, Enum):
    INSERT = "insert"
    UPDATE = "update"
    AUTO = "auto"

# ✅ ADD THIS HERE - BEFORE ZohoSyncRequest
class ZohoConfig(BaseModel):
    owner_name: str
    app_name: str
    form_name: str

# ✅ NOW THIS WORKS
class ZohoSyncRequest(BaseModel):
    config: ZohoConfig  # ← Can now reference ZohoConfig
    record_ids: List[Any]
    sync_mode: Literal['insert', 'update', 'auto'] = 'auto'
    tag: Optional[str] = None
    tag_color: Optional[str] = 'blue'

load_dotenv()

# ============================================================
# AUTHENTICATION CONFIGURATION
# ============================================================

AUTHENTIK_URL = os.getenv("AUTHENTIK_URL", "https://authentik.teameverest.ngo")
AUTHENTIK_API_TOKEN = os.getenv("AUTHENTIK_API_TOKEN", "")
FRONTEND_URL = os.getenv("FRONTEND_URL", "https://810lwl70-5174.inc1.devtunnels.ms")
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

WORKDRIVE_TOKENS = [
    {
        "name": "Workdrive_Token1",
        "client_id": os.getenv("WORKDRIVE_CLIENT_ID_1"),
        "client_secret": os.getenv("WORKDRIVE_CLIENT_SECRET_1"),
        "refresh_token": os.getenv("WORKDRIVE_REFRESH_TOKEN_1"),
        "request_count": 0,
        "error_count": 0,
        "last_used": 0,
        "status": "active"
    },
    # Add more tokens as needed
]

WORKDRIVE_TOKENS = [t for t in WORKDRIVE_TOKENS if t["client_id"] and t["client_secret"] and t["refresh_token"]]


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

def get_workdrive_token() -> tuple:
    """Get Workdrive access token with caching"""
    token_name = "workdrive_token"
    
    # Check cache
    with token_cache_lock:
        if token_name in token_cache:
            cached = token_cache[token_name]
            remaining_time = cached["expires_at"] - time.time()
            
            if remaining_time > 300:
                print(f"[WORKDRIVE] ✓ Using cached token (expires in {int(remaining_time/60)} min)")
                return cached["access_token"], token_name
    
    # Get first available token
    if not WORKDRIVE_TOKENS:
        print("[WORKDRIVE] No tokens configured")
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
        
        # Cache token
        with token_cache_lock:
            token_cache[token_name] = {
                "access_token": access_token,
                "expires_at": time.time() + expires_in
            }
        
        print(f"[WORKDRIVE] ✓ Generated new token (valid for {expires_in}s)")
        return access_token, token_name
        
    except Exception as e:
        print(f"[WORKDRIVE] ✗ Token generation failed: {e}")
        return None, None


# ============================================================
# WORKDRIVE BARCODE EXTRACTION (FIXED)
# ============================================================

def fetch_workdrive_files(folder_id: str) -> List[Dict]:
    """Fetch files from Workdrive folder - FIXED VERSION"""
    try:
        access_token, token_name = get_workdrive_token()
        
        if not access_token:
            raise Exception("Failed to get Workdrive token")
        
        # ✅ CORRECT API ENDPOINT - matches your working script
        api_url = f"https://workdrive.zoho.com/api/v1/files/{folder_id}/files"
        headers = {
            "Authorization": f"Zoho-oauthtoken {access_token}",
            "Accept": "application/vnd.api+json"
        }
        
        print(f"[WORKDRIVE] Fetching files from folder: {folder_id}")
        print(f"[WORKDRIVE] API URL: {api_url}")
        
        response = requests.get(api_url, headers=headers, timeout=30)
        
        print(f"[WORKDRIVE] Status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"[WORKDRIVE] Error Response: {response.text[:500]}")
        
        response.raise_for_status()
        
        data = response.json()
        items = data.get("data", [])
        
        print(f"[WORKDRIVE] Found {len(items)} total items")
        
        # ✅ FIX: Include PDFs!
        image_files = []
        for item in items:
            item_type = item.get("type")
            attributes = item.get("attributes", {})
            name = attributes.get("name", "")
            item_id = item.get("id")
            size = attributes.get("size", 0)
            created_time = attributes.get("created_time")
            modified_time = attributes.get("modified_time")
            
            # Check if it's a file and is an image OR PDF
            if item_type == "files":
                # ✅ ADDED .pdf HERE
                is_supported = name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.svg', '.pdf'))
                
                if is_supported:
                    image_files.append({
                        "file_id": item_id,
                        "filename": name,
                        "size": size,
                        "created_time": created_time,
                        "modified_time": modified_time,
                        "extension": name.split(".")[-1].lower() if "." in name else "unknown"
                    })
                    print(f"[WORKDRIVE]   📄 {name} ({size} bytes)")
        
        print(f"[WORKDRIVE] ✓ Found {len(image_files)} supported files")
        return image_files
        
    except requests.exceptions.HTTPError as e:
        print(f"[WORKDRIVE] ✗ HTTP Error: {e}")
        print(f"[WORKDRIVE] Response: {e.response.text[:500]}")
        raise
    except Exception as e:
        print(f"[WORKDRIVE] ✗ Error fetching files: {e}")
        import traceback
        traceback.print_exc()
        raise

def download_workdrive_file(file_id: str) -> tuple:
    """Download file from Workdrive - FIXED VERSION"""
    try:
        access_token, token_name = get_workdrive_token()
        
        if not access_token:
            raise Exception("Failed to get Workdrive token")
        
        # ✅ CORRECT DOWNLOAD ENDPOINT - matches your script
        api_url = f"https://workdrive.zoho.com/api/v1/download/{file_id}"
        headers = {
            "Authorization": f"Zoho-oauthtoken {access_token}",
            "Accept": "application/vnd.api+json"
        }
        
        print(f"[WORKDRIVE] Downloading file: {file_id}")
        
        response = requests.get(api_url, headers=headers, timeout=30, stream=True)
        response.raise_for_status()
        
        file_content = response.content
        
        # Get filename from Content-Disposition header or use file_id
        filename = file_id
        if 'Content-Disposition' in response.headers:
            import re
            cd = response.headers['Content-Disposition']
            matches = re.findall('filename="?([^"]+)"?', cd)
            if matches:
                filename = matches[0]
        
        # If still no proper filename, default to jpg
        if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.pdf', '.gif', '.webp']):
            filename = f"{file_id}.jpg"
        
        print(f"[WORKDRIVE] ✓ Downloaded {filename} ({len(file_content):,} bytes)")
        return file_content, filename
        
    except Exception as e:
        print(f"[WORKDRIVE] ✗ Download failed: {e}")
        raise


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


# ============================================================
# ✅ FINAL FIX: save_extraction_result
# Saves to BOTH record_id and Record_id columns
# ============================================================

@retry_on_network_error(max_retries=3, delay=2)
def save_extraction_result(
    job_id: str,
    record_id: str,
    app_link_name: str,
    report_link_name: str,
    student_name: str,
    bank_image_supabase: Optional[str],
    bill_image_supabase: Optional[List[str]],
    bank_data: Optional[Dict],
    bill_data: Optional[List[Dict]],
    status: str,
    error_message: Optional[str] = None,
    processing_time_ms: Optional[int] = None,
    tokens_used: Optional[int] = None,
    cost_usd: Optional[float] = None,
    scholar_id: Optional[str] = None,
    tracking_id: Optional[str] = None,
    email: Optional[str] = None
):
    """
    Save extraction result to Supabase
    ✅ VERIFIED: Saves to both record_id AND Record_id columns
    ✅ Schema verified with your exact column list
    """
    if not supabase:
        return None
    
    # Build data with ALL your schema columns
    data = {
        # Required fields
        "job_id": job_id,
        "record_id": record_id,              # ✅ Lowercase (primary)
        "Record_id": record_id,              # ✅ CAPITAL R (display field)
        "app_link_name": app_link_name,
        "report_link_name": report_link_name,
        "status": status,
        
        # Student info
        "student_name": student_name,
        "email": email,
        "scholar_id": scholar_id,
        "tracking_id": tracking_id,
        
        # Extraction data
        "bank_image_supabase": bank_image_supabase,
        "bill_image_supabase": bill_image_supabase,
        "bank_data": bank_data,
        "bill_data": bill_data,
        
        # Status tracking
        "error_message": error_message,
        "push_status": None,  # Will be set when pushed to Zoho
        
        # Processing metrics
        "processing_time_ms": processing_time_ms,
        "tokens_used": tokens_used,
        "cost_usd": float(cost_usd) if cost_usd else 0.0,
        
        # Timestamp
        "processed_at": datetime.now().isoformat()
    }
    
    try:
        result = supabase.table("auto_extraction_results").insert(data).execute()
        
        if result.data:
            row = result.data[0]
            db_id = row.get('id')
            db_record_id = row.get('record_id')
            db_Record_id = row.get('Record_id')
            db_email = row.get('email')
            
            print(f"[SAVE] ✅ Saved to Supabase (row_id: {db_id})")
            print(f"[SAVE]    record_id: {db_record_id}")
            print(f"[SAVE]    Record_id: {db_Record_id}")
            print(f"[SAVE]    email: {db_email}")
            
            return row
        else:
            print(f"[SAVE] ⚠️ Insert succeeded but no data returned")
            return None
            
    except Exception as save_error:
        error_msg = str(save_error)
        print(f"[SAVE] ✗ Error saving to Supabase:")
        print(f"[SAVE]    {error_msg}")
        
        # Helpful diagnostics
        if "constraint" in error_msg.lower():
            print(f"[SAVE] 💡 Constraint error - check NOT NULL columns")
        elif "column" in error_msg.lower():
            print(f"[SAVE] 💡 Column error - check column names")
        
        raise


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
    """Fetch records from Zoho Creator - DESCENDING ID pagination"""
    try:
        access_token, token_name = get_zoho_token(scope_needed="read")
        
        if not access_token:
            raise Exception("Failed to get READ token")
        
        api_url = f"https://creator.zoho.com/api/v2.1/{app_link_name}/report/{report_link_name}"
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
        
        if criteria:
            # With criteria - single request
            print(f"[ZOHO] Fetching with criteria (no pagination due to Zoho API limitation)...")
            params = {"criteria": criteria, "from": 1, "limit": 200}
            
            response = requests.get(api_url, headers=headers, params=params, timeout=30)
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
            
            print(f"[ZOHO] ✓ Fetched {len(records)} records, {len(unique_records)} unique")
            print(f"[ZOHO] ⚠️ Zoho API limitation: Can only fetch 200 records max with filters")
            return unique_records
            
        else:
            # ✅ DESCENDING ID pagination
            all_records = []
            seen_ids = set()
            page = 1
            max_id = None  # Track the smallest ID we've seen
            
            print(f"[ZOHO] Fetching records from {report_link_name}...")
            
            while True:
                # Build parameters
                if max_id is None:
                    # First page
                    params = {"from": 1, "limit": 200}
                else:
                    # Subsequent pages: fetch records with ID < max_id (descending)
                    params = {
                        "criteria": f"ID < {max_id}",
                        "from": 1,
                        "limit": 200
                    }
                
                print(f"[ZOHO] Page {page}: Requesting {params}")
                
                response = requests.get(api_url, headers=headers, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                records = data.get("data", [])
                
                if not records:
                    print(f"[ZOHO] No more records returned")
                    break
                
                # Find min ID in this batch for next iteration
                batch_ids = [int(r.get("ID", 0)) for r in records if r.get("ID")]
                if batch_ids:
                    current_min_id = min(batch_ids)
                    current_max_id = max(batch_ids)
                    print(f"[ZOHO] Received {len(records)} records, ID range: {current_max_id} to {current_min_id}")
                    max_id = current_min_id  # ← Use minimum for next page
                
                # Deduplicate
                new_records = 0
                for record in records:
                    record_id = str(record.get("ID", ""))
                    if record_id and record_id not in seen_ids:
                        seen_ids.add(record_id)
                        all_records.append(record)
                        new_records += 1
                
                print(f"[ZOHO] Page {page}: {new_records} new records ({len(all_records)} total unique)")
                
                # Stop if no new records
                if new_records == 0:
                    print(f"[ZOHO] ⚠️ No new records on page {page}, stopping pagination")
                    break
                
                # Stop if max_records reached
                if max_records and len(all_records) >= max_records:
                    all_records = all_records[:max_records]
                    print(f"[ZOHO] ⚠️ Reached max_records limit of {max_records}")
                    break
                
                # Stop if we got less than a full page
                if len(records) < 200:
                    print(f"[ZOHO] Received partial page ({len(records)} records), assuming end of data")
                    break
                
                page += 1
                time.sleep(0.5)
            
            print(f"[ZOHO] ✓ Total: {len(all_records)} unique records")
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


def check_active_jobs_for_records(record_ids: List[str]) -> Optional[str]:
    if not supabase or not record_ids:
        return None

    try:
        response = supabase.table("auto_extraction_jobs") \
            .select("job_id, status") \
            .in_("status", ["pending", "running"]) \
            .execute()

        if not response.data:
            return None

        for job in response.data:
            job_id = job["job_id"]

            # ✅ FIX: paginate the results sub-query
            active_record_ids = set()
            from_idx = 0
            PAGE = 1000
            while True:
                results = supabase.table("auto_extraction_results") \
                    .select("record_id") \
                    .eq("job_id", job_id) \
                    .range(from_idx, from_idx + PAGE - 1) \
                    .execute()
                batch = results.data or []
                active_record_ids.update(str(r["record_id"]) for r in batch)
                if len(batch) < PAGE:
                    break
                from_idx += PAGE

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
# ✅ UPDATED: process_extraction_job with flexible field extraction
# ============================================================

def process_extraction_job(job_id: str, config: Dict):
    """
    Background job processor with flexible field extraction
    ✅ Extracts scholar_id/tracking_id if available
    ✅ Falls back to email/record_id if needed
    ✅ Supports multi-bill PDFs (one PDF → bill1, bill2, bill3…)
    """
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

        # ── Fetch records ─────────────────────────────────────────────
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

        # ── Helpers ───────────────────────────────────────────────────
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

        def extract_field_value(record, field_names):
            """
            Extract field value, trying multiple possible field names.
            Returns (value, was_found)
            """
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

        # ── Detect available fields from first record ─────────────────
        first_record = records[0] if records else {}
        first_keys = set(first_record.keys())

        has_scholar_id  = any(k in first_keys for k in ["Actual_Scholar_ID", "Scholar_ID", "ScholarID"])
        has_tracking_id = any(k in first_keys for k in ["Tracking_ID", "Tracking_Id", "TrackingID"])
        has_email       = any(k in first_keys for k in ["Email", "email", "student_email"])

        print(f"[AUTO EXTRACT] Field detection:")
        print(f"  Scholar ID:  {has_scholar_id}")
        print(f"  Tracking ID: {has_tracking_id}")
        print(f"  Email:       {has_email}\n")

        # ── Record loop ───────────────────────────────────────────────
        for idx, record in enumerate(records, 1):
            record_start = time.time()

            record_id    = str(record.get("ID"))
            student_name = extract_student_name(record)

            scholar_id  = None
            tracking_id = None
            email       = None

            if has_scholar_id:
                scholar_id, _ = extract_field_value(
                    record, ["Actual_Scholar_ID", "Scholar_ID", "ScholarID"]
                )
            if has_tracking_id:
                tracking_id, _ = extract_field_value(
                    record, ["Tracking_ID", "Tracking_Id", "TrackingID"]
                )
            if has_email:
                email, _ = extract_field_value(
                    record, ["Email", "email", "student_email"]
                )

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
                    bank_zoho_urls   = extract_all_image_urls(bank_field_value, max_images=1)

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
                                record_cost   += calculate_cost(
                                    tokens.get('input_tokens', 0),
                                    tokens.get('output_tokens', 0),
                                    'gemini_vision'
                                )
                                print(f"[AUTO EXTRACT]   ✅ Bank extracted")
                            else:
                                error_msg = f"Bank OCR failed: {result.get('error')}"
                                print(f"[AUTO EXTRACT]   ✗ {error_msg}")

                        except Exception as e:
                            error_msg = f"Bank processing error: {str(e)}"
                            print(f"[AUTO EXTRACT]   ✗ {error_msg}")

                # ── Process Bill Images (supports multi-bill PDFs) ────
                if config.get('bill_field_name'):
                    bill_field_value = record.get(config['bill_field_name'])
                    bill_zoho_urls   = extract_all_image_urls(bill_field_value, max_images=5)

                    if bill_zoho_urls:
                        print(f"[AUTO EXTRACT]   📥 Found {len(bill_zoho_urls)} bill file(s) in field")

                        for file_idx, bill_zoho_url in enumerate(bill_zoho_urls, 1):
                            try:
                                print(f"[AUTO EXTRACT]   📥 Downloading bill file {file_idx}...")
                                file_content, filename = download_file_from_url(bill_zoho_url)

                                # ── Is it a PDF? ──────────────────────
                                is_pdf = (
                                    file_content[:4] == b'%PDF'
                                    or file_content[:5] == b'\x0a%PDF'
                                )

                                if is_pdf:
                                    # Multi-page PDF → potentially multiple bills
                                    print(f"[AUTO EXTRACT]   📄 PDF detected for file {file_idx} "
                                          f"– running multi-page bill extraction")

                                    from ai_analyzer import analyze_bill_multi_page

                                    try:
                                        page_bills = analyze_bill_multi_page(file_content, filename)
                                    except Exception as mp_err:
                                        print(f"[AUTO EXTRACT]   ✗ Multi-page extraction failed: {mp_err}")
                                        if not error_msg:
                                            error_msg = f"Bill file {file_idx}: multi-page extraction failed"
                                        page_bills = []

                                    for bill_result in page_bills:
                                        # Wrap to match the shape process_single_file returns
                                        wrapped = {
                                            **bill_result,
                                            "success": True,
                                            "method": "gemini_vision",
                                            "filename": filename,
                                            "token_usage": {
                                                "input_tokens": 258,
                                                "output_tokens": 80,
                                                "total_tokens": 338,
                                            },
                                        }
                                        bill_data_array.append(wrapped)
                                        record_tokens += 338
                                        record_cost   += calculate_cost(258, 80, "gemini_vision")

                                    if page_bills:
                                        print(f"[AUTO EXTRACT]   ✅ File {file_idx}: "
                                              f"{len(page_bills)} bill(s) extracted from PDF")
                                    else:
                                        print(f"[AUTO EXTRACT]   ⚠️ File {file_idx}: "
                                              f"No bills extracted from PDF")
                                        if not error_msg:
                                            error_msg = f"Bill file {file_idx}: no bills found in PDF"

                                else:
                                    # Single image → one bill
                                    print(f"[AUTO EXTRACT]   🤖 Processing bill file {file_idx} (image)...")
                                    result = process_single_file(file_content, filename, "bill")

                                    if result.get('success'):
                                        bill_data_array.append(result)
                                        tokens = result.get('token_usage', {})
                                        record_tokens += tokens.get('total_tokens', 0)
                                        record_cost   += calculate_cost(
                                            tokens.get('input_tokens', 0),
                                            tokens.get('output_tokens', 0),
                                            'gemini_vision'
                                        )
                                        print(f"[AUTO EXTRACT]   ✅ Bill file {file_idx} extracted "
                                              f"(Amount: ₹{result.get('amount')})")
                                    else:
                                        err_detail = result.get('error')
                                        print(f"[AUTO EXTRACT]   ✗ Bill file {file_idx} failed: {err_detail}")
                                        if not error_msg:
                                            error_msg = f"Bill file {file_idx} OCR failed"

                            except Exception as e:
                                print(f"[AUTO EXTRACT]   ✗ Bill file {file_idx} error: {str(e)}")
                                if not error_msg:
                                    error_msg = f"Bill file {file_idx} processing error"

                        # Summary log
                        print(f"[AUTO EXTRACT]   📊 Total bills extracted for this record: "
                              f"{len(bill_data_array)}")
                        for i, b in enumerate(bill_data_array, 1):
                            print(f"[AUTO EXTRACT]      Bill {i}: "
                                  f"Page={b.get('page_number', '?')} | "
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

            # ── Save result ───────────────────────────────────────────
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
            processed  += 1

            update_job_status(job_id, {
                "processed_records":  processed,
                "successful_records": successful,
                "failed_records":     failed,
                "total_cost_usd":     round(total_cost, 6)
            })

            print(f"[AUTO EXTRACT]   💰 ${record_cost:.6f} | ⏱️ {processing_time}ms | {status}")
            time.sleep(0.3)

        # ── Job completed ─────────────────────────────────────────────
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

    except Exception as e:
        print(f"[AUTO EXTRACT] ✗ Job failed: {e}")
        import traceback
        traceback.print_exc()

        update_job_status(job_id, {
            "status":       "failed",
            "completed_at": datetime.now().isoformat()
        })
# ============================================================
# UPDATED ENDPOINTS (No Team ID needed!)
# ============================================================

@app.post("/barcode/workdrive/list-files")
async def list_workdrive_files(
    folder_id: str = Form(...)
):
    """List image files from Workdrive folder"""
    try:
        files = fetch_workdrive_files(folder_id)
        
        # Format files for frontend
        formatted_files = []
        for f in files:
            formatted_files.append({
                "file_id": f["file_id"],
                "filename": f["filename"],
                "size": f["size"],
                "created_time": f["created_time"],
                "modified_time": f["modified_time"],
                "extension": f["extension"]
            })
        
        return JSONResponse(content={
            "success": True,
            "total_files": len(formatted_files),
            "files": formatted_files
        })
        
    except Exception as e:
        print(f"[WORKDRIVE] ✗ Error: {e}")
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })

# Add these endpoints to your FastAPI app (in main.py or server file)
# These handle local file uploads for barcode extraction

@app.post("/barcode/local/start")
async def start_local_barcode_extraction(
    request: Request
):
    """Start barcode extraction job from local uploaded files"""
    if not supabase:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "Supabase not configured"
        })
    
    try:
        form = await request.form()
        uploaded_files = form.getlist("files")
        
        if not uploaded_files:
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": "No files provided"
            })
        
        job_id = f"barcode_local_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Save uploaded files temporarily
        temp_files = []
        for uploaded_file in uploaded_files:
            file_content = await uploaded_file.read()
            
            # Create temporary file path
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
        
        # Create job in Supabase
        supabase.table("barcode_extraction_jobs").insert({
            "job_id": job_id,
            "source": "local",
            "total_files": len(temp_files),
            "status": "pending"
        }).execute()
        
        # Start processing in background
        config = {
            "job_id": job_id,
            "source": "local",
            "temp_files": temp_files
        }
        
        thread = threading.Thread(
            target=process_local_barcode_extraction_job,
            args=(job_id, config)
        )
        thread.daemon = True
        thread.start()
        
        return JSONResponse(content={
            "success": True,
            "job_id": job_id,
            "status": "started",
            "message": f"🚀 Processing {len(temp_files)} files",
            "check_status_url": f"/barcode/workdrive/status/{job_id}"
        })
        
    except Exception as e:
        print(f"[LOCAL] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })

def process_local_barcode_extraction_job(job_id: str, config: Dict):
    """Background job processor with FIXES"""
    if not supabase:
        print("[LOCAL BARCODE] ✗ Supabase not configured")
        return
    
    try:
        print(f"\n{'='*80}")
        print(f"[LOCAL BARCODE] Starting Local Job: {job_id}")
        print(f"{'='*80}\n")
        
        supabase.table("barcode_extraction_jobs").update({
            "status": "running",
            "started_at": datetime.now().isoformat()
        }).eq("job_id", job_id).execute()
        
        temp_files = config['temp_files']
        total_cost = 0.0
        processed = 0
        successful = 0
        failed = 0
        total_barcodes_extracted = 0
        
        for idx, file_info in enumerate(temp_files, 1):
            record_start = time.time()
            
            file_id = file_info['file_id']
            filename = file_info['filename']
            temp_path = file_info['temp_path']
            
            print(f"\n[LOCAL BARCODE] [{idx}/{len(temp_files)}] Processing: {filename}")
            
            try:
                # Read file content
                with open(temp_path, "rb") as f:
                    file_content = f.read()
                
                # Extract barcodes
                result = analyze_barcode_gemini_vision(file_content, filename)
                
                if result.get('success'):
                    # ✅ FIX 1: SKIP image upload - just set to None
                    supabase_url = None
                    print(f"[LOCAL BARCODE]   📤 Image upload disabled (bucket not found)")
                    
                    # Calculate cost
                    tokens = result.get('token_usage', {})
                    cost = calculate_cost(
                        tokens.get('input_tokens', 0),
                        tokens.get('output_tokens', 0),
                        'gemini_vision'
                    )
                    total_cost += cost
                    
                    # Extract barcodes
                    all_barcodes = result.get('all_barcodes', [])
                    total_found = len(all_barcodes)
                    is_multipage = result.get('is_multipage', False)
                    pages_processed = result.get('total_pages_processed', 1)
                    
                    total_barcodes_extracted += total_found
                    
                    # ✅ FIX 2: Add delay between inserts to avoid "Resource temporarily unavailable"
                    if total_found > 0:
                        print(f"[LOCAL BARCODE]   💾 Saving {total_found} barcode(s) to database...")
                        print(f"[LOCAL BARCODE]   ⏱️ Adding 0.1s delay between inserts to prevent network errors...\n")
                        
                        saved_count = 0
                        for barcode_idx, barcode in enumerate(all_barcodes, 1):
                            try:
                                insert_data = {
                                    "job_id": job_id,
                                    "file_id": file_id,
                                    "filename": filename,
                                    "barcode_number": barcode_idx,
                                    "total_barcodes_in_file": total_found,
                                    "barcode_data": barcode.get('data'),
                                    "barcode_type": barcode.get('type'),
                                    "page_number": barcode.get('page', 1),
                                    "is_primary": barcode_idx == 1,
                                    "image_url": supabase_url,  # Will be None
                                    "status": "success",
                                    "tokens_used": tokens.get('total_tokens', 0),
                                    "cost_usd": cost / total_found if total_found > 0 else 0,
                                    "processing_time_ms": int((time.time() - record_start) * 1000),
                                    "is_multipage": is_multipage,
                                    "pages_processed": pages_processed,
                                    "source": "local"
                                }
                                
                                # ✅ FIX 2: Add delay BEFORE insert
                                if barcode_idx > 1:
                                    time.sleep(0.1)
                                
                                supabase.table("barcode_extraction_results").insert(insert_data).execute()
                                saved_count += 1
                                
                                # Only log every 5th barcode to reduce spam
                                if barcode_idx % 5 == 1 or barcode_idx == 1:
                                    print(f"[LOCAL BARCODE]      ✅ Saved barcode {barcode_idx}/{total_found}")
                                
                            except Exception as save_error:
                                print(f"[LOCAL BARCODE]      ❌ Failed barcode {barcode_idx}: {save_error}")
                                import traceback
                                traceback.print_exc()
                        
                        if saved_count == total_found:
                            successful += 1
                            print(f"[LOCAL BARCODE]   ✅ All {total_found} barcodes saved successfully\n")
                        else:
                            failed += 1
                            print(f"[LOCAL BARCODE]   ⚠️ Only {saved_count}/{total_found} barcodes saved\n")
                    else:
                        failed += 1
                        supabase.table("barcode_extraction_results").insert({
                            "job_id": job_id,
                            "file_id": file_id,
                            "filename": filename,
                            "status": "no_barcode",
                            "error_message": "No barcodes detected in image",
                            "source": "local"
                        }).execute()
                        print(f"[LOCAL BARCODE]   ⚠️ No barcodes found\n")
                else:
                    failed += 1
                    supabase.table("barcode_extraction_results").insert({
                        "job_id": job_id,
                        "file_id": file_id,
                        "filename": filename,
                        "status": "failed",
                        "error_message": result.get('error'),
                        "source": "local"
                    }).execute()
                    print(f"[LOCAL BARCODE]   ✗ Failed: {result.get('error')}\n")
                
            except Exception as file_error:
                failed += 1
                print(f"[LOCAL BARCODE]   ✗ Error: {file_error}\n")
                try:
                    supabase.table("barcode_extraction_results").insert({
                        "job_id": job_id,
                        "file_id": file_id,
                        "filename": filename,
                        "status": "failed",
                        "error_message": str(file_error),
                        "source": "local"
                    }).execute()
                except:
                    pass
            finally:
                # Clean up temp file
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except:
                    pass
            
            processed += 1
            
            # Update job progress
            try:
                supabase.table("barcode_extraction_jobs").update({
                    "processed_files": processed,
                    "successful_files": successful,
                    "failed_files": failed,
                    "total_cost_usd": round(total_cost, 6),
                    "total_barcodes_extracted": total_barcodes_extracted
                }).eq("job_id", job_id).execute()
            except:
                pass
        
        # Job completed
        supabase.table("barcode_extraction_jobs").update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "total_cost_usd": round(total_cost, 6),
            "total_barcodes_extracted": total_barcodes_extracted
        }).eq("job_id", job_id).execute()
        
        print(f"\n{'='*80}")
        print(f"[LOCAL BARCODE] ✅ Job Completed: {job_id}")
        print(f"{'='*80}")
        print(f"[LOCAL BARCODE]   Files Processed: {successful}/{len(temp_files)}")
        print(f"[LOCAL BARCODE]   Total Barcodes Extracted: {total_barcodes_extracted} ✅")
        print(f"[LOCAL BARCODE]   Failed: {failed}/{len(temp_files)}")
        print(f"[LOCAL BARCODE]   Total Cost: ${total_cost:.6f}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"[LOCAL BARCODE] ✗ Job failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            supabase.table("barcode_extraction_jobs").update({
                "status": "failed",
                "completed_at": datetime.now().isoformat()
            }).eq("job_id", job_id).execute()
        except:
            pass


# ============================================================
# BARCODE EXTRACTION DEBUGGER
# Add these endpoints to find exactly where extraction fails
# ============================================================

@app.post("/barcode/test-extraction")
async def test_barcode_extraction(
    file_url: str = Form(...)
):
    """
    Test barcode extraction end-to-end with detailed logging
    Helps identify exactly where the process fails
    """
    print(f"\n{'='*80}")
    print(f"[DEBUG] TESTING BARCODE EXTRACTION PIPELINE")
    print(f"{'='*80}")
    print(f"[DEBUG] File URL: {file_url}\n")
    
    debug_log = []
    
    try:
        # Step 1: Download file
        print(f"[DEBUG] STEP 1: Downloading file...")
        debug_log.append("STEP 1: Starting download")
        
        file_content, filename = download_file_from_url(file_url)
        
        print(f"[DEBUG] ✅ Downloaded: {filename}")
        print(f"[DEBUG]    Size: {len(file_content):,} bytes")
        print(f"[DEBUG]    First 20 bytes: {file_content[:20].hex()}\n")
        debug_log.append(f"Downloaded: {filename} ({len(file_content)} bytes)")
        
        # Step 2: Validate image
        print(f"[DEBUG] STEP 2: Validating image...")
        debug_log.append("STEP 2: Validating image")
        
        try:
            from ai_analyzer import validate_and_optimize_image_single
            optimized_content, mime_type = validate_and_optimize_image_single(file_content, filename)
            print(f"[DEBUG] ✅ Image valid")
            print(f"[DEBUG]    MIME type: {mime_type}")
            print(f"[DEBUG]    Size after optimization: {len(optimized_content):,} bytes\n")
            debug_log.append(f"Image validated: {mime_type}")
        except Exception as validation_error:
            print(f"[DEBUG] ❌ Image validation failed: {validation_error}\n")
            debug_log.append(f"Image validation failed: {validation_error}")
            raise
        
        # Step 3: Call Gemini Vision
        print(f"[DEBUG] STEP 3: Calling Gemini Vision API...")
        debug_log.append("STEP 3: Calling Gemini Vision")
        
        try:
            # ✅ Call the ACTUAL extraction function
            result = analyze_barcode_gemini_vision(file_content, filename)
            
            print(f"[DEBUG] ✅ Gemini response received")
            print(f"[DEBUG]    success: {result.get('success')}")
            print(f"[DEBUG]    error: {result.get('error')}")
            print(f"[DEBUG]    barcode_type: {result.get('barcode_type')}")
            print(f"[DEBUG]    barcode_data: {result.get('barcode_data')}")
            print(f"[DEBUG]    total_barcodes_found: {result.get('total_barcodes_found', 0)}")
            print(f"[DEBUG]    is_multipage: {result.get('is_multipage')}")
            print(f"[DEBUG]    total_pages_processed: {result.get('total_pages_processed')}")
            
            debug_log.append(f"Gemini response: success={result.get('success')}")
            
            if not result.get('success'):
                print(f"[DEBUG] ❌ Analysis failed!\n")
                debug_log.append(f"Analysis failed: {result.get('error')}")
                
                return JSONResponse(content={
                    "success": False,
                    "stage": "gemini_analysis",
                    "error": result.get('error'),
                    "debug_log": debug_log
                })
            
            # Step 4: Check all_barcodes
            print(f"\n[DEBUG] STEP 4: Checking extracted barcodes...")
            debug_log.append("STEP 4: Extracting barcodes")
            
            all_barcodes = result.get('all_barcodes', [])
            print(f"[DEBUG] ✅ all_barcodes extracted")
            print(f"[DEBUG]    Count: {len(all_barcodes)}")
            print(f"[DEBUG]    Type: {type(all_barcodes)}")
            print(f"[DEBUG]    Content:\n")
            
            for idx, bc in enumerate(all_barcodes[:5], 1):
                print(f"[DEBUG]       [{idx}] type={bc.get('type')}, data={bc.get('data')}, page={bc.get('page')}")
            
            if len(all_barcodes) > 5:
                print(f"[DEBUG]       ... and {len(all_barcodes) - 5} more")
            
            print()
            debug_log.append(f"Extracted {len(all_barcodes)} barcodes")
            
            if len(all_barcodes) == 0:
                print(f"[DEBUG] ⚠️ WARNING: No barcodes found!")
                print(f"[DEBUG]    This might be expected if image has no barcodes\n")
            
            # Step 5: Test database insert
            print(f"[DEBUG] STEP 5: Testing database insert...")
            debug_log.append("STEP 5: Testing insert")
            
            if len(all_barcodes) > 0:
                test_barcode = all_barcodes[0]
                
                test_data = {
                    "job_id": "debug_test",
                    "file_id": "debug_test",
                    "filename": filename,
                    "barcode_number": 1,
                    "total_barcodes_in_file": len(all_barcodes),
                    "barcode_data": test_barcode.get('data'),
                    "barcode_type": test_barcode.get('type'),
                    "page_number": test_barcode.get('page', 1),
                    "is_primary": True,
                    "status": "debug_test"
                }
                
                print(f"[DEBUG]    Test data: {test_data}\n")
                
                try:
                    supabase.table("barcode_extraction_results").insert(test_data).execute()
                    print(f"[DEBUG] ✅ Insert successful!\n")
                    debug_log.append("Insert successful")
                    
                    # Clean up test record
                    supabase.table("barcode_extraction_results")\
                        .delete()\
                        .eq("job_id", "debug_test")\
                        .execute()
                    print(f"[DEBUG]    (Test record cleaned up)\n")
                    
                except Exception as insert_error:
                    print(f"[DEBUG] ❌ Insert failed: {insert_error}\n")
                    debug_log.append(f"Insert failed: {insert_error}")
                    import traceback
                    traceback.print_exc()
                    raise
            else:
                print(f"[DEBUG] ⚠️ Skipping insert test (no barcodes to insert)\n")
                debug_log.append("No barcodes to insert")
            
            # Success!
            print(f"{'='*80}")
            print(f"[DEBUG] ✅ ALL TESTS PASSED!")
            print(f"{'='*80}\n")
            
            return JSONResponse(content={
                "success": True,
                "stage": "completed",
                "result": result,
                "debug_log": debug_log,
                "message": "Extraction pipeline working perfectly!"
            })
            
        except Exception as gemini_error:
            print(f"[DEBUG] ❌ Gemini error: {gemini_error}\n")
            debug_log.append(f"Gemini error: {gemini_error}")
            import traceback
            traceback.print_exc()
            
            return JSONResponse(status_code=500, content={
                "success": False,
                "stage": "gemini_analysis",
                "error": str(gemini_error),
                "debug_log": debug_log
            })
    
    except Exception as e:
        print(f"[DEBUG] ❌ FATAL ERROR: {e}\n")
        debug_log.append(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(status_code=500, content={
            "success": False,
            "stage": "unknown",
            "error": str(e),
            "debug_log": debug_log
        })


@app.get("/barcode/test-local-file")
async def test_with_sample_file():
    """
    Test extraction with a known image file
    """
    print(f"\n[DEBUG] Testing with sample image...\n")
    
    # Try to find a sample image
    import os
    
    sample_paths = [
        "uploads/sample.jpg",
        "test_image.jpg",
        "sample.jpg",
    ]
    
    sample_file = None
    for path in sample_paths:
        if os.path.exists(path):
            sample_file = path
            break
    
    if not sample_file:
        return JSONResponse(status_code=400, content={
            "success": False,
            "error": "No sample file found",
            "message": "Place a test image at: uploads/sample.jpg"
        })
    
    try:
        with open(sample_file, 'rb') as f:
            file_content = f.read()
        
        print(f"[DEBUG] Testing with: {sample_file} ({len(file_content):,} bytes)\n")
        
        # Run extraction
        result = analyze_barcode_gemini_vision(file_content, sample_file)
        
        print(f"[DEBUG] Result:")
        print(f"[DEBUG]   success: {result.get('success')}")
        print(f"[DEBUG]   barcodes found: {result.get('total_barcodes_found', 0)}")
        print(f"[DEBUG]   all_barcodes: {result.get('all_barcodes')}\n")
        
        return JSONResponse(content={
            "success": result.get('success'),
            "filename": sample_file,
            "total_barcodes_found": result.get('total_barcodes_found', 0),
            "all_barcodes": result.get('all_barcodes', []),
            "result": result
        })
        
    except Exception as e:
        print(f"[DEBUG] Error: {e}\n")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.post("/barcode/check-gemini-response")
async def check_gemini_response(
    prompt: str = Form(...),
    file_url: str = Form(...)
):
    """
    Test Gemini Vision with a custom prompt
    See exactly what Gemini returns
    """
    print(f"\n[DEBUG] Testing Gemini Response...\n")
    
    try:
        # Download file
        file_content, filename = download_file_from_url(file_url)
        
        from ai_analyzer import validate_and_optimize_image_single
        optimized_content, mime_type = validate_and_optimize_image_single(file_content, filename)
        
        from google.genai import types
        
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=optimized_content,
                    mime_type=mime_type
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        response_text = response.text
        
        print(f"[DEBUG] Raw Gemini Response:")
        print(f"[DEBUG] {response_text}\n")
        
        # Try to parse as JSON
        try:
            import json
            clean_text = response_text.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(clean_text)
            
            return JSONResponse(content={
                "success": True,
                "raw_response": response_text,
                "parsed_json": parsed,
                "type": str(type(parsed)),
                "keys": list(parsed.keys()) if isinstance(parsed, dict) else "N/A"
            })
        except json.JSONDecodeError as json_error:
            return JSONResponse(content={
                "success": False,
                "error": f"Failed to parse JSON: {json_error}",
                "raw_response": response_text
            })
        
    except Exception as e:
        print(f"[DEBUG] Error: {e}\n")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.get("/barcode/list-recent-results/{limit}")
async def list_recent_results(limit: int = 10):
    """
    List recent barcode extraction results
    Shows what's actually being stored
    """
    try:
        response = supabase.table("barcode_extraction_results")\
            .select("*")\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        
        print(f"\n[DEBUG] Found {len(response.data)} recent results\n")
        
        for result in response.data[:5]:
            print(f"[DEBUG] ID: {result.get('id')}")
            print(f"[DEBUG]   Job: {result.get('job_id')}")
            print(f"[DEBUG]   File: {result.get('filename')}")
            print(f"[DEBUG]   Barcode: {result.get('barcode_data')}")
            print(f"[DEBUG]   Status: {result.get('status')}")
            print(f"[DEBUG]   Created: {result.get('created_at')}\n")
        
        return JSONResponse(content={
            "success": True,
            "total_results": len(response.data),
            "results": response.data
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })

@app.post("/barcode/workdrive/preview")
async def preview_workdrive_extraction(
    folder_id: str = Form(...),
    selected_file_ids: Optional[str] = Form(None)
):
    """Preview files to be processed"""
    try:
        files = fetch_workdrive_files(folder_id)
        
        if selected_file_ids:
            selected_ids = json.loads(selected_file_ids)
            files = [f for f in files if f["file_id"] in selected_ids]
        
        return JSONResponse(content={
            "success": True,
            "total_files": len(files),
            "estimated_cost": f"${len(files) * 0.002:.4f}",
            "estimated_time_minutes": math.ceil(len(files) * 2 / 60),
            "files": [
                {
                    "file_id": f["file_id"],
                    "filename": f["filename"],
                    "size": f["size"]
                }
                for f in files
            ]
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


@app.post("/barcode/workdrive/start")
async def start_workdrive_barcode_extraction(
    folder_id: str = Form(...),
    selected_file_ids: str = Form(...)
):
    """Start barcode extraction job from Workdrive"""
    if not supabase:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "Supabase not configured"
        })
    
    try:
        selected_ids = json.loads(selected_file_ids)
        
        if not selected_ids:
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": "No files selected"
            })
        
        job_id = f"barcode_job_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Create job in Supabase
        supabase.table("barcode_extraction_jobs").insert({
            "job_id": job_id,
            "folder_id": folder_id,
            "total_files": len(selected_ids),
            "status": "pending"
        }).execute()
        
        # Start processing in background
        config = {
            "folder_id": folder_id,
            "selected_file_ids": selected_ids
        }
        
        thread = threading.Thread(
            target=process_barcode_extraction_job,
            args=(job_id, config)
        )
        thread.daemon = True
        thread.start()
        
        return JSONResponse(content={
            "success": True,
            "job_id": job_id,
            "status": "started",
            "message": f"🚀 Processing {len(selected_ids)} files",
            "check_status_url": f"/barcode/workdrive/status/{job_id}"
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })

# ============================================================


@app.post("/zoho/sync-records")
async def sync_records_to_zoho(request: ZohoSyncRequest):
    """
    Smart sync endpoint that INSERT/UPDATE and tracks sync status
    
    ✅ NEW: Updates Supabase with push_status = "synced"
    This ensures records won't be fetched again in preview
    """
    try:
        if not supabase:
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": "Supabase not configured"
            })
        
        print(f"\n{'='*80}")
        print(f"SMART SYNC TO ZOHO CREATOR")
        print(f"{'='*80}")
        print(f"Sync Mode: {request.sync_mode.upper()}")
        print(f"Total Records: {len(request.record_ids)}")
        if request.tag:
            print(f"Tag: {request.tag} ({request.tag_color})")
        print(f"{'='*80}\n")
        
        # Convert record_ids to strings
        record_ids_str = [str(rid) for rid in request.record_ids]
        
        # Fetch full records from Supabase
        records_to_sync = []
        supabase_ids = []  # Track Supabase record IDs for sync status update
        
        for record_id in record_ids_str:
            try:
                response = supabase.table('auto_extraction_results')\
                    .select('*')\
                    .eq('id', record_id)\
                    .execute()
                
                if response.data and len(response.data) > 0:
                    record = response.data[0]
                    records_to_sync.append(record)
                    supabase_ids.append(record.get('id'))
                else:
                    print(f"⚠️ Record {record_id} not found")
            except Exception as fetch_error:
                print(f"❌ Error fetching record {record_id}: {fetch_error}")
        
        if not records_to_sync:
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": "No valid records found"
            })
        
        print(f"✅ Fetched {len(records_to_sync)} records from Supabase\n")
        
        # Get Zoho access token
        access_token = os.getenv("ZOHO_ACCESS_TOKEN")
        if not access_token:
            access_token = get_zoho_access_token()
        
        if not access_token:
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": "Failed to get Zoho access token"
            })
        
        # Build Zoho form URL
        form_url = f"https://creator.zoho.com/api/v2/{request.config.owner_name}/{request.config.app_name}/form/{request.config.form_name}"
        
        headers = {
            'Authorization': f'Zoho-oauthtoken {access_token}',
            'Content-Type': 'application/json'
        }
        
        results = {
            "total_records": len(records_to_sync),
            "successful": 0,
            "inserted": 0,
            "updated": 0,
            "failed": 0,
            "errors": [],
            "sync_details": []
        }
        
        # Track successfully synced IDs
        successfully_synced_ids = []
        
        # Process each record
        for idx, record in enumerate(records_to_sync, 1):
            record_id = record.get('id')
            zoho_record_id = record.get('record_id')
            scholar_id = record.get('scholar_id', record.get('Scholar_ID', 'unknown'))
            
            print(f"[{idx}/{len(records_to_sync)}] Processing {scholar_id}...", end=' ')
            
            try:
                # Format record for Zoho
                zoho_record = format_record_for_zoho_creator(record)
                
                # Add tag if provided
                if request.tag:
                    zoho_record['tag'] = request.tag
                    zoho_record['tag_color'] = request.tag_color
                
                # Determine sync action
                sync_action = request.sync_mode
                existing_id = None
                
                if request.sync_mode == 'auto':
                    # Check if record exists in Zoho by Scholar_ID
                    scholar_id_field = zoho_record.get('Scholar_ID')
                    
                    if scholar_id_field:
                        check_url = f"https://creator.zoho.com/api/v2/{request.config.owner_name}/{request.config.app_name}/report/All_{request.config.form_name}"
                        
                        check_response = requests.get(
                            check_url,
                            headers=headers,
                            params={
                                'criteria': f'Scholar_ID == "{scholar_id_field}"',
                                'max_records': 1
                            },
                            timeout=10
                        )
                        
                        if check_response.status_code == 200:
                            data = check_response.json()
                            if data.get('data') and len(data['data']) > 0:
                                existing_record = data['data'][0]
                                sync_action = 'update'
                                existing_id = existing_record.get('ID')
                            else:
                                sync_action = 'insert'
                        else:
                            sync_action = 'insert'
                
                # Perform the sync action
                if sync_action == 'insert':
                    # POST for new record
                    payload = {"data": zoho_record}
                    
                    response = requests.post(
                        form_url,
                        json=payload,
                        headers=headers,
                        timeout=30
                    )
                    
                    response.raise_for_status()
                    zoho_response = response.json()
                    
                    results['successful'] += 1
                    results['inserted'] += 1
                    successfully_synced_ids.append(record_id)
                    print("✅ INSERTED")
                    
                    results['sync_details'].append({
                        "record_id": record_id,
                        "zoho_record_id": zoho_record_id,
                        "scholar_id": scholar_id,
                        "action": "INSERT",
                        "status": "success"
                    })
                
                elif sync_action == 'update':
                    if not existing_id and request.sync_mode != 'update':
                        results['failed'] += 1
                        print("⚠️ NOT FOUND (update mode)")
                        results['sync_details'].append({
                            "record_id": record_id,
                            "scholar_id": scholar_id,
                            "action": "UPDATE",
                            "status": "not_found"
                        })
                    else:
                        if existing_id:
                            update_url = f"{form_url}/{existing_id}"
                            payload = {"data": zoho_record}
                            
                            response = requests.put(
                                update_url,
                                json=payload,
                                headers=headers,
                                timeout=30
                            )
                            
                            response.raise_for_status()
                            zoho_response = response.json()
                            
                            results['successful'] += 1
                            results['updated'] += 1
                            successfully_synced_ids.append(record_id)
                            print("🔄 UPDATED")
                            
                            results['sync_details'].append({
                                "record_id": record_id,
                                "zoho_record_id": zoho_record_id,
                                "scholar_id": scholar_id,
                                "action": "UPDATE",
                                "zoho_id": existing_id,
                                "status": "success"
                            })
                        else:
                            results['failed'] += 1
                            print("❌ NOT FOUND")
                            results['sync_details'].append({
                                "record_id": record_id,
                                "scholar_id": scholar_id,
                                "action": "UPDATE",
                                "status": "not_found"
                            })
                
            except Exception as record_error:
                results['failed'] += 1
                error_msg = str(record_error)
                print(f"❌ {error_msg[:50]}")
                
                results['errors'].append({
                    "record_id": record_id,
                    "scholar_id": scholar_id,
                    "error": error_msg
                })
                
                results['sync_details'].append({
                    "record_id": record_id,
                    "scholar_id": scholar_id,
                    "status": "failed",
                    "error": error_msg
                })
            
            time.sleep(0.5)  # Rate limiting
        
        # ============================================================
        # ✅ NEW: Update Supabase records with sync status
        # This marks them as synced so they're excluded in future previews
        # ============================================================
        print(f"\n[SYNC] Updating Supabase sync status for {len(successfully_synced_ids)} records...")
        
        for supabase_id in successfully_synced_ids:
            try:
                update_data = {
                    "push_status": "synced",
                    "synced_at": datetime.now().isoformat(),
                    "sync_mode": request.sync_mode
                }
                
                if request.tag:
                    update_data["tag"] = request.tag
                    update_data["tag_color"] = request.tag_color
                
                supabase.table('auto_extraction_results')\
                    .update(update_data)\
                    .eq('id', supabase_id)\
                    .execute()
                    
            except Exception as update_error:
                print(f"⚠️ Failed to update sync status for {supabase_id}: {update_error}")
        
        print(f"[SYNC] ✅ Updated {len(successfully_synced_ids)} records in Supabase")
        
        # Final summary
        print(f"\n{'='*80}")
        print(f"SYNC COMPLETE")
        print(f"{'='*80}")
        print(f"✅ Total Successful: {results['successful']}/{len(records_to_sync)}")
        print(f"  ➕ Inserted: {results['inserted']}")
        print(f"  🔄 Updated: {results['updated']}")
        print(f"❌ Failed: {results['failed']}/{len(records_to_sync)}")
        print(f"📊 Marked in Supabase as synced: {len(successfully_synced_ids)}")
        
        if results['errors']:
            print(f"\nErrors:")
            for error in results['errors'][:3]:
                print(f"  - {error['scholar_id']}: {error['error'][:60]}")
        
        print(f"{'='*80}\n")
        
        return JSONResponse(content={
            "success": results['successful'] > 0,
            "message": f"✅ Synced {results['successful']}/{len(records_to_sync)} records to Creator",
            "details": results
        })
        
    except Exception as e:
        print(f"❌ Sync error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })

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
    """
    Cross-report OCR - accepts bank URL + bill download URLs
    Returns IMMEDIATELY, processes in background
    """
    try:
        print(f"\n{'='*80}")
        print(f"[CROSS-REPORT] New request: {scholar_name} ({scholar_id})")
        print(f"[CROSS-REPORT] Record ID: {source_record_id}")
        print(f"{'='*80}")

        # Parse bill URLs from JSON string
        bill_url_list = []
        if bill_urls:
            try:
                bill_url_list = json.loads(bill_urls)
                print(f"[CROSS-REPORT] Bill URLs: {len(bill_url_list)}")
            except Exception as e:
                print(f"[CROSS-REPORT] ⚠️ Failed to parse bill_urls: {e}")

        if not bank_passbook_url and not bill_url_list:
            return JSONResponse(status_code=400, content={
                "success": False,
                "error": "No files provided"
            })

        # Build config for background job
        config = {
            "source_record_id": source_record_id,
            "source_report_name": source_report_name,
            "app_link_name": app_link_name,
            "scholar_id": scholar_id,
            "scholar_name": scholar_name,
            "tracking_id": tracking_id, 
            "dest_form_name": dest_form_name,
            "bank_passbook_url": bank_passbook_url,
            "bill_url_list": bill_url_list
        }

        # ✅ Start background thread - return immediately
        thread = threading.Thread(
            target=process_cross_report_background,
            args=(config,)
        )
        thread.daemon = True
        thread.start()

        return JSONResponse(content={
            "success": True,
            "status": "processing",
            "message": f"OCR started for {scholar_name}",
            "record_id": source_record_id
        })

    except Exception as e:
        print(f"[CROSS-REPORT] ✗ Error: {e}")
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })


def process_cross_report_background(config: Dict):
    """Background processor for cross-report OCR"""
    scholar_name = config['scholar_name']
    scholar_id = config['scholar_id']

    print(f"\n[CROSS-REPORT BG] Starting: {scholar_name}")

    bank_data = None
    bill_data_list = []
    total_amount = 0.0

    # ✅ STEP 1: Process Bank
    if config.get('bank_passbook_url'):
        try:
            print(f"[CROSS-REPORT BG] Downloading bank passbook...")
            file_content, filename = download_file_from_url(config['bank_passbook_url'])
            result = process_single_file(file_content, filename, "bank")
            if result.get('success'):
                bank_data = result
                print(f"[CROSS-REPORT BG] ✅ Bank OCR success")
            else:
                print(f"[CROSS-REPORT BG] ✗ Bank OCR failed: {result.get('error')}")
        except Exception as e:
            print(f"[CROSS-REPORT BG] ✗ Bank error: {e}")

    # ✅ STEP 2: Process Bills
    for idx, bill_url in enumerate(config.get('bill_url_list', []), 1):
        try:
            print(f"[CROSS-REPORT BG] Downloading bill {idx}...")
            file_content, filename = download_file_from_url(bill_url)
            result = process_single_file(file_content, filename, "bill")
            if result.get('success'):
                bill_data_list.append(result)
                amount = result.get('amount') or 0
                try:
                    total_amount += float(amount)
                except:
                    pass
                print(f"[CROSS-REPORT BG] ✅ Bill {idx} OCR success")
            else:
                print(f"[CROSS-REPORT BG] ✗ Bill {idx} failed: {result.get('error')}")
        except Exception as e:
            print(f"[CROSS-REPORT BG] ✗ Bill {idx} error: {e}")

    # ✅ STEP 3: Save to destination form
    try:
        access_token, _ = get_zoho_token(scope_needed="create")
        if not access_token:
            print(f"[CROSS-REPORT BG] ✗ No create token available")
            return

        owner, app = config['app_link_name'].split('/', 1)
        form_url = f"https://creator.zoho.com/api/v2/{owner}/{app}/form/{config['dest_form_name']}"

        # Build bill data string
        bill_text = ""
        for i, bill in enumerate(bill_data_list, 1):
            bill_text += f"Bill {i}: Student: {bill.get('student_name')} | College: {bill.get('college_name')} | Amount: ₹{bill.get('amount', 0)}\n\n"

        data = {
            "Scholar_ID": scholar_id,
            "Scholar_Name": scholar_name,
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
            "Bill6_Amount": bill_data_list[5].get('amount', 0) if len(bill_data_list) > 5 else 0,
            "Bill7_Amount": bill_data_list[6].get('amount', 0) if len(bill_data_list) > 6 else 0,
            "Bill8_Amount": bill_data_list[7].get('amount', 0) if len(bill_data_list) > 7 else 0,
            "Amount": total_amount,
            "status": "OCR Completed"
        }

        response = requests.post(
            form_url,
            json={"data": data},
            headers={"Authorization": f"Zoho-oauthtoken {access_token}"},
            timeout=30
        )

        print(f"[CROSS-REPORT BG] ✅ Record saved: {response.status_code}")
        print(f"[CROSS-REPORT BG] Response: {response.text[:200]}")

    except Exception as e:
        print(f"[CROSS-REPORT BG] ✗ Save error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================
# ✅ HELPER: Format Record for Zoho with Tags
# ============================================================

def format_record_for_zoho_creator(record: Dict) -> Dict:
    """
    Format extraction result for Zoho Creator form
    Includes tag support for batch tracking
    """
    bill_data = record.get('bill_data', {})
    bank_data = record.get('bank_data', {})
    
    # Handle JSON strings
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
    
    # Convert bill_data to array
    bill_data_array = []
    if isinstance(bill_data, list):
        bill_data_array = bill_data
    elif isinstance(bill_data, dict) and bill_data:
        bill_data_array = [bill_data]
    
    # Safe float conversion
    def safe_float(value, default=0.0):
        if value is None or value == "null" or value == "":
            return default
        try:
            return float(value)
        except:
            return default
    
    # Extract amounts
    bill1_amount = safe_float(bill_data_array[0].get('amount')) if len(bill_data_array) > 0 else 0.0
    bill2_amount = safe_float(bill_data_array[1].get('amount')) if len(bill_data_array) > 1 else 0.0
    bill3_amount = safe_float(bill_data_array[2].get('amount')) if len(bill_data_array) > 2 else 0.0
    bill4_amount = safe_float(bill_data_array[3].get('amount')) if len(bill_data_array) > 3 else 0.0
    bill5_amount = safe_float(bill_data_array[4].get('amount')) if len(bill_data_array) > 4 else 0.0
    bill6_amount = safe_float(bill_data_array[5].get('amount')) if len(bill_data_array) > 5 else 0.0
    bill7_amount = safe_float(bill_data_array[6].get('amount')) if len(bill_data_array) > 6 else 0.0
    bill8_amount = safe_float(bill_data_array[7].get('amount')) if len(bill_data_array) > 7 else 0.0
    
    total_amount = bill1_amount + bill2_amount + bill3_amount + bill4_amount + bill5_amount + bill6_amount + bill7_amount + bill8_amount
    
    # Get scholar info
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
    
    # Format bill data with warnings
    def format_bill_text(bills):
        if not bills:
            return ""
        
        formatted_parts = []
        for idx, bill in enumerate(bills):
            amount = bill.get('amount')
            
            if amount is None or amount == 'null':
                amount_str = "⚠️ Not extracted"
            else:
                amount_str = f"₹{safe_float(amount):,.2f}"
            
            prefix = f"Bill {idx + 1}: " if len(bills) > 1 else ""
            part = (
                f"{prefix}"
                f"Student: {bill.get('student_name', 'N/A')} | "
                f"College: {bill.get('college_name', 'N/A')} | "
                f"Receipt: {bill.get('receipt_number', 'N/A')} | "
                f"Amount: {amount_str}"
            )
            formatted_parts.append(part)
        
        return " || ".join(formatted_parts)
    
    # Check for null amounts
    has_null_amounts = any(
        bill.get('amount') is None or bill.get('amount') == 'null'
        for bill in bill_data_array
    ) if bill_data_array else False
    
    status = record.get('status', 'completed')
    if has_null_amounts:
        status = "⚠️ Review - Missing amounts"
    
    # Build Zoho record
    return {
        "Scholar_Name": scholar_name,
        "Scholar_ID": scholar_id,
        "Tracking_ID": record.get('Tracking_id') or record.get('tracking_id', ''),
        "Account_No": bank_data.get('account_number', ''),
        "Bank_Name": bank_data.get('bank_name', ''),
        "Holder_Name": bank_data.get('account_holder_name', ''),
        "IFSC_Code": bank_data.get('ifsc_code', ''),
        "Branch_Name": bank_data.get('branch_name', ''),
        "Bill_Data": format_bill_text(bill_data_array),
        "Bill1_Amount": bill1_amount,
        "Bill2_Amount": bill2_amount,
        "Bill3_Amount": bill3_amount,
        "Bill4_Amount": bill4_amount,
        "Bill5_Amount": bill5_amount,
        "Bill6_Amount": bill6_amount,
        "Bill7_Amount": bill7_amount,
        "Bill8_Amount": bill8_amount,
        "Total_Amount": total_amount,
        "Status": status,
        "Tokens_Used": record.get('tokens_used', 0),
        "Cost_USD": float(record.get('cost_usd', 0)),
        # Tags will be added in the endpoint
    }
# Replace the barcode processing functions in your main.py

def process_barcode_extraction_job(job_id: str, config: Dict):
    """
    Background job processor for barcode extraction
    ✅ Handles ALL barcodes from multi-page PDFs
    """
    if not supabase:
        print("[BARCODE] ✗ Supabase not configured")
        return
    
    try:
        print(f"\n{'='*80}")
        print(f"[BARCODE] Starting Job: {job_id}")
        print(f"{'='*80}\n")
        
        # Update job status
        supabase.table("barcode_extraction_jobs").update({
            "status": "running",
            "started_at": datetime.now().isoformat()
        }).eq("job_id", job_id).execute()
        
        selected_ids = config['selected_file_ids']
        total_cost = 0.0
        processed = 0
        successful = 0
        failed = 0
        total_barcodes_extracted = 0
        
        for idx, file_id in enumerate(selected_ids, 1):
            record_start = time.time()
            
            print(f"\n[BARCODE] [{idx}/{len(selected_ids)}] Processing file: {file_id}")
            
            try:
                # Download file
                file_content, filename = download_workdrive_file(file_id)
                
                print(f"[BARCODE]   Downloaded: {filename} ({len(file_content):,} bytes)")
                
                # ✅ NEW: Extract ALL barcodes (handles multi-page PDFs)
                result = analyze_barcode_gemini_vision(file_content, filename)
                
                if result.get('success'):
                    # Upload file to Supabase storage
                    supabase_url = None
                    if supabase:
                        try:
                            supabase_url = upload_to_supabase_storage(
                                file_content,
                                f"barcode_{file_id}_{filename}",
                                folder="barcode-extractions"
                            )
                        except Exception as upload_error:
                            print(f"[BARCODE]   ⚠️ Supabase upload failed: {upload_error}")
                    
                    # Calculate cost
                    tokens = result.get('token_usage', {})
                    cost = calculate_cost(
                        tokens.get('input_tokens', 0),
                        tokens.get('output_tokens', 0),
                        'gemini_vision'
                    )
                    total_cost += cost
                    
                    # ✅ Extract ALL barcodes
                    all_barcodes = result.get('all_barcodes', [])
                    total_found = len(all_barcodes)
                    is_multipage = result.get('is_multipage', False)
                    pages_processed = result.get('total_pages_processed', 1)
                    
                    total_barcodes_extracted += total_found
                    
                    # ✅ Get primary barcode (first one) for compatibility
                    primary_barcode = result.get('barcode_data')
                    primary_type = result.get('barcode_type')
                    
                    # ✅ NEW: Save ALL barcodes individually to database
                    # This allows searching/filtering by individual barcode
                    if total_found > 0:
                        print(f"[BARCODE]   Saving {total_found} barcode(s) to database...")
                        
                        for barcode_idx, barcode in enumerate(all_barcodes, 1):
                            try:
                                supabase.table("barcode_extraction_results").insert({
                                    "job_id": job_id,
                                    "file_id": file_id,
                                    "filename": filename,
                                    "barcode_number": barcode_idx,  # Position in file
                                    "total_barcodes_in_file": total_found,
                                    "barcode_data": barcode.get('data'),
                                    "barcode_type": barcode.get('type'),
                                    "page_number": barcode.get('page', 1),  # For PDFs
                                    "is_primary": barcode_idx == 1,
                                    "image_url": supabase_url,
                                    "status": "success",
                                    "tokens_used": tokens.get('total_tokens', 0),
                                    "cost_usd": cost / total_found if total_found > 0 else 0,
                                    "processing_time_ms": int((time.time() - record_start) * 1000),
                                    "is_multipage": is_multipage,
                                    "pages_processed": pages_processed
                                }).execute()
                            except Exception as save_error:
                                print(f"[BARCODE]   ⚠️ Failed to save barcode {barcode_idx}: {save_error}")
                        
                        successful += 1
                        
                        # Enhanced logging
                        print(f"[BARCODE]   ✅ Extracted {total_found} barcode(s):")
                        
                        if is_multipage:
                            print(f"[BARCODE]      📄 Multi-page PDF: {pages_processed} pages")
                        
                        # Show first 5
                        for i, bc in enumerate(all_barcodes[:5]):
                            page_info = f" (page {bc.get('page')})" if bc.get('page') else ""
                            print(f"[BARCODE]      [{i+1}] {bc.get('type')}: {bc.get('data')}{page_info}")
                        
                        if total_found > 5:
                            print(f"[BARCODE]      ... and {total_found - 5} more")
                    else:
                        # No barcodes found but extraction succeeded
                        failed += 1
                        supabase.table("barcode_extraction_results").insert({
                            "job_id": job_id,
                            "file_id": file_id,
                            "filename": filename,
                            "status": "no_barcode",
                            "error_message": "No barcodes detected in image",
                            "processing_time_ms": int((time.time() - record_start) * 1000)
                        }).execute()
                        print(f"[BARCODE]   ⚠️ No barcodes found")
                else:
                    # Extraction failed
                    failed += 1
                    supabase.table("barcode_extraction_results").insert({
                        "job_id": job_id,
                        "file_id": file_id,
                        "filename": filename,
                        "status": "failed",
                        "error_message": result.get('error'),
                        "processing_time_ms": int((time.time() - record_start) * 1000)
                    }).execute()
                    print(f"[BARCODE]   ✗ Failed: {result.get('error')}")
                
            except Exception as file_error:
                failed += 1
                print(f"[BARCODE]   ✗ Error: {file_error}")
                try:
                    supabase.table("barcode_extraction_results").insert({
                        "job_id": job_id,
                        "file_id": file_id,
                        "status": "failed",
                        "error_message": str(file_error)
                    }).execute()
                except:
                    pass
            
            processed += 1
            
            # Update progress
            try:
                supabase.table("barcode_extraction_jobs").update({
                    "processed_files": processed,
                    "successful_files": successful,
                    "failed_files": failed,
                    "total_cost_usd": round(total_cost, 6),
                    "total_barcodes_extracted": total_barcodes_extracted  # ✅ NEW
                }).eq("job_id", job_id).execute()
            except:
                pass
            
            time.sleep(0.5)
        
        # Job completed
        supabase.table("barcode_extraction_jobs").update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "total_cost_usd": round(total_cost, 6),
            "total_barcodes_extracted": total_barcodes_extracted  # ✅ NEW
        }).eq("job_id", job_id).execute()
        
        print(f"\n{'='*80}")
        print(f"[BARCODE] ✅ Job Completed: {job_id}")
        print(f"{'='*80}")
        print(f"[BARCODE]   Files Processed: {successful}/{len(selected_ids)}")
        print(f"[BARCODE]   Total Barcodes Extracted: {total_barcodes_extracted} ✅")
        print(f"[BARCODE]   Failed: {failed}/{len(selected_ids)}")
        print(f"[BARCODE]   Total Cost: ${total_cost:.6f}")
        print(f"{'='*80}\n")
        
    except Exception as e:
        print(f"[BARCODE] ✗ Job failed: {e}")
        import traceback
        traceback.print_exc()
        try:
            supabase.table("barcode_extraction_jobs").update({
                "status": "failed",
                "completed_at": datetime.now().isoformat()
            }).eq("job_id", job_id).execute()
        except:
            pass

@app.get("/barcode/workdrive/status/{job_id}")
async def get_barcode_job_status(job_id: str):
    """Get barcode extraction job status"""
    if not supabase:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "Supabase not configured"
        })
    
    try:
        response = supabase.table("barcode_extraction_jobs")\
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
        if job.get("total_files", 0) > 0:
            progress_percent = round(
                (job.get("processed_files", 0) / job["total_files"]) * 100, 2
            )
        
        return JSONResponse(content={
            "success": True,
            "job_id": job_id,
            "status": job["status"],
            "progress": {
                "total_files": job.get("total_files", 0),
                "processed_files": job.get("processed_files", 0),
                "successful_files": job.get("successful_files", 0),
                "failed_files": job.get("failed_files", 0),
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


@app.get("/barcode/workdrive/results/{job_id}")
async def get_barcode_job_results(job_id: str, limit: int = 100):
    """Get barcode extraction results"""
    if not supabase:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "Supabase not configured"
        })
    
    try:
        response = supabase.table("barcode_extraction_results")\
            .select("*")\
            .eq("job_id", job_id)\
            .order("created_at", desc=True)\
            .limit(limit)\
            .execute()
        
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
# BANK FOLDER EXTRACTION
# Reuses auto_extraction_jobs + auto_extraction_results tables
# ============================================================

SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.pdf', '.bmp'}

def scan_folder_for_images(folder_path: str, recursive: bool = False) -> List[Dict]:
    """Scan a local folder for supported image/PDF files."""
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Path is not a folder: {folder_path}")

    files = []
    walk = os.walk(folder_path) if recursive else [(folder_path, [], os.listdir(folder_path))]

    for root, dirs, filenames in walk:
        for fname in filenames:
            if fname.startswith('.'):          # skip hidden files
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                full_path = os.path.join(root, fname)
                if os.path.isfile(full_path):
                    files.append({
                        "file_id":  f"local_{uuid.uuid4().hex}",
                        "filename": fname,
                        "filepath": full_path,
                        "size":     os.path.getsize(full_path),
                        "rel_path": os.path.relpath(full_path, folder_path)
                    })

    files.sort(key=lambda x: x["rel_path"])
    return files


# ── PREVIEW ────────────────────────────────────────────────

@app.post("/bank/folder/preview")
async def preview_bank_folder(
    folder_path: str = Form(...),
    recursive:   str = Form("false")
):
    """Preview files that will be processed - no OCR run yet."""
    try:
        is_recursive = recursive.lower() in ("true", "1", "yes")
        files = scan_folder_for_images(folder_path, recursive=is_recursive)

        return JSONResponse(content={
            "success":       True,
            "folder_path":   folder_path,
            "recursive":     is_recursive,
            "total_files":   len(files),
            "estimated_cost": f"${len(files) * 0.002:.4f}",
            "estimated_time_minutes": max(1, math.ceil(len(files) * 3 / 60)),
            "files": [
                {"filename": f["filename"], "rel_path": f["rel_path"],
                 "size_kb": round(f["size"] / 1024, 1)}
                for f in files
            ]
        })

    except (FileNotFoundError, NotADirectoryError) as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ── START ──────────────────────────────────────────────────

@app.post("/bank/folder/start")
async def start_bank_folder_extraction(
    folder_path: str = Form(...),
    recursive:   str = Form("false")
):
    """
    Start bank OCR for every supported file in a folder.
    Uses auto_extraction_jobs + auto_extraction_results (existing tables).
    """
    if not supabase:
        return JSONResponse(status_code=500, content={
            "success": False, "error": "Supabase not configured"
        })

    try:
        is_recursive = recursive.lower() in ("true", "1", "yes")
        files = scan_folder_for_images(folder_path, recursive=is_recursive)

        if not files:
            return JSONResponse(status_code=400, content={
                "success": False,
                "error":   f"No supported files found in: {folder_path}",
                "supported_formats": sorted(SUPPORTED_EXTENSIONS)
            })

        job_id = f"bank_folder_{int(time.time())}_{uuid.uuid4().hex[:8]}"

        # ✅ Reuse auto_extraction_jobs table
        supabase.table("auto_extraction_jobs").insert({
            "job_id":          job_id,
            "app_link_name":   "local_folder",   # sentinel value
            "report_link_name": folder_path,     # store folder path here
            "bank_field_name": "local_file",
            "bill_field_name": None,
            "status":          "pending",
            "total_records":   len(files),
            "folder_path":     folder_path       # dedicated column already exists
        }).execute()

        thread = threading.Thread(
            target=process_bank_folder_job,
            args=(job_id, files, folder_path)
        )
        thread.daemon = True
        thread.start()

        return JSONResponse(content={
            "success":          True,
            "job_id":           job_id,
            "status":           "started",
            "folder_path":      folder_path,
            "total_files":      len(files),
            "message":          f"🚀 Processing {len(files)} file(s) from folder",
            "check_status_url": f"/bank/folder/status/{job_id}",
            "results_url":      f"/bank/folder/results/{job_id}"
        })

    except (FileNotFoundError, NotADirectoryError) as e:
        return JSONResponse(status_code=400, content={"success": False, "error": str(e)})
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ── BACKGROUND PROCESSOR ───────────────────────────────────

def process_bank_folder_job(job_id: str, files: List[Dict], folder_path: str):
    """
    Processes every file and saves to auto_extraction_results.
    Maps local-file fields onto the existing schema:
      record_id   → file_id  (unique per file)
      student_name → filename
      bank_data   → full OCR result dict
      status      → success / failed
    """
    if not supabase:
        return

    print(f"\n{'='*80}")
    print(f"[BANK FOLDER] Job: {job_id}  |  Files: {len(files)}")
    print(f"{'='*80}\n")

    try:
        supabase.table("auto_extraction_jobs").update({
            "status":     "running",
            "started_at": datetime.now().isoformat()
        }).eq("job_id", job_id).execute()

        processed = successful = failed = 0
        total_cost = 0.0

        for idx, file_info in enumerate(files, 1):
            file_id   = file_info["file_id"]
            filename  = file_info["filename"]
            filepath  = file_info["filepath"]
            rel_path  = file_info["rel_path"]
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
                cost   = calculate_cost(
                    tokens.get("input_tokens", 0),
                    tokens.get("output_tokens", 0),
                    "gemini_vision"
                )
                total_cost    += cost
                processing_ms  = int((time.time() - t0) * 1000)

                # ✅ Save into auto_extraction_results
                supabase.table("auto_extraction_results").insert({
                    "job_id":             job_id,
                    "record_id":          file_id,        # file_id as record_id
                    "app_link_name":      "local_folder",
                    "report_link_name":   folder_path,
                    "student_name":       filename,       # filename as student_name
                    "bank_data":          result,         # full OCR result
                    "bill_data":          None,
                    "status":             "success",
                    "processing_time_ms": processing_ms,
                    "tokens_used":        tokens.get("total_tokens", 0),
                    "cost_usd":           cost,
                    "processed_at":       datetime.now().isoformat(),
                    "rel_path":           rel_path        # column already exists
                }).execute()

                successful += 1
                print(f"[BANK FOLDER]   ✅ {result.get('bank_name')} | "
                      f"A/C: {result.get('account_number')} | "
                      f"IFSC: {result.get('ifsc_code')} | ${cost:.6f}")

            except Exception as err:
                failed += 1
                print(f"[BANK FOLDER]   ✗ {err}")
                try:
                    supabase.table("auto_extraction_results").insert({
                        "job_id":          job_id,
                        "record_id":       file_id,
                        "app_link_name":   "local_folder",
                        "report_link_name": folder_path,
                        "student_name":    filename,
                        "status":          "failed",
                        "error_message":   str(err),
                        "processed_at":    datetime.now().isoformat(),
                        "rel_path":        rel_path
                    }).execute()
                except Exception:
                    pass

            processed += 1

            # Update progress
            try:
                supabase.table("auto_extraction_jobs").update({
                    "processed_records":  processed,
                    "successful_records": successful,
                    "failed_records":     failed,
                    "total_cost_usd":     round(total_cost, 6)
                }).eq("job_id", job_id).execute()
            except Exception:
                pass

            time.sleep(0.3)

        supabase.table("auto_extraction_jobs").update({
            "status":        "completed",
            "completed_at":  datetime.now().isoformat(),
            "total_cost_usd": round(total_cost, 6)
        }).eq("job_id", job_id).execute()

        print(f"\n[BANK FOLDER] ✅ Done | "
              f"Success: {successful}/{len(files)} | "
              f"Failed: {failed}/{len(files)} | "
              f"Cost: ${total_cost:.6f}\n")

    except Exception as e:
        print(f"[BANK FOLDER] ✗ Job crashed: {e}")
        import traceback; traceback.print_exc()
        try:
            supabase.table("auto_extraction_jobs").update({
                "status":       "failed",
                "completed_at": datetime.now().isoformat()
            }).eq("job_id", job_id).execute()
        except Exception:
            pass


# ── STATUS ─────────────────────────────────────────────────

@app.get("/bank/folder/status/{job_id}")
async def get_bank_folder_status(job_id: str):
    """Poll job progress."""
    if not supabase:
        return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured"})

    try:
        resp = supabase.table("auto_extraction_jobs").select("*").eq("job_id", job_id).execute()
        if not resp.data:
            return JSONResponse(status_code=404, content={"success": False, "error": "Job not found"})

        job   = resp.data[0]
        total = job.get("total_records", 0)
        done  = job.get("processed_records", 0)
        pct   = round((done / total) * 100, 2) if total else 0

        return JSONResponse(content={
            "success":     True,
            "job_id":      job_id,
            "status":      job["status"],
            "folder_path": job.get("folder_path"),
            "progress": {
                "total_files":      total,
                "processed_files":  done,
                "successful_files": job.get("successful_records", 0),
                "failed_files":     job.get("failed_records", 0),
                "progress_percent": pct
            },
            "cost": {"total_cost_usd": float(job.get("total_cost_usd", 0))},
            "timestamps": {
                "created_at":   job.get("created_at"),
                "started_at":   job.get("started_at"),
                "completed_at": job.get("completed_at")
            }
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


# ── RESULTS ────────────────────────────────────────────────

@app.get("/bank/folder/results/{job_id}")
async def get_bank_folder_results(
    job_id:        str,
    limit:         int = 500,
    status_filter: Optional[str] = None   # "success" | "failed"
):
    """Fetch all extraction results for a folder job."""
    if not supabase:
        return JSONResponse(status_code=500, content={"success": False, "error": "Supabase not configured"})

    try:
        query = supabase.table("auto_extraction_results")\
            .select("id, record_id, student_name, rel_path, status, bank_data, error_message, cost_usd, processing_time_ms, processed_at")\
            .eq("job_id", job_id)\
            .order("processed_at", desc=False)\
            .limit(limit)

        if status_filter:
            query = query.eq("status", status_filter)

        resp = query.execute()

        # Flatten bank_data fields for easy reading
        flat_results = []
        for row in resp.data:
            bank = row.get("bank_data") or {}
            flat_results.append({
                "filename":       row.get("student_name"),
                "rel_path":       row.get("rel_path"),
                "status":         row.get("status"),
                "bank_name":      bank.get("bank_name"),
                "account_number": bank.get("account_number"),
                "account_holder": bank.get("account_holder_name"),
                "ifsc_code":      bank.get("ifsc_code"),
                "branch_name":    bank.get("branch_name"),
                "error_message":  row.get("error_message"),
                "cost_usd":       row.get("cost_usd"),
                "processing_time_ms": row.get("processing_time_ms")
            })

        return JSONResponse(content={
            "success":       True,
            "job_id":        job_id,
            "total_results": len(flat_results),
            "results":       flat_results
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})
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


# ============================================================
# ✅ UPDATED: Filter Already-Extracted Records by record_id
# Works without tracking_id - uses record_id as primary key
# ============================================================




def get_already_extracted_record_ids(
    app_link_name: str,
    report_link_name: str,
    include_failed: bool = False,
    include_synced: bool = False
) -> set:
    if not supabase:
        return set()

    try:
        print(f"\n[FILTER] Checking for already-extracted records...")
        print(f"[FILTER] App: {app_link_name} | Report: {report_link_name}")

        # ✅ FIX: Paginate Supabase — default limit is 1000 rows per request
        all_results = []
        PAGE_SIZE  = 1000
        from_index = 0

        while True:
            response = supabase.table("auto_extraction_results") \
                .select("id, record_id, status, push_status, scholar_id, tracking_id, created_at") \
                .eq("app_link_name", app_link_name) \
                .eq("report_link_name", report_link_name) \
                .range(from_index, from_index + PAGE_SIZE - 1) \
                .execute()

            batch = response.data or []
            all_results.extend(batch)

            print(f"[FILTER]   Page {from_index // PAGE_SIZE + 1}: "
                  f"fetched {len(batch)} rows (running total: {len(all_results)})")

            if len(batch) < PAGE_SIZE:
                break          # last page reached
            from_index += PAGE_SIZE

        if not all_results:
            print(f"[FILTER] ✅ No previous extractions found — all records are new")
            return set()

        print(f"[FILTER] Found {len(all_results)} total historical records")

        # ── count by status & push_status ──────────────────────────
        status_counts     = {}
        push_status_counts = {}
        for r in all_results:
            s  = r.get("status",      "unknown")
            ps = r.get("push_status", "not_synced")
            status_counts[s]      = status_counts.get(s, 0) + 1
            push_status_counts[ps] = push_status_counts.get(ps, 0) + 1

        print(f"[FILTER] Status breakdown:")
        for s, c in sorted(status_counts.items()):
            print(f"[FILTER]   Extraction Status - {s}: {c}")

        print(f"[FILTER] Sync Status breakdown:")
        for ps, c in sorted(push_status_counts.items()):
            print(f"[FILTER]   Sync Status - {ps}: {c}")

        # ── build exclusion set ─────────────────────────────────────
        already_extracted = set()

        # 1️⃣  Always exclude successful extractions
        successful_ids = {
            str(r["record_id"])
            for r in all_results
            if r.get("status") == "success" and r.get("record_id")
        }
        already_extracted.update(successful_ids)
        print(f"[FILTER] ✅ Excluding {len(successful_ids)} successful extractions")

        # 2️⃣  Always exclude synced records (unless caller opts in for re-sync)
        if not include_synced:
            synced_ids = {
                str(r["record_id"])
                for r in all_results
                if r.get("push_status") == "synced" and r.get("record_id")
            }
            already_extracted.update(synced_ids)
            print(f"[FILTER] ✅ Excluding {len(synced_ids)} already synced to Zoho Creator")
        else:
            print(f"[FILTER] ⚠️ Including {push_status_counts.get('synced', 0)} synced records for re-sync")

        # 3️⃣  Optionally exclude failed attempts
        if include_failed:
            failed_ids = {
                str(r["record_id"])
                for r in all_results
                if r.get("status") in ("failed", "error") and r.get("record_id")
            }
            already_extracted.update(failed_ids)
            print(f"[FILTER] ⚠️ Also excluding {len(failed_ids)} failed extraction attempts")
        else:
            failed_count = status_counts.get("failed", 0) + status_counts.get("error", 0)
            print(f"[FILTER] 💡 Including {failed_count} failed records for retry")

        # ── summary ────────────────────────────────────────────────
        available = len(all_results) - len(already_extracted)
        print(f"\n[FILTER] 📊 EXCLUSION SUMMARY")
        print(f"[FILTER] {'='*60}")
        print(f"[FILTER] Total extracted (Supabase):  {len(all_results)}")
        print(f"[FILTER] - Successful:                {len(successful_ids)}")
        print(f"[FILTER] - Synced to Creator:         {push_status_counts.get('synced', 0)}")
        print(f"[FILTER] - Failed:                    {status_counts.get('failed', 0)}")
        print(f"[FILTER] {'='*60}")
        print(f"[FILTER] Total to EXCLUDE:            {len(already_extracted)}")
        print(f"[FILTER] Available for processing:    {available}")
        print(f"[FILTER] {'='*60}\n")

        return already_extracted

    except Exception as e:
        print(f"[FILTER] ✗ Error: {e}")
        import traceback; traceback.print_exc()
        return set()

# ============================================================
# ✅ UPDATED: fetch_zoho_records_with_dedup
# Now uses the new filter function
# ============================================================

def fetch_zoho_records_with_dedup(
    app_link_name: str, 
    report_link_name: str,
    criteria: Optional[str] = None,
    max_records: int = None,
    exclude_already_extracted: bool = True,
    include_failed_retries: bool = False,
    include_synced_for_retry: bool = False
) -> tuple:
    """
    Fetch records from Zoho Creator with automatic deduplication
    ✅ Now excludes synced records by default
    ✅ Supports retry modes for failed and synced records
    
    Returns:
        (records, stats_dict) where stats includes counts and exclusions
    """
    try:
        access_token, token_name = get_zoho_token(scope_needed="read")
        
        if not access_token:
            raise Exception("Failed to get READ token")
        
        api_url = f"https://creator.zoho.com/api/v2.1/{app_link_name}/report/{report_link_name}"
        headers = {"Authorization": f"Zoho-oauthtoken {access_token}"}
        
        # Fetch all records (or filtered subset)
        all_records = []
        seen_ids = set()
        page = 1
        max_id = None
        
        print(f"\n{'='*80}")
        print(f"[ZOHO FETCH] Fetching from {report_link_name}")
        print(f"[ZOHO FETCH] Deduplication: {exclude_already_extracted}")
        print(f"[ZOHO FETCH] Retry Failed: {include_failed_retries}")
        print(f"[ZOHO FETCH] Retry Synced: {include_synced_for_retry}")
        print(f"{'='*80}\n")
        
        while True:
            if criteria:
                params = {"criteria": criteria, "from": 1, "limit": 200}
            else:
                if max_id is None:
                    params = {"from": 1, "limit": 200}
                else:
                    params = {"criteria": f"ID < {max_id}", "from": 1, "limit": 200}
            
            response = requests.get(api_url, headers=headers, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            records = data.get("data", [])
            
            if not records:
                break
            
            # Extract ID range for next page
            batch_ids = [int(r.get("ID", 0)) for r in records if r.get("ID")]
            if batch_ids:
                max_id = min(batch_ids)
            
            # Deduplicate
            for record in records:
                record_id = str(record.get("ID", ""))
                if record_id and record_id not in seen_ids:
                    seen_ids.add(record_id)
                    all_records.append(record)
            
            print(f"[ZOHO FETCH] Page {page}: Fetched {len(records)} records")
            
            if len(records) < 200 or (max_records and len(all_records) >= max_records):
                break
            
            page += 1
            time.sleep(0.5)
        
        print(f"[ZOHO FETCH] ✅ Fetched {len(all_records)} unique records from Zoho\n")
        
        # Filter out already extracted (if enabled)
        stats = {
            "total_fetched": len(all_records),
            "excluded_successful": 0,
            "excluded_synced": 0,
            "excluded_failed": 0,
            "total_excluded": 0,
            "new_count": len(all_records),
            "final_records": len(all_records),
            "excluded_ids": []
        }
        
        if exclude_already_extracted:
            already_extracted = get_already_extracted_record_ids(
                app_link_name,
                report_link_name,
                include_failed=include_failed_retries,
                include_synced=include_synced_for_retry
            )
            
            if already_extracted:
                original_count = len(all_records)
                all_records = [r for r in all_records if str(r.get("ID", "")) not in already_extracted]
                excluded_count = original_count - len(all_records)
                
                stats["total_excluded"] = excluded_count
                stats["new_count"] = len(all_records)
                stats["final_records"] = len(all_records)
                stats["excluded_ids"] = list(already_extracted)[:10]  # Sample
                
                print(f"[FILTER] Excluded {excluded_count} already-processed records")
                print(f"[FILTER] New records to process: {len(all_records)}\n")
        
        # Apply max_records limit
        if max_records and len(all_records) > max_records:
            all_records = all_records[:max_records]
        
        return all_records, stats
        
    except Exception as e:
        print(f"[ZOHO FETCH] ✗ Error: {e}")
        raise


# ============================================================
# ✅ UPDATED: Auto-Extract Preview 
# Now shows synced record exclusion
# ============================================================

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
    include_synced_for_retry: str = Form("false"),  # ✅ NEW
    max_records_limit: int = Form(1000),
    selected_fields: Optional[str] = Form(None)
):
    """
    Preview records for extraction - WITH FULL DEDUPLICATION
    
    ✅ NEW: include_synced_for_retry parameter to re-sync synced records
    ✅ Excludes: Successful + Synced + (Optional) Failed records
    """
    try:
        print(f"[PREVIEW] Loading records from {report_link_name}...")

        # Parse options
        exclude_extracted = exclude_already_extracted.lower() in ('true', '1', 'yes')
        include_failed = include_failed_retries.lower() in ('true', '1', 'yes')
        include_synced = include_synced_for_retry.lower() in ('true', '1', 'yes')

        # Parse filters
        zoho_criteria = None
        if filter_criteria:
            try:
                filters = json.loads(filter_criteria)
                zoho_criteria = convert_filters_to_zoho_criteria(filters)
                print(f"[PREVIEW] 🔍 Filter criteria: {zoho_criteria}")
            except Exception as e:
                print(f"[PREVIEW] ⚠️ Filter error: {e}")
                if isinstance(filter_criteria, str) and filter_criteria.strip():
                    zoho_criteria = filter_criteria
        
        # ✅ UPDATED: Fetch with deduplication including synced records
        records, fetch_stats = fetch_zoho_records_with_dedup(
            app_link_name=app_link_name,
            report_link_name=report_link_name,
            criteria=zoho_criteria,
            max_records=max_records_limit,
            exclude_already_extracted=exclude_extracted,
            include_failed_retries=include_failed,
            include_synced_for_retry=include_synced  # ✅ NEW
        )

        print(f"[PREVIEW] ✅ After filtering: {len(records)} records ready")

        # Parse selected fields
        fields_to_fetch = []
        if selected_fields:
            try:
                fields_to_fetch = json.loads(selected_fields)
                print(f"[PREVIEW] 🎯 User selected {len(fields_to_fetch)} fields")
            except Exception as e:
                print(f"[PREVIEW] ⚠️ Failed to parse fields: {e}")
        
        # Extract image URLs
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

        # Build preview
        preview_records = []
        for record in records:
            record_id = str(record.get("ID", ""))
            student_name = extract_name(record)
            
            bank_value = record.get(bank_field_name) if bank_field_name else None
            bill_value = record.get(bill_field_name) if bill_field_name else None
            
            preview_record = {
                "record_id": record_id,
                "student_name": student_name,
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
        
        # Response with detailed stats
        return JSONResponse(content={
            "success": True,
            "total_records": len(preview_records),
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
            "include_synced_for_retry": include_synced  # ✅ NEW
        })
        
    except Exception as e:
        print(f"[PREVIEW] ✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": str(e)
        })

# ============================================================
# HELPER: Updated start extraction to pass selected fields
# ============================================================

@app.post("/ocr/auto-extract/start")
async def start_extraction(
    app_link_name: str = Form(...),
    report_link_name: str = Form(...),
    bank_field_name: Optional[str] = Form(None),
    bill_field_name: Optional[str] = Form(None),
    filter_criteria: Optional[str] = Form(None),
    selected_record_ids: Optional[str] = Form(None),
    selected_fields: Optional[str] = Form(None)  # ✅ NEW: Pass selected fields
):
    """Start extraction job with selected fields"""
    if not supabase:
        return JSONResponse(status_code=500, content={
            "success": False,
            "error": "Supabase not configured. Auto-extract requires Supabase."
        })
    
    try:
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
            "selected_record_ids": selected_ids,
            "selected_fields": selected_fields  # ✅ NEW: Store selected fields in config
        }
        
        supabase.table("auto_extraction_jobs").insert({
            "job_id": job_id,
            "app_link_name": app_link_name,
            "report_link_name": report_link_name,
            "bank_field_name": bank_field_name,
            "bill_field_name": bill_field_name,
            "filter_criteria": zoho_criteria,
            "selected_fields": selected_fields,
            "status": "pending"  # ✅ NEW: Store in Supabase
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
                "Bill6_Amount",
                "Bill7_Amount",
                "Bill8_Amount",
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