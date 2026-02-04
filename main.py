# main.py
from fastapi import FastAPI, Depends, HTTPException, status, Request, Response, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import httpx
from datetime import datetime, timedelta
import secrets
import os
from io import BytesIO
import asyncio

# Import your existing libraries
from supabase import create_client, Client
import google.generativeai as genai
from PIL import Image

# Configuration
AUTHENTIK_URL = os.getenv("AUTHENTIK_URL", "https://authentik.teameverest.ngo")
AUTHENTIK_API_TOKEN = os.getenv("AUTHENTIK_API_TOKEN", "KPu2Ow7RVjIZFFZ5DdXSu1LqaI5oxFBPxsnORgQHykECbiCYoReoBE5vEx2U")  # For user lookup
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")
SECRET_KEY = os.getenv("SECRET_KEY", secrets.token_urlsafe(32))

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

# Google Gemini Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Initialize services
supabase: Client = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

app = FastAPI(title="TeamEverest API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session storage (use Redis in production)
sessions = {}

# Models
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

# Helper functions
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

async def get_current_user(request: Request) -> User:
    """Dependency to get current authenticated user"""
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

# Authentication Routes
@app.get("/")
async def root():
    return {
        "message": "TeamEverest API",
        "version": "2.0.0",
        "auth_type": "flow_based",
        "docs": "/docs"
    }

@app.post("/auth/login", response_model=LoginResponse)
async def login(login_data: LoginRequest, response: Response):
    try:
        username = login_data.username
        password = login_data.password

        # Validate input
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

        # Enable redirect following with history tracking
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            # Step 1: Initialize authentication flow
            flow_response = await client.post(
                f"{AUTHENTIK_URL}/api/v3/flows/executor/default-authentication-flow/",
                json={}
            )
            
            print(f"📥 Step 1 Status: {flow_response.status_code}")
            
            # Get cookies from response
            cookies = flow_response.cookies
            
            # Step 2: Submit username
            id_response = await client.post(
                f"{AUTHENTIK_URL}/api/v3/flows/executor/default-authentication-flow/",
                json={"uid_field": username},
                cookies=cookies
            )
            
            print(f"📥 Step 2 Final Status: {id_response.status_code}")
            
            # Update cookies
            if id_response.cookies:
                cookies.update(id_response.cookies)
            
            # Parse JSON response
            try:
                id_data = id_response.json()
                print(f"📥 Step 2 Component: {id_data.get('component')}")
            except Exception as e:
                print(f"❌ JSON parse error at Step 2: {e}")
                print(f"Response: {id_response.text[:200]}")
                return LoginResponse(
                    success=False,
                    message="Invalid username. Please check and try again."
                )
            
            # Check if username was accepted
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
            print(f"📥 Step 3 Response: {password_response.text[:500]}")
            
            # Parse password response
            try:
                password_data = password_response.json()
            except Exception as e:
                print(f"❌ JSON parse error at Step 3: {e}")
                print(f"Response text: {password_response.text[:200]}")
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
async def get_me(current_user: User = Depends(get_current_user)):
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

# Document Processing Routes
@app.post("/api/extract")
async def extract_document(
    file: UploadFile = File(...),
    prompt: str = "",
    current_user: User = Depends(get_current_user)
):
    """Extract data from uploaded document using Gemini AI"""
    try:
        contents = await file.read()
        image = Image.open(BytesIO(contents))
        
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if not prompt:
            prompt = "Extract all important information from this document including text, tables, and key data points. Structure the output as JSON."
        
        response = model.generate_content([prompt, image])
        
        input_tokens = len(prompt.split()) * 1.3
        output_tokens = len(response.text.split()) * 1.3
        cost = (input_tokens / 1000000 * 0.075) + (output_tokens / 1000000 * 0.30)
        
        if supabase:
            try:
                supabase.table("extractions").insert({
                    "user_id": current_user.id,
                    "user_email": current_user.email,
                    "username": current_user.username,
                    "filename": file.filename,
                    "prompt": prompt,
                    "result": response.text,
                    "cost": cost,
                    "timestamp": datetime.now().isoformat()
                }).execute()
            except Exception as e:
                print(f"Error storing in Supabase: {e}")
        
        return {
            "success": True,
            "data": {
                "filename": file.filename,
                "extracted_text": response.text,
                "user": current_user.username
            },
            "cost": round(cost, 6),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Extraction failed: {str(e)}"
        )

@app.post("/api/batch-extract")
async def batch_extract(
    files: List[UploadFile] = File(...),
    prompt: str = "",
    current_user: User = Depends(get_current_user)
):
    """Extract data from multiple documents"""
    results = []
    total_cost = 0.0
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(BytesIO(contents))
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            
            if not prompt:
                prompt = "Extract all important information from this document."
            
            response = model.generate_content([prompt, image])
            
            input_tokens = len(prompt.split()) * 1.3
            output_tokens = len(response.text.split()) * 1.3
            cost = (input_tokens / 1000000 * 0.075) + (output_tokens / 1000000 * 0.30)
            total_cost += cost
            
            if supabase:
                try:
                    supabase.table("extractions").insert({
                        "user_id": current_user.id,
                        "user_email": current_user.email,
                        "username": current_user.username,
                        "filename": file.filename,
                        "prompt": prompt,
                        "result": response.text,
                        "cost": cost,
                        "timestamp": datetime.now().isoformat()
                    }).execute()
                except Exception as e:
                    print(f"Error storing in Supabase: {e}")
            
            results.append({
                "filename": file.filename,
                "success": True,
                "extracted_text": response.text,
                "cost": round(cost, 6)
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e),
                "cost": 0
            })
    
    return {
        "success": True,
        "total_files": len(files),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "total_cost": round(total_cost, 6),
        "results": results,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/extractions")
async def get_extractions(
    limit: int = 100,
    current_user: User = Depends(get_current_user)
):
    """Get extraction history for current user"""
    if not supabase:
        return {"success": False, "message": "Supabase not configured", "data": []}
    
    try:
        response = supabase.table("extractions")\
            .select("*")\
            .eq("user_id", current_user.id)\
            .order("timestamp", desc=True)\
            .limit(limit)\
            .execute()
        
        return {"success": True, "data": response.data, "count": len(response.data)}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch extractions: {str(e)}"
        )

@app.get("/api/cost-summary")
async def get_cost_summary(current_user: User = Depends(get_current_user)):
    """Get cost summary for current user"""
    if not supabase:
        return {"success": False, "message": "Supabase not configured", "data": {}}
    
    try:
        response = supabase.table("extractions")\
            .select("cost, timestamp")\
            .eq("user_id", current_user.id)\
            .execute()
        
        total_cost = sum(item["cost"] for item in response.data)
        count = len(response.data)
        avg_cost = total_cost / count if count > 0 else 0
        
        from collections import defaultdict
        daily_costs = defaultdict(float)
        for item in response.data:
            date = item["timestamp"].split("T")[0]
            daily_costs[date] += item["cost"]
        
        return {
            "success": True,
            "data": {
                "total_cost": round(total_cost, 6),
                "total_extractions": count,
                "average_cost": round(avg_cost, 6),
                "daily_costs": dict(daily_costs)
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch cost summary: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "auth_type": "flow_based",
        "services": {
            "supabase": "connected" if supabase else "not configured",
            "gemini": "configured" if GEMINI_API_KEY else "not configured",
            "authentik": "configured" if AUTHENTIK_URL else "not configured"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)