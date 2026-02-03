"""
Simple OCR API - COMPLETE VERSION with Gemini Vision (NEW API)
✅ Gemini Vision only (no macOCR dependency)
✅ Works on Linux/Render
✅ PDF support with proper conversion
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import Optional, List
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
from datetime import datetime
import time
from database import log_processing, get_usage_stats, get_all_logs, delete_log

load_dotenv()

app = FastAPI(title="OCR API - Complete with Vision (NEW API)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
zoho_bulk = ZohoBulkAPI()


# ============================================================
# Cost calculation function
# ============================================================
def calculate_cost(input_tokens: int, output_tokens: int, method: str) -> float:
    """
    Calculate cost based on Gemini pricing (2.5 Flash)
    FREE TIER LIMITS:
    - 15 requests/minute
    - 1,000 requests/day
    
    PAID PRICING (after free tier):
    - Input: $0.075 per 1M tokens
    - Output: $0.30 per 1M tokens
    - Images: ~258 tokens standard
    """
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
    """Download file from URL"""
    print(f"[DOWNLOAD] Downloading from URL...")
    
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        
        response = requests.get(file_url, timeout=30, headers=headers, stream=True)
        response.raise_for_status()
        
        file_content = response.content
        filename = file_url.split('/')[-1].split('?')[0]
        
        if not any(filename.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.pdf']):
            filename = "downloaded_file.jpg"
        
        print(f"[DOWNLOAD] ✓ Downloaded: {filename} ({len(file_content):,} bytes)")
        return file_content, filename
        
    except Exception as e:
        print(f"[DOWNLOAD] ✗ Error: {str(e)}")
        raise


def process_single_file(file_content: bytes, filename: str, doc_type: str) -> dict:
    """
    Process a single file using Gemini Vision
    ✅ Handles images and PDFs
    """
    start_time = time.time()
    
    try:
        def calculate_tokens(text):
            """1 token ≈ 4 characters"""
            return max(1, len(text) // 4)
        
        # Check if Gemini is available
        if not USE_GEMINI:
            return {
                "error": "Gemini Vision not configured. Please set GEMINI_API_KEY.",
                "success": False,
                "filename": filename
            }
        
        print(f"[PROCESS] Using Gemini Vision...")
        
        # ✅ Handle PDF files
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
                
                # Convert PIL Image to bytes
                from io import BytesIO
                img_byte_arr = BytesIO()
                images[0].save(img_byte_arr, format='JPEG', quality=95)
                
                # ✅ Update file_content with converted image
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
        
        # Process with Gemini Vision
        input_tokens = 258  # Standard Gemini image token cost
        
        if doc_type == "bank":
          result = analyze_bank_gemini_vision(image_content, filename)
    
    # ✅ FIXED: Prioritize Gemini's bank name, use IFSC only as fallback
          gemini_bank_name = result.get('bank_name')
          if not gemini_bank_name or gemini_bank_name == 'null':
               ifsc = result.get('ifsc_code')
               result['bank_name'] = get_bank_name_from_ifsc(ifsc or '')
        else:
            result = analyze_bill_gemini_vision(image_content, filename)
        
        # Save image (always save as image, not PDF)
        image_id = str(uuid.uuid4())
        if converted_from_pdf:
            image_ext = ".jpg"
        else:
            image_ext = os.path.splitext(original_filename)[1] or ".jpg"
        
        save_path = f"uploads/{image_id}{image_ext}"
        with open(save_path, "wb") as f:
            f.write(image_content)
        
        # Use environment variable for base URL or default to localhost
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
        "service": "OCR API - Gemini Vision",
        "version": "5.0",
        "features": [
            "✅ Gemini Vision (NEW API - most accurate)",
            "✅ PDF support with automatic conversion",
            "✅ Bank passbook extraction",
            "✅ College bill extraction",
            "✅ Batch processing"
        ],
        "gemini_vision_status": "✅ Available" if USE_GEMINI else "❌ Not configured",
        "endpoints": {
            "POST /ocr/bank": "Process bank passbook",
            "POST /ocr/bill": "Process college bill",
            "GET /health": "Health check",
            "GET /stats": "Usage statistics",
            "GET /logs": "Processing logs"
        }
    }


@app.post("/ocr/bank")
async def process_bank_passbook(
    files: List[UploadFile] = File(default=[]),
    file: Optional[UploadFile] = File(None),
    File_upload_Bank: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    File_upload_Bank_url: Optional[str] = Form(None),
    scholarship_id: Optional[str] = Form(None),
    student_name: Optional[str] = Form(None)
):
    """Process bank passbook - supports single or multiple files"""
    try:
        print(f"\n{'='*80}")
        print(f"[BANK OCR] Processing bank passbook(s)")
        if student_name:
            print(f"[BANK OCR] Student: {student_name}")
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
    files: List[UploadFile] = File(default=[]),
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    scholarship_id: Optional[str] = Form(None),
    student_name: Optional[str] = Form(None)
):
    """Process college bill - supports single or multiple files"""
    try:
        print(f"\n{'='*80}")
        print(f"[BILL OCR] Processing bill(s)")
        if student_name:
            print(f"[BILL OCR] Student: {student_name}")
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

@app.post("/sync/bulk-push-to-zoho")
async def bulk_push_to_zoho(limit: int = 1000):
    """Bulk push records from Supabase to Zoho Creator FORM"""
    try:
        print(f"\n{'='*80}")
        print(f"INITIATING BULK SYNC TO ZOHO CREATOR FORM")
        print(f"{'='*80}")
        
        # Fetch records from Supabase
        print(f"[FETCH] Loading records from Supabase...")
        
        from supabase import create_client
        supabase_url = os.getenv("VITE_SUPABASE_URL", "https://ohfnriyabohbvgxebllt.supabase.co")
        supabase_key = os.getenv("VITE_SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im9oZm5yaXlhYm9oYnZneGVibGx0Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzQ2ODI2MTksImV4cCI6MjA1MDI1ODYxOX0.KI_E7vVgzDPpKj5Sh0fZvfaG7h5mq6c5NmqfvU7vU7c")
        supabase = create_client(supabase_url, supabase_key)
        
        response = supabase.table('auto_extraction_results').select('*').limit(limit).execute()
        records = response.data
        
        if not records:
            return JSONResponse(content={
                "success": False,
                "message": "No records found in Supabase"
            })
        
        print(f"[FETCH] ✓ Loaded {len(records)} records from Supabase")
        
        # Bulk push to Zoho Form
        result = zoho_bulk.bulk_insert(records)
        
        return JSONResponse(content={
            "success": True,
            "message": f"Bulk sync completed: {result['successful']}/{result['total_records']} successful",
            "details": result
        })
        
    except Exception as e:
        print(f"[ERROR] Bulk sync failed: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@app.get("/sync/test-zoho-connection")
async def test_zoho_connection():
    """Test Zoho Creator FORM connection"""
    try:
        result = zoho_bulk.test_connection()
        
        return JSONResponse(content={
            "success": result['success'],
            "message": "Zoho Form connection test",
            "result": result
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "service": "OCR API - Gemini Vision",
        "version": "5.0",
        "gemini_vision": "enabled" if USE_GEMINI else "disabled"
    }


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
        
        # Use existing bulk push functionality
        result = zoho_bulk.bulk_insert(records)
        
        return JSONResponse(content={
            "success": True,
            "message": f"Pushed {result['successful']}/{result['total_records']} records",
            "details": result
        })
        
    except Exception as e:
        print(f"[ERROR] {str(e)}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )
        
# ============================================================
# 🆕 NEW ENDPOINTS - Add these here
# ============================================================

@app.get("/stats/student/{student_name}")
async def get_student_stats(student_name: str):
    """Get stats for a specific student"""
    try:
        from database import get_logs_by_student
        logs = get_logs_by_student(student_name)
        
        total_cost = sum(float(log.get('cost_usd', 0)) for log in logs)
        
        return JSONResponse(content={
            "student_name": student_name,
            "total_requests": len(logs),
            "total_cost_usd": round(total_cost, 6),
            "logs": logs
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/stats/scholarship/{scholarship_id}")
async def get_scholarship_stats(scholarship_id: str):
    """Get stats for a specific scholarship"""
    try:
        from database import get_logs_by_scholarship
        logs = get_logs_by_scholarship(scholarship_id)
        
        total_cost = sum(float(log.get('cost_usd', 0)) for log in logs)
        
        return JSONResponse(content={
            "scholarship_id": scholarship_id,
            "total_requests": len(logs),
            "total_cost_usd": round(total_cost, 6),
            "logs": logs
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/stats/total-cost")
async def get_total_api_cost():
    """Get total cost across all processing"""
    try:
        from database import get_total_cost
        total = get_total_cost()
        
        return JSONResponse(content={
            "total_cost_usd": total,
            "message": f"Total API cost: ${total}"
        })
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/ocr/generic")
async def process_generic_document(
    files: List[UploadFile] = File(default=[]),
    file: Optional[UploadFile] = File(None),
    file_url: Optional[str] = Form(None),
    extraction_prompt: str = Form(...),  # Required custom prompt
    student_name: Optional[str] = Form(None),
    scholarship_id: Optional[str] = Form(None)
):
    """Generic OCR - Extract anything based on user's custom prompt"""
    try:
        print(f"\n{'='*80}")
        print(f"[GENERIC OCR] Processing document(s)")
        print(f"[GENERIC OCR] Extraction prompt: {extraction_prompt[:100]}...")
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
        
        print(f"[GENERIC OCR] Processing {len(files_to_process)} file(s)")
        
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
            
            # Use generic processing
            result = process_generic_file(file_content, filename, extraction_prompt)
            
            # Calculate cost and log
            token_usage = result.get('token_usage', {})
            input_tok = token_usage.get('input_tokens', 0)
            output_tok = token_usage.get('output_tokens', 0)
            cost = calculate_cost(input_tok, output_tok, result.get('method', 'unknown'))
            
            log_processing(
                doc_type="generic",
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
            
            print(f"[GENERIC OCR] ✓ Completed: {filename}")
            print(f"[GENERIC OCR]   Cost: ${cost}")
        
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
        print(f"[GENERIC OCR] ✗ Error: {str(e)}")
        
        log_processing(
            doc_type="generic",
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


def process_generic_file(file_content: bytes, filename: str, extraction_prompt: str) -> dict:
    """Process any document with custom extraction prompt"""
    start_time = time.time()
    
    try:
        from ai_analyzer import analyze_generic_gemini_vision, USE_GEMINI
        
        def calculate_tokens(text):
            return max(1, len(text) // 4)
        
        if not USE_GEMINI:
            return {
                "error": "Gemini Vision not configured. Please set GEMINI_API_KEY.",
                "success": False,
                "filename": filename
            }
        
        print(f"[PROCESS] Using Gemini Vision for generic extraction...")
        
        # Handle PDF conversion
        image_content = file_content
        original_filename = filename
        converted_from_pdf = False
        
        if filename.lower().endswith('.pdf'):
            print(f"[PROCESS] Converting PDF to image...")
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
                
                print(f"[PROCESS] ✓ PDF converted to image")
            except ImportError:
                return {
                    "error": "pdf2image not installed",
                    "success": False,
                    "filename": original_filename
                }
            finally:
                if os.path.exists(temp_pdf_path):
                    os.unlink(temp_pdf_path)
        
        # Process with Gemini Vision using custom prompt
        input_tokens = 258
        result = analyze_generic_gemini_vision(image_content, filename, extraction_prompt)
        
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


if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment (Render provides PORT variable)
    port = int(os.environ.get("PORT", 8000))
    
    print("="*80)
    print("OCR API - GEMINI VISION ONLY")
    print("="*80)
    print(f"✓ Gemini Vision: {'ENABLED ✅' if USE_GEMINI else 'DISABLED ❌'}")
    print("✓ PDF support with auto-conversion")
    print("✓ Bank passbook + Bill extraction")
    print("✓ Batch processing")
    print("="*80)
    print(f"\nStarting server on http://0.0.0.0:{port}")
    print("="*80 + "\n")
    
    uvicorn.run(app, host="0.0.0.0", port=port)