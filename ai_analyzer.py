"""
Enhanced AI Analyzer Module - FIXED WITH IMAGE VALIDATION + PDF SUPPORT
- Proper image validation before sending to Gemini
- Automatic image optimization (resize, compress)
- PDF to image conversion support
- Better MIME type detection
- Comprehensive error handling
"""
import json
import re
from typing import Dict, Any, List, Tuple
import os
import time
from dotenv import load_dotenv
from PIL import Image
import io

# ✅ NEW: Register HEIC support at module load
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    HEIC_SUPPORT = True
    print("[AI Config] ✓ HEIC support enabled (pillow-heif)")
except ImportError:
    HEIC_SUPPORT = False
    print("[AI Config] ⚠️ HEIC support disabled - install: pip install pillow-heif")

# Try to import PDF support
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
    print("[AI Config] ✓ PDF support enabled (pdf2image)")
except ImportError:
    PDF_SUPPORT = False
    print("[AI Config] ⚠️ PDF support disabled - install: pip3 install pdf2image")
    print("[AI Config] ⚠️ Also install poppler: brew install poppler (macOS)")


# ============================================================
# ADD/REPLACE in ai_analyzer.py
# ============================================================

# ── Try PPTX support at module level (alongside existing imports) ──
try:
    from pptx import Presentation
    from pptx.util import Inches
    PPTX_SUPPORT = True
    print("[AI Config] ✓ PPTX support enabled (python-pptx)")
except ImportError:
    PPTX_SUPPORT = False
    print("[AI Config] ⚠️ PPTX support disabled - install: pip install python-pptx")

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
GEMINI_MODEL = 'gemini-3-flash-preview'

print(f"[AI Config] Selected Model: {OLLAMA_MODEL if OLLAMA_AVAILABLE else GEMINI_MODEL if USE_GEMINI else 'OpenAI GPT' if USE_OPENAI else 'None'}")


def validate_and_optimize_image(file_content: bytes, filename: str) -> Tuple[bytes, str]:
    """
    Validate and optimize image for Gemini Vision
    - Validates image can be opened
    - Converts PDFs to images
    - ✅ NEW: Converts HEIC to JPEG
    - Resizes if too large
    - Converts to JPEG if needed
    - Compresses if file size is too large
    
    Returns: (optimized_bytes, mime_type)
    """
    print(f"[IMAGE VALIDATION] Validating: {filename} ({len(file_content):,} bytes)")
    
    # Check if file is a PDF
    is_pdf = file_content[:4] == b'%PDF' or file_content[:5] == b'\x0a%PDF'
    
    # ✅ NEW: Check if file is HEIC
    is_heic = len(file_content) > 12 and b'ftyp' in file_content[:20] and (b'heic' in file_content[:20] or b'heif' in file_content[:20])
    
    if is_pdf:
        print(f"[IMAGE VALIDATION]   📄 Detected PDF file")
        
        if not PDF_SUPPORT:
            raise Exception(f"PDF file detected but PDF support not installed. Install: pip3 install pdf2image && brew install poppler")
        
        try:
            # Convert PDF to images (take first page only)
            print(f"[IMAGE VALIDATION]   Converting PDF to image...")
            images = convert_from_bytes(file_content, dpi=200, first_page=1, last_page=1)
            
            if not images:
                raise Exception("Failed to convert PDF - no pages found")
            
            img = images[0]
            print(f"[IMAGE VALIDATION]   ✓ PDF converted: {img.size} ({img.mode})")
            
            # Convert to bytes
            output = io.BytesIO()
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(output, format='JPEG', quality=90, optimize=True)
            file_content = output.getvalue()
            
            print(f"[IMAGE VALIDATION]   ✓ Converted to JPEG: {len(file_content):,} bytes")
            
            # Continue with normal image processing
            img = Image.open(io.BytesIO(file_content))
            
        except Exception as e:
            print(f"[IMAGE VALIDATION]   ✗ PDF conversion failed: {e}")
            raise Exception(f"Failed to convert PDF to image: {e}")
    
    # ✅ NEW: Handle HEIC files
    elif is_heic:
        print(f"[IMAGE VALIDATION]   📱 Detected HEIC file")
        
        if not HEIC_SUPPORT:
            raise Exception(f"HEIC file detected but pillow-heif not installed. Install: pip install pillow-heif")
        
        try:
            # Open HEIC image
            print(f"[IMAGE VALIDATION]   Converting HEIC to JPEG...")
            img = Image.open(io.BytesIO(file_content))
            
            print(f"[IMAGE VALIDATION]   ✓ HEIC opened: {img.size} ({img.mode})")
            
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA', 'P'):
                print(f"[IMAGE VALIDATION]   Converting {img.mode} to RGB")
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPEG
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=95, optimize=True)
            file_content = output.getvalue()
            
            print(f"[IMAGE VALIDATION]   ✓ Converted to JPEG: {len(file_content):,} bytes")
            
            # Continue with normal image processing
            img = Image.open(io.BytesIO(file_content))
            
        except Exception as e:
            print(f"[IMAGE VALIDATION]   ✗ HEIC conversion failed: {e}")
            raise Exception(f"Failed to convert HEIC to JPEG: {e}")
    
    try:
        # Try to open the image (or converted PDF/HEIC)
        if not is_pdf and not is_heic:
            img = Image.open(io.BytesIO(file_content))
        
        # Get original format and size
        original_format = img.format
        original_size = img.size
        original_mode = img.mode
        
        print(f"[IMAGE VALIDATION]   Format: {original_format}, Size: {original_size}, Mode: {original_mode}")
        
        # Define limits (Gemini supports up to 4096x4096, but we'll be conservative)
        MAX_DIMENSION = 3072
        MAX_FILE_SIZE_MB = 4
        TARGET_FILE_SIZE_MB = 2
        
        needs_optimization = False
        
        # Check if dimensions are too large
        if max(original_size) > MAX_DIMENSION:
            print(f"[IMAGE VALIDATION]   ⚠️ Image too large, will resize")
            needs_optimization = True
        
        # Check if file size is too large
        size_mb = len(file_content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            print(f"[IMAGE VALIDATION]   ⚠️ File size {size_mb:.2f}MB too large, will compress")
            needs_optimization = True
        
        # Convert RGBA to RGB if needed (JPEG doesn't support alpha)
        if img.mode in ('RGBA', 'LA', 'P'):
            print(f"[IMAGE VALIDATION]   Converting {img.mode} to RGB")
            # Create white background
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
            needs_optimization = True
        
        # Optimize if needed
        if needs_optimization:
            # Calculate resize ratio
            if max(original_size) > MAX_DIMENSION:
                ratio = MAX_DIMENSION / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                print(f"[IMAGE VALIDATION]   Resizing to: {new_size}")
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            # Convert to JPEG with progressive quality reduction
            output = io.BytesIO()
            quality = 95
            
            while quality >= 60:
                output.seek(0)
                output.truncate()
                
                img.save(output, format='JPEG', quality=quality, optimize=True)
                
                output_size_mb = output.tell() / (1024 * 1024)
                
                if output_size_mb <= TARGET_FILE_SIZE_MB or quality <= 60:
                    break
                
                quality -= 5
            
            optimized_bytes = output.getvalue()
            mime_type = "image/jpeg"
            
            print(f"[IMAGE VALIDATION]   ✓ Optimized: {len(optimized_bytes):,} bytes (quality: {quality})")
            print(f"[IMAGE VALIDATION]   Size reduction: {len(file_content):,} → {len(optimized_bytes):,} bytes")
            
            return optimized_bytes, mime_type
        
        else:
            # Image is fine, just detect MIME type
            if original_format == 'PNG':
                mime_type = "image/png"
            elif original_format in ('JPEG', 'JPG'):
                mime_type = "image/jpeg"
            elif original_format == 'WEBP':
                mime_type = "image/webp"
            else:
                # Convert unknown formats to JPEG
                print(f"[IMAGE VALIDATION]   Converting {original_format} to JPEG")
                output = io.BytesIO()
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output, format='JPEG', quality=90, optimize=True)
                return output.getvalue(), "image/jpeg"
            
            print(f"[IMAGE VALIDATION]   ✓ Image is valid, no optimization needed")
            return file_content, mime_type
    
    except Exception as e:
        if is_pdf or is_heic:
            # Already handled PDF/HEIC errors above
            raise
        
        print(f"[IMAGE VALIDATION]   ✗ Failed to validate image: {e}")
        
        # Check if file is actually an image by magic bytes
        is_jpeg = file_content[:2] == b'\xff\xd8'
        is_png = file_content[:4] == b'\x89PNG'
        is_webp = len(file_content) > 12 and file_content[8:12] == b'WEBP'
        
        if not any([is_jpeg, is_png, is_webp, is_heic]):
            # Not a valid image format
            print(f"[IMAGE VALIDATION]   ✗ File is NOT a valid image!")
            print(f"[IMAGE VALIDATION]   First 20 bytes: {file_content[:20].hex()}")
            
            # Check if it's a PDF
            if file_content[:4] == b'%PDF' or file_content[:5] == b'\x0a%PDF':
                print(f"[IMAGE VALIDATION]   ✗ File is a PDF but conversion failed!")
                raise Exception(f"PDF file detected but conversion failed. Ensure poppler is installed: brew install poppler")
            
            # Check if it's HEIC
            if is_heic:
                print(f"[IMAGE VALIDATION]   ✗ File is HEIC but conversion failed!")
                raise Exception(f"HEIC file detected but conversion failed. Install pillow-heif: pip install pillow-heif")
            
            # Try to see if it's text (HTML/JSON error)
            try:
                text_preview = file_content[:200].decode('utf-8', errors='ignore')
                if '<html' in text_preview.lower() or '<!doctype' in text_preview.lower():
                    print(f"[IMAGE VALIDATION]   ✗ File is HTML, not an image!")
                    print(f"[IMAGE VALIDATION]   Preview: {text_preview[:150]}")
                    raise Exception(f"Downloaded file is HTML error page, not an image")
                elif '{' in text_preview and ('"error"' in text_preview or '"message"' in text_preview):
                    print(f"[IMAGE VALIDATION]   ✗ File is JSON error, not an image!")
                    print(f"[IMAGE VALIDATION]   Preview: {text_preview[:150]}")
                    raise Exception(f"Downloaded file is JSON error, not an image")
            except UnicodeDecodeError:
                pass
            
            raise Exception(f"File is not a valid image format. Cannot process.")
        
        # Detect MIME type from magic bytes as fallback
        if is_png:
            mime_type = "image/png"
        elif is_jpeg:
            mime_type = "image/jpeg"
        elif is_webp:
            mime_type = "image/webp"
        elif is_heic:
            mime_type = "image/heic"
        else:
            mime_type = "image/jpeg"
        
        print(f"[IMAGE VALIDATION]   ⚠️ Using fallback MIME type: {mime_type}")
        print(f"[IMAGE VALIDATION]   ⚠️ Sending raw bytes without optimization")
        
        return file_content, mime_type


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
    ✅ FIXED: Analyze bank passbook using Gemini Vision with image validation
    """
    if not USE_GEMINI:
        raise Exception("Gemini Vision not available - add GEMINI_API_KEY to .env")
    
    print(f"[GEMINI VISION] Analyzing bank passbook: {filename}")
    
    try:
        from google.genai import types
        
        # ✅ NEW: Validate and optimize image first
        optimized_content, mime_type = validate_and_optimize_image(file_content, filename)
        
        print(f"[GEMINI VISION] Sending to Gemini: {mime_type}, {len(optimized_content):,} bytes")
        
        prompt = """
Analyze this Indian bank passbook image.

STEP 1 - TRANSCRIBE THE ACCOUNT NUMBER:
Write each digit of the account number with its position number:
Position 1: [digit]
Position 2: [digit]
Position 3: [digit]
...continue for every digit...
Total digit count: [N]

STEP 2 - VERIFY REPEATED SEQUENCES:
List any runs of identical digits you found:
Example format: "positions 7-11 are all zeros (5 zeros)"

STEP 3 - OUTPUT JSON:
Using ONLY what you transcribed above (not memory or guessing), return:
{
    "bank_name": "<as seen>",
    "account_holder_name": "<as seen>",
    "account_number": "<paste exact digits from Step 1>",
    "ifsc_code": "<4 letters + 0 + 6 chars, exactly as seen>",
    "branch_name": "<as seen>",
    "digit_count": <number from Step 1>,
    "confidence": "high|medium|low",
    "repeated_sequences": "<from Step 2, or 'none'>"
}

RULES:
- account_number must be a STRING (not a number) to preserve leading zeros
- digit_count must equal len(account_number)
- If anything is unclear, set confidence to "low" — never guess digits
"""
        
        # ✅ Call Gemini with optimized image
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
        
        # Parse response
        response_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(response_text)

                # ✅ FIX: Handle case where Gemini returns a list instead of dict
        if isinstance(data, list):
            print(f"[GEMINI VISION] ⚠️ Received array response, taking first element")
            data = data[0] if data else {}

        if not isinstance(data, dict):
            data = {}
        
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
    ✅ FIXED: Analyze bill using Gemini Vision with image validation
    """
    if not USE_GEMINI:
        raise Exception("Gemini Vision not available")
    
    print(f"[GEMINI VISION] Analyzing bill: {filename}")
    
    try:
        from google.genai import types
        
        # ✅ NEW: Validate and optimize image first
        optimized_content, mime_type = validate_and_optimize_image(file_content, filename)
        
        print(f"[GEMINI VISION] Sending to Gemini: {mime_type}, {len(optimized_content):,} bytes")
        
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
        
        # ✅ Call Gemini with optimized image
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
        
        response_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(response_text)
        
        # ✅ Handle different response types
        if isinstance(data, list):
            print(f"[GEMINI VISION] ⚠️ Received array response, taking first element")
            if len(data) > 0:
                data = data[0]
            else:
                data = {
                    "student_name": None,
                    "college_name": None,
                    "roll_number": None,
                    "receipt_number": None,
                    "class_course": None,
                    "bill_date": None,
                    "amount": None
                }
        
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


# ============================================================
# ADD THIS FUNCTION to ai_analyzer.py
# Place it right after analyze_bill_gemini_vision()
# ============================================================

def analyze_bill_multi_page(file_content: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extract bills from a PDF that may contain MULTIPLE bills (one per page).
    
    - Single-page PDFs / plain images → returns a list with 1 bill dict
    - Multi-page PDFs                 → returns a list with N bill dicts
                                        (index 0 = bill1, index 1 = bill2, …)
    
    Each dict is the same shape as analyze_bill_gemini_vision() output:
    {
        "student_name": ...,
        "college_name": ...,
        "roll_number": ...,
        "receipt_number": ...,
        "class_course": ...,
        "bill_date": ...,
        "amount": ...,
        "page_number": 1          # which page this bill came from
    }
    """
    if not USE_GEMINI:
        raise Exception("Gemini Vision not available - add GEMINI_API_KEY to .env")

    print(f"[BILL MULTI-PAGE] Starting: {filename}")

    try:
        from google.genai import types

        # ─── Check if it's a PDF ──────────────────────────────────
        is_pdf = (
            file_content[:4] == b'%PDF'
            or file_content[:5] == b'\x0a%PDF'
        )

        if is_pdf:
            print(f"[BILL MULTI-PAGE] 📄 PDF detected – converting all pages")
            if not PDF_SUPPORT:
                raise Exception(
                    "PDF support not installed. "
                    "Run: pip3 install pdf2image && brew install poppler"
                )
            images_to_process = convert_pdf_to_images(file_content)
            print(f"[BILL MULTI-PAGE] {len(images_to_process)} page(s) extracted from PDF")
        else:
            print(f"[BILL MULTI-PAGE] 🖼️ Single image – treating as one bill")
            optimized, mime_type = validate_and_optimize_image_single(file_content, filename)
            images_to_process = [(optimized, mime_type)]

        # ─── Bill extraction prompt ───────────────────────────────
        BILL_PROMPT = """
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
- If the page is blank or not a bill, return amount as null

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

        # ─── Process each page ────────────────────────────────────
        bills: List[Dict[str, Any]] = []
        skipped = 0

        for page_idx, (image_bytes, mime_type) in enumerate(images_to_process, 1):
            print(f"[BILL MULTI-PAGE] [{page_idx}/{len(images_to_process)}] Extracting bill from page…")

            try:
                response = genai_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        BILL_PROMPT,
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )

                response_text = response.text.replace("```json", "").replace("```", "").strip()
                data = json.loads(response_text)

                # Normalise to dict (Gemini occasionally wraps in a list)
                if isinstance(data, list):
                    data = data[0] if data else {}
                if not isinstance(data, dict):
                    data = {}

                # Stamp which page this came from
                data["page_number"] = page_idx

                # Skip pages that look blank / are not bills
                if data.get("amount") is None and data.get("college_name") is None:
                    print(f"[BILL MULTI-PAGE]   ⚠️ Page {page_idx} appears blank or not a bill – skipping")
                    skipped += 1
                    continue

                print(f"[BILL MULTI-PAGE]   ✅ Page {page_idx}: "
                      f"College={data.get('college_name')} | "
                      f"Student={data.get('student_name')} | "
                      f"Amount=₹{data.get('amount')}")

                bills.append(data)

            except json.JSONDecodeError as e:
                print(f"[BILL MULTI-PAGE]   ✗ Page {page_idx}: JSON parse error – {e}")
            except Exception as e:
                print(f"[BILL MULTI-PAGE]   ✗ Page {page_idx}: {e}")
                import traceback; traceback.print_exc()

        print(f"\n[BILL MULTI-PAGE] Done – {len(bills)} bill(s) extracted"
              f" ({skipped} page(s) skipped)")

        return bills

    except Exception as e:
        print(f"[BILL MULTI-PAGE] ✗ Fatal error: {e}")
        import traceback; traceback.print_exc()
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


def analyze_generic_gemini_vision(file_content: bytes, filename: str, user_prompt: str) -> Dict[str, Any]:
    """
    ✅ FIXED: Generic document analysis using Gemini Vision with image validation
    """
    if not USE_GEMINI:
        raise Exception("Gemini Vision not available - add GEMINI_API_KEY to .env")
    
    print(f"[GEMINI VISION] Analyzing document: {filename}")
    print(f"[GEMINI VISION] Custom prompt: {user_prompt[:100]}...")
    
    try:
        from google.genai import types
        
        # ✅ NEW: Validate and optimize image first
        optimized_content, mime_type = validate_and_optimize_image(file_content, filename)
        
        print(f"[GEMINI VISION] Sending to Gemini: {mime_type}, {len(optimized_content):,} bytes")
        
        full_prompt = f"""
You are a highly accurate OCR and data extraction AI. Analyze this document image and extract the information requested by the user.

USER'S EXTRACTION REQUEST:
{user_prompt}

INSTRUCTIONS:
1. Read all text in the image carefully, including rotated, small, or handwritten text
2. Extract ONLY the information the user requested
3. If a field is not found or unclear, use null
4. IMPORTANT: Always return a JSON OBJECT, never just a string value
5. If extracting a single value, wrap it in an object like {{"value": "extracted_value"}}
6. Be as accurate as possible with numbers, dates, and names

Return your response as a valid JSON object with the fields the user requested.
"""
        
        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=optimized_content,
                    mime_type=mime_type
                ),
                full_prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )
        
        response_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(response_text)
        
        # Handle different response types
        if isinstance(data, str):
            print(f"[GEMINI VISION] ⚠️ Received string response, wrapping in object")
            data = {"extracted_value": data}
        elif isinstance(data, list):
            print(f"[GEMINI VISION] ⚠️ Received array response")
            if len(data) > 0:
                if isinstance(data[0], dict):
                    data = data[0]
                else:
                    data = {"extracted_values": data}
            else:
                data = {"extracted_values": []}
        elif isinstance(data, (int, float, bool)):
            print(f"[GEMINI VISION] ⚠️ Received primitive value, wrapping in object")
            data = {"extracted_value": data}
        elif not isinstance(data, dict):
            print(f"[GEMINI VISION] ⚠️ Unexpected type: {type(data)}, wrapping in object")
            data = {"extracted_value": str(data)}
        
        if not isinstance(data, dict):
            data = {"error": "Invalid response format", "raw_response": str(data)}
        
        print(f"[GEMINI VISION] ✓ Extraction completed")
        print(f"[GEMINI VISION]   Extracted {len(data)} fields")
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"[GEMINI VISION] ✗ JSON Parse Error: {e}")
        print(f"[GEMINI VISION] Raw response: {response_text[:200]}...")
        return {
            "error": "Failed to parse JSON response",
            "raw_response": response_text[:500]
        }
    except Exception as e:
        print(f"[GEMINI VISION] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise

# Replace these functions in your ai_analyzer.py

def convert_pdf_to_images(file_content: bytes) -> List[Tuple[bytes, str]]:
    """
    Convert ALL pages of PDF to images
    Returns list of (image_bytes, mime_type) tuples
    """
    print(f"[PDF CONVERTER] Processing multi-page PDF...")
    
    if not PDF_SUPPORT:
        raise Exception("PDF support not installed. Install: pip3 install pdf2image && brew install poppler")
    
    try:
        # Convert ALL pages (not just first page)
        images = convert_from_bytes(file_content, dpi=300)
        
        if not images:
            raise Exception("Failed to convert PDF - no pages found")
        
        print(f"[PDF CONVERTER] ✓ Converted {len(images)} pages from PDF")
        
        # Convert each image to JPEG bytes
        image_list = []
        for idx, img in enumerate(images, 1):
            print(f"[PDF CONVERTER]   Page {idx}: {img.size} ({img.mode})")
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPEG
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=90, optimize=True)
            image_bytes = output.getvalue()
            
            image_list.append((image_bytes, "image/jpeg"))
            print(f"[PDF CONVERTER]   Page {idx}: {len(image_bytes):,} bytes")
        
        return image_list
        
    except Exception as e:
        print(f"[PDF CONVERTER] ✗ Conversion failed: {e}")
        raise Exception(f"Failed to convert PDF to images: {e}")


def validate_and_optimize_image_single(file_content: bytes, filename: str) -> Tuple[bytes, str]:
    """
    ✅ UPDATED: Validate and optimize a single image with HEIC support
    Used for individual image processing
    """
    print(f"[IMAGE VALIDATION] Validating: {filename} ({len(file_content):,} bytes)")
    
    try:
        # Check for HEIC first
        is_heic = len(file_content) > 12 and b'ftyp' in file_content[:20] and (b'heic' in file_content[:20] or b'heif' in file_content[:20])
        
        if is_heic:
            print(f"[IMAGE VALIDATION]   📱 Detected HEIC - converting to JPEG...")
            if not HEIC_SUPPORT:
                raise Exception("HEIC support not available - install: pip install pillow-heif")
            
            img = Image.open(io.BytesIO(file_content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=95, optimize=True)
            file_content = output.getvalue()
            print(f"[IMAGE VALIDATION]   ✓ HEIC converted to JPEG: {len(file_content):,} bytes")
        
        # Open and validate image
        img = Image.open(io.BytesIO(file_content))
        
        original_format = img.format
        original_size = img.size
        original_mode = img.mode
        
        print(f"[IMAGE VALIDATION]   Format: {original_format}, Size: {original_size}, Mode: {original_mode}")
        
        # Define limits
        MAX_DIMENSION = 3072
        MAX_FILE_SIZE_MB = 4
        TARGET_FILE_SIZE_MB = 2
        
        needs_optimization = False
        
        # Check dimensions
        if max(original_size) > MAX_DIMENSION:
            print(f"[IMAGE VALIDATION]   ⚠️ Image too large, will resize")
            needs_optimization = True
        
        # Check file size
        size_mb = len(file_content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            print(f"[IMAGE VALIDATION]   ⚠️ File size {size_mb:.2f}MB too large, will compress")
            needs_optimization = True
        
        # Convert RGBA to RGB
        if img.mode in ('RGBA', 'LA', 'P'):
            print(f"[IMAGE VALIDATION]   Converting {img.mode} to RGB")
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
            needs_optimization = True
        
        # Optimize if needed
        if needs_optimization:
            if max(original_size) > MAX_DIMENSION:
                ratio = MAX_DIMENSION / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
                print(f"[IMAGE VALIDATION]   Resizing to: {new_size}")
                img = img.resize(new_size, Image.Resampling.LANCZOS)
            
            output = io.BytesIO()
            quality = 95
            
            while quality >= 60:
                output.seek(0)
                output.truncate()
                img.save(output, format='JPEG', quality=quality, optimize=True)
                
                if output.tell() / (1024 * 1024) <= TARGET_FILE_SIZE_MB or quality <= 60:
                    break
                
                quality -= 5
            
            optimized_bytes = output.getvalue()
            mime_type = "image/jpeg"
            
            print(f"[IMAGE VALIDATION]   ✓ Optimized: {len(optimized_bytes):,} bytes (quality: {quality})")
            return optimized_bytes, mime_type
        
        else:
            # Determine MIME type
            if original_format == 'PNG':
                mime_type = "image/png"
            elif original_format in ('JPEG', 'JPG'):
                mime_type = "image/jpeg"
            elif original_format == 'WEBP':
                mime_type = "image/webp"
            else:
                # Convert unknown formats
                output = io.BytesIO()
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output, format='JPEG', quality=90, optimize=True)
                return output.getvalue(), "image/jpeg"
            
            print(f"[IMAGE VALIDATION]   ✓ Valid image: {mime_type}")
            return file_content, mime_type
    
    except Exception as e:
        print(f"[IMAGE VALIDATION]   ✗ Validation failed: {e}")
        raise

def analyze_barcode_gemini_vision_multi_page(file_content: bytes, filename: str) -> Dict[str, Any]:
    """
    ✅ NEW: Extract barcodes from ALL pages of a PDF
    Handles single images and multi-page PDFs
    """
    if not USE_GEMINI:
        raise Exception("Gemini Vision not available - add GEMINI_API_KEY to .env")
    
    print(f"[BARCODE EXTRACTION] Starting: {filename}")
    
    try:
        from google.genai import types
        
        # Check if file is PDF
        is_pdf = file_content[:4] == b'%PDF' or file_content[:5] == b'\x0a%PDF'
        
        if is_pdf:
            print(f"[BARCODE EXTRACTION] 📄 Multi-page PDF detected")
            images_to_process = convert_pdf_to_images(file_content)
        else:
            print(f"[BARCODE EXTRACTION] 🖼️ Single image file")
            optimized, mime_type = validate_and_optimize_image_single(file_content, filename)
            images_to_process = [(optimized, mime_type)]
        
        print(f"[BARCODE EXTRACTION] Processing {len(images_to_process)} image(s)...")
        
        all_barcodes = []
        total_files_processed = 0
        
        # Process each image
        for page_idx, (image_bytes, mime_type) in enumerate(images_to_process, 1):
            print(f"\n[BARCODE EXTRACTION] [{page_idx}/{len(images_to_process)}] Processing page/image...")
            
            try:
                prompt = """
You are a barcode/QR code scanner. Extract ALL barcodes from this image.

CRITICAL INSTRUCTIONS:
- Find and extract EVERY barcode in the image, no matter how many
- For each barcode, extract the type (Code 128, EAN-13, QR Code, etc.) and the exact data
- Return ALL barcodes found, even if there are many (9, 12, 18, or more)
- If no barcodes found, return empty array

Return ONLY this JSON:
{
    "page": 1,
    "total_barcodes_on_page": 9,
    "barcodes": [
        {"type": "Code 128", "data": "CT270168007IN"},
        {"type": "Code 128", "data": "CT270168136IN"}
    ]
}
"""
                
                # Call Gemini for this image
                response = genai_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        types.Part.from_bytes(
                            data=image_bytes,
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
                
                # Handle response format
                if isinstance(data, list):
                    data = {"page": page_idx, "total_barcodes_on_page": len(data), "barcodes": data}
                elif not isinstance(data, dict):
                    data = {"page": page_idx, "total_barcodes_on_page": 0, "barcodes": []}
                
                page_barcodes = data.get('barcodes', [])
                
                print(f"[BARCODE EXTRACTION]   ✅ Page {page_idx}: Found {len(page_barcodes)} barcode(s)")
                
                # Show first few barcodes from this page
                for i, bc in enumerate(page_barcodes[:3]):
                    print(f"[BARCODE EXTRACTION]      [{i+1}] {bc.get('type')}: {bc.get('data')}")
                
                if len(page_barcodes) > 3:
                    print(f"[BARCODE EXTRACTION]      ... and {len(page_barcodes) - 3} more on this page")
                
                # Add to all barcodes with page info
                for barcode in page_barcodes:
                    barcode['page'] = page_idx
                    all_barcodes.append(barcode)
                
                total_files_processed += 1
                
            except json.JSONDecodeError as e:
                print(f"[BARCODE EXTRACTION]   ⚠️ Page {page_idx}: JSON parse error: {e}")
                print(f"[BARCODE EXTRACTION]   Raw: {response_text[:200]}...")
            except Exception as e:
                print(f"[BARCODE EXTRACTION]   ⚠️ Page {page_idx}: Error: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n[BARCODE EXTRACTION] ✅ Complete Summary:")
        print(f"[BARCODE EXTRACTION]   Total Pages Processed: {total_files_processed}")
        print(f"[BARCODE EXTRACTION]   Total Barcodes Found: {len(all_barcodes)}")
        
        if is_pdf:
            print(f"[BARCODE EXTRACTION]   Average per page: {len(all_barcodes) / len(images_to_process):.1f}")
        
        # Group by type for summary
        type_summary = {}
        for bc in all_barcodes:
            bc_type = bc.get('type', 'Unknown')
            type_summary[bc_type] = type_summary.get(bc_type, 0) + 1
        
        print(f"[BARCODE EXTRACTION]   By type:")
        for bc_type, count in sorted(type_summary.items()):
            print(f"[BARCODE EXTRACTION]      {bc_type}: {count}")
        
        # Return all barcodes
        primary = all_barcodes[0] if all_barcodes else {}
        
        return {
            "success": True,
            "filename": filename,
            "is_multipage": is_pdf,
            "total_pages_processed": total_files_processed,
            "barcode_type": primary.get('type'),  # Primary for compatibility
            "barcode_data": primary.get('data'),  # Primary for compatibility
            "total_barcodes_found": len(all_barcodes),
            "all_barcodes": all_barcodes,  # ✅ ALL barcodes with page info
            "confidence": "high",
            "method": "gemini_vision",
            "token_usage": {
                "input_tokens": 258 * total_files_processed,
                "output_tokens": 100 + (len(all_barcodes) * 20),
                "total_tokens": (258 * total_files_processed) + 100 + (len(all_barcodes) * 20)
            }
        }
        
    except Exception as e:
        print(f"[BARCODE EXTRACTION] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "error": str(e),
            "filename": filename
        }


def split_image_into_tiles(image_bytes: bytes, rows: int = 3, cols: int = 2) -> List[Tuple[bytes, str]]:
    """
    Split a single page image into a grid of tiles.
    Default: 3 rows × 2 cols = 6 tiles per page.
    Each tile is sent to Gemini separately → fewer barcodes per call → fewer missed reads.
    """
    img = Image.open(io.BytesIO(image_bytes))
    w, h = img.size

    tile_w = w // cols
    tile_h = h // rows

    tiles = []
    for r in range(rows):
        for c in range(cols):
            left   = c * tile_w
            upper  = r * tile_h
            right  = left + tile_w  if c < cols - 1 else w
            lower  = upper + tile_h if r < rows - 1 else h

            tile = img.crop((left, upper, right, lower))

            if tile.mode != 'RGB':
                tile = tile.convert('RGB')

            out = io.BytesIO()
            tile.save(out, format='JPEG', quality=95, optimize=True)
            tiles.append((out.getvalue(), "image/jpeg"))

    print(f"[TILE SPLIT] Split into {len(tiles)} tiles ({rows}r × {cols}c), each ~{tile_w}×{tile_h}px")
    return tiles


def deduplicate_barcodes(barcodes: List[Dict]) -> List[Dict]:
    """
    Remove duplicate barcodes (same data value).
    Keeps first occurrence (lowest page/tile).
    """
    seen = set()
    unique = []
    for bc in barcodes:
        key = bc.get('data', '').strip()
        if key and key.lower() != 'none' and key not in seen:
            seen.add(key)
            unique.append(bc)
    return unique


def extract_barcodes_from_image_tiled(
    image_bytes: bytes,
    mime_type: str,
    page_idx: int,
    rows: int = 3,
    cols: int = 2
) -> List[Dict]:
    """
    Extract barcodes from a single page by splitting into tiles first.
    Each tile is sent to Gemini separately → more reliable extraction.
    Returns deduplicated list of barcodes found across all tiles.
    """
    from google.genai import types

    PROMPT = """
You are a barcode scanner. Find and extract ALL barcodes visible in this image tile.

Return ONLY this JSON:
{
    "barcodes": [
        {"type": "Code 128", "data": "CT270192855IN"},
        {"type": "Code 128", "data": "CT270192974IN"}
    ]
}

- If no barcodes are found, return: {"barcodes": []}
- Do NOT skip any barcode, even partial ones
- Extract the data exactly as printed
"""

    tiles = split_image_into_tiles(image_bytes, rows=rows, cols=cols)
    page_barcodes = []

    for tile_idx, (tile_bytes, tile_mime) in enumerate(tiles, 1):
        try:
            response = genai_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=tile_bytes, mime_type=tile_mime),
                    PROMPT,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )

            raw = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)

            if isinstance(data, list):
                data = {"barcodes": data}
            if not isinstance(data, dict):
                data = {"barcodes": []}

            tile_barcodes = data.get("barcodes", [])

            # Tag each barcode with page + tile info
            for bc in tile_barcodes:
                bc["page"] = page_idx
                bc["tile"] = tile_idx

            print(f"[TILE] Page {page_idx}, Tile {tile_idx}/{len(tiles)}: {len(tile_barcodes)} barcodes")
            page_barcodes.extend(tile_barcodes)

        except json.JSONDecodeError as e:
            print(f"[TILE] Page {page_idx}, Tile {tile_idx}: JSON error - {e}")
        except Exception as e:
            print(f"[TILE] Page {page_idx}, Tile {tile_idx}: Error - {e}")

    # Deduplicate (tiles may overlap slightly at edges)
    unique = deduplicate_barcodes(page_barcodes)
    print(f"[TILE] Page {page_idx}: {len(page_barcodes)} raw → {len(unique)} unique barcodes")
    return unique


def analyze_barcode_gemini_vision(file_content: bytes, filename: str) -> Dict[str, Any]:
    """
    ✅ UPDATED: Extract ALL barcodes using tiled approach for accuracy.
    Each page is split into a 3×2 grid, each tile processed separately.
    Results are deduplicated before returning.
    """
    if not USE_GEMINI:
        raise Exception("Gemini Vision not available - add GEMINI_API_KEY to .env")

    print(f"[BARCODE EXTRACTION] Starting (tiled mode): {filename}")

    try:
        from google.genai import types

        is_pdf = file_content[:4] == b'%PDF' or file_content[:5] == b'\x0a%PDF'

        if is_pdf:
            print(f"[BARCODE EXTRACTION] 📄 Multi-page PDF detected")
            images_to_process = convert_pdf_to_images(file_content)
        else:
            print(f"[BARCODE EXTRACTION] 🖼️ Single image file")
            optimized, mime_type = validate_and_optimize_image_single(file_content, filename)
            images_to_process = [(optimized, mime_type)]

        print(f"[BARCODE EXTRACTION] Processing {len(images_to_process)} page(s) with tiling...")

        all_barcodes = []
        total_pages_processed = 0

        for page_idx, (image_bytes, mime_type) in enumerate(images_to_process, 1):
            print(f"\n[BARCODE EXTRACTION] ── Page {page_idx}/{len(images_to_process)} ──")

            page_barcodes = extract_barcodes_from_image_tiled(
                image_bytes, mime_type, page_idx,
                rows=3, cols=2   # ← Tune this: more tiles = more accurate but slower
            )

            print(f"[BARCODE EXTRACTION] ✅ Page {page_idx}: {len(page_barcodes)} barcodes extracted")
            all_barcodes.extend(page_barcodes)
            total_pages_processed += 1

        # Final dedup across all pages
        all_barcodes = deduplicate_barcodes(all_barcodes)

        # Summary
        print(f"\n[BARCODE EXTRACTION] ✅ FINAL SUMMARY")
        print(f"[BARCODE EXTRACTION]   Pages Processed : {total_pages_processed}")
        print(f"[BARCODE EXTRACTION]   Total Barcodes  : {len(all_barcodes)}")

        type_summary = {}
        for bc in all_barcodes:
            t = bc.get('type', 'Unknown')
            type_summary[t] = type_summary.get(t, 0) + 1
        for t, c in sorted(type_summary.items()):
            print(f"[BARCODE EXTRACTION]     {t}: {c}")

        primary = all_barcodes[0] if all_barcodes else {}

        input_tokens  = 350 * total_pages_processed * 6   # 6 tiles per page
        output_tokens = 150 + (len(all_barcodes) * 25)

        return {
            "success": True,
            "filename": filename,
            "is_multipage": is_pdf,
            "total_pages_processed": total_pages_processed,
            "barcode_type": primary.get('type'),
            "barcode_data": primary.get('data'),
            "total_barcodes_found": len(all_barcodes),
            "all_barcodes": all_barcodes,
            "barcode_types_summary": type_summary,
            "confidence": "high",
            "method": "gemini_vision_tiled",
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
        }

    except Exception as e:
        print(f"[BARCODE EXTRACTION] ✗ Error: {e}")
        import traceback; traceback.print_exc()
        return {"success": False, "error": str(e), "filename": filename}
