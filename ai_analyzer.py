"""
Enhanced AI Analyzer Module - FIXED WITH IMAGE VALIDATION + PDF SUPPORT + MULTI-BILL IMAGE
- Proper image validation before sending to Gemini
- Automatic image optimization (resize, compress)
- PDF to image conversion support
- Better MIME type detection
- Comprehensive error handling
- Multi-bill extraction from single images (side-by-side/stacked receipts)
"""
import json
import re
from typing import Dict, Any, List, Tuple
import os
import time
from dotenv import load_dotenv
from PIL import Image
import io

# ✅ Register HEIC support at module load
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
GEMINI_MODEL = 'gemini-2.0-flash'

print(f"[AI Config] Selected Model: {OLLAMA_MODEL if OLLAMA_AVAILABLE else GEMINI_MODEL if USE_GEMINI else 'OpenAI GPT' if USE_OPENAI else 'None'}")


def validate_and_optimize_image(file_content: bytes, filename: str) -> Tuple[bytes, str]:
    """
    Validate and optimize image for Gemini Vision
    Returns: (optimized_bytes, mime_type)
    """
    print(f"[IMAGE VALIDATION] Validating: {filename} ({len(file_content):,} bytes)")

    is_pdf = file_content[:4] == b'%PDF' or file_content[:5] == b'\x0a%PDF'
    is_heic = len(file_content) > 12 and b'ftyp' in file_content[:20] and (b'heic' in file_content[:20] or b'heif' in file_content[:20])

    if is_pdf:
        print(f"[IMAGE VALIDATION]   📄 Detected PDF file")
        if not PDF_SUPPORT:
            raise Exception(f"PDF file detected but PDF support not installed.")
        try:
            print(f"[IMAGE VALIDATION]   Converting PDF to image...")
            images = convert_from_bytes(file_content, dpi=200, first_page=1, last_page=1)
            if not images:
                raise Exception("Failed to convert PDF - no pages found")
            img = images[0]
            print(f"[IMAGE VALIDATION]   ✓ PDF converted: {img.size} ({img.mode})")
            output = io.BytesIO()
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img.save(output, format='JPEG', quality=90, optimize=True)
            file_content = output.getvalue()
            print(f"[IMAGE VALIDATION]   ✓ Converted to JPEG: {len(file_content):,} bytes")
            img = Image.open(io.BytesIO(file_content))
        except Exception as e:
            print(f"[IMAGE VALIDATION]   ✗ PDF conversion failed: {e}")
            raise Exception(f"Failed to convert PDF to image: {e}")

    elif is_heic:
        print(f"[IMAGE VALIDATION]   📱 Detected HEIC file")
        if not HEIC_SUPPORT:
            raise Exception(f"HEIC file detected but pillow-heif not installed.")
        try:
            print(f"[IMAGE VALIDATION]   Converting HEIC to JPEG...")
            img = Image.open(io.BytesIO(file_content))
            print(f"[IMAGE VALIDATION]   ✓ HEIC opened: {img.size} ({img.mode})")
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
                img = background
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=95, optimize=True)
            file_content = output.getvalue()
            print(f"[IMAGE VALIDATION]   ✓ Converted to JPEG: {len(file_content):,} bytes")
            img = Image.open(io.BytesIO(file_content))
        except Exception as e:
            print(f"[IMAGE VALIDATION]   ✗ HEIC conversion failed: {e}")
            raise Exception(f"Failed to convert HEIC to JPEG: {e}")

    try:
        if not is_pdf and not is_heic:
            img = Image.open(io.BytesIO(file_content))

        original_format = img.format
        original_size = img.size
        original_mode = img.mode

        print(f"[IMAGE VALIDATION]   Format: {original_format}, Size: {original_size}, Mode: {original_mode}")

        MAX_DIMENSION = 3072
        MAX_FILE_SIZE_MB = 4
        TARGET_FILE_SIZE_MB = 2
        needs_optimization = False

        if max(original_size) > MAX_DIMENSION:
            print(f"[IMAGE VALIDATION]   ⚠️ Image too large, will resize")
            needs_optimization = True

        size_mb = len(file_content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            print(f"[IMAGE VALIDATION]   ⚠️ File size {size_mb:.2f}MB too large, will compress")
            needs_optimization = True

        if img.mode in ('RGBA', 'LA', 'P'):
            print(f"[IMAGE VALIDATION]   Converting {img.mode} to RGB")
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
            needs_optimization = True

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
                output_size_mb = output.tell() / (1024 * 1024)
                if output_size_mb <= TARGET_FILE_SIZE_MB or quality <= 60:
                    break
                quality -= 5

            optimized_bytes = output.getvalue()
            mime_type = "image/jpeg"
            print(f"[IMAGE VALIDATION]   ✓ Optimized: {len(optimized_bytes):,} bytes (quality: {quality})")
            return optimized_bytes, mime_type
        else:
            if original_format == 'PNG':
                mime_type = "image/png"
            elif original_format in ('JPEG', 'JPG'):
                mime_type = "image/jpeg"
            elif original_format == 'WEBP':
                mime_type = "image/webp"
            else:
                output = io.BytesIO()
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img.save(output, format='JPEG', quality=90, optimize=True)
                return output.getvalue(), "image/jpeg"

            print(f"[IMAGE VALIDATION]   ✓ Image is valid, no optimization needed")
            return file_content, mime_type

    except Exception as e:
        if is_pdf or is_heic:
            raise

        print(f"[IMAGE VALIDATION]   ✗ Failed to validate image: {e}")

        is_jpeg = file_content[:2] == b'\xff\xd8'
        is_png = file_content[:4] == b'\x89PNG'
        is_webp = len(file_content) > 12 and file_content[8:12] == b'WEBP'

        if not any([is_jpeg, is_png, is_webp, is_heic]):
            print(f"[IMAGE VALIDATION]   ✗ File is NOT a valid image!")
            print(f"[IMAGE VALIDATION]   First 20 bytes: {file_content[:20].hex()}")
            try:
                text_preview = file_content[:200].decode('utf-8', errors='ignore')
                if '<html' in text_preview.lower() or '<!doctype' in text_preview.lower():
                    raise Exception(f"Downloaded file is HTML error page, not an image")
                elif '{' in text_preview and ('"error"' in text_preview or '"message"' in text_preview):
                    raise Exception(f"Downloaded file is JSON error, not an image")
            except UnicodeDecodeError:
                pass
            raise Exception(f"File is not a valid image format. Cannot process.")

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
    """Analyze bank passbook using Gemini Vision with image validation"""
    if not USE_GEMINI:
        raise Exception("Gemini Vision not available - add GEMINI_API_KEY to .env")

    print(f"[GEMINI VISION] Analyzing bank passbook: {filename}")

    try:
        from google.genai import types

        optimized_content, mime_type = validate_and_optimize_image(file_content, filename)
        print(f"[GEMINI VISION] Sending to Gemini: {mime_type}, {len(optimized_content):,} bytes")

        prompt = """
Analyze this Indian bank passbook image.

STEP 1 - TRANSCRIBE THE ACCOUNT NUMBER:
Write each digit of the account number with its position number:
Position 1: [digit]
Position 2: [digit]
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

        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=optimized_content, mime_type=mime_type),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )

        response_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(response_text)

                # ✅ FIX: Handle case where Gemini returns a list instead of dict
        if isinstance(data, list):
            print(f"[GEMINI VISION] ⚠️ Received array response, taking first element")
            data = data[0] if data else {}

        if not isinstance(data, dict):
            data = {}
        
        print(f"[GEMINI VISION] ✓ Extracted bank details")
        
        # Handle case where Gemini returns a list instead of dict
        if isinstance(data, list):
            print(f"[GEMINI VISION] ⚠️ Received list with {len(data)} items")
            if len(data) > 0 and isinstance(data[0], dict):
                # Take the first item if it's a list of dicts
                data = data[0]
                print(f"[GEMINI VISION] ⚠️ Using first item from list")
            else:
                raise ValueError(f"Unexpected response format: list with {len(data)} items")
        
        if not isinstance(data, dict):
            raise ValueError(f"Unexpected response type: {type(data)}")

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
    """Analyze bill using Gemini Vision with image validation"""
    if not USE_GEMINI:
        raise Exception("Gemini Vision not available")

    print(f"[GEMINI VISION] Analyzing bill: {filename}")

    try:
        from google.genai import types

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

        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=optimized_content, mime_type=mime_type),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )

        response_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(response_text)

        if isinstance(data, list):
            print(f"[GEMINI VISION] ⚠️ Received array response, taking first element")
            data = data[0] if len(data) > 0 else {}

        if not isinstance(data, dict):
            data = {
                "student_name": None, "college_name": None,
                "roll_number": None, "receipt_number": None,
                "class_course": None, "bill_date": None, "amount": None
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


def analyze_bill_multi_page(file_content: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extract bills from a PDF that may contain MULTIPLE bills per page
    (side-by-side / stacked receipts in a single scanned image).
    Returns a flat list of bill dicts.
    """
    if not USE_GEMINI:
        raise Exception("Gemini Vision not available - add GEMINI_API_KEY to .env")

    print(f"[BILL MULTI-PAGE] Starting: {filename}")

    try:
        from google.genai import types

        is_pdf = (
            file_content[:4] == b'%PDF'
            or file_content[:5] == b'\x0a%PDF'
        )

        if is_pdf:
            print(f"[BILL MULTI-PAGE] 📄 PDF detected – converting all pages")
            if not PDF_SUPPORT:
                raise Exception("PDF support not installed.")
            images_to_process = convert_pdf_to_images(file_content)
            print(f"[BILL MULTI-PAGE] {len(images_to_process)} page(s) extracted from PDF")
        else:
            print(f"[BILL MULTI-PAGE] 🖼️ Single image – treating as one page")
            optimized, mime_type = validate_and_optimize_image_single(file_content, filename)
            images_to_process = [(optimized, mime_type)]

        # ── MULTI-BILL prompt: always returns a JSON *array* ──────────────
        MULTI_BILL_PROMPT = """
You are analysing a fee receipt / college bill image that MAY contain
ONE or MORE separate receipts in the same photo (side-by-side, stacked,
or as a collage).

STEP 1 – COUNT THE RECEIPTS
Look carefully: how many distinct fee receipts / bills are visible?

STEP 2 – EXTRACT EACH RECEIPT SEPARATELY
For every receipt you found, extract:
  • student_name   – from the "Name:" field (may be handwritten)
  • college_name   – usually large text at the top of the receipt
  • roll_number    – Reg No / Roll No / Admission No
  • receipt_number – Receipt No / Challan No / Bill No
  • class_course   – Class, Section, or Course name
  • bill_date      – Date in YYYY-MM-DD format (convert if needed)
  • amount         – The TOTAL amount (number only, e.g. 18000)

RULES:
- For amounts like "18000-00" or "18,000/-", extract as 18000
- student_name is in the Name field, NOT the signature
- If a field is unclear or absent, use null
- Return EXACTLY as many objects as receipts found (even if only 1)
- Do NOT merge two receipts into one

Return ONLY a JSON array (no surrounding text):
[
  {
    "bill_index": 1,
    "student_name": "...",
    "college_name": "...",
    "roll_number": "...",
    "receipt_number": "...",
    "class_course": "...",
    "bill_date": "YYYY-MM-DD",
    "amount": 18000.00
  }
]
"""

        bills: List[Dict[str, Any]] = []
        skipped = 0

        for page_idx, (image_bytes, mime_type) in enumerate(images_to_process, 1):
            print(f"[BILL MULTI-PAGE] [{page_idx}/{len(images_to_process)}] Scanning page for bills…")

            try:
                response = genai_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                        MULTI_BILL_PROMPT,
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )

                response_text = response.text.replace("```json", "").replace("```", "").strip()
                data = json.loads(response_text)

                # Normalise: always expect a list
                if isinstance(data, dict):
                    data = [data]
                if not isinstance(data, list):
                    data = []

                page_bills_added = 0
                for item in data:
                    if not isinstance(item, dict):
                        continue

                    # Skip blank pages
                    if item.get("amount") is None and item.get("college_name") is None:
                        continue

                    item["page_number"] = page_idx
                    if "bill_index" not in item:
                        item["bill_index"] = page_bills_added + 1

                    bills.append(item)
                    page_bills_added += 1

                if page_bills_added == 0:
                    print(f"[BILL MULTI-PAGE]   ⚠️ Page {page_idx} appears blank – skipping")
                    skipped += 1
                else:
                    print(f"[BILL MULTI-PAGE]   ✅ Page {page_idx}: {page_bills_added} bill(s) found")
                    for b in data[:page_bills_added]:
                        print(f"[BILL MULTI-PAGE]      Bill {b.get('bill_index')}: "
                              f"College={b.get('college_name')} | "
                              f"Student={b.get('student_name')} | "
                              f"Amount=₹{b.get('amount')}")

            except json.JSONDecodeError as e:
                print(f"[BILL MULTI-PAGE]   ✗ Page {page_idx}: JSON parse error – {e}")
            except Exception as e:
                print(f"[BILL MULTI-PAGE]   ✗ Page {page_idx}: {e}")
                import traceback; traceback.print_exc()

        print(f"\n[BILL MULTI-PAGE] Done – {len(bills)} bill(s) extracted ({skipped} page(s) skipped)")
        return bills

    except Exception as e:
        print(f"[BILL MULTI-PAGE] ✗ Fatal error: {e}")
        import traceback; traceback.print_exc()
        raise

def analyze_bill_multi_bills_from_image(file_content: bytes, filename: str) -> List[Dict[str, Any]]:
    """
    Extract ALL bills from a single image (JPG/PNG) that may contain
    multiple fee receipts side-by-side or stacked in one photo.

    - If only 1 bill found  → returns list with 1 item
    - If 2+ bills found     → returns list with N items
    - Each item has page_number = 1 (single image, not a PDF)

    Shape of each dict:
    {
        "student_name": ...,
        "college_name": ...,
        "roll_number": ...,
        "receipt_number": ...,
        "class_course": ...,
        "bill_date": ...,
        "amount": 18000.00,
        "page_number": 1,
        "bill_index": 1   # position within this image (1, 2, 3…)
    }
    """
    if not USE_GEMINI:
        raise Exception("Gemini Vision not available - add GEMINI_API_KEY to .env")

    print(f"[MULTI-BILL IMAGE] Scanning for multiple bills: {filename}")

    try:
        from google.genai import types

        optimised_content, mime_type = validate_and_optimize_image_single(file_content, filename)
        print(f"[MULTI-BILL IMAGE] Sending to Gemini: {mime_type}, {len(optimised_content):,} bytes")

        prompt = """
You are analysing a fee receipt / college bill image that MAY contain
ONE or MORE separate receipts in the same photo (side-by-side, stacked,
or as a collage).

STEP 1 – COUNT THE RECEIPTS
Look carefully: how many distinct fee receipts / bills are visible?

STEP 2 – EXTRACT EACH RECEIPT SEPARATELY
For every receipt you found, extract:
  • student_name   – from the "Name:" field (may be handwritten)
  • college_name   – usually large text at the top of the receipt
  • roll_number    – Reg No / Roll No / Admission No
  • receipt_number – Receipt No / Challan No / Bill No
  • class_course   – Class, Section, or Course name
  • bill_date      – Date in YYYY-MM-DD format (convert if needed)
  • amount         – The TOTAL amount (number only, e.g. 18000)

RULES:
- For amounts like "18000-00" or "18,000/-", extract as 18000
- student_name is in the Name field, NOT the signature
- If a field is unclear or absent, use null
- Return EXACTLY as many objects as receipts found
- Do NOT merge two receipts into one

Return ONLY a JSON array (no surrounding text):
[
  {
    "bill_index": 1,
    "student_name": "...",
    "college_name": "...",
    "roll_number": "...",
    "receipt_number": "...",
    "class_course": "...",
    "bill_date": "YYYY-MM-DD",
    "amount": 18000.00
  },
  {
    "bill_index": 2,
    "student_name": "...",
    "college_name": "...",
    "roll_number": "...",
    "receipt_number": "...",
    "class_course": "...",
    "bill_date": "YYYY-MM-DD",
    "amount": 26000.00
  }
]
"""

        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=optimised_content, mime_type=mime_type),
                prompt,
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json",
            ),
        )

        response_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(response_text)

        # Normalise to list
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            data = []

        bills = []
        for item in data:
            if not isinstance(item, dict):
                continue
            # Ensure bill_index is set
            if 'bill_index' not in item:
                item['bill_index'] = len(bills) + 1
            item["page_number"] = 1
            bills.append(item)

        print(f"[MULTI-BILL IMAGE] Found {len(bills)} bill(s) in image:")
        for b in bills:
            print(f"[MULTI-BILL IMAGE]   Bill {b.get('bill_index', '?')}: "
                  f"{b.get('college_name')} | "
                  f"{b.get('student_name')} | "
                  f"₹{b.get('amount')}")

        return bills

    except json.JSONDecodeError as e:
        print(f"[MULTI-BILL IMAGE] ✗ JSON parse error: {e}")
        raise
    except Exception as e:
        print(f"[MULTI-BILL IMAGE] ✗ Error: {e}")
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

    college_match = re.search(r'([A-Z][A-Za-z\s]+COLLEGE)', markdown_text, re.IGNORECASE)
    if college_match:
        college_name = college_match.group(1).strip()
        print(f"[MANUAL EXTRACTION]   ✓ Found college: {college_name}")

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
            'bank_name': None, 'account_holder_name': None,
            'account_number': None, 'ifsc_code': None, 'branch_name': None
        }


def analyze_bill_from_markdown(markdown_text: str) -> Dict[str, Any]:
    """Analyze bill from OCR markdown"""
    try:
        return extract_bill_details_from_markdown(markdown_text)
    except Exception as e:
        print(f"[ANALYSIS] ✗ Error: {e}")
        return {
            'student_name': None, 'college_name': None,
            'roll_number': None, 'bill_date': None, 'amount': None
        }


def analyze_generic_gemini_vision(file_content: bytes, filename: str, user_prompt: str) -> Dict[str, Any]:
    """Generic document analysis using Gemini Vision with image validation"""
    if not USE_GEMINI:
        raise Exception("Gemini Vision not available - add GEMINI_API_KEY to .env")

    print(f"[GEMINI VISION] Analyzing document: {filename}")

    try:
        from google.genai import types

        optimized_content, mime_type = validate_and_optimize_image(file_content, filename)

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

Return your response as a valid JSON object with the fields the user requested.
"""

        response = genai_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=optimized_content, mime_type=mime_type),
                full_prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                response_mime_type="application/json"
            )
        )

        response_text = response.text.replace("```json", "").replace("```", "").strip()
        data = json.loads(response_text)

        if isinstance(data, str):
            data = {"extracted_value": data}
        elif isinstance(data, list):
            if len(data) > 0:
                data = data[0] if isinstance(data[0], dict) else {"extracted_values": data}
            else:
                data = {"extracted_values": []}
        elif isinstance(data, (int, float, bool)):
            data = {"extracted_value": data}
        elif not isinstance(data, dict):
            data = {"extracted_value": str(data)}

        if not isinstance(data, dict):
            data = {"error": "Invalid response format"}

        print(f"[GEMINI VISION] ✓ Extraction completed, {len(data)} fields")
        return data

    except json.JSONDecodeError as e:
        return {"error": "Failed to parse JSON response"}
    except Exception as e:
        print(f"[GEMINI VISION] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        raise


def convert_pdf_to_images(file_content: bytes) -> List[Tuple[bytes, str]]:
    """Convert ALL pages of PDF to images. Returns list of (image_bytes, mime_type)."""
    print(f"[PDF CONVERTER] Processing multi-page PDF...")

    if not PDF_SUPPORT:
        raise Exception("PDF support not installed.")

    try:
        images = convert_from_bytes(file_content, dpi=200)
        if not images:
            raise Exception("Failed to convert PDF - no pages found")

        print(f"[PDF CONVERTER] ✓ Converted {len(images)} pages from PDF")

        image_list = []
        for idx, img in enumerate(images, 1):
            print(f"[PDF CONVERTER]   Page {idx}: {img.size} ({img.mode})")
            if img.mode != 'RGB':
                img = img.convert('RGB')
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
    """Validate and optimize a single image with HEIC support."""
    print(f"[IMAGE VALIDATION] Validating: {filename} ({len(file_content):,} bytes)")

    try:
        is_heic = len(file_content) > 12 and b'ftyp' in file_content[:20] and (b'heic' in file_content[:20] or b'heif' in file_content[:20])

        if is_heic:
            print(f"[IMAGE VALIDATION]   📱 Detected HEIC - converting to JPEG...")
            if not HEIC_SUPPORT:
                raise Exception("HEIC support not available")
            img = Image.open(io.BytesIO(file_content))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            output = io.BytesIO()
            img.save(output, format='JPEG', quality=95, optimize=True)
            file_content = output.getvalue()
            print(f"[IMAGE VALIDATION]   ✓ HEIC converted to JPEG: {len(file_content):,} bytes")

        img = Image.open(io.BytesIO(file_content))
        original_format = img.format
        original_size = img.size
        original_mode = img.mode

        print(f"[IMAGE VALIDATION]   Format: {original_format}, Size: {original_size}, Mode: {original_mode}")

        MAX_DIMENSION = 3072
        MAX_FILE_SIZE_MB = 4
        TARGET_FILE_SIZE_MB = 2
        needs_optimization = False

        if max(original_size) > MAX_DIMENSION:
            needs_optimization = True

        size_mb = len(file_content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            needs_optimization = True

        if img.mode in ('RGBA', 'LA', 'P'):
            background = Image.new('RGB', img.size, (255, 255, 255))
            if img.mode == 'P':
                img = img.convert('RGBA')
            background.paste(img, mask=img.split()[-1] if img.mode in ('RGBA', 'LA') else None)
            img = background
            needs_optimization = True

        if needs_optimization:
            if max(original_size) > MAX_DIMENSION:
                ratio = MAX_DIMENSION / max(original_size)
                new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
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

            return output.getvalue(), "image/jpeg"
        else:
            if original_format == 'PNG':
                mime_type = "image/png"
            elif original_format in ('JPEG', 'JPG'):
                mime_type = "image/jpeg"
            elif original_format == 'WEBP':
                mime_type = "image/webp"
            else:
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


def split_image_into_strips(image_bytes: bytes, num_strips: int = 3) -> list:
    """Split image horizontally into N strips for better barcode detection."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size
        strip_height = height // num_strips
        strips = []

        for i in range(num_strips):
            top = i * strip_height
            bottom = height if i == num_strips - 1 else (i + 1) * strip_height
            strip = img.crop((0, top, width, bottom))

            output = io.BytesIO()
            if strip.mode != 'RGB':
                strip = strip.convert('RGB')
            strip.save(output, format='JPEG', quality=95, optimize=True)
            strips.append((output.getvalue(), "image/jpeg"))
            print(f"[STRIP] Strip {i+1}/{num_strips}: y={top}-{bottom}, size={len(output.getvalue()):,} bytes")

        return strips
    except Exception as e:
        print(f"[STRIP] ✗ Failed to split image: {e}")
        return [(image_bytes, "image/jpeg")]


def extract_barcodes_from_image_bytes(image_bytes: bytes, mime_type: str, page_idx: int, strip_idx: int = None) -> list:
    """
    Core barcode extraction from a single image/strip.
    Returns list of valid barcode dicts.
    """
    from google.genai import types

    label = f"Page {page_idx}" + (f" Strip {strip_idx}" if strip_idx else "")

    BARCODE_PROMPT = """
You are a precise barcode scanner. This image contains barcode labels (Indian Speed Post tracking barcodes).

YOUR TASK:
1. Scan the image systematically: LEFT to RIGHT, TOP to BOTTOM
2. Extract EVERY single barcode text printed below each barcode graphic
3. Each barcode label has text like "CT270XXXXXXIN" printed below the black bars
4. Extract ALL of them without stopping early

EXTRACTION RULES:
- Read the printed text BELOW each barcode (not the bars themselves)
- Format is always: CT + 9 digits + IN (e.g. CT270187815IN)
- Do NOT skip any barcode, do NOT stop early
- Scan every row and every column

Return ONLY this JSON:
{
    "barcodes": [
        {"type": "Code 128", "data": "CT270187815IN"},
        {"type": "Code 128", "data": "CT270187930IN"}
    ]
}
"""

    max_attempts = 3
    best_result = []

    for attempt in range(1, max_attempts + 1):
        try:
            response = genai_client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    BARCODE_PROMPT,
                ],
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    max_output_tokens=8192,
                )
            )

            response_text = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(response_text)

            if isinstance(data, list):
                data = {"barcodes": data}
            elif not isinstance(data, dict):
                data = {"barcodes": []}

            valid_barcodes = [
                bc for bc in data.get('barcodes', [])
                if bc.get('data') and str(bc.get('data')).strip() not in ('', 'None', 'null')
            ]

            extracted_count = len(valid_barcodes)
            print(f"[BARCODE EXTRACTION]   {label} | Attempt {attempt}: extracted={extracted_count}")

            # Keep best result across attempts
            if extracted_count > len(best_result):
                best_result = valid_barcodes

            # If we got a good count, stop retrying
            if extracted_count >= 10:
                break

            if attempt < max_attempts:
                print(f"[BARCODE EXTRACTION]   ⚠️ Low count, retrying {label}...")
                time.sleep(1)

        except json.JSONDecodeError as e:
            print(f"[BARCODE EXTRACTION]   ⚠️ {label} Attempt {attempt} JSON error: {e}")
            if attempt < max_attempts:
                time.sleep(1)
        except Exception as e:
            print(f"[BARCODE EXTRACTION]   ⚠️ {label} Attempt {attempt} error: {e}")
            if attempt < max_attempts:
                time.sleep(1)

    return best_result


def analyze_barcode_gemini_vision(file_content: bytes, filename: str) -> Dict[str, Any]:
    """Extract ALL barcodes from single images and multi-page PDFs using strip-based extraction."""
    if not USE_GEMINI:
        raise Exception("Gemini Vision not available - add GEMINI_API_KEY to .env")

    print(f"[BARCODE EXTRACTION] Starting: {filename}")

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

        print(f"[BARCODE EXTRACTION] Processing {len(images_to_process)} image(s)...")

        all_barcodes = []
        total_pages_processed = 0

        for page_idx, (image_bytes, mime_type) in enumerate(images_to_process, 1):
            print(f"\n[BARCODE EXTRACTION] [{page_idx}/{len(images_to_process)}] Processing page/image...")

            # ── Step 1: Try full page first ───────────────────────────────────
            print(f"[BARCODE EXTRACTION]   🔍 Step 1: Full page extraction...")
            full_page_barcodes = extract_barcodes_from_image_bytes(
                image_bytes, mime_type, page_idx
            )
            print(f"[BARCODE EXTRACTION]   Full page result: {len(full_page_barcodes)} barcodes")

            # ── Step 2: Always do strip-based extraction and merge ────────────
            # Split into 3 horizontal strips (top / middle / bottom rows)
            print(f"[BARCODE EXTRACTION]   🔍 Step 2: Strip-based extraction (3 strips)...")
            strips = split_image_into_strips(image_bytes, num_strips=3)

            strip_barcodes = []
            for strip_idx, (strip_bytes, strip_mime) in enumerate(strips, 1):
                print(f"[BARCODE EXTRACTION]   Processing strip {strip_idx}/3...")
                result = extract_barcodes_from_image_bytes(
                    strip_bytes, strip_mime, page_idx, strip_idx=strip_idx
                )
                print(f"[BARCODE EXTRACTION]   Strip {strip_idx}: {len(result)} barcodes")
                strip_barcodes.extend(result)

            print(f"[BARCODE EXTRACTION]   Strip total (before dedup): {len(strip_barcodes)} barcodes")

            # ── Step 3: Merge full page + strip results ───────────────────────
            combined = full_page_barcodes + strip_barcodes

            # ── Step 4: Deduplicate by barcode data ───────────────────────────
            seen_data = set()
            deduped = []
            for bc in combined:
                d = str(bc.get('data', '')).strip()
                if d and d not in seen_data:
                    seen_data.add(d)
                    bc['page'] = page_idx
                    deduped.append(bc)

            # ── Step 5: If still low, try 6-strip extraction ──────────────────
            if len(deduped) < 30:
                print(f"[BARCODE EXTRACTION]   ⚠️ Only {len(deduped)} found, trying 6-strip extraction...")
                strips_6 = split_image_into_strips(image_bytes, num_strips=6)

                for strip_idx, (strip_bytes, strip_mime) in enumerate(strips_6, 1):
                    result = extract_barcodes_from_image_bytes(
                        strip_bytes, strip_mime, page_idx, strip_idx=f"6s_{strip_idx}"
                    )
                    for bc in result:
                        d = str(bc.get('data', '')).strip()
                        if d and d not in seen_data:
                            seen_data.add(d)
                            bc['page'] = page_idx
                            deduped.append(bc)

                print(f"[BARCODE EXTRACTION]   After 6-strip: {len(deduped)} barcodes")

            # ── Step 6: Sort barcodes naturally ───────────────────────────────
            deduped.sort(key=lambda x: str(x.get('data', '')))

            print(f"[BARCODE EXTRACTION]   ✅ Page {page_idx}: Final count = {len(deduped)} barcode(s)")
            for i, bc in enumerate(deduped[:3], 1):
                print(f"[BARCODE EXTRACTION]      [{i}] {bc.get('type')}: {bc.get('data')}")
            if len(deduped) > 3:
                print(f"[BARCODE EXTRACTION]      ... and {len(deduped) - 3} more")

            all_barcodes.extend(deduped)
            total_pages_processed += 1

            # Delay between pages to avoid rate limiting
            if page_idx < len(images_to_process):
                time.sleep(0.5)

        # ── Global dedup across all pages ─────────────────────────────────────
        seen_global = set()
        final_barcodes = []
        for bc in all_barcodes:
            d = str(bc.get('data', '')).strip()
            if d and d not in seen_global:
                seen_global.add(d)
                final_barcodes.append(bc)

        if len(final_barcodes) < len(all_barcodes):
            print(f"[BARCODE EXTRACTION] ℹ️ Removed {len(all_barcodes) - len(final_barcodes)} duplicates across pages")

        type_summary = {}
        for bc in final_barcodes:
            bc_type = bc.get('type', 'Unknown')
            type_summary[bc_type] = type_summary.get(bc_type, 0) + 1

        primary = final_barcodes[0] if final_barcodes else {}
        input_tokens = 350 * total_pages_processed
        output_tokens = 150 + (len(final_barcodes) * 25)

        print(f"\n[BARCODE EXTRACTION] ✅ COMPLETE: {len(final_barcodes)} total barcodes from {total_pages_processed} pages")

        return {
            "success": True,
            "filename": filename,
            "is_multipage": is_pdf,
            "total_pages_processed": total_pages_processed,
            "barcode_type": primary.get('type'),
            "barcode_data": primary.get('data'),
            "total_barcodes_found": len(final_barcodes),
            "all_barcodes": final_barcodes,
            "barcode_types_summary": type_summary,
            "confidence": "high",
            "method": "gemini_vision_tiled",
            "token_usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens
            }
        }

    except Exception as e:
        print(f"[BARCODE EXTRACTION] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e), "filename": filename}