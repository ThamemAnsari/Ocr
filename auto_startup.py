"""
auto_startup.py
───────────────
Automatically starts an extraction job on backend startup and pushes to Zoho.

Target:
  App   : teameverest / iatc-scholarship
  Report: Scholar_Fee_Request_OCR_View
  Form  : OCR_Extraction_From
"""

import os
import time
import threading
import requests
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────
AUTO_EXTRACT_CONFIG = {
    "app_link_name":    "teameverest/iatc-scholarship",
    "report_link_name": "Scholar_Fee_Request_OCR_View",

    # Zoho Creator criteria syntax - filter by OCR_Status
    # Leave empty to fetch all records (filtering will be done in code)
    "filter_criteria": "",  # Fetch all records, filter in application

    # Field names in the Zoho report (verified from actual data)
    "bank_field_name":  "Soft_Copy_Passbook_Multi_Line",
    "bill_field_name":  "Original_Paid_Bill_or_Online_Bill",

    # How long (seconds) to wait after FastAPI boots before triggering
    "startup_delay_seconds": 5,
    
    # Auto-push to Zoho after extraction
    "auto_push_to_zoho": True,
    "zoho_owner_name": "teameverest",
    "zoho_app_name": "iatc-scholarship",
    "zoho_form_name": "OCR_Extraction_From",
}

# Internal API base (same process – call ourselves via localhost)
_API_BASE = os.getenv("VITE_API_URL", "http://localhost:8000")

# ── State shared with the frontend polling endpoint ──────────────────────────
startup_job_state: dict = {
    "triggered": False,
    "job_id": None,
    "status": "idle",          # idle | starting | running | completed | failed
    "message": "",
    "triggered_at": None,
    "records_found": 0,
}

def _trigger_auto_extract():
    """Called once in a background thread shortly after startup."""
    global startup_job_state

    delay = AUTO_EXTRACT_CONFIG["startup_delay_seconds"]
    print(f"\n[AUTO-STARTUP] Waiting {delay}s before triggering auto-extract...")
    time.sleep(delay)

    startup_job_state["status"] = "starting"
    startup_job_state["triggered_at"] = datetime.now().isoformat()
    startup_job_state["message"] = "Starting extraction job..."

    try:
        start_resp = requests.post(
            f"{_API_BASE}/ocr/auto-extract/start",
            data={
                "app_link_name":    AUTO_EXTRACT_CONFIG["app_link_name"],
                "report_link_name": AUTO_EXTRACT_CONFIG["report_link_name"],
                "bank_field_name":  AUTO_EXTRACT_CONFIG["bank_field_name"],
                "bill_field_name":  AUTO_EXTRACT_CONFIG["bill_field_name"] or "",
                "filter_criteria":  AUTO_EXTRACT_CONFIG["filter_criteria"],
            },
            timeout=60,
        )
        start_data = start_resp.json()

        if start_data.get("success"):
            job_id = start_data["job_id"]
            startup_job_state["job_id"]    = job_id
            startup_job_state["status"]    = "running"
            startup_job_state["triggered"] = True
            startup_job_state["message"]   = f"🚀 Job {job_id} started"
            print(f"[AUTO-STARTUP] ✅ Job started: {job_id}")
            _poll_job_until_done(job_id)

        else:
            err = start_data.get("error", "Unknown error")
            if "already been extracted" in err or "All selected" in err:
                startup_job_state["status"]  = "completed"
                startup_job_state["message"] = "✅ All records already processed"
                print(f"[AUTO-STARTUP] All records already extracted — skipping")
            else:
                startup_job_state["status"]  = "failed"
                startup_job_state["message"] = f"❌ {err}"
                print(f"[AUTO-STARTUP] ✗ {err}")

    except Exception as exc:
        startup_job_state["status"]  = "failed"
        startup_job_state["message"] = f"❌ {exc}"
        print(f"[AUTO-STARTUP] ✗ Exception: {exc}")
        import traceback; traceback.print_exc()


def _poll_job_until_done(job_id: str):
    """Keeps polling /ocr/auto-extract/status/{job_id} until terminal state."""
    global startup_job_state

    poll_interval = 10   # seconds between polls
    max_polls     = 360  # give up after 1 hour

    for _ in range(max_polls):
        time.sleep(poll_interval)
        try:
            resp = requests.get(
                f"{_API_BASE}/ocr/auto-extract/status/{job_id}", timeout=15
            )
            data = resp.json()
            job_status = data.get("status", "unknown")
            progress   = data.get("progress", {})

            startup_job_state["message"] = (
                f"Processing: {progress.get('processed_records', 0)}"
                f"/{progress.get('total_records', '?')} records"
            )

            if job_status in ("completed", "failed"):
                startup_job_state["status"] = job_status
                if job_status == "completed":
                    ok  = progress.get("successful_records", 0)
                    bad = progress.get("failed_records", 0)
                    startup_job_state["message"] = (
                        f"✅ Done — {ok} succeeded, {bad} failed"
                    )
                    print(f"[AUTO-STARTUP] Job {job_id} completed: "
                          f"{ok} ok / {bad} failed")
                    
                    # Auto-push to Zoho if enabled
                    if AUTO_EXTRACT_CONFIG.get("auto_push_to_zoho"):
                        print(f"[AUTO-STARTUP] Starting auto-push to Zoho...")
                        _push_to_zoho()
                else:
                    startup_job_state["message"] = f"❌ Job {job_id} failed"
                    print(f"[AUTO-STARTUP] Job {job_id} FAILED")
                return

        except Exception as poll_exc:
            print(f"[AUTO-STARTUP] Poll error: {poll_exc}")

    # Timed out
    startup_job_state["status"]  = "failed"
    startup_job_state["message"] = "⏰ Polling timed out after 1 hour"
    print("[AUTO-STARTUP] Polling timed out")


def _push_to_zoho():
    """Push extracted records to Zoho Creator form."""
    try:
        print(f"[AUTO-STARTUP] 📤 Pushing to Zoho...")
        
        push_resp = requests.post(
            f"{_API_BASE}/sync/bulk-push-to-zoho",
            params={
                "limit": 1000,
                "today_only": True,
                "retry_failed": False
            },
            timeout=300,  # 5 minutes
        )
        
        push_data = push_resp.json()
        
        if push_data.get("success"):
            details = push_data.get("details", {})
            successful = details.get("successful", 0)
            failed = details.get("failed", 0)
            print(f"[AUTO-STARTUP] ✅ Zoho push completed: {successful} succeeded, {failed} failed")
            startup_job_state["message"] += f" | Pushed to Zoho: {successful} ok, {failed} failed"
        else:
            error = push_data.get("error", "Unknown error")
            print(f"[AUTO-STARTUP] ❌ Zoho push failed: {error}")
            startup_job_state["message"] += f" | Zoho push failed: {error}"
            
    except Exception as push_exc:
        print(f"[AUTO-STARTUP] ❌ Zoho push exception: {push_exc}")
        startup_job_state["message"] += f" | Zoho push error: {push_exc}"



def register_startup_hook(app):
    """
    Call this in main.py / app.py after creating the FastAPI `app`.

    Usage:
        from auto_startup import register_startup_hook
        register_startup_hook(app)
    """
    from fastapi import FastAPI

    @app.on_event("startup")
    async def _on_startup():
        print("\n" + "="*60)
        print("[AUTO-STARTUP] Backend started — scheduling auto-extract")
        print(f"  App    : {AUTO_EXTRACT_CONFIG['app_link_name']}")
        print(f"  Report : {AUTO_EXTRACT_CONFIG['report_link_name']}")
        print(f"  Bank   : {AUTO_EXTRACT_CONFIG['bank_field_name']}")
        print(f"  Bill   : {AUTO_EXTRACT_CONFIG['bill_field_name']}")
        if AUTO_EXTRACT_CONFIG.get("auto_push_to_zoho"):
            print(f"  Push   : Auto-push to Zoho ENABLED")
            print(f"  Form   : {AUTO_EXTRACT_CONFIG['zoho_form_name']}")
        print("="*60 + "\n")

        thread = threading.Thread(target=_trigger_auto_extract, daemon=True)
        thread.start()

    # ── Status endpoint so the frontend can poll ──────────────────────────────
    from fastapi.responses import JSONResponse

    @app.get("/auto-startup/status")
    async def get_startup_status():
        return JSONResponse(content=startup_job_state)

    print("[AUTO-STARTUP] ✅ Startup hook registered")
