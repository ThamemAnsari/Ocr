"""
Database module for logging OCR processing - SUPABASE VERSION
Includes cost tracking stored in Supabase
"""
from datetime import datetime, timedelta
import json
from typing import Dict, Any, List
import os
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise Exception("❌ SUPABASE_URL and SUPABASE_KEY must be set in .env file")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

print(f"[DATABASE] ✓ Connected to Supabase")


def init_db():
    """
    Initialize Supabase table (run this SQL in Supabase SQL Editor):
    
    CREATE TABLE IF NOT EXISTS processing_logs (
        id BIGSERIAL PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        doc_type TEXT NOT NULL,
        filename TEXT NOT NULL,
        method TEXT NOT NULL,
        input_tokens INTEGER DEFAULT 0,
        output_tokens INTEGER DEFAULT 0,
        total_tokens INTEGER DEFAULT 0,
        cost_usd DECIMAL(10, 6) DEFAULT 0.0,
        success BOOLEAN NOT NULL,
        error_message TEXT,
        student_name TEXT,
        scholarship_id TEXT,
        extracted_data JSONB,
        processing_time_ms INTEGER,
        image_url TEXT,
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    
    CREATE INDEX idx_timestamp ON processing_logs(timestamp);
    CREATE INDEX idx_doc_type ON processing_logs(doc_type);
    CREATE INDEX idx_student_name ON processing_logs(student_name);
    CREATE INDEX idx_scholarship_id ON processing_logs(scholarship_id);
    """
    print("[DATABASE] ℹ️  Run the SQL schema in Supabase SQL Editor (see function docstring)")


def log_processing(
    doc_type: str,
    filename: str,
    method: str,
    input_tokens: int,
    output_tokens: int,
    total_tokens: int,
    cost_usd: float,
    success: bool,
    error_message: str = None,
    student_name: str = None,
    scholarship_id: str = None,
    extracted_data: Dict = None,
    processing_time_ms: int = None,
    image_url: str = None
):
    """Log a processing event to Supabase"""
    try:
        data = {
            "timestamp": datetime.now().isoformat(),
            "doc_type": doc_type,
            "filename": filename,
            "method": method,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": float(cost_usd),  # Ensure it's a float
            "success": success,
            "error_message": error_message,
            "student_name": student_name,
            "scholarship_id": scholarship_id,
            "extracted_data": extracted_data,  # Supabase handles JSON automatically
            "processing_time_ms": processing_time_ms,
            "image_url": image_url
        }
        
        result = supabase.table("processing_logs").insert(data).execute()
        
        print(f"[DATABASE] ✓ Logged: {filename} (Cost: ${cost_usd})")
        return result.data[0] if result.data else None
        
    except Exception as e:
        print(f"[DATABASE] ✗ Error logging to Supabase: {e}")
        raise


def get_usage_stats(days: int = 30) -> Dict[str, Any]:
    """Get usage statistics for the specified period"""
    try:
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        # Get all logs within the date range
        response = supabase.table("processing_logs")\
            .select("*")\
            .gte("timestamp", start_date)\
            .execute()
        
        logs = response.data
        
        # Calculate summary stats
        total_requests = len(logs)
        total_input_tokens = sum(log.get('input_tokens', 0) for log in logs)
        total_output_tokens = sum(log.get('output_tokens', 0) for log in logs)
        total_tokens = sum(log.get('total_tokens', 0) for log in logs)
        total_cost = sum(float(log.get('cost_usd', 0)) for log in logs)
        
        gemini_vision_count = sum(1 for log in logs if log.get('method') == 'gemini_vision')
        ocr_text_count = sum(1 for log in logs if log.get('method') == 'ocr_text')
        bank_count = sum(1 for log in logs if log.get('doc_type') == 'bank')
        bill_count = sum(1 for log in logs if log.get('doc_type') == 'bill')
        success_count = sum(1 for log in logs if log.get('success'))
        
        summary = {
            'total_requests': total_requests,
            'total_input_tokens': total_input_tokens,
            'total_output_tokens': total_output_tokens,
            'total_tokens': total_tokens,
            'total_cost_usd': round(total_cost, 6),
            'gemini_vision_count': gemini_vision_count,
            'ocr_text_count': ocr_text_count,
            'bank_count': bank_count,
            'bill_count': bill_count,
            'success_count': success_count
        }
        
        # Daily breakdown
        daily_stats_dict = {}
        for log in logs:
            date = log.get('timestamp', '').split('T')[0]
            if date not in daily_stats_dict:
                daily_stats_dict[date] = {
                    'date': date,
                    'total_tokens': 0,
                    'cost_usd': 0.0,
                    'requests': 0
                }
            daily_stats_dict[date]['total_tokens'] += log.get('total_tokens', 0)
            daily_stats_dict[date]['cost_usd'] += float(log.get('cost_usd', 0))
            daily_stats_dict[date]['requests'] += 1
        
        daily_stats = sorted(daily_stats_dict.values(), key=lambda x: x['date'])
        
        # Round daily costs
        for stat in daily_stats:
            stat['cost_usd'] = round(stat['cost_usd'], 6)
        
        return {
            'summary': summary,
            'daily_stats': daily_stats,
            'period_days': days
        }
        
    except Exception as e:
        print(f"[DATABASE] ✗ Error getting stats: {e}")
        raise


def get_all_logs(limit: int = 100) -> List[Dict]:
    """Get all processing logs"""
    try:
        response = supabase.table("processing_logs")\
            .select("*")\
            .order("timestamp", desc=True)\
            .limit(limit)\
            .execute()
        
        return response.data
        
    except Exception as e:
        print(f"[DATABASE] ✗ Error getting logs: {e}")
        raise


def delete_log(log_id: int) -> bool:
    """Delete a specific log"""
    try:
        response = supabase.table("processing_logs")\
            .delete()\
            .eq("id", log_id)\
            .execute()
        
        return len(response.data) > 0
        
    except Exception as e:
        print(f"[DATABASE] ✗ Error deleting log: {e}")
        raise


def get_logs_by_student(student_name: str, limit: int = 50) -> List[Dict]:
    """Get logs for a specific student"""
    try:
        response = supabase.table("processing_logs")\
            .select("*")\
            .eq("student_name", student_name)\
            .order("timestamp", desc=True)\
            .limit(limit)\
            .execute()
        
        return response.data
        
    except Exception as e:
        print(f"[DATABASE] ✗ Error getting student logs: {e}")
        raise


def get_logs_by_scholarship(scholarship_id: str, limit: int = 50) -> List[Dict]:
    """Get logs for a specific scholarship"""
    try:
        response = supabase.table("processing_logs")\
            .select("*")\
            .eq("scholarship_id", scholarship_id)\
            .order("timestamp", desc=True)\
            .limit(limit)\
            .execute()
        
        return response.data
        
    except Exception as e:
        print(f"[DATABASE] ✗ Error getting scholarship logs: {e}")
        raise


def get_total_cost() -> float:
    """Get total cost across all time"""
    try:
        response = supabase.table("processing_logs")\
            .select("cost_usd")\
            .execute()
        
        total = sum(float(log.get('cost_usd', 0)) for log in response.data)
        return round(total, 6)
        
    except Exception as e:
        print(f"[DATABASE] ✗ Error getting total cost: {e}")
        raise


# Initialize on import
init_db()