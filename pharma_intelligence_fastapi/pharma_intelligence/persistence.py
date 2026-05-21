from datetime import datetime
import json
import re
from pathlib import Path

from .config import REPORTS_DIR
from .mongo_database import analysis_results_collection
from .models import BaseIntelligenceResponse


def save_analysis_response(response: BaseIntelligenceResponse) -> str:
    document = {
        "file_name": response.file_name,
        "file_type": response.file_type,
        "detected_language": response.detected_language,
        "target_language": response.target_language,
        "translated": response.translated,
        "summary": response.summary,
        "extracted_keywords": response.extracted_keywords,
        "description_of_image_video": response.description_of_image_video,
        "transcript": response.transcript,
        "frame_descriptions": response.frame_descriptions,
        "content_summary": response.content_summary,
        "full_transcript": response.full_transcript,
        "transcript_available": response.transcript_available,
        "summary_available": response.summary_available,
        "metadata": response.metadata,
        "accuracy_report": response.accuracy_report,
        "key_message_matches": [match.model_dump() for match in response.key_message_matches],
        "created_at": datetime.utcnow(),
    }
    result = analysis_results_collection.insert_one(document)
    return str(result.inserted_id)


def _safe_report_name(value: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", value or "material").strip("._-")
    return name[:80] or "material"


def save_accuracy_report(response: BaseIntelligenceResponse, analysis_result_id: str = "") -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    identifier = analysis_result_id or response.metadata.get("analysis_result_id") or timestamp
    report_path = REPORTS_DIR / f"accuracy_{timestamp}_{_safe_report_name(str(identifier))}.json"
    report = response.accuracy_report or {
        "file_name": response.file_name,
        "file_type": response.file_type,
        "model_used": None,
        "accuracy": None,
        "model_accuracy": None,
        "is_measured_accuracy": False,
        "method": "accuracy_not_available",
        "warnings": [],
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path
