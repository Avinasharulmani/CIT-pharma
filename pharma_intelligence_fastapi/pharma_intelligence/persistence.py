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
    accuracy = response.metadata.get("extraction_accuracy") or {}
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S_%f")
    identifier = analysis_result_id or response.metadata.get("analysis_result_id") or timestamp
    report_path = REPORTS_DIR / f"accuracy_{timestamp}_{_safe_report_name(str(identifier))}.json"
    report = {
        "analysis_result_id": analysis_result_id or response.metadata.get("analysis_result_id"),
        "file_name": response.file_name,
        "file_type": response.file_type,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "accuracy_score": response.metadata.get("accuracy_score"),
        "accuracy_percentage": response.metadata.get("accuracy_percentage"),
        "accuracy_type": response.metadata.get("accuracy_type"),
        "accuracy_module": response.metadata.get("accuracy_module"),
        "extraction_accuracy": accuracy,
        "accuracy_available": response.metadata.get("accuracy_available", False),
        "accuracy_note": response.metadata.get("accuracy_note"),
        "accuracy_source": response.metadata.get("accuracy_source"),
        "measured_accuracy": response.metadata.get("measured_accuracy"),
        "estimated_quality_score": response.metadata.get("estimated_quality_score"),
        "estimated_extraction_quality": response.metadata.get("estimated_extraction_quality"),
        "confidence_score": response.metadata.get("confidence_score"),
        "score_label": "Measured Accuracy" if response.metadata.get("accuracy_available") else "Estimated Extraction Quality",
        "quality_score": response.metadata.get("quality_score"),
        "quality_note": response.metadata.get("quality_note"),
        "unit_level_scores": response.metadata.get("unit_level_scores", []),
        "overall_cer": response.metadata.get("overall_cer"),
        "overall_wer": response.metadata.get("overall_wer"),
        "extraction_method_summary": response.metadata.get("extraction_method_summary", {}),
        "page_level_output": response.metadata.get("page_level_output", []),
        "slide_level_output": response.metadata.get("slide_level_output", []),
        "frame_level_output": response.metadata.get("frame_level_output", []),
        "warnings": response.metadata.get("warnings", []),
        "total_source_units": response.metadata.get("total_source_units"),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report_path
