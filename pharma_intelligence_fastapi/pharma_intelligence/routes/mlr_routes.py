from __future__ import annotations

from datetime import datetime
import hashlib
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from bson import ObjectId
from fastapi import APIRouter, File, HTTPException, UploadFile

from ..mongo_database import analysis_results_collection, mlr_checklist_items_collection, mlr_review_results_collection
from ..services.mlr_checklist_parser import parse_mlr_checklist_docx
from ..services.mlr_review_engine import load_checklist_items, run_mlr_review_for_analysis


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/mlr", tags=["MLR Review"])


def _json_safe(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items() if key != "checklist_embedding" and key != "embedding"}
    return value


@router.post("/checklist/upload")
async def upload_mlr_checklist(file: UploadFile = File(...)):
    if not file.filename or Path(file.filename).suffix.lower() != ".docx":
        raise HTTPException(status_code=400, detail="Upload a .docx MLR checklist file.")
    logger.info("MLR checklist upload started: %s", file.filename)
    with NamedTemporaryFile(delete=False, suffix=".docx") as temp:
        temp_path = Path(temp.name)
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            temp.write(chunk)
    try:
        _, items = parse_mlr_checklist_docx(temp_path)
        digest = hashlib.sha256(temp_path.read_bytes()).hexdigest()[:16]
        version = f"{Path(file.filename).stem}:{digest}"
        if not items:
            raise HTTPException(status_code=400, detail="No checklist items were parsed from the uploaded DOCX.")
        logger.info("Parsed and embedded %s MLR checklist items for version=%s", len(items), version)
        for item in items:
            item.checklist_version = version
            item.source_file_name = file.filename
            payload = item.model_dump()
            created_date = payload.pop("created_date", None)
            mlr_checklist_items_collection.update_one(
                {"checklist_version": item.checklist_version, "item_code": item.item_code},
                {"$set": payload, "$setOnInsert": {"created_date": created_date or datetime.utcnow()}},
                upsert=True,
            )
        sections: dict[str, int] = {}
        for item in items:
            sections[item.section_name] = sections.get(item.section_name, 0) + 1
        return {"checklist_version": version, "item_count": len(items), "sections": sections, "file_name": file.filename}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("MLR checklist upload failed")
        raise HTTPException(status_code=500, detail=f"MLR checklist upload failed: {exc}") from exc
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass


@router.get("/checklist")
def get_mlr_checklist():
    items = load_checklist_items()
    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        section = str(item.get("section_name") or "Checklist")
        grouped.setdefault(section, []).append(_json_safe(item))
    return {"checklist_version": str(items[0].get("checklist_version") or "") if items else "", "sections": grouped, "total": len(items)}


def _latest_analysis_for_upload(upload_id: str) -> dict[str, Any] | None:
    return analysis_results_collection.find_one({"metadata.upload_id": upload_id}, sort=[("created_at", -1)])


@router.post("/review/{upload_id}")
def run_mlr_review(upload_id: str):
    analysis = _latest_analysis_for_upload(upload_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="No completed analysis found for this upload. Run normal processing first.")
    filename = str(analysis.get("file_name") or upload_id)
    try:
        return run_mlr_review_for_analysis(upload_id, filename, _json_safe(analysis))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("MLR review failed for upload_id=%s", upload_id)
        raise HTTPException(status_code=500, detail=f"MLR review failed: {exc}") from exc


@router.get("/review/{upload_id}")
def get_mlr_review(upload_id: str):
    review = mlr_review_results_collection.find_one({"upload_id": upload_id}, sort=[("created_at", -1)])
    if not review:
        raise HTTPException(status_code=404, detail="No MLR review found for this upload.")
    return _json_safe(review)


@router.get("/reviews")
def list_mlr_reviews(limit: int = 50):
    safe_limit = max(1, min(int(limit or 50), 200))
    reviews = mlr_review_results_collection.find({}, {"content_chunks": 0, "extracted_text": 0}).sort("created_at", -1).limit(safe_limit)
    return {"reviews": [_json_safe(review) for review in reviews]}
