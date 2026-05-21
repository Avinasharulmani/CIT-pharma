from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Body, File, HTTPException, UploadFile

from ..correction.dictionary_loader import (
    DICTIONARY_COLLECTIONS,
    get_all_terms,
    invalidate_dictionary_cache,
    load_dictionary_from_csv,
    refresh_dictionary_from_fda,
)
from ..correction.ocr_error_map import invalidate_pattern_cache
from ..mongo_database import db


router = APIRouter(prefix="/dictionary", tags=["dictionary"])


def _collection_for_type(value: str) -> str:
    collection = DICTIONARY_COLLECTIONS.get(str(value or "").strip())
    if not collection:
        raise HTTPException(status_code=400, detail="type must be one of: drug_names, terms, dosage_units")
    return collection


@router.post("/upload-csv")
async def upload_csv(type: str, file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Dictionary upload must be a .csv file")
    collection_name = _collection_for_type(type)
    content = (await file.read()).decode("utf-8", errors="ignore")
    try:
        count = await load_dictionary_from_csv(db, content, collection_name)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"inserted": count, "collection": collection_name}


@router.post("/refresh-fda")
async def refresh_fda():
    return await refresh_dictionary_from_fda(db)


@router.get("/terms")
async def terms():
    items = await get_all_terms(db)
    return {"terms": items, "count": len(items)}


@router.delete("/term/{term}")
async def delete_term(term: str):
    normalized = str(term or "").strip().lower()
    if not normalized:
        raise HTTPException(status_code=400, detail="term must not be empty")
    deactivated = False
    for collection_name in DICTIONARY_COLLECTIONS.values():
        result = db[collection_name].update_many({"term": normalized, "active": True}, {"$set": {"active": False}})
        deactivated = deactivated or result.modified_count > 0
    if deactivated:
        invalidate_dictionary_cache()
    return {"term": normalized, "deactivated": True}


@router.post("/term")
async def add_term(payload: dict = Body(...)):
    normalized = str(payload.get("term") or "").strip().lower()
    if not normalized:
        raise HTTPException(status_code=400, detail="term must not be empty")
    collection_name = _collection_for_type(str(payload.get("type") or ""))
    if db[collection_name].find_one({"term": normalized, "active": True}, {"_id": 1}):
        raise HTTPException(status_code=400, detail="term already exists")
    db[collection_name].insert_one(
        {
            "term": normalized,
            "source": "manual",
            "created_at": datetime.utcnow(),
            "active": True,
        }
    )
    invalidate_dictionary_cache()
    return {"term": normalized, "inserted": True}


@router.post("/ocr-pattern")
async def add_ocr_pattern(payload: dict = Body(...)):
    pattern = str(payload.get("pattern") or "")
    replacement = str(payload.get("replacement") or "")
    context = str(payload.get("context") or "any")
    if not pattern:
        raise HTTPException(status_code=400, detail="pattern must not be empty")
    allowed_contexts = {"between_vowels", "word_only", "word_start", "any"}
    if context not in allowed_contexts:
        raise HTTPException(status_code=400, detail="context must be one of: between_vowels, word_only, word_start, any")
    db["ocr_error_patterns"].insert_one(
        {
            "pattern": pattern,
            "replacement": replacement,
            "context": context,
            "active": True,
            "source": "manual",
            "created_at": datetime.utcnow(),
        }
    )
    invalidate_pattern_cache()
    return {"inserted": True}
