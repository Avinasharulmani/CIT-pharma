from __future__ import annotations

import csv
import io
import logging
from datetime import datetime
from typing import Any

import httpx

from ..config import FDA_API_URL


logger = logging.getLogger(__name__)

DICTIONARY_COLLECTIONS = {
    "drug_names": "pharma_drug_names",
    "terms": "pharma_terms",
    "dosage_units": "pharma_dosage_units",
}

_cache: dict[str, Any] = {"terms": [], "loaded_at": None}


def invalidate_dictionary_cache() -> None:
    _cache["loaded_at"] = None
    _cache["terms"] = []


def _utcnow() -> datetime:
    return datetime.utcnow()


def _normalize_term(value: str) -> str:
    return str(value or "").strip().lower()


async def fetch_fda_drug_names(db) -> int:
    inserted = 0
    names: set[str] = set()
    async with httpx.AsyncClient(timeout=20.0) as client:
        for field in ("generic_name", "brand_name"):
            field_names = [f"{field}.exact", f"openfda.{field}.exact"]
            for field_name in field_names:
                try:
                    response = await client.get(FDA_API_URL, params={"count": field_name, "limit": 1000})
                    response.raise_for_status()
                except Exception as exc:
                    logger.warning("FDA dictionary refresh failed for %s: %s", field_name, exc)
                    continue
                for item in response.json().get("results", []) or []:
                    term = _normalize_term(item.get("term", ""))
                    if term:
                        names.add(term)
                break
    if not names:
        return 0

    collection = db["pharma_drug_names"]
    for name in sorted(names):
        if collection.find_one({"term": name, "active": True}, {"_id": 1}):
            continue
        collection.insert_one(
            {
                "term": name,
                "source": "fda_api",
                "created_at": _utcnow(),
                "active": True,
            }
        )
        inserted += 1
    if inserted:
        invalidate_dictionary_cache()
    return inserted


async def load_dictionary_from_csv(db, csv_content: str, collection_name: str) -> int:
    if collection_name not in DICTIONARY_COLLECTIONS.values():
        raise ValueError("Unsupported dictionary collection")
    reader = csv.DictReader(io.StringIO(csv_content or ""))
    if not reader.fieldnames:
        return 0
    term_field = next((field for field in reader.fieldnames if str(field or "").strip().lower() == "term"), None)
    if not term_field:
        raise ValueError('CSV must include a "term" column')

    collection = db[collection_name]
    inserted = 0
    for row in reader:
        term = _normalize_term(row.get(term_field, ""))
        if not term or collection.find_one({"term": term, "active": True}, {"_id": 1}):
            continue
        collection.insert_one(
            {
                "term": term,
                "source": "csv_upload",
                "created_at": _utcnow(),
                "active": True,
            }
        )
        inserted += 1
    if inserted:
        invalidate_dictionary_cache()
    return inserted


async def get_all_terms(db) -> list[str]:
    now = _utcnow()
    loaded_at = _cache.get("loaded_at")
    if loaded_at is not None and (now - loaded_at).total_seconds() < 600:
        return list(_cache.get("terms") or [])

    terms: set[str] = set()
    for collection_name in DICTIONARY_COLLECTIONS.values():
        for document in db[collection_name].find({"active": True}, {"term": 1, "_id": 0}):
            term = _normalize_term(document.get("term", ""))
            if term:
                terms.add(term)
    result = sorted(terms)
    _cache["terms"] = result
    _cache["loaded_at"] = now
    return list(result)


async def refresh_dictionary_from_fda(db) -> dict:
    count = await fetch_fda_drug_names(db)
    return {"inserted": count, "source": "fda_api", "timestamp": _utcnow()}
