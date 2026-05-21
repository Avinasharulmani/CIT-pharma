from __future__ import annotations

from datetime import datetime
from typing import Any


_cache: dict[str, Any] = {"patterns": [], "loaded_at": None}


def invalidate_pattern_cache() -> None:
    _cache["loaded_at"] = None
    _cache["patterns"] = []


def _utcnow() -> datetime:
    return datetime.utcnow()


async def get_ocr_error_patterns(db) -> list[dict]:
    now = _utcnow()
    loaded_at = _cache.get("loaded_at")
    if loaded_at is not None and (now - loaded_at).total_seconds() < 600:
        return list(_cache.get("patterns") or [])
    patterns = []
    for document in db["ocr_error_patterns"].find({"active": True}, {"_id": 0}):
        patterns.append(dict(document))
    _cache["patterns"] = patterns
    _cache["loaded_at"] = now
    return list(patterns)


async def seed_default_patterns(db) -> None:
    collection = db["ocr_error_patterns"]
    if collection.count_documents({}) > 0:
        return
    defaults = [
        {"pattern": "rn", "replacement": "m", "context": "between_vowels"},
        {"pattern": "0", "replacement": "o", "context": "word_only"},
        {"pattern": "1", "replacement": "l", "context": "word_only"},
        {"pattern": "|", "replacement": "l", "context": "word_only"},
        {"pattern": "5", "replacement": "S", "context": "word_start"},
    ]
    collection.insert_many(
        [
            {
                **item,
                "active": True,
                "source": "default",
                "created_at": _utcnow(),
            }
            for item in defaults
        ]
    )
    invalidate_pattern_cache()
