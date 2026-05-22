from __future__ import annotations

from typing import Any

from .mlr_review_engine import run_mlr_review_for_analysis


def run_mlr_review(payload: dict[str, Any]) -> dict[str, Any]:
    result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    upload_id = str(payload.get("upload_id") or (result.get("metadata") or {}).get("upload_id") or "adhoc-review")
    filename = str(result.get("file_name") or upload_id)
    return run_mlr_review_for_analysis(upload_id, filename, result)
