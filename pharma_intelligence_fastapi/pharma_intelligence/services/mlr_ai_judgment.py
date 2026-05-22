from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from ..config import GEMINI_API_KEY, GEMINI_MODEL_NAME, MLR_EVIDENCE_MIN_MATCHED_TERMS, MLR_EVIDENCE_TERM_MATCH_RATIO, MLR_REVIEW_SIMILARITY_THRESHOLD
from ..utils import clean_text

_EVIDENCE_STOPWORDS = {
    "a",
    "all",
    "an",
    "and",
    "are",
    "as",
    "be",
    "by",
    "content",
    "each",
    "for",
    "from",
    "has",
    "in",
    "is",
    "must",
    "no",
    "not",
    "of",
    "on",
    "or",
    "per",
    "the",
    "to",
    "with",
}


def _evidence_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", clean_text(text).lower())
        if term not in _EVIDENCE_STOPWORDS
    }


def _has_direct_evidence(criteria: str, evidence: str) -> bool:
    criteria_terms = _evidence_terms(criteria)
    if not criteria_terms:
        return False
    evidence_terms = _evidence_terms(evidence)
    matched_count = len(criteria_terms.intersection(evidence_terms))
    ratio = matched_count / len(criteria_terms)
    required_count = min(max(1, MLR_EVIDENCE_MIN_MATCHED_TERMS), max(1, len(criteria_terms)))
    return matched_count >= required_count or ratio >= MLR_EVIDENCE_TERM_MATCH_RATIO


def _fallback_judgment(matched_chunks: list[Any], reason: str, checklist_description: str = "") -> dict[str, str]:
    best = matched_chunks[0] if matched_chunks else None
    score = float(getattr(best, "similarity_score", 0.0) if best is not None else 0.0)
    evidence = clean_text(getattr(best, "text", "") if best is not None else "")
    if len(evidence) > 320:
        evidence = evidence[:320].rsplit(" ", 1)[0] + "..."
    status = "Satisfied" if score >= MLR_REVIEW_SIMILARITY_THRESHOLD and _has_direct_evidence(checklist_description, evidence) else "Not Satisfied"
    return {
        "status": status,
        "evidence": evidence or "No readable mapped evidence found in uploaded content.",
        "reasoning": reason,
        "recommendation": "Reviewer should manually verify this checklist item against the uploaded material.",
    }


def _extract_json(text: str) -> dict[str, Any]:
    text = clean_text(text)
    if text.startswith("```"):
        text = text.strip("`").replace("json", "", 1).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            return {}
        value = json.loads(text[start : end + 1])
        return value if isinstance(value, dict) else {}


def run_ai_mlr_judgment(checklist_item: dict[str, Any], matched_chunks: list[Any], uploaded_filename: str = "") -> dict[str, str]:
    """
    Sends checklist item and BGE-mapped content evidence to Gemini API.
    Returns status, evidence, reasoning, and recommendation.
    """
    if not GEMINI_API_KEY:
        criteria = str(
            checklist_item.get("checklist_description")
            or checklist_item.get("checklist_criteria")
            or checklist_item.get("criteria")
            or checklist_item.get("description")
            or ""
        )
        return _fallback_judgment(matched_chunks, "Gemini API key is not configured; AI judgment could not be completed.", criteria)

    evidence_payload = [
        {
            "location": getattr(chunk, "location", ""),
            "similarity_score": getattr(chunk, "similarity_score", 0.0),
            "text": clean_text(getattr(chunk, "text", ""))[:1200],
        }
        for chunk in matched_chunks
    ]
    prompt = {
        "instruction": (
            "You are performing pharma MLR compliance review. Use only the mapped uploaded-content evidence provided. "
            "Do not infer facts that are not present. Return exactly one status: Satisfied or Not Satisfied."
        ),
        "uploaded_filename": uploaded_filename,
        "checklist_item": {
            "item_code": checklist_item.get("item_code"),
            "section_name": checklist_item.get("section_name"),
            "checklist_description": (
                checklist_item.get("checklist_description")
                or checklist_item.get("checklist_criteria")
                or checklist_item.get("criteria")
                or checklist_item.get("description")
            ),
            "reviewer_type": checklist_item.get("reviewer_type"),
            "mandatory_status": checklist_item.get("mandatory_status"),
        },
        "mapped_content_chunks": evidence_payload,
        "required_json": {
            "status": "Satisfied | Not Satisfied",
            "evidence": "specific supporting or missing evidence",
            "reasoning": "short explanation grounded in mapped content",
            "recommendation": "next action for reviewer",
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    request = urllib.request.Request(
        url,
        data=json.dumps(
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": json.dumps(prompt, ensure_ascii=True)}],
                    }
                ],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": 900,
                    "responseMimeType": "application/json",
                },
            }
        ).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        criteria = str(
            checklist_item.get("checklist_description")
            or checklist_item.get("checklist_criteria")
            or checklist_item.get("criteria")
            or checklist_item.get("description")
            or ""
        )
        return _fallback_judgment(matched_chunks, f"Gemini judgment failed or was unavailable: {exc}", criteria)

    text_parts = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            if isinstance(part, dict) and part.get("text"):
                text_parts.append(str(part.get("text") or ""))
    parsed = _extract_json("\n".join(text_parts))
    status = str(parsed.get("status") or "Not Satisfied").strip()
    if status not in {"Satisfied", "Not Satisfied"}:
        status = "Not Satisfied"
    return {
        "status": status,
        "evidence": clean_text(str(parsed.get("evidence") or "")),
        "reasoning": clean_text(str(parsed.get("reasoning") or "")),
        "recommendation": clean_text(str(parsed.get("recommendation") or "")),
    }


def run_ai_mlr_judgments_batch(review_items: list[dict[str, Any]], uploaded_filename: str = "") -> dict[str, dict[str, str]]:
    """
    Sends all checklist rows and their BGE-mapped evidence in one Gemini request.
    Returns a mapping by item_code with status, evidence, reasoning, and recommendation.
    """
    if not GEMINI_API_KEY:
        return {
            str(item.get("item_code") or ""): _fallback_judgment(
                item.get("matched_chunks") or [],
                "Gemini API key is not configured; AI judgment could not be completed.",
                str(item.get("checklist_description") or ""),
            )
            for item in review_items
        }

    checklist_payload = []
    for item in review_items:
        matched_chunks = item.get("matched_chunks") or []
        evidence_payload = [
            {
                "location": getattr(chunk, "location", ""),
                "similarity_score": getattr(chunk, "similarity_score", 0.0),
                "text": clean_text(getattr(chunk, "text", ""))[:700],
            }
            for chunk in matched_chunks
        ]
        checklist_payload.append(
            {
                "item_code": item.get("item_code"),
                "section_name": item.get("section_name"),
                "checklist_description": item.get("checklist_description"),
                "reviewer_type": item.get("reviewer_type"),
                "mandatory_status": item.get("mandatory_status"),
                "mapped_content_chunks": evidence_payload,
            }
        )

    prompt = {
        "instruction": (
            "You are performing pharma MLR compliance review for uploaded promotional material. For each checklist item, "
            "use only the mapped uploaded-content evidence provided for that item. Mark Satisfied when the evidence clearly "
            "shows the uploaded material contains, supports, or complies with the checklist requirement. Mark Not Satisfied "
            "when evidence is missing, unrelated, insufficient, contradicts the requirement, or the item is a workflow/system "
            "approval requirement that cannot be proven from uploaded content text. Do not infer facts that are not present. "
            "Return one result for every checklist item."
        ),
        "uploaded_filename": uploaded_filename,
        "checklist_items": checklist_payload,
        "required_json": {
            "results": [
                {
                    "item_code": "same item_code from checklist_items",
                    "status": "Satisfied | Not Satisfied",
                    "evidence": "specific supporting or missing evidence",
                    "reasoning": "short explanation grounded in mapped content",
                    "recommendation": "next action for reviewer",
                }
            ]
        },
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL_NAME}:generateContent?key={GEMINI_API_KEY}"
    request = urllib.request.Request(
        url,
        data=json.dumps(
            {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": json.dumps(prompt, ensure_ascii=True)}],
                    }
                ],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": 6000,
                    "responseMimeType": "application/json",
                },
            }
        ).encode("utf-8"),
        headers={"content-type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        reason = f"Gemini batch judgment failed or was unavailable: {exc}"
        return {
            str(item.get("item_code") or ""): _fallback_judgment(
                item.get("matched_chunks") or [],
                reason,
                str(item.get("checklist_description") or ""),
            )
            for item in review_items
        }

    text_parts = []
    for candidate in data.get("candidates", []):
        content = candidate.get("content") or {}
        for part in content.get("parts", []):
            if isinstance(part, dict) and part.get("text"):
                text_parts.append(str(part.get("text") or ""))
    parsed = _extract_json("\n".join(text_parts))
    results = parsed.get("results") if isinstance(parsed.get("results"), list) else []
    judgments: dict[str, dict[str, str]] = {}
    for result in results:
        if not isinstance(result, dict):
            continue
        item_code = str(result.get("item_code") or "").strip()
        if not item_code:
            continue
        status = str(result.get("status") or "Not Satisfied").strip()
        if status not in {"Satisfied", "Not Satisfied"}:
            status = "Not Satisfied"
        judgments[item_code] = {
            "status": status,
            "evidence": clean_text(str(result.get("evidence") or "")),
            "reasoning": clean_text(str(result.get("reasoning") or "")),
            "recommendation": clean_text(str(result.get("recommendation") or "")),
        }
    return judgments
