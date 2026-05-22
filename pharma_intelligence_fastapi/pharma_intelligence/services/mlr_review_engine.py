from __future__ import annotations

from datetime import datetime
import logging
import re
import time
from typing import Any

from bson import ObjectId

from ..config import (
    MLR_AI_JUDGMENT_ENABLED,
    MLR_EVIDENCE_MIN_MATCHED_TERMS,
    MLR_EVIDENCE_TERM_MATCH_RATIO,
    MLR_REVIEW_SECTIONS,
    MLR_REVIEW_SIMILARITY_THRESHOLD,
    MLR_TOP_MATCHES,
)
from ..mongo_database import mlr_checklist_items_collection, mlr_review_results_collection
from ..utils import clean_text, split_into_chunks
from .mlr_embedding_service import embed_texts, find_relevant_content_for_checklist
from .mlr_ai_judgment import run_ai_mlr_judgments_batch
from .mlr_models import MLRContentChunk, MLRItemResult, MLRReviewReport


logger = logging.getLogger(__name__)
MLR_AI_BATCH_SIZE = 5
MLR_AI_BATCH_PAUSE_SECONDS = 1.0
WORKFLOW_ONLY_ITEM_CODES = {
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "1",
    "2",
    "3",
    "5",
    "6",
    "7",
    "8",
    "9",
    "L1",
    "L3",
    "L4",
    "L6",
    "L8",
    "R8",
    "R9",
}

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


def _json_safe(value: Any) -> Any:
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    return value


def latest_checklist_version() -> str:
    document = mlr_checklist_items_collection.find_one({}, sort=[("updated_date", -1)])
    return str(document.get("checklist_version") or "") if document else ""


def load_checklist_items(checklist_version: str | None = None) -> list[dict[str, Any]]:
    version = checklist_version or latest_checklist_version()
    if not version:
        return []
    items = list(mlr_checklist_items_collection.find({"checklist_version": version}).sort([("section_name", 1), ("item_code", 1)]))
    allowed_sections = {clean_text(section).lower() for section in MLR_REVIEW_SECTIONS}
    if not allowed_sections:
        return items
    return [
        item
        for item in items
        if clean_text(str(item.get("section_name") or "")).lower() in allowed_sections
    ]


def _checklist_id(item: dict[str, Any]) -> str:
    return str(item.get("checklist_id") or item.get("item_code") or item.get("id") or "")


def _checklist_criteria(item: dict[str, Any]) -> str:
    return str(
        item.get("checklist_criteria")
        or item.get("checklist_description")
        or item.get("criteria")
        or item.get("description")
        or ""
    )


def _evidence_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-zA-Z][a-zA-Z0-9-]{2,}", clean_text(text).lower())
        if term not in _EVIDENCE_STOPWORDS
    }


def _term_overlap(criteria: str, evidence: str) -> tuple[float, int]:
    criteria_terms = _evidence_terms(criteria)
    if not criteria_terms:
        return 0.0, 0
    evidence_terms = _evidence_terms(evidence)
    matched = criteria_terms.intersection(evidence_terms)
    return len(matched) / len(criteria_terms), len(matched)


def _has_enough_direct_evidence(criteria: str, evidence: str) -> bool:
    ratio, matched_count = _term_overlap(criteria, evidence)
    required_count = min(max(1, MLR_EVIDENCE_MIN_MATCHED_TERMS), max(1, len(_evidence_terms(criteria))))
    return matched_count >= required_count or ratio >= MLR_EVIDENCE_TERM_MATCH_RATIO


def _status_from_evidence(score: float, criteria: str, evidence: str) -> str:
    if score >= MLR_REVIEW_SIMILARITY_THRESHOLD and _has_enough_direct_evidence(criteria, evidence):
        return "Satisfied"
    if _has_content_signal(criteria, evidence) and score >= 0.45:
        return "Satisfied"
    return "Not Satisfied"


def _has_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = clean_text(text).lower()
    for term in terms:
        term = term.lower()
        if " " in term:
            if term in lowered:
                return True
        elif re.search(rf"\b{re.escape(term)}\b", lowered):
            return True
    return False


def _find_signal_chunk(content_chunks: list[dict[str, Any]], terms: tuple[str, ...]) -> dict[str, Any] | None:
    for chunk in content_chunks:
        if _has_any(str(chunk.get("text") or ""), terms):
            return chunk
    return None


def _matched_chunk_from_payload(chunk: dict[str, Any] | None, score: float = 1.0):
    if not chunk:
        return None
    from .mlr_models import MLRMatchedChunk

    return MLRMatchedChunk(
        chunk_id=str(chunk.get("chunk_id") or ""),
        location=str(chunk.get("location") or ""),
        text=str(chunk.get("text") or ""),
        similarity_score=round(float(score), 4),
    )


def _rule_based_content_status(
    item_code: str,
    criteria: str,
    content_chunks: list[dict[str, Any]],
    matches: list[Any],
) -> tuple[str | None, Any | None, str]:
    text = clean_text(" ".join(str(chunk.get("text") or "") for chunk in content_chunks))
    if not text:
        return "Not Satisfied", None, "No extracted OCR/transcript evidence available."

    item_code = str(item_code or "").strip()
    criteria_lower = clean_text(criteria).lower()
    content_lower = text.lower()

    benefit_terms = (
        "effective",
        "efficacy",
        "achieves",
        "treatment",
        "treat",
        "supports",
        "protect",
        "prevent",
        "reduction",
        "reduce",
        "improve",
        "relief",
        "maintain",
        "control",
        "controlling",
        "active against",
        "rapid penetration",
    )
    safety_terms = (
        "side effect",
        "adverse",
        "warning",
        "allergic",
        "contraindication",
        "precaution",
        "does not protect everyone",
        "severe allergic",
        "weakened immune",
        "fatigue",
        "headache",
        "muscle pain",
        "joint pain",
        "safety",
        "safe",
        "well tolerated",
    )
    dose_terms = (
        "mg",
        "mcg",
        "iu",
        "ml",
        "tablet",
        "tablets",
        "capsule",
        "cream",
        "dose",
        "daily",
        "weekly",
        "twice",
        "granules",
        "sachet",
    )
    clinical_terms = (
        "clinical",
        "study",
        "trial",
        "patients",
        "patient",
        "month",
        "weeks",
        "reduction",
        "%",
        "score",
        "density",
        "compared",
    )
    indication_terms = (
        "indication",
        "indicated",
        "used to",
        "prevent",
        "treat",
        "treatment of",
        "hypertension",
        "angina",
        "osteoporosis",
        "osteoarthritis",
        "diabetes",
        "rsv",
        "respiratory",
        "inflammation",
        "infection",
        "blepharitis",
        "keratitis",
        "conjunctivitis",
        "post surgical",
        "bone loss",
        "back pain",
    )
    comparison_terms = ("compared", "versus", " vs ", "superior", "better", "comparison", "lower than", "higher than")
    visual_terms = ("chart", "graph", "visual", "image", "poster", "claims", "layout", "red", "white", "video")
    pi_terms = ("prescribing information", " pi ", "smpc", "package insert", "full prescribing")
    mechanism_terms = ("mechanism", "inhibit", "inhibits", "block", "blocks", "receptor", "regulating", "calcium", "phosphorus")
    patient_outcome_terms = ("pain", "symptom", "weakness", "quality of life", "patient-reported", "relief")

    code_terms: dict[str, tuple[str, ...]] = {
        "M1": benefit_terms,
        "M2": safety_terms,
        "M3": dose_terms,
        "M4": clinical_terms,
        "M5": indication_terms,
        "M6": comparison_terms,
        "M7": ("%", "score", "rate", "reduction", "chart", "graph", "month", "weeks"),
        "M8": visual_terms,
        "M9": indication_terms + dose_terms,
        "M10": mechanism_terms,
        "M11": patient_outcome_terms,
        "R1": benefit_terms + safety_terms + indication_terms,
        "R2": safety_terms,
        "R3": pi_terms,
        "R4": ("adverse", "side effect", "report", "reaction"),
        "R5": indication_terms,
        "R6": visual_terms + dose_terms,
        "R7": benefit_terms + safety_terms,
        "R8": ("registration", "approved indication", "indicated", "used to", "treat", "prevent"),
        "R9": ("country", "mandatory", "warning", "disclaimer", "prescribing information", "pi"),
        "4": safety_terms + pi_terms,
        "10": visual_terms + dose_terms,
    }

    terms = code_terms.get(item_code)
    if not terms:
        if "safety" in criteria_lower or "warning" in criteria_lower or "isi" in criteria_lower:
            terms = safety_terms
        elif "dosing" in criteria_lower or "administration" in criteria_lower:
            terms = dose_terms
        elif "efficacy" in criteria_lower or "claims accuracy" in criteria_lower:
            terms = benefit_terms
        elif "indication" in criteria_lower:
            terms = indication_terms
        elif "visual" in criteria_lower or "format" in criteria_lower:
            terms = visual_terms

    if not terms:
        return None, None, ""

    matched_signal = _has_any(content_lower, terms)
    if item_code == "R7":
        matched_signal = _has_any(content_lower, benefit_terms) and _has_any(content_lower, safety_terms)
    if item_code in {"R3"}:
        matched_signal = _has_any(content_lower, pi_terms)

    if matched_signal:
        chunk = _find_signal_chunk(content_chunks, terms)
        matched = _matched_chunk_from_payload(chunk, score=1.0) or (matches[0] if matches else None)
        return "Satisfied", matched, "Satisfied because extracted OCR/transcript text contains evidence for this checklist criterion."

    return "Not Satisfied", matches[0] if matches else None, "Not satisfied because extracted OCR/transcript text does not contain evidence for this checklist criterion."


def _has_content_signal(criteria: str, evidence: str) -> bool:
    lowered_criteria = clean_text(criteria).lower()
    lowered_evidence = clean_text(evidence).lower()
    if not lowered_criteria or not lowered_evidence:
        return False
    safety_terms = (
        "side effect",
        "allergic reaction",
        "severe allergic",
        "weakened immune",
        "fatigue",
        "muscle pain",
        "headache",
        "joint pain",
        "adverse",
    )
    indication_terms = (
        "prevent lower respiratory",
        "lower respiratory disease",
        "rsv",
        "people 60 years",
        "60 years and older",
        "vaccine used to prevent",
    )
    limitation_terms = (
        "does not protect everyone",
        "not protect everyone",
        "severe allergic reactions",
        "lower response",
    )
    if _has_any(lowered_criteria, ("safety", "warning", "warnings", "isi", "important safety")):
        return _has_any(lowered_evidence, safety_terms)
    if _has_any(lowered_criteria, ("approved indication", "indications restriction")):
        return _has_any(lowered_evidence, indication_terms)
    if _has_any(lowered_criteria, ("disclaimer", "limitation", "fair balance")):
        return _has_any(lowered_evidence, limitation_terms) and _has_any(lowered_evidence, safety_terms)
    if _has_any(lowered_criteria, ("efficacy claim", "efficacy claims", "claim accuracy")):
        return _has_any(lowered_evidence, ("protect against", "prevent lower respiratory", "vaccine used to prevent"))
    return False


def _evidence_text(match: Any | None, status: str) -> str:
    if match is None:
        return "No matching evidence found."
    text = clean_text(getattr(match, "text", "") or "")
    if not text:
        return "No matching evidence found."
    if status == "Not Satisfied":
        return "No matching evidence found."
    location = clean_text(getattr(match, "location", "") or "")
    prefix = f"{location}: " if location else ""
    return f"{prefix}{text}"


def _fallback_evidence(matches: list[Any], status: str) -> str:
    if not matches:
        return "No matching evidence found."
    return _evidence_text(matches[0], status)


def _select_status_match(criteria: str, matches: list[Any]) -> tuple[str, Any | None]:
    for match in matches:
        text = getattr(match, "text", "") or ""
        score = float(getattr(match, "similarity_score", 0.0) or 0.0)
        if _status_from_evidence(score, criteria, text) == "Satisfied":
            return "Satisfied", match
    return "Not Satisfied", matches[0] if matches else None


def content_chunks_from_analysis(result: dict[str, Any]) -> list[MLRContentChunk]:
    metadata = result.get("metadata") or {}
    source_units = metadata.get("source_units") or []
    chunks: list[MLRContentChunk] = []
    for index, unit in enumerate(source_units, start=1):
        if not isinstance(unit, dict):
            continue
        unit_metadata = unit.get("metadata") if isinstance(unit.get("metadata"), dict) else {}
        text = clean_text(
            "\n".join(
                str(value or "")
                for value in (
                    unit.get("text"),
                    unit.get("description"),
                    unit.get("description_of_image_video"),
                    unit.get("ocr_text"),
                    unit.get("transcript"),
                    unit.get("frame_text"),
                    unit.get("frame_descriptions"),
                    unit_metadata.get("ocr_text"),
                    unit_metadata.get("transcript"),
                    unit_metadata.get("frame_text"),
                    unit_metadata.get("frame_descriptions"),
                )
            )
        )
        if not text:
            continue
        source_type = unit.get("source_type") or unit.get("type") or "unit"
        source_no = unit.get("source_no") or unit.get("number") or index
        text_parts = split_into_chunks(text, max_words=45, overlap=10) or [text]
        for part_index, part in enumerate(text_parts, start=1):
            location = f"{source_type} {source_no}" if len(text_parts) == 1 else f"{source_type} {source_no} part {part_index}"
            chunks.append(
                MLRContentChunk(
                    chunk_id=f"unit-{index}-{part_index}",
                    location=location,
                    text=part,
                )
            )
    if chunks:
        return chunks

    fallback_text = clean_text(
        "\n".join(
            str(result.get(key) or "")
            for key in (
                "full_transcript",
                "transcript",
                "content_summary",
                "summary",
                "description_of_image_video",
                "frame_descriptions",
            )
        )
        + "\n"
        + str(metadata.get("full_extracted_text") or "")
        + "\n"
        + str(metadata.get("frame_text") or "")
        + "\n"
        + str(metadata.get("frame_descriptions") or "")
    )
    return [
        MLRContentChunk(chunk_id=f"chunk-{index}", location=f"Text chunk {index}", text=text)
        for index, text in enumerate(split_into_chunks(fallback_text, max_words=120, overlap=25), start=1)
        if clean_text(text)
    ]


def _embed_content_chunks(chunks: list[MLRContentChunk]) -> list[dict[str, Any]]:
    embeddings = embed_texts([chunk.text for chunk in chunks])
    payload = []
    for chunk, embedding in zip(chunks, embeddings):
        item = chunk.model_dump()
        item["embedding"] = embedding
        payload.append(item)
    return payload


def _empty_report(upload_id: str, filename: str, checklist_items: list[dict[str, Any]], reason: str) -> MLRReviewReport:
    results = [
        MLRItemResult(
            item_code=str(item.get("item_code") or ""),
            checklist_id=_checklist_id(item),
            section=str(item.get("section_name") or ""),
            checklist_description=_checklist_criteria(item),
            checklist_criteria=_checklist_criteria(item),
            reviewer_type=str(item.get("reviewer_type") or ""),
            mandatory_status=str(item.get("mandatory_status") or ""),
            status="Not Satisfied",
            evidence=reason,
            evidence_found=reason,
            reasoning=reason,
            recommendation="Re-run extraction or manually review the source content.",
        )
        for item in checklist_items
    ]
    return _build_report(upload_id, filename, checklist_items, results, [], latest_checklist_version())


def _build_report(
    upload_id: str,
    filename: str,
    checklist_items: list[dict[str, Any]],
    item_results: list[MLRItemResult],
    content_chunks: list[dict[str, Any]],
    checklist_version: str,
) -> MLRReviewReport:
    sections = []
    for section_name in dict.fromkeys(result.section for result in item_results):
        section_items = [result for result in item_results if result.section == section_name]
        sections.append(
            {
                "section_name": section_name,
                "total": len(section_items),
                "satisfied_count": sum(1 for item in section_items if item.status == "Satisfied"),
                "not_satisfied_count": sum(1 for item in section_items if item.status == "Not Satisfied"),
                "needs_review_count": sum(1 for item in section_items if item.status == "Needs Review"),
                "cannot_determine_count": sum(1 for item in section_items if item.status == "Needs Review"),
                "items": [item.model_dump() for item in section_items],
            }
        )
    satisfied = sum(1 for item in item_results if item.status == "Satisfied")
    not_satisfied = sum(1 for item in item_results if item.status == "Not Satisfied")
    needs_review = sum(1 for item in item_results if item.status == "Needs Review")
    overall = "Non-Compliant" if not_satisfied else "Compliant"
    return MLRReviewReport(
        upload_id=upload_id,
        filename=filename,
        overall_mlr_status=overall,
        total_checklist_items=len(checklist_items),
        satisfied_count=satisfied,
        not_satisfied_count=not_satisfied,
        cannot_determine_count=needs_review,
        section_wise_results=sections,
        item_wise_results=item_results,
        checklist_version=checklist_version,
        created_date=datetime.utcnow(),
        status_summary={
            "overall": overall,
            "satisfied": satisfied,
            "not_satisfied": not_satisfied,
            "needs_review": needs_review,
            "cannot_determine": needs_review,
        },
    )


def run_mlr_review_for_analysis(upload_id: str, filename: str, analysis_result: dict[str, Any]) -> dict[str, Any]:
    logger.info("MLR review start for upload_id=%s filename=%s", upload_id, filename)
    checklist_version = latest_checklist_version()
    checklist_items = load_checklist_items(checklist_version)
    if not checklist_items:
        raise ValueError("No MLR checklist found in database.")

    chunks = content_chunks_from_analysis(analysis_result)
    if not chunks:
        report = _empty_report(upload_id, filename, checklist_items, "No extracted evidence available")
        return save_mlr_review_report(report, analysis_result, [])

    logger.info("Embedding %s content chunks for MLR review upload_id=%s", len(chunks), upload_id)
    content_chunks = _embed_content_chunks(chunks)
    matched_items: list[dict[str, Any]] = []
    for item in checklist_items:
        criteria = _checklist_criteria(item)
        matches = find_relevant_content_for_checklist(item, content_chunks, top_k=MLR_TOP_MATCHES)
        matched_items.append(
            {
                "item": item,
                "item_code": str(item.get("item_code") or ""),
                "section_name": str(item.get("section_name") or ""),
                "checklist_description": criteria,
                "reviewer_type": str(item.get("reviewer_type") or ""),
                "mandatory_status": str(item.get("mandatory_status") or ""),
                "matched_chunks": matches,
            }
        )
    judgments: dict[str, dict[str, str]] = {}
    if MLR_AI_JUDGMENT_ENABLED:
        for start in range(0, len(matched_items), MLR_AI_BATCH_SIZE):
            batch = matched_items[start : start + MLR_AI_BATCH_SIZE]
            try:
                judgments.update(run_ai_mlr_judgments_batch(batch, uploaded_filename=filename))
            except Exception as exc:
                logger.warning("MLR AI judgment batch failed; using embedding fallback for this batch: %s", exc)
            if start + MLR_AI_BATCH_SIZE < len(matched_items):
                time.sleep(MLR_AI_BATCH_PAUSE_SECONDS)

    item_results: list[MLRItemResult] = []
    for matched_item in matched_items:
        item = matched_item["item"]
        matches = matched_item["matched_chunks"]
        criteria = matched_item["checklist_description"]
        item_code = str(item.get("item_code") or "")
        rule_status, rule_match, rule_reasoning = _rule_based_content_status(item_code, criteria, content_chunks, matches)
        judgment = judgments.get(item_code) or {}
        selected_match = matches[0] if matches else None
        status = str(judgment.get("status") or "").strip()
        if item_code in WORKFLOW_ONLY_ITEM_CODES:
            status = "Not Satisfied"
            rule_reasoning = "Not satisfied because this checklist item requires workflow, approval, legal clearance, or system metadata that cannot be proven from OCR/transcript content alone."
        elif rule_status == "Satisfied":
            status = "Satisfied"
            selected_match = rule_match or selected_match
        elif status not in {"Satisfied", "Not Satisfied"}:
            status, selected_match = _select_status_match(criteria=criteria, matches=matches)
        best_score = float(getattr(selected_match, "similarity_score", 0.0) or 0.0) if selected_match else 0.0
        best_location = getattr(selected_match, "location", "") if selected_match else ""
        evidence_found = _evidence_text(selected_match, status) if rule_status == "Satisfied" else clean_text(judgment.get("evidence") or "") or _evidence_text(selected_match, status)
        reasoning = rule_reasoning if rule_status == "Satisfied" else clean_text(judgment.get("reasoning") or rule_reasoning or "")
        item_results.append(
            MLRItemResult(
                item_code=str(item.get("item_code") or ""),
                checklist_id=_checklist_id(item),
                section=str(item.get("section_name") or ""),
                checklist_description=criteria,
                checklist_criteria=criteria,
                reviewer_type=str(item.get("reviewer_type") or ""),
                mandatory_status=str(item.get("mandatory_status") or ""),
                status=status,
                evidence=evidence_found,
                evidence_found=evidence_found,
                reasoning=reasoning,
                recommendation=clean_text(judgment.get("recommendation") or ""),
                similarity_score=best_score,
                matched_location=best_location,
                matched_chunks=matches,
            )
        )

    report = _build_report(upload_id, filename, checklist_items, item_results, content_chunks, checklist_version)
    logger.info("MLR review complete for upload_id=%s status=%s", upload_id, report.overall_mlr_status)
    return save_mlr_review_report(report, analysis_result, content_chunks)


def save_mlr_review_report(report: MLRReviewReport, analysis_result: dict[str, Any], content_chunks: list[dict[str, Any]]) -> dict[str, Any]:
    document = report.model_dump()
    document["extracted_text"] = (analysis_result.get("metadata") or {}).get("full_extracted_text") or analysis_result.get("full_transcript") or ""
    document["content_chunks"] = content_chunks
    document["created_at"] = datetime.utcnow()
    result = mlr_review_results_collection.insert_one(document)
    response = _json_safe(document)
    response["review_id"] = str(result.inserted_id)
    return response
