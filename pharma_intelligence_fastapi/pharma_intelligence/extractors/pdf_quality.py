import re
from typing import Any

from ..utils import clean_text


def _edit_distance(left: list[str] | str, right: list[str] | str) -> int:
    previous = list(range(len(right) + 1))
    for index, left_item in enumerate(left, start=1):
        current = [index]
        for other_index, right_item in enumerate(right, start=1):
            insert_cost = current[other_index - 1] + 1
            delete_cost = previous[other_index] + 1
            replace_cost = previous[other_index - 1] + (left_item != right_item)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def _tokens(value: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+", clean_text(value).lower())


def calculate_measured_accuracy(extracted_text: str, ground_truth_text: str) -> dict[str, Any]:
    expected = clean_text(ground_truth_text)
    actual = clean_text(extracted_text)
    if not expected and not actual:
        return {"measured_accuracy": 100.0, "cer": 0.0, "wer": 0.0}
    if not expected:
        return {"measured_accuracy": None, "cer": None, "wer": None, "warning": "ground truth text is empty"}

    cer = round((_edit_distance(expected, actual) / max(len(expected), 1)) * 100, 2)
    expected_words = _tokens(expected)
    actual_words = _tokens(actual)
    wer = round((_edit_distance(expected_words, actual_words) / max(len(expected_words), 1)) * 100, 2)
    accuracy = round(max(0.0, 100.0 - wer), 2)
    return {"measured_accuracy": accuracy, "cer": cer, "wer": wer}


def _noise_ratio(text: str) -> float:
    value = clean_text(text)
    if not value:
        return 1.0
    allowed = re.findall(r"[A-Za-z0-9\s.,;:!?%/()+\-–—~®™•'’]", value)
    return round(max(0.0, 1.0 - (len(allowed) / max(len(value), 1))), 4)


def _garbled_word_ratio(text: str) -> float:
    words = re.findall(r"[A-Za-z][A-Za-z0-9\-]*", clean_text(text))
    if not words:
        return 1.0
    garbled = 0
    for word in words:
        letters = re.findall(r"[A-Za-z]", word)
        if len(letters) < 5:
            continue
        vowels = re.findall(r"[AEIOUaeiou]", word)
        if len(vowels) / max(len(letters), 1) < 0.16:
            garbled += 1
        elif re.search(r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]{6,}", word):
            garbled += 1
    return round(garbled / max(len(words), 1), 4)


def calculate_pdf_quality_score(
    page_text: str,
    extraction_method: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = metadata or {}
    text = clean_text(page_text)
    text_length = len(text)
    words = _tokens(text)
    word_count = len(words)
    warnings = list(metadata.get("warnings", []) or [])
    ocr_confidence = metadata.get("ocr_confidence") or metadata.get("confidence_score")
    noise_ratio = _noise_ratio(text)
    garbled_ratio = _garbled_word_ratio(text)
    method = (extraction_method or "").lower()

    if not text:
        return {
            "quality_score": 0.0,
            "confidence_score": ocr_confidence,
            "noise_ratio": noise_ratio,
            "garbled_word_ratio": garbled_ratio,
            "word_count": word_count,
            "warnings": [*warnings, "empty extracted text"],
        }

    if method == "native":
        score = 90.0
        if text_length >= 2000:
            score += 4.0
        elif text_length >= 800:
            score += 3.0
        elif text_length >= 250:
            score += 1.5
        if word_count >= 250:
            score += 2.0
        elif word_count >= 80:
            score += 1.0
        if noise_ratio <= 0.015:
            score += 1.5
        if garbled_ratio <= 0.01:
            score += 1.0
        if warnings:
            score -= min(8.0, len(warnings) * 2.0)
        score -= min(8.0, noise_ratio * 100.0)
        score -= min(6.0, garbled_ratio * 80.0)
    elif method == "ocr":
        score = 65.0
        if ocr_confidence is not None:
            try:
                score = max(score, min(94.0, float(ocr_confidence)))
            except Exception:
                pass
        if text_length >= 1000:
            score += 5.0
        elif text_length >= 250:
            score += 2.0
        if noise_ratio > 0.06:
            score -= 8.0
        if garbled_ratio > 0.08:
            score -= 8.0
        if warnings:
            score -= min(12.0, len(warnings) * 3.0)
    else:
        score = 84.0
        if text_length >= 800:
            score += 4.0
        if ocr_confidence is not None:
            try:
                score = (score * 0.65) + (float(ocr_confidence) * 0.35)
            except Exception:
                pass
        score -= min(8.0, noise_ratio * 100.0)
        score -= min(6.0, garbled_ratio * 80.0)
        if warnings:
            score -= min(10.0, len(warnings) * 2.5)

    quality_score = round(max(0.0, min(99.0, score)), 2)
    confidence_score = round(float(ocr_confidence), 2) if ocr_confidence is not None and method == "ocr" else quality_score
    return {
        "quality_score": quality_score,
        "confidence_score": confidence_score,
        "noise_ratio": noise_ratio,
        "garbled_word_ratio": garbled_ratio,
        "word_count": word_count,
        "warnings": warnings,
    }
