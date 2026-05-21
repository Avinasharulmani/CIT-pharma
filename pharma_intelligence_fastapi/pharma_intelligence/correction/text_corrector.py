from __future__ import annotations

import logging
import re
from typing import Any

from rapidfuzz import fuzz, process

from ..config import CORRECTION_ENABLED, CORRECTION_THRESHOLD, MIN_WORD_LENGTH
from .dictionary_loader import get_all_terms
from .ocr_error_map import get_ocr_error_patterns


logger = logging.getLogger(__name__)


def _preserve_capitalization(original: str, replacement: str) -> str:
    if original[:1].isupper():
        return replacement[:1].upper() + replacement[1:]
    return replacement


async def apply_ocr_error_patterns(text: str, db) -> str:
    corrected = str(text or "")
    patterns = await get_ocr_error_patterns(db)
    vowels = "aeiouAEIOU"
    for item in patterns:
        pattern = str(item.get("pattern") or "")
        replacement = str(item.get("replacement") or "")
        context = str(item.get("context") or "any")
        if not pattern:
            continue
        escaped = re.escape(pattern)
        if context == "between_vowels":
            corrected = re.sub(rf"(?<=[{vowels}]){escaped}(?=[{vowels}])", replacement, corrected)
        elif context == "word_only":
            def replace_inside_word(match: re.Match) -> str:
                word = match.group(0)
                if word.isdigit():
                    return word
                return word.replace(pattern, replacement)

            corrected = re.sub(r"\b\w+\b", replace_inside_word, corrected)
        elif context == "word_start":
            corrected = re.sub(rf"\b{escaped}(?=[A-Za-z])", replacement, corrected)
        elif context == "any":
            corrected = corrected.replace(pattern, replacement)
    return corrected


async def correct_word(word: str, dictionary: list) -> str:
    if not CORRECTION_ENABLED:
        return word
    if len(word) < MIN_WORD_LENGTH:
        return word
    if any(char.isdigit() for char in word):
        return word
    match = re.match(r"^(\W*)([\w-]+)(\W*)$", word)
    if not match:
        return word
    prefix, core, suffix = match.groups()
    if len(core) < MIN_WORD_LENGTH or any(char.isdigit() for char in core):
        return word
    result = process.extractOne(core.lower(), dictionary, scorer=fuzz.ratio)
    if not result:
        return word
    best_match, score, _index = result
    if float(score) >= CORRECTION_THRESHOLD:
        return prefix + _preserve_capitalization(core, str(best_match)) + suffix
    return word


async def correct_dosage_format(text: str, db) -> str:
    units = [
        str(document.get("term") or "").strip()
        for document in db["pharma_dosage_units"].find({"active": True}, {"term": 1, "_id": 0})
    ]
    units = [unit for unit in units if unit]
    if not units:
        return text
    units_pattern = "|".join(re.escape(unit) for unit in sorted(units, key=len, reverse=True))
    pattern = re.compile(rf"(\d+\.?\d*)\s+({units_pattern})\b", flags=re.IGNORECASE)
    return pattern.sub(lambda match: f"{match.group(1)}{match.group(2)}", text)


async def correct_extracted_text(text: str, db) -> tuple[str, dict[str, Any]]:
    original = str(text or "")
    if not CORRECTION_ENABLED:
        return original, {"corrections_made": 0, "correction_enabled": False}
    try:
        corrected = await apply_ocr_error_patterns(original, db)
        corrected = await correct_dosage_format(corrected, db)
        dictionary = await get_all_terms(db)
        warnings: list[str] = []
        if not dictionary:
            logger.warning("Pharma dictionary is empty - skipping word correction")
            warnings.append("Pharma dictionary is empty")
            corrections_made = sum(1 for before, after in zip(original.split(), corrected.split()) if before != after)
            return (
                corrected,
                {
                    "corrections_made": corrections_made,
                    "dictionary_size": 0,
                    "correction_enabled": True,
                    "warnings": warnings,
                },
            )

        words = corrected.split()
        corrected_words = [await correct_word(word, dictionary) for word in words]
        final_text = " ".join(corrected_words)
        original_words = original.split()
        corrections_made = sum(
            1
            for index, word in enumerate(corrected_words)
            if index >= len(original_words) or original_words[index] != word
        )
        return (
            final_text,
            {
                "corrections_made": corrections_made,
                "dictionary_size": len(dictionary),
                "correction_enabled": True,
                "warnings": warnings,
            },
        )
    except Exception as exc:
        logger.warning("OCR text correction failed: %s", exc)
        return (
            original,
            {
                "corrections_made": 0,
                "correction_enabled": True,
                "warnings": [f"OCR correction failed: {exc}"],
            },
        )
