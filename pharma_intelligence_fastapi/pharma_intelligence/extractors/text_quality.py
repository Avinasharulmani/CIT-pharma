import re
from collections import OrderedDict
from typing import Iterable

from ..utils import clean_text


def normalize_line(value: str) -> str:
    return re.sub(r"\W+", " ", clean_text(value).lower()).strip()


def remove_duplicate_lines(values: Iterable[str] | str) -> str:
    if isinstance(values, str):
        lines = values.splitlines()
    else:
        lines = []
        for value in values:
            lines.extend(str(value).splitlines())
    unique: OrderedDict[str, str] = OrderedDict()
    for raw_line in lines:
        line = clean_text(raw_line).strip()
        key = normalize_line(line)
        if not key:
            continue
        unique.setdefault(key, line)
    return clean_text("\n".join(unique.values()))


def merge_text_blocks(*blocks: str) -> str:
    return remove_duplicate_lines(block for block in blocks if clean_text(block))


def text_quality_score(value: str) -> float:
    text = clean_text(value)
    if not text:
        return 0.0
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9%+./-]*", text)
    letters = re.findall(r"[A-Za-z]", text)
    if not words or not letters:
        return 5.0
    alpha_ratio = len(letters) / max(len(text), 1)
    unique_ratio = len({word.lower() for word in words}) / max(len(words), 1)
    long_word_ratio = len([word for word in words if len(word) >= 3]) / max(len(words), 1)
    length_bonus = min(20.0, len(text) / 30.0)
    score = (alpha_ratio * 35.0) + (unique_ratio * 25.0) + (long_word_ratio * 20.0) + length_bonus
    return round(max(0.0, min(100.0, score)), 2)


def is_weak_text(value: str, *, min_chars: int = 30, min_quality: float = 35.0) -> bool:
    text = clean_text(value)
    if len(text) < min_chars:
        return True
    return text_quality_score(text) < min_quality


def extraction_unit_metadata(
    *,
    source_type: str,
    unit_number: int | None = None,
    timestamp: float | None = None,
    raw_text: str = "",
    cleaned_text: str = "",
    extraction_method: str = "",
    confidence_score: float | None = None,
    warnings: list[str] | None = None,
) -> dict:
    cleaned = clean_text(cleaned_text or raw_text)
    metadata = {
        "source_type": source_type,
        "unit_number": unit_number,
        "timestamp": timestamp,
        "raw_text": clean_text(raw_text),
        "cleaned_text": cleaned,
        "extraction_method": extraction_method,
        "confidence_score": confidence_score,
        "text_length": len(cleaned),
        "text_quality_score": text_quality_score(cleaned),
        "warnings": warnings or [],
    }
    return metadata
