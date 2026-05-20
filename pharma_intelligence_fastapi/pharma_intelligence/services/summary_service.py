import logging
import re
from typing import Iterable

from ..config import BRIEF_SUMMARY_MAX_CHARS, MAX_CHARS_FOR_SUMMARY, NO_SUMMARY_MESSAGE
from ..utils import clean_text, remove_repetition

logger = logging.getLogger(__name__)


def _trim_summary(text: str, max_chars: int = BRIEF_SUMMARY_MAX_CHARS) -> str:
    summary = clean_text(text)
    if len(summary) <= max_chars:
        return summary

    trimmed = summary[:max_chars].rsplit(" ", 1)[0].strip()
    return f"{trimmed}..." if trimmed else summary[:max_chars].strip()


def _normalized(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _unique_parts(values: Iterable[str]) -> list[str]:
    seen = set()
    output = []
    for value in values:
        cleaned = clean_text(str(value or ""))
        key = _normalized(cleaned)
        if cleaned and key and key not in seen:
            seen.add(key)
            output.append(cleaned)
    return output


def _join_details(parts: list[str]) -> str:
    cleaned = [part.rstrip(" .;") for part in parts if part]
    if len(cleaned) <= 1:
        return cleaned[0] if cleaned else ""
    return "; ".join(cleaned)


def generate_content_summary(text: str) -> str:
    logger.info("Summary generation started.")
    cleaned = remove_repetition(clean_text(text))
    if not cleaned:
        logger.warning("Summary generation skipped because no readable content was found.")
        return NO_SUMMARY_MESSAGE

    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", cleaned)
        if sentence.strip()
    ]
    if not sentences:
        logger.warning("Summary generation skipped because extracted content had no usable sentences.")
        return NO_SUMMARY_MESSAGE

    selected = []
    total_length = 0
    for sentence in sentences:
        if sentence in selected:
            continue
        selected.append(sentence)
        total_length += len(sentence)
        if len(selected) >= 1 or total_length >= MAX_CHARS_FOR_SUMMARY:
            break

    summary = clean_text(" ".join(selected))
    if len(summary) > MAX_CHARS_FOR_SUMMARY:
        summary = summary[:MAX_CHARS_FOR_SUMMARY].rsplit(" ", 1)[0] + "..."

    logger.info("Summary generation completed.")
    return _trim_summary(summary) or NO_SUMMARY_MESSAGE


def generate_mapped_summary(matches, fallback_summary: str = "") -> str:
    """Create a brief user-facing summary from mapped product/message details."""
    if not matches:
        return _trim_summary(fallback_summary) if fallback_summary else NO_SUMMARY_MESSAGE

    summary_parts = []
    for match in matches[:3]:
        brand = clean_text(getattr(match, "brand_product", "") or "")
        message = clean_text(getattr(match, "key_message", "") or "")
        description = clean_text(getattr(match, "description_of_image_video", "") or "")

        details = _unique_parts([message, description])
        if not details:
            continue

        detail_text = _join_details(details)
        summary_parts.append(f"{brand}: {detail_text}" if brand else detail_text)

    if not summary_parts:
        return _trim_summary(fallback_summary) if fallback_summary else NO_SUMMARY_MESSAGE

    return _trim_summary(" | ".join(summary_parts))


def _metadata_title(sentence: str) -> str:
    cleaned = clean_text(sentence)
    cleaned = re.sub(r"^(?:the|this|these|it)\s+", "", cleaned, flags=re.IGNORECASE)
    words = re.findall(r"[A-Za-z0-9%+-]+", cleaned)
    if not words:
        return "Content Insight"
    title = " ".join(words[:7]).strip()
    return title[:1].upper() + title[1:]


def _metadata_sentences(text: str) -> list[str]:
    cleaned = remove_repetition(clean_text(text))
    cleaned = re.sub(r"\b(?:upload|file|extension|duration|page count|slide count|file size)\b[^.?!]*[.?!]?", "", cleaned, flags=re.IGNORECASE)
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", cleaned)
        if sentence.strip()
    ]
    output = []
    seen = set()
    for sentence in sentences:
        normalized = _normalized(sentence)
        if not normalized or normalized in seen:
            continue
        if len(normalized.split()) < 4:
            continue
        seen.add(normalized)
        output.append(sentence)
    return output


def generate_content_metadata(analysis_text: str) -> list[dict[str, str]]:
    """Generate content-based metadata points from extracted text/transcript/visual content."""
    sentences = _metadata_sentences(analysis_text)
    if not sentences:
        return [
            {
                "title": "No content available",
                "description": "The uploaded file did not contain enough readable or transcribable content to generate content-based metadata.",
            }
        ]

    points: list[dict[str, str]] = []
    index = 0
    while index < len(sentences):
        lead = sentences[index]
        details = [lead]
        index += 1
        while index < len(sentences) and len(details) < 4:
            candidate = sentences[index]
            details.append(candidate)
            index += 1
            if len(details) >= 3 and len(" ".join(details)) >= 260:
                break

        description = clean_text(" ".join(details))
        if len(description) > 720:
            description = description[:720].rsplit(" ", 1)[0].strip() + "..."
        points.append(
            {
                "title": _metadata_title(lead),
                "description": description,
            }
        )

    unique_points: list[dict[str, str]] = []
    seen_titles = set()
    for point in points:
        key = _normalized(point["title"] + " " + point["description"][:120])
        if key and key not in seen_titles:
            seen_titles.add(key)
            unique_points.append(point)
    return unique_points
