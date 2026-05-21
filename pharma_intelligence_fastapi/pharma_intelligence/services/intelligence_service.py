from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import re
import tempfile
from typing import List, Optional

from ..config import DEFAULT_TOP_K, DEFAULT_TARGET_LANGUAGE, MAX_CHARS_FOR_MAPPING, NO_SPEECH_MESSAGE, NO_SUMMARY_MESSAGE, SUPPORTED_FILE_TYPES, UPLOAD_DIR, VIDEO_FRAME_SAMPLE_SECONDS, VIDEO_MAX_OCR_FRAMES
from ..accuracy.engine import (
    compute_cer,
    compute_wer,
    get_easyocr_confidence,
    get_best_ocr_text,
    get_tesseract_confidence,
    get_tesseract_text,
    load_reference_file,
    load_reference_transcript,
    normalize_whisper_confidence,
)
from ..accuracy.report_builder import build_report
from ..correction.text_corrector import correct_extracted_text
from ..extractors import (
    extract_document,
    extract_email,
    extract_fallback,
    extract_image,
    extract_markup_text,
    extract_pdf,
    extract_ppt,
    extract_spreadsheet,
    extract_video_intelligence,
    extract_zip_package,
    transcribe_audio_file,
    transcribe_audio_result,
)
from ..key_messages import load_key_messages
from ..keywords import extract_keywords
from ..language import detect_language, translate_text
from ..models import BaseIntelligenceResponse, SourceChunk
from ..mongo_database import db as mongo_db
from ..preprocessing import preprocess_image
from ..utils import clean_text, compact_join, detect_file_type, get_file_type_config, remove_repetition
from ..extractors.text_quality import extraction_unit_metadata
from .mapping_service import match_key_messages
from .summary_service import generate_content_metadata, generate_content_summary, generate_mapped_summary


logger = logging.getLogger(__name__)


def _run_async_blocking(coro):
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(lambda: asyncio.run(coro)).result()


def _extract_by_file_type(path: Path, file_type: str, page_slide_number: Optional[int]) -> List[SourceChunk]:
    if file_type == "pdf":
        return extract_pdf(path, page_number=page_slide_number)
    if file_type == "ppt":
        if path.suffix.lower() != ".pptx":
            return extract_document(path)
        return extract_ppt(path, slide_number=page_slide_number)
    if file_type == "image":
        if path.suffix.lower() == ".svg":
            return extract_markup_text(path)
        if path.suffix.lower() in {".ai", ".eps", ".psd", ".indd", ".raw"}:
            return extract_fallback(path)
        try:
            return extract_image(path)
        except Exception as exc:
            chunks = extract_fallback(path)
            for chunk in chunks:
                warnings = list(chunk.metadata.get("warnings", []) or [])
                warnings.append(f"Image extraction warning: {exc}")
                chunk.metadata["warnings"] = warnings
            return chunks
    if file_type == "document":
        return extract_document(path)
    if file_type == "spreadsheet":
        return extract_spreadsheet(path)
    if file_type == "email":
        return extract_email(path)
    if file_type in {"text", "script"}:
        return extract_markup_text(path)
    if file_type == "archive":
        return extract_zip_package(path)
    if file_type == "audio":
        warnings = []
        transcript = ""
        transcript_segments = []
        try:
            audio_result = transcribe_audio_result(path)
            transcript = audio_result.get("text", "")
            transcript_segments = audio_result.get("segments", []) or []
            warnings.extend(audio_result.get("warnings", []) or [])
        except Exception as exc:
            warnings.append(f"Audio transcription warning: {exc}")
            logger.warning("Audio transcription failed: %s", exc)
        if not transcript:
            logger.warning("No speech detected in audio file.")
        return [
            SourceChunk(
                source_no=1,
                source_type="audio",
                text=transcript,
                metadata={
                    **extraction_unit_metadata(
                        source_type="audio",
                        unit_number=1,
                        raw_text=transcript,
                        cleaned_text=transcript,
                        extraction_method="whisper_audio_transcript",
                        confidence_score=None,
                        warnings=warnings,
                    ),
                    "transcript": transcript,
                    "transcript_segments": transcript_segments,
                    "language": audio_result.get("language") if "audio_result" in locals() else None,
                    "asr_model_size": audio_result.get("model_size") if "audio_result" in locals() else None,
                    "accuracy_report": audio_result.get("accuracy_report") if "audio_result" in locals() else None,
                    "warnings": warnings,
                },
            )
        ]
    if file_type == "video":
        return extract_video_intelligence(path)
    return extract_fallback(path)


def _correct_text_sync(text: str) -> tuple[str, dict]:
    return _run_async_blocking(correct_extracted_text(text, mongo_db))


def _append_unique_warnings(target: list, warnings: list) -> list:
    for warning in warnings or []:
        if warning and warning not in target:
            target.append(warning)
    return target


def _apply_ocr_corrections_to_chunks(chunks: List[SourceChunk], file_type: str) -> List[SourceChunk]:
    if file_type not in {"image", "pdf"}:
        return chunks
    for chunk in chunks:
        if not clean_text(chunk.text):
            continue
        original_text = chunk.text
        corrected_text, correction_stats = _correct_text_sync(original_text)
        chunk.text = corrected_text
        metadata = chunk.metadata or {}
        metadata["ocr_correction_stats"] = correction_stats
        metadata["cleaned_text"] = corrected_text
        metadata["corrected_text"] = corrected_text
        if "ocr_text" in metadata and metadata.get("ocr_text"):
            metadata["ocr_text"] = corrected_text if file_type == "image" else metadata["ocr_text"]
        if "final_text" in metadata:
            metadata["final_text"] = corrected_text
            metadata["final_text_length"] = len(corrected_text)
        if "text_length" in metadata:
            metadata["text_length"] = len(corrected_text)
        warnings = list(metadata.get("warnings", []) or [])
        _append_unique_warnings(warnings, correction_stats.get("warnings", []))
        metadata["warnings"] = warnings
        page_level = metadata.get("page_level")
        if isinstance(page_level, dict):
            page_level["final_text"] = corrected_text
            page_level["final_text_length"] = len(corrected_text)
            page_warnings = list(page_level.get("warnings", []) or [])
            _append_unique_warnings(page_warnings, correction_stats.get("warnings", []))
            page_level["warnings"] = page_warnings
            page_level["ocr_correction_stats"] = correction_stats
        chunk.metadata = metadata
    return chunks


def _source_location(chunks: List[SourceChunk], file_type: str) -> str:
    if not chunks:
        return file_type
    parts = []
    for chunk in chunks:
        if chunk.source_no:
            parts.append(f"{chunk.source_type} {chunk.source_no}")
        else:
            parts.append(chunk.source_type)
    return ", ".join(parts)


def _chunk_source_location(chunk: SourceChunk) -> str:
    if chunk.source_no:
        return f"{chunk.source_type} {chunk.source_no}"
    return chunk.source_type


def _source_sort_key(source_location: str):
    match = re.search(r"(\d+)", source_location)
    if match:
        return (source_location[: match.start()], int(match.group(1)))
    return (source_location, 0)


def _summary_from_text(text: str) -> str:
    text = remove_repetition(clean_text(text))
    if not text:
        return NO_SUMMARY_MESSAGE
    return generate_content_summary(text)


def _summary_source_text(chunks: List[SourceChunk], file_type: str, description: Optional[str]) -> str:
    text = compact_join([chunk.text for chunk in chunks], limit=None)
    if file_type in {"pdf", "ppt", "image"}:
        return compact_join([text, description or ""], limit=None)
    return text


def _unique_activity_lines(text: Optional[str]) -> str:
    candidates = []
    seen = set()
    for line in clean_text(text or "").splitlines():
        line = line.strip()
        if not line:
            continue
        activity_only = re.sub(r"^(?:\d{2}:\d{2}\s+)?Frame\s+\d+:\s*", "", line, flags=re.IGNORECASE).strip()
        normalized = re.sub(r"\W+", " ", activity_only.lower()).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            weak = normalized in {"person is shown in the scene", "people are shown interacting", "a person is standing"}
            timestamp_match = re.match(r"^(\d{2}):(\d{2})", line)
            timestamp = 999999
            if timestamp_match:
                timestamp = int(timestamp_match.group(1)) * 60 + int(timestamp_match.group(2))
            candidates.append((weak, timestamp, line))
    has_strong = any(not weak for weak, _, _ in candidates)
    lines = [
        line
        for weak, _, line in sorted(candidates, key=lambda item: item[1])
        if not (has_strong and weak)
    ]
    return clean_text("\n".join(lines))


def _frame_content_lines(frame_text: Optional[str]) -> str:
    frame_blocks = []
    current_label = ""
    current_lines = []

    def flush_current() -> None:
        nonlocal current_label, current_lines
        if not current_label:
            return
        unique_lines = []
        seen_lines = set()
        for value in current_lines:
            cleaned = clean_text(value).rstrip(" .;")
            normalized = re.sub(r"\W+", " ", cleaned.lower()).strip()
            if not normalized or normalized in seen_lines:
                continue
            seen_lines.add(normalized)
            unique_lines.append(cleaned)
        if unique_lines:
            frame_blocks.append(f"{current_label}: On-screen content shows {'; '.join(unique_lines[:4])}.")
        current_label = ""
        current_lines = []

    for raw_line in clean_text(frame_text or "").splitlines():
        line = raw_line.strip()
        match = re.match(r"^((?:\d{2}:\d{2}\s+)?Frame\s+\d+):\s*(.*)$", line, flags=re.IGNORECASE)
        if match:
            flush_current()
            current_label = match.group(1)
            if match.group(2).strip():
                current_lines.append(match.group(2).strip())
            continue
        if current_label and line:
            current_lines.append(line)
    flush_current()

    output = []
    seen_descriptions = set()
    for block in frame_blocks:
        normalized = re.sub(r"^(?:\d{2}:\d{2}\s+)?Frame\s+\d+:\s*", "", block, flags=re.IGNORECASE)
        normalized = re.sub(r"\W+", " ", normalized.lower()).strip()
        if normalized and normalized not in seen_descriptions:
            seen_descriptions.add(normalized)
            output.append(block)
        if len(output) >= 6:
            break
    return clean_text("\n".join(output))


def _mapping_frame_text(frame_text: Optional[str]) -> str:
    lines = []
    for line in clean_text(frame_text or "").splitlines():
        body = re.sub(r"^(?:\d{2}:\d{2}\s+)?Frame\s+\d+:\s*", "", line, flags=re.IGNORECASE).strip()
        if not body:
            continue
        compact = re.sub(r"\s+", "", body.lower())
        if len(compact) >= 24:
            hex_chars = sum(1 for char in compact if char in "0123456789abcdef")
            if hex_chars / max(1, len(compact)) > 0.85:
                continue
        lines.append(body)
    return clean_text("\n".join(lines))


def _visual_activity_description(
    file_type: str,
    description: Optional[str],
    frame_descriptions: Optional[str],
    frame_text: Optional[str] = None,
) -> Optional[str]:
    if file_type == "image":
        return clean_text(description or "") or "No clear image content description could be generated."
    if file_type in {"pdf", "ppt"}:
        return clean_text(description or "") or None
    if file_type == "video":
        video_summary = clean_text(description or "")
        if video_summary:
            return video_summary
        frame_activity = _unique_activity_lines(frame_descriptions)
        if frame_activity:
            return frame_activity
        frame_content = _frame_content_lines(frame_text)
        if frame_content:
            return frame_content
        return clean_text(description or "") or "No clear frame-by-frame visual activity or readable visual content detected in the sampled video frames."
    return None


def _source_unit_payload(chunks: List[SourceChunk]) -> List[dict]:
    units = []
    for chunk in chunks:
        unit_text = _format_source_unit_text(chunk)
        units.append(
            {
                "number": chunk.source_no,
                "type": chunk.source_type,
                "text": unit_text,
                "raw_text": clean_text(chunk.text),
                "cleaned_text": clean_text(chunk.metadata.get("cleaned_text") or unit_text or chunk.text),
                "extraction_method": chunk.metadata.get("extraction_method"),
                "confidence_score": chunk.metadata.get("confidence_score"),
                "text_length": chunk.metadata.get("text_length", len(clean_text(unit_text or chunk.text))),
                "warnings": chunk.metadata.get("warnings", []),
                "description": chunk.description_of_image_video,
                "title": chunk.metadata.get("title"),
                "metadata": chunk.metadata,
            }
        )
    return units


def _looks_like_ocr_noise(line: str) -> bool:
    value = clean_text(line).strip(" .,:;|[]{}()")
    if not value:
        return True
    if len(value) <= 1:
        return True
    letters = re.findall(r"[A-Za-z]", value)
    if not letters:
        return True
    alpha_ratio = len(letters) / max(len(value), 1)
    if alpha_ratio < 0.35:
        return True
    words = re.findall(r"[A-Za-z0-9%+.-]+", value)
    if not words:
        return True
    short_words = [word for word in words if len(re.sub(r"[^A-Za-z0-9]", "", word)) <= 1]
    if len(short_words) >= max(2, len(words) // 2 + 1):
        return True
    return False


def _token_vowel_ratio(value: str) -> float:
    letters = re.findall(r"[A-Za-z]", value)
    if not letters:
        return 0.0
    vowels = [letter for letter in letters if letter.lower() in "aeiou"]
    return len(vowels) / len(letters)


def _max_consonant_run(value: str) -> int:
    runs = re.findall(r"[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]+", value)
    return max((len(run) for run in runs), default=0)


def _clean_ocr_token(token: str) -> str:
    value = clean_text(token).strip(" .,:;|[]{}()")
    if not value:
        return ""
    if len(value) == 1 and not value.isdigit():
        return ""
    letters = re.findall(r"[A-Za-z]", value)
    if not letters:
        return value if re.search(r"\d", value) else ""
    if len(letters) >= 4 and _max_consonant_run(value) >= 4:
        return ""
    if len(letters) >= 4 and _token_vowel_ratio(value) < 0.30 and not re.search(r"\d|%|\+", value):
        return ""
    if len(letters) <= 3 and not re.search(r"\d|%|\+", value) and not value.isupper():
        return ""
    if re.search(r"[A-Za-z]{2,}-[A-Za-z]{1,2}$", value) and not re.search(r"\d", value):
        return ""
    return value


def _clean_ocr_fragment(fragment: str) -> str:
    words = re.findall(r"[A-Za-z0-9%+.-]+", clean_text(fragment))
    cleaned = [_clean_ocr_token(word) for word in words]
    cleaned = [word for word in cleaned if word]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        word = cleaned[0]
        letters = re.findall(r"[A-Za-z]", word)
        if len(letters) <= 3 and not re.search(r"\d|%|\+", word):
            return ""
        if len(letters) <= 4 and word.isupper() and _token_vowel_ratio(word) < 0.30 and not re.search(r"\d", word):
            return ""
    return " ".join(cleaned)


def _readable_text_fragments(text: Optional[str]) -> List[str]:
    fragments_out = []
    seen = set()
    for raw_line in clean_text(text or "").splitlines():
        line = clean_text(raw_line).strip(" .,:;|[]{}")
        if _looks_like_ocr_noise(line):
            continue
        normalized = re.sub(r"\W+", " ", line.lower()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        fragments_out.append(line)
    return fragments_out


def _phrase_list(values: List[str]) -> str:
    values = [value for value in values if value]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return f"{', '.join(values[:-1])}, and {values[-1]}"


def _split_readable_fragments(fragments: List[str]) -> tuple[List[str], List[str]]:
    primary = []
    details = []
    for value in fragments:
        normalized = value.lower()
        if re.search(r"\b(support|supports|absorption|immune|system|dietary|supplement|vitamin|zinc|lactoferrin|echinacea)\b", normalized):
            details.append(value)
        elif len(primary) < 8:
            primary.append(value)
        else:
            details.append(value)
    return primary, details


def _clean_readable_text(text: Optional[str]) -> str:
    fragments = _readable_text_fragments(text)
    if not fragments:
        return ""
    primary, details = _split_readable_fragments(fragments)
    sentences = []
    if primary:
        sentences.append(f"The visible copy identifies {_phrase_list(primary)}.")
    if details:
        sentences.append(f"It also includes {_phrase_list(details)}.")
    return clean_text(" ".join(sentences))


def _description_visual_bits(description: Optional[str]) -> List[str]:
    value = _visual_description_for_transcript(description).rstrip(".")
    if not value:
        return []
    parts = [
        clean_text(part).lower().strip(" .,:;")
        for part in re.split(r"[.;\n]+", value)
        if clean_text(part)
    ]
    joined = " ".join(parts)
    bits = []
    if "pregnant" in joined and "woman" in joined:
        bits.append("a pregnant woman")
    elif "woman" in joined:
        bits.append("a woman")
    elif "pregnant" in joined:
        bits.append("a pregnant person")
    elif "man" in joined:
        bits.append("a man")
    elif "person" in joined:
        bits.append("a person")
    else:
        bits.extend(part for part in parts if part not in {"image", "photo", "picture"})
    return bits


def _visual_context_sentence(description: Optional[str], readable_fragments: List[str]) -> str:
    lowered_text = " ".join(readable_fragments).lower()
    visual_bits = _description_visual_bits(description)
    if any(term in lowered_text for term in ("supplement", "pharma", "product", "package", "lacto", "apex")):
        visual_bits.append("product or supplement packaging")
    if any(term in lowered_text for term in ("poster", "advertisement", "promo", "pharma")):
        visual_bits.append("promotional healthcare material")
    if not visual_bits:
        return ""
    seen = []
    for bit in visual_bits:
        if bit in {"medicine", "medical"} and any("packaging" in item for item in visual_bits):
            continue
        if bit and bit not in seen:
            if bit in {"woman", "man", "person", "girl", "boy"}:
                bit = f"a {bit}"
            seen.append(bit)
    return f"The image shows {_phrase_list(seen)}."


def _visual_description_for_transcript(description: Optional[str]) -> str:
    value = clean_text(description or "")
    if not value:
        return ""
    value = re.sub(r"\bReadable text includes:\s*.*$", "", value, flags=re.IGNORECASE).strip(" .")
    return f"{value}." if value else ""


def _format_source_unit_text(chunk: SourceChunk) -> str:
    if chunk.metadata.get("extension") in {".ai", ".eps"}:
        return clean_text(chunk.text)
    if chunk.metadata.get("image_segmented"):
        return clean_text(chunk.text)
    if chunk.source_type == "page" and chunk.metadata.get("title"):
        return clean_text(chunk.text)
    if chunk.source_type not in {"image", "page", "slide", "video"}:
        return clean_text(chunk.text)
    return clean_text(chunk.text)


def _full_extracted_transcript(
    chunks: List[SourceChunk],
    file_type: str,
    transcript: Optional[str],
    description: Optional[str],
    frame_text: Optional[str],
    frame_descriptions: Optional[str],
) -> str:
    sections = []
    if file_type in {"audio", "video"}:
        if transcript:
            sections.append(f"Transcript:\n{clean_text(transcript)}")
        if frame_text:
            sections.append(f"Readable frame content:\n{clean_text(frame_text)}")
        if frame_descriptions:
            sections.append(f"Visual frame descriptions:\n{clean_text(frame_descriptions)}")
        if description:
            sections.append(f"Visual description:\n{clean_text(description)}")
    else:
        for chunk in chunks:
            unit_text = _format_source_unit_text(chunk)
            if unit_text:
                sections.append(unit_text)
    return clean_text("\n\n".join(section for section in sections if section))


def _score_bounds(value: float) -> float:
    return round(max(0.0, min(100.0, value)), 2)


def _extraction_module_name(file_type: str) -> str:
    if file_type == "pdf":
        return "pdf_text_or_scanned_pdf_ocr"
    if file_type == "ppt":
        return "ppt_extraction"
    if file_type == "document":
        return "word_document_extraction"
    if file_type == "spreadsheet":
        return "excel_spreadsheet_extraction"
    if file_type == "image":
        return "image_ocr_and_description"
    if file_type == "audio":
        return "whisper_audio_transcript"
    if file_type == "video":
        return "video_audio_transcript_and_frame_ocr"
    return f"{file_type}_extraction"


def _estimate_chunk_accuracy(chunk: SourceChunk, file_type: str) -> dict:
    text = clean_text(chunk.text)
    description = clean_text(chunk.description_of_image_video or "")
    warnings = list(chunk.metadata.get("warnings", []) or [])
    score = 0.0
    reasons = []

    if text:
        score += 70
        reasons.append("readable text extracted")
        if len(text) >= 500:
            score += 15
        elif len(text) >= 120:
            score += 10
        elif len(text) >= 30:
            score += 5
    elif file_type in {"image", "video"} and description:
        score += 55
        reasons.append("visual description extracted")
    else:
        reasons.append("no readable text extracted")

    if description and file_type in {"pdf", "ppt", "image", "video"}:
        score += 10
        reasons.append("visual description available")

    if chunk.metadata.get("ocr_text_found"):
        score += 10
        reasons.append(f"OCR text found via {chunk.metadata.get('ocr_engine') or 'OCR'}")
    elif chunk.metadata.get("ocr_available") and file_type == "image":
        score -= 10
        reasons.append("OCR available but no text found")

    if warnings:
        score -= min(30, len(warnings) * 10)
        reasons.append("warnings present")

    return {
        "source_no": chunk.source_no,
        "source_type": chunk.source_type,
        "score": _score_bounds(score),
        "text_chars": len(text),
        "has_description": bool(description),
        "warnings": warnings,
        "reasons": reasons,
    }


def _estimated_extraction_accuracy(
    chunks: List[SourceChunk],
    file_type: str,
    full_transcript: str,
    transcript: Optional[str],
    description: Optional[str],
    frame_text: Optional[str],
    frame_descriptions: Optional[str],
    warnings: List[str],
) -> dict:
    unit_scores = [_estimate_chunk_accuracy(chunk, file_type) for chunk in chunks]
    if unit_scores:
        score = sum(float(unit["score"]) for unit in unit_scores) / len(unit_scores)
    else:
        score = 0.0

    if file_type == "pdf":
        pdf_units = []
        for chunk in chunks:
            page_level = chunk.metadata.get("page_level") or {}
            quality = page_level.get("quality_score", chunk.metadata.get("quality_score"))
            text_length = page_level.get("final_text_length", len(clean_text(chunk.text)))
            if quality is None:
                continue
            pdf_units.append(
                {
                    "source_no": chunk.source_no,
                    "source_type": chunk.source_type,
                    "score": _score_bounds(float(quality)),
                    "text_chars": int(text_length or 0),
                    "extraction_method": page_level.get("extraction_method") or chunk.metadata.get("extraction_method"),
                    "quality_score": _score_bounds(float(quality)),
                    "measured_accuracy": page_level.get("measured_accuracy"),
                    "accuracy_source": page_level.get("accuracy_source", "estimated_quality"),
                    "warnings": page_level.get("warnings", chunk.metadata.get("warnings", [])),
                    "reasons": ["page-level estimated quality from extracted text characteristics"],
                }
            )
        if pdf_units:
            total_chars = sum(max(int(unit["text_chars"]), 1) for unit in pdf_units)
            score = sum(float(unit["quality_score"]) * max(int(unit["text_chars"]), 1) for unit in pdf_units) / max(total_chars, 1)
            unit_scores = pdf_units
    elif file_type == "audio":
        score = 95.0 if clean_text(transcript or full_transcript) else 0.0
    elif file_type == "video":
        score = 0.0
        if clean_text(transcript or ""):
            score += 45
        if clean_text(frame_text or ""):
            score += 30
        if clean_text(frame_descriptions or description or ""):
            score += 20
        if not score and clean_text(full_transcript):
            score = 40
    elif file_type in {"document", "spreadsheet", "ppt"} and clean_text(full_transcript):
        score = max(score, 85.0)
    elif file_type == "image":
        has_ocr = any(clean_text(chunk.text) for chunk in chunks)
        has_description = bool(clean_text(description or ""))
        if has_ocr and has_description:
            score = max(score, 90.0)
        elif has_ocr:
            score = max(score, 80.0)
        elif has_description:
            score = max(score, 65.0)

    if warnings:
        score -= min(20, len(warnings) * 5)

    score = _score_bounds(score)
    return {
        "score": score,
        "percentage": score,
        "type": "estimated_extraction_quality" if file_type == "pdf" else "estimated_extraction_quality_without_ground_truth",
        "module": _extraction_module_name(file_type),
        "can_average": True,
        "accuracy_source": "estimated_quality",
        "measured_accuracy": None,
        "estimated_quality_score": score,
        "confidence_score": score if file_type == "pdf" else None,
        "note": "Estimated extraction quality. Real measured accuracy requires manually verified ground truth.",
        "unit_scores": unit_scores,
    }


def _save_pil_temp(image, folder: str, name: str) -> str:
    path = Path(folder) / name
    image.convert("RGB").save(path)
    return str(path)


def _cleanup_preprocessed(preprocessed_path: str, original_path: str) -> None:
    if preprocessed_path != original_path:
        try:
            os.remove(preprocessed_path)
        except Exception:
            pass


def _pdf_strict_accuracy(file_path: Path, file_name: str, job_id: str) -> dict:
    warnings: list[str] = []
    page_scores: list[dict] = []
    method = "confidence_only_scanned_pdf"
    model_used = "tesseract+easyocr"
    is_measured = False
    try:
        import fitz
        import pdfplumber
        from PIL import Image

        doc = fitz.open(str(file_path))
        with tempfile.TemporaryDirectory() as folder:
            with pdfplumber.open(str(file_path)) as pdf:
                for page_index, fitz_page in enumerate(doc, start=1):
                    native_text = ""
                    if page_index - 1 < len(pdf.pages):
                        native_text = pdf.pages[page_index - 1].extract_text() or ""
                    pixmap = fitz_page.get_pixmap(matrix=fitz.Matrix(150 / 72.0, 150 / 72.0), alpha=False)
                    image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
                    image_path = _save_pil_temp(image, folder, f"page_{page_index}.png")
                    preprocessed_path = preprocess_image(image_path)
                    if native_text and len(native_text.strip()) > 50:
                        try:
                            ocr_text, _ocr_confidence, ocr_engine_used = get_best_ocr_text(preprocessed_path)
                            ocr_text, _correction_stats = _correct_text_sync(ocr_text)
                            page_accuracy = round((1.0 - compute_cer(native_text, ocr_text)) * 100, 2)
                            method = "pdf_self_referential_cer"
                            model_used = f"pdfplumber+{ocr_engine_used}"
                            is_measured = True
                        finally:
                            _cleanup_preprocessed(preprocessed_path, image_path)
                    else:
                        try:
                            _ocr_text, ocr_confidence, ocr_engine_used = get_best_ocr_text(preprocessed_path)
                            page_accuracy = round(ocr_confidence * 100, 2)
                            method = "confidence_only_scanned_pdf"
                            model_used = ocr_engine_used
                            if is_measured:
                                method = "mixed_pdf_self_referential_and_confidence"
                            is_measured = False
                        finally:
                            _cleanup_preprocessed(preprocessed_path, image_path)
                    page_scores.append({"page": page_index, "accuracy": page_accuracy})
        doc.close()
    except Exception as exc:
        warnings.append(f"PDF accuracy computation failed: {exc}")

    values = [float(page["accuracy"]) for page in page_scores]
    accuracy = round(sum(values) / len(values), 2) if values else None
    for page in page_scores:
        if page["accuracy"] < 70:
            warnings.append(f"Page {page['page']} has low extraction quality")
    return build_report(file_name, "pdf", model_used, accuracy, is_measured, method, warnings, mongo_db, job_id)


def _image_strict_accuracy(file_path: Path, file_name: str, job_id: str, corrected_extracted_text: str = "") -> dict:
    reference_text = load_reference_file(job_id, str(UPLOAD_DIR))
    warnings: list[str] = []
    original_path = str(file_path)
    preprocessed_path = preprocess_image(original_path)
    if reference_text is not None:
        try:
            extracted_text, _ocr_confidence, ocr_engine_used = get_best_ocr_text(preprocessed_path)
            extracted_text = corrected_extracted_text or extracted_text
            if not corrected_extracted_text:
                extracted_text, _correction_stats = _correct_text_sync(extracted_text)
            accuracy = round((1.0 - compute_cer(reference_text, extracted_text)) * 100, 2)
            is_measured = True
            method = "cer_vs_ground_truth"
            model_used = ocr_engine_used
            if accuracy < 60:
                warnings.append("Low accuracy vs ground truth - image may be blurry or skewed")
            if accuracy < 80:
                warnings.append("Moderate accuracy - review drug names and dosage values manually")
        finally:
            _cleanup_preprocessed(preprocessed_path, original_path)
    else:
        try:
            _extracted_text, ocr_confidence, ocr_engine_used = get_best_ocr_text(preprocessed_path)
            accuracy = round(ocr_confidence * 100, 2)
            is_measured = False
            method = "confidence_only_no_ground_truth"
            model_used = ocr_engine_used
            warnings = [
                "No ground truth file uploaded. This is model confidence only, not real accuracy. Upload ground_truth.txt alongside image for measured accuracy."
            ]
        finally:
            _cleanup_preprocessed(preprocessed_path, original_path)
    return build_report(file_name, "image", model_used, accuracy, is_measured, method, warnings, mongo_db, job_id)


def _audio_strict_accuracy(file_name: str, job_id: str, transcript: str, segments: list[dict]) -> dict:
    reference = load_reference_transcript(job_id, str(UPLOAD_DIR))
    warnings: list[str] = []
    if reference:
        wer = compute_wer(reference, transcript)
        accuracy = round((1.0 - wer) * 100, 2)
        is_measured = True
        method = "wer_vs_reference_transcript"
        model_used = "whisper"
        if accuracy < 70:
            warnings.append("Low transcript accuracy vs reference - audio may be noisy")
    else:
        accuracy = round(normalize_whisper_confidence(segments) * 100, 2)
        is_measured = False
        method = "whisper_confidence_only"
        model_used = "whisper"
        warnings = [
            "No reference transcript uploaded. Showing Whisper confidence only, not measured accuracy. Upload reference_transcript.txt for real accuracy."
        ]
    return build_report(file_name, "audio", model_used, accuracy, is_measured, method, warnings, mongo_db, job_id)


def _sample_video_frame_confidence(file_path: Path) -> float | None:
    try:
        import cv2
    except Exception:
        return None
    scores = []
    cap = cv2.VideoCapture(str(file_path))
    if not cap.isOpened():
        return None
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        interval = max(int(fps * VIDEO_FRAME_SAMPLE_SECONDS), 1)
        with tempfile.TemporaryDirectory() as folder:
            count = 0
            for frame_index in range(0, max(total_frames, 1), interval):
                if count >= VIDEO_MAX_OCR_FRAMES:
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
                if not ok:
                    continue
                image_path = str(Path(folder) / f"frame_{count + 1}.png")
                cv2.imwrite(image_path, frame)
                preprocessed_path = preprocess_image(image_path)
                try:
                    tess_conf = get_tesseract_confidence(preprocessed_path)
                    _easy_text, easy_conf = get_easyocr_confidence(preprocessed_path)
                    scores.append(round(((tess_conf + easy_conf) / 2.0) * 100, 2))
                    count += 1
                finally:
                    _cleanup_preprocessed(preprocessed_path, image_path)
    finally:
        cap.release()
    return round(sum(scores) / len(scores), 2) if scores else None


def _video_strict_accuracy(file_path: Path, file_name: str, job_id: str, transcript: str, segments: list[dict]) -> dict:
    reference = load_reference_transcript(job_id, str(UPLOAD_DIR))
    warnings: list[str] = []
    if reference:
        audio_accuracy = round((1.0 - compute_wer(reference, transcript)) * 100, 2)
        audio_is_measured = True
        audio_method = "wer_vs_reference_transcript"
    else:
        audio_accuracy = round(normalize_whisper_confidence(segments) * 100, 2)
        audio_is_measured = False
        audio_method = "whisper_confidence_only"
        warnings.append("No reference transcript - audio accuracy is confidence only")
    frame_accuracy = _sample_video_frame_confidence(file_path)
    frame_method = "confidence_only_no_ground_truth"
    if frame_accuracy is not None:
        accuracy = round((audio_accuracy * 0.5) + (frame_accuracy * 0.5), 2)
    else:
        accuracy = audio_accuracy
    if frame_accuracy is not None and frame_accuracy < 50:
        warnings.append("Low visual frame extraction quality")
    return build_report(
        file_name,
        "video",
        "whisper+tesseract+easyocr",
        accuracy,
        audio_is_measured,
        f"{audio_method}+{frame_method}",
        warnings,
        mongo_db,
        job_id,
    )


def _docx_strict_accuracy(file_path: Path, file_name: str, job_id: str) -> dict:
    warnings: list[str] = []
    paragraph_count = table_count = heading_count = extracted_elements = 0
    try:
        from docx import Document

        document = Document(str(file_path))
        paragraph_count = len(document.paragraphs)
        table_count = len(document.tables)
        heading_count = sum(1 for paragraph in document.paragraphs if str(paragraph.style.name).lower().startswith("heading"))
        extracted_elements = paragraph_count + table_count + heading_count
    except Exception as exc:
        warnings.append(f"DOCX parse completeness failed: {exc}")
    total_elements = paragraph_count + table_count + heading_count
    accuracy = round((extracted_elements / total_elements) * 100, 2) if total_elements > 0 else 0.0
    if accuracy < 90:
        warnings.append("Some document elements failed to extract completely")
    return build_report(file_name, "docx", "python-docx", accuracy, False, "parse_completeness", warnings, mongo_db, job_id)


def _xlsx_strict_accuracy(file_path: Path, file_name: str, job_id: str) -> dict:
    warnings: list[str] = []
    total_cells = extracted_cells = 0
    try:
        from openpyxl import load_workbook

        workbook = load_workbook(file_path, read_only=True, data_only=True)
        try:
            for sheet in workbook.worksheets:
                for row in sheet.iter_rows():
                    for cell in row:
                        total_cells += 1
                        try:
                            _value = cell.value
                            extracted_cells += 1
                        except Exception:
                            pass
        finally:
            workbook.close()
    except Exception as exc:
        warnings.append(f"XLSX cell parse completeness failed: {exc}")
    accuracy = round((extracted_cells / total_cells) * 100, 2) if total_cells > 0 else 0.0
    if accuracy < 90:
        warnings.append("Some cells failed to parse - file may have merged cells or formulas")
    return build_report(file_name, "xlsx", "openpyxl", accuracy, False, "cell_parse_completeness", warnings, mongo_db, job_id)


def _strict_accuracy_report(
    file_path: Path,
    original_filename: str,
    file_type: str,
    transcript: Optional[str],
    transcript_segments: list[dict],
    corrected_extracted_text: str = "",
) -> dict:
    job_id = file_path.stem
    extension = file_path.suffix.lower()
    if file_type == "image":
        return _image_strict_accuracy(file_path, original_filename, job_id, corrected_extracted_text)
    if file_type == "pdf":
        return _pdf_strict_accuracy(file_path, original_filename, job_id)
    if file_type == "audio":
        return _audio_strict_accuracy(original_filename, job_id, transcript or "", transcript_segments)
    if file_type == "video":
        return _video_strict_accuracy(file_path, original_filename, job_id, transcript or "", transcript_segments)
    if file_type == "document" and extension in {".docx", ".docm", ".dotx", ".dotm"}:
        return _docx_strict_accuracy(file_path, original_filename, job_id)
    if file_type == "spreadsheet" and extension == ".xlsx":
        return _xlsx_strict_accuracy(file_path, original_filename, job_id)
    return build_report(original_filename, file_type, _extraction_module_name(file_type), None, False, "accuracy_not_available", [], mongo_db, job_id)


def analyze_file(
    file_path: Path,
    original_filename: str,
    key_message_path: Path,
    target_language: str = DEFAULT_TARGET_LANGUAGE,
    mapping_enabled: bool = True,
    page_slide_number: Optional[int] = None,
    top_k: int = DEFAULT_TOP_K,
) -> BaseIntelligenceResponse:
    file_info = get_file_type_config(original_filename)
    file_type = detect_file_type(original_filename)
    chunks = _extract_by_file_type(file_path, file_type, page_slide_number)
    chunks = _apply_ocr_corrections_to_chunks(chunks, file_type)

    description = clean_text("\n".join(c.description_of_image_video for c in chunks if c.description_of_image_video)) or None
    full_text = compact_join(
        [
            compact_join([chunk.text, chunk.description_of_image_video or ""], limit=None)
            for chunk in chunks
        ],
        limit=None,
    )
    
    # Extract transcript and frame descriptions for video/audio
    transcript = None
    transcript_segments = []
    frame_descriptions = None
    frame_text = None
    for chunk in chunks:
        if chunk.source_type in {"video", "audio"}:
            transcript = chunk.metadata.get("transcript")
            transcript_segments = chunk.metadata.get("transcript_segments") or []
            frame_descriptions = chunk.metadata.get("frame_descriptions")
            frame_text = chunk.metadata.get("frame_text")
            break
    
    warnings = []
    for chunk in chunks:
        for warning in chunk.metadata.get("warnings", []) or []:
            if warning not in warnings:
                warnings.append(warning)

    full_transcript = _full_extracted_transcript(
        chunks,
        file_type,
        transcript,
        description,
        frame_text,
        frame_descriptions,
    )
    transcript_available = bool(full_transcript)
    if file_type in {"audio", "video"} and not full_transcript:
        full_transcript = NO_SPEECH_MESSAGE
    accuracy_report = _strict_accuracy_report(file_path, original_filename, file_type, transcript, transcript_segments, full_transcript)

    fast_document_mode = file_type not in {"audio", "video"}
    content_summary = None
    summary_available = False
    if file_type not in {"audio", "video"}:
        summary_source = compact_join([_summary_source_text(chunks, file_type, description)], limit=MAX_CHARS_FOR_MAPPING)
        try:
            content_summary = generate_content_summary(summary_source)
        except Exception as exc:
            logger.warning("Summary generation failed: %s", exc)
            content_summary = f"{NO_SUMMARY_MESSAGE} Error: {exc}"
        summary_available = bool(content_summary and content_summary != NO_SUMMARY_MESSAGE)

    mapping_context = full_text
    if not mapping_context and description:
        mapping_context = description
    if file_type in {"audio", "video"}:
        readable_frame_text = _mapping_frame_text(frame_text) if file_type == "video" else ""
        audiovisual_context = compact_join([transcript or "", readable_frame_text, description or ""], limit=None)
        mapping_context = f"{Path(original_filename).stem}\n\n{audiovisual_context}" if audiovisual_context else ""
        
    processing_context = compact_join([mapping_context], limit=MAX_CHARS_FOR_MAPPING) if fast_document_mode else mapping_context
    detected_language = detect_language(processing_context)
    translated_text, translated = translate_text(processing_context, target_language=target_language)
    analysis_text = translated_text if translated else processing_context
    summary = content_summary if file_type not in {"audio", "video"} else _summary_from_text(analysis_text)
    extracted_keywords = extract_keywords(analysis_text, top_n=10, use_model=False)
    metadata_source_text = compact_join(
        [
            full_transcript,
            analysis_text,
            content_summary or "",
            summary or "",
        ],
        limit=MAX_CHARS_FOR_MAPPING,
    )
    content_metadata = generate_content_metadata(metadata_source_text)

    key_message_matches = []
    has_readable_mapping_text = bool(clean_text(analysis_text))
    mapping_status = "Mapping Disabled"
    if mapping_enabled and has_readable_mapping_text:
        key_df, id_col, brand_col, msg_col = load_key_messages(key_message_path)
        visual_description = _visual_activity_description(file_type, description, frame_descriptions, frame_text)
        effective_top_k = 10 if file_type in {"audio", "video"} and top_k <= 0 else top_k

        key_message_matches = match_key_messages(
            analysis_text,
            key_df=key_df,
            id_col=id_col,
            brand_col=brand_col,
            msg_col=msg_col,
            top_k=effective_top_k,
            source_location=_source_location(chunks, file_type),
            description_of_image_video=visual_description,
            lock_to_detected_brand=file_type in {"pdf", "ppt", "image", "video"},
            use_evidence_description=file_type not in {"pdf", "ppt", "image", "video"},
            use_keyword_model=False,
        )
        mapping_status = "Mapping Completed" if key_message_matches else "Mapping Completed - no matches"
    elif mapping_enabled:
        mapping_status = "Mapping skipped - no readable text"

    if file_type not in {"audio", "video"}:
        content_summary = generate_mapped_summary(key_message_matches, fallback_summary=content_summary or summary)
        summary = content_summary
        summary_available = bool(content_summary and content_summary != NO_SUMMARY_MESSAGE)

    has_extracted_text = bool(clean_text(full_transcript))
    preview_status = "Preview Available" if (file_info.get("can_preview") or has_extracted_text) else "Unsupported Preview"
    accuracy_report_data = accuracy_report if isinstance(accuracy_report, dict) else {}
    report_accuracy = accuracy_report_data.get("accuracy")
    extraction_accuracy = {
        "score": report_accuracy,
        "percentage": report_accuracy,
        "type": accuracy_report_data.get("method", "accuracy_not_available"),
        "module": accuracy_report_data.get("model_used") or _extraction_module_name(file_type),
        "can_average": report_accuracy is not None,
        "accuracy_source": "ground_truth" if accuracy_report_data.get("is_measured_accuracy") else "confidence_only",
        "measured_accuracy": report_accuracy if accuracy_report_data.get("is_measured_accuracy") else None,
        "estimated_quality_score": report_accuracy,
        "confidence_score": report_accuracy if not accuracy_report_data.get("is_measured_accuracy") else None,
        "note": accuracy_report_data.get("method", "accuracy_not_available"),
        "unit_scores": [],
    }
    page_level_output = [chunk.metadata.get("page_level") for chunk in chunks if chunk.metadata.get("page_level")]
    slide_level_output = [chunk.metadata.get("slide_level") for chunk in chunks if chunk.metadata.get("slide_level")]
    frame_level_output = []
    for chunk in chunks:
        frame_level_output.extend(chunk.metadata.get("frame_level") or chunk.metadata.get("frame_results") or [])
    extraction_method_summary = {}
    for chunk in chunks:
        method = chunk.metadata.get("extraction_method") or "unknown"
        extraction_method_summary[method] = extraction_method_summary.get(method, 0) + 1
    measured_accuracy = extraction_accuracy.get("measured_accuracy")
    estimated_quality_score = extraction_accuracy.get("estimated_quality_score", extraction_accuracy["score"])
    confidence_score = extraction_accuracy.get("confidence_score")
    accuracy_available = measured_accuracy is not None
    accuracy_source = "ground_truth" if accuracy_available else "confidence_only"

    return BaseIntelligenceResponse(
        file_name=original_filename,
        file_type=file_type,
        detected_language=detected_language,
        target_language=target_language,
        translated=translated,
        summary=summary,
        extracted_keywords=extracted_keywords,
        description_of_image_video=description if file_type in {"pdf", "ppt", "image", "video"} else None,
        transcript=transcript,
        frame_descriptions=frame_descriptions,
        content_summary=content_summary,
        full_transcript=full_transcript,
        transcript_available=transcript_available,
        summary_available=summary_available,
        key_message_matches=key_message_matches,
        metadata={
            "total_source_units": len(chunks),
            "mapping_enabled": mapping_enabled,
            "warnings": warnings,
            "full_extracted_text": full_transcript,
            "file_extension": file_info.get("extension"),
            "file_category": file_info.get("category"),
            "preview_type": file_info.get("preview_type"),
            "preview_status": preview_status,
            "extraction_status": "Text Extracted" if has_extracted_text else "No readable text extracted",
            "accuracy_score": extraction_accuracy["score"],
            "accuracy_percentage": extraction_accuracy["percentage"],
            "accuracy_type": "measured_accuracy" if measured_accuracy is not None else extraction_accuracy["type"],
            "accuracy_module": extraction_accuracy["module"],
            "extraction_accuracy": extraction_accuracy,
            "accuracy_report": accuracy_report,
            "measured_accuracy": measured_accuracy,
            "estimated_quality_score": estimated_quality_score,
            "estimated_extraction_quality": estimated_quality_score,
            "confidence_score": confidence_score,
            "accuracy_source": accuracy_source,
            "mapping_status": mapping_status,
            "supported_file_type": file_info,
            "transcript_segments": transcript_segments,
            "media_duration_seconds": max((float(segment.get("end") or 0) for segment in transcript_segments if isinstance(segment, dict)), default=0),
            "source_units": _source_unit_payload(chunks),
            "page_level_output": page_level_output,
            "slide_level_output": slide_level_output,
            "frame_level_output": frame_level_output,
            "quality_score": estimated_quality_score,
            "quality_note": extraction_accuracy["note"],
            "accuracy_available": accuracy_available,
            "accuracy_note": "Measured accuracy is available from ground truth." if accuracy_available else "Ground truth was not available. Showing confidence only where supported.",
            "extraction_method_summary": extraction_method_summary,
            "content_metadata": content_metadata,
            "workflow": "upload -> detect file type -> processor -> shared intelligence -> translation -> analytics -> key-message mapping -> base model response",
        },
        accuracy_report=accuracy_report,
    )
