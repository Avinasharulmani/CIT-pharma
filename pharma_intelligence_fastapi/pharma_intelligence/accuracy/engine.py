from __future__ import annotations

import math
import re
from pathlib import Path
from statistics import mean
from typing import Optional

from ..ocr.open_weight_vlm_ocr import extract_text_open_weight_vlm, is_open_weight_vlm_available


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def normalize_text(text: str) -> str:
    value = str(text or "").lower()
    value = re.sub(r"[^a-z0-9 ]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _edit_distance(reference: list[str] | str, hypothesis: list[str] | str) -> int:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    previous = list(range(cols))
    for row in range(1, rows):
        current = [row] + [0] * (cols - 1)
        for col in range(1, cols):
            substitution_cost = 0 if reference[row - 1] == hypothesis[col - 1] else 1
            current[col] = min(
                previous[col] + 1,
                current[col - 1] + 1,
                previous[col - 1] + substitution_cost,
            )
        previous = current
    return previous[-1]


def compute_cer(reference: str, hypothesis: str) -> float:
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref and not hyp:
        return 0.0
    if not ref:
        return 1.0
    return _clamp(_edit_distance(ref, hyp) / len(ref))


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_tokens = normalize_text(reference).split()
    hyp_tokens = normalize_text(hypothesis).split()
    if not ref_tokens and not hyp_tokens:
        return 0.0
    if not ref_tokens:
        return 1.0
    return _clamp(_edit_distance(ref_tokens, hyp_tokens) / len(ref_tokens))


def get_tesseract_confidence(image_path: str) -> float:
    try:
        import pytesseract
        from pytesseract import Output

        data = pytesseract.image_to_data(str(image_path), output_type=Output.DICT)
    except Exception:
        return 0.0
    values = []
    for value in data.get("conf", []) or []:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            continue
        if confidence != -1:
            values.append(confidence)
    return _clamp(mean(values) / 100.0) if values else 0.0


def get_easyocr_confidence(image_path: str) -> tuple[str, float]:
    try:
        import easyocr

        results = easyocr.Reader(["en"], gpu=False, verbose=False).readtext(str(image_path))
    except Exception:
        return "", 0.0
    if not results:
        return "", 0.0
    text = " ".join(str(result[1]) for result in results if len(result) > 1).strip()
    values = []
    for result in results:
        try:
            values.append(float(result[2]))
        except (TypeError, ValueError, IndexError):
            continue
    return text, _clamp(mean(values)) if values else 0.0


def get_tesseract_text(image_path: str) -> str:
    try:
        import pytesseract

        return str(pytesseract.image_to_string(str(image_path)) or "").strip()
    except Exception:
        return ""


def get_best_ocr_text(image_path: str) -> tuple[str, float, str]:
    if is_open_weight_vlm_available():
        text, confidence = extract_text_open_weight_vlm(image_path)
        if text.strip() and confidence > 0:
            return text, confidence, "open_weight_vlm"
    text = get_tesseract_text(image_path)
    confidence = get_tesseract_confidence(image_path)
    return text, confidence, "tesseract_fallback"


def normalize_whisper_confidence(segments: list) -> float:
    signals = []
    for segment in segments or []:
        try:
            no_speech_prob = float(segment["no_speech_prob"])
            avg_logprob = float(segment["avg_logprob"])
        except (KeyError, TypeError, ValueError):
            continue
        signals.append(_clamp((1.0 - no_speech_prob) * min(1.0, math.exp(avg_logprob + 1.0))))
    return _clamp(mean(signals)) if signals else 0.0


def load_reference_file(job_id: str, upload_dir: str) -> Optional[str]:
    path = Path(upload_dir) / str(job_id) / "ground_truth.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _plain_srt_text(value: str) -> str:
    lines = []
    for line in value.splitlines():
        item = line.strip()
        if not item or item.isdigit() or "-->" in item:
            continue
        lines.append(item)
    return " ".join(lines).strip()


def load_reference_transcript(job_id: str, upload_dir: str) -> Optional[str]:
    folder = Path(upload_dir) / str(job_id)
    txt_path = folder / "reference_transcript.txt"
    srt_path = folder / "reference_transcript.srt"
    if txt_path.exists():
        return txt_path.read_text(encoding="utf-8", errors="ignore").strip()
    if srt_path.exists():
        return _plain_srt_text(srt_path.read_text(encoding="utf-8", errors="ignore"))
    return None


def normalize_tesseract_confidence(ocr_data: dict) -> float:
    values = []
    for value in (ocr_data or {}).get("conf", []) or []:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            continue
        if confidence != -1:
            values.append(confidence)
    return _clamp(mean(values) / 100.0) if values else 0.0


def normalize_easyocr_confidence(results: list) -> float:
    values = []
    for result in results or []:
        try:
            values.append(float(result[2]))
        except (TypeError, ValueError, IndexError):
            continue
    return _clamp(mean(values)) if values else 0.0
