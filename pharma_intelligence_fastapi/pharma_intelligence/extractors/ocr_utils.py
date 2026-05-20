import importlib.util
import logging
import re
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

from ..config import IMAGE_EASYOCR_ENABLED
from ..utils import clean_text
from .text_quality import merge_text_blocks, text_quality_score

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str = ""
    engine: str = ""
    selected_variant: str = ""
    confidence_score: float | None = None
    text_length: int = 0
    word_count: int = 0
    quality_score: float = 0.0
    warnings: list[str] | None = None

    def as_metadata(self) -> dict[str, Any]:
        return {
            "ocr_text": self.text,
            "ocr_engine": self.engine,
            "selected_variant": self.selected_variant,
            "confidence_score": self.confidence_score,
            "text_length": self.text_length,
            "word_count": self.word_count,
            "text_quality_score": self.quality_score,
            "warnings": self.warnings or [],
        }


def ocr_available() -> bool:
    return bool(shutil.which("tesseract") or (IMAGE_EASYOCR_ENABLED and importlib.util.find_spec("easyocr")))


def preprocess_image_variants(image) -> list[tuple[str, Any]]:
    from PIL import ImageEnhance, ImageFilter, ImageOps

    rgb = image.convert("RGB")
    max_side = max(rgb.size)
    scale = 2 if max_side < 1800 else 1
    enlarged = rgb.resize((rgb.width * scale, rgb.height * scale))
    gray = ImageOps.grayscale(enlarged)
    contrast = ImageEnhance.Contrast(gray).enhance(1.8)
    sharpened = contrast.filter(ImageFilter.SHARPEN)
    denoised = sharpened.filter(ImageFilter.MedianFilter(size=3))
    threshold = denoised.point(lambda pixel: 255 if pixel > 165 else 0)
    adaptive = _adaptive_threshold(denoised)
    variants = [
        ("original", rgb),
        ("enlarged", enlarged),
        ("grayscale", gray),
        ("contrast", contrast),
        ("sharpened", sharpened),
        ("denoised", denoised),
        ("threshold", threshold),
    ]
    if adaptive is not None:
        variants.append(("adaptive_threshold", adaptive))
    return variants


def _adaptive_threshold(image):
    try:
        import cv2
        import numpy as np
        from PIL import Image

        array = np.array(image.convert("L"))
        threshold = cv2.adaptiveThreshold(
            array,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            11,
        )
        return Image.fromarray(threshold)
    except Exception:
        return None


def _normalize_ocr_line(value: str) -> str:
    return re.sub(r"\W+", " ", clean_text(value).lower()).strip()


def _ocr_tesseract_variant(image, variant_name: str, config: str) -> OCRResult:
    try:
        import pytesseract
        from pytesseract import Output
    except Exception as exc:
        return OCRResult(warnings=[f"Tesseract unavailable: {exc}"])

    warnings: list[str] = []
    lines: list[str] = []
    confidence_values: list[float] = []
    try:
        data = pytesseract.image_to_data(image, config=config, output_type=Output.DICT)
        grouped: dict[tuple[int, int, int], list[str]] = {}
        for index, word in enumerate(data.get("text", [])):
            word = clean_text(str(word)).strip()
            if not word:
                continue
            try:
                confidence = float(data.get("conf", [])[index])
            except Exception:
                confidence = -1.0
            if confidence >= 0:
                confidence_values.append(confidence)
            if confidence < 35:
                continue
            key = (
                int(data.get("block_num", [0])[index] or 0),
                int(data.get("par_num", [0])[index] or 0),
                int(data.get("line_num", [0])[index] or 0),
            )
            grouped.setdefault(key, []).append(word)
        for key in sorted(grouped):
            line = clean_text(" ".join(grouped[key]))
            if line:
                lines.append(line)
    except Exception as exc:
        warnings.append(f"Tesseract data OCR failed for {variant_name}: {exc}")

    try:
        raw_text = pytesseract.image_to_string(image, config=config)
        lines.extend(line for line in clean_text(raw_text).splitlines() if clean_text(line))
    except Exception as exc:
        warnings.append(f"Tesseract string OCR failed for {variant_name}: {exc}")

    seen = set()
    unique_lines = []
    for line in lines:
        key = _normalize_ocr_line(line)
        if key and key not in seen:
            seen.add(key)
            unique_lines.append(line)
    text = clean_text("\n".join(unique_lines))
    confidence = round(sum(confidence_values) / len(confidence_values), 2) if confidence_values else None
    return _scored_result(
        text=text,
        engine="tesseract",
        selected_variant=variant_name,
        confidence_score=confidence,
        warnings=warnings,
    )


@lru_cache(maxsize=1)
def _get_easyocr_reader():
    if not IMAGE_EASYOCR_ENABLED:
        return None
    try:
        import easyocr

        return easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception:
        return None


def _ocr_easyocr_image(image) -> OCRResult:
    if not IMAGE_EASYOCR_ENABLED:
        return OCRResult()
    reader = _get_easyocr_reader()
    if reader is None:
        return OCRResult()
    try:
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "ocr.png"
            image.convert("RGB").save(image_path)
            detail = reader.readtext(str(image_path), detail=1, paragraph=False)
            texts = []
            confidences = []
            for item in detail:
                if len(item) >= 2 and clean_text(str(item[1])):
                    texts.append(str(item[1]))
                if len(item) >= 3:
                    try:
                        confidences.append(float(item[2]) * 100.0)
                    except Exception:
                        pass
            confidence = round(sum(confidences) / len(confidences), 2) if confidences else None
            return _scored_result(
                text=merge_text_blocks("\n".join(texts)),
                engine="easyocr",
                selected_variant="easyocr_original",
                confidence_score=confidence,
            )
    except Exception as exc:
        logger.debug("EasyOCR failed: %s", exc)
        return OCRResult(warnings=[f"EasyOCR failed: {exc}"])


def _scored_result(
    *,
    text: str,
    engine: str,
    selected_variant: str,
    confidence_score: float | None = None,
    warnings: list[str] | None = None,
) -> OCRResult:
    cleaned = clean_text(text)
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9%+./-]*", cleaned)
    return OCRResult(
        text=cleaned,
        engine=engine,
        selected_variant=selected_variant,
        confidence_score=confidence_score,
        text_length=len(cleaned),
        word_count=len(words),
        quality_score=text_quality_score(cleaned),
        warnings=warnings or [],
    )


def _selection_score(result: OCRResult) -> float:
    confidence = result.confidence_score if result.confidence_score is not None else 45.0
    length_score = min(30.0, result.text_length / 12.0)
    word_score = min(20.0, result.word_count * 1.5)
    quality = result.quality_score * 0.35
    return confidence * 0.45 + length_score + word_score + quality


def choose_best_ocr_result(results: Iterable[OCRResult]) -> OCRResult:
    candidates = [result for result in results if clean_text(result.text)]
    if not candidates:
        warnings = []
        for result in results:
            warnings.extend(result.warnings or [])
        return OCRResult(warnings=warnings)
    best = max(candidates, key=_selection_score)
    if best.confidence_score is not None and best.confidence_score < 45:
        best.warnings = [*(best.warnings or []), "low OCR confidence"]
    return best


def run_ocr_on_image(image, *, configs: list[str] | None = None, fast: bool = False) -> OCRResult:
    if configs is None:
        configs = ["--psm 6"] if fast else ["--psm 6", "--psm 11", "--psm 4"]
    results: list[OCRResult] = []
    if shutil.which("tesseract"):
        variants = preprocess_image_variants(image)
        if fast:
            preferred = {"enlarged", "contrast", "threshold"}
            variants = [(name, variant) for name, variant in variants if name in preferred]
        for variant_name, variant in variants:
            for config in configs:
                result = _ocr_tesseract_variant(variant, variant_name, config)
                if result.text:
                    results.append(result)
    else:
        results.append(OCRResult(warnings=["Tesseract OCR is not available on PATH."]))
    # EasyOCR is stronger but slower. In fast mode, use it only when Tesseract found no text.
    if not fast or not any(result.text for result in results):
        easyocr_result = _ocr_easyocr_image(image)
        if easyocr_result.text or easyocr_result.warnings:
            results.append(easyocr_result)
    return choose_best_ocr_result(results)
