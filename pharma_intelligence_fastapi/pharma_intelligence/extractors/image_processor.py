import shutil
import re
import importlib.util
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

try:
    import pillow_avif  # noqa: F401
except Exception:
    pass

from ..config import (
    IMAGE_DESCRIPTION_ENABLED,
    IMAGE_EASYOCR_ENABLED,
    IMAGE_OCR_MODE,
    LOCAL_IMAGE_DESCRIPTION_ENABLED,
)
from ..models import SourceChunk
from ..utils import clean_text
from ..vision_activity import describe_visible_activity, pil_image_to_data_url
from .ocr_utils import ocr_available, run_ocr_on_image
from .text_quality import extraction_unit_metadata


def _ocr_image_tesseract(image) -> str:
    result = run_ocr_on_image(image)
    if result.engine == "tesseract":
        return result.text
    if result.text:
        return result.text
    try:
        import pytesseract
        from pytesseract import Output
        from PIL import ImageFilter, ImageOps

        rgb_image = image.convert("RGB")
        gray_image = ImageOps.grayscale(rgb_image)
        enhanced_image = ImageOps.autocontrast(gray_image).filter(ImageFilter.SHARPEN)
        threshold_image = enhanced_image.point(lambda pixel: 255 if pixel > 170 else 0)

        if IMAGE_OCR_MODE == "thorough":
            variants = [
                rgb_image,
                rgb_image.resize((rgb_image.width * 2, rgb_image.height * 2)),
                enhanced_image.resize((enhanced_image.width * 2, enhanced_image.height * 2)),
                threshold_image.resize((threshold_image.width * 2, threshold_image.height * 2)),
            ]
            configs = ["--psm 6", "--psm 11"]
        else:
            variants = [enhanced_image.resize((enhanced_image.width * 2, enhanced_image.height * 2))]
            configs = ["--psm 6"]
        confidence_lines = []
        raw_lines = []
        seen = set()

        def add_line(value: str, target: list[str]) -> None:
            value = clean_text(value).strip()
            normalized = re.sub(r"\W+", " ", value.lower()).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                target.append(value)

        for variant in variants:
            for config in configs:
                try:
                    data = pytesseract.image_to_data(variant, config=config, output_type=Output.DICT)
                    grouped: dict[tuple[int, int, int], list[str]] = {}
                    for index, word in enumerate(data.get("text", [])):
                        word = clean_text(str(word)).strip()
                        if not word:
                            continue
                        try:
                            confidence = float(data.get("conf", [])[index])
                        except Exception:
                            confidence = -1
                        if confidence < 42:
                            continue
                        key = (
                            int(data.get("block_num", [0])[index] or 0),
                            int(data.get("par_num", [0])[index] or 0),
                            int(data.get("line_num", [0])[index] or 0),
                        )
                        grouped.setdefault(key, []).append(word)
                    for key in sorted(grouped):
                        add_line(" ".join(grouped[key]), confidence_lines)
                except Exception:
                    pass

                text = pytesseract.image_to_string(variant, config=config)
                for line in clean_text(text).splitlines():
                    add_line(line, raw_lines)
        return clean_text("\n".join(confidence_lines + raw_lines))
    except Exception:
        return ""


@lru_cache(maxsize=1)
def _get_easyocr_reader():
    try:
        import easyocr

        return easyocr.Reader(["en"], gpu=False, verbose=False)
    except Exception:
        return None


def _ocr_image_easyocr(path: Path) -> str:
    if not IMAGE_EASYOCR_ENABLED:
        return ""
    reader = _get_easyocr_reader()
    if reader is None:
        return ""
    try:
        result = reader.readtext(str(path), detail=0, paragraph=False)
        return clean_text("\n".join(str(item) for item in result if item))
    except Exception:
        return ""


def _ocr_image(path: Path, image) -> tuple[str, str]:
    result = run_ocr_on_image(image)
    return result.text, result.engine


def _ocr_pil_image(image) -> tuple[str, str]:
    result = run_ocr_on_image(image)
    return result.text, result.engine


def _document_image_segments(image) -> list[tuple[int, object]]:
    width, height = image.size
    if width <= 0 or height / max(width, 1) < 2.2:
        return [(1, image)]
    try:
        import numpy as np
    except Exception:
        return [(1, image)]

    sample_width = 200
    gray = image.convert("L").resize((sample_width, height))
    rows = np.array(gray)
    ink_ratio = (rows < 230).mean(axis=1)
    window = 15
    smoothed = np.convolve(ink_ratio, np.ones(window) / window, mode="same")
    min_gap_height = max(28, int(height * 0.004))
    min_segment_height = max(260, int(width * 0.35))
    gaps: list[tuple[int, int]] = []
    start = None
    for row_index, value in enumerate(smoothed):
        if value < 0.005:
            if start is None:
                start = row_index
        elif start is not None:
            if row_index - start >= min_gap_height:
                gaps.append((start, row_index))
            start = None
    if start is not None and height - start >= min_gap_height:
        gaps.append((start, height))

    boundaries = [0]
    last_boundary = 0
    for start, end in gaps:
        center = (start + end) // 2
        if center - last_boundary < min_segment_height:
            continue
        if height - center < min_segment_height // 2:
            continue
        boundaries.append(center)
        last_boundary = center
    boundaries.append(height)

    if len(boundaries) <= 2:
        return [(1, image)]
    segments = []
    for index, (top, bottom) in enumerate(zip(boundaries, boundaries[1:]), start=1):
        if bottom - top < 80:
            continue
        segments.append((index, image.crop((0, top, width, bottom))))
    return segments or [(1, image)]


@lru_cache(maxsize=1)
def _local_blip_vqa():
    try:
        from transformers import BlipForQuestionAnswering, BlipProcessor

        model_name = "Salesforce/blip-vqa-base"
        processor = BlipProcessor.from_pretrained(model_name, local_files_only=True)
        model = BlipForQuestionAnswering.from_pretrained(model_name, local_files_only=True)
        return processor, model
    except Exception:
        return None


@lru_cache(maxsize=1)
def _local_blip_captioner():
    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        model_name = "Salesforce/blip-image-captioning-base"
        processor = BlipProcessor.from_pretrained(model_name, local_files_only=True)
        model = BlipForConditionalGeneration.from_pretrained(model_name, local_files_only=True)
        return processor, model
    except Exception:
        return None


def _ask_local_vlm(image, question: str) -> str:
    vlm = _local_blip_vqa()
    if vlm is None:
        return ""
    try:
        processor, model = vlm
        rgb = image.convert("RGB")
        inputs = processor(rgb, question, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=24)
        return clean_text(processor.decode(output[0], skip_special_tokens=True))
    except Exception:
        return ""


def _caption_local_vlm(image) -> str:
    vlm = _local_blip_captioner()
    if vlm is None:
        return ""
    try:
        processor, model = vlm
        rgb = image.convert("RGB")
        inputs = processor(rgb, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=36)
        return clean_text(processor.decode(output[0], skip_special_tokens=True))
    except Exception:
        return ""


def _is_weak_vlm_answer(answer: str) -> bool:
    answer = clean_text(answer)
    if not answer:
        return True
    normalized = re.sub(r"\W+", " ", answer.lower()).strip()
    if len(normalized) <= 2:
        return True
    if normalized in {
        "no",
        "none",
        "nothing",
        "unknown",
        "unclear",
        "n a",
        "na",
        "image",
        "picture",
        "photo",
        "a picture",
        "an image",
        "a photo",
    }:
        return True
    if any(
        phrase in normalized
        for phrase in (
            "no idea",
            "do not know",
            "don t know",
            "i don t know",
            "not sure",
            "cannot tell",
            "can t tell",
        )
    ):
        return True
    if len(normalized.split()) == 1 and normalized in {
        "ad",
        "advertisement",
        "diet",
        "food",
        "graphic",
        "health",
        "poster",
        "product",
        "text",
    }:
        return True
    weak_phrases = {
        "a blurry image",
        "a close up",
        "a screenshot",
        "a document",
        "a poster",
        "text",
        "words",
    }
    return normalized in weak_phrases


def _normalize_vlm_caption(answer: str) -> str:
    answer = clean_text(answer)
    if _is_weak_vlm_answer(answer):
        return ""
    if not answer.lower().startswith(("a ", "an ", "the ", "visible ", "people ", "person ")):
        answer = f"Visible content shows {answer}"
    return _sentence(answer)


def _sentence(value: str) -> str:
    value = clean_text(value).strip(" .")
    if not value:
        return ""
    return value[0].upper() + value[1:] + "."


def _normalize_vlm_activity(answer: str) -> str:
    answer = clean_text(answer)
    if _is_weak_vlm_answer(answer):
        return ""
    answer_lower = answer.lower().strip(" .")
    if answer_lower.startswith(("a person ", "people ", "a woman ", "a man ", "an illustration ")):
        sentence = answer
    elif answer_lower.endswith("ing") or answer_lower in {"sitting", "standing", "walking", "presenting", "talking"}:
        sentence = f"A person is {answer_lower}"
    else:
        sentence = f"Visible activity: {answer}"
    if not sentence.endswith("."):
        sentence += "."
    if len(sentence) > 220:
        sentence = sentence[:220].rsplit(" ", 1)[0] + "..."
    return sentence


def _normalize_vlm_scene(answer: str) -> str:
    answer = clean_text(answer)
    if _is_weak_vlm_answer(answer):
        return ""
    answer_lower = answer.lower().strip(" .")
    sentence = answer
    if not sentence.lower().startswith(("a ", "an ", "the ", "person ", "people ", "visible ")):
        sentence = f"Visible content: {sentence}"
    if not sentence.endswith("."):
        sentence += "."
    if len(sentence) > 220:
        sentence = sentence[:220].rsplit(" ", 1)[0] + "..."
    return sentence


def _normalize_visible_value(answer: str) -> str:
    answer = clean_text(answer).strip(" .")
    if _is_weak_vlm_answer(answer):
        return ""
    return answer


def _local_vlm_caption_description(image) -> str:
    return _normalize_vlm_caption(_caption_local_vlm(image))


def _ocr_lines_for_description(ocr_text: Optional[str]) -> List[str]:
    lines: List[str] = []
    seen = set()
    for raw_line in clean_text(ocr_text or "").splitlines():
        line = clean_text(raw_line).strip(" .,:;|[]{}()")
        if not line:
            continue
        normalized = re.sub(r"\W+", " ", line.lower()).strip()
        letters = re.findall(r"[A-Za-z]", line)
        if len(letters) < 3 or not normalized or normalized in seen:
            continue
        seen.add(normalized)
        lines.append(line)
    return lines


def _line_looks_like_product_name(line: str) -> bool:
    words = re.findall(r"[A-Za-z][A-Za-z0-9%.-]*", line)
    if not words or len(words) > 7:
        return False
    normalized = " ".join(words).lower()
    if normalized in {
        "achieves superior clinical",
        "achieves superior reduction",
        "possesses good dermal penetration",
        "visible content",
    }:
        return False
    if re.search(r"\b(cream|tablet|capsule|injection|syrup|gel|ointment|spray|dose|mg|gm|ml|%)\b", normalized):
        return True
    if len(words) <= 3 and any(word[:1].isupper() for word in words):
        return True
    return False


def _product_title_from_ocr(lines: List[str]) -> str:
    for index, line in enumerate(lines[:14]):
        if not _line_looks_like_product_name(line):
            continue
        title = line
        if index + 1 < len(lines) and re.search(
            r"\b(cream|tablet|capsule|injection|syrup|gel|ointment|spray|dose|mg|gm|ml|%)\b",
            lines[index + 1],
            flags=re.IGNORECASE,
        ):
            title = f"{title} {lines[index + 1]}"
        return clean_text(title)
    return ""


def _looks_repetitive_or_ocr_noisy(description: str) -> bool:
    value = clean_text(description).lower()
    if not value:
        return True
    if "no idea" in value or "do not know" in value or "don't know" in value:
        return True
    tokens = re.findall(r"[a-z0-9]+", value)
    if len(tokens) < 4:
        return True
    unique_ratio = len(set(tokens)) / max(len(tokens), 1)
    if len(tokens) >= 8 and unique_ratio < 0.45:
        return True
    counts = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    if max(counts.values(), default=0) >= 4:
        return True
    return False


def _text_heavy_material_description(ocr_text: Optional[str]) -> str:
    lines = _ocr_lines_for_description(ocr_text)
    if len(lines) < 6:
        return ""

    text_lower = " ".join(lines).lower()
    title = _product_title_from_ocr(lines)
    visual_parts: List[str] = []

    if any(term in text_lower for term in ("chart", "graph", "score", "rate", "rates", "compared", "reduction")):
        visual_parts.append("comparison charts")
    if any(term in text_lower for term in ("patient", "patients", "clinical", "cure", "efficacy", "study")):
        visual_parts.append("clinical efficacy statements")
    if len(lines) >= 10:
        visual_parts.append("product claims")
    if any(term in text_lower for term in ("cream", "tablet", "capsule", "injection", "syrup", "gel", "ointment", "spray")):
        visual_parts.append("dosage details")

    seen_parts: List[str] = []
    for part in visual_parts:
        if part not in seen_parts:
            seen_parts.append(part)
    if not seen_parts:
        seen_parts.append("promotional content")

    subject = f" for {title}" if title else ""
    return f"Text-heavy promotional/clinical material{subject}, showing {', '.join(seen_parts[:3])}."


def _prefer_ocr_material_description(description: str, ocr_text: Optional[str]) -> str:
    ocr_description = _text_heavy_material_description(ocr_text)
    if not ocr_description:
        return description
    if not description or _looks_repetitive_or_ocr_noisy(description):
        return ocr_description
    return description


def _local_vlm_sees_person(image) -> bool:
    answers = [
        _ask_local_vlm(image, "Is a person visible in this image?"),
        _ask_local_vlm(image, "How many people are visible?"),
    ]
    normalized = [answer.lower().strip(" .") for answer in answers if answer]
    if any(answer in {"no", "none", "0", "zero"} for answer in normalized):
        return False
    return any(answer in {"yes", "1", "one"} or "person" in answer or "people" in answer for answer in normalized)


def _local_vlm_activity_description(image, require_person: bool = True) -> str:
    if require_person and not _local_vlm_sees_person(image):
        return ""

    questions = [
        "Describe the visible human activity or posture in this image.",
        "What is the person doing in this image?",
        "What activity is happening in the image?",
    ]
    for question in questions:
        description = _normalize_vlm_activity(_ask_local_vlm(image, question))
        if description:
            return description
    return ""


def _local_vlm_scene_description(image) -> str:
    questions = [
        "What is shown in this image?",
        "Describe the visible scene in this image.",
        "What are the main visible elements in this image?",
    ]
    for question in questions:
        description = _normalize_vlm_scene(_ask_local_vlm(image, question))
        if description:
            return description
    return ""


def _local_vlm_best_image_description(image, ocr_text: Optional[str] = None) -> str:
    ocr_description = _text_heavy_material_description(ocr_text)
    caption = _local_vlm_caption_description(image)
    probes = [
        ("main", "What is the main subject of this image?"),
        ("person_state", "Describe any visible person's appearance or posture in a short phrase."),
        ("scene", "Describe the visible scene in this image."),
        ("objects", "What important visible objects or packaging are shown?"),
        ("people", "What people are visible and what are they doing?"),
        ("setting", "What setting or environment is visible?"),
    ]
    answers = {}
    for key, question in probes:
        value = _normalize_visible_value(_ask_local_vlm(image, question))
        if value:
            answers[key] = value

    parts = []
    if caption:
        parts.append(caption)
    if answers.get("main"):
        parts.append(answers["main"])

    if answers.get("scene") and answers["scene"].lower() != answers.get("main", "").lower():
        parts.append(answers["scene"])

    detail_bits = []
    for key in ("person_state", "objects", "people", "setting"):
        value = answers.get(key)
        if value:
            normalized = re.sub(r"\W+", " ", value.lower()).strip()
            if normalized not in {re.sub(r"\W+", " ", item.lower()).strip() for item in parts}:
                detail_bits.append(value)
    if detail_bits:
        parts.append("; ".join(detail_bits[:3]))

    seen = set()
    sentences = []
    for part in parts:
        normalized = re.sub(r"\W+", " ", part.lower()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        sentence = _sentence(part)
        if sentence:
            sentences.append(sentence)
        if len(sentences) >= 2:
            break

    description = clean_text(" ".join(sentences))
    if len(description) > 340:
        description = description[:340].rsplit(" ", 1)[0] + "..."
    return _prefer_ocr_material_description(description, ocr_text) or ocr_description


def describe_pil_image_content(image, warnings: Optional[list[str]] = None, allow_local_fallback: bool = True) -> str:
    description = ""
    if IMAGE_DESCRIPTION_ENABLED:
        image_data_url = pil_image_to_data_url(image)
        description = describe_visible_activity(
            [image_data_url] if image_data_url else [],
            warnings=warnings,
            media_label="image",
        )
    if not description and allow_local_fallback and LOCAL_IMAGE_DESCRIPTION_ENABLED:
        description = _local_vlm_best_image_description(image) or _local_vlm_activity_description(
            image,
            require_person=False,
        ) or _local_vlm_scene_description(image)
    if description and warnings is not None:
        warnings[:] = [
            warning
            for warning in warnings
            if not str(warning).startswith("Exact visual activity description")
        ]
    return description


def extract_image(path: Path) -> List[SourceChunk]:
    try:
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("Pillow is required for image extraction. Install with: pip install pillow") from exc

    image = Image.open(str(path))
    segments = _document_image_segments(image)
    if len(segments) > 1:
        chunks: List[SourceChunk] = []
        for source_no, segment in segments:
            ocr_result = run_ocr_on_image(segment)
            ocr_text, ocr_engine = ocr_result.text, ocr_result.engine
            warnings = list(ocr_result.warnings or [])
            if not ocr_text:
                warnings.append("no OCR text found in image segment")
            chunks.append(
                SourceChunk(
                    source_no=source_no,
                    source_type="page",
                    text=ocr_text,
                    metadata={
                        **extraction_unit_metadata(
                            source_type="image_segment",
                            unit_number=source_no,
                            raw_text=ocr_text,
                            cleaned_text=ocr_text,
                            extraction_method=f"{ocr_engine or 'ocr'}:{ocr_result.selected_variant or 'none'}",
                            confidence_score=ocr_result.confidence_score,
                            warnings=warnings,
                        ),
                        "width": segment.size[0],
                        "height": segment.size[1],
                        "ocr_available": bool(ocr_text or ocr_available()),
                        "ocr_text_found": bool(ocr_text),
                        "ocr_engine": ocr_engine,
                        "ocr_text": ocr_text,
                        "selected_variant": ocr_result.selected_variant,
                        "confidence_score": ocr_result.confidence_score,
                        "image_segmented": True,
                        "warnings": warnings,
                    },
                )
            )
        return chunks

    ocr_result = run_ocr_on_image(image)
    ocr_text, ocr_engine = ocr_result.text, ocr_result.engine
    warnings = []
    warnings.extend(ocr_result.warnings or [])
    if not ocr_text:
        warnings.append("no OCR text found")
    description = describe_pil_image_content(image, warnings, allow_local_fallback=False)
    if not description and LOCAL_IMAGE_DESCRIPTION_ENABLED:
        description = (
            _local_vlm_best_image_description(image, ocr_text)
            or _local_vlm_activity_description(image, require_person=False)
            or _local_vlm_scene_description(image)
        )
    description = _prefer_ocr_material_description(description, ocr_text)
    if description:
        warnings = [warning for warning in warnings if not str(warning).startswith("Exact visual activity description")]
    has_ocr_available = bool(ocr_text or ocr_available())
    return [
        SourceChunk(
            source_no=1,
            source_type="image",
            text=ocr_text,
            description_of_image_video=description,
            metadata={
                **extraction_unit_metadata(
                    source_type="image",
                    unit_number=1,
                    raw_text=ocr_text,
                    cleaned_text=ocr_text,
                    extraction_method=f"{ocr_engine or 'ocr'}:{ocr_result.selected_variant or 'none'}",
                    confidence_score=ocr_result.confidence_score,
                    warnings=warnings,
                ),
                "width": image.size[0],
                "height": image.size[1],
                "ocr_available": has_ocr_available,
                "ocr_text_found": bool(ocr_text),
                "ocr_engine": ocr_engine,
                "ocr_text": ocr_text,
                "image_description": description,
                "selected_variant": ocr_result.selected_variant,
                "confidence_score": ocr_result.confidence_score,
                "warnings": warnings,
            },
        )
    ]
