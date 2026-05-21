from pathlib import Path
from typing import List, Optional
import io
import os
import tempfile

from ..models import SourceChunk
from ..preprocessing import preprocess_image
from ..utils import clean_text
from .image_processor import describe_pil_image_content
from .ocr_utils import run_ocr_on_image_path
from .pdf_quality import calculate_pdf_quality_score
from .text_quality import extraction_unit_metadata, is_weak_text, merge_text_blocks


PDF_TEXT_PAGE_IMAGE_DESCRIPTION_THRESHOLD = 80
PDF_WEAK_TEXT_CHARS = 80


def _pdf_page_image_descriptions(doc, page, warnings: list[str]) -> tuple[list[str], int]:
    try:
        from PIL import Image
    except Exception:
        return [], 0

    descriptions: list[str] = []
    image_refs = page.get_images(full=True)
    for image_index, image_ref in enumerate(image_refs, start=1):
        try:
            image_data = doc.extract_image(image_ref[0])
            image = Image.open(io.BytesIO(image_data["image"]))
            description = describe_pil_image_content(image, warnings, allow_local_fallback=True)
            if description:
                descriptions.append(f"Image {image_index}: {description}")
        except Exception:
            continue
    return descriptions, len(image_refs)


def _extract_page_text_pdfplumber(path: Path, page_index: int, warnings: list[str]) -> str:
    try:
        import pdfplumber
    except Exception:
        return ""
    try:
        with pdfplumber.open(str(path)) as pdf:
            if page_index >= len(pdf.pages):
                return ""
            return clean_text(pdf.pages[page_index].extract_text() or "")
    except Exception as exc:
        warnings.append(f"pdfplumber text extraction warning: {exc}")
        return ""


def _extract_page_text_pypdf2(path: Path, page_index: int, warnings: list[str]) -> str:
    try:
        from PyPDF2 import PdfReader
    except Exception:
        return ""
    try:
        reader = PdfReader(str(path))
        if page_index >= len(reader.pages):
            return ""
        return clean_text(reader.pages[page_index].extract_text() or "")
    except Exception as exc:
        warnings.append(f"PyPDF2 text extraction warning: {exc}")
        return ""


def _render_page_to_image(page, dpi: int = 144):
    from PIL import Image
    import fitz

    scale = dpi / 72.0
    pixmap = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
    return Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)


def extract_pdf(path: Path, page_number: Optional[int] = None) -> List[SourceChunk]:
    try:
        import fitz  # PyMuPDF
    except Exception as exc:
        raise RuntimeError("PyMuPDF is required for PDF extraction. Install with: pip install PyMuPDF") from exc

    chunks: List[SourceChunk] = []
    doc = fitz.open(str(path))
    try:
        for index, page in enumerate(doc, start=1):
            if page_number and index != page_number:
                continue
            warnings: list[str] = []
            pymupdf_text = clean_text(page.get_text("text"))
            pdfplumber_text = _extract_page_text_pdfplumber(path, index - 1, warnings)
            pypdf2_text = _extract_page_text_pypdf2(path, index - 1, warnings)
            native_text = max([pymupdf_text, pdfplumber_text, pypdf2_text], key=lambda value: len(clean_text(value)))
            ocr_text = ""
            ocr_result = None
            extraction_method = "native"
            page_image = _render_page_to_image(page, dpi=150)
            if is_weak_text(native_text, min_chars=PDF_WEAK_TEXT_CHARS):
                warnings.append("empty native text" if not native_text else "weak native text")
                warnings.append("scanned page detected")
                try:
                    with tempfile.TemporaryDirectory() as folder:
                        rasterized_page_path = Path(folder) / f"page_{index}.png"
                        page_image.save(rasterized_page_path)
                        preprocessed_path = preprocess_image(str(rasterized_page_path))
                        try:
                            ocr_result = run_ocr_on_image_path(preprocessed_path)
                        finally:
                            if preprocessed_path != str(rasterized_page_path):
                                try:
                                    os.remove(preprocessed_path)
                                except Exception:
                                    pass
                    ocr_text = ocr_result.text
                    if ocr_result.warnings:
                        warnings.extend(ocr_result.warnings)
                    extraction_method = "hybrid" if native_text and ocr_text else ("ocr" if ocr_text else "native")
                except Exception as exc:
                    warnings.append(f"PDF page OCR warning: {exc}")
            text = merge_text_blocks(native_text, ocr_text) if ocr_text else native_text
            image_count = len(page.get_images(full=True))
            image_descriptions: list[str] = []
            if len(text) < PDF_TEXT_PAGE_IMAGE_DESCRIPTION_THRESHOLD and image_count:
                image_descriptions, image_count = _pdf_page_image_descriptions(doc, page, warnings)
            quality = calculate_pdf_quality_score(
                text,
                extraction_method,
                {
                    "warnings": warnings,
                    "ocr_confidence": getattr(ocr_result, "confidence_score", None),
                },
            )
            page_level = {
                "page_number": index,
                "native_text": native_text,
                "ocr_text": ocr_text,
                "final_text": text,
                "extraction_method": extraction_method,
                "native_text_length": len(native_text),
                "ocr_text_length": len(ocr_text),
                "final_text_length": len(text),
                "quality_score": quality["quality_score"],
                "measured_accuracy": None,
                "accuracy_source": "strict_report_builder",
                "confidence_score": getattr(ocr_result, "confidence_score", None),
                "estimated_confidence_score": quality["confidence_score"],
                "noise_ratio": quality["noise_ratio"],
                "garbled_word_ratio": quality["garbled_word_ratio"],
                "word_count": quality["word_count"],
                "warnings": quality["warnings"],
                "text_length": len(text),
            }
            chunks.append(
                SourceChunk(
                    source_no=index,
                    source_type="page",
                    text=text,
                    description_of_image_video=clean_text("\n".join(image_descriptions)) or None,
                    metadata={
                        **extraction_unit_metadata(
                            source_type="page",
                            unit_number=index,
                            raw_text=clean_text("\n".join([native_text, ocr_text])),
                            cleaned_text=text,
                            extraction_method=extraction_method,
                            confidence_score=quality["confidence_score"],
                            warnings=warnings,
                        ),
                        "quality_score": quality["quality_score"],
                        "estimated_quality_score": quality["quality_score"],
                        "measured_accuracy": None,
                        "accuracy_source": "strict_report_builder",
                        "confidence_score": quality["confidence_score"],
                        "native_text_length": len(native_text),
                        "ocr_text_length": len(ocr_text),
                        "final_text_length": len(text),
                        "noise_ratio": quality["noise_ratio"],
                        "garbled_word_ratio": quality["garbled_word_ratio"],
                        "word_count": quality["word_count"],
                        "image_count": image_count,
                        "has_images": image_count > 0,
                        "native_text": native_text,
                        "ocr_text": ocr_text,
                        "final_text": text,
                        "page_level": page_level,
                        "warnings": warnings,
                    },
                )
            )
    finally:
        doc.close()
    return chunks
