from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .extractors.audio_processor import transcribe_audio_file
from .extractors.generic_processor import extract_document, extract_spreadsheet
from .extractors.image_processor import _ocr_pil_image, extract_image
from .extractors.pdf_processor import extract_pdf
from .extractors.ppt_processor import extract_ppt
from .extractors.video_processor import (
    extract_audio_from_video,
    extract_sampled_frame_text,
    extract_video_intelligence,
)
from .models import SourceChunk
from .utils import clean_text


TEXT_EXTENSIONS = {".txt", ".csv", ".html", ".htm", ".xml", ".svg", ".rtf"}
PDF_EXTENSIONS = {".pdf"}
WORD_EXTENSIONS = {".doc", ".docx", ".docm", ".dot", ".dotx", ".dotm", ".rtf"}
PPT_EXTENSIONS = {".ppt", ".pptx"}
EXCEL_EXTENSIONS = {".xls", ".xlsx", ".csv"}
IMAGE_EXTENSIONS = {
    ".avif",
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
VIDEO_EXTENSIONS = {".avi", ".mkv", ".mov", ".mp4"}
AUDIO_EXTENSIONS = {".flac", ".m4a", ".mp3", ".ogg", ".wav"}


MODULE_ALIASES = {
    "pymupdf": "pdf_text",
    "pdfplumber": "pdf_text_pdfplumber",
    "pdf_text": "pdf_text",
    "scanned_pdf": "scanned_pdf_ocr",
    "scanned_pdf_ocr": "scanned_pdf_ocr",
    "tesseract": "image_ocr",
    "easyocr": "image_ocr",
    "image_ocr": "image_ocr",
    "python-docx": "word_extraction",
    "python_docx": "word_extraction",
    "word": "word_extraction",
    "docx": "word_extraction",
    "python-pptx": "ppt_extraction",
    "python_pptx": "ppt_extraction",
    "ppt": "ppt_extraction",
    "pptx": "ppt_extraction",
    "openpyxl": "excel_extraction",
    "pandas": "excel_extraction",
    "excel": "excel_extraction",
    "libreoffice": "office_to_pdf_rendering",
    "libreoffice_headless": "office_to_pdf_rendering",
    "office_to_pdf": "office_to_pdf_rendering",
    "blip": "image_description",
    "vlm": "image_description",
    "image_description": "image_description",
    "whisper": "audio_transcript",
    "audio": "audio_transcript",
    "audio_transcript": "audio_transcript",
    "ffmpeg": "video_audio_transcript",
    "moviepy": "video_audio_transcript",
    "video_audio": "video_audio_transcript",
    "video_audio_transcript": "video_audio_transcript",
    "opencv": "video_frame_ocr",
    "opencv_ocr": "video_frame_ocr",
    "video_frame_ocr": "video_frame_ocr",
    "video": "video_frame_ocr",
}


MODULE_TOOL_LABELS = {
    "pdf_text": "PyMuPDF PDF text extraction",
    "pdf_text_pdfplumber": "pdfplumber PDF text extraction",
    "scanned_pdf_ocr": "PyMuPDF rendering + Tesseract/EasyOCR scanned PDF OCR",
    "word_extraction": "python-docx / OOXML Word extraction",
    "ppt_extraction": "python-pptx PPT extraction",
    "excel_extraction": "openpyxl/pandas Excel extraction",
    "office_to_pdf_rendering": "LibreOffice headless Office-to-PDF rendering + PyMuPDF text extraction",
    "image_ocr": "EasyOCR/Tesseract image OCR",
    "image_description": "BLIP/VLM image description",
    "audio_transcript": "Whisper audio transcript",
    "video_audio_transcript": "FFmpeg/MoviePy video audio extraction + Whisper transcript",
    "video_frame_ocr": "OpenCV frame sampling + OCR",
}


MODULE_METRIC_LABELS = {
    "pdf_text": "word_accuracy",
    "pdf_text_pdfplumber": "word_accuracy",
    "scanned_pdf_ocr": "word_accuracy",
    "word_extraction": "word_accuracy",
    "ppt_extraction": "word_accuracy",
    "excel_extraction": "word_accuracy",
    "office_to_pdf_rendering": "text_similarity",
    "image_ocr": "word_accuracy",
    "image_description": "manual_or_semantic",
    "audio_transcript": "WER accuracy (100 - WER)",
    "video_audio_transcript": "WER accuracy (100 - WER)",
    "video_frame_ocr": "word_accuracy",
}


def _normalize_tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", clean_text(value).lower())


def _edit_distance(left: list[str], right: list[str]) -> int:
    rows = len(left) + 1
    cols = len(right) + 1
    previous = list(range(cols))
    for row in range(1, rows):
        current = [row] + [0] * (cols - 1)
        for col in range(1, cols):
            substitution_cost = 0 if left[row - 1] == right[col - 1] else 1
            current[col] = min(
                previous[col] + 1,
                current[col - 1] + 1,
                previous[col - 1] + substitution_cost,
            )
        previous = current
    return previous[-1]


def text_similarity(expected: str, actual: str) -> float:
    expected_clean = clean_text(expected).lower()
    actual_clean = clean_text(actual).lower()
    if not expected_clean and not actual_clean:
        return 100.0
    return round(SequenceMatcher(None, expected_clean, actual_clean).ratio() * 100, 2)


def word_accuracy(expected: str, actual: str) -> float:
    expected_words = _normalize_tokens(expected)
    actual_words = _normalize_tokens(actual)
    if not expected_words and not actual_words:
        return 100.0
    if not expected_words:
        return 0.0
    distance = _edit_distance(expected_words, actual_words)
    return round(max(0.0, 1.0 - (distance / len(expected_words))) * 100, 2)


def word_error_rate(expected: str, actual: str) -> float:
    expected_words = _normalize_tokens(expected)
    actual_words = _normalize_tokens(actual)
    if not expected_words and not actual_words:
        return 0.0
    if not expected_words:
        return 100.0
    distance = _edit_distance(expected_words, actual_words)
    return round((distance / len(expected_words)) * 100, 2)


def token_f1(expected: str, actual: str) -> float:
    expected_words = _normalize_tokens(expected)
    actual_words = _normalize_tokens(actual)
    if not expected_words and not actual_words:
        return 100.0
    if not expected_words or not actual_words:
        return 0.0
    expected_counts: dict[str, int] = defaultdict(int)
    actual_counts: dict[str, int] = defaultdict(int)
    for token in expected_words:
        expected_counts[token] += 1
    for token in actual_words:
        actual_counts[token] += 1
    overlap = sum(min(expected_counts[token], actual_counts[token]) for token in expected_counts)
    precision = overlap / max(len(actual_words), 1)
    recall = overlap / max(len(expected_words), 1)
    if precision + recall == 0:
        return 0.0
    return round((2 * precision * recall / (precision + recall)) * 100, 2)


def _chunks_text(chunks: list[SourceChunk]) -> str:
    return clean_text("\n".join(chunk.text for chunk in chunks if chunk.text))


def _chunks_description(chunks: list[SourceChunk]) -> str:
    values = []
    for chunk in chunks:
        if chunk.description_of_image_video:
            values.append(chunk.description_of_image_video)
        frame_descriptions = chunk.metadata.get("frame_descriptions") if chunk.metadata else None
        if frame_descriptions:
            values.append(str(frame_descriptions))
    return clean_text("\n".join(values))


def _read_text_file(path: Path) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return clean_text(path.read_text(encoding=encoding))
        except UnicodeDecodeError:
            continue
    return clean_text(path.read_text(encoding="latin-1", errors="ignore"))


def _extract_pdf_with_pdfplumber(path: Path) -> str:
    try:
        import pdfplumber
    except Exception as exc:
        raise RuntimeError("pdfplumber is required for this evaluation module. Install with: pip install pdfplumber") from exc

    parts = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            parts.append(page.extract_text() or "")
    return clean_text("\n".join(parts))


def _render_pdf_pages_for_ocr(path: Path) -> str:
    try:
        import fitz
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("PyMuPDF and Pillow are required for scanned PDF OCR evaluation.") from exc

    texts = []
    doc = fitz.open(str(path))
    try:
        for page in doc:
            pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
            ocr_text, _engine = _ocr_pil_image(image)
            texts.append(ocr_text)
    finally:
        doc.close()
    return clean_text("\n".join(texts))


def _render_office_to_pdf_text(path: Path) -> str:
    executable = shutil.which("soffice") or shutil.which("libreoffice")
    if not executable:
        raise RuntimeError("LibreOffice headless was not found on PATH.")

    with tempfile.TemporaryDirectory() as folder:
        output_dir = Path(folder)
        command = [
            executable,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(output_dir),
            str(path),
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            raise RuntimeError(f"LibreOffice conversion failed: {result.stderr[-600:]}")
        pdf_path = output_dir / f"{path.stem}.pdf"
        if not pdf_path.exists():
            pdfs = list(output_dir.glob("*.pdf"))
            if not pdfs:
                raise RuntimeError("LibreOffice conversion did not produce a PDF.")
            pdf_path = pdfs[0]
        return _chunks_text(extract_pdf(pdf_path))


def _extract_video_audio_transcript(path: Path) -> str:
    with tempfile.TemporaryDirectory() as folder:
        audio_path = Path(folder) / f"{path.stem}_audio.wav"
        extracted_audio = extract_audio_from_video(path, audio_path)
        return transcribe_audio_file(extracted_audio)


def _extract_output(case: dict[str, Any], module: str) -> str:
    if case.get("actual_text") is not None:
        return clean_text(str(case["actual_text"]))
    if case.get("actual_output") is not None:
        return clean_text(str(case["actual_output"]))
    if case.get("actual_description") is not None:
        return clean_text(str(case["actual_description"]))

    path_value = case.get("file_path") or case.get("source_path") or case.get("path")
    if not path_value:
        raise ValueError("Each case needs file_path/source_path/path unless actual_text or actual_output is provided.")
    path = Path(path_value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file was not found: {path}")

    if module == "pdf_text":
        return _chunks_text(extract_pdf(path))
    if module == "pdf_text_pdfplumber":
        return _extract_pdf_with_pdfplumber(path)
    if module == "scanned_pdf_ocr":
        return _render_pdf_pages_for_ocr(path)
    if module == "word_extraction":
        return _chunks_text(extract_document(path))
    if module == "ppt_extraction":
        return _chunks_text(extract_ppt(path))
    if module == "excel_extraction":
        return _chunks_text(extract_spreadsheet(path))
    if module == "office_to_pdf_rendering":
        return _render_office_to_pdf_text(path)
    if module == "image_ocr":
        return _chunks_text(extract_image(path))
    if module == "image_description":
        return _chunks_description(extract_image(path))
    if module == "audio_transcript":
        return transcribe_audio_file(path)
    if module == "video_audio_transcript":
        return _extract_video_audio_transcript(path)
    if module == "video_frame_ocr":
        frame_result = extract_sampled_frame_text(path)
        frame_text, frame_descriptions = frame_result[0], frame_result[1]
        return clean_text("\n".join([frame_text, frame_descriptions]))
    if module == "video_full":
        chunks = extract_video_intelligence(path)
        return clean_text("\n".join([_chunks_text(chunks), _chunks_description(chunks)]))
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return _read_text_file(path)
    raise ValueError(f"No extractor is configured for module '{module}'.")


def detect_module(case: dict[str, Any]) -> str:
    explicit = clean_text(str(case.get("module") or case.get("tool") or case.get("extractor") or "")).lower()
    normalized = explicit.replace(" ", "_").replace("-", "_")
    if normalized in MODULE_ALIASES:
        return MODULE_ALIASES[normalized]

    path_value = case.get("file_path") or case.get("source_path") or case.get("path") or ""
    ext = Path(str(path_value)).suffix.lower()
    task = clean_text(str(case.get("task") or case.get("metric") or "")).lower()
    if "description" in task:
        return "image_description" if ext in IMAGE_EXTENSIONS else "video_frame_ocr"
    if "ocr" in task and ext in PDF_EXTENSIONS:
        return "scanned_pdf_ocr"
    if "ocr" in task and ext in IMAGE_EXTENSIONS:
        return "image_ocr"
    if "frame" in task and ext in VIDEO_EXTENSIONS:
        return "video_frame_ocr"
    if "transcript" in task and ext in VIDEO_EXTENSIONS:
        return "video_audio_transcript"
    if ext in PDF_EXTENSIONS:
        return "pdf_text"
    if ext in WORD_EXTENSIONS:
        return "word_extraction"
    if ext in PPT_EXTENSIONS:
        return "ppt_extraction"
    if ext in EXCEL_EXTENSIONS:
        return "excel_extraction"
    if ext in IMAGE_EXTENSIONS:
        return "image_ocr"
    if ext in AUDIO_EXTENSIONS:
        return "audio_transcript"
    if ext in VIDEO_EXTENSIONS:
        return "video_frame_ocr"
    return "text"


def detect_expected(case: dict[str, Any], module: str) -> str:
    keys = [
        "expected_text",
        "ground_truth",
        "expected_output",
        "expected_transcript",
        "expected_description",
    ]
    for key in keys:
        if case.get(key) is not None:
            return clean_text(str(case[key]))
    expected_path = case.get("ground_truth_path") or case.get("expected_path")
    if expected_path:
        path = Path(str(expected_path))
        if not path.is_absolute():
            path = Path.cwd() / path
        return _read_text_file(path)
    raise ValueError(f"Case for module '{module}' needs expected_text/ground_truth or ground_truth_path.")


def detect_metric(case: dict[str, Any], module: str) -> str:
    explicit = clean_text(str(case.get("metric") or "")).lower().replace(" ", "_")
    if explicit:
        return explicit
    if module in {"audio_transcript", "video_audio_transcript"}:
        return "wer_accuracy"
    if module in {"image_description"}:
        return "manual_or_semantic"
    if module in {"video_frame_ocr"}:
        return "word_accuracy"
    if "ocr" in module or module in {"word_extraction", "ppt_extraction", "excel_extraction", "pdf_text", "pdf_text_pdfplumber"}:
        return "word_accuracy"
    return "text_similarity"


def _score_case(expected: str, actual: str, metric: str, case: dict[str, Any]) -> dict[str, Any]:
    if metric in {"wer", "wer_accuracy", "transcript_accuracy"}:
        wer = word_error_rate(expected, actual)
        return {
            "metric": "WER",
            "wer": wer,
            "accuracy": round(max(0.0, 100.0 - wer), 2),
            "accuracy_formula": "100 - WER",
        }
    if metric in {"word_accuracy", "ocr_accuracy"}:
        return {"metric": "word_accuracy", "accuracy": word_accuracy(expected, actual)}
    if metric in {"text_similarity", "similarity"}:
        return {"metric": "text_similarity", "accuracy": text_similarity(expected, actual)}
    if metric in {"token_f1", "semantic", "manual_or_semantic"}:
        manual_score = case.get("manual_score")
        if manual_score is not None:
            return {"metric": "manual_score", "accuracy": round(float(manual_score), 2)}
        return {
            "metric": "semantic_token_f1_proxy",
            "accuracy": token_f1(expected, actual),
            "note": "Use manual_score in ground truth JSON for human semantic scoring of descriptions.",
        }
    raise ValueError(f"Unsupported metric '{metric}'.")


def evaluate_case(case: dict[str, Any]) -> dict[str, Any]:
    module = detect_module(case)
    metric = detect_metric(case, module)
    expected = detect_expected(case, module)
    actual = _extract_output(case, module)
    score = _score_case(expected, actual, metric, case)
    return {
        "case_id": case.get("id") or case.get("case_id") or Path(str(case.get("file_path", "case"))).stem,
        "module": module,
        "tool": MODULE_TOOL_LABELS.get(module, module),
        "metric_detected": score["metric"],
        "accuracy": score["accuracy"],
        "details": {key: value for key, value in score.items() if key not in {"metric", "accuracy"}},
        "expected_chars": len(expected),
        "actual_chars": len(actual),
        "expected_preview": expected[:500],
        "actual_preview": actual[:500],
    }


def evaluate_ground_truth(data: dict[str, Any]) -> dict[str, Any]:
    cases = data.get("cases", data if isinstance(data, list) else [])
    if not isinstance(cases, list):
        raise ValueError("Ground truth JSON must be a list or an object with a 'cases' list.")

    results_by_module: dict[str, list[dict[str, Any]]] = defaultdict(list)
    failures: list[dict[str, Any]] = []
    for case in cases:
        try:
            result = evaluate_case(case)
            results_by_module[result["module"]].append(result)
        except Exception as exc:
            module = detect_module(case) if isinstance(case, dict) else "unknown"
            failures.append(
                {
                    "case_id": case.get("id") or case.get("case_id") if isinstance(case, dict) else "unknown",
                    "module": module,
                    "error": str(exc),
                }
            )

    summary = {}
    for module, results in sorted(results_by_module.items()):
        accuracies = [float(result["accuracy"]) for result in results]
        summary[module] = {
            "tool": MODULE_TOOL_LABELS.get(module, module),
            "case_count": len(results),
            "average_accuracy": round(sum(accuracies) / max(len(accuracies), 1), 2),
            "minimum_accuracy": round(min(accuracies), 2) if accuracies else 0.0,
            "maximum_accuracy": round(max(accuracies), 2) if accuracies else 0.0,
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool_catalog": {
            module: {
                "tool": tool,
                "default_metric": MODULE_METRIC_LABELS.get(module, "text_similarity"),
            }
            for module, tool in sorted(MODULE_TOOL_LABELS.items())
        },
        "summary": summary,
        "results_by_module": dict(sorted(results_by_module.items())),
        "failures": failures,
    }


def write_evaluation_report(ground_truth_path: Path, output_path: Path) -> dict[str, Any]:
    data = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    report = evaluate_ground_truth(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate extraction accuracy from manually verified ground truth JSON.")
    parser.add_argument("ground_truth_json", type=Path, help="Path to manually verified ground truth JSON.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports") / "extraction_accuracy_report.json",
        help="Where to write the module-wise JSON report.",
    )
    args = parser.parse_args()
    report = write_evaluation_report(args.ground_truth_json, args.output)
    print(json.dumps({"output": str(args.output), "summary": report["summary"], "failures": report["failures"]}, indent=2))


if __name__ == "__main__":
    main()
