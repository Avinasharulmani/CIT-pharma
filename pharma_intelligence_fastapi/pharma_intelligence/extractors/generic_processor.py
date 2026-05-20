from __future__ import annotations

import csv
import html
import io
import re
import zipfile
from email import policy
from email.parser import BytesParser
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, List
from xml.etree import ElementTree

from ..config import SUPPORTED_FILE_TYPES
from ..models import SourceChunk
from ..utils import clean_text


TEXT_EXTENSIONS = {
    ".bash",
    ".cov",
    ".csh",
    ".ext",
    ".htm",
    ".html",
    ".inp",
    ".jsl",
    ".lst",
    ".odc",
    ".op",
    ".param",
    ".r",
    ".sas",
    ".sbml",
    ".scm",
    ".sh",
    ".ssc",
    ".svg",
    ".txt",
    ".xml",
}

BINARY_RASTER_EXTENSIONS = {
    ".3fr",
    ".arw",
    ".avif",
    ".bmp",
    ".cr2",
    ".crw",
    ".dcr",
    ".dng",
    ".gif",
    ".heic",
    ".heif",
    ".jpg",
    ".jpeg",
    ".kdc",
    ".nef",
    ".nrw",
    ".orf",
    ".pef",
    ".png",
    ".raf",
    ".raw",
    ".rw2",
    ".sr2",
    ".srf",
    ".tif",
    ".tiff",
    ".webp",
}


class _HTMLTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        value = clean_text(html.unescape(data))
        if value:
            self.parts.append(value)


def _read_bytes(path: Path, max_bytes: int | None = 4 * 1024 * 1024) -> bytes:
    with path.open("rb") as file:
        return file.read(max_bytes) if max_bytes is not None else file.read()


def _decode_bytes(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="ignore")


def _printable_strings(data: bytes, min_length: int = 4) -> str:
    text = _decode_bytes(data)
    fragments = re.findall(rf"[A-Za-z0-9][A-Za-z0-9\s.,;:!?%()/#&+\-_'\"=]{{{min_length - 1},}}", text)
    seen: set[str] = set()
    values: list[str] = []
    for fragment in fragments:
        value = clean_text(fragment)
        if re.match(r"^%{0,2}(?:PS-Adobe|Creator:|BoundingBox:|Pages:|EndComments|Page:|ImageData:|BeginData|EndData)\b", value, flags=re.IGNORECASE):
            continue
        normalized = value.lower()
        compact = re.sub(r"\s+", "", normalized)
        hex_chars = sum(1 for char in compact if char in "0123456789abcdef")
        letters = re.findall(r"[A-Za-z]", value)
        if len(compact) > 24 and hex_chars / max(1, len(compact)) > 0.85:
            continue
        if len(value) > 10 and not letters:
            continue
        if any(token in normalized for token in ("readhexstring", "colorimage", "currentfile", "imagemask")):
            continue
        if normalized in {"gsave", "grestore", "colorimage", "endstream", "endobj", "trailer", "startxref", "bind"}:
            continue
        if len(value) < min_length or normalized in seen:
            continue
        seen.add(normalized)
        values.append(value)
        if len(values) >= 250:
            break
    return clean_text("\n".join(values))


def _postscript_text(data: bytes) -> str:
    return clean_text("\n".join(text for _, _, text in _postscript_text_items(data)))


def _postscript_text_items(data: bytes) -> list[tuple[float, float, str]]:
    raw = _decode_bytes(data)
    pattern = re.compile(r"(?m)([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+moveto\s+\(((?:\\.|[^)])*)\)\s*show")
    items: list[tuple[float, float, str]] = []
    for match in pattern.finditer(raw):
        value = match.group(3)
        value = value.replace(r"\(", "(").replace(r"\)", ")").replace(r"\\", "\\")
        value = re.sub(r"\\[0-7]{1,3}", " ", value)
        value = clean_text(value)
        if value:
            items.append((float(match.group(1)), float(match.group(2)), value))
    return items


def _postscript_section_heading(line: str, next_line: str = "") -> bool:
    if not line or re.match(r"^[-*•]", line) or re.match(r"^PAGE\s+\d+\s+-", line, flags=re.IGNORECASE):
        return False
    if re.search(r"[.!?]$", line):
        return False
    words = line.split()
    return len(words) <= 6 and len(line) <= 80 and bool(re.match(r"^[-*•]", next_line or ""))


def _merge_wrapped_postscript_lines(lines: list[str]) -> list[str]:
    merged: list[str] = []
    for index, line in enumerate(lines):
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        if _postscript_section_heading(line, next_line):
            merged.append(line)
            continue
        if re.match(r"^[-*•]", line):
            merged.append(line)
            continue
        if merged and re.match(r"^[-*•]", merged[-1]) and not _postscript_section_heading(line, next_line):
            merged[-1] = clean_text(f"{merged[-1]} {line}")
            continue
        merged.append(line)
    return merged


def _postscript_chunks(path: Path, data: bytes) -> list[SourceChunk]:
    items = _postscript_text_items(data)
    pages: list[dict[str, object]] = []
    current: dict[str, object] | None = None
    intro: list[str] = []
    page_pattern = re.compile(r"^PAGE\s+(\d+)\s*-\s*(.+)$", flags=re.IGNORECASE)
    for x, y, value in items:
        page_match = page_pattern.match(value)
        if page_match:
            current = {
                "number": int(page_match.group(1)),
                "title": clean_text(page_match.group(2)),
                "lines": [value],
                "positions": [{"x": x, "y": y, "text": value}],
            }
            pages.append(current)
            continue
        if current is None:
            intro.append(value)
            continue
        current["lines"].append(value)  # type: ignore[index]
        current["positions"].append({"x": x, "y": y, "text": value})  # type: ignore[index]

    if pages:
        chunks: list[SourceChunk] = []
        for page in pages:
            page_lines = _merge_wrapped_postscript_lines(list(page["lines"]))  # type: ignore[arg-type]
            page_text = clean_text("\n".join(page_lines))
            chunks.append(
                _chunk(
                    page_text,
                    "page",
                    int(page["number"]),
                    extension=path.suffix.lower(),
                    title=page.get("title"),
                    positions=page.get("positions"),
                    intro=clean_text("\n".join(intro)) or None,
                )
            )
        return chunks

    values: list[str] = []
    for _, _, value in items:
        values.append(value)
    text = clean_text("\n".join(values))
    return [_chunk(text, "design file", 1, extension=path.suffix.lower())] if text else []


def _chunk(text: str, source_type: str, source_no: int | None = 1, **metadata) -> SourceChunk:
    return SourceChunk(
        source_no=source_no,
        source_type=source_type,
        text=clean_text(text),
        metadata={key: value for key, value in metadata.items() if value is not None},
    )


def _strip_shell_heredoc_wrappers(text: str) -> str:
    lines = clean_text(text).splitlines()
    if not lines:
        return ""
    marker = ""
    cleaned: list[str] = []
    heredoc_pattern = re.compile(r"^\s*(?:cat|tee)\b.*<<[-]?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*$")
    for line in lines:
        match = heredoc_pattern.match(line)
        if match:
            marker = match.group(1)
            continue
        if marker and line.strip() == marker:
            marker = ""
            continue
        if not marker and re.match(r"^[A-Z][A-Z0-9_]{8,}$", line.strip()) and "_" in line:
            continue
        cleaned.append(line)
    return clean_text("\n".join(cleaned))


def _normalize_page_marker_text(text: str) -> str:
    text = _strip_shell_heredoc_wrappers(text)
    text = re.sub(r"\s+(PAGE\s+\d+\s+-)", r"\n\1", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+((?:Page\s*(?:Number|No\.?|#)?\s*:?\s*)\d+\b)", r"\n\1", text, flags=re.IGNORECASE)
    return clean_text(text)


def _generic_page_chunks(path: Path, text: str, default_source_type: str, warnings: list[str] | None = None) -> list[SourceChunk]:
    text = _normalize_page_marker_text(text)
    page_pattern = re.compile(
        r"^(?:PAGE\s+(\d+)\s*-\s*(.+)|Page\s*(?:Number|No\.?|#)?\s*:?\s*(\d+)\b.*)$",
        flags=re.IGNORECASE,
    )
    pages: list[tuple[int, str, list[str]]] = []
    current_number: int | None = None
    current_title = ""
    current_lines: list[str] = []
    intro: list[str] = []

    def flush() -> None:
        nonlocal current_number, current_title, current_lines
        if current_number is not None:
            pages.append((current_number, current_title, list(current_lines)))
        current_number = None
        current_title = ""
        current_lines = []

    for line in text.splitlines():
        value = clean_text(line)
        if not value:
            continue
        match = page_pattern.match(value)
        if match:
            flush()
            current_number = int(match.group(1) or match.group(3))
            current_title = clean_text(match.group(2) or "")
            current_lines = [value]
            continue
        if current_number is None:
            intro.append(value)
        else:
            current_lines.append(value)
    flush()

    if not pages:
        return [_chunk(text, default_source_type, 1, extension=path.suffix.lower(), warnings=warnings or [])]

    chunks: list[SourceChunk] = []
    for number, title, lines in pages:
        page_text = clean_text("\n".join(lines))
        chunks.append(
            _chunk(
                page_text,
                "page",
                number,
                extension=path.suffix.lower(),
                title=title,
                intro=clean_text("\n".join(intro)) or None,
                warnings=warnings or [],
            )
        )
    return chunks


def _html_to_text(value: str) -> str:
    parser = _HTMLTextParser()
    parser.feed(value)
    text = "\n".join(parser.parts)
    return clean_text(text or re.sub(r"<[^>]+>", " ", value))


def _xml_to_text(value: str) -> str:
    try:
        root = ElementTree.fromstring(value)
        return clean_text("\n".join(item.strip() for item in root.itertext() if item and item.strip()))
    except Exception:
        return clean_text(re.sub(r"<[^>]+>", " ", value))


def _docx_xml_to_text(value: str) -> str:
    try:
        root = ElementTree.fromstring(value)
    except Exception:
        return _xml_to_text(value)

    paragraphs: list[str] = []
    for paragraph in root.iter("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p"):
        parts: list[str] = []
        for node in paragraph.iter():
            if node.tag == "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t" and node.text:
                parts.append(node.text)
            elif node.tag == "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}tab":
                parts.append(" ")
        text = clean_text("".join(parts))
        if text:
            paragraphs.append(text)
    return clean_text("\n".join(paragraphs)) if paragraphs else _xml_to_text(value)


def _rtf_to_text(value: str) -> str:
    value = re.sub(r"{\\\*?\\[^{}]+}|[{}]", " ", value)
    value = re.sub(r"\\'[0-9a-fA-F]{2}", " ", value)
    value = re.sub(r"\\[a-zA-Z]+\d* ?", " ", value)
    return clean_text(value)


def _docx_text_from_zip(path: Path) -> str:
    parts: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for member in ("word/document.xml", "word/footnotes.xml", "word/endnotes.xml", "word/comments.xml"):
            if member not in archive.namelist():
                continue
            parts.append(_docx_xml_to_text(_decode_bytes(archive.read(member))))
    return clean_text("\n".join(parts))


def _xlsx_chunks(path: Path) -> list[SourceChunk]:
    try:
        import pandas as pd
    except Exception as exc:
        return [_chunk("", "workbook", warnings=[f"pandas is required for spreadsheet extraction: {exc}"])]

    chunks: list[SourceChunk] = []
    try:
        sheets = pd.read_excel(path, sheet_name=None, dtype=str, nrows=200)
    except Exception as exc:
        return [_chunk(_printable_strings(_read_bytes(path)), "workbook", warnings=[f"Spreadsheet extraction warning: {exc}"])]
    for index, (sheet_name, frame) in enumerate(sheets.items(), start=1):
        frame = frame.fillna("")
        text = frame.to_csv(index=False)
        chunks.append(_chunk(text, "sheet", index, sheet_name=str(sheet_name), rows=int(len(frame))))
    return chunks or [_chunk("", "sheet")]


def _csv_text(path: Path) -> str:
    raw = _decode_bytes(_read_bytes(path))
    sample = raw[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample)
    except Exception:
        dialect = csv.excel
    rows: list[str] = []
    reader = csv.reader(io.StringIO(raw), dialect)
    for index, row in enumerate(reader):
        rows.append(" | ".join(clean_text(cell) for cell in row))
        if index >= 250:
            break
    return clean_text("\n".join(rows))


def extract_document(path: Path) -> List[SourceChunk]:
    ext = path.suffix.lower()
    warnings: list[str] = []
    text = ""
    try:
        if ext in {".docx", ".docm", ".dotx", ".dotm"}:
            text = _docx_text_from_zip(path)
        elif ext == ".rtf":
            text = _rtf_to_text(_decode_bytes(_read_bytes(path)))
        else:
            text = _printable_strings(_read_bytes(path))
    except Exception as exc:
        warnings.append(f"Document extraction warning: {exc}")
    return _generic_page_chunks(path, text, "document", warnings=warnings)


def extract_spreadsheet(path: Path) -> List[SourceChunk]:
    ext = path.suffix.lower()
    if ext == ".xlsx":
        return _xlsx_chunks(path)
    if ext == ".csv":
        return [_chunk(_csv_text(path), "table", extension=ext)]
    if ext == ".xls":
        try:
            import pandas as pd
            frame = pd.read_excel(path, dtype=str, nrows=200).fillna("")
            return [_chunk(frame.to_csv(index=False), "sheet", 1, extension=ext, rows=int(len(frame)))]
        except Exception as exc:
            return [_chunk(_printable_strings(_read_bytes(path)), "sheet", 1, extension=ext, warnings=[f"XLS extraction warning: {exc}"])]
    return [_chunk(_decode_bytes(_read_bytes(path)), "data", 1, extension=ext)]


def extract_markup_text(path: Path) -> List[SourceChunk]:
    ext = path.suffix.lower()
    raw = _decode_bytes(_read_bytes(path, max_bytes=None))
    if ext in {".html", ".htm", ".svg"}:
        text = _html_to_text(raw)
    elif ext in {".xml", ".sbml"}:
        text = _xml_to_text(raw)
    else:
        text = raw
    return _generic_page_chunks(path, text, "text")


def extract_email(path: Path) -> List[SourceChunk]:
    ext = path.suffix.lower()
    warnings: list[str] = []
    if ext == ".eml":
        try:
            message = BytesParser(policy=policy.default).parsebytes(_read_bytes(path, max_bytes=12 * 1024 * 1024))
            body_parts: list[str] = []
            if message.is_multipart():
                for part in message.walk():
                    content_type = part.get_content_type()
                    if content_type not in {"text/plain", "text/html"}:
                        continue
                    payload = part.get_content()
                    body_parts.append(_html_to_text(payload) if content_type == "text/html" else str(payload))
            else:
                payload = message.get_content()
                body_parts.append(_html_to_text(payload) if message.get_content_type() == "text/html" else str(payload))
            headers = [
                f"Subject: {message.get('subject', '')}",
                f"From: {message.get('from', '')}",
                f"To: {message.get('to', '')}",
                f"Date: {message.get('date', '')}",
            ]
            text = clean_text("\n".join(headers + body_parts))
            return [_chunk(text, "email", 1, extension=ext, subject=str(message.get("subject", "")), sender=str(message.get("from", "")), recipient=str(message.get("to", "")))]
        except Exception as exc:
            warnings.append(f"EML extraction warning: {exc}")
    return _generic_page_chunks(path, _printable_strings(_read_bytes(path)), "email", warnings=warnings)


def _safe_zip_members(archive: zipfile.ZipFile) -> Iterable[zipfile.ZipInfo]:
    for info in archive.infolist():
        name = info.filename.replace("\\", "/")
        if info.is_dir() or name.startswith("/") or ".." in name.split("/"):
            continue
        yield info


def extract_zip_package(path: Path) -> List[SourceChunk]:
    chunks: list[SourceChunk] = []
    listing: list[str] = []
    warnings: list[str] = []
    try:
        with zipfile.ZipFile(path) as archive:
            members = list(_safe_zip_members(archive))
            for info in members[:200]:
                listing.append(f"{info.filename} ({info.file_size} bytes)")
            chunks.append(_chunk("\n".join(listing), "package", 1, file_count=len(members)))
            source_no = 2
            for info in members[:30]:
                member_ext = Path(info.filename).suffix.lower()
                if info.file_size > 1_000_000 or member_ext not in TEXT_EXTENSIONS | {".csv"}:
                    continue
                text = _decode_bytes(archive.read(info))
                if member_ext in {".html", ".htm", ".svg"}:
                    text = _html_to_text(text)
                elif member_ext in {".xml", ".sbml"}:
                    text = _xml_to_text(text)
                chunks.append(_chunk(text, "package_file", source_no, member_name=info.filename, extension=member_ext))
                source_no += 1
    except Exception as exc:
        warnings.append(f"ZIP extraction warning: {exc}")
    if warnings:
        chunks.append(_chunk("", "package", len(chunks) + 1, warnings=warnings))
    return chunks or [_chunk("", "package")]


def extract_fallback(path: Path) -> List[SourceChunk]:
    ext = path.suffix.lower()
    data = _read_bytes(path)
    if ext in {".ai", ".eps"}:
        chunks = _postscript_chunks(path, data)
        if chunks:
            return chunks
    if ext in BINARY_RASTER_EXTENSIONS:
        return [
            _chunk(
                "",
                "image",
                1,
                extension=ext,
                warnings=["Image preview is available, but server-side text extraction requires image decoder/OCR support for this format."],
            )
        ]
    text = ""
    if not text:
        text = _printable_strings(data)
    registry = SUPPORTED_FILE_TYPES.get(ext, {})
    return [
        _chunk(
            text,
            "design file" if registry.get("category") == "Images/Design" else (registry.get("file_type") or "file"),
            1,
            extension=ext,
            warnings=[] if text else ["No readable text could be extracted from this file."],
        )
    ]
