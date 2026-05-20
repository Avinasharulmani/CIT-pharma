import os
import re
import uuid
from pathlib import Path
from typing import Iterable, List, Optional

from .config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES, SUPPORTED_FILE_TYPES, UPLOAD_DIR


def ensure_directories() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_file_type_config(filename: str, content_type: Optional[str] = None) -> dict:
    ext = Path(filename).suffix.lower()
    config = SUPPORTED_FILE_TYPES.get(ext)
    if not config:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {', '.join(sorted(ALLOWED_EXTENSIONS))}")
    return {
        "extension": ext,
        "mime_type": content_type or "",
        **config,
    }


def detect_file_type(filename: str, content_type: Optional[str] = None) -> str:
    return str(get_file_type_config(filename, content_type).get("file_type") or "")


def safe_filename(filename: str) -> str:
    name = Path(filename).name
    name = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    while re.match(r"^[0-9a-f]{32}_", name, flags=re.IGNORECASE):
        name = name.split("_", 1)[1]
    if len(name) > 120:
        stem = Path(name).stem[:90]
        suffix = Path(name).suffix[:20]
        name = f"{stem}{suffix}"
    return f"{uuid.uuid4().hex}_{name}"


async def save_upload_file(upload_file, folder: Optional[Path] = None) -> Path:
    ensure_directories()
    folder = folder or UPLOAD_DIR
    folder.mkdir(parents=True, exist_ok=True)
    get_file_type_config(upload_file.filename or "uploaded_file", getattr(upload_file, "content_type", None))
    destination = folder / safe_filename(upload_file.filename or "uploaded_file")
    total_size = 0
    with destination.open("wb") as output_file:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            total_size += len(chunk)
            if total_size > MAX_UPLOAD_SIZE_BYTES:
                output_file.close()
                destination.unlink(missing_ok=True)
                raise ValueError(f"File is too large. Maximum upload size is {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB.")
            output_file.write(chunk)
    return destination


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_repetition(text: str) -> str:
    """Remove repeated neighboring sentences/lines without changing meaning."""
    if not text:
        return ""
    seen = set()
    output = []
    for part in re.split(r"(?<=[.!?])\s+|\n+", text):
        normalized = re.sub(r"\W+", " ", part.lower()).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            output.append(part.strip())
    return "\n".join(output)


def split_into_chunks(text: str, max_words: int = 160, overlap: int = 30) -> List[str]:
    words = clean_text(text).split()
    if not words:
        return []
    chunks = []
    step = max(max_words - overlap, 1)
    for start in range(0, len(words), step):
        chunk_words = words[start : start + max_words]
        if chunk_words:
            chunks.append(" ".join(chunk_words))
    return chunks


def compact_join(values: Iterable[str], limit: Optional[int] = 6000) -> str:
    text = "\n".join([v for v in values if v])
    if limit is None:
        return text
    return text[:limit]


def extract_possible_brand_name(text: str) -> str:
    """Simple brand/medicine-name signal used before embedding fallback."""
    for line in clean_text(text).splitlines():
        candidate = line.strip()
        if 3 <= len(candidate) <= 60 and re.match(r"^[A-Z][A-Z0-9\s\-/()]+$", candidate):
            candidate = re.sub(r"\(.*?\)", "", candidate).strip()
            return candidate.lower()
    return ""
