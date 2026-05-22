from __future__ import annotations

import base64
import logging
import mimetypes
import os
from pathlib import Path
from typing import Any

import httpx

from ..config import (
    OPEN_WEIGHT_VLM_API_KEY,
    OPEN_WEIGHT_VLM_API_URL,
    OPEN_WEIGHT_VLM_CONFIDENCE,
    OPEN_WEIGHT_VLM_ENABLED,
    OPEN_WEIGHT_VLM_MODEL,
    OPEN_WEIGHT_VLM_TIMEOUT_SECONDS,
)


logger = logging.getLogger(__name__)

OCR_PROMPT = (
    "Extract all readable text from this image. Return only the text exactly as it appears, "
    "preserving line breaks where useful. If there is no readable text, return an empty response."
)


def _chat_completions_url(value: str) -> str:
    url = str(value or "").strip().rstrip("/")
    if not url:
        return ""
    if url.endswith("/v1"):
        return f"{url}/chat/completions"
    if url.endswith("/chat/completions"):
        return url
    return url


def _image_data_url(image_path: str) -> str:
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    content = path.read_bytes()
    encoded = base64.b64encode(content).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_message_content(payload: dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        return ""
    message = choices[0].get("message") or {}
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part.strip() for part in parts if part.strip()).strip()
    return str(content or "").strip()


def extract_text_open_weight_vlm(image_path: str) -> tuple[str, float]:
    try:
        url = _chat_completions_url(os.getenv("OPEN_WEIGHT_VLM_API_URL", OPEN_WEIGHT_VLM_API_URL))
        model = os.getenv("OPEN_WEIGHT_VLM_MODEL", OPEN_WEIGHT_VLM_MODEL).strip()
        if not url or not model:
            return "", 0.0

        api_key = os.getenv("OPEN_WEIGHT_VLM_API_KEY", OPEN_WEIGHT_VLM_API_KEY).strip()
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        body = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": _image_data_url(image_path)}},
                        {"type": "text", "text": OCR_PROMPT},
                    ],
                }
            ],
            "max_tokens": 2048,
            "temperature": 0,
        }
        with httpx.Client(timeout=OPEN_WEIGHT_VLM_TIMEOUT_SECONDS) as client:
            response = client.post(url, headers=headers, json=body)
            response.raise_for_status()
        text = _extract_message_content(response.json())
        if text.startswith("```"):
            text = text.strip("`").strip()
            if text.lower().startswith("text"):
                text = text[4:].strip()
        confidence = max(0.0, min(1.0, float(OPEN_WEIGHT_VLM_CONFIDENCE)))
        return text, confidence if text else 0.0
    except Exception as exc:
        logger.warning("Open-weight VLM OCR failed: %s", exc)
        return "", 0.0


def is_open_weight_vlm_available() -> bool:
    if not OPEN_WEIGHT_VLM_ENABLED:
        return False
    url = os.getenv("OPEN_WEIGHT_VLM_API_URL", OPEN_WEIGHT_VLM_API_URL)
    model = os.getenv("OPEN_WEIGHT_VLM_MODEL", OPEN_WEIGHT_VLM_MODEL)
    return bool(str(url or "").strip() and str(model or "").strip())
