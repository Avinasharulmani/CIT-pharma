from __future__ import annotations

import base64
import logging
import os
from statistics import mean

from google.cloud import vision

from ..config import GOOGLE_VISION_API_KEY


logger = logging.getLogger(__name__)


def get_vision_client():
    return vision.ImageAnnotatorClient(client_options={"api_key": GOOGLE_VISION_API_KEY})


def extract_text_google_vision(image_path: str) -> tuple[str, float]:
    try:
        with open(image_path, "rb") as image_file:
            content = image_file.read()
        _encoded = base64.b64encode(content).decode("ascii")
        image = vision.Image(content=content)
        response = get_vision_client().document_text_detection(image=image)
        if response.error and response.error.message:
            raise RuntimeError(response.error.message)
        annotation = response.full_text_annotation
        extracted_text = str(annotation.text or "").strip() if annotation else ""
        page_confidences = [float(page.confidence) for page in (annotation.pages or []) if page.confidence is not None] if annotation else []
        confidence = mean(page_confidences) if page_confidences else 0.0
        return extracted_text, max(0.0, min(1.0, float(confidence)))
    except Exception as exc:
        logger.warning("Google Vision OCR failed: %s", exc)
        return "", 0.0


def is_vision_available() -> bool:
    value = GOOGLE_VISION_API_KEY or os.getenv("GOOGLE_VISION_API_KEY", "")
    return bool(str(value or "").strip())
