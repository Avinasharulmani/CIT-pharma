import base64
import json
import urllib.error
import urllib.request
from io import BytesIO
from typing import List, Optional

from .config import OPENAI_API_KEY, VISION_ACTIVITY_FRAME_LIMIT, VISION_ACTIVITY_MODEL
from .utils import clean_text


def pil_image_to_data_url(image) -> Optional[str]:
    try:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=82)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except Exception:
        return None


def describe_visible_activity(
    image_data_urls: List[str],
    warnings: Optional[List[str]] = None,
    media_label: str = "image",
) -> str:
    if not OPENAI_API_KEY or not image_data_urls:
        if warnings is not None and not OPENAI_API_KEY:
            warnings.append("Exact visual activity description is disabled because OPENAI_API_KEY is not configured.")
        return ""

    if media_label == "image":
        prompt = (
            "Describe the visible content of this pharma commercial image with concrete visual detail. "
            "Mention the main subjects, setting, visible objects or product packaging, composition, and any clearly readable "
            "on-screen text that helps identify the material. Do not invent claims, diagnoses, brand names, or actions that "
            "are not visible. Avoid generic phrases such as 'an image' or 'a common scene'. "
            "Return 2 concise sentences under 55 words. If the image is too unclear to inspect, return: Visual content unclear."
        )
        detail = "high"
        max_output_tokens = 140
    else:
        prompt = (
            f"Look at this {media_label} and describe only the visible human activities. "
            "Focus on what people are doing, for example presenting, talking, consulting, walking, "
            "showing a product, receiving care, or interacting. "
            "Do not describe or quote on-screen text, brand names, logos, captions, or claims. "
            "Do not guess locations, objects, products, or actions that are not clearly visible. "
            "Return one concise sentence under 24 words. If no clear human activity is visible, "
            "return: Visual activity unclear."
        )
        detail = "low"
        max_output_tokens = 80

    content = [
        {
            "type": "input_text",
            "text": prompt,
        }
    ]
    content.extend(
        {"type": "input_image", "image_url": image_url, "detail": detail}
        for image_url in image_data_urls[:VISION_ACTIVITY_FRAME_LIMIT]
    )

    payload = {
        "model": VISION_ACTIVITY_MODEL,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": max_output_tokens,
    }
    request = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        try:
            exc.read()
        except Exception:
            pass
        if warnings is not None:
            warnings.append(
                "Exact visual activity description is unavailable from the vision model; using local visual fallback when possible."
            )
        return ""
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        if warnings is not None:
            warnings.append(
                "Exact visual activity description is unavailable from the vision model; using local visual fallback when possible."
            )
        return ""

    text = clean_text(str(data.get("output_text", "")))
    if not text:
        parts = []
        for item in data.get("output", []):
            for content_item in item.get("content", []):
                if content_item.get("type") in {"output_text", "text"}:
                    parts.append(str(content_item.get("text", "")))
        text = clean_text(" ".join(parts))

    lowered = text.lower()
    if not text or "visual activity unclear" in lowered or "visual content unclear" in lowered:
        return ""
    if len(text) > 220:
        text = text[:220].rsplit(" ", 1)[0] + "..."
    return text
