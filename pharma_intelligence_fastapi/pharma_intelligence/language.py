from typing import Tuple


def detect_language(text: str) -> str:
    if not text or len(text.strip()) < 20:
        return "unknown"
    try:
        from langdetect import detect

        return detect(text[:4000])
    except Exception:
        return "unknown"


def translate_text(text: str, target_language: str = "en") -> Tuple[str, bool]:
    """Translate text when possible. Falls back to original text if translator is unavailable."""
    if not text or not target_language:
        return text, False

    source_language = detect_language(text)
    if source_language == "unknown" or source_language == target_language:
        return text, False

    try:
        from deep_translator import GoogleTranslator

        translated = GoogleTranslator(source="auto", target=target_language).translate(text[:4800])
        if len(text) > 4800:
            translated += "\n\n[Translation note: only the first 4800 characters were translated for this response.]"
        return translated, True
    except Exception:
        return text, False
