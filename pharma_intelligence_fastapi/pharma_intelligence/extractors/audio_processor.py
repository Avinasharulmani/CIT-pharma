from pathlib import Path
import os
import shutil
import logging
import re
import subprocess
import tempfile
from typing import Any

from ..config import (
    ASR_BEAM_SIZE,
    ASR_BEST_OF,
    ASR_COMPRESSION_RATIO_THRESHOLD,
    ASR_CONDITION_ON_PREVIOUS_TEXT,
    ASR_CONTEXT_FROM_FILENAME,
    ASR_FALLBACK_MODEL_NAME,
    ASR_INITIAL_PROMPT,
    ASR_LANGUAGE,
    ASR_LOGPROB_THRESHOLD,
    ASR_MODEL_NAME,
    ASR_NO_SPEECH_THRESHOLD,
    ASR_PROVIDER,
    ASR_TEMPERATURE,
    BASE_DIR,
)
from ..utils import clean_text

logger = logging.getLogger(__name__)


def _looks_like_garbage_transcript(text: str) -> bool:
    cleaned = clean_text(text)
    if not cleaned:
        return True
    tokens = re.findall(r"[A-Za-z0-9]+", cleaned)
    if not tokens:
        return True
    alpha_tokens = [token for token in tokens if re.search(r"[A-Za-z]", token)]
    if len(cleaned) < 8 and len(alpha_tokens) <= 1:
        return True
    if len(alpha_tokens) == 0:
        return True
    short_or_numeric = [
        token
        for token in tokens
        if len(token) <= 2 or token.isdigit() or re.fullmatch(r"[A-Za-z]\d+|\d+[A-Za-z]", token)
    ]
    if len(tokens) >= 4 and len(short_or_numeric) / len(tokens) > 0.7:
        return True
    letters = re.findall(r"[A-Za-z]", cleaned)
    if letters:
        vowels = re.findall(r"[AEIOUaeiou]", cleaned)
        if len(letters) >= 12 and len(vowels) / len(letters) < 0.18:
            return True
    if re.search(r"\b([A-Za-z0-9])(?:\s+\1\b){4,}", cleaned, flags=re.IGNORECASE):
        return True
    if re.search(r"([A-Za-z0-9])\1{5,}", cleaned):
        return True
    return False


def _segment_is_usable(segment: dict[str, Any]) -> bool:
    text = clean_text(str(segment.get("text", "")))
    if _looks_like_garbage_transcript(text):
        return False
    no_speech_prob = segment.get("no_speech_prob")
    avg_logprob = segment.get("avg_logprob")
    compression_ratio = segment.get("compression_ratio")
    try:
        no_speech_value = float(no_speech_prob)
    except Exception:
        no_speech_value = 0.0
    try:
        avg_logprob_value = float(avg_logprob)
    except Exception:
        avg_logprob_value = 0.0
    try:
        compression_value = float(compression_ratio)
    except Exception:
        compression_value = 0.0
    if no_speech_value >= 0.65 and avg_logprob_value <= -0.8:
        return False
    if avg_logprob_value <= -1.2:
        return False
    if compression_value >= 2.8:
        return False
    return True


def _filtered_transcription(result: dict[str, Any]) -> dict[str, Any]:
    raw_segments = result.get("segments", []) or []
    usable_segments = [segment for segment in raw_segments if _segment_is_usable(segment)]
    segment_text = clean_text(" ".join(str(segment.get("text", "")) for segment in usable_segments))
    raw_text = clean_text(result.get("text", ""))
    if segment_text:
        return {"text": segment_text, "segments": usable_segments}
    if raw_segments:
        return {"text": "", "segments": []}
    if _looks_like_garbage_transcript(raw_text):
        return {"text": "", "segments": []}
    return {"text": raw_text, "segments": []}


def _ensure_ffmpeg_on_path() -> None:
    if shutil.which("ffmpeg"):
        return
    try:
        import imageio_ffmpeg

        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        ffmpeg_dir_path = BASE_DIR / "scratch" / "ffmpeg"
        ffmpeg_dir_path.mkdir(parents=True, exist_ok=True)
        ffmpeg_link = ffmpeg_dir_path / "ffmpeg.exe"
        if not ffmpeg_link.exists():
            shutil.copyfile(ffmpeg_exe, ffmpeg_link)
        ffmpeg_dir = str(ffmpeg_dir_path)
        if ffmpeg_dir not in os.environ["PATH"]:
            os.environ["PATH"] += os.pathsep + ffmpeg_dir
    except Exception:
        pass


def normalize_audio_for_asr(audio_path: Path, warnings: list[str] | None = None) -> Path:
    """Convert audio to Whisper-friendly WAV: mono, 16 kHz, normalized volume."""
    _ensure_ffmpeg_on_path()
    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        if warnings is not None:
            warnings.append("audio normalization skipped because ffmpeg is not available")
        return audio_path
    output_dir = Path(tempfile.mkdtemp(prefix="pharma_asr_"))
    output_path = output_dir / f"{audio_path.stem}_normalized.wav"
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
        str(output_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and output_path.exists():
            return output_path
        if warnings is not None:
            warnings.append(f"audio normalization warning: {result.stderr[-400:]}")
    except Exception as exc:
        if warnings is not None:
            warnings.append(f"audio normalization warning: {exc}")
    return audio_path


def _build_initial_prompt(audio_path: Path, extra_context: str = "") -> str | None:
    prompt_parts = []
    if ASR_INITIAL_PROMPT:
        prompt_parts.append(ASR_INITIAL_PROMPT)
    if ASR_CONTEXT_FROM_FILENAME:
        filename_context = re.sub(r"[_-]+", " ", audio_path.stem)
        filename_context = re.sub(r"\b[a-f0-9]{16,}\b", " ", filename_context, flags=re.IGNORECASE)
        filename_context = clean_text(filename_context)
        if filename_context:
            prompt_parts.append(filename_context)
    if extra_context:
        prompt_parts.append(clean_text(extra_context))
    prompt = clean_text(". ".join(part for part in prompt_parts if part))
    return prompt or None


def _transcribe_options(audio_path: Path, context: str = "", *, use_prompt: bool = True, condition_on_previous_text: bool | None = None) -> dict[str, Any]:
    prompt = _build_initial_prompt(audio_path, extra_context=context) if use_prompt else None
    return {
        "language": ASR_LANGUAGE,
        "beam_size": ASR_BEAM_SIZE,
        "best_of": ASR_BEST_OF,
        "temperature": ASR_TEMPERATURE,
        "condition_on_previous_text": ASR_CONDITION_ON_PREVIOUS_TEXT if condition_on_previous_text is None else condition_on_previous_text,
        "initial_prompt": prompt,
        "no_speech_threshold": ASR_NO_SPEECH_THRESHOLD,
        "logprob_threshold": ASR_LOGPROB_THRESHOLD,
        "compression_ratio_threshold": ASR_COMPRESSION_RATIO_THRESHOLD,
        "fp16": False,
        "verbose": False,
    }


def _run_transcribe(model: Any, audio_path: Path, context: str = "") -> dict[str, Any]:
    result = model.transcribe(str(audio_path), **_transcribe_options(audio_path, context=context))
    filtered = _filtered_transcription(result)
    if filtered["text"] or not (_build_initial_prompt(audio_path, extra_context=context) or ASR_CONDITION_ON_PREVIOUS_TEXT):
        return result
    retry_result = model.transcribe(
        str(audio_path),
        **_transcribe_options(audio_path, context="", use_prompt=False, condition_on_previous_text=False),
    )
    retry_filtered = _filtered_transcription(retry_result)
    return retry_result if retry_filtered["text"] else result


def transcribe_audio_file(
    audio_path: Path,
    model_size: str = ASR_MODEL_NAME,
    provider: str = ASR_PROVIDER,
    context: str = "",
) -> str:
    """Audio code merged from the second file and adapted for FastAPI/library usage."""
    logger.info("Transcription started.")
    _ensure_ffmpeg_on_path()
    if provider != "openai_whisper":
        raise RuntimeError(f"Unsupported ASR provider configured: {provider}")
    try:
        import whisper
    except Exception as exc:
        raise RuntimeError("openai-whisper is required for audio transcription. Install with: pip install openai-whisper") from exc

    try:
        model = whisper.load_model(model_size)
    except Exception as exc:
        if not ASR_FALLBACK_MODEL_NAME or ASR_FALLBACK_MODEL_NAME == model_size:
            raise
        logger.warning("Could not load ASR model %s. Falling back to %s: %s", model_size, ASR_FALLBACK_MODEL_NAME, exc)
        model = whisper.load_model(ASR_FALLBACK_MODEL_NAME)
    warnings: list[str] = []
    normalized_audio_path = normalize_audio_for_asr(audio_path, warnings)
    result = _run_transcribe(model, normalized_audio_path, context=context)
    filtered = _filtered_transcription(result)
    transcript = filtered["text"]
    if transcript:
        logger.info("Transcription completed.")
    else:
        logger.warning("Transcription completed with no speech detected.")
    return transcript


def _transcript_segments(result: dict[str, Any]) -> list[dict[str, Any]]:
    segments = []
    language = result.get("language")
    for segment in result.get("segments", []) or []:
        text = clean_text(str(segment.get("text", "")))
        start = segment.get("start")
        end = segment.get("end")
        try:
            start_value = float(start)
            end_value = float(end)
        except Exception:
            continue
        if not text:
            continue
        segments.append(
            {
                "start": max(0.0, start_value),
                "end": max(max(0.0, start_value), end_value),
                "start_time": max(0.0, start_value),
                "end_time": max(max(0.0, start_value), end_value),
                "text": text,
                "language": language,
                "confidence": _segment_confidence(segment),
                "avg_logprob": segment.get("avg_logprob"),
                "no_speech_prob": segment.get("no_speech_prob"),
            }
        )
    return segments


def _segment_confidence(segment: dict[str, Any]) -> float | None:
    try:
        avg_logprob = float(segment.get("avg_logprob"))
    except Exception:
        return None
    # Whisper exposes log probabilities, not calibrated confidence. This is a bounded quality proxy.
    return round(max(0.0, min(1.0, (avg_logprob + 1.0))), 3)


def transcribe_audio_result(
    audio_path: Path,
    model_size: str = ASR_MODEL_NAME,
    provider: str = ASR_PROVIDER,
    context: str = "",
) -> dict[str, Any]:
    """Return transcript text plus timestamped segments for annotation workflows."""
    logger.info("Transcription with segments started.")
    warnings: list[str] = []
    _ensure_ffmpeg_on_path()
    if provider != "openai_whisper":
        raise RuntimeError(f"Unsupported ASR provider configured: {provider}")
    try:
        import whisper
    except Exception as exc:
        raise RuntimeError("openai-whisper is required for audio transcription. Install with: pip install openai-whisper") from exc

    try:
        model = whisper.load_model(model_size)
    except Exception as exc:
        if not ASR_FALLBACK_MODEL_NAME or ASR_FALLBACK_MODEL_NAME == model_size:
            raise
        logger.warning("Could not load ASR model %s. Falling back to %s: %s", model_size, ASR_FALLBACK_MODEL_NAME, exc)
        model = whisper.load_model(ASR_FALLBACK_MODEL_NAME)

    normalized_audio_path = normalize_audio_for_asr(audio_path, warnings)
    result = _run_transcribe(model, normalized_audio_path, context=context)
    filtered = _filtered_transcription(result)
    transcript = filtered["text"]
    segments = _transcript_segments({"segments": filtered["segments"], "language": result.get("language") or ASR_LANGUAGE})
    if transcript:
        logger.info("Transcription with segments completed.")
    else:
        logger.warning("Transcription with segments completed with no speech detected.")
    return {
        "text": transcript,
        "segments": segments,
        "language": result.get("language") or ASR_LANGUAGE,
        "model_size": model_size,
        "normalized_audio": str(normalized_audio_path),
        "warnings": warnings,
    }
