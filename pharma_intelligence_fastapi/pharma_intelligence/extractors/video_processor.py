import re
import shutil
import subprocess
import base64
import json
import os
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple
from uuid import uuid4

from ..config import (
    OPENAI_API_KEY,
    ASR_MODEL_NAME,
    LOCAL_VIDEO_DESCRIPTION_ENABLED,
    VIDEO_DESCRIPTION_ENABLED,
    VIDEO_FRAME_SAMPLE_SECONDS,
    VIDEO_LOCAL_VLM_FRAME_LIMIT,
    VIDEO_MAX_OCR_FRAMES,
    VIDEO_MAX_SCENE_FRAMES,
    VIDEO_SCENE_SCAN_SECONDS,
    VISION_ACTIVITY_FRAME_LIMIT,
    VISION_ACTIVITY_MODEL,
)
from ..models import SourceChunk
from ..preprocessing import preprocess_image
from ..utils import clean_text
from .audio_processor import transcribe_audio_result
from .ocr_utils import run_ocr_on_image
from .text_quality import extraction_unit_metadata


def check_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            # Add the directory containing ffmpeg to PATH for current process
            ffmpeg_dir = str(Path(ffmpeg_exe).parent)
            import os
            if ffmpeg_dir not in os.environ["PATH"]:
                os.environ["PATH"] += os.pathsep + ffmpeg_dir
        except ImportError:
            raise RuntimeError("FFmpeg is required for video/audio extraction. Please install imageio-ffmpeg: pip install imageio-ffmpeg")
        except Exception as exc:
            raise RuntimeError(f"FFmpeg is required for video/audio extraction. Error finding binary: {exc}")


def extract_audio_from_video(video_path: Path, output_audio_path: Path) -> Path:
    """Video-audio extraction logic merged from the second file."""
    moviepy_error = None
    try:
        try:
            from moviepy.editor import VideoFileClip
        except Exception:
            from moviepy import VideoFileClip
    except Exception as exc:
        moviepy_error = exc
    else:
        clip = VideoFileClip(str(video_path))
        try:
            if clip.audio is None:
                raise RuntimeError("No audio track found in video.")
            clip.audio.write_audiofile(str(output_audio_path), codec="pcm_s16le", fps=16000, nbytes=2, logger=None)
            return output_audio_path
        except Exception as exc:
            moviepy_error = exc
        finally:
            clip.close()

    ffmpeg_exe = shutil.which("ffmpeg")
    if not ffmpeg_exe:
        try:
            import imageio_ffmpeg

            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = None

    if not ffmpeg_exe:
        raise RuntimeError(f"Could not extract video audio. MoviePy error: {moviepy_error}")

    fallback_output_audio_path = output_audio_path.with_name(f"{output_audio_path.stem}_ffmpeg{output_audio_path.suffix}")
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        str(fallback_output_audio_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            "Could not extract video audio. "
            f"MoviePy error: {moviepy_error}. FFmpeg error: {result.stderr[-600:]}"
        )
    return fallback_output_audio_path


def _detect_visual_activity(frame, previous_frame, has_ocr_text: bool) -> List[str]:
    """Return conservative visual activity signals without relying on generative captions."""
    try:
        import cv2
    except Exception:
        return []

    activities = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    try:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) == 1:
            activities.append("person")
        elif len(faces) > 1:
            activities.append("people")
    except Exception:
        pass

    if previous_frame is not None:
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.resize(previous_gray, (gray.shape[1], gray.shape[0]))
        frame_delta = cv2.absdiff(previous_gray, gray)
        if float(frame_delta.mean()) > 18:
            activities.append("scene_change")

    if has_ocr_text and not activities:
        activities.append("promotional_visual")

    return activities


def _activity_description_from_signals(signals: List[str], has_transcript: bool) -> str:
    signal_set = set(signals)
    descriptions = []

    if "people" in signal_set:
        if has_transcript:
            descriptions.append("People are speaking or interacting")
        else:
            descriptions.append("People are shown interacting")
    elif "person" in signal_set:
        if has_transcript:
            descriptions.append("Person is speaking or presenting")
        else:
            descriptions.append("Person is shown in the scene")

    if "promotional_visual" in signal_set:
        descriptions.append("Promotional visuals are displayed")

    unique_descriptions = []
    seen = set()
    for description in descriptions:
        normalized = re.sub(r"\W+", " ", description.lower()).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique_descriptions.append(description)

    return clean_text("; ".join(unique_descriptions[:3]))


def _frame_to_data_url(frame) -> Optional[str]:
    try:
        import cv2
    except Exception:
        return None

    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
    if not ok:
        return None
    encoded = base64.b64encode(buffer).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _describe_activity_with_vision(frame_data_urls: List[str], warnings: Optional[List[str]] = None) -> str:
    if not OPENAI_API_KEY or not frame_data_urls:
        if warnings is not None and not OPENAI_API_KEY:
            warnings.append("Exact visual activity description is disabled because OPENAI_API_KEY is not configured.")
        return ""

    content = [
        {
            "type": "input_text",
            "text": (
                "Look across these sampled video frames and describe only the visible human activities. "
                "Focus on what people are doing, for example presenting, talking to someone, sitting in a clinic, "
                "walking, showing a product, receiving care, or interacting. "
                "Do not describe or quote on-screen text, brand names, logos, captions, or claims. "
                "Do not guess locations, objects, products, or actions that are not clearly visible. "
                "Return one concise sentence under 24 words. If the activity is unclear, return: Visual activity unclear."
            ),
        }
    ]
    content.extend(
        {"type": "input_image", "image_url": image_url, "detail": "low"}
        for image_url in frame_data_urls[:VISION_ACTIVITY_FRAME_LIMIT]
    )

    payload = {
        "model": VISION_ACTIVITY_MODEL,
        "input": [{"role": "user", "content": content}],
        "max_output_tokens": 80,
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
                "Exact visual activity description is unavailable from the vision model; using local frame VLM fallback when possible."
            )
        return ""
    except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
        if warnings is not None:
            warnings.append(
                "Exact visual activity description is unavailable from the vision model; using local frame VLM fallback when possible."
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

    if not text or "visual activity unclear" in text.lower():
        return ""
    if len(text) > 220:
        text = text[:220].rsplit(" ", 1)[0] + "..."
    return text


def _describe_frame_with_local_vlm(frame) -> str:
    try:
        import cv2
        from PIL import Image

        from .image_processor import _ask_local_vlm, _local_vlm_caption_description, _normalize_visible_value

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        def ask(question: str) -> str:
            return clean_text(_normalize_visible_value(_ask_local_vlm(image, question))).lower().strip(" .")

        activity = ask("What are the people doing?")
        location = ask("Where are the people?")
        shown = ask("What is shown?")
        objects = ask("What objects are shown?")
        people_swimming = ask("Are people swimming?")
        people_walking = ask("Are people walking?")
        caption = clean_text(_local_vlm_caption_description(image)).strip()

        no_people_visible = location in {"none in photo", "none", "no people", "unknown"}

        if people_swimming in {"yes", "yeah"}:
            return "People are swimming in the water."
        if people_walking in {"yes", "yeah"}:
            if location.startswith("on "):
                place = f" {location}"
            elif location and not no_people_visible:
                place = f" on the {location}"
            else:
                place = ""
            return f"People are walking{place}."
        if no_people_visible and (
            any(term in shown for term in {"bottle", "juice", "tea", "product"})
            or any(term in objects for term in {"bottle", "juice", "tea", "product"})
        ):
            visible_product = objects if objects and objects not in {"none", "unknown"} else shown
            return f"Product bottles are shown ({visible_product})."
        if activity and activity not in {"none", "unknown", "unclear", "nothing"}:
            place = ""
            if location.startswith("on "):
                place = f" {location}"
            elif location and not no_people_visible and location not in {"unclear"}:
                place = f" on the {location}"
            if activity in {"standing", "sitting", "walking", "swimming", "playing", "dancing"}:
                return f"People are {activity}{place}."
            return f"Visible activity: {activity}{place}."
        if shown and shown not in {"none", "unknown", "unclear", "nothing"}:
            if any(term in shown for term in {"bottle", "juice", "tea", "product"}) or any(
                term in objects for term in {"bottle", "juice", "tea", "product"}
            ):
                visible_product = objects if objects and objects not in {"none", "unknown"} else shown
                return f"Product bottles are shown ({visible_product})."
            return f"Visible scene shows {shown}."
        if objects and objects not in {"none", "unknown", "unclear", "nothing"}:
            return f"Visible objects include {objects}."
        if caption:
            return caption
        return ""
    except Exception:
        return ""


def _format_seconds(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    minutes, secs = divmod(total_seconds, 60)
    return f"{minutes:02d}:{secs:02d}"


def _strip_activity_prefix(description: str) -> str:
    description = clean_text(description).rstrip(".")
    lowered = description.lower()
    prefixes = [
        "visible activity:",
    ]
    for prefix in prefixes:
        if lowered.startswith(prefix):
            return description[len(prefix) :].strip()
    if lowered.startswith("a person is "):
        return "a person " + description[len("a person is ") :].strip()
    if lowered.startswith("a person "):
        return description
    if lowered.startswith("people are "):
        return "people " + description[len("people are ") :].strip()
    if lowered.startswith("people "):
        return description
    return description


def _describe_frames_with_local_vlm(frames: List[Tuple[float, object]]) -> str:
    descriptions = []
    seen = set()
    for index, (timestamp_seconds, frame) in enumerate(frames, start=1):
        description = _describe_frame_with_local_vlm(frame)
        if not description:
            continue
        normalized = re.sub(r"\W+", " ", description.lower()).strip()
        if normalized in seen:
            continue
        seen.add(normalized)
        descriptions.append(f"{_format_seconds(timestamp_seconds)} Frame {index}: {description}")
        if len(descriptions) >= VISION_ACTIVITY_FRAME_LIMIT:
            break
    return clean_text("\n".join(descriptions))


def _frame_description_from_signals(signals: List[str], has_transcript: bool) -> str:
    description = _activity_description_from_signals(signals, has_transcript=has_transcript)
    return f"Sampled frames: {description}." if description else ""


def _summary_from_frame_descriptions(frame_descriptions: str) -> str:
    descriptions: List[str] = []
    seen = set()
    for line in clean_text(frame_descriptions).splitlines():
        description = re.sub(r"^(?:\d{2}:\d{2}\s+)?Frame\s+\d+:\s*", "", line).strip()
        normalized = re.sub(r"\W+", " ", description.lower()).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            descriptions.append(description.rstrip("."))
    if not descriptions:
        return ""

    cleaned = [_strip_activity_prefix(description) for description in descriptions[:3]]
    if len(cleaned) == 1:
        summary = f"Throughout the video, {cleaned[0]}."
    elif len(cleaned) == 2:
        summary = f"The video starts with {cleaned[0]}, then shows {cleaned[1]}."
    else:
        summary = f"The video starts with {cleaned[0]}, then shows {cleaned[1]}, and ends with {cleaned[2]}."

    if len(summary) > 300:
        summary = summary[:260].rsplit(" ", 1)[0] + "..."
    return clean_text(summary)


def _looks_like_ocr_noise(value: str) -> bool:
    text = clean_text(value)
    if not text:
        return True
    compact = re.sub(r"\s+", "", text.lower())
    if len(compact) >= 24:
        hex_chars = sum(1 for char in compact if char in "0123456789abcdef")
        if hex_chars / max(1, len(compact)) > 0.85:
            return True
    if re.search(r"([a-f0-9])\1{8,}", compact):
        return True
    words = re.findall(r"[a-zA-Z]{2,}", text)
    return len(text) > 40 and not words


def _clean_frame_ocr_text(frame_text: str) -> str:
    cleaned_lines = []
    for line in clean_text(frame_text).splitlines():
        label_match = re.match(r"^((?:\d{2}:\d{2}\s+)?Frame\s+\d+:\s*)(.*)$", line, flags=re.IGNORECASE)
        label = label_match.group(1) if label_match else ""
        body = label_match.group(2) if label_match else line
        if _looks_like_ocr_noise(body):
            continue
        cleaned_lines.append(label + clean_text(body))
    return clean_text("\n".join(cleaned_lines))


def extract_text_from_frame(frame) -> str:
    """Frame OCR logic merged from the second file."""
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python is required for video frame OCR.") from exc

    from PIL import Image

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    return run_ocr_on_image(image, fast=True).text


def extract_text_from_frame_result(frame):
    try:
        import cv2
        from PIL import Image
    except Exception as exc:
        raise RuntimeError("opencv-python and Pillow are required for video frame OCR.") from exc
    with tempfile.TemporaryDirectory() as folder:
        frame_path = Path(folder) / "frame.png"
        cv2.imwrite(str(frame_path), frame)
        preprocessed_path = preprocess_image(str(frame_path))
        try:
            rgb_image = Image.open(preprocessed_path).convert("RGB")
            return run_ocr_on_image(rgb_image, fast=True)
        finally:
            if preprocessed_path != str(frame_path):
                try:
                    os.remove(preprocessed_path)
                except Exception:
                    pass


def _frame_difference_score(frame, previous_frame) -> float:
    if previous_frame is None:
        return 100.0
    try:
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
        previous_gray = cv2.resize(previous_gray, (gray.shape[1], gray.shape[0]))
        return float(cv2.absdiff(previous_gray, gray).mean())
    except Exception:
        return 100.0


def extract_sampled_frame_text(
    video_path: Path,
    sample_seconds: int = VIDEO_FRAME_SAMPLE_SECONDS,
    warnings: Optional[List[str]] = None,
) -> Tuple[str, str, int]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python is required for video frame extraction. Install with: pip install opencv-python") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(fps * sample_seconds), 1)
    frame_index = 0
    sampled_count = 0
    previous_frame = None
    ocr_texts: List[str] = []
    activity_signals: List[str] = []
    frame_signal_descriptions: List[str] = []
    frame_data_urls: List[str] = []
    sampled_frames_for_vlm: List[Tuple[float, object]] = []
    frame_results: list[dict] = []
    accepted_frames: list[object] = []

    def collect_frame(frame, timestamp_seconds: float, selected_reason: str) -> None:
        nonlocal previous_frame, sampled_count
        if sampled_count >= VIDEO_MAX_OCR_FRAMES:
            return
        duplicate_score = min((_frame_difference_score(frame, accepted) for accepted in accepted_frames[-5:]), default=100.0)
        if duplicate_score < 3.0:
            if warnings is not None:
                warnings.append(f"duplicate frame skipped at {_format_seconds(timestamp_seconds)}")
            return
        sampled_count += 1
        ocr_result = extract_text_from_frame_result(frame)
        text = ocr_result.text
        if text:
            ocr_texts.append(f"{_format_seconds(timestamp_seconds)} Frame {sampled_count}: {text}")

        frame_signals = _detect_visual_activity(frame, previous_frame, bool(text))
        activity_signals.extend(frame_signals)
        frame_signal_description = _activity_description_from_signals(frame_signals, has_transcript=False)
        if frame_signal_description:
            frame_signal_descriptions.append(
                f"{_format_seconds(timestamp_seconds)} Frame {sampled_count}: {frame_signal_description}."
            )
        previous_frame = frame.copy()
        accepted_frames.append(frame.copy())
        description_frame_limit = min(VISION_ACTIVITY_FRAME_LIMIT, VIDEO_MAX_OCR_FRAMES, VIDEO_LOCAL_VLM_FRAME_LIMIT)
        if len(frame_data_urls) < description_frame_limit:
            frame_data_url = _frame_to_data_url(frame)
            if frame_data_url:
                frame_data_urls.append(frame_data_url)
            sampled_frames_for_vlm.append((timestamp_seconds, frame.copy()))
        frame_results.append(
            {
                **extraction_unit_metadata(
                    source_type="video_frame",
                    unit_number=sampled_count,
                    timestamp=timestamp_seconds,
                    raw_text=text,
                    cleaned_text=text,
                    extraction_method=f"frame_ocr:{ocr_result.engine or 'none'}:{ocr_result.selected_variant or 'none'}",
                    confidence_score=ocr_result.confidence_score,
                    warnings=ocr_result.warnings or [],
                ),
                "timestamp": timestamp_seconds,
                "timestamp_label": _format_seconds(timestamp_seconds),
                "frame_index": int(round(timestamp_seconds * fps)),
                "ocr_text": text,
                "visual_description": frame_signal_description,
                "selected_reason": selected_reason,
                "confidence_score": ocr_result.confidence_score,
            }
        )

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames > 0:
        last_scene_frame = None
        for frame_index in range(0, total_frames, frame_interval):
            if sampled_count >= VIDEO_MAX_OCR_FRAMES:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if ok:
                collect_frame(frame, frame_index / fps, "fixed_interval")
        scene_step = max(int(fps * max(VIDEO_SCENE_SCAN_SECONDS, 1)), 1)
        scene_collected = 0
        for frame_index in range(0, total_frames, scene_step):
            if sampled_count >= VIDEO_MAX_OCR_FRAMES or scene_collected >= VIDEO_MAX_SCENE_FRAMES:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                continue
            diff = _frame_difference_score(frame, last_scene_frame)
            if diff > 18.0:
                before_count = sampled_count
                collect_frame(frame, frame_index / fps, "scene_change")
                if sampled_count > before_count:
                    scene_collected += 1
                last_scene_frame = frame.copy()
    else:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % frame_interval == 0:
                collect_frame(frame, frame_index / fps, "fixed_interval")
            if sampled_count >= VIDEO_MAX_OCR_FRAMES:
                break
            frame_index += 1

    cap.release()
    ocr_result = _clean_frame_ocr_text("\n".join(ocr_texts))
    vlm_result = _describe_frames_with_local_vlm(sampled_frames_for_vlm) if LOCAL_VIDEO_DESCRIPTION_ENABLED and sampled_frames_for_vlm else ""
    if frame_signal_descriptions:
        described_frame_labels = {
            match.group(1).lower()
            for match in re.finditer(r"((?:\d{2}:\d{2}\s+)?Frame\s+\d+):", vlm_result, flags=re.IGNORECASE)
        }
        remaining_signal_descriptions = [
            value
            for value in frame_signal_descriptions
            if not (
                (match := re.match(r"^((?:\d{2}:\d{2}\s+)?Frame\s+\d+):", value, flags=re.IGNORECASE))
                and match.group(1).lower() in described_frame_labels
            )
        ]
        vlm_result = clean_text("\n".join([vlm_result, *remaining_signal_descriptions]))
    if vlm_result and warnings is not None:
        warnings[:] = [
            warning
            for warning in warnings
            if not str(warning).startswith("Exact visual activity description is unavailable from the vision model")
        ]
    if not vlm_result:
        vision_description = _describe_activity_with_vision(frame_data_urls, warnings=warnings) if VIDEO_DESCRIPTION_ENABLED else ""
        if vision_description:
            vlm_result = vision_description
    if not vlm_result and not OPENAI_API_KEY and (VIDEO_DESCRIPTION_ENABLED or LOCAL_VIDEO_DESCRIPTION_ENABLED):
        vlm_result = _frame_description_from_signals(activity_signals, has_transcript=False)
    for frame_result in frame_results:
        label = frame_result.get("timestamp_label")
        description = ""
        for line in clean_text(vlm_result).splitlines():
            if label and line.startswith(str(label)):
                description = re.sub(r"^(?:\d{2}:\d{2}\s+)?Frame\s+\d+:\s*", "", line).strip()
                break
        if description:
            frame_result["visual_description"] = description
    return ocr_result, vlm_result, sampled_count, frame_results


def extract_video_intelligence(video_path: Path) -> List[SourceChunk]:
    check_ffmpeg()
    warnings = []
    scratch_files = []
    try:
        audio_path = video_path.parent / f"{video_path.stem}_{uuid4().hex}_audio.wav"
        scratch_files.extend([audio_path, audio_path.with_name(f"{audio_path.stem}_ffmpeg{audio_path.suffix}")])
        transcript = ""
        transcript_segments = []
        try:
            extract_audio_from_video(video_path, audio_path)
            audio_result = transcribe_audio_result(audio_path, model_size=ASR_MODEL_NAME)
            transcript = audio_result.get("text", "")
            transcript_segments = audio_result.get("segments", []) or []
            warnings.extend(audio_result.get("warnings", []) or [])
        except Exception as exc:
            warnings.append(f"Audio extraction/transcription warning: {exc}")

        frame_text = ""
        vlm_descriptions = ""
        sampled_count = 0
        frame_results = []
        try:
            frame_text, vlm_descriptions, sampled_count, frame_results = extract_sampled_frame_text(video_path, warnings=warnings)
        except Exception as exc:
            warnings.append(f"Frame extraction warning: {exc}")

        combined_text = clean_text("\n".join([transcript, frame_text]))
        
        # Description should capture visual activity only, not OCR/transcript text.
        fallback_signal_values = {"person", "people", "scene_change", "promotional_visual"}
        raw_descriptions = clean_text(vlm_descriptions)
        activity_signals = raw_descriptions.split("; ") if raw_descriptions else []
        if activity_signals and all(signal in fallback_signal_values for signal in activity_signals):
            description = _activity_description_from_signals(activity_signals, has_transcript=bool(transcript.strip()))
        elif re.match(r"^(?:\d{2}:\d{2}\s+)?Frame\s+\d+:", raw_descriptions):
            description = _summary_from_frame_descriptions(raw_descriptions)
        else:
            description = raw_descriptions
        
        # Return transcript and frame descriptions separately for display before table
        return [
            SourceChunk(
                source_no=1,
                source_type="video",
                text=combined_text,
                description_of_image_video=description,
                metadata={
                    **extraction_unit_metadata(
                        source_type="video",
                        unit_number=1,
                        raw_text=combined_text,
                        cleaned_text=combined_text,
                        extraction_method="video_audio_transcript_and_frame_ocr",
                        confidence_score=None,
                        warnings=warnings,
                    ),
                    "sampled_frames": sampled_count, 
                    "warnings": warnings,
                    "transcript": transcript,
                    "transcript_segments": transcript_segments,
                    "frame_text": frame_text,
                    "frame_descriptions": vlm_descriptions,
                    "frame_results": frame_results,
                    "frame_level": frame_results,
                },
            )
        ]
    finally:
        for scratch_file in scratch_files:
            try:
                scratch_file.unlink(missing_ok=True)
            except Exception:
                pass
