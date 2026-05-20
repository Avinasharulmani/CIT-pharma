import os
import shutil
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

def _setup_ffmpeg():
    if not shutil.which("ffmpeg"):
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            ffmpeg_dir = str(Path(ffmpeg_exe).parent)
            if ffmpeg_dir not in os.environ["PATH"]:
                os.environ["PATH"] += os.pathsep + ffmpeg_dir
        except ImportError:
            pass

_setup_ffmpeg()

BASE_DIR = Path(__file__).resolve().parent.parent
if load_dotenv:
    load_dotenv(BASE_DIR / ".env")

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORTS_DIR = BASE_DIR / "reports"
DEFAULT_KEY_MESSAGE_PATH = BASE_DIR / "Key_Messages_Final_NEW_all_media_rows.xlsx"
SAMPLE_KEY_MESSAGE_PATH = BASE_DIR / "sample_key_messages.csv"
MLR_GUIDELINES_DIR = Path(os.getenv("MLR_GUIDELINES_DIR", "")).expanduser() if os.getenv("MLR_GUIDELINES_DIR") else None
MLR_MARKETING_GUIDELINE_PATH = Path(os.getenv("MLR_MARKETING_GUIDELINE_PATH", "")).expanduser() if os.getenv("MLR_MARKETING_GUIDELINE_PATH") else None
MLR_LEGAL_GUIDELINE_PATH = Path(os.getenv("MLR_LEGAL_GUIDELINE_PATH", "")).expanduser() if os.getenv("MLR_LEGAL_GUIDELINE_PATH") else None
MLR_REGULATORY_GUIDELINE_PATH = Path(os.getenv("MLR_REGULATORY_GUIDELINE_PATH", "")).expanduser() if os.getenv("MLR_REGULATORY_GUIDELINE_PATH") else None
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB = os.getenv("MONGODB_DB", "pharma_intelligence")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:root@localhost:5432/Keymessages")
ASR_PROVIDER = os.getenv("ASR_PROVIDER", "openai_whisper")
ASR_MODEL_NAME = os.getenv("ASR_MODEL_NAME", os.getenv("WHISPER_MODEL_SIZE", "small"))
ASR_FALLBACK_MODEL_NAME = os.getenv("ASR_FALLBACK_MODEL_NAME", "base")
ASR_LANGUAGE = os.getenv("ASR_LANGUAGE", "en").strip() or None
ASR_BEAM_SIZE = int(os.getenv("ASR_BEAM_SIZE", "5"))
ASR_BEST_OF = int(os.getenv("ASR_BEST_OF", "5"))
ASR_TEMPERATURE = float(os.getenv("ASR_TEMPERATURE", "0"))
ASR_CONDITION_ON_PREVIOUS_TEXT = os.getenv("ASR_CONDITION_ON_PREVIOUS_TEXT", "false").lower() in {"1", "true", "yes", "on"}
ASR_INITIAL_PROMPT = os.getenv("ASR_INITIAL_PROMPT", "")
ASR_CONTEXT_FROM_FILENAME = os.getenv("ASR_CONTEXT_FROM_FILENAME", "false").lower() in {"1", "true", "yes", "on"}
ASR_NO_SPEECH_THRESHOLD = float(os.getenv("ASR_NO_SPEECH_THRESHOLD", "0.55"))
ASR_LOGPROB_THRESHOLD = float(os.getenv("ASR_LOGPROB_THRESHOLD", "-0.85"))
ASR_COMPRESSION_RATIO_THRESHOLD = float(os.getenv("ASR_COMPRESSION_RATIO_THRESHOLD", "2.4"))
ASR_SOURCE_SEPARATION_ENABLED = os.getenv("ASR_SOURCE_SEPARATION_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
ASR_SOURCE_SEPARATION_PROVIDER = os.getenv("ASR_SOURCE_SEPARATION_PROVIDER", "auto").lower()
ASR_SOURCE_SEPARATION_KEEP_FILES = os.getenv("ASR_SOURCE_SEPARATION_KEEP_FILES", "false").lower() in {"1", "true", "yes", "on"}
NO_SUMMARY_MESSAGE = os.getenv("NO_SUMMARY_MESSAGE", "No readable content found for summary generation.")
NO_SPEECH_MESSAGE = os.getenv("NO_SPEECH_MESSAGE", "No speech detected in the audio/video.")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
VISION_ACTIVITY_MODEL = os.getenv("VISION_ACTIVITY_MODEL", "gpt-4.1-mini")
VISION_ACTIVITY_FRAME_LIMIT = int(os.getenv("VISION_ACTIVITY_FRAME_LIMIT", "6"))
VIDEO_DESCRIPTION_ENABLED = os.getenv("VIDEO_DESCRIPTION_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
LOCAL_VIDEO_DESCRIPTION_ENABLED = os.getenv("LOCAL_VIDEO_DESCRIPTION_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
IMAGE_DESCRIPTION_ENABLED = os.getenv("IMAGE_DESCRIPTION_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
LOCAL_IMAGE_DESCRIPTION_ENABLED = os.getenv("LOCAL_IMAGE_DESCRIPTION_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
EMBEDDING_LOCAL_FILES_ONLY = os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "true").lower() in {"1", "true", "yes", "on"}
EMBEDDING_MODEL_ENABLED = os.getenv("EMBEDDING_MODEL_ENABLED", "false").lower() in {"1", "true", "yes", "on"}
KEYWORD_MODEL_ENABLED = os.getenv("KEYWORD_MODEL_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
IMAGE_OCR_MODE = os.getenv("IMAGE_OCR_MODE", "fast").lower()
IMAGE_EASYOCR_ENABLED = os.getenv("IMAGE_EASYOCR_ENABLED", "true").lower() in {"1", "true", "yes", "on"}

SUPPORTED_FILE_TYPES = {
    ".pdf": {"file_type": "pdf", "category": "Documents", "can_preview": True, "can_full_text_search": True, "extractor_name": "pdf", "preview_type": "pdf"},
    ".doc": {"file_type": "document", "category": "Documents", "can_preview": True, "can_full_text_search": True, "extractor_name": "document", "preview_type": "document"},
    ".docx": {"file_type": "document", "category": "Documents", "can_preview": True, "can_full_text_search": True, "extractor_name": "document", "preview_type": "document"},
    ".docm": {"file_type": "document", "category": "Documents", "can_preview": True, "can_full_text_search": True, "extractor_name": "document", "preview_type": "document"},
    ".dot": {"file_type": "document", "category": "Documents", "can_preview": True, "can_full_text_search": True, "extractor_name": "document", "preview_type": "document"},
    ".dotx": {"file_type": "document", "category": "Documents", "can_preview": True, "can_full_text_search": True, "extractor_name": "document", "preview_type": "document"},
    ".dotm": {"file_type": "document", "category": "Documents", "can_preview": True, "can_full_text_search": True, "extractor_name": "document", "preview_type": "document"},
    ".rtf": {"file_type": "document", "category": "Documents", "can_preview": True, "can_full_text_search": True, "extractor_name": "document", "preview_type": "document"},
    ".hwp": {"file_type": "document", "category": "Documents", "can_preview": False, "can_full_text_search": True, "extractor_name": "document", "preview_type": "fallback"},
    ".ppt": {"file_type": "ppt", "category": "Presentations", "can_preview": True, "can_full_text_search": True, "extractor_name": "ppt", "preview_type": "slides"},
    ".pptx": {"file_type": "ppt", "category": "Presentations", "can_preview": True, "can_full_text_search": True, "extractor_name": "ppt", "preview_type": "slides"},
    ".xls": {"file_type": "spreadsheet", "category": "Spreadsheets/Data", "can_preview": True, "can_full_text_search": True, "extractor_name": "spreadsheet", "preview_type": "table"},
    ".xlsx": {"file_type": "spreadsheet", "category": "Spreadsheets/Data", "can_preview": True, "can_full_text_search": True, "extractor_name": "spreadsheet", "preview_type": "table"},
    ".csv": {"file_type": "spreadsheet", "category": "Spreadsheets/Data", "can_preview": True, "can_full_text_search": True, "extractor_name": "spreadsheet", "preview_type": "table"},
    ".odc": {"file_type": "spreadsheet", "category": "Spreadsheets/Data", "can_preview": True, "can_full_text_search": True, "extractor_name": "spreadsheet", "preview_type": "text"},
    ".cov": {"file_type": "spreadsheet", "category": "Spreadsheets/Data", "can_preview": True, "can_full_text_search": True, "extractor_name": "spreadsheet", "preview_type": "text"},
    ".ext": {"file_type": "spreadsheet", "category": "Spreadsheets/Data", "can_preview": True, "can_full_text_search": True, "extractor_name": "spreadsheet", "preview_type": "text"},
    ".inp": {"file_type": "spreadsheet", "category": "Spreadsheets/Data", "can_preview": True, "can_full_text_search": True, "extractor_name": "spreadsheet", "preview_type": "text"},
    ".lst": {"file_type": "spreadsheet", "category": "Spreadsheets/Data", "can_preview": True, "can_full_text_search": True, "extractor_name": "spreadsheet", "preview_type": "text"},
    ".op": {"file_type": "spreadsheet", "category": "Spreadsheets/Data", "can_preview": True, "can_full_text_search": True, "extractor_name": "spreadsheet", "preview_type": "text"},
    ".param": {"file_type": "spreadsheet", "category": "Spreadsheets/Data", "can_preview": True, "can_full_text_search": True, "extractor_name": "spreadsheet", "preview_type": "text"},
    ".r": {"file_type": "spreadsheet", "category": "Spreadsheets/Data", "can_preview": True, "can_full_text_search": True, "extractor_name": "spreadsheet", "preview_type": "text"},
    ".sas": {"file_type": "spreadsheet", "category": "Spreadsheets/Data", "can_preview": True, "can_full_text_search": True, "extractor_name": "spreadsheet", "preview_type": "text"},
    ".scm": {"file_type": "spreadsheet", "category": "Spreadsheets/Data", "can_preview": True, "can_full_text_search": True, "extractor_name": "spreadsheet", "preview_type": "text"},
    ".jsl": {"file_type": "spreadsheet", "category": "Spreadsheets/Data", "can_preview": True, "can_full_text_search": True, "extractor_name": "spreadsheet", "preview_type": "text"},
    ".ssc": {"file_type": "spreadsheet", "category": "Spreadsheets/Data", "can_preview": True, "can_full_text_search": True, "extractor_name": "spreadsheet", "preview_type": "text"},
    ".sbml": {"file_type": "spreadsheet", "category": "Spreadsheets/Data", "can_preview": True, "can_full_text_search": True, "extractor_name": "spreadsheet", "preview_type": "text"},
    ".jpg": {"file_type": "image", "category": "Images/Design", "can_preview": True, "can_full_text_search": True, "extractor_name": "image", "preview_type": "image"},
    ".jpeg": {"file_type": "image", "category": "Images/Design", "can_preview": True, "can_full_text_search": True, "extractor_name": "image", "preview_type": "image"},
    ".png": {"file_type": "image", "category": "Images/Design", "can_preview": True, "can_full_text_search": True, "extractor_name": "image", "preview_type": "image"},
    ".bmp": {"file_type": "image", "category": "Images/Design", "can_preview": True, "can_full_text_search": True, "extractor_name": "image", "preview_type": "image"},
    ".gif": {"file_type": "image", "category": "Images/Design", "can_preview": True, "can_full_text_search": True, "extractor_name": "image", "preview_type": "image"},
    ".webp": {"file_type": "image", "category": "Images/Design", "can_preview": True, "can_full_text_search": True, "extractor_name": "image", "preview_type": "image"},
    ".avif": {"file_type": "image", "category": "Images/Design", "can_preview": True, "can_full_text_search": True, "extractor_name": "image", "preview_type": "image"},
    ".heif": {"file_type": "image", "category": "Images/Design", "can_preview": True, "can_full_text_search": True, "extractor_name": "image", "preview_type": "image"},
    ".heic": {"file_type": "image", "category": "Images/Design", "can_preview": True, "can_full_text_search": True, "extractor_name": "image", "preview_type": "image"},
    ".tif": {"file_type": "image", "category": "Images/Design", "can_preview": True, "can_full_text_search": True, "extractor_name": "image", "preview_type": "image"},
    ".tiff": {"file_type": "image", "category": "Images/Design", "can_preview": True, "can_full_text_search": True, "extractor_name": "image", "preview_type": "image"},
    ".raw": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": True, "extractor_name": "fallback", "preview_type": "fallback"},
    ".svg": {"file_type": "image", "category": "Images/Design", "can_preview": True, "can_full_text_search": True, "extractor_name": "markup", "preview_type": "image"},
    ".eps": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": True, "extractor_name": "fallback", "preview_type": "fallback"},
    ".ai": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": True, "extractor_name": "fallback", "preview_type": "fallback"},
    ".psd": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": True, "extractor_name": "fallback", "preview_type": "fallback"},
    ".indd": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": True, "extractor_name": "fallback", "preview_type": "fallback"},
    ".dng": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": False, "extractor_name": "fallback", "preview_type": "fallback"},
    ".crw": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": False, "extractor_name": "fallback", "preview_type": "fallback"},
    ".cr2": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": False, "extractor_name": "fallback", "preview_type": "fallback"},
    ".raf": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": False, "extractor_name": "fallback", "preview_type": "fallback"},
    ".3fr": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": False, "extractor_name": "fallback", "preview_type": "fallback"},
    ".dcr": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": False, "extractor_name": "fallback", "preview_type": "fallback"},
    ".kdc": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": False, "extractor_name": "fallback", "preview_type": "fallback"},
    ".nef": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": False, "extractor_name": "fallback", "preview_type": "fallback"},
    ".nrw": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": False, "extractor_name": "fallback", "preview_type": "fallback"},
    ".orf": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": False, "extractor_name": "fallback", "preview_type": "fallback"},
    ".rw2": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": False, "extractor_name": "fallback", "preview_type": "fallback"},
    ".pef": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": False, "extractor_name": "fallback", "preview_type": "fallback"},
    ".arw": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": False, "extractor_name": "fallback", "preview_type": "fallback"},
    ".srf": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": False, "extractor_name": "fallback", "preview_type": "fallback"},
    ".sr2": {"file_type": "image", "category": "Images/Design", "can_preview": False, "can_full_text_search": False, "extractor_name": "fallback", "preview_type": "fallback"},
    ".eml": {"file_type": "email", "category": "Email", "can_preview": True, "can_full_text_search": True, "extractor_name": "email", "preview_type": "email"},
    ".msg": {"file_type": "email", "category": "Email", "can_preview": True, "can_full_text_search": True, "extractor_name": "email", "preview_type": "email"},
    ".html": {"file_type": "text", "category": "Web/Markup/Text", "can_preview": True, "can_full_text_search": True, "extractor_name": "markup", "preview_type": "html"},
    ".htm": {"file_type": "text", "category": "Web/Markup/Text", "can_preview": True, "can_full_text_search": True, "extractor_name": "markup", "preview_type": "html"},
    ".xml": {"file_type": "text", "category": "Web/Markup/Text", "can_preview": True, "can_full_text_search": True, "extractor_name": "markup", "preview_type": "text"},
    ".txt": {"file_type": "text", "category": "Web/Markup/Text", "can_preview": True, "can_full_text_search": True, "extractor_name": "text", "preview_type": "text"},
    ".bash": {"file_type": "script", "category": "Scripts", "can_preview": True, "can_full_text_search": True, "extractor_name": "text", "preview_type": "code"},
    ".csh": {"file_type": "script", "category": "Scripts", "can_preview": True, "can_full_text_search": True, "extractor_name": "text", "preview_type": "code"},
    ".sh": {"file_type": "script", "category": "Scripts", "can_preview": True, "can_full_text_search": True, "extractor_name": "text", "preview_type": "code"},
    ".zip": {"file_type": "archive", "category": "Archive/Package", "can_preview": True, "can_full_text_search": True, "extractor_name": "zip", "preview_type": "package"},
    ".mp3": {"file_type": "audio", "category": "Audio", "can_preview": True, "can_full_text_search": True, "extractor_name": "audio", "preview_type": "audio"},
    ".wav": {"file_type": "audio", "category": "Audio", "can_preview": True, "can_full_text_search": True, "extractor_name": "audio", "preview_type": "audio"},
    ".m4a": {"file_type": "audio", "category": "Audio", "can_preview": True, "can_full_text_search": True, "extractor_name": "audio", "preview_type": "audio"},
    ".flac": {"file_type": "audio", "category": "Audio", "can_preview": True, "can_full_text_search": True, "extractor_name": "audio", "preview_type": "audio"},
    ".ogg": {"file_type": "audio", "category": "Audio", "can_preview": True, "can_full_text_search": True, "extractor_name": "audio", "preview_type": "audio"},
    ".mp4": {"file_type": "video", "category": "Video", "can_preview": True, "can_full_text_search": True, "extractor_name": "video", "preview_type": "video"},
    ".avi": {"file_type": "video", "category": "Video", "can_preview": True, "can_full_text_search": True, "extractor_name": "video", "preview_type": "video"},
    ".mov": {"file_type": "video", "category": "Video", "can_preview": True, "can_full_text_search": True, "extractor_name": "video", "preview_type": "video"},
    ".mkv": {"file_type": "video", "category": "Video", "can_preview": True, "can_full_text_search": True, "extractor_name": "video", "preview_type": "video"},
}

ALLOWED_EXTENSIONS = {extension: config["file_type"] for extension, config in SUPPORTED_FILE_TYPES.items()}
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(200 * 1024 * 1024)))

DEFAULT_TARGET_LANGUAGE = "en"
DEFAULT_TOP_K = 0  # 0 means return every candidate key-message row.
SIMILARITY_THRESHOLD = 0.25
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
WHISPER_MODEL_SIZE = ASR_MODEL_NAME  # Backward-compatible alias for existing audio/video code.
VIDEO_FRAME_SAMPLE_SECONDS = 10
VIDEO_MAX_OCR_FRAMES = int(os.getenv("VIDEO_MAX_OCR_FRAMES", "4"))
VIDEO_SCENE_SCAN_SECONDS = int(os.getenv("VIDEO_SCENE_SCAN_SECONDS", "2"))
VIDEO_MAX_SCENE_FRAMES = int(os.getenv("VIDEO_MAX_SCENE_FRAMES", "2"))
VIDEO_LOCAL_VLM_FRAME_LIMIT = int(os.getenv("VIDEO_LOCAL_VLM_FRAME_LIMIT", "0"))
MAX_CHARS_FOR_SUMMARY = 1800
BRIEF_SUMMARY_MAX_CHARS = int(os.getenv("BRIEF_SUMMARY_MAX_CHARS", "320"))
MAX_CHARS_FOR_MAPPING = 6000
