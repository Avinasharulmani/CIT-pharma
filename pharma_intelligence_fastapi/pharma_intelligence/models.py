from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SourceChunk(BaseModel):
    """Base source unit extracted from one page, slide, image, audio, or video."""

    source_no: Optional[int] = None
    source_type: str
    text: str = ""
    description_of_image_video: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class KeyMessageMatch(BaseModel):
    """Base model for one mapped key-message result."""

    key_message_id: str
    brand_product: str
    key_message: str
    key_words: str
    confidence_score: float
    source_location: str
    description_of_image_video: Optional[str] = None


class BaseIntelligenceResponse(BaseModel):
    """Base model returned by the Pharma Intelligence Engine."""

    file_name: str
    file_type: str
    detected_language: str
    target_language: str
    translated: bool
    summary: str
    extracted_keywords: List[str]
    description_of_image_video: Optional[str] = None
    transcript: Optional[str] = None  # For video/audio transcript
    frame_descriptions: Optional[str] = None  # For video frame VLM descriptions
    content_summary: Optional[str] = None
    full_transcript: Optional[str] = None
    transcript_available: bool = False
    summary_available: bool = False
    key_message_matches: List[KeyMessageMatch] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    accuracy_report: Optional[dict] = None


class ErrorResponse(BaseModel):
    status: str = "error"
    message: str
