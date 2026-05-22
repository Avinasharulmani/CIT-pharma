from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class MLRChecklistItem(BaseModel):
    item_code: str
    section_name: str
    checklist_description: str
    reviewer_type: str = ""
    mandatory_status: str = ""
    checklist_embedding: list[float] = Field(default_factory=list)
    checklist_version: str = ""
    source_file_name: str = ""
    created_date: datetime | None = None
    updated_date: datetime | None = None


class MLRContentChunk(BaseModel):
    chunk_id: str
    location: str
    text: str
    embedding: list[float] = Field(default_factory=list)


class MLRMatchedChunk(BaseModel):
    chunk_id: str
    location: str
    text: str
    similarity_score: float


class MLRItemResult(BaseModel):
    item_code: str
    checklist_id: str = ""
    section: str
    checklist_description: str
    checklist_criteria: str = ""
    reviewer_type: str = ""
    mandatory_status: str = ""
    status: Literal["Satisfied", "Not Satisfied", "Needs Review"]
    evidence: str = ""
    evidence_found: str = ""
    reasoning: str = ""
    recommendation: str = ""
    similarity_score: float = 0.0
    matched_location: str = ""
    matched_chunks: list[MLRMatchedChunk] = Field(default_factory=list)


class MLRReviewReport(BaseModel):
    upload_id: str
    filename: str
    overall_mlr_status: str
    total_checklist_items: int
    satisfied_count: int
    not_satisfied_count: int
    cannot_determine_count: int
    section_wise_results: list[dict[str, Any]]
    item_wise_results: list[MLRItemResult]
    checklist_version: str = ""
    created_date: datetime | None = None
    status_summary: dict[str, Any] = Field(default_factory=dict)
