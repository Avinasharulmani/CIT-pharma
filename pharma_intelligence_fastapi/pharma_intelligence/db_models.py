from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


class KeyMessageRecord(Base):
    __tablename__ = "key_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    key_message_id: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    brand_product: Mapped[str] = mapped_column(Text, nullable=False, default="")
    key_message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    key_words: Mapped[str] = mapped_column(Text, nullable=False, default="")
    source_file: Mapped[str] = mapped_column(String(512), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)


class AnalysisResultRecord(Base):
    __tablename__ = "analysis_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    file_name: Mapped[str] = mapped_column(String(512), nullable=False)
    file_type: Mapped[str] = mapped_column(String(64), nullable=False)
    detected_language: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    target_language: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    translated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    extracted_keywords: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    description_of_image_video: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    frame_descriptions: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_metadata: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)

    matches: Mapped[list["AnalysisMatchRecord"]] = relationship(
        "AnalysisMatchRecord",
        back_populates="analysis",
        cascade="all, delete-orphan",
    )


class AnalysisMatchRecord(Base):
    __tablename__ = "analysis_matches"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analysis_results.id", ondelete="CASCADE"), nullable=False, index=True)
    key_message_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    brand_product: Mapped[str] = mapped_column(Text, nullable=False, default="")
    key_message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    key_words: Mapped[str] = mapped_column(Text, nullable=False, default="")
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    source_location: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    description_of_image_video: Mapped[str | None] = mapped_column(Text, nullable=True)

    analysis: Mapped[AnalysisResultRecord] = relationship("AnalysisResultRecord", back_populates="matches")
