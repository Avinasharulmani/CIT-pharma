from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np

from ..config import MLR_EMBEDDING_LOCAL_FILES_ONLY, MLR_EMBEDDING_MODEL_NAME, MLR_TOP_MATCHES
from ..utils import clean_text
from .mlr_models import MLRMatchedChunk


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MLR_EMBEDDING_MODEL_NAME, local_files_only=MLR_EMBEDDING_LOCAL_FILES_ONLY)


def embed_text(text: str) -> list[float]:
    """
    Converts text into an embedding using the configured MLR embedding model.
    """
    value = clean_text(text)
    if not value:
        return []
    embedding = _model().encode([value], convert_to_numpy=True, normalize_embeddings=True)[0]
    return np.asarray(embedding, dtype=np.float32).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    values = [clean_text(text) for text in texts]
    if not values:
        return []
    embeddings = _model().encode(values, convert_to_numpy=True, normalize_embeddings=True)
    return np.asarray(embeddings, dtype=np.float32).tolist()


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    a_array = np.asarray(a, dtype=np.float32)
    b_array = np.asarray(b, dtype=np.float32)
    denominator = float(np.linalg.norm(a_array) * np.linalg.norm(b_array))
    if denominator <= 0:
        return 0.0
    return float(np.dot(a_array, b_array) / denominator)


def find_relevant_content_for_checklist(
    checklist_item: dict[str, Any],
    content_chunks: list[dict[str, Any]],
    top_k: int = MLR_TOP_MATCHES,
) -> list[MLRMatchedChunk]:
    """
    Uses MLR embeddings and cosine similarity to find the most relevant uploaded content chunks for one checklist item.
    """
    criteria = (
        checklist_item.get("checklist_criteria")
        or checklist_item.get("checklist_description")
        or checklist_item.get("criteria")
        or ""
    )
    checklist_embedding = embed_text(str(criteria))
    ranked = []
    for chunk in content_chunks:
        score = cosine_similarity(checklist_embedding, chunk.get("embedding") or [])
        ranked.append(
            MLRMatchedChunk(
                chunk_id=str(chunk.get("chunk_id") or ""),
                location=str(chunk.get("location") or ""),
                text=str(chunk.get("text") or ""),
                similarity_score=round(score, 4),
            )
        )
    ranked.sort(key=lambda item: item.similarity_score, reverse=True)
    return ranked[: max(int(top_k or 1), 1)]
