from functools import lru_cache
import re
from typing import List

from .config import EMBEDDING_LOCAL_FILES_ONLY, EMBEDDING_MODEL_NAME, KEYWORD_MODEL_ENABLED
from .utils import clean_text


GENERIC_KEYWORD_TERMS = {
    "adult",
    "adults",
    "asset",
    "audio",
    "awareness",
    "focused",
    "impact",
    "healthcare",
    "people",
    "person",
    "shown",
    "video",
    "image",
}


@lru_cache(maxsize=1)
def _get_kw_model():
    if not KEYWORD_MODEL_ENABLED:
        return None
    try:
        from keybert import KeyBERT

        if EMBEDDING_LOCAL_FILES_ONLY:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
            return KeyBERT(model=model)
        return KeyBERT(model=EMBEDDING_MODEL_NAME)
    except Exception:
        return None



def _normalize_keyword(value: str) -> str:
    value = clean_text(value).lower()
    value = re.sub(r"[^a-z0-9+/\-\s]", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" -/")
    return value


def _keyword_terms(value: str) -> set[str]:
    return {term for term in re.findall(r"[a-z0-9]+", value.lower()) if len(term) > 2}


def _is_generic_keyword(value: str) -> bool:
    terms = _keyword_terms(value)
    return bool(terms) and terms <= GENERIC_KEYWORD_TERMS


def _dedupe_keywords(keywords: List[str], top_n: int) -> List[str]:
    output = []
    seen = set()
    for keyword in keywords:
        normalized = _normalize_keyword(keyword)
        if not normalized or _is_generic_keyword(normalized):
            continue
        token_key = re.sub(r"[^a-z0-9]+", " ", normalized).strip()
        if token_key in seen:
            continue
        if any(token_key in existing or existing in token_key for existing in seen):
            continue
        seen.add(token_key)
        output.append(normalized)
        if len(output) >= top_n:
            break
    return output


def extract_keywords(text: str, top_n: int = 8, seed_terms: List[str] | None = None, use_model: bool = True) -> List[str]:
    """Create keywords using KeyBERT. If KeyBERT is unavailable, use CountVectorizer fallback."""
    if not text or not text.strip():
        return []

    seed_terms = seed_terms or []
    seed_keywords = _dedupe_keywords(seed_terms, top_n=top_n)

    kw_model = _get_kw_model() if use_model else None
    if kw_model is not None:
        try:
            result = kw_model.extract_keywords(
                text,
                keyphrase_ngram_range=(2, 4),
                stop_words="english",
                top_n=max(top_n * 4, 16),
                use_mmr=True,
                diversity=0.85,
            )
            return _dedupe_keywords(seed_keywords + [kw for kw, _ in result], top_n=top_n)
        except Exception:
            pass
    return _dedupe_keywords(seed_keywords + _count_vectorizer_keywords(text, top_n=max(top_n * 4, 16)), top_n=top_n)


def _count_vectorizer_keywords(text: str, top_n: int = 8) -> List[str]:
    try:
        from sklearn.feature_extraction.text import CountVectorizer

        vectorizer = CountVectorizer(
            stop_words="english",
            ngram_range=(2, 4),
            max_features=120,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9+\-/]{2,}\b",
        )
        matrix = vectorizer.fit_transform([text])
        terms = vectorizer.get_feature_names_out()
        counts = matrix.toarray()[0]
        ranked = sorted(zip(terms, counts), key=lambda x: (x[1], len(x[0])), reverse=True)
        return [term for term, _ in ranked[:top_n]]
    except Exception:
        words = [w.strip(".,:;()[]{}!?\"'").lower() for w in text.split()]
        words = [w for w in words if len(w) > 4]
        seen = []
        for word in words:
            if word not in seen:
                seen.append(word)
            if len(seen) >= top_n:
                break
        return seen
