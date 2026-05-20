from difflib import SequenceMatcher
from functools import lru_cache
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config import EMBEDDING_LOCAL_FILES_ONLY, EMBEDDING_MODEL_ENABLED, EMBEDDING_MODEL_NAME, SIMILARITY_THRESHOLD
from ..keywords import extract_keywords
from ..models import KeyMessageMatch
from ..utils import clean_text, extract_possible_brand_name, split_into_chunks


def _brand_aliases(value: str) -> List[str]:
    brand = str(value).strip()
    if not brand:
        return []

    without_parentheses = re.sub(r"\(.*?\)", "", brand).strip()
    without_dosage_form = re.sub(
        r"\b(tablets?|capsules?|drops?|syrups?|injections?|inhalers?|ointments?|cream|sachet|eye drops)\b",
        "",
        without_parentheses,
        flags=re.IGNORECASE,
    ).strip()

    aliases = {
        brand,
        without_parentheses,
        without_dosage_form,
        re.split(r"[-\s/(]", without_dosage_form, maxsplit=1)[0].strip(),
    }
    return [alias.lower() for alias in aliases if len(alias.strip()) > 2]


def _build_alias_brand_counts(brands) -> dict:
    alias_brand_counts = {}
    for brand in brands:
        for alias in _brand_aliases(str(brand)):
            alias_brand_counts.setdefault(alias, set()).add(str(brand))
    return {alias: len(values) for alias, values in alias_brand_counts.items()}


def _contains_phrase(text_lower: str, phrase_lower: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(phrase_lower) + r"(?![a-z0-9])"
    return re.search(pattern, text_lower) is not None


def _phrase_position(text_lower: str, phrase_lower: str) -> Optional[int]:
    pattern = r"(?<![a-z0-9])" + re.escape(phrase_lower) + r"(?![a-z0-9])"
    match = re.search(pattern, text_lower)
    if match:
        return match.start()
    return None


def _normalized_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _text_terms(*values: str) -> set:
    terms = set()
    for value in values:
        for term in re.findall(r"[a-z0-9]+", str(value).lower()):
            if len(term) > 3:
                terms.add(term)
    return terms


def _display_keywords(brand: str, key_message: str, source_text: str, stored_keywords: str, use_model: bool = True) -> str:
    brand_terms = [
        alias
        for alias in _brand_aliases(brand)
        if len(alias) > 3
    ][:2]
    seed_terms = [
        "adult RSV prevention" if "adult rsv" in key_message.lower() or "adult rsv" in source_text.lower() else "",
        "lower respiratory disease" if "lower respiratory" in source_text.lower() else "",
        "RSV vaccine" if "rsv" in source_text.lower() and "vaccine" in source_text.lower() else "",
        "severe allergic reactions" if "severe allergic" in source_text.lower() else "",
        "weakened immune systems" if "weakened immune" in source_text.lower() else "",
        "injection site pain" if "injection site pain" in source_text.lower() else "",
        *brand_terms,
    ]
    stored = clean_text(stored_keywords)
    if stored:
        values = []
        seen = set()
        for value in [*seed_terms, *re.split(r"[,;|]\s*", stored)]:
            cleaned = clean_text(value)
            key = cleaned.lower()
            if cleaned and key not in seen:
                seen.add(key)
                values.append(cleaned)
            if len(values) >= 6:
                break
        if values:
            return ", ".join(values)
    keyword_text = clean_text("\n".join([brand, key_message, stored_keywords, source_text]))
    return ", ".join(extract_keywords(keyword_text, top_n=6, seed_terms=seed_terms, use_model=use_model))


def _evidence_description(text: str, brand: str, key_message: str) -> str:
    action_words = {
        "boosts",
        "controls",
        "delivers",
        "enhances",
        "ensures",
        "improves",
        "increases",
        "induces",
        "lowers",
        "prevents",
        "promotes",
        "provides",
        "reduces",
        "relieves",
        "supports",
        "treats",
    }
    terms = _text_terms(brand, key_message)
    lines = []
    seen = set()
    for line in clean_text(text).splitlines():
        cleaned = clean_text(line)
        normalized = re.sub(r"\W+", " ", cleaned.lower()).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        words = set(re.findall(r"[a-z0-9]+", normalized))
        has_action = bool(words & action_words)
        overlap = len(words & terms)
        if has_action or overlap >= 2:
            lines.append((overlap + (2 if has_action else 0), cleaned))

    if not lines:
        return ""

    selected = []
    for _, line in sorted(lines, key=lambda item: item[0], reverse=True):
        selected.append(line)
        if len(selected) >= 4:
            break
    description = "; ".join(selected)
    if len(description) > 320:
        description = description[:320].rsplit(" ", 1)[0] + "..."
    return description


def _contains_fuzzy_brand(text_lower: str, alias_lower: str) -> bool:
    return _fuzzy_brand_score(text_lower, alias_lower) >= 0.75


def _fuzzy_brand_score(text_lower: str, alias_lower: str) -> float:
    alias = _normalized_token(alias_lower)
    if len(alias) < 6:
        return 0.0

    tokens = [_normalized_token(token) for token in re.findall(r"[a-z0-9-]+", text_lower)]
    tokens = [token for token in tokens if len(token) >= 5]
    best_score = 0.0
    for token in tokens:
        if token == alias or alias in token:
            return 1.0
        if token[:3] == alias[:3]:
            best_score = max(best_score, SequenceMatcher(None, alias, token).ratio())
    return best_score


@lru_cache(maxsize=1)
def _embedding_model():
    if not EMBEDDING_MODEL_ENABLED:
        return None
    try:
        from sentence_transformers import SentenceTransformer

        return SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=EMBEDDING_LOCAL_FILES_ONLY)
    except Exception:
        return None


@lru_cache(maxsize=128)
def _get_embeddings(texts_tuple: Tuple[str, ...]) -> Optional[np.ndarray]:
    model = _embedding_model()
    if model is not None:
        return model.encode(list(texts_tuple), convert_to_numpy=True, normalize_embeddings=False)
    return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_norm = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-9, None)
    b_norm = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-9, None)
    return np.dot(a_norm, b_norm.T)


def _semantic_scores(query_texts: List[str], key_texts: List[str], use_model: bool = True) -> np.ndarray:
    query_embeddings = _get_embeddings(tuple(query_texts)) if use_model else None
    key_embeddings = _get_embeddings(tuple(key_texts)) if use_model else None

    if query_embeddings is not None and key_embeddings is not None:
        return _cosine_similarity(query_embeddings, key_embeddings)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(query_texts + key_texts)
    return cosine_similarity(matrix[: len(query_texts)], matrix[len(query_texts) :])


def match_key_messages(
    text: str,
    key_df: pd.DataFrame,
    id_col: str,
    brand_col: str,
    msg_col: str,
    top_k: int,
    source_location: str,
    description_of_image_video: Optional[str] = None,
    lock_to_detected_brand: bool = False,
    use_evidence_description: bool = True,
    use_keyword_model: bool = True,
) -> List[KeyMessageMatch]:
    chunks = split_into_chunks(text)
    if not chunks:
        chunks = [text]

    key_texts = key_df["__combined_text"].astype(str).tolist()
    if not key_texts:
        return []

    scores = _semantic_scores(chunks, key_texts, use_model=use_keyword_model)
    best_scores = scores.max(axis=0)
    semantic_scores = best_scores.copy()

    text_lower = text.lower()
    matched_brand_values = {}
    unique_brands = key_df[brand_col].unique()
    alias_brand_counts = _build_alias_brand_counts(unique_brands)
    for brand in unique_brands:
        aliases = [
            alias
            for alias in _brand_aliases(str(brand))
            if alias_brand_counts.get(alias, 0) == 1
        ]
        exact_positions = [
            position
            for alias in aliases
            for position in [_phrase_position(text_lower, alias)]
            if position is not None
        ]
        exact_match = bool(exact_positions)
        fuzzy_score = max((_fuzzy_brand_score(text_lower, alias) for alias in aliases), default=0.0)
        if exact_match or fuzzy_score >= 0.75:
            brand_mask = key_df[brand_col].astype(str) == str(brand)
            match_score = fuzzy_score
            if exact_match:
                first_position = min(exact_positions)
                brand_score = 0.99 - (min(first_position, 200) * 0.0001)
                best_scores[brand_mask] = np.minimum(0.99, semantic_scores[brand_mask] + 0.18)
                match_score = brand_score
            else:
                best_scores[brand_mask] = np.minimum(0.95, semantic_scores[brand_mask] + 0.08)
            matched_brand_values[str(brand)] = max(matched_brand_values.get(str(brand), 0.0), match_score)

    possible_brand = extract_possible_brand_name(text)
    if possible_brand:
        for idx, row in key_df.iterrows():
            brand = str(row[brand_col]).lower()
            if brand and (brand in possible_brand or possible_brand in brand):
                best_scores[idx] = min(0.95, float(semantic_scores[idx]) + 0.18)
                matched_brand_values[str(row[brand_col])] = 1.0

    effective_threshold = SIMILARITY_THRESHOLD if matched_brand_values else max(SIMILARITY_THRESHOLD, 0.4)
    show_all_candidates = top_k <= 0
    limit = len(key_df) if show_all_candidates else top_k
    if matched_brand_values and lock_to_detected_brand:
        best_brand_score = max(matched_brand_values.values())
        matched_brand_names = {
            brand
            for brand, score in matched_brand_values.items()
            if score >= best_brand_score - 0.0005
        }
        candidate_indexes = [
            idx
            for idx, row in key_df.iterrows()
            if str(row[brand_col]) in matched_brand_names
        ]
        if best_brand_score < 1.0 and candidate_indexes:
            best_candidate_score = max(float(best_scores[idx]) for idx in candidate_indexes)
            effective_threshold = max(0.05, best_candidate_score - 0.05)
        ranked_indexes = sorted(candidate_indexes, key=lambda idx: best_scores[idx], reverse=True)[:limit]
    else:
        ranked_indexes = np.argsort(best_scores)[::-1][:limit]

    matches: List[KeyMessageMatch] = []
    seen = set()

    for idx in ranked_indexes:
        score = float(best_scores[idx])
        if not show_all_candidates and score < effective_threshold:
            continue
        row = key_df.iloc[int(idx)]
        record_key = str(row.get("__record_key", row[id_col]))
        if record_key in seen:
            continue
        seen.add(record_key)
        key_message_id = str(row[id_col])
        evidence_description = ""
        if description_of_image_video is None and use_evidence_description:
            evidence_description = _evidence_description(text, str(row[brand_col]), str(row[msg_col]))
        matches.append(
            KeyMessageMatch(
                key_message_id=key_message_id,
                brand_product=str(row[brand_col]),
                key_message=str(row[msg_col]),
                key_words=_display_keywords(
                    str(row[brand_col]),
                    str(row[msg_col]),
                    text,
                    str(row.get("__generated_keywords", "")),
                    use_model=use_keyword_model,
                ),
                confidence_score=round(score, 4),
                source_location=source_location,
                description_of_image_video=description_of_image_video or evidence_description,
            )
        )
    return matches
