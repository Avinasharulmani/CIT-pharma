from pathlib import Path
from threading import Lock
from typing import Tuple
from datetime import datetime

import pandas as pd

from .keywords import extract_keywords
from .mongo_database import key_messages_collection


def _normalized_column_name(col: str) -> str:
    return str(col).lower().strip().replace("_", "").replace("/", "").replace(" ", "")


def _find_column(df: pd.DataFrame, options, fallback_index: int) -> str:
    normalized = {_normalized_column_name(col): col for col in df.columns}
    for option in options:
        key = _normalized_column_name(option)
        if key in normalized:
            return normalized[key]
    if len(df.columns) > fallback_index:
        return df.columns[fallback_index]
    raise ValueError(f"Could not find required column. Available columns: {list(df.columns)}")


DB_ID_COL = "Key Message ID"
DB_BRAND_COL = "Brand/Product"
DB_MSG_COL = "Key Message"
DB_KEYWORDS_COL = "Key Words"
DB_RECORD_KEY_COL = "__record_key"

_KEY_MESSAGE_CACHE_LOCK = Lock()
_KEY_MESSAGE_CACHE: dict[tuple[str, int, int], pd.DataFrame] = {}


def _file_signature(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return (str(path.resolve()), stat.st_mtime_ns, stat.st_size)


def _read_key_message_file(path: Path) -> Tuple[pd.DataFrame, str, str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Key message file not found: {path}")

    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, engine="openpyxl")
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Key message file must be .xlsx, .xls, or .csv")

    df.columns = [str(c).strip() for c in df.columns]

    id_col = _find_column(
        df,
        ["Key Message ID", "Key_Message_ID", "Message ID", "KM ID", "ID"],
        0,
    )
    brand_col = _find_column(
        df,
        ["Brand/Product", "Brand Product", "Brand", "Product", "Brand / Product"],
        1,
    )
    msg_col = _find_column(
        df,
        ["Key Message", "KeyMessage", "Message", "Topic Key Message"],
        2,
    )

    df = df.fillna("")
    return df, id_col, brand_col, msg_col


def _replace_key_messages(df: pd.DataFrame, id_col: str, brand_col: str, msg_col: str, source_file: str) -> int:
    records = []
    now = datetime.utcnow()
    for row_index, row in df.iterrows():
        key_message_id = str(row[id_col]).strip()
        if not key_message_id:
            continue

        brand_product = str(row[brand_col]).strip()
        key_message = str(row[msg_col]).strip()
        key_words = str(row.get(DB_KEYWORDS_COL, "")).strip()
        if not key_words:
            key_words = ", ".join(extract_keywords(key_message, top_n=6, use_model=False))
        record_key = f"{key_message_id}::{row_index}"

        records.append(
            {
                "record_key": record_key,
                "key_message_id": key_message_id,
                "brand_product": brand_product,
                "key_message": key_message,
                "key_words": key_words,
                "source_file": source_file,
                "created_at": now,
                "updated_at": now,
            }
        )

    key_messages_collection.delete_many({})
    if records:
        key_messages_collection.insert_many(records, ordered=False)
    return len(records)


def _load_key_messages_from_db() -> pd.DataFrame:
    records = key_messages_collection.find({}, {"_id": 0}).sort("key_message_id", 1)
    rows = [
        {
            DB_RECORD_KEY_COL: record.get("record_key", record.get("key_message_id", "")),
            DB_ID_COL: record.get("key_message_id", ""),
            DB_BRAND_COL: record.get("brand_product", ""),
            DB_MSG_COL: record.get("key_message", ""),
            DB_KEYWORDS_COL: record.get("key_words", ""),
        }
        for record in records
    ]

    if not rows:
        raise ValueError("No key messages found in the database. Upload a key-message Excel/CSV file first.")

    return pd.DataFrame(rows).fillna("")


def load_key_messages(path: Path) -> Tuple[pd.DataFrame, str, str, str]:
    signature = _file_signature(path)
    with _KEY_MESSAGE_CACHE_LOCK:
        cached = _KEY_MESSAGE_CACHE.get(signature)
    if cached is not None:
        return cached.copy(deep=False), DB_ID_COL, DB_BRAND_COL, DB_MSG_COL

    df, id_col, brand_col, msg_col = _read_key_message_file(path)
    _replace_key_messages(df, id_col, brand_col, msg_col, source_file=path.name)

    db_df = _load_key_messages_from_db()
    if DB_RECORD_KEY_COL not in db_df.columns:
        db_df[DB_RECORD_KEY_COL] = db_df[DB_ID_COL].astype(str)
    db_df["__combined_text"] = db_df[DB_BRAND_COL].astype(str) + " " + db_df[DB_MSG_COL].astype(str)
    db_df["__generated_keywords"] = db_df[DB_KEYWORDS_COL].astype(str)
    with _KEY_MESSAGE_CACHE_LOCK:
        _KEY_MESSAGE_CACHE.clear()
        _KEY_MESSAGE_CACHE[signature] = db_df
    return db_df.copy(deep=False), DB_ID_COL, DB_BRAND_COL, DB_MSG_COL
