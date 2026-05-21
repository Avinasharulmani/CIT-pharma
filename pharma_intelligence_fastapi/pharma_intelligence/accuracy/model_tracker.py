from __future__ import annotations

from datetime import datetime
from statistics import mean
from typing import Optional


def save_accuracy_record(db, model_name: str, file_type: str, accuracy: float, is_measured: bool, job_id: str) -> None:
    db["model_accuracy_history"].insert_one(
        {
            "job_id": job_id,
            "model_name": model_name,
            "file_type": file_type,
            "accuracy": float(accuracy),
            "is_measured": bool(is_measured),
            "timestamp": datetime.utcnow().isoformat(),
        }
    )


def get_model_accuracy_average(db, model_name: str) -> Optional[float]:
    records = list(
        db["model_accuracy_history"].find(
            {"model_name": model_name, "is_measured": True},
            {"accuracy": 1, "_id": 0},
        )
    )
    values = [float(record["accuracy"]) for record in records if record.get("accuracy") is not None]
    return round(mean(values), 2) if values else None


def get_model_accuracy_average_all(db, model_name: str) -> Optional[float]:
    records = list(
        db["model_accuracy_history"].find(
            {"model_name": model_name},
            {"accuracy": 1, "_id": 0},
        )
    )
    values = [float(record["accuracy"]) for record in records if record.get("accuracy") is not None]
    return round(mean(values), 2) if values else None
