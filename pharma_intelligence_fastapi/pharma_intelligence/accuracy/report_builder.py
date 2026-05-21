from __future__ import annotations

from typing import Optional

from .model_tracker import (
    get_model_accuracy_average,
    get_model_accuracy_average_all,
    save_accuracy_record,
)


def build_report(
    file_name: str,
    file_type: str,
    model_used: str,
    accuracy: Optional[float],
    is_measured_accuracy: bool,
    method: str,
    warnings: list,
    db,
    job_id: str,
) -> dict:
    if accuracy is not None:
        save_accuracy_record(db, model_used, file_type, accuracy, is_measured_accuracy, job_id)
    model_acc = get_model_accuracy_average(db, model_used)
    if model_acc is None:
        model_acc = get_model_accuracy_average_all(db, model_used)
    return {
        "file_name": file_name,
        "file_type": file_type,
        "model_used": model_used,
        "accuracy": accuracy,
        "model_accuracy": model_acc,
        "is_measured_accuracy": bool(is_measured_accuracy),
        "method": method,
        "warnings": list(warnings or []),
    }
