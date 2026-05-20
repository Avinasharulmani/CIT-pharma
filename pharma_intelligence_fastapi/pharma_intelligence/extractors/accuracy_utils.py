from datetime import datetime, timezone
from typing import Any

from ..utils import clean_text
from ..extraction_accuracy import word_error_rate


def character_error_rate(expected: str, actual: str) -> float:
    expected_clean = clean_text(expected)
    actual_clean = clean_text(actual)
    if not expected_clean and not actual_clean:
        return 0.0
    if not expected_clean:
        return 100.0
    previous = list(range(len(actual_clean) + 1))
    for index, expected_char in enumerate(expected_clean, start=1):
        current = [index]
        for other_index, actual_char in enumerate(actual_clean, start=1):
            insert_cost = current[other_index - 1] + 1
            delete_cost = previous[other_index] + 1
            replace_cost = previous[other_index - 1] + (expected_char != actual_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return round((previous[-1] / len(expected_clean)) * 100, 2)


def build_accuracy_report(
    *,
    file_name: str,
    file_type: str,
    units: list[dict[str, Any]],
    ground_truth_units: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not ground_truth_units:
        return {
            "file_name": file_name,
            "file_type": file_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "accuracy_available": False,
            "message": "Real accuracy cannot be calculated without ground truth. Returning extraction quality only.",
        }
    unit_scores = []
    for index, unit in enumerate(units):
        expected = ""
        if index < len(ground_truth_units):
            expected = clean_text(str(ground_truth_units[index].get("text") or ground_truth_units[index].get("expected_text") or ""))
        actual = clean_text(str(unit.get("cleaned_text") or unit.get("text") or unit.get("raw_text") or ""))
        unit_scores.append(
            {
                "unit_number": unit.get("unit_number") or unit.get("number") or index + 1,
                "cer": character_error_rate(expected, actual),
                "wer": word_error_rate(expected, actual),
                "expected_length": len(expected),
                "actual_length": len(actual),
            }
        )
    return {
        "file_name": file_name,
        "file_type": file_type,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "accuracy_available": True,
        "unit_level_scores": unit_scores,
        "overall_cer": round(sum(score["cer"] for score in unit_scores) / max(len(unit_scores), 1), 2),
        "overall_wer": round(sum(score["wer"] for score in unit_scores) / max(len(unit_scores), 1), 2),
        "extraction_method_summary": {},
    }
