from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .training_contracts import evidence_value


def validate_annotation(annotation: dict[str, Any]) -> None:
    if not isinstance(annotation.get("record_id"), str): raise ValueError("annotation requires record_id")
    if not isinstance(annotation.get("fields"), dict): raise ValueError("annotation requires fields")
    for field, value in annotation["fields"].items():
        if not isinstance(value, dict) or value.get("status") not in {"value", "unknown", "ambiguous", "not_inferable"}:
            raise ValueError(f"invalid annotation for {field}")
        if value["status"] == "value" and not isinstance(value.get("acceptable_values"), list):
            raise ValueError(f"value annotation {field} needs acceptable_values")


def evaluate_gold(briefs: dict[str, dict[str, Any]], annotations: list[dict[str, Any]]) -> dict[str, Any]:
    by_field: dict[str, Counter[str]] = defaultdict(Counter)
    errors: list[dict[str, Any]] = []
    for annotation in annotations:
        validate_annotation(annotation)
        brief = briefs.get(annotation["record_id"])
        if not brief: continue
        for field, label in annotation["fields"].items():
            predicted = evidence_value(brief.get(field, {}))
            status = label["status"]
            if status in {"unknown", "not_inferable", "ambiguous"}:
                outcome = "correct_abstention" if predicted is None else "false_inference"
            else:
                acceptable = {str(x).strip().lower() for x in label.get("acceptable_values", [])}
                outcome = "accepted" if str(predicted).strip().lower() in acceptable else "wrong"
            by_field[field][outcome] += 1
            if outcome in {"wrong", "false_inference"}:
                errors.append({"record_id": annotation["record_id"], "field": field, "category": label.get("error_category", "model_failed_to_abstain" if outcome == "false_inference" else "wrong_field"), "predicted": predicted})
    result = {}
    for field, counts in by_field.items():
        total = sum(counts.values())
        result[field] = {**counts, "total": total, "accept_rate": round(counts["accepted"] / total, 4) if total else None, "false_inference_rate": round(counts["false_inference"] / total, 4) if total else None, "correct_abstention_rate": round(counts["correct_abstention"] / total, 4) if total else None}
    return {"fields": result, "error_count": len(errors), "errors": errors, "human_acceptance_available": bool(annotations)}


def reviewer_agreement(annotations: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in annotations: grouped[item["record_id"]].append(item["fields"].get(field, {"status":"not_inferable"}))
    comparable = [values for values in grouped.values() if len(values) > 1]
    matches = sum(len({(v.get("status"), tuple(v.get("acceptable_values", []))) for v in values}) == 1 for values in comparable)
    return {"field": field, "multi_review_records": len(comparable), "exact_agreement_rate": round(matches / len(comparable), 4) if comparable else None}
