from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from .contracts import OUTCOME_SCHEMA_VERSION


COUNT_METRICS = {
    "views", "reach", "plays", "likes", "comments", "shares", "saves",
    "total_interactions", "follows", "profile_visits",
}
NONNEGATIVE_METRICS = {
    "total_watch_time_seconds", "average_watch_time_seconds",
}
RATE_METRICS = {
    "completion_rate", "three_second_view_rate", "skip_rate",
}


class OutcomeValidationError(ValueError):
    pass


def _parse_timestamp(value: Any, field: str) -> datetime:
    if not isinstance(value, str):
        raise OutcomeValidationError(f"{field} must be an ISO-8601 timestamp")
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise OutcomeValidationError(f"{field} must be an ISO-8601 timestamp") from exc


def validate_outcome_record(record: dict[str, Any]) -> dict[str, Any]:
    required = {
        "schema_version", "report_id", "platform", "account_context_id",
        "published_at", "measured_at", "measurement_window_hours", "metrics", "rights",
    }
    missing = required - record.keys()
    if missing:
        raise OutcomeValidationError(f"outcome missing fields: {sorted(missing)}")
    if record["schema_version"] != OUTCOME_SCHEMA_VERSION:
        raise OutcomeValidationError("unsupported outcome schema version")
    for field in ("report_id", "platform", "account_context_id"):
        if not isinstance(record[field], str) or not record[field].strip():
            raise OutcomeValidationError(f"{field} must be a non-empty string")
    published = _parse_timestamp(record["published_at"], "published_at")
    measured = _parse_timestamp(record["measured_at"], "measured_at")
    if measured < published:
        raise OutcomeValidationError("measured_at cannot precede published_at")
    window = record["measurement_window_hours"]
    if not isinstance(window, (int, float)) or window <= 0:
        raise OutcomeValidationError("measurement_window_hours must be positive")
    metrics = record["metrics"]
    if not isinstance(metrics, dict):
        raise OutcomeValidationError("metrics must be an object")
    for name in COUNT_METRICS:
        value = metrics.get(name)
        if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 0):
            raise OutcomeValidationError(f"metrics.{name} must be a nonnegative integer")
    for name in NONNEGATIVE_METRICS:
        value = metrics.get(name)
        if value is not None and (
            not isinstance(value, (int, float)) or isinstance(value, bool) or value < 0
        ):
            raise OutcomeValidationError(f"metrics.{name} must be nonnegative")
    for name in RATE_METRICS:
        value = metrics.get(name)
        if value is not None and (
            not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1
        ):
            raise OutcomeValidationError(f"metrics.{name} must be between 0 and 1")
    rights = record["rights"]
    if not isinstance(rights, dict) or not isinstance(rights.get("training_allowed"), bool):
        raise OutcomeValidationError("rights.training_allowed must be boolean")
    if not isinstance(rights.get("basis"), str) or not rights["basis"].strip():
        raise OutcomeValidationError("rights.basis must be a non-empty string")
    normalization = record.get("cohort_normalization")
    if normalization is not None:
        if not isinstance(normalization, dict):
            raise OutcomeValidationError("cohort_normalization must be an object or null")
        if not normalization.get("cohort_key") or not normalization.get("method"):
            raise OutcomeValidationError("normalized outcomes require cohort_key and method")
        if not isinstance(normalization.get("cohort_size"), int) or normalization["cohort_size"] <= 0:
            raise OutcomeValidationError("cohort_size must be a positive integer")
    return record


def canonical_outcome_json(record: dict[str, Any]) -> str:
    validate_outcome_record(record)
    return json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
