from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from .domain import ValidationResult


class ReportValidationError(ValueError):
    """A permanent report/schema problem suitable for quarantine."""


def _required_mapping(parent: dict[str, Any], key: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ReportValidationError(f"{key} must be an object")
    return value


def _required_text(parent: dict[str, Any], key: str, *, where: str = "report") -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ReportValidationError(f"{where}.{key} must be a non-empty string")
    return value.strip()


def _finite_positive(value: Any, field: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ReportValidationError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ReportValidationError(f"{field} must be finite and positive")
    return result


def assign_split(group_key: str, salt: str) -> str:
    """Assign 90/5/5 train/validation/test deterministically by source group."""
    digest = hashlib.blake2b(
        group_key.encode("utf-8"), key=salt.encode("utf-8"), digest_size=8
    ).digest()
    bucket = int.from_bytes(digest, "big") % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def validate_report(raw: bytes, *, split_salt: str) -> ValidationResult:
    try:
        decoded = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReportValidationError("report must be UTF-8 JSON") from exc
    try:
        report = json.loads(decoded)
    except json.JSONDecodeError as exc:
        raise ReportValidationError(f"invalid JSON: {exc.msg}") from exc
    if not isinstance(report, dict):
        raise ReportValidationError("report root must be an object")

    report_id = _required_text(report, "report_id")
    source = _required_mapping(report, "source")
    content_hash = _required_text(source, "content_hash", where="source")
    duration = _finite_positive(source.get("duration_seconds"), "source.duration_seconds")

    processing = _required_mapping(report, "processing")
    if processing.get("status") != "complete":
        raise ReportValidationError("processing.status must be complete")
    extractor_version = _required_text(
        processing, "extractor_version", where="processing"
    )

    transcript = _required_mapping(report, "transcript")
    if transcript.get("status") != "complete":
        raise ReportValidationError("transcript.status must be complete")
    full_text = _required_text(transcript, "full_text", where="transcript")
    if len(full_text.split()) < 3:
        raise ReportValidationError("transcript.full_text is implausibly short")

    training_features = report.get("training_features")
    if training_features is not None and not isinstance(training_features, dict):
        raise ReportValidationError("training_features must be an object when present")

    # This projection deliberately excludes raw word/frame arrays and unverified
    # semantic labels. The immutable raw report remains available for reprocessing.
    projection = {
        "schema_version": "1.0",
        "report_id": report_id,
        "group_key": content_hash,
        "source": {
            "content_hash": content_hash,
            "duration_seconds": duration,
            "fps": source.get("fps"),
            "resolution": source.get("resolution"),
        },
        "extractor": {
            "version": extractor_version,
            "confidence_policy": report.get("confidence"),
        },
        "transcript": {
            "text": full_text,
            "word_count": transcript.get("delivery", {}).get("word_count"),
            "words_per_minute": transcript.get("delivery", {}).get(
                "overall_words_per_minute"
            ),
        },
        "verified_features": (training_features or {}).get("values", {}),
        "feature_provenance": (training_features or {}).get("provenance", {}),
        "excluded_features": (training_features or {}).get("excluded", {}),
        "outcome": None,
        "rights": None,
    }
    canonical = json.dumps(projection, sort_keys=True, separators=(",", ":"))
    transcript_hash = hashlib.sha256(full_text.encode("utf-8")).hexdigest()
    return ValidationResult(
        report_id=report_id,
        source_content_hash=content_hash,
        group_key=content_hash,
        split=assign_split(content_hash, split_salt),
        quality_status="observation_only",
        extractor_version=extractor_version,
        transcript_sha256=transcript_hash,
        canonical_json=canonical,
    )
