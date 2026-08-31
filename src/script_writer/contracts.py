from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


SCRIPT_INTELLIGENCE_SCHEMA_VERSION = "1.0.0"
OUTCOME_SCHEMA_VERSION = "1.0.0"
GENERATION_CONTRACT_VERSION = "1.0.0"


class EvidenceType(StrEnum):
    OBSERVED = "observed"
    DETERMINISTIC = "deterministic_derivation"
    HEURISTIC = "heuristic_inference"
    MODEL = "model_inference"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SourceRef:
    path: str
    start_seconds: float | None = None
    end_seconds: float | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"path": self.path}
        if self.start_seconds is not None:
            result["start_seconds"] = round(float(self.start_seconds), 4)
        if self.end_seconds is not None:
            result["end_seconds"] = round(float(self.end_seconds), 4)
        if self.note:
            result["note"] = self.note
        return result


@dataclass(frozen=True)
class Evidence:
    value: Any
    evidence_type: EvidenceType
    confidence: float | None = None
    method: str | None = None
    sources: tuple[SourceRef, ...] = field(default_factory=tuple)
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.evidence_type is EvidenceType.UNKNOWN and self.value is not None:
            raise ValueError("unknown evidence must have a null value")
        if self.evidence_type is not EvidenceType.UNKNOWN and self.value is None:
            raise ValueError("known evidence must have a value")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "value": self.value,
            "evidence_type": self.evidence_type.value,
            "sources": [source.to_dict() for source in self.sources],
        }
        if self.confidence is not None:
            result["confidence"] = round(float(self.confidence), 4)
        if self.method:
            result["method"] = self.method
        if self.reason:
            result["reason"] = self.reason
        return result


def observed(
    value: Any,
    path: str,
    *,
    confidence: float | None = None,
    start: float | None = None,
    end: float | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    return Evidence(
        value,
        EvidenceType.OBSERVED,
        confidence=confidence,
        method=method,
        sources=(SourceRef(path, start, end),),
    ).to_dict()


def derived(
    value: Any,
    sources: list[SourceRef],
    method: str,
) -> dict[str, Any]:
    return Evidence(
        value,
        EvidenceType.DETERMINISTIC,
        confidence=1.0,
        method=method,
        sources=tuple(sources),
    ).to_dict()


def heuristic(
    value: Any,
    sources: list[SourceRef],
    method: str,
    confidence: float,
) -> dict[str, Any]:
    return Evidence(
        value,
        EvidenceType.HEURISTIC,
        confidence=confidence,
        method=method,
        sources=tuple(sources),
    ).to_dict()


def model_inference(
    value: Any,
    sources: list[SourceRef],
    method: str,
    confidence: float,
) -> dict[str, Any]:
    return Evidence(
        value,
        EvidenceType.MODEL,
        confidence=confidence,
        method=method,
        sources=tuple(sources),
    ).to_dict()


def unknown(reason: str) -> dict[str, Any]:
    return Evidence(None, EvidenceType.UNKNOWN, reason=reason).to_dict()


def validate_evidence(node: Any, path: str = "$", *, recursive: bool = True) -> None:
    """Validate evidence envelopes without requiring a JSON Schema dependency."""
    if isinstance(node, dict):
        if "evidence_type" in node:
            required = {"value", "evidence_type", "sources"}
            missing = required - node.keys()
            if missing:
                raise ValueError(f"{path} missing evidence fields: {sorted(missing)}")
            try:
                kind = EvidenceType(node["evidence_type"])
            except ValueError as exc:
                raise ValueError(f"{path}.evidence_type is invalid") from exc
            if kind is EvidenceType.UNKNOWN and node["value"] is not None:
                raise ValueError(f"{path} unknown evidence must have null value")
            if kind is not EvidenceType.UNKNOWN and node["value"] is None:
                raise ValueError(f"{path} known evidence cannot have null value")
            confidence = node.get("confidence")
            if confidence is not None and not (
                isinstance(confidence, (int, float)) and 0 <= confidence <= 1
            ):
                raise ValueError(f"{path}.confidence must be between 0 and 1")
            if not isinstance(node["sources"], list):
                raise ValueError(f"{path}.sources must be an array")
        if recursive:
            for key, value in node.items():
                validate_evidence(value, f"{path}.{key}", recursive=True)
    elif isinstance(node, list) and recursive:
        for index, value in enumerate(node):
            validate_evidence(value, f"{path}[{index}]", recursive=True)


def validate_script_intelligence_record(record: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "record_id",
        "compiler",
        "identity",
        "content",
        "script_structure",
        "hook_intelligence",
        "retention_devices",
        "linguistic_characteristics",
        "storytelling",
        "persuasion",
        "information_density",
        "delivery",
        "script_edit_relationships",
        "quality",
        "index_projections",
    }
    missing = required - record.keys()
    if missing:
        raise ValueError(f"record missing fields: {sorted(missing)}")
    if record["schema_version"] != SCRIPT_INTELLIGENCE_SCHEMA_VERSION:
        raise ValueError("unsupported ScriptIntelligenceRecord schema version")
    if not isinstance(record["record_id"], str) or not record["record_id"]:
        raise ValueError("record_id must be a non-empty string")
    validate_evidence(record)


def validate_generation_request(request: dict[str, Any]) -> None:
    required = {"contract_version", "client_context_id", "topic", "objective", "audience"}
    missing = required - request.keys()
    if missing:
        raise ValueError(f"generation request missing fields: {sorted(missing)}")
    if request["contract_version"] != GENERATION_CONTRACT_VERSION:
        raise ValueError("unsupported generation contract version")


def validate_generation_result(result: dict[str, Any]) -> None:
    required = {
        "contract_version",
        "spoken_script",
        "hook",
        "sections",
        "on_screen_text_suggestions",
        "delivery_notes",
        "visual_cues",
        "cta",
        "claims_requiring_verification",
        "creative_mechanisms",
        "retrieved_evidence",
        "rationale",
    }
    missing = required - result.keys()
    if missing:
        raise ValueError(f"generation result missing fields: {sorted(missing)}")
    if result["contract_version"] != GENERATION_CONTRACT_VERSION:
        raise ValueError("unsupported generation contract version")
