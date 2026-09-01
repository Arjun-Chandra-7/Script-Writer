from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from .contracts import EvidenceType, observed, unknown, validate_evidence


CLIENT_TRAINING_CONTEXT_VERSION = "1.0.0"
TRAINING_EXAMPLE_SCHEMA_VERSION = "1.0.0"
TRAINING_COMPILER_VERSION = "1.0.0"
TRAINING_MANIFEST_VERSION = "1.0.0"
TRAINING_READINESS_VERSION = "1.0.0"


class TrainingObjective(StrEnum):
    FULL_SCRIPT = "full_script_sft"
    HOOK = "hook_generation"
    CONTINUATION = "continuation"
    STRUCTURE = "structure_planning"
    SECTION = "section_generation"
    CTA = "cta_generation"
    STYLE = "style_conditioned"


class Eligibility(StrEnum):
    ELIGIBLE = "eligible"
    WARNING = "eligible_with_warning"
    INELIGIBLE = "ineligible"


SCRIPT_RELEVANT_CLIENT_FIELDS: dict[str, tuple[str, ...]] = {
    "identity_positioning": ("identity_positioning", "positioning", "identity"),
    "niche": ("niche",),
    "subniche": ("subniche", "sub_niche"),
    "audience": ("audience", "target_audience"),
    "content_pillars": ("content_pillars", "pillars"),
    "brand_voice": ("brand_voice", "voice"),
    "tone": ("tone",),
    "products_offers": ("products_offers", "products", "offers"),
    "beliefs": ("beliefs",),
    "vocabulary": ("vocabulary", "vocabulary_preferences"),
    "cta_rules": ("cta_rules", "cta_preferences"),
    "factual_constraints": ("factual_constraints", "claim_constraints"),
    "prohibited_topics_claims": ("prohibited_topics_claims", "prohibited_claims"),
    "recurring_narratives": ("recurring_narratives",),
    "differentiation": ("differentiation",),
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def content_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


@dataclass(frozen=True)
class ClientTrainingContext:
    projection: dict[str, Any]
    canonical_json: str
    sha256: str

    @classmethod
    def from_document(cls, document: dict[str, Any], *, source_path: str) -> "ClientTrainingContext":
        if not isinstance(document, dict):
            raise ValueError("client context root must be an object")
        client_id = document.get("client_id") or document.get("id")
        if not isinstance(client_id, str) or not client_id.strip():
            raise ValueError("client.json requires a non-empty client_id or id")
        source_hash = content_sha256(document)
        fields: dict[str, Any] = {}
        for output_name, aliases in SCRIPT_RELEVANT_CLIENT_FIELDS.items():
            match = next((name for name in aliases if document.get(name) not in (None, "", [], {})), None)
            fields[output_name] = (
                observed(document[match], f"{source_path}$.{match}")
                if match
                else unknown(f"{output_name} is absent from client context")
            )
        payload = {
            "schema_version": CLIENT_TRAINING_CONTEXT_VERSION,
            "context_id": f"clientctx:{client_id.strip()}:{source_hash[:16]}",
            "client_id": client_id.strip(),
            "source": {"path": source_path, "sha256": source_hash},
            "fields": fields,
        }
        validate_client_training_context(payload)
        rendered = canonical_json(payload)
        return cls(payload, rendered, hashlib.sha256(rendered.encode()).hexdigest())

    @classmethod
    def from_path(cls, path: Path) -> "ClientTrainingContext":
        return cls.from_document(json.loads(path.read_text()), source_path=str(path))


def validate_client_training_context(context: dict[str, Any]) -> None:
    required = {"schema_version", "context_id", "client_id", "source", "fields"}
    missing = required - context.keys()
    if missing:
        raise ValueError(f"client training context missing fields: {sorted(missing)}")
    if context["schema_version"] != CLIENT_TRAINING_CONTEXT_VERSION:
        raise ValueError("unsupported ClientTrainingContext version")
    if not isinstance(context["fields"], dict):
        raise ValueError("client training fields must be an object")
    validate_evidence(context["fields"])


def validate_training_example(example: dict[str, Any]) -> None:
    required = {
        "schema_version", "example_id", "identity", "client_context_ref",
        "training_input", "creative_plan", "target_output", "provenance",
        "quality", "review",
    }
    missing = required - example.keys()
    if missing:
        raise ValueError(f"training example missing fields: {sorted(missing)}")
    if example["schema_version"] != TRAINING_EXAMPLE_SCHEMA_VERSION:
        raise ValueError("unsupported ScriptTrainingExample version")
    objective = example["identity"].get("dataset_objective")
    try:
        TrainingObjective(objective)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid training objective") from exc
    if example["quality"].get("eligibility") not in {item.value for item in Eligibility}:
        raise ValueError("invalid eligibility")
    validate_evidence(example["training_input"])
    validate_evidence(example["creative_plan"])


def evidence_value(node: Any) -> Any:
    return node.get("value") if isinstance(node, dict) and "evidence_type" in node else None


def known_evidence(node: Any) -> bool:
    return (
        isinstance(node, dict)
        and node.get("evidence_type") in {kind.value for kind in EvidenceType if kind is not EvidenceType.UNKNOWN}
        and node.get("value") is not None
    )
