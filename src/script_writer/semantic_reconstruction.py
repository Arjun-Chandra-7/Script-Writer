from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from .contracts import SourceRef, model_inference, unknown, validate_evidence
from .training_contracts import canonical_json, evidence_value
from .training_leakage import leakage_metrics


MINIMUM_SUFFICIENT_BRIEF_VERSION = "1.0.0"
SEMANTIC_PROMPT_VERSION = "intent-reconstruction-v1"
SEMANTIC_RECONSTRUCTION_VERSION = "semantic-intent-reconstruction-v1"
CORE_FIELDS = (
    "topic", "central_idea", "content_objective", "content_format",
    "target_audience", "audience_problem_desire", "target_duration_seconds", "language",
)
OPTIONAL_FIELDS = (
    "subtopic", "desired_outcome", "tone", "sophistication_level", "perspective",
    "hook_intent", "progression", "cta_intent", "required_concepts",
    "prohibited_concepts", "factual_context", "stylistic_constraints",
)


class SemanticInferenceError(RuntimeError):
    pass


class SemanticIntentAdapter(Protocol):
    version: str
    prompt_version: str

    def infer(self, request: dict[str, Any]) -> dict[str, Any]: ...


def _brief_template(intelligence: dict[str, Any], client: dict[str, Any]) -> dict[str, Any]:
    content = intelligence["content"]
    return {
        "schema_version": MINIMUM_SUFFICIENT_BRIEF_VERSION,
        "brief_id": "",
        "adapter": {},
        "topic": unknown("semantic inference did not provide a topic"),
        "central_idea": unknown("semantic inference did not provide a central idea"),
        "content_objective": unknown("semantic inference did not provide an objective"),
        "content_format": unknown("semantic inference did not provide a format"),
        "target_audience": unknown("audience is not inferable from source evidence"),
        "audience_problem_desire": unknown("problem/desire is not inferable from source evidence"),
        "target_duration_seconds": content["video_duration_seconds"],
        "language": content.get("language", unknown("language unavailable")),
        "subtopic": unknown("not inferred"), "desired_outcome": unknown("not inferred"),
        "tone": client.get("fields", {}).get("tone", unknown("client tone unavailable")),
        "sophistication_level": content.get("vocabulary", unknown("not measured")),
        "perspective": intelligence["linguistic_characteristics"].get("person_usage", unknown("not measured")),
        "hook_intent": unknown("not inferred"), "progression": unknown("not inferred"),
        "cta_intent": unknown("no verified CTA"), "required_concepts": unknown("not inferred"),
        "prohibited_concepts": client.get("fields", {}).get("prohibited_topics_claims", unknown("none supplied")),
        "factual_context": unknown("not inferred"), "stylistic_constraints": unknown("not inferred"),
    }


def build_semantic_input(intelligence: dict[str, Any], client: dict[str, Any], variant: str = "full") -> dict[str, Any]:
    """Canonical, versioned input shared by every semantic adapter experiment."""
    if variant not in {"transcript", "structure", "delivery", "full", "full_with_client"}:
        raise ValueError("unsupported semantic input variant")
    transcript = str(evidence_value(intelligence["content"]["clean_transcript"]) or "")
    value = {
        "semantic_input_version": "1.0.0",
        "variant": variant,
        "transcript": transcript,
    }
    if variant in {"structure", "delivery", "full", "full_with_client"}:
        value.update({
            "hook_mechanisms": [x.get("mechanism") for x in intelligence["hook_intelligence"].get("mechanisms", [])],
            "beat_roles": [x.get("role") for x in intelligence["script_structure"].get("major_beats", [])],
        })
    if variant in {"delivery", "full", "full_with_client"}:
        value["delivery"] = intelligence.get("delivery", {})
    if variant in {"full", "full_with_client"}:
        value["script_relevant_evidence"] = {
            "persuasion": intelligence.get("persuasion", {}),
            "information_density": intelligence.get("information_density", {}),
            "linguistic_characteristics": intelligence.get("linguistic_characteristics", {}),
        }
    if variant == "full_with_client":
        value["client_context"] = {key: field for key, field in client.get("fields", {}).items() if key in {"niche", "audience", "tone", "content_pillars", "factual_constraints", "prohibited_topics_claims"}}
    return value


def build_prompt_request(intelligence: dict[str, Any], client: dict[str, Any], stage: str, *, input_variant: str = "full") -> dict[str, Any]:
    source = build_semantic_input(intelligence, client, input_variant)
    return {
        "schema_version": SEMANTIC_PROMPT_VERSION,
        "semantic_input_version": source["semantic_input_version"],
        "stage": stage,
        "instructions": [
            "Infer minimum sufficient conditioning, never the literal creator prompt.",
            "Evidence outranks completeness. Unknown is valid.",
            "Do not copy target wording, paraphrase every sentence, infer performance, demographics, business intent, or CTA without evidence.",
            "Client context is a weak disambiguation prior, not source truth.",
            "Return concise field values with transcript or canonical paths supporting each inference.",
        ],
        **source,
    }


def validate_adapter_fields(value: dict[str, Any]) -> None:
    if not isinstance(value, dict):
        raise SemanticInferenceError("adapter response must be an object")
    for name, node in value.items():
        if name not in {*CORE_FIELDS, *OPTIONAL_FIELDS}:
            raise SemanticInferenceError(f"adapter returned unsupported field: {name}")
        if not isinstance(node, dict) or "value" not in node or "confidence" not in node:
            raise SemanticInferenceError(f"adapter field {name} lacks value/confidence")
        if node["value"] is not None and not isinstance(node["value"], (str, list, dict, int, float)):
            raise SemanticInferenceError(f"adapter field {name} has invalid value")
        if not isinstance(node["confidence"], (int, float)) or not 0 <= node["confidence"] <= 1:
            raise SemanticInferenceError(f"adapter field {name} has invalid confidence")
        if not isinstance(node.get("evidence_paths", []), list) or not all(isinstance(x, str) for x in node.get("evidence_paths", [])):
            raise SemanticInferenceError(f"adapter field {name} has invalid evidence paths")


@dataclass(frozen=True)
class RuleBasedSemanticIntentAdapter:
    """Conservative local baseline; its semantic quality must be gold-evaluated."""
    version: str = "rule-semantic-intent-v1"
    prompt_version: str = SEMANTIC_PROMPT_VERSION

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        text = request["transcript"].lower()
        stage = request["stage"]
        result: dict[str, Any] = {}
        def item(value: Any, confidence: float) -> dict[str, Any]:
            return {"value": value, "confidence": confidence, "evidence_paths": ["$.content.clean_transcript"]}
        if "doge" in text and "government workers" in text:
            if stage in {"topic_central_idea", "single_pass"}:
                result.update({
                    "topic": item("DOGE government workforce cuts", 0.72),
                    "central_idea": item("The speaker argues that arbitrary workforce cuts harm public workers.", 0.68),
                })
            if stage in {"objective_format", "single_pass"}:
                result.update({
                    "content_objective": item("persuade", 0.63),
                    "content_format": item("commentary", 0.78),
                })
            if stage in {"conditional_context", "single_pass"}:
                result.update({
                    "required_concepts": item(["DOGE", "government workers", "consequences"], 0.75),
                    "hook_intent": item("open a critical discussion through a question", 0.58),
                })
        elif stage in {"objective_format", "single_pass"} and ("how to" in text or "step" in text):
            result.update({"content_objective": item("educate", 0.62), "content_format": item("tutorial", 0.62)})
        return result


@dataclass(frozen=True)
class MockSemanticIntentAdapter:
    fields: dict[str, Any]
    version: str = "mock-semantic-intent-v1"
    prompt_version: str = SEMANTIC_PROMPT_VERSION

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.fields


@dataclass(frozen=True)
class HttpStructuredSemanticIntentAdapter:
    """Provider-neutral transport boundary; no network call occurs until infer()."""
    endpoint: str
    api_key: str
    model: str
    transport: Callable[[dict[str, Any]], dict[str, Any]]
    temperature: float = 0.0
    timeout_seconds: int = 45
    max_tokens: int = 1200
    version: str = "http-structured-semantic-intent-v1"
    prompt_version: str = SEMANTIC_PROMPT_VERSION

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        if not self.api_key:
            raise SemanticInferenceError("semantic API credential is not configured")
        payload = {"model": self.model, "temperature": self.temperature, "max_tokens": self.max_tokens, "request": request}
        response = self.transport(payload)
        fields = response.get("fields") if isinstance(response, dict) else None
        validate_adapter_fields(fields)
        return fields


class SemanticInferenceCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS semantic_results (request_hash TEXT PRIMARY KEY, payload_json TEXT NOT NULL, cost_json TEXT NOT NULL)")
        self.connection.execute("CREATE TABLE IF NOT EXISTS semantic_failures (request_hash TEXT PRIMARY KEY, error TEXT NOT NULL, attempts INTEGER NOT NULL)")
        self.connection.commit()

    def get(self, key: str) -> dict[str, Any] | None:
        row = self.connection.execute("SELECT payload_json FROM semantic_results WHERE request_hash=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, value: dict[str, Any], cost: dict[str, Any]) -> None:
        self.connection.execute("INSERT OR REPLACE INTO semantic_results VALUES(?,?,?)", (key, canonical_json(value), canonical_json(cost)))
        self.connection.commit()

    def fail(self, key: str, error: str) -> None:
        self.connection.execute("INSERT INTO semantic_failures VALUES(?,?,1) ON CONFLICT(request_hash) DO UPDATE SET error=excluded.error, attempts=attempts+1", (key, error[:2000]))
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


class SemanticReconstructionService:
    def __init__(self, adapter: SemanticIntentAdapter, cache: SemanticInferenceCache | None = None, retries: int = 2):
        self.adapter, self.cache, self.retries = adapter, cache, retries

    def reconstruct(self, intelligence: dict[str, Any], client: dict[str, Any], *, mode: str = "staged", input_variant: str = "full_with_client") -> dict[str, Any]:
        if mode not in {"staged", "single_pass"}:
            raise ValueError("mode must be staged or single_pass")
        merged: dict[str, Any] = {}
        cache_hits = 0
        stages = ("topic_central_idea", "objective_format", "conditional_context") if mode == "staged" else ("single_pass",)
        for stage in stages:
            request = build_prompt_request(intelligence, client, stage, input_variant=input_variant)
            key = hashlib.sha256((canonical_json(request) + self.adapter.version + self.adapter.prompt_version).encode()).hexdigest()
            response = self.cache.get(key) if self.cache else None
            if response is not None:
                cache_hits += 1
            else:
                last: Exception | None = None
                response = None
                for _ in range(self.retries + 1):
                    try:
                        candidate = self.adapter.infer(request)
                        validate_adapter_fields(candidate)
                        response = candidate
                        break
                    except Exception as exc:
                        last = exc
                if response is None:
                    if self.cache: self.cache.fail(key, str(last))
                    raise SemanticInferenceError(f"semantic stage {stage} failed closed: {last}")
                if self.cache: self.cache.put(key, response, {"estimated_input_tokens": len(request["transcript"].split()), "estimated_output_tokens": len(canonical_json(response).split())})
            merged.update(response)
        brief = _brief_template(intelligence, client)
        for field, node in merged.items():
            if node["value"] is None:
                continue
            text = str(evidence_value(intelligence["content"]["clean_transcript"]) or "")
            leakage = leakage_metrics({field: node["value"]}, text)
            compact = len(str(node["value"]).split()) <= 32 if field == "central_idea" else True
            if leakage["rejected"] or not compact:
                brief[field] = unknown("semantic inference rejected by field-level leakage/compression control")
                continue
            brief[field] = model_inference(node["value"], [SourceRef(path) for path in node.get("evidence_paths", [])], f"{self.adapter.version}:{self.adapter.prompt_version}", float(node["confidence"]))
        source = str(evidence_value(intelligence["identity"]["source_content_hash"]) or intelligence["record_id"])
        brief["brief_id"] = f"msb:{hashlib.sha256((source + self.adapter.version).encode()).hexdigest()[:24]}"
        brief["adapter"] = {"version": self.adapter.version, "prompt_version": self.adapter.prompt_version, "stages": len(stages), "mode": mode, "semantic_input_version": "1.0.0", "input_variant": input_variant, "cache_hits": cache_hits}
        validate_minimum_sufficient_brief(brief)
        return brief


def validate_minimum_sufficient_brief(brief: dict[str, Any]) -> None:
    required = {"schema_version", "brief_id", "adapter", *CORE_FIELDS, *OPTIONAL_FIELDS}
    missing = required - brief.keys()
    if missing: raise ValueError(f"brief missing fields: {sorted(missing)}")
    if brief["schema_version"] != MINIMUM_SUFFICIENT_BRIEF_VERSION: raise ValueError("unsupported brief version")
    validate_evidence({key: brief[key] for key in {*CORE_FIELDS, *OPTIONAL_FIELDS}})


def field_leakage_report(brief: dict[str, Any], target: str) -> dict[str, Any]:
    return {field: leakage_metrics({field: evidence_value(brief[field])}, target) for field in (*CORE_FIELDS, *OPTIONAL_FIELDS) if evidence_value(brief[field]) is not None}


def estimate_corpus(records: int, average_words: int, *, adapter: SemanticIntentAdapter, input_price_per_million: float = 0.0, output_price_per_million: float = 0.0, requests_per_record: int = 3, concurrency: int = 4) -> dict[str, Any]:
    input_tokens = records * average_words * requests_per_record
    output_tokens = records * 180
    return {"records": records, "requests": records * requests_per_record, "estimated_input_tokens": input_tokens, "estimated_output_tokens": output_tokens, "estimated_cost": round(input_tokens / 1_000_000 * input_price_per_million + output_tokens / 1_000_000 * output_price_per_million, 4), "adapter_version": adapter.version, "concurrency": concurrency, "quality_note": "estimate only; no request is sent"}
