from __future__ import annotations

import hashlib
import json
import os
import tempfile
import copy
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .dataset_design import (
    LeakageGuard,
    LeakageIdentity,
    assign_universal_training_split,
    simhash64,
)
from .text_analysis import normalize_transcript
from .training_compiler import TrainingExampleCompiler
from .training_contracts import (
    TRAINING_COMPILER_VERSION,
    TRAINING_MANIFEST_VERSION,
    TRAINING_READINESS_VERSION,
    TrainingObjective,
    canonical_json,
    evidence_value,
    validate_training_example,
)


class OutcomeAwareSamplingProvider(Protocol):
    """Future interface; no implementation or outcome weighting exists today."""

    version: str

    def weight(self, example: dict[str, Any], outcome: dict[str, Any]) -> float: ...


@dataclass(frozen=True)
class SamplingPolicy:
    max_examples_per_cluster_per_objective: int | None = None
    max_examples_per_source_per_objective: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_examples_per_cluster_per_objective": self.max_examples_per_cluster_per_objective,
            "max_examples_per_source_per_objective": self.max_examples_per_source_per_objective,
            "automatic_rebalancing": False,
        }


@dataclass(frozen=True)
class CorpusCompilation:
    examples: tuple[dict[str, Any], ...]
    rejections: tuple[dict[str, Any], ...]
    source_groups: dict[str, str]
    exact_duplicate_count: int
    near_duplicate_cluster_count: int
    cache_hits: int = 0
    cache_misses: int = 0


class CorpusTrainingCompiler:
    def __init__(self, *, split_salt: str, near_duplicate_distance: int = 3, cache: Any = None):
        self.split_salt = split_salt
        self.guard = LeakageGuard(near_duplicate_distance)
        self.example_compiler = TrainingExampleCompiler()
        self.cache = cache

    def compile(
        self, records: list[dict[str, Any]], client_context: dict[str, Any]
    ) -> CorpusCompilation:
        identities: list[LeakageIdentity] = []
        usable: list[dict[str, Any]] = []
        early_rejections: list[dict[str, Any]] = []
        seen_sources: set[str] = set()
        seen_transcripts: set[str] = set()
        exact_duplicate_count = 0
        for record in records:
            transcript = str(evidence_value(record.get("content", {}).get("clean_transcript")) or "")
            source_hash = str(evidence_value(record.get("identity", {}).get("source_content_hash")) or "")
            if not transcript or not source_hash:
                early_rejections.append({"record_id": record.get("record_id"), "reasons": ["missing_script_or_source_identity"]})
                continue
            normalized = normalize_transcript(transcript)
            transcript_sha = hashlib.sha256(normalized.encode()).hexdigest()
            if source_hash in seen_sources or transcript_sha in seen_transcripts:
                exact_duplicate_count += 1
                early_rejections.append(
                    {"record_id": record.get("record_id"), "reasons": ["exact_duplicate_suppressed"]}
                )
                continue
            seen_sources.add(source_hash)
            seen_transcripts.add(transcript_sha)
            identities.append(
                LeakageIdentity(
                    source_content_hash=source_hash,
                    transcript_sha256=transcript_sha,
                    transcript_simhash64=simhash64(normalized),
                )
            )
            usable.append(record)
        groups = self.guard.cluster(identities)
        cluster_sizes = Counter(groups.values())
        near_duplicate_cluster_count = sum(1 for count in cluster_sizes.values() if count > 1)
        examples: list[dict[str, Any]] = []
        rejections = list(early_rejections)
        cache_hits = 0
        cache_misses = 0
        for record, identity in zip(usable, identities):
            group_id = groups[identity.source_content_hash]
            split = assign_universal_training_split(group_id, salt=self.split_salt)
            cache_key = hashlib.sha256(
                (
                    canonical_json(record)
                    + canonical_json(client_context)
                    + TRAINING_COMPILER_VERSION
                    + self.example_compiler.reconstructor.version
                ).encode()
            ).hexdigest()
            result = self.cache.get(cache_key) if self.cache is not None else None
            if result is None:
                cache_misses += 1
                result = self.example_compiler.compile(
                    record, client_context, group_id=group_id, split=split
                )
                if self.cache is not None:
                    self.cache.put(cache_key, identity.source_content_hash, result)
            else:
                cache_hits += 1
            patched_examples = []
            for example in result.examples:
                patched = copy.deepcopy(example)
                patched["identity"]["source_group_id"] = group_id
                patched["identity"]["split"] = split
                patched_examples.append(patched)
            examples.extend(patched_examples)
            rejections.extend(result.rejections)
        self.guard.validate_no_leakage(
            [
                {
                    "split": example["identity"]["split"],
                    "source_content_hash": example["identity"]["source_content_hash"],
                    "near_duplicate_group": example["identity"]["source_group_id"],
                }
                for example in examples
            ]
        )
        return CorpusCompilation(
            tuple(examples), tuple(rejections), groups,
            exact_duplicate_count, near_duplicate_cluster_count,
            cache_hits, cache_misses,
        )


def _atomic_write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise OSError(f"immutable dataset collision at {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}-", suffix=".part", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


class TrainingDatasetBuilder:
    def __init__(self, output_dir: Path, policy: SamplingPolicy = SamplingPolicy()):
        self.output_dir = output_dir
        self.policy = policy

    def build(
        self,
        compilation: CorpusCompilation,
        client_context: dict[str, Any],
    ) -> dict[str, Any]:
        accepted = [
            item for item in compilation.examples
            if item["quality"]["eligibility"] != "ineligible"
        ]
        selected, sampling_exclusions = self._sample(accepted)
        manifests: dict[str, Any] = {}
        for objective in TrainingObjective:
            members = sorted(
                [item for item in selected if item["identity"]["dataset_objective"] == objective.value],
                key=lambda item: item["example_id"],
            )
            if not members:
                continue
            for member in members:
                validate_training_example(member)
            lines = [canonical_json(member) for member in members]
            jsonl = ("\n".join(lines) + "\n").encode()
            payload_sha = hashlib.sha256(jsonl).hexdigest()
            dataset_id = f"tds:{objective.value}:{payload_sha[:16]}"
            data_path = self.output_dir / f"{objective.value}-{payload_sha[:16]}.jsonl"
            _atomic_write_once(data_path, jsonl)
            split_counts = Counter(item["identity"]["split"] for item in members)
            manifest = {
                "manifest_version": TRAINING_MANIFEST_VERSION,
                "dataset_id": dataset_id,
                "objective": objective.value,
                "compiler_version": TRAINING_COMPILER_VERSION,
                "client_context_id": client_context["context_id"],
                "client_context_sha256": client_context["source"]["sha256"],
                "data_file": data_path.name,
                "data_sha256": payload_sha,
                "example_count": len(members),
                "split_counts": dict(sorted(split_counts.items())),
                "source_group_count": len({item["identity"]["source_group_id"] for item in members}),
                "sampling_policy": self.policy.to_dict(),
                "sampling_exclusion_count": sum(
                    rejection["objective"] == objective.value for rejection in sampling_exclusions
                ),
                "performance_signal_used": False,
                "training_execution_enabled": False,
            }
            manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            manifest_path = self.output_dir / f"manifest-{objective.value}-{payload_sha[:16]}.json"
            _atomic_write_once(manifest_path, manifest_bytes)
            manifests[objective.value] = {**manifest, "manifest_path": str(manifest_path)}
        rejected = sorted(
            [*compilation.rejections, *sampling_exclusions],
            key=lambda item: (str(item.get("objective")), str(item.get("example_id", item.get("record_id")))),
        )
        rejection_bytes = ("\n".join(canonical_json(item) for item in rejected) + ("\n" if rejected else "")).encode()
        rejection_path = self.output_dir / f"rejections-{hashlib.sha256(rejection_bytes).hexdigest()[:16]}.jsonl"
        _atomic_write_once(rejection_path, rejection_bytes)
        return {"manifests": manifests, "rejection_path": str(rejection_path), "selected_examples": selected, "sampling_exclusions": sampling_exclusions}

    def _sample(self, examples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ranked = sorted(
            examples,
            key=lambda item: (
                -float(item["quality"]["training_evidence_quality"]["value"]),
                item["example_id"],
            ),
        )
        selected: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        cluster_counts: Counter[tuple[str, str]] = Counter()
        source_counts: Counter[tuple[str, str]] = Counter()
        for item in ranked:
            identity = item["identity"]
            objective = identity["dataset_objective"]
            cluster_key = (identity["source_group_id"], objective)
            source_key = (identity["source_content_hash"], objective)
            reason = None
            if self.policy.max_examples_per_cluster_per_objective is not None and cluster_counts[cluster_key] >= self.policy.max_examples_per_cluster_per_objective:
                reason = "sampling_cluster_cap"
            if self.policy.max_examples_per_source_per_objective is not None and source_counts[source_key] >= self.policy.max_examples_per_source_per_objective:
                reason = "sampling_source_cap"
            if reason:
                rejected.append({"example_id": item["example_id"], "objective": objective, "reasons": [reason]})
                continue
            selected.append(item)
            cluster_counts[cluster_key] += 1
            source_counts[source_key] += 1
        return sorted(selected, key=lambda item: item["example_id"]), rejected


def dataset_audit(compilation: CorpusCompilation, build: dict[str, Any]) -> dict[str, Any]:
    examples = list(compilation.examples)
    selected = list(build["selected_examples"])
    source_ids = {item["identity"]["source_content_hash"] for item in examples}
    eligible_sources = {
        item["identity"]["source_content_hash"]
        for item in examples if item["quality"]["eligibility"] != "ineligible"
    }
    def distribution(values: list[Any]) -> dict[str, int]:
        return dict(sorted(Counter(str(value) for value in values).items()))
    hooks: list[str] = []
    structures: list[str] = []
    unknowns: Counter[str] = Counter()
    provenance: Counter[str] = Counter()
    source_examples: dict[str, dict[str, Any]] = {}
    for item in examples:
        source_examples.setdefault(item["identity"]["source_content_hash"], item)
    for item in source_examples.values():
        plan = item["creative_plan"]
        hooks.extend(evidence_value(plan.get("hook_mechanisms")) or [])
        roles = evidence_value(plan.get("progression")) or []
        if roles:
            structures.append(" -> ".join(roles))
        provenance.update(item["provenance"]["reconstruction_evidence_types"])
        for name, node in item["training_input"]["content_brief"].items():
            if isinstance(node, dict) and node.get("evidence_type") == "unknown":
                unknowns[name] += 1
    def bucket(value: Any, boundaries: tuple[float, ...], labels: tuple[str, ...]) -> str:
        if not isinstance(value, (int, float)):
            return "unknown"
        for boundary, label in zip(boundaries, labels):
            if float(value) < boundary:
                return label
        return labels[-1]
    attributes = [item["identity"]["source_attributes"] for item in source_examples.values()]
    objectives = distribution([item["identity"]["dataset_objective"] for item in selected])
    splits = distribution([item["identity"]["split"] for item in selected])
    leakage_failures = sum(item["quality"]["leakage"]["severity"] == "high" for item in examples)
    warnings: list[str] = []
    for name, count in objectives.items():
        if selected and count / len(selected) > 0.7:
            warnings.append(f"objective imbalance: {name} is {count}/{len(selected)} examples")
    return {
        "audit_version": "1.0.0",
        "total_source_videos": len(source_ids),
        "eligible_sources": len(eligible_sources),
        "rejected_sources": len(source_ids - eligible_sources),
        "total_compiled_examples": len(examples),
        "total_training_examples": len(selected),
        "eligibility": distribution([item["quality"]["eligibility"] for item in examples]),
        "examples_per_objective": objectives,
        "split_sizes": splits,
        "leakage_failures": leakage_failures,
        "exact_duplicate_count": compilation.exact_duplicate_count,
        "near_duplicate_cluster_count": compilation.near_duplicate_cluster_count,
        "hook_distribution": distribution(hooks),
        "structure_distribution": distribution(structures),
        "language_distribution": distribution([item.get("language") or "unknown" for item in attributes]),
        "topic_distribution": distribution([item.get("topic") or "unknown" for item in attributes]),
        "format_distribution": distribution([item.get("content_format") or "unknown" for item in attributes]),
        "duration_distribution_seconds": distribution([
            bucket(item.get("duration_seconds"), (15, 30, 60, float("inf")), ("<15", "15-29", "30-59", "60+"))
            for item in attributes
        ]),
        "word_count_distribution": distribution([
            bucket(item.get("word_count"), (50, 100, 200, float("inf")), ("<50", "50-99", "100-199", "200+"))
            for item in attributes
        ]),
        "speaking_rate_distribution_wps": distribution([
            bucket(item.get("words_per_second"), (2, 3, 4, float("inf")), ("<2", "2-2.99", "3-3.99", "4+"))
            for item in attributes
        ]),
        "cta_source_count": len({
            item["identity"]["source_content_hash"] for item in examples
            if item["identity"]["dataset_objective"] == TrainingObjective.CTA.value
        }),
        "mean_brief_reconstruction_completeness": round(
            sum(item["quality"]["training_evidence_quality"]["components"]["brief_completeness"] for item in source_examples.values())
            / len(source_examples), 4
        ) if source_examples else 0.0,
        "mean_training_evidence_quality": round(
            sum(item["quality"]["training_evidence_quality"]["value"] for item in examples) / len(examples), 4
        ) if examples else 0.0,
        "unknown_fields": dict(sorted(unknowns.items())),
        "provenance_mix": dict(sorted(provenance.items())),
        "rejection_count": len(compilation.rejections) + len(build["sampling_exclusions"]),
        "rejection_reasons": distribution([
            reason for item in [*compilation.rejections, *build["sampling_exclusions"]] for reason in item.get("reasons", [])
        ]),
        "warnings": warnings,
        "cache": {"hits": compilation.cache_hits, "misses": compilation.cache_misses},
    }


def training_readiness_report(
    compilation: CorpusCompilation,
    build: dict[str, Any],
    audit: dict[str, Any],
    *,
    reviewed_examples: int = 0,
    minimum_reviewed_examples: int = 25,
    minimum_exported_examples: int = 100,
    deterministic_regeneration_verified: bool = False,
) -> dict[str, Any]:
    examples = list(compilation.examples)
    selected = list(build["selected_examples"])
    group_splits: dict[str, set[str]] = defaultdict(set)
    source_splits: dict[str, set[str]] = defaultdict(set)
    for item in examples:
        group_splits[item["identity"]["source_group_id"]].add(item["identity"]["split"])
        source_splits[item["identity"]["source_content_hash"]].add(item["identity"]["split"])
    gates = {
        "all_examples_schema_valid": all(_valid(item) for item in examples),
        "zero_cross_split_source_leakage": all(len(splits) == 1 for splits in source_splits.values()),
        "zero_cross_split_duplicate_leakage": all(len(splits) == 1 for splits in group_splits.values()),
        "zero_high_severity_target_leakage_in_export": all(item["quality"]["leakage"]["severity"] != "high" for item in selected),
        "immutable_objective_manifests_generated": bool(build["manifests"]),
        "compiler_version_frozen": all(item["identity"]["compiler_version"] == TRAINING_COMPILER_VERSION for item in examples),
        "client_context_version_frozen": len({item["client_context_ref"]["context_id"] for item in examples}) <= 1 and bool(examples),
        "rejection_reasons_recorded": all(item.get("reasons") for item in compilation.rejections),
        "deterministic_regeneration_verified": deterministic_regeneration_verified,
        "validation_and_test_membership_frozen": bool(build["manifests"]),
        "sufficient_human_inspection": reviewed_examples >= minimum_reviewed_examples,
        "sufficient_eligible_examples": len(selected) >= minimum_exported_examples,
        "validation_and_test_sets_present": (
            any(item["identity"]["split"] == "validation" for item in selected)
            and any(item["identity"]["split"] == "test" for item in selected)
        ),
    }
    failed = sorted(name for name, passed in gates.items() if not passed)
    return {
        "schema_version": TRAINING_READINESS_VERSION,
        "status": "training_ready" if not failed else "not_training_ready",
        "gates": gates,
        "failed_gates": failed,
        "review": {"reviewed_examples": reviewed_examples, "minimum_required": minimum_reviewed_examples},
        "minimum_exported_examples": minimum_exported_examples,
        "dataset_summary": {
            "source_count": audit["total_source_videos"],
            "exported_example_count": len(selected),
            "manifest_count": len(build["manifests"]),
        },
        "training_execution_enabled": False,
    }


def _valid(example: dict[str, Any]) -> bool:
    try:
        validate_training_example(example)
    except ValueError:
        return False
    return True


class ReviewStore:
    def __init__(self, path: Path):
        self.path = path

    def decisions(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        value = json.loads(self.path.read_text())
        return value if isinstance(value, dict) else {}

    def record(self, example_id: str, decision: str, *, note: str = "") -> dict[str, Any]:
        if decision not in {"accept", "reject", "flag"}:
            raise ValueError("decision must be accept, reject, or flag")
        decisions = self.decisions()
        decisions[example_id] = {"decision": decision, "note": note}
        payload = (json.dumps(decisions, indent=2, sort_keys=True) + "\n").encode()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(prefix=f".{self.path.name}-", suffix=".part", dir=self.path.parent)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, self.path)
        return decisions[example_id]
