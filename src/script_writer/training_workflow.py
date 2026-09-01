from __future__ import annotations

import hashlib
import json
import copy
from pathlib import Path
from typing import Any

from .training_contracts import ClientTrainingContext, canonical_json, evidence_value
from .training_cache import TrainingCompilationCache
from .training_dataset import (
    CorpusCompilation,
    CorpusTrainingCompiler,
    ReviewStore,
    SamplingPolicy,
    TrainingDatasetBuilder,
    _atomic_write_once,
    dataset_audit,
    training_readiness_report,
)


def _write_content_addressed(directory: Path, stem: str, suffix: str, payload: bytes) -> Path:
    digest = hashlib.sha256(payload).hexdigest()[:16]
    path = directory / f"{stem}-{digest}.{suffix}"
    _atomic_write_once(path, payload)
    return path


def build_training_artifacts(
    intelligence_paths: list[Path],
    client_path: Path,
    output_dir: Path,
    *,
    split_salt: str,
    sampling_policy: SamplingPolicy = SamplingPolicy(),
    minimum_reviewed_examples: int = 25,
    minimum_exported_examples: int = 100,
) -> dict[str, Any]:
    if not intelligence_paths:
        raise ValueError("at least one ScriptIntelligenceRecord is required")
    records = [json.loads(path.read_text()) for path in sorted(intelligence_paths)]
    client = ClientTrainingContext.from_path(client_path)
    decisions = ReviewStore(output_dir / "reviews.json").decisions()
    cache = TrainingCompilationCache(output_dir / ".training-cache" / "compilations.sqlite3")
    compiler = CorpusTrainingCompiler(split_salt=split_salt, cache=cache)
    first = compiler.compile(records, client.projection)
    second = compiler.compile(records, client.projection)
    cache.close()
    deterministic = (
        first.examples == second.examples
        and first.rejections == second.rejections
        and first.source_groups == second.source_groups
    )
    reviewed_examples = []
    reviewed_rejections = list(first.rejections)
    for example in first.examples:
        updated = copy.deepcopy(example)
        decision = decisions.get(example["example_id"])
        if decision:
            updated["review"] = {"status": "reviewed", **decision}
            if decision["decision"] in {"reject", "flag"}:
                reason = "human_rejected" if decision["decision"] == "reject" else "human_review_flagged"
                updated["quality"]["eligibility"] = "ineligible"
                updated["quality"]["exclusion_reasons"] = sorted(
                    set([*updated["quality"]["exclusion_reasons"], reason])
                )
                reviewed_rejections.append(
                    {"example_id": updated["example_id"], "objective": updated["identity"]["dataset_objective"], "reasons": [reason]}
                )
        reviewed_examples.append(updated)
    compilation = CorpusCompilation(
        tuple(reviewed_examples), tuple(reviewed_rejections), first.source_groups,
        first.exact_duplicate_count, first.near_duplicate_cluster_count,
        first.cache_hits, first.cache_misses,
    )
    builder = TrainingDatasetBuilder(output_dir / "datasets", sampling_policy)
    build = builder.build(compilation, client.projection)
    audit = dataset_audit(compilation, build)
    readiness = training_readiness_report(
        compilation,
        build,
        audit,
        reviewed_examples=sum(example_id in decisions for example_id in {item["example_id"] for item in compilation.examples}),
        minimum_reviewed_examples=minimum_reviewed_examples,
        minimum_exported_examples=minimum_exported_examples,
        deterministic_regeneration_verified=deterministic,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    client_bytes = (json.dumps(client.projection, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode()
    client_output = _write_content_addressed(output_dir, "client-training-context", "json", client_bytes)
    examples_bytes = ("\n".join(canonical_json(item) for item in compilation.examples) + "\n").encode()
    examples_output = _write_content_addressed(output_dir, "compiled-examples", "jsonl", examples_bytes)
    source_lines = []
    for record in records:
        source_lines.append(
            canonical_json(
                {
                    "record_id": record["record_id"],
                    "report_id": evidence_value(record["identity"]["report_id"]),
                    "source_content_hash": evidence_value(record["identity"]["source_content_hash"]),
                    "source_transcript": evidence_value(record["content"]["clean_transcript"]),
                }
            )
        )
    sources_output = _write_content_addressed(
        output_dir, "review-sources", "jsonl", ("\n".join(source_lines) + "\n").encode()
    )
    audit_output = _write_content_addressed(
        output_dir, "dataset-audit", "json", (json.dumps(audit, indent=2, sort_keys=True) + "\n").encode()
    )
    readiness_output = _write_content_addressed(
        output_dir, "training-readiness", "json", (json.dumps(readiness, indent=2, sort_keys=True) + "\n").encode()
    )
    return {
        "client_context": str(client_output),
        "compiled_examples": str(examples_output),
        "review_sources": str(sources_output),
        "dataset_audit": str(audit_output),
        "training_readiness": str(readiness_output),
        "manifests": build["manifests"],
        "summary": audit,
        "readiness": readiness,
    }


def inspect_example(directory: Path, example_id: str) -> dict[str, Any]:
    examples: dict[str, Any] | None = None
    for path in sorted(directory.glob("compiled-examples-*.jsonl"), reverse=True):
        for line in path.read_text().splitlines():
            item = json.loads(line)
            if item["example_id"] == example_id:
                examples = item
                break
        if examples:
            break
    if examples is None:
        raise KeyError(example_id)
    transcript = None
    source_hash = examples["identity"]["source_content_hash"]
    for path in sorted(directory.glob("review-sources-*.jsonl"), reverse=True):
        for line in path.read_text().splitlines():
            source = json.loads(line)
            if source["source_content_hash"] == source_hash:
                transcript = source["source_transcript"]
                break
        if transcript is not None:
            break
    client = None
    client_paths = sorted(directory.glob("client-training-context-*.json"), reverse=True)
    if client_paths:
        client = json.loads(client_paths[0].read_text())
    return {
        "example_id": example_id,
        "source_transcript": transcript,
        "client_context_projection": client,
        "reconstructed_brief": examples["training_input"]["content_brief"],
        "creative_plan": examples["creative_plan"],
        "target": examples["target_output"],
        "provenance": examples["provenance"],
        "leakage": examples["quality"]["leakage"],
        "eligibility": examples["quality"]["eligibility"],
        "exclusion_reasons": examples["quality"]["exclusion_reasons"],
        "review": examples["review"],
    }
