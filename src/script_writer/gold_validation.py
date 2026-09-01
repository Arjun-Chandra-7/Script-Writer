"""Human-gold semantic validation, deliberately separate from training data."""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from .contracts import unknown
from .dataset_design import hamming_distance, simhash64
from .semantic_reconstruction import (
    SemanticReconstructionService,
    build_semantic_input,
    field_leakage_report,
)
from .text_analysis import normalize_transcript
from .training_contracts import canonical_json, evidence_value


GOLD_SELECTION_VERSION = "1.0.0"
GOLD_ANNOTATION_VERSION = "1.0.0"
GOLD_MANIFEST_VERSION = "1.0.0"
SEMANTIC_INPUT_VERSION = "1.0.0"

CORE_FIELDS = (
    "topic",
    "central_idea",
    "content_objective",
    "content_format",
    "target_audience",
    "audience_problem_desire",
    "cta_intent",
)
OPTIONAL_FIELDS = (
    "subtopic",
    "desired_outcome",
    "tone",
    "sophistication_level",
    "perspective",
    "hook_intent",
    "progression",
    "required_concepts",
    "prohibited_concepts",
    "factual_context",
    "stylistic_constraints",
)
ALL_FIELDS = (*CORE_FIELDS, *OPTIONAL_FIELDS)
CONTROLLED_FIELDS = {"content_objective", "content_format", "cta_intent"}

ANNOTATION_STATUSES = {
    "value",
    "accepted",
    "partial",
    "wrong",
    "too_broad",
    "too_narrow",
    "too_vague",
    "too_detailed",
    "leakage",
    "unknown",
    "ambiguous",
    "not_inferable",
    "reject",
}


def semantic_quality_gate_report(
    benchmark: dict[str, Any],
    *,
    reviewed_sources: int,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Apply frozen proposed gates; absence of a measured field or synthetic fixture evaluation is a failed gate."""
    thresholds = config.get("thresholds", {})
    fields = benchmark.get("metrics", {}).get("fields", {})
    leakage = benchmark.get("leakage", {})

    def metric(field: str, name: str) -> float | None:
        value = fields.get(field, {}).get(name)
        return float(value) if isinstance(value, (int, float)) else None

    topic_acc = metric("topic", "acceptance_rate")
    central_acc = metric("central_idea", "acceptance_rate")
    obj_acc = metric("content_objective", "acceptance_rate")
    fmt_acc = metric("content_format", "acceptance_rate")
    audience_unsupported = metric("target_audience", "unsupported_inference_rate")
    cta_unsupported = metric("cta_intent", "unsupported_inference_rate")
    high_risk_leakage = leakage.get("high_risk_count")

    min_sources_threshold = thresholds.get("minimum_human_reviewed_sources", 100)
    topic_threshold = thresholds.get("topic_acceptance", 0.90)
    central_threshold = thresholds.get("central_idea_acceptance", 0.85)
    obj_threshold = thresholds.get("objective_acceptance", 0.80)
    fmt_threshold = thresholds.get("format_acceptance", 0.80)
    audience_threshold = thresholds.get("unsupported_audience_inference", 0.05)
    cta_threshold = thresholds.get("false_cta_inference", 0.02)

    is_fixture = benchmark.get("metrics", {}).get("fixture_only", False) or benchmark.get("is_fixture", False)

    gates = {
        "minimum_human_reviewed_sources": (reviewed_sources >= min_sources_threshold) and not is_fixture,
        "topic_acceptance": topic_acc is not None and topic_acc >= topic_threshold,
        "central_idea_acceptance": central_acc is not None and central_acc >= central_threshold,
        "objective_acceptance": obj_acc is not None and obj_acc >= obj_threshold,
        "format_acceptance": fmt_acc is not None and fmt_acc >= fmt_threshold,
        "unsupported_audience_inference": (
            audience_unsupported is not None and audience_unsupported <= audience_threshold
        ),
        "false_cta_inference": (
            cta_unsupported is not None and cta_unsupported <= cta_threshold
        ),
        "zero_high_severity_leakage": high_risk_leakage is not None and high_risk_leakage == 0,
    }
    return {
        "schema_version": "1.0.0",
        "config_version": config.get("version", "unknown"),
        "reviewed_sources": reviewed_sources,
        "is_fixture": is_fixture,
        "gates": gates,
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
        "semantic_reconstruction_gold_quality_verified": all(gates.values()),
        "note": config.get("note", ""),
    }


def verified_quality_report(report: dict[str, Any]) -> bool:
    """Fail-closed check ensuring report satisfies schema and all production readiness criteria."""
    if not isinstance(report, dict):
        return False
    if report.get("schema_version") != "1.0.0":
        return False
    if report.get("semantic_reconstruction_gold_quality_verified") is not True:
        return False
    if report.get("failed_gates") != []:
        return False
    if report.get("gates", {}).get("minimum_human_reviewed_sources") is not True:
        return False
    return True


def _value(record: dict[str, Any], path: str) -> Any:
    current: Any = record
    for part in path.split("."):
        current = current.get(part, {}) if isinstance(current, dict) else {}
    return evidence_value(current)


def _bucket(value: float | int | None, edges: tuple[float, ...], labels: tuple[str, ...]) -> str:
    if value is None:
        return "unknown"
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def record_strata(record: dict[str, Any]) -> dict[str, str]:
    transcript = str(_value(record, "content.clean_transcript") or "")
    duration = _value(record, "content.video_duration_seconds")
    words = _value(record, "content.spoken_word_count")
    wps = _value(record, "content.words_per_second")
    mechanisms = record.get("hook_intelligence", {}).get("mechanisms", [])
    hook = str(mechanisms[0].get("mechanism", "none")) if mechanisms else "none"
    beats = record.get("script_structure", {}).get("major_beats", [])
    structure = "+".join(str(item.get("role", "unknown")) for item in beats[:3]) or "unknown"
    transcript_conf = _value(record, "content.transcript_confidence")
    return {
        "duration": _bucket(
            duration if isinstance(duration, (int, float)) else None,
            (20, 45, 75),
            ("short", "medium", "long", "very_long"),
        ),
        "word_count": _bucket(
            words if isinstance(words, (int, float)) else None,
            (60, 140, 250),
            ("very_short", "short", "medium", "long"),
        ),
        "speaking_rate": _bucket(
            wps if isinstance(wps, (int, float)) else None,
            (1.8, 2.8, 3.8),
            ("slow", "typical", "fast", "very_fast"),
        ),
        "hook": hook,
        "structure": structure,
        "transcript_confidence": _bucket(
            transcript_conf if isinstance(transcript_conf, (int, float)) else None,
            (0.6, 0.85),
            ("low", "medium", "high"),
        ),
        "semantic_complexity": _bucket(
            len(transcript.split()),
            (80, 180, 320),
            ("low", "medium", "high", "very_high"),
        ),
        "cta": (
            "present"
            if record.get("persuasion", {}).get("cta_mechanism", {}).get("evidence_type")
            not in {None, "unknown"}
            else "absent_or_unknown"
        ),
    }


def stratified_gold_sample(
    records: Iterable[dict[str, Any]],
    count: int,
    *,
    seed: str = "gold-v1",
) -> dict[str, Any]:
    """Deterministic diversity-first sample; it never implies a random human study."""
    candidates = []
    for record in records:
        source_hash = str(_value(record, "identity.source_content_hash") or "")
        transcript = str(_value(record, "content.clean_transcript") or "")
        if source_hash and transcript:
            candidates.append((record, source_hash, transcript, record_strata(record)))
    candidates.sort(key=lambda item: hashlib.sha256((seed + item[1]).encode()).hexdigest())
    chosen: list[tuple[dict[str, Any], str, str, dict[str, str]]] = []
    represented: Counter[tuple[str, str]] = Counter()
    while candidates and len(chosen) < count:
        def score(item: tuple[dict[str, Any], str, str, dict[str, str]]) -> tuple[int, str]:
            strata = item[3]
            return (sum(represented[(key, value)] for key, value in strata.items()), item[1])
        item = min(candidates, key=score)
        candidates.remove(item)
        chosen.append(item)
        represented.update(item[3].items())
    entries = []
    for record, source_hash, transcript, strata in chosen:
        normalized = normalize_transcript(transcript)
        entries.append({
            "record_id": record["record_id"],
            "source_content_hash": source_hash,
            "transcript_sha256": hashlib.sha256(normalized.encode()).hexdigest(),
            "transcript_simhash64": f"{simhash64(normalized):016x}",
            "strata": strata,
        })
    return {
        "schema_version": GOLD_SELECTION_VERSION,
        "selection_id": f"goldsel:{hashlib.sha256(canonical_json(entries).encode()).hexdigest()[:16]}",
        "seed": seed,
        "requested_count": count,
        "available_records": len(candidates) + len(chosen),
        "entries": entries,
        "evaluation_only": True,
    }


def review_payload(
    record: dict[str, Any],
    *,
    mode: str,
    client: dict[str, Any],
    proposal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if mode not in {"blind", "assisted"}:
        raise ValueError("mode must be blind or assisted")
    transcript = str(_value(record, "content.clean_transcript") or "")
    payload: dict[str, Any] = {
        "record_id": record["record_id"],
        "mode": mode,
        "source": {
            "transcript": transcript,
            "timestamps": record.get("script_structure", {}).get("major_beats", []),
            "hook": record.get("hook_intelligence", {}),
            "structure": record.get("script_structure", {}),
            "client_context": client.get("fields", {}),
        },
        "fields": list(CORE_FIELDS),
        "annotation_instructions": (
            "Use unknown, ambiguous, or not_inferable freely. Do not infer private intent, "
            "demographics, or business goals without transcript/creative evidence."
        ),
    }
    if mode == "assisted":
        payload["proposal"] = proposal or {}
    return payload


def validate_gold_annotation(annotation: dict[str, Any]) -> None:
    if annotation.get("schema_version") != GOLD_ANNOTATION_VERSION:
        raise ValueError("unsupported gold annotation version")
    if not isinstance(annotation.get("record_id"), str) or not isinstance(annotation.get("reviewer_id"), str):
        raise ValueError("gold annotation requires record_id and reviewer_id")
    if annotation.get("mode") not in {"blind", "assisted"}:
        raise ValueError("invalid review mode")
    if not isinstance(annotation.get("fields"), dict):
        raise ValueError("annotation fields must be an object")
    for name, value in annotation["fields"].items():
        if name not in ALL_FIELDS and not isinstance(name, str):
            raise ValueError(f"invalid annotation field {name}")
        if not isinstance(value, dict) or value.get("status") not in ANNOTATION_STATUSES:
            raise ValueError(f"invalid annotation field {name}: unsupported status {value.get("status")}")
        if value["status"] == "value" and not isinstance(value.get("acceptable_values"), list) and "human_value" not in value:
            raise ValueError(f"{name} value status requires acceptable_values or human_value")
        if "confidence" in value and value["confidence"] is not None:
            if not isinstance(value["confidence"], (int, float, str)):
                raise ValueError(f"invalid confidence in {name}")
            if isinstance(value["confidence"], (int, float)) and not (0 <= value["confidence"] <= 1):
                raise ValueError(f"confidence in {name} must be in [0, 1]")


def new_annotation(
    record_id: str,
    reviewer_id: str,
    mode: str,
    fields: dict[str, Any],
    *,
    timestamp: str | None = None,
) -> dict[str, Any]:
    value = {
        "schema_version": GOLD_ANNOTATION_VERSION,
        "record_id": record_id,
        "reviewer_id": reviewer_id,
        "mode": mode,
        "fields": fields,
        "created_at": timestamp or datetime.now(timezone.utc).isoformat(),
    }
    validate_gold_annotation(value)
    return value


def multi_reviewer_agreement(
    annotations: list[dict[str, Any]],
    fields: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Calculate inter-annotator agreement metrics across multiple reviewers on the same records."""
    target_fields = tuple(fields) if fields else CORE_FIELDS
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in annotations:
        validate_gold_annotation(item)
        grouped[item["record_id"]].append(item)

    results: dict[str, Any] = {}
    for field in target_fields:
        comparable_records = 0
        exact_agreements = 0
        inferability_agreements = 0
        disagreements = []

        for record_id, records in grouped.items():
            field_annotations = [
                (rec["reviewer_id"], rec["fields"][field])
                for rec in records
                if field in rec.get("fields", {})
            ]
            if len(field_annotations) < 2:
                continue
            comparable_records += 1

            inferable_flags = [
                fa[1]["status"] in {"value", "accepted"}
                for fa in field_annotations
            ]
            inferability_match = len(set(inferable_flags)) == 1
            if inferability_match:
                inferability_agreements += 1

            canonical_signatures = []
            for reviewer_id, f_dict in field_annotations:
                status = f_dict["status"]
                values = tuple(sorted(str(v).strip().lower() for v in f_dict.get("acceptable_values", [])))
                canonical_signatures.append((status, values))

            if len(set(canonical_signatures)) == 1:
                exact_agreements += 1
            else:
                disagreements.append({
                    "record_id": record_id,
                    "reviewers": [fa[0] for fa in field_annotations],
                    "values": [fa[1] for fa in field_annotations],
                })

        results[field] = {
            "multi_reviewed_records": comparable_records,
            "exact_agreement_count": exact_agreements,
            "exact_agreement_rate": (
                round(exact_agreements / comparable_records, 4) if comparable_records else None
            ),
            "inferability_agreement_rate": (
                round(inferability_agreements / comparable_records, 4) if comparable_records else None
            ),
            "disagreement_count": len(disagreements),
        }

    return {
        "fields": results,
        "total_records_multi_reviewed": len([r for r, recs in grouped.items() if len(recs) > 1]),
    }


def adjudicate(
    record_id: str,
    field: str,
    resolver_id: str,
    resolved: dict[str, Any],
    annotations: list[dict[str, Any]],
    note: str = "",
) -> dict[str, Any]:
    if field not in ALL_FIELDS and not isinstance(field, str):
        raise ValueError("unsupported gold field")
    validate_gold_annotation(new_annotation(record_id, resolver_id, "blind", {field: resolved}))
    source = [a for a in annotations if a.get("record_id") == record_id and field in a.get("fields", {})]
    return {
        "schema_version": "1.0.0",
        "adjudication_version": "1.0.0",
        "record_id": record_id,
        "field": field,
        "resolver_id": resolver_id,
        "resolved": resolved,
        "source_reviewers": sorted(set(a["reviewer_id"] for a in source)),
        "reviewer_count": len(source),
        "note": note,
        "adjudicated_at": datetime.now(timezone.utc).isoformat(),
    }


def resolved_gold(
    annotations: list[dict[str, Any]],
    adjudications: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    resolved: dict[tuple[str, str], dict[str, Any]] = {}
    for item in adjudications:
        resolved[(item["record_id"], item["field"])] = item["resolved"]
    grouped: dict[str, dict[str, Any]] = defaultdict(dict)
    for annotation in annotations:
        validate_gold_annotation(annotation)
        for field, value in annotation["fields"].items():
            grouped[annotation["record_id"]].setdefault(
                field, resolved.get((annotation["record_id"], field), value)
            )
    return [{"record_id": record_id, "fields": fields} for record_id, fields in sorted(grouped.items())]


def freeze_gold_set(
    selection: dict[str, Any],
    annotations: list[dict[str, Any]],
    adjudications: list[dict[str, Any]],
    output_dir: Path,
) -> dict[str, Any]:
    if not selection.get("evaluation_only"):
        raise ValueError("gold selection must be evaluation-only")
    for annotation in annotations:
        validate_gold_annotation(annotation)
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = selection.get("entries", [])
    reviewers = sorted(set(a.get("reviewer_id", "") for a in annotations if a.get("reviewer_id")))
    frozen = {
        "schema_version": GOLD_MANIFEST_VERSION,
        "gold_schema_version": GOLD_MANIFEST_VERSION,
        "annotation_version": GOLD_ANNOTATION_VERSION,
        "adjudication_version": "1.0.0",
        "selection": selection,
        "annotations": annotations,
        "adjudications": adjudications,
        "resolved_annotations": resolved_gold(annotations, adjudications),
        "reviewer_metadata": {
            "total_reviewers": len(reviewers),
            "reviewer_ids": reviewers,
            "has_human_reviewers": any(not r.startswith("fixture-") for r in reviewers),
        },
        "evaluation_only": True,
        "training_excluded_source_hashes": sorted(item["source_content_hash"] for item in entries),
        "training_excluded_simhashes": sorted(item["transcript_simhash64"] for item in entries),
        "training_excluded_transcript_shas": sorted(
            item.get("transcript_sha256", "") for item in entries if item.get("transcript_sha256")
        ),
        "semantic_input_version": SEMANTIC_INPUT_VERSION,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
    }
    payload = (json.dumps(frozen, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()
    digest = hashlib.sha256(payload).hexdigest()
    path = output_dir / f"gold-manifest-{digest[:16]}.json"
    if path.exists() and path.read_bytes() != payload:
        raise OSError("immutable gold manifest collision")
    if not path.exists():
        path.write_bytes(payload)
    return {
        "manifest_path": str(path),
        "sha256": digest,
        "source_count": len(entries),
        "annotation_count": len(annotations),
        "adjudication_count": len(adjudications),
        "evaluation_only": True,
    }


def training_exclusions_from_gold_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return exact and lexical-near-duplicate identities that training must reject."""
    if not manifest.get("evaluation_only"):
        raise ValueError("only frozen evaluation-only gold manifests may create exclusions")
    entries = manifest.get("selection", {}).get("entries", [])
    source_hashes = sorted(
        set(manifest.get("training_excluded_source_hashes", [item["source_content_hash"] for item in entries]))
    )
    simhashes = sorted(
        set(manifest.get("training_excluded_simhashes", [item["transcript_simhash64"] for item in entries]))
    )
    transcript_shas = sorted(
        set(
            manifest.get(
                "training_excluded_transcript_shas",
                [item.get("transcript_sha256", "") for item in entries if item.get("transcript_sha256")],
            )
        )
    )
    return {
        "source_content_hashes": source_hashes,
        "transcript_simhash64": simhashes,
        "transcript_sha256": transcript_shas,
        "version": "gold-training-exclusion-v1",
    }


def _label_result(predicted: Any, label: dict[str, Any]) -> str:
    status = label.get("status")
    if status in {"unknown", "not_inferable", "ambiguous", "reject"}:
        return "correct_abstention" if predicted is None else "unsupported_inference"
    if predicted is None:
        return "unnecessary_abstention"
    if status == "accepted":
        return "accepted"
    if status in {"partial", "wrong", "too_broad", "too_narrow", "too_vague", "too_detailed", "leakage"}:
        return status
    allowed: set[str] = set()
    for val in label.get("acceptable_values", []):
        if val is not None:
            allowed.add(str(val).strip().lower())
    if "human_value" in label and label["human_value"] is not None:
        allowed.add(str(label["human_value"]).strip().lower())

    predicted_values = predicted if isinstance(predicted, list) else [predicted]
    if any(str(value).strip().lower() in allowed for value in predicted_values if value is not None):
        return "accepted"
    return "wrong"


def field_metrics(briefs: dict[str, dict[str, Any]], labels: list[dict[str, Any]]) -> dict[str, Any]:
    """Calculate field-level evaluation metrics, confusion matrices, and confidence calibration."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    confusion_matrices: dict[str, dict[str, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    truth_counts: dict[str, Counter[str]] = defaultdict(Counter)
    predicted_counts: dict[str, Counter[str]] = defaultdict(Counter)
    true_positives: dict[str, Counter[str]] = defaultdict(Counter)
    errors: list[dict[str, Any]] = []

    band_calibration: dict[str, Counter[str]] = defaultdict(Counter)
    band_conf_sum: dict[str, float] = defaultdict(float)
    numeric_buckets: dict[str, Counter[str]] = defaultdict(Counter)
    numeric_conf_sum: dict[str, float] = defaultdict(float)

    bucket_ranges = (
        ("0.0-0.2", 0.0, 0.2),
        ("0.2-0.4", 0.2, 0.4),
        ("0.4-0.6", 0.4, 0.6),
        ("0.6-0.8", 0.6, 0.8),
        ("0.8-1.0", 0.8, 1.01),
    )

    for item in labels:
        brief = briefs.get(item["record_id"])
        if not brief:
            continue
        for field, label in item.get("fields", {}).items():
            predicted = evidence_value(brief.get(field, {}))
            outcome = _label_result(predicted, label)
            counts[field][outcome] += 1

            if outcome == "unsupported_inference":
                counts[field]["false_inference"] += 1

            if field in CONTROLLED_FIELDS and label.get("status") in {"value", "accepted", "wrong", "partial"}:
                truth = str(
                    label.get("acceptable_values", [label.get("human_value", "unknown")])[0]
                ).strip().lower()
                prediction = str(
                    predicted[0] if isinstance(predicted, list) and predicted else (predicted or "abstain")
                ).strip().lower()
                confusion[field][f"{prediction} -> {truth}"] += 1
                confusion_matrices[field][prediction][truth] += 1
                truth_counts[field][truth] += 1
                predicted_counts[field][prediction] += 1
                if prediction == truth:
                    true_positives[field][truth] += 1

            node = brief.get(field, {})
            confidence = node.get("confidence") if isinstance(node, dict) else None
            is_accepted = outcome == "accepted"

            if isinstance(confidence, (int, float)):
                band = "high" if confidence >= 0.8 else "medium" if confidence >= 0.6 else "low"
                band_calibration[band]["accepted" if is_accepted else "not_accepted"] += 1
                band_conf_sum[band] += float(confidence)

                for b_name, b_min, b_max in bucket_ranges:
                    if b_min <= confidence < b_max:
                        numeric_buckets[b_name]["accepted" if is_accepted else "not_accepted"] += 1
                        numeric_conf_sum[b_name] += float(confidence)
                        break
            else:
                band_calibration["unknown"]["accepted" if is_accepted else "not_accepted"] += 1

            if outcome not in {"accepted", "correct_abstention"}:
                errors.append({
                    "record_id": item["record_id"],
                    "field": field,
                    "outcome": outcome,
                    "predicted": predicted,
                    "confidence": confidence,
                    "category": label.get("error_category", outcome),
                })

    fields: dict[str, Any] = {}
    for field, value in counts.items():
        total = value["accepted"] + value["partial"] + value["wrong"] + value["too_broad"] + value["too_narrow"] + value["too_vague"] + value["too_detailed"] + value["leakage"] + value["correct_abstention"] + value["unnecessary_abstention"] + value["unsupported_inference"]
        if total == 0:
            total = sum(value.values())
        fields[field] = {
            **value,
            "total": total,
            "acceptance_rate": round(value["accepted"] / total, 4) if total else None,
            "unsupported_inference_rate": round(value["unsupported_inference"] / total, 4) if total else None,
            "false_inference_rate": round(value["unsupported_inference"] / total, 4) if total else None,
            "correct_abstention_rate": round(value["correct_abstention"] / total, 4) if total else None,
            "unnecessary_abstention_rate": round(value["unnecessary_abstention"] / total, 4) if total else None,
        }

    classification: dict[str, Any] = {}
    for field in CONTROLLED_FIELDS:
        labels_seen = sorted(set(truth_counts[field]) | set(predicted_counts[field]))
        if not labels_seen:
            continue
        per_class: dict[str, Any] = {}
        scores: list[float] = []
        for lbl in labels_seen:
            tp = true_positives[field][lbl]
            pred_cnt = predicted_counts[field][lbl]
            truth_cnt = truth_counts[field][lbl]
            prec = round(tp / pred_cnt, 4) if pred_cnt else 0.0
            rec = round(tp / truth_cnt, 4) if truth_cnt else 0.0
            f1 = round(2 * prec * rec / (prec + rec), 4) if (prec + rec) else 0.0
            scores.append(f1)
            per_class[lbl] = {
                "truth_count": truth_cnt,
                "predicted_count": pred_cnt,
                "true_positives": tp,
                "precision": prec,
                "recall": rec,
                "f1": f1,
            }
        classification[field] = {
            "macro_f1": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "label_count": len(labels_seen),
            "per_class": per_class,
        }

    calibration_report: dict[str, Any] = {}
    for band in ("high", "medium", "low", "unknown"):
        val = band_calibration.get(band, Counter())
        tot = sum(val.values())
        acc = val["accepted"]
        acc_rate = round(acc / tot, 4) if tot else None
        mean_conf = round(band_conf_sum[band] / tot, 4) if tot and band != "unknown" else None
        calibration_report[band] = {
            "total": tot,
            "accepted": acc,
            "not_accepted": val["not_accepted"],
            "acceptance_rate": acc_rate,
            "mean_confidence": mean_conf,
            "calibration_gap": round(abs(mean_conf - acc_rate), 4) if mean_conf is not None and acc_rate is not None else None,
        }

    numeric_report: dict[str, Any] = {}
    total_evaluated_cal = 0
    weighted_gap_sum = 0.0
    for b_name, _, _ in bucket_ranges:
        val = numeric_buckets.get(b_name, Counter())
        tot = sum(val.values())
        acc = val["accepted"]
        acc_rate = round(acc / tot, 4) if tot else None
        mean_conf = round(numeric_conf_sum[b_name] / tot, 4) if tot else None
        gap = round(abs(mean_conf - acc_rate), 4) if mean_conf is not None and acc_rate is not None else None
        numeric_report[b_name] = {
            "total": tot,
            "accepted": acc,
            "not_accepted": val["not_accepted"],
            "acceptance_rate": acc_rate,
            "mean_confidence": mean_conf,
            "calibration_gap": gap,
        }
        if tot > 0 and gap is not None:
            total_evaluated_cal += tot
            weighted_gap_sum += gap * tot

    ece = round(weighted_gap_sum / total_evaluated_cal, 4) if total_evaluated_cal > 0 else None

    human_count = sum(1 for item in labels if not str(item.get("reviewer_id", "")).startswith("fixture-"))
    fixture_only = human_count == 0 and len(labels) > 0

    return {
        "fields": fields,
        "classification": classification,
        "confusion_matrices": {field: dict(matrix) for field, matrix in confusion.items()},
        "confusion_matrix_grid": {
            field: {pred: dict(truths) for pred, truths in grid.items()}
            for field, grid in confusion_matrices.items()
        },
        "errors": errors,
        "confidence_calibration": calibration_report,
        "numeric_calibration": numeric_report,
        "expected_calibration_error": ece,
        "fixture_only": fixture_only,
        "human_annotation_count": human_count,
    }


def leakage_audit(briefs: dict[str, dict[str, Any]], records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    reports = {}
    for record in records:
        if record["record_id"] in briefs:
            transcript = str(_value(record, "content.clean_transcript") or "")
            reports[record["record_id"]] = field_leakage_report(briefs[record["record_id"]], transcript)
    risky = [
        {"record_id": rid, "field": field, **metrics}
        for rid, fields in reports.items()
        for field, metrics in fields.items()
        if metrics.get("severity") != "none"
    ]
    return {
        "record_count": len(reports),
        "high_risk_count": sum(item.get("severity") == "high" for item in risky),
        "warn_risk_count": sum(item.get("severity") == "warn" for item in risky),
        "risky_fields": risky,
        "reports": reports,
    }


def detect_anomalies(briefs: dict[str, dict[str, Any]], records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Corpus-level anomaly detection for collapse, saturation, excessive leakage, and empty evidence."""
    total_records = len(briefs)
    central_ideas = Counter(str(evidence_value(item.get("central_idea")) or "") for item in briefs.values() if evidence_value(item.get("central_idea")) is not None)
    topics = Counter(str(evidence_value(item.get("topic")) or "") for item in briefs.values() if evidence_value(item.get("topic")) is not None)
    audiences = Counter(str(evidence_value(item.get("target_audience")) or "") for item in briefs.values() if evidence_value(item.get("target_audience")) is not None)
    objectives = Counter(str(evidence_value(item.get("content_objective")) or "") for item in briefs.values() if evidence_value(item.get("content_objective")) is not None)
    formats = Counter(str(evidence_value(item.get("content_format")) or "") for item in briefs.values() if evidence_value(item.get("content_format")) is not None)

    anomalies: list[dict[str, Any]] = []
    category_counts: Counter[str] = Counter()

    if total_records >= 5:
        for topic_str, count in topics.items():
            if count / total_records > 0.5:
                category_counts["generic_topic_collapse"] += count
                for rid, b in briefs.items():
                    if str(evidence_value(b.get("topic")) or "") == topic_str:
                        anomalies.append({"record_id": rid, "category": "generic_topic_collapse", "topic": topic_str, "frequency": count / total_records})
        for aud_str, count in audiences.items():
            if count / total_records > 0.5:
                category_counts["identical_audience_everywhere"] += count
                for rid, b in briefs.items():
                    if str(evidence_value(b.get("target_audience")) or "") == aud_str:
                        anomalies.append({"record_id": rid, "category": "identical_audience_everywhere", "audience": aud_str, "frequency": count / total_records})
        for obj_str, count in objectives.items():
            if count / total_records > 0.8:
                category_counts["objective_collapse"] += count
                for rid, b in briefs.items():
                    if str(evidence_value(b.get("content_objective")) or "") == obj_str:
                        anomalies.append({"record_id": rid, "category": "objective_collapse", "objective": obj_str, "frequency": count / total_records})
        for fmt_str, count in formats.items():
            if count / total_records > 0.8:
                category_counts["format_collapse"] += count
                for rid, b in briefs.items():
                    if str(evidence_value(b.get("content_format")) or "") == fmt_str:
                        anomalies.append({"record_id": rid, "category": "format_collapse", "format": fmt_str, "frequency": count / total_records})

    for record in records:
        rid = record["record_id"]
        brief = briefs.get(rid)
        if not brief:
            continue
        idea = str(evidence_value(brief.get("central_idea")) or "")
        if idea and len(idea.split()) > 32:
            anomalies.append({"record_id": rid, "category": "central_idea_too_long", "word_count": len(idea.split())})
            category_counts["central_idea_too_long"] += 1
        if idea and central_ideas[idea] > 1:
            anomalies.append({"record_id": rid, "category": "repeated_central_idea", "count": central_ideas[idea]})
            category_counts["repeated_central_idea"] += 1
        if all(evidence_value(brief.get(field)) is not None for field in CORE_FIELDS):
            anomalies.append({"record_id": rid, "category": "all_core_fields_filled"})
            category_counts["all_core_fields_filled"] += 1
        if any(
            (brief.get(field, {}).get("confidence", 0) if isinstance(brief.get(field), dict) else 0) >= 0.98
            for field in CORE_FIELDS
        ):
            anomalies.append({"record_id": rid, "category": "suspiciously_high_confidence"})
            category_counts["suspiciously_high_confidence"] += 1

        for field in CORE_FIELDS:
            node = brief.get(field, {})
            if isinstance(node, dict) and node.get("evidence_type") == "model_inference":
                sources = node.get("sources", [])
                if not sources:
                    anomalies.append({"record_id": rid, "field": field, "category": "empty_evidence"})
                    category_counts["empty_evidence"] += 1

    return {
        "anomaly_count": len(anomalies),
        "anomalies": anomalies,
        "anomalies_by_category": dict(category_counts),
        "anomaly_rate": round(len(set(a["record_id"] for a in anomalies)) / total_records, 4) if total_records else 0.0,
    }


def contamination_test(
    record: dict[str, Any],
    contexts: dict[str, dict[str, Any]],
    service: SemanticReconstructionService,
) -> dict[str, Any]:
    """Adversarial contamination testing to ensure client context does not mutate source meaning."""
    outputs = {name: service.reconstruct(record, context) for name, context in contexts.items()}
    protected = ("topic", "central_idea", "content_objective", "content_format")
    values = {field: {name: evidence_value(brief[field]) for name, brief in outputs.items()} for field in protected}
    changed = [
        field for field, per_context in values.items()
        if len(set(canonical_json(value) for value in per_context.values())) > 1
    ]
    per_field_analysis = {}
    for field in protected:
        stable = field not in changed
        per_field_analysis[field] = {
            "stable": stable,
            "values": values[field],
        }
    return {
        "contexts": sorted(contexts),
        "protected_field_values": values,
        "per_field_analysis": per_field_analysis,
        "contaminated_fields": changed,
        "client_context_contamination_rate": round(len(changed) / len(protected), 4),
        "pass": not changed,
    }


def benchmark_adapter(
    records: list[dict[str, Any]],
    client: dict[str, Any],
    labels: list[dict[str, Any]],
    factory: Callable[[], SemanticReconstructionService],
    *,
    mode: str = "staged",
    input_variant: str = "full",
    name: str = "adapter",
) -> dict[str, Any]:
    service = factory()
    started = time.perf_counter()
    briefs: dict[str, dict[str, Any]] = {}
    failures: list[dict[str, Any]] = []

    for record in records:
        try:
            briefs[record["record_id"]] = service.reconstruct(
                record, client, mode=mode, input_variant=input_variant
            )
        except Exception as exc:
            failures.append({"record_id": record["record_id"], "error": str(exc)})

    elapsed = time.perf_counter() - started
    evaluated = field_metrics(briefs, labels)
    stages_multiplier = 3 if mode == "staged" else 1
    input_tokens = sum(
        len(str(_value(record, "content.clean_transcript") or "").split()) * stages_multiplier
        for record in records
    )
    output_tokens = sum(len(canonical_json(brief).split()) for brief in briefs.values())

    return {
        "adapter": name,
        "adapter_version": service.adapter.version,
        "prompt_version": service.adapter.prompt_version,
        "mode": mode,
        "semantic_input_version": SEMANTIC_INPUT_VERSION,
        "input_variant": input_variant,
        "record_count": len(records),
        "successful_inferences": len(briefs),
        "failed_inferences": len(failures),
        "failure_rate": round(len(failures) / len(records), 4) if records else 0.0,
        "failures": failures,
        "latency_seconds": round(elapsed, 6),
        "mean_latency_ms": round(elapsed * 1000 / len(records), 3) if records else None,
        "estimated_requests": len(records) * stages_multiplier,
        "measured_input_token_proxy": input_tokens,
        "measured_output_token_proxy": output_tokens,
        "metrics": evaluated,
        "leakage": leakage_audit(briefs, records),
        "anomalies": detect_anomalies(briefs, records),
        "briefs": briefs,
    }


def run_ablation(
    records: list[dict[str, Any]],
    client: dict[str, Any],
    labels: list[dict[str, Any]],
    factory: Callable[[], SemanticReconstructionService],
    *,
    mode: str = "staged",
) -> dict[str, Any]:
    variants = ("transcript", "structure", "delivery", "full", "full_with_client")
    results = [
        benchmark_adapter(records, client, labels, factory, mode=mode, input_variant=variant, name=f"ablation-{variant}")
        for variant in variants
    ]
    summary_comparison = []
    for res in results:
        f_metrics = res.get("metrics", {}).get("fields", {})
        topic_acc = f_metrics.get("topic", {}).get("acceptance_rate")
        central_acc = f_metrics.get("central_idea", {}).get("acceptance_rate")
        obj_acc = f_metrics.get("content_objective", {}).get("acceptance_rate")
        fmt_acc = f_metrics.get("content_format", {}).get("acceptance_rate")
        summary_comparison.append({
            "variant": res["input_variant"],
            "input_token_proxy": res["measured_input_token_proxy"],
            "output_token_proxy": res["measured_output_token_proxy"],
            "mean_latency_ms": res["mean_latency_ms"],
            "topic_acceptance": topic_acc,
            "central_idea_acceptance": central_acc,
            "objective_acceptance": obj_acc,
            "format_acceptance": fmt_acc,
            "leakage_high_risk": res.get("leakage", {}).get("high_risk_count", 0),
            "anomalies": res.get("anomalies", {}).get("anomaly_count", 0),
        })
    return {
        "mode": mode,
        "semantic_input_version": SEMANTIC_INPUT_VERSION,
        "results": results,
        "variant_comparison": summary_comparison,
        "recommended_minimal_variant": "transcript" if not results[0].get("failures") else "full",
    }


def pilot(
    records: list[dict[str, Any]],
    client: dict[str, Any],
    factory: Callable[[], SemanticReconstructionService],
    *,
    limit: int = 500,
    mode: str = "staged",
    input_variant: str = "full",
) -> dict[str, Any]:
    """Execute a limited semantic pilot before full-corpus inference."""
    selected = records[:limit]
    result = benchmark_adapter(selected, client, [], factory, mode=mode, input_variant=input_variant, name="pilot")

    known = lambda brief, field: evidence_value(brief.get(field)) is not None
    briefs = result["briefs"]

    eligible_full_script = sum(
        all(known(brief, field) for field in ("topic", "central_idea"))
        and (known(brief, "content_objective") or known(brief, "content_format"))
        for brief in briefs.values()
    )
    eligible_hook = sum(known(brief, "hook_intent") or known(brief, "topic") for brief in briefs.values())
    eligible_continuation = sum(known(brief, "topic") and known(brief, "central_idea") for brief in briefs.values())
    eligible_structure = sum(known(brief, "content_format") or known(brief, "content_objective") for brief in briefs.values())
    eligible_section = sum(known(brief, "topic") for brief in briefs.values())
    eligible_cta = sum(known(brief, "cta_intent") for brief in briefs.values())
    eligible_style = sum(known(brief, "tone") or known(brief, "perspective") for brief in briefs.values())

    abstention_summary = {
        field: sum(not known(brief, field) for brief in briefs.values())
        for field in ALL_FIELDS
    }

    cache_hits_total = sum(b.get("adapter", {}).get("cache_hits", 0) for b in briefs.values())

    return {
        **result,
        "pilot_limit": limit,
        "pilot_records_examined": len(selected),
        "successful_inference_count": len(briefs),
        "failed_inference_count": result["failed_inferences"],
        "eligible_full_script_conditioning_count": eligible_full_script,
        "eligible_per_objective": {
            "full_script": eligible_full_script,
            "hook": eligible_hook,
            "continuation": eligible_continuation,
            "structure": eligible_structure,
            "section": eligible_section,
            "cta": eligible_cta,
            "measurable_style": eligible_style,
        },
        "abstention_summary": abstention_summary,
        "cache_hits_total": cache_hits_total,
        "cache_behavior": "adapter/service cache is supplied by caller; per-brief cache accounting recorded",
    }


def full_corpus_projection(
    pilot_report: dict[str, Any],
    total_sources: int,
    *,
    input_price_per_million: float = 0.0,
    output_price_per_million: float = 0.0,
) -> dict[str, Any]:
    """Extrapolate from a measured pilot; zero prices mean cost is intentionally unknown."""
    count = int(pilot_report.get("record_count", 0))
    if count <= 0:
        raise ValueError("pilot report has no processed records")
    input_per_source = float(pilot_report.get("measured_input_token_proxy", 0)) / count
    output_per_source = float(pilot_report.get("measured_output_token_proxy", 0)) / count
    latency_per_source = float(pilot_report.get("latency_seconds", 0)) / count
    requests_per_source = float(pilot_report.get("estimated_requests", 0)) / count
    cost_per_source = (
        input_per_source / 1_000_000 * input_price_per_million
        + output_per_source / 1_000_000 * output_price_per_million
    )

    expected = {
        "requests": round(requests_per_source * total_sources),
        "input_token_proxy": round(input_per_source * total_sources),
        "output_token_proxy": round(output_per_source * total_sources),
        "latency_seconds_serial": round(latency_per_source * total_sources, 3),
        "cost": round(cost_per_source * total_sources, 6) if (input_price_per_million or output_price_per_million) else None,
    }
    best = {
        "requests": round(requests_per_source * total_sources),
        "input_token_proxy": round(input_per_source * total_sources * 0.8),
        "output_token_proxy": round(output_per_source * total_sources * 0.8),
        "latency_seconds_serial": round(latency_per_source * total_sources * 0.8, 3),
        "cost": round(cost_per_source * total_sources * 0.8, 6) if (input_price_per_million or output_price_per_million) else None,
    }
    worst = {
        "requests": round(requests_per_source * total_sources * 1.3),
        "input_token_proxy": round(input_per_source * total_sources * 1.3),
        "output_token_proxy": round(output_per_source * total_sources * 1.3),
        "latency_seconds_serial": round(latency_per_source * total_sources * 1.3, 3),
        "cost": round(cost_per_source * total_sources * 1.3, 6) if (input_price_per_million or output_price_per_million) else None,
    }

    return {
        "sources": total_sources,
        "based_on_pilot_sources": count,
        "pricing_configured": bool(input_price_per_million or output_price_per_million),
        "best": best,
        "expected": expected,
        "worst": worst,
        "note": (
            "Token counts and latencies are local request/output proxies and rule-adapter times; "
            "remote API billing and throughput must be measured separately with a live provider pilot."
        ),
    }
