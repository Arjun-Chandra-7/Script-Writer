from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from .contracts import validate_generation_request, validate_generation_result
from .text_analysis import normalize_transcript


EVALUATION_SCHEMA_VERSION = "1.0.0"
EVALUATOR_VERSION = "offline-deterministic-1.0.0"


JUDGMENT_DIMENSIONS = (
    "hook_quality",
    "coherence",
    "information_progression",
    "originality",
    "relevance",
    "audience_fit",
    "client_fit",
    "structural_quality",
    "factual_grounding",
    "unsupported_claims",
    "cta_quality",
    "retrieval_relevance",
    "retrieval_diversity",
)


def longest_common_token_phrase(left: str, right: str) -> dict[str, Any]:
    left_tokens = normalize_transcript(left).split()
    right_tokens = normalize_transcript(right).split()
    previous = [0] * (len(right_tokens) + 1)
    best_length = 0
    best_end = 0
    for left_token in left_tokens:
        current = [0] * (len(right_tokens) + 1)
        for index, right_token in enumerate(right_tokens, start=1):
            if left_token == right_token:
                current[index] = previous[index - 1] + 1
                if current[index] > best_length:
                    best_length = current[index]
                    best_end = index
        previous = current
    return {
        "token_count": best_length,
        "phrase": " ".join(right_tokens[best_end - best_length : best_end]),
    }


def ngram_overlap_share(candidate: str, source: str, n: int = 5) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    candidate_tokens = normalize_transcript(candidate).split()
    source_tokens = normalize_transcript(source).split()
    candidate_ngrams = {
        tuple(candidate_tokens[index : index + n])
        for index in range(max(0, len(candidate_tokens) - n + 1))
    }
    source_ngrams = {
        tuple(source_tokens[index : index + n])
        for index in range(max(0, len(source_tokens) - n + 1))
    }
    if not candidate_ngrams:
        return 0.0
    return len(candidate_ngrams & source_ngrams) / len(candidate_ngrams)


def _metric(
    name: str,
    status: str,
    value: Any,
    *,
    threshold: Any = None,
    details: Any = None,
) -> dict[str, Any]:
    result = {
        "name": name,
        "evaluation_type": "deterministic",
        "status": status,
        "value": value,
    }
    if threshold is not None:
        result["threshold"] = threshold
    if details is not None:
        result["details"] = details
    return result


class OfflineEvaluator:
    version = EVALUATOR_VERSION

    def evaluate(
        self,
        request: dict[str, Any],
        result: dict[str, Any],
        *,
        source_records: list[dict[str, Any]],
        candidate_version: str,
        fixture_set_version: str,
        baseline_version: str | None = None,
    ) -> dict[str, Any]:
        validate_generation_request(request)
        validate_generation_result(result)
        metrics: list[dict[str, Any]] = []
        spoken = str(result["spoken_script"])
        desired_duration = float(request.get("desired_duration_seconds", 0))
        word_count = len(normalize_transcript(spoken).split())
        estimated_seconds = word_count / 2.5 if word_count else 0
        duration_tolerance = max(3.0, desired_duration * 0.25)
        metrics.append(
            _metric(
                "estimated_duration_compliance",
                "pass" if abs(estimated_seconds - desired_duration) <= duration_tolerance else "fail",
                round(estimated_seconds, 3),
                threshold={"target_seconds": desired_duration, "tolerance_seconds": duration_tolerance},
                details="2.5 words/second is an explicit planning proxy, not measured delivery",
            )
        )
        banned = [
            pattern
            for pattern in request.get("banned_patterns", [])
            if str(pattern).casefold() in spoken.casefold()
        ]
        metrics.append(_metric("banned_pattern_compliance", "pass" if not banned else "fail", banned))

        sections = result.get("sections", [])
        timestamps_valid = True
        previous_end = 0.0
        roles = set()
        for section in sections:
            if not isinstance(section, dict):
                timestamps_valid = False
                continue
            roles.add(section.get("role"))
            start, end = section.get("start_seconds"), section.get("end_seconds")
            if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
                timestamps_valid = False
            elif start < previous_end or end < start or end > desired_duration + 0.001:
                timestamps_valid = False
            else:
                previous_end = float(end)
        metrics.append(_metric("section_timestamp_validity", "pass" if timestamps_valid else "fail", timestamps_valid))
        required_roles = {"hook", "body"}
        metrics.append(
            _metric(
                "required_section_coverage",
                "pass" if required_roles.issubset(roles) else "fail",
                sorted(str(role) for role in roles),
                threshold=sorted(required_roles),
            )
        )
        desired_cta = request.get("desired_cta")
        cta = str(result.get("cta") or "")
        cta_pass = bool(cta.strip()) and (
            not desired_cta or normalize_transcript(str(desired_cta)) in normalize_transcript(cta)
        )
        metrics.append(_metric("cta_contract_compliance", "pass" if cta_pass else "fail", cta))

        referenced_ids = {
            str(item.get("record_id"))
            for item in result.get("retrieved_evidence", [])
            if isinstance(item, dict)
        }
        available_ids = {str(record.get("record_id")) for record in source_records}
        unknown_references = sorted(referenced_ids - available_ids)
        metrics.append(
            _metric(
                "retrieval_reference_integrity",
                "pass" if not unknown_references else "fail",
                {"referenced": len(referenced_ids), "unknown": unknown_references},
            )
        )
        referenced_sources = {
            str(item.get("source_content_hash"))
            for item in result.get("retrieved_evidence", [])
            if isinstance(item, dict) and item.get("source_content_hash")
        }
        metrics.append(
            _metric(
                "retrieval_source_diversity",
                "info",
                {"unique_sources": len(referenced_sources), "references": len(referenced_ids)},
                details="Reported without a universal pass threshold; task-specific judgment is required.",
            )
        )

        overlap_details = []
        max_phrase = 0
        max_ngram_share = 0.0
        for record in source_records:
            source_text = str(record.get("content", {}).get("clean_transcript", {}).get("value", ""))
            phrase = longest_common_token_phrase(spoken, source_text)
            share = ngram_overlap_share(spoken, source_text, 5)
            max_phrase = max(max_phrase, int(phrase["token_count"]))
            max_ngram_share = max(max_ngram_share, share)
            if phrase["token_count"] >= 5 or share > 0:
                overlap_details.append(
                    {
                        "record_id": record.get("record_id"),
                        "longest_phrase_tokens": phrase["token_count"],
                        "longest_phrase": phrase["phrase"],
                        "five_gram_overlap_share": round(share, 4),
                    }
                )
        memorization_pass = max_phrase < 8 and max_ngram_share <= 0.15
        metrics.append(
            _metric(
                "corpus_similarity_guard",
                "pass" if memorization_pass else "fail",
                {
                    "max_contiguous_tokens": max_phrase,
                    "max_five_gram_overlap_share": round(max_ngram_share, 4),
                },
                threshold={"max_contiguous_tokens_exclusive": 8, "max_five_gram_overlap_share": 0.15},
                details=overlap_details,
            )
        )
        flagged_claims = result.get("claims_requiring_verification", [])
        metrics.append(
            _metric(
                "claim_review_queue_present",
                "info",
                {"flagged_claim_count": len(flagged_claims)},
                details="This does not determine whether unflagged unsupported claims exist.",
            )
        )

        judgment = {
            dimension: {
                "status": "not_evaluated",
                "evaluation_type": "human_or_versioned_model_judgment_required",
                "score": None,
            }
            for dimension in JUDGMENT_DIMENSIONS
        }
        payload = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "evaluator_version": self.version,
            "candidate_version": candidate_version,
            "baseline_version": baseline_version,
            "fixture_set_version": fixture_set_version,
            "deterministic_metrics": metrics,
            "judgment_dimensions": judgment,
            "summary": {
                "deterministic_passes": sum(metric["status"] == "pass" for metric in metrics),
                "deterministic_failures": sum(metric["status"] == "fail" for metric in metrics),
                "judgment_dimensions_pending": len(judgment),
                "promotion_decision": "not_available_offline",
            },
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["evaluation_id"] = "eval-" + hashlib.sha256(canonical.encode()).hexdigest()[:20]
        return payload

    @staticmethod
    def compare(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
        candidate_metrics = {metric["name"]: metric for metric in candidate["deterministic_metrics"]}
        baseline_metrics = {metric["name"]: metric for metric in baseline["deterministic_metrics"]}
        changes = []
        for name in sorted(candidate_metrics.keys() & baseline_metrics.keys()):
            current = candidate_metrics[name]
            previous = baseline_metrics[name]
            if current["status"] != previous["status"] or current["value"] != previous["value"]:
                changes.append(
                    {
                        "metric": name,
                        "baseline_status": previous["status"],
                        "candidate_status": current["status"],
                        "baseline_value": previous["value"],
                        "candidate_value": current["value"],
                    }
                )
        return {
            "candidate_evaluation_id": candidate["evaluation_id"],
            "baseline_evaluation_id": baseline["evaluation_id"],
            "changes": changes,
            "aggregate_score": None,
            "note": "No cross-dimension aggregate is invented; review metric changes and pending judgments.",
        }


def evaluation_timestamp() -> str:
    return datetime.now(UTC).isoformat()
