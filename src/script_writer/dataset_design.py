from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from .text_analysis import normalize_transcript


class DatasetObjective(StrEnum):
    CORPUS_UNDERSTANDING = "corpus_understanding"
    GENERATION_SFT = "script_generation_sft"
    PREFERENCE_RANKING = "preference_ranking"
    RETRIEVAL_EVALUATION = "retrieval_evaluation"
    CHALLENGE_REGRESSION = "challenge_regression"
    PERFORMANCE_LEARNING = "performance_learning"


@dataclass(frozen=True)
class ObjectivePolicy:
    objective: DatasetObjective
    split_percentages: dict[str, int]
    requires_script_target: bool = False
    requires_preference_pair: bool = False
    requires_outcome: bool = False
    frozen_membership: bool = False
    notes: str = ""


OBJECTIVE_POLICIES = {
    DatasetObjective.CORPUS_UNDERSTANDING: ObjectivePolicy(
        DatasetObjective.CORPUS_UNDERSTANDING,
        {"corpus": 100},
        notes="All rights-permitted content evidence can be indexed; this is not a training split.",
    ),
    DatasetObjective.GENERATION_SFT: ObjectivePolicy(
        DatasetObjective.GENERATION_SFT,
        {"train": 90, "validation": 5, "test": 5},
        requires_script_target=True,
        notes="Requires an explicit prompt/target pair; extractor transcripts alone are not SFT examples.",
    ),
    DatasetObjective.PREFERENCE_RANKING: ObjectivePolicy(
        DatasetObjective.PREFERENCE_RANKING,
        {"train": 80, "validation": 10, "test": 10},
        requires_preference_pair=True,
        notes="Requires comparable chosen/rejected outputs and preference provenance.",
    ),
    DatasetObjective.RETRIEVAL_EVALUATION: ObjectivePolicy(
        DatasetObjective.RETRIEVAL_EVALUATION,
        {"development": 50, "test": 50},
        notes="Requires labeled query/relevance judgments; corpus records remain the searchable collection.",
    ),
    DatasetObjective.CHALLENGE_REGRESSION: ObjectivePolicy(
        DatasetObjective.CHALLENGE_REGRESSION,
        {"challenge": 100},
        frozen_membership=True,
        notes="Owner-reviewed fixtures are append-only and never used for training.",
    ),
    DatasetObjective.PERFORMANCE_LEARNING: ObjectivePolicy(
        DatasetObjective.PERFORMANCE_LEARNING,
        {"train": 80, "validation": 10, "test": 10},
        requires_outcome=True,
        notes="Requires cohort-aware outcomes; raw cross-account view counts are not labels.",
    ),
}


def simhash64(text: str) -> int:
    tokens = normalize_transcript(text).split()
    features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
    weights = [0] * 64
    for feature in features:
        digest = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        for bit in range(64):
            weights[bit] += 1 if digest & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def assign_universal_training_split(
    group_id: str,
    *,
    salt: str,
    split_percentages: dict[str, int] | None = None,
) -> str:
    """Assign one source cluster once for every derived training objective."""
    percentages = split_percentages or {"train": 90, "validation": 5, "test": 5}
    if sum(percentages.values()) != 100:
        raise ValueError("split percentages must sum to 100")
    bucket = int.from_bytes(
        hashlib.blake2b(group_id.encode(), key=salt.encode(), digest_size=8).digest(),
        "big",
    ) % 100
    cumulative = 0
    for split, percentage in percentages.items():
        cumulative += percentage
        if bucket < cumulative:
            return split
    raise AssertionError("unreachable split assignment")


@dataclass(frozen=True)
class LeakageIdentity:
    source_content_hash: str
    transcript_sha256: str
    transcript_simhash64: int
    derived_from_source_hash: str | None = None


class LeakageGuard:
    """Groups exact, derived, and near-duplicate scripts before objective splitting."""

    def __init__(self, near_duplicate_hamming_distance: int = 3):
        if not 0 <= near_duplicate_hamming_distance <= 16:
            raise ValueError("near_duplicate_hamming_distance must be between 0 and 16")
        self.threshold = near_duplicate_hamming_distance

    def cluster(self, records: list[LeakageIdentity]) -> dict[str, str]:
        parent = list(range(len(records)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[max(left_root, right_root)] = min(left_root, right_root)

        exact_sources: dict[str, int] = {}
        exact_transcripts: dict[str, int] = {}
        band_buckets: dict[tuple[int, int], list[int]] = {}
        for index, record in enumerate(records):
            for key in filter(None, (record.source_content_hash, record.derived_from_source_hash)):
                if key in exact_sources:
                    union(index, exact_sources[key])
                exact_sources[key] = index
            if record.transcript_sha256 in exact_transcripts:
                union(index, exact_transcripts[record.transcript_sha256])
            exact_transcripts[record.transcript_sha256] = index
            candidates: set[int] = set()
            for band in range(4):
                value = (record.transcript_simhash64 >> (band * 16)) & 0xFFFF
                candidates.update(band_buckets.get((band, value), []))
            for candidate in candidates:
                if hamming_distance(record.transcript_simhash64, records[candidate].transcript_simhash64) <= self.threshold:
                    union(index, candidate)
            for band in range(4):
                value = (record.transcript_simhash64 >> (band * 16)) & 0xFFFF
                band_buckets.setdefault((band, value), []).append(index)

        groups: dict[int, list[str]] = {}
        for index, record in enumerate(records):
            groups.setdefault(find(index), []).append(record.source_content_hash)
        group_ids = {
            root: hashlib.sha256("\n".join(sorted(sources)).encode()).hexdigest()[:20]
            for root, sources in groups.items()
        }
        return {
            record.source_content_hash: group_ids[find(index)]
            for index, record in enumerate(records)
        }

    def assign_split(
        self, group_id: str, policy: ObjectivePolicy, *, salt: str
    ) -> str:
        if sum(policy.split_percentages.values()) != 100:
            raise ValueError("split percentages must sum to 100")
        bucket = int.from_bytes(
            hashlib.blake2b(
                f"{policy.objective}:{group_id}".encode(),
                key=salt.encode(),
                digest_size=8,
            ).digest(),
            "big",
        ) % 100
        cumulative = 0
        for split, percentage in policy.split_percentages.items():
            cumulative += percentage
            if bucket < cumulative:
                return split
        raise AssertionError("unreachable split assignment")

    @staticmethod
    def validate_no_leakage(members: list[dict[str, Any]]) -> None:
        seen: dict[tuple[str, str], str] = {}
        for member in members:
            split = str(member["split"])
            for kind in ("source_content_hash", "transcript_sha256", "near_duplicate_group"):
                value = member.get(kind)
                if not value:
                    continue
                key = (kind, str(value))
                previous = seen.get(key)
                if previous is not None and previous != split:
                    raise ValueError(
                        f"dataset leakage: {kind}={value} appears in {previous} and {split}"
                    )
                seen[key] = split


def objective_eligibility(
    record: dict[str, Any], objective: DatasetObjective
) -> tuple[bool, str]:
    policy = OBJECTIVE_POLICIES[objective]
    if policy.requires_script_target and not record.get("script_target"):
        return False, "missing explicit script generation target"
    if policy.requires_preference_pair and not record.get("preference_pair"):
        return False, "missing chosen/rejected preference pair"
    if policy.requires_outcome and not record.get("outcome"):
        return False, "missing cohort-aware outcome evidence"
    if objective is DatasetObjective.RETRIEVAL_EVALUATION and not record.get("relevance_judgments"):
        return False, "missing labeled retrieval query/relevance judgments"
    return True, "eligible"
