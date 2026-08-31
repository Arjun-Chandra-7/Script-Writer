from __future__ import annotations

import hashlib
import unittest

from script_writer.dataset_design import (
    OBJECTIVE_POLICIES,
    DatasetObjective,
    LeakageGuard,
    LeakageIdentity,
    objective_eligibility,
    simhash64,
)


def identity(source: str, transcript: str, derived_from: str | None = None) -> LeakageIdentity:
    return LeakageIdentity(
        source_content_hash=source,
        transcript_sha256=hashlib.sha256(transcript.encode()).hexdigest(),
        transcript_simhash64=simhash64(transcript),
        derived_from_source_hash=derived_from,
    )


class DatasetDesignTests(unittest.TestCase):
    def test_exact_near_and_derived_examples_share_a_group(self) -> None:
        guard = LeakageGuard(near_duplicate_hamming_distance=8)
        records = [
            identity("source-a", "Walk ten minutes every morning for better consistency."),
            identity("source-b", "Walk ten minutes every morning for better consistency!"),
            identity("derived-c", "A rewritten target.", derived_from="source-a"),
        ]
        groups = guard.cluster(records)
        self.assertEqual(len(set(groups.values())), 1)

    def test_split_is_objective_specific_and_deterministic(self) -> None:
        guard = LeakageGuard()
        group = "near-duplicate-group"
        generation = guard.assign_split(
            group, OBJECTIVE_POLICIES[DatasetObjective.GENERATION_SFT], salt="fixed"
        )
        repeated = guard.assign_split(
            group, OBJECTIVE_POLICIES[DatasetObjective.GENERATION_SFT], salt="fixed"
        )
        corpus = guard.assign_split(
            group, OBJECTIVE_POLICIES[DatasetObjective.CORPUS_UNDERSTANDING], salt="fixed"
        )
        self.assertEqual(generation, repeated)
        self.assertIn(generation, {"train", "validation", "test"})
        self.assertEqual(corpus, "corpus")

    def test_cross_split_near_duplicate_leakage_is_rejected(self) -> None:
        members = [
            {
                "source_content_hash": "a",
                "transcript_sha256": "one",
                "near_duplicate_group": "cluster-1",
                "split": "train",
            },
            {
                "source_content_hash": "b",
                "transcript_sha256": "two",
                "near_duplicate_group": "cluster-1",
                "split": "test",
            },
        ]
        with self.assertRaisesRegex(ValueError, "dataset leakage"):
            LeakageGuard.validate_no_leakage(members)

    def test_extractor_record_is_useful_for_corpus_but_not_automatically_sft(self) -> None:
        record = {"record_id": "sir:one"}
        self.assertEqual(
            objective_eligibility(record, DatasetObjective.CORPUS_UNDERSTANDING),
            (True, "eligible"),
        )
        self.assertEqual(
            objective_eligibility(record, DatasetObjective.GENERATION_SFT),
            (False, "missing explicit script generation target"),
        )
        self.assertEqual(
            objective_eligibility(record, DatasetObjective.PERFORMANCE_LEARNING),
            (False, "missing cohort-aware outcome evidence"),
        )
