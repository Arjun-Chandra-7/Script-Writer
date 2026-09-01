from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from script_writer.contracts import heuristic
from script_writer.dataset_design import (
    LeakageGuard,
    LeakageIdentity,
    assign_universal_training_split,
    simhash64,
)
from script_writer.training_compiler import TrainingExampleCompiler
from script_writer.training_contracts import (
    ClientTrainingContext,
    TrainingObjective,
    validate_client_training_context,
    validate_training_example,
)
from script_writer.training_dataset import (
    CorpusTrainingCompiler,
    ReviewStore,
    SamplingPolicy,
    TrainingDatasetBuilder,
    dataset_audit,
    training_readiness_report,
)
from script_writer.training_leakage import leakage_metrics
from script_writer.training_cache import TrainingCompilationCache
from script_writer.training_workflow import build_training_artifacts


ROOT = Path(__file__).resolve().parents[1]
REAL = json.loads((ROOT / "examples/compiled/86c1671e8a3a7b46.script-intelligence.v1.json").read_text())


def context() -> ClientTrainingContext:
    return ClientTrainingContext.from_path(ROOT / "fixtures/client.example.json")


def record_with_test_topic() -> dict:
    record = copy.deepcopy(REAL)
    record["content"]["topic"] = heuristic(
        "technology accountability", [], "explicit_test_semantic_adapter", 0.9
    )
    return record


class TrainingDataTests(unittest.TestCase):
    def test_client_projection_is_compact_grounded_and_deterministic(self) -> None:
        first, second = context(), context()
        validate_client_training_context(first.projection)
        self.assertEqual(first.sha256, second.sha256)
        self.assertNotIn("private_notes", first.projection["fields"])
        self.assertEqual(first.projection["fields"]["niche"]["evidence_type"], "observed")

    def test_missing_client_fields_remain_unknown(self) -> None:
        value = ClientTrainingContext.from_document({"client_id": "sparse"}, source_path="fixture")
        self.assertEqual(value.projection["fields"]["audience"]["evidence_type"], "unknown")

    def test_real_record_generates_supported_objectives_without_cta(self) -> None:
        result = CorpusTrainingCompiler(split_salt="tests").compile([REAL], context().projection)
        objectives = [item["identity"]["dataset_objective"] for item in result.examples]
        self.assertIn(TrainingObjective.FULL_SCRIPT.value, objectives)
        self.assertIn(TrainingObjective.HOOK.value, objectives)
        self.assertIn(TrainingObjective.CONTINUATION.value, objectives)
        self.assertIn(TrainingObjective.STRUCTURE.value, objectives)
        self.assertIn(TrainingObjective.SECTION.value, objectives)
        self.assertIn(TrainingObjective.STYLE.value, objectives)
        self.assertNotIn(TrainingObjective.CTA.value, objectives)
        self.assertTrue(all(item["quality"]["performance_signal_used"] is False for item in result.examples))
        full = next(item for item in result.examples if item["identity"]["dataset_objective"] == TrainingObjective.FULL_SCRIPT.value)
        self.assertEqual(full["quality"]["eligibility"], "ineligible")
        self.assertIn("missing_reliable_topic_or_central_idea", full["quality"]["exclusion_reasons"])

    def test_every_real_example_validates_and_preserves_source_split(self) -> None:
        result = CorpusTrainingCompiler(split_salt="tests").compile([REAL], context().projection)
        for item in result.examples:
            validate_training_example(item)
        self.assertEqual(len({item["identity"]["split"] for item in result.examples}), 1)
        self.assertEqual(len({item["identity"]["source_group_id"] for item in result.examples}), 1)

    def test_intent_does_not_invent_topic_or_private_objective(self) -> None:
        result = CorpusTrainingCompiler(split_salt="tests").compile([REAL], context().projection)
        brief = result.examples[0]["training_input"]["content_brief"]
        self.assertEqual(brief["topic"]["evidence_type"], "unknown")
        self.assertEqual(brief["content_objective"]["evidence_type"], "unknown")
        self.assertEqual(brief["required_concepts"]["evidence_type"], "heuristic_inference")

    def test_leakage_detector_rejects_verbatim_conditioning(self) -> None:
        target = "Most beginners think stronger arms are the key to their first pull up"
        metrics = leakage_metrics({"instruction": f"Start by saying {target}"}, target)
        self.assertTrue(metrics["rejected"])
        self.assertEqual(metrics["severity"], "high")

    def test_abstract_conditioning_passes_leakage_detector(self) -> None:
        metrics = leakage_metrics(
            {"hook_mechanism": "misconception correction", "belief": "arm strength is the main blocker"},
            "Most beginners think stronger arms are the key to their first pull up",
        )
        self.assertFalse(metrics["rejected"])

    def test_structure_target_is_removed_from_its_conditioning(self) -> None:
        result = CorpusTrainingCompiler(split_salt="tests").compile([REAL], context().projection)
        item = next(x for x in result.examples if x["identity"]["dataset_objective"] == TrainingObjective.STRUCTURE.value)
        self.assertNotIn("progression", item["training_input"]["creative_plan"])
        self.assertEqual(item["quality"]["leakage"]["severity"], "none")

    def test_low_confidence_hook_is_rejected_with_reason(self) -> None:
        record = copy.deepcopy(REAL)
        record["script_structure"]["hook"]["confidence"] = 0.2
        result = CorpusTrainingCompiler(split_salt="tests").compile([record], context().projection)
        hook = next(x for x in result.examples if x["identity"]["dataset_objective"] == TrainingObjective.HOOK.value)
        self.assertEqual(hook["quality"]["eligibility"], "ineligible")
        self.assertIn("hook_boundary_confidence_too_low", hook["quality"]["exclusion_reasons"])

    def test_missing_hook_and_structure_do_not_create_those_objectives(self) -> None:
        record = copy.deepcopy(REAL)
        record["script_structure"]["hook"] = {"value": None, "evidence_type": "unknown", "sources": [], "reason": "missing"}
        record["script_structure"]["major_beats"] = []
        result = CorpusTrainingCompiler(split_salt="tests").compile([record], context().projection)
        objectives = {item["identity"]["dataset_objective"] for item in result.examples}
        self.assertNotIn(TrainingObjective.HOOK.value, objectives)
        self.assertNotIn(TrainingObjective.CONTINUATION.value, objectives)
        self.assertNotIn(TrainingObjective.STRUCTURE.value, objectives)
        self.assertNotIn(TrainingObjective.SECTION.value, objectives)

    def test_verified_cta_creates_cta_example(self) -> None:
        record = copy.deepcopy(REAL)
        record["script_structure"]["cta"] = heuristic(
            {"text": "Follow for the next explanation.", "start_seconds": 55, "end_seconds": 59},
            [], "test_reviewed_boundary", 0.9,
        )
        result = CorpusTrainingCompiler(split_salt="tests").compile([record], context().projection)
        self.assertTrue(any(item["identity"]["dataset_objective"] == TrainingObjective.CTA.value for item in result.examples))

    def test_short_script_is_ineligible_with_explicit_reason(self) -> None:
        record = copy.deepcopy(REAL)
        record["content"]["clean_transcript"]["value"] = "Only three words"
        result = CorpusTrainingCompiler(split_salt="tests").compile([record], context().projection)
        full = next(x for x in result.examples if x["identity"]["dataset_objective"] == TrainingObjective.FULL_SCRIPT.value)
        self.assertIn("script_too_short", full["quality"]["exclusion_reasons"])

    def test_low_transcript_confidence_is_rejected_when_available(self) -> None:
        record = copy.deepcopy(REAL)
        record["content"]["transcript_confidence"] = heuristic(0.2, [], "test", 0.9)
        result = CorpusTrainingCompiler(split_salt="tests").compile([record], context().projection)
        full = next(x for x in result.examples if x["identity"]["dataset_objective"] == TrainingObjective.FULL_SCRIPT.value)
        self.assertIn("transcript_confidence_too_low", full["quality"]["exclusion_reasons"])

    def test_malformed_timing_rejects_affected_structure_and_section(self) -> None:
        record = copy.deepcopy(REAL)
        record["script_structure"]["major_beats"][0]["end_seconds"] = -1
        result = CorpusTrainingCompiler(split_salt="tests").compile([record], context().projection)
        structure = next(x for x in result.examples if x["identity"]["dataset_objective"] == TrainingObjective.STRUCTURE.value)
        section = next(x for x in result.examples if x["identity"]["variant_id"] == "section_0")
        self.assertIn("malformed_section_timing", structure["quality"]["exclusion_reasons"])
        self.assertIn("malformed_section_timing", section["quality"]["exclusion_reasons"])

    def test_exact_and_near_duplicates_cluster(self) -> None:
        base = "a clear script about a daily walking habit for complete beginners"
        near = base + " today"
        identities = [
            LeakageIdentity("a", "one", simhash64(base)),
            LeakageIdentity("b", "two", simhash64(near)),
            LeakageIdentity("a", "three", simhash64("different words")),
        ]
        groups = LeakageGuard(near_duplicate_hamming_distance=16).cluster(identities)
        self.assertEqual(groups["a"], groups["b"])

    def test_exact_duplicate_source_is_suppressed_before_example_expansion(self) -> None:
        compiler = CorpusTrainingCompiler(split_salt="tests")
        single = compiler.compile([REAL], context().projection)
        duplicate = compiler.compile([REAL, copy.deepcopy(REAL)], context().projection)
        self.assertEqual(len(single.examples), len(duplicate.examples))
        self.assertEqual(duplicate.exact_duplicate_count, 1)
        self.assertIn("exact_duplicate_suppressed", duplicate.rejections[0]["reasons"])

    def test_universal_split_does_not_depend_on_objective(self) -> None:
        values = {assign_universal_training_split("cluster-a", salt="stable") for _ in TrainingObjective}
        self.assertEqual(len(values), 1)

    def test_manifests_are_immutable_and_reproducible(self) -> None:
        compilation = CorpusTrainingCompiler(split_salt="tests").compile([record_with_test_topic()], context().projection)
        with tempfile.TemporaryDirectory() as directory:
            builder = TrainingDatasetBuilder(Path(directory))
            first = builder.build(compilation, context().projection)
            second = builder.build(compilation, context().projection)
            self.assertEqual(first["manifests"], second["manifests"])
            for manifest in first["manifests"].values():
                self.assertTrue(Path(manifest["manifest_path"]).exists())

    def test_sampling_policy_is_explicit_and_recorded(self) -> None:
        compilation = CorpusTrainingCompiler(split_salt="tests").compile([record_with_test_topic()], context().projection)
        with tempfile.TemporaryDirectory() as directory:
            build = TrainingDatasetBuilder(
                Path(directory), SamplingPolicy(max_examples_per_source_per_objective=1)
            ).build(compilation, context().projection)
            section_manifest = build["manifests"][TrainingObjective.SECTION.value]
            self.assertEqual(section_manifest["example_count"], 1)
            self.assertEqual(section_manifest["sampling_policy"]["max_examples_per_source_per_objective"], 1)

    def test_audit_and_readiness_make_human_review_failure_visible(self) -> None:
        compilation = CorpusTrainingCompiler(split_salt="tests").compile([REAL], context().projection)
        with tempfile.TemporaryDirectory() as directory:
            build = TrainingDatasetBuilder(Path(directory)).build(compilation, context().projection)
            audit = dataset_audit(compilation, build)
            report = training_readiness_report(
                compilation, build, audit, deterministic_regeneration_verified=True
            )
            self.assertEqual(audit["total_source_videos"], 1)
            self.assertEqual(audit["language_distribution"], {"en": 1})
            self.assertEqual(audit["duration_distribution_seconds"], {"30-59": 1})
            self.assertEqual(audit["cta_source_count"], 0)
            self.assertEqual(report["status"], "not_training_ready")
            self.assertIn("sufficient_human_inspection", report["failed_gates"])
            self.assertIn("sufficient_eligible_examples", report["failed_gates"])
            self.assertIn("validation_and_test_sets_present", report["failed_gates"])
            self.assertTrue(report["gates"]["zero_cross_split_source_leakage"])

    def test_review_store_accept_reject_and_flag(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ReviewStore(Path(directory) / "reviews.json")
            store.record("one", "accept", note="good")
            store.record("two", "flag", note="check intent")
            self.assertEqual(store.decisions()["one"]["decision"], "accept")
            with self.assertRaises(ValueError):
                store.record("three", "maybe")

    def test_rejected_review_is_excluded_on_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            initial = build_training_artifacts(
                [ROOT / "examples/compiled/86c1671e8a3a7b46.script-intelligence.v1.json"],
                ROOT / "fixtures/client.example.json", output,
                split_salt="tests", minimum_exported_examples=1,
            )
            continuation_path = Path(initial["manifests"][TrainingObjective.CONTINUATION.value]["data_file"])
            full_path = output / "datasets" / continuation_path
            example = json.loads(full_path.read_text().splitlines()[0])
            ReviewStore(output / "reviews.json").record(example["example_id"], "reject", note="bad reconstruction")
            rebuilt = build_training_artifacts(
                [ROOT / "examples/compiled/86c1671e8a3a7b46.script-intelligence.v1.json"],
                ROOT / "fixtures/client.example.json", output,
                split_salt="tests", minimum_exported_examples=1,
            )
            self.assertNotIn(TrainingObjective.CONTINUATION.value, rebuilt["manifests"])
            self.assertIn("human_rejected", rebuilt["summary"]["rejection_reasons"])

    def test_compilation_is_deterministic(self) -> None:
        compiler = CorpusTrainingCompiler(split_salt="tests")
        first = compiler.compile([REAL], context().projection)
        second = compiler.compile([REAL], context().projection)
        self.assertEqual(first.examples, second.examples)
        self.assertEqual(first.rejections, second.rejections)

    def test_incremental_cache_avoids_recompiling_unchanged_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            cache = TrainingCompilationCache(Path(directory) / "cache.sqlite3")
            compiler = CorpusTrainingCompiler(split_salt="tests", cache=cache)
            first = compiler.compile([REAL], context().projection)
            second = compiler.compile([REAL], context().projection)
            cache.close()
            self.assertEqual((first.cache_hits, first.cache_misses), (0, 1))
            self.assertEqual((second.cache_hits, second.cache_misses), (1, 0))
            self.assertEqual(first.examples, second.examples)


if __name__ == "__main__":
    unittest.main()
