from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from script_writer.semantic_evaluation import evaluate_gold, reviewer_agreement
from script_writer.semantic_reconstruction import (
    MockSemanticIntentAdapter, RuleBasedSemanticIntentAdapter, SemanticInferenceCache,
    SemanticInferenceError, SemanticReconstructionService, estimate_corpus, field_leakage_report,
)
from script_writer.sharded_assembly import build_shards
from script_writer.training_compiler import TrainingExampleCompiler
from script_writer.training_contracts import ClientTrainingContext
from script_writer.training_intent import SemanticIntentReconstructor


ROOT = Path(__file__).resolve().parents[1]
REAL = json.loads((ROOT / "examples/compiled/86c1671e8a3a7b46.script-intelligence.v1.json").read_text())

def client() -> dict:
    return ClientTrainingContext.from_path(ROOT / "fixtures/client.example.json").projection

class SemanticReconstructionTests(unittest.TestCase):
    def test_rule_adapter_reconstructs_real_subject_and_abstains(self) -> None:
        brief = SemanticReconstructionService(RuleBasedSemanticIntentAdapter()).reconstruct(REAL, client())
        self.assertEqual(brief["topic"]["value"], "DOGE government workforce cuts")
        self.assertEqual(brief["content_format"]["value"], "commentary")
        self.assertEqual(brief["target_audience"]["evidence_type"], "unknown")
        self.assertEqual(brief["cta_intent"]["evidence_type"], "unknown")
        self.assertEqual(brief["topic"]["evidence_type"], "model_inference")

    def test_invalid_adapter_response_fails_closed(self) -> None:
        adapter = MockSemanticIntentAdapter({"topic": {"value": "x"}})
        with self.assertRaises(SemanticInferenceError):
            SemanticReconstructionService(adapter, retries=0).reconstruct(REAL, client())

    def test_cache_and_retries_are_deterministic(self) -> None:
        fields = {"topic": {"value":"topic", "confidence":0.8,"evidence_paths":["$.content.clean_transcript"]}}
        with tempfile.TemporaryDirectory() as directory:
            cache = SemanticInferenceCache(Path(directory) / "cache.sqlite3")
            service = SemanticReconstructionService(MockSemanticIntentAdapter(fields), cache)
            first, second = service.reconstruct(REAL, client()), service.reconstruct(REAL, client())
            cache.close()
        self.assertEqual(first["brief_id"], second["brief_id"])
        self.assertEqual(second["adapter"]["cache_hits"], 3)

    def test_field_leakage_rejects_transcript_like_central_idea(self) -> None:
        record = copy.deepcopy(REAL)
        transcript = record["content"]["clean_transcript"]["value"]
        fields = {"central_idea":{"value":transcript,"confidence":.9,"evidence_paths":["$.content.clean_transcript"]}}
        brief = SemanticReconstructionService(MockSemanticIntentAdapter(fields)).reconstruct(record, client())
        self.assertEqual(brief["central_idea"]["evidence_type"], "unknown")

    def test_semantic_brief_enables_eligible_objectives_without_audience_fabrication(self) -> None:
        reconstructor = SemanticIntentReconstructor(SemanticReconstructionService(RuleBasedSemanticIntentAdapter()))
        result = TrainingExampleCompiler(reconstructor).compile(REAL, client(), group_id="g", split="train")
        full = next(x for x in result.examples if x["identity"]["dataset_objective"] == "full_script_sft")
        self.assertNotEqual(full["quality"]["eligibility"], "ineligible")
        self.assertEqual(full["training_input"]["content_brief"]["audience"]["evidence_type"], "unknown")

    def test_gold_metrics_measure_abstention_and_errors(self) -> None:
        brief = SemanticReconstructionService(RuleBasedSemanticIntentAdapter()).reconstruct(REAL, client())
        annotations = [{"record_id":REAL["record_id"],"fields":{"topic":{"status":"value","acceptable_values":["DOGE government workforce cuts"]},"target_audience":{"status":"not_inferable"},"cta_intent":{"status":"not_inferable"}}}]
        report = evaluate_gold({REAL["record_id"]:brief}, annotations)
        self.assertEqual(report["fields"]["topic"]["accepted"], 1)
        self.assertEqual(report["fields"]["target_audience"]["correct_abstention"], 1)
        self.assertEqual(reviewer_agreement(annotations + annotations, "topic")["exact_agreement_rate"], 1.0)

    def test_estimate_and_disk_shards_are_deterministic(self) -> None:
        estimate = estimate_corpus(7500, 160, adapter=RuleBasedSemanticIntentAdapter(), input_price_per_million=1, output_price_per_million=2)
        self.assertEqual(estimate["requests"], 22500)
        with tempfile.TemporaryDirectory() as directory:
            examples = [{"id": index} for index in range(2501)]
            first = build_shards(examples, Path(directory), shard_size=1000)
            second = build_shards(examples, Path(directory), shard_size=1000)
            self.assertEqual(first["manifest_sha256"], second["manifest_sha256"])
            self.assertEqual(first["example_count"], 2501)
            self.assertEqual(len(first["shards"]), 3)
