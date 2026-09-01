from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from script_writer.cli import main
from script_writer.contracts import model_inference, SourceRef, unknown
from script_writer.gold_validation import (
    adjudicate,
    benchmark_adapter,
    contamination_test,
    detect_anomalies,
    field_metrics,
    freeze_gold_set,
    full_corpus_projection,
    multi_reviewer_agreement,
    new_annotation,
    pilot,
    record_strata,
    resolved_gold,
    review_payload,
    run_ablation,
    semantic_quality_gate_report,
    stratified_gold_sample,
    training_exclusions_from_gold_manifest,
    validate_gold_annotation,
    verified_quality_report,
)
from script_writer.semantic_reconstruction import (
    RuleBasedSemanticIntentAdapter,
    SemanticReconstructionService,
    build_semantic_input,
)
from script_writer.training_contracts import ClientTrainingContext
from script_writer.training_dataset import CorpusTrainingCompiler
from script_writer.training_workflow import build_training_artifacts


ROOT = Path(__file__).resolve().parents[1]
REAL = json.loads((ROOT / "examples/compiled/86c1671e8a3a7b46.script-intelligence.v1.json").read_text())


def client() -> dict:
    return ClientTrainingContext.from_path(ROOT / "fixtures/client.example.json").projection


def copies(n: int = 3) -> list[dict]:
    result = []
    for index in range(n):
        record = copy.deepcopy(REAL)
        record["record_id"] = f"gold-{index}"
        record["identity"]["source_content_hash"]["value"] = f"source-{index}"
        result.append(record)
    return result


class GoldValidationTests(unittest.TestCase):
    def test_stratified_selection_is_deterministic_and_evaluation_only(self) -> None:
        first = stratified_gold_sample(copies(), 2, seed="unit")
        second = stratified_gold_sample(copies(), 2, seed="unit")
        self.assertEqual(first, second)
        self.assertTrue(first["evaluation_only"])
        self.assertEqual(len(first["entries"]), 2)
        self.assertIn("duration", first["entries"][0]["strata"])
        self.assertIn("hook", record_strata(REAL))

    def test_blind_assisted_adjudication_and_freeze_preserve_review_history(self) -> None:
        proposal = SemanticReconstructionService(RuleBasedSemanticIntentAdapter()).reconstruct(REAL, client())
        blind = review_payload(REAL, mode="blind", client=client(), proposal=proposal)
        assisted = review_payload(REAL, mode="assisted", client=client(), proposal=proposal)
        self.assertNotIn("proposal", blind)
        self.assertIn("proposal", assisted)
        annotations = [
            new_annotation(REAL["record_id"], "reviewer-a", "blind", {"topic": {"status": "value", "acceptable_values": ["DOGE government workforce cuts"]}}),
            new_annotation(REAL["record_id"], "reviewer-b", "blind", {"topic": {"status": "ambiguous"}}),
        ]
        resolved = adjudicate(REAL["record_id"], "topic", "adjudicator", {"status": "value", "acceptable_values": ["DOGE government workforce cuts", "federal workforce cuts"]}, annotations, note="lead resolved")
        self.assertEqual(resolved["source_reviewers"], ["reviewer-a", "reviewer-b"])
        self.assertEqual(resolved["note"], "lead resolved")
        selection = stratified_gold_sample([REAL], 1)
        with tempfile.TemporaryDirectory() as directory:
            frozen = freeze_gold_set(selection, annotations, [resolved], Path(directory))
            manifest = json.loads(Path(frozen["manifest_path"]).read_text())
        self.assertTrue(manifest["evaluation_only"])
        self.assertEqual(len(manifest["annotations"]), 2)
        self.assertEqual(manifest["resolved_annotations"][0]["fields"]["topic"]["status"], "value")
        self.assertEqual(len(manifest["training_excluded_source_hashes"]), 1)

    def test_gold_exclusion_blocks_exact_and_lexical_near_duplicate_training(self) -> None:
        selection = stratified_gold_sample([REAL], 1)
        with tempfile.TemporaryDirectory() as directory:
            frozen = freeze_gold_set(selection, [], [], Path(directory))
            exclusions = training_exclusions_from_gold_manifest(json.loads(Path(frozen["manifest_path"]).read_text()))
        near = copy.deepcopy(REAL)
        near["record_id"] = "near"
        near["identity"]["source_content_hash"]["value"] = "different-source"
        compiled = CorpusTrainingCompiler(split_salt="gold", gold_exclusions=exclusions).compile([near], client())
        self.assertEqual(compiled.examples, ())
        self.assertIn("gold_evaluation_source_or_near_duplicate_excluded", compiled.rejections[0]["reasons"])

    def test_benchmark_ablation_and_contamination_are_explicit(self) -> None:
        labels = [{"record_id": REAL["record_id"], "fields": {"topic": {"status": "value", "acceptable_values": ["DOGE government workforce cuts"]}, "target_audience": {"status": "not_inferable"}}}]
        factory = lambda: SemanticReconstructionService(RuleBasedSemanticIntentAdapter())
        single = benchmark_adapter([REAL], client(), labels, factory, mode="single_pass", input_variant="full", name="single")
        staged = benchmark_adapter([REAL], client(), labels, factory, mode="staged", input_variant="full", name="staged")
        self.assertEqual(single["estimated_requests"], 1)
        self.assertEqual(staged["estimated_requests"], 3)
        self.assertEqual(staged["metrics"]["fields"]["topic"]["accepted"], 1)
        self.assertEqual(len(run_ablation([REAL], client(), labels, factory)["results"]), 5)
        contexts = {"a": client(), "b": client()}
        self.assertTrue(contamination_test(REAL, contexts, factory())["pass"])
        estimate = full_corpus_projection(staged, 7500, input_price_per_million=1, output_price_per_million=2)
        self.assertEqual(estimate["expected"]["requests"], 22500)
        self.assertTrue(estimate["pricing_configured"])

    def test_field_metrics_controlled_fields_precision_recall_f1_confusion(self) -> None:
        brief_a = {
            "content_objective": model_inference("persuade", [SourceRef("$.content")], "rule", 0.85),
            "content_format": model_inference("commentary", [SourceRef("$.content")], "rule", 0.75),
        }
        brief_b = {
            "content_objective": model_inference("educate", [SourceRef("$.content")], "rule", 0.65),
            "content_format": model_inference("tutorial", [SourceRef("$.content")], "rule", 0.55),
        }
        briefs = {"rec-1": brief_a, "rec-2": brief_b}
        labels = [
            {"record_id": "rec-1", "fields": {"content_objective": {"status": "value", "acceptable_values": ["persuade"]}, "content_format": {"status": "value", "acceptable_values": ["commentary"]}}},
            {"record_id": "rec-2", "fields": {"content_objective": {"status": "value", "acceptable_values": ["persuade"]}, "content_format": {"status": "value", "acceptable_values": ["tutorial"]}}},
        ]
        metrics = field_metrics(briefs, labels)
        obj_class = metrics["classification"]["content_objective"]
        self.assertIn("per_class", obj_class)
        self.assertEqual(obj_class["per_class"]["persuade"]["true_positives"], 1)
        self.assertEqual(obj_class["per_class"]["persuade"]["truth_count"], 2)
        self.assertEqual(obj_class["per_class"]["persuade"]["predicted_count"], 1)
        self.assertEqual(obj_class["per_class"]["persuade"]["precision"], 1.0)
        self.assertEqual(obj_class["per_class"]["persuade"]["recall"], 0.5)
        self.assertIn("persuade -> persuade", metrics["confusion_matrices"]["content_objective"])
        self.assertIn("educate -> persuade", metrics["confusion_matrices"]["content_objective"])
        self.assertEqual(metrics["fields"]["content_objective"]["total"], 2)
        self.assertEqual(metrics["fields"]["content_objective"]["accepted"], 1)
        self.assertEqual(metrics["fields"]["content_objective"]["acceptance_rate"], 0.5)

    def test_field_metrics_abstention_and_unsupported_inference(self) -> None:
        briefs = {
            "rec-1": {"target_audience": unknown("not inferred"), "cta_intent": model_inference("follow channel", [SourceRef("$.persuasion")], "rule", 0.9)},
            "rec-2": {"target_audience": model_inference("teens", [SourceRef("$.content")], "rule", 0.7), "cta_intent": unknown("none")},
        }
        labels = [
            {"record_id": "rec-1", "fields": {"target_audience": {"status": "not_inferable"}, "cta_intent": {"status": "unknown"}}},
            {"record_id": "rec-2", "fields": {"target_audience": {"status": "value", "acceptable_values": ["students"]}, "cta_intent": {"status": "not_inferable"}}},
        ]
        metrics = field_metrics(briefs, labels)
        self.assertEqual(metrics["fields"]["target_audience"]["correct_abstention"], 1)
        self.assertEqual(metrics["fields"]["target_audience"]["wrong"], 1)
        self.assertEqual(metrics["fields"]["cta_intent"]["unsupported_inference"], 1)
        self.assertEqual(metrics["fields"]["cta_intent"]["false_inference"], 1)
        self.assertEqual(metrics["fields"]["cta_intent"]["correct_abstention"], 1)
        self.assertEqual(metrics["fields"]["cta_intent"]["unsupported_inference_rate"], 0.5)

    def test_field_metrics_free_text_human_statuses(self) -> None:
        briefs = {
            "rec-1": {"topic": model_inference("broad politics", [SourceRef("$.content")], "rule", 0.7)},
            "rec-2": {"topic": model_inference("detailed analysis", [SourceRef("$.content")], "rule", 0.7)},
            "rec-3": {"topic": model_inference("leaked transcript text", [SourceRef("$.content")], "rule", 0.7)},
        }
        labels = [
            {"record_id": "rec-1", "fields": {"topic": {"status": "too_broad"}}},
            {"record_id": "rec-2", "fields": {"topic": {"status": "partial"}}},
            {"record_id": "rec-3", "fields": {"topic": {"status": "leakage"}}},
        ]
        metrics = field_metrics(briefs, labels)
        self.assertEqual(metrics["fields"]["topic"]["too_broad"], 1)
        self.assertEqual(metrics["fields"]["topic"]["partial"], 1)
        self.assertEqual(metrics["fields"]["topic"]["leakage"], 1)
        self.assertEqual(len(metrics["errors"]), 3)

    def test_multi_reviewer_agreement(self) -> None:
        annotations = [
            new_annotation("rec-1", "rev-1", "blind", {
                "content_objective": {"status": "value", "acceptable_values": ["persuade"]},
                "target_audience": {"status": "not_inferable"},
            }),
            new_annotation("rec-1", "rev-2", "blind", {
                "content_objective": {"status": "value", "acceptable_values": ["persuade"]},
                "target_audience": {"status": "not_inferable"},
            }),
            new_annotation("rec-2", "rev-1", "blind", {
                "content_objective": {"status": "value", "acceptable_values": ["educate"]},
                "target_audience": {"status": "value", "acceptable_values": ["engineers"]},
            }),
            new_annotation("rec-2", "rev-2", "blind", {
                "content_objective": {"status": "value", "acceptable_values": ["entertain"]},
                "target_audience": {"status": "unknown"},
            }),
        ]
        agreement = multi_reviewer_agreement(annotations)
        obj_aggr = agreement["fields"]["content_objective"]
        self.assertEqual(obj_aggr["multi_reviewed_records"], 2)
        self.assertEqual(obj_aggr["exact_agreement_count"], 1)
        self.assertEqual(obj_aggr["exact_agreement_rate"], 0.5)
        aud_aggr = agreement["fields"]["target_audience"]
        self.assertEqual(aud_aggr["inferability_agreement_rate"], 0.5)

    def test_confidence_calibration(self) -> None:
        briefs = {
            f"rec-{i}": {
                "topic": model_inference(f"topic-{i}", [SourceRef("$.content")], "rule", 0.9 if i < 5 else 0.5)
            }
            for i in range(10)
        }
        labels = [
            {"record_id": f"rec-{i}", "fields": {"topic": {"status": "value", "acceptable_values": [f"topic-{i}"] if i % 2 == 0 else ["other"]}}}
            for i in range(10)
        ]
        metrics = field_metrics(briefs, labels)
        cal = metrics["confidence_calibration"]
        self.assertIn("high", cal)
        self.assertIn("low", cal)
        self.assertEqual(cal["high"]["total"], 5)
        self.assertEqual(cal["low"]["total"], 5)
        self.assertIsNotNone(metrics["expected_calibration_error"])

    def test_anomalies_surface_all_categories(self) -> None:
        brief = SemanticReconstructionService(RuleBasedSemanticIntentAdapter()).reconstruct(REAL, client())
        brief["central_idea"]["value"] = "word " * 35
        report = detect_anomalies({REAL["record_id"]: brief}, [REAL])
        self.assertIn("central_idea_too_long", report["anomalies_by_category"])

        # Multiple records with identical topic collapse & identical audience
        collapsed_briefs = {}
        fake_records = []
        for i in range(6):
            b = copy.deepcopy(brief)
            b["topic"]["value"] = "identical generic topic"
            b["target_audience"]["value"] = "identical generic audience"
            b["content_objective"]["value"] = "persuade"
            b["content_format"]["value"] = "commentary"
            rid = f"rec-{i}"
            collapsed_briefs[rid] = b
            rec = copy.deepcopy(REAL)
            rec["record_id"] = rid
            fake_records.append(rec)
        collapse_report = detect_anomalies(collapsed_briefs, fake_records)
        self.assertIn("generic_topic_collapse", collapse_report["anomalies_by_category"])
        self.assertIn("identical_audience_everywhere", collapse_report["anomalies_by_category"])
        self.assertIn("objective_collapse", collapse_report["anomalies_by_category"])
        self.assertIn("format_collapse", collapse_report["anomalies_by_category"])

    def test_adversarial_contamination_detects_mutation(self) -> None:
        class ContaminatingAdapter(RuleBasedSemanticIntentAdapter):
            def infer(self, request: dict[str, Any]) -> dict[str, Any]:
                res = super().infer(request)
                # mutate topic based on client niche
                client_niche = request.get("client_context", {}).get("niche", {}).get("value", "default")
                if "topic" in res:
                    res["topic"]["value"] = f"mutated for {client_niche}"
                return res
        contaminating_service = SemanticReconstructionService(ContaminatingAdapter())
        fitness_client = copy.deepcopy(client())
        fitness_client["fields"]["niche"] = {"value": "fitness", "evidence_type": "observed", "sources": []}
        finance_client = copy.deepcopy(client())
        finance_client["fields"]["niche"] = {"value": "finance", "evidence_type": "observed", "sources": []}
        contexts = {"fitness": fitness_client, "finance": finance_client}
        result = contamination_test(REAL, contexts, contaminating_service)
        self.assertFalse(result["pass"])
        self.assertIn("topic", result["contaminated_fields"])
        self.assertGreater(result["client_context_contamination_rate"], 0.0)

    def test_quality_gate_does_not_pass_fixture_or_missing_metrics(self) -> None:
        config = json.loads((ROOT / "fixtures/semantic/quality-gates.v1.json").read_text())
        report = semantic_quality_gate_report({"metrics": {"fields": {}}, "leakage": {}}, reviewed_sources=1, config=config)
        self.assertFalse(report["semantic_reconstruction_gold_quality_verified"])
        self.assertFalse(verified_quality_report(report))
        self.assertIn("minimum_human_reviewed_sources", report["failed_gates"])

        # Fixture-only benchmark fails gate even if reviewed_sources >= 100
        fixture_benchmark = {
            "metrics": {
                "fixture_only": True,
                "fields": {
                    "topic": {"acceptance_rate": 0.95},
                    "central_idea": {"acceptance_rate": 0.90},
                    "content_objective": {"acceptance_rate": 0.85},
                    "content_format": {"acceptance_rate": 0.85},
                    "target_audience": {"unsupported_inference_rate": 0.01},
                    "cta_intent": {"unsupported_inference_rate": 0.01},
                },
            },
            "leakage": {"high_risk_count": 0},
        }
        fixture_report = semantic_quality_gate_report(fixture_benchmark, reviewed_sources=150, config=config)
        self.assertFalse(fixture_report["semantic_reconstruction_gold_quality_verified"])
        self.assertIn("minimum_human_reviewed_sources", fixture_report["failed_gates"])

        # Human benchmark passing all gates
        human_benchmark = {
            "metrics": {
                "fixture_only": False,
                "fields": {
                    "topic": {"acceptance_rate": 0.95},
                    "central_idea": {"acceptance_rate": 0.90},
                    "content_objective": {"acceptance_rate": 0.85},
                    "content_format": {"acceptance_rate": 0.85},
                    "target_audience": {"unsupported_inference_rate": 0.01},
                    "cta_intent": {"unsupported_inference_rate": 0.01},
                },
            },
            "leakage": {"high_risk_count": 0},
        }
        passing_report = semantic_quality_gate_report(human_benchmark, reviewed_sources=150, config=config)
        self.assertTrue(passing_report["semantic_reconstruction_gold_quality_verified"])
        self.assertTrue(verified_quality_report(passing_report))

    def test_pilot_workflow_and_projection(self) -> None:
        factory = lambda: SemanticReconstructionService(RuleBasedSemanticIntentAdapter())
        pilot_result = pilot([REAL], client(), factory, limit=10)
        self.assertEqual(pilot_result["pilot_records_examined"], 1)
        self.assertEqual(pilot_result["successful_inference_count"], 1)
        self.assertIn("full_script", pilot_result["eligible_per_objective"])
        self.assertIn("hook", pilot_result["eligible_per_objective"])

        projection = full_corpus_projection(pilot_result, 7500, input_price_per_million=2.5, output_price_per_million=10.0)
        self.assertEqual(projection["sources"], 7500)
        self.assertTrue(projection["pricing_configured"])
        self.assertIsNotNone(projection["expected"]["cost"])
        self.assertIsNotNone(projection["best"]["cost"])
        self.assertIsNotNone(projection["worst"]["cost"])

    def test_dataset_build_integration_with_gold_manifest_and_quality_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            selection = stratified_gold_sample([REAL], 1)
            freeze_gold_set(selection, [], [], out_dir / "frozen")
            manifest_file = next((out_dir / "frozen").glob("gold-manifest-*.json"))
            manifest = json.loads(manifest_file.read_text())

            # 1. Dataset build with gold manifest excludes the gold source
            build_result = build_training_artifacts(
                [ROOT / "examples/compiled/86c1671e8a3a7b46.script-intelligence.v1.json"],
                ROOT / "fixtures/client.example.json",
                out_dir / "build1",
                split_salt="test-salt",
                gold_manifest=manifest,
                semantic_rules=True,
            )
            self.assertFalse(build_result["readiness"]["gates"]["semantic_reconstruction_gold_quality_verified"])

            # 2. Build with verified quality report unlocks semantic gold quality gate
            passing_report = {
                "schema_version": "1.0.0",
                "semantic_reconstruction_gold_quality_verified": True,
                "failed_gates": [],
                "gates": {"minimum_human_reviewed_sources": True},
            }
            build_result_verified = build_training_artifacts(
                [ROOT / "examples/compiled/86c1671e8a3a7b46.script-intelligence.v1.json"],
                ROOT / "fixtures/client.example.json",
                out_dir / "build2",
                split_salt="test-salt",
                semantic_rules=True,
                semantic_quality_gold_evaluated=verified_quality_report(passing_report),
            )
            self.assertTrue(build_result_verified["readiness"]["gates"]["semantic_reconstruction_gold_quality_verified"])

    def test_cli_gold_commands_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            out_dir = Path(directory)
            rec_path = ROOT / "examples/compiled/86c1671e8a3a7b46.script-intelligence.v1.json"
            client_path = ROOT / "fixtures/client.example.json"
            gates_path = ROOT / "fixtures/semantic/quality-gates.v1.json"

            # 1. gold sample
            sel_path = out_dir / "selection.json"
            ret = main(["gold", "sample", "--intelligence", str(rec_path), "--count", "1", "--output", str(sel_path)])
            self.assertEqual(ret, 0)
            self.assertTrue(sel_path.exists())

            # 2. gold review blind
            rev_path = out_dir / "review_blind.json"
            ret = main(["gold", "review", "--intelligence", str(rec_path), "--client", str(client_path), "--mode", "blind", "--output", str(rev_path)])
            self.assertEqual(ret, 0)
            self.assertTrue(rev_path.exists())

            # 3. gold annotate
            ann_path = out_dir / "annotations.json"
            ret = main(["gold", "annotate", "--record-id", REAL["record_id"], "--reviewer", "rev-1", "--mode", "blind", "--field", "topic", "--status", "value", "--value", "DOGE government workforce cuts", "--output", str(ann_path)])
            self.assertEqual(ret, 0)
            ret = main(["gold", "annotate", "--record-id", REAL["record_id"], "--reviewer", "rev-2", "--mode", "blind", "--field", "topic", "--status", "too_broad", "--notes", "needs focus", "--output", str(ann_path)])
            self.assertEqual(ret, 0)

            # 4. gold adjudicate
            adj_path = out_dir / "adjudications.json"
            ret = main(["gold", "adjudicate", "--annotations", str(ann_path), "--record-id", REAL["record_id"], "--field", "topic", "--resolver", "lead", "--status", "value", "--value", "DOGE government workforce cuts", "--notes", "lead resolution", "--output", str(adj_path)])
            self.assertEqual(ret, 0)

            # 5. gold freeze
            frozen_dir = out_dir / "frozen"
            ret = main(["gold", "freeze", "--selection", str(sel_path), "--annotations", str(ann_path), "--adjudications", str(adj_path), "--output", str(frozen_dir)])
            self.assertEqual(ret, 0)

            # 6. gold benchmark
            bench_path = out_dir / "benchmark.json"
            ret = main(["gold", "benchmark", "--intelligence", str(rec_path), "--client", str(client_path), "--annotations", str(ann_path), "--output", str(bench_path)])
            self.assertEqual(ret, 0)

            # 7. gold ablation
            ablation_path = out_dir / "ablation.json"
            ret = main(["gold", "ablation", "--intelligence", str(rec_path), "--client", str(client_path), "--annotations", str(ann_path), "--output", str(ablation_path)])
            self.assertEqual(ret, 0)

            # 8. gold contamination-test
            contam_path = out_dir / "contamination.json"
            ret = main(["gold", "contamination-test", "--intelligence", str(rec_path), "--clients", str(client_path), "--output", str(contam_path)])
            self.assertEqual(ret, 0)

            # 9. gold pilot
            pilot_path = out_dir / "pilot.json"
            ret = main(["gold", "pilot", "--intelligence", str(rec_path), "--client", str(client_path), "--limit", "1", "--output", str(pilot_path)])
            self.assertEqual(ret, 0)

            # 10. gold estimate
            est_path = out_dir / "projection.json"
            ret = main(["gold", "estimate", "--pilot", str(pilot_path), "--sources", "7500", "--input-price-per-million", "1.0", "--output-price-per-million", "2.0", "--output", str(est_path)])
            self.assertEqual(ret, 0)

            # 11. gold report
            rep_path = out_dir / "report.json"
            ret = main(["gold", "report", "--benchmark", str(bench_path), "--reviewed-sources", "1", "--gates", str(gates_path), "--output", str(rep_path)])
            self.assertEqual(ret, 0)
