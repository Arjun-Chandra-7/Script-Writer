from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from script_writer.contracts import validate_generation_request
from script_writer.database import Registry
from script_writer.evaluation import (
    OfflineEvaluator,
    longest_common_token_phrase,
    ngram_overlap_share,
)


ROOT = Path(__file__).resolve().parents[1]


def request() -> dict:
    return {
        "contract_version": "1.0.0",
        "client_context_id": "client-1",
        "niche": "fitness",
        "topic": "walking",
        "objective": "start a consistent habit",
        "audience": "busy beginners",
        "desired_duration_seconds": 20,
        "platform": "instagram_reels",
        "banned_patterns": ["secret hack"],
        "desired_cta": "Walk today",
    }


def result(spoken_script: str) -> dict:
    return {
        "contract_version": "1.0.0",
        "generator_version": "test-v1",
        "spoken_script": spoken_script,
        "hook": "Busy beginners, start smaller.",
        "sections": [
            {"role": "hook", "start_seconds": 0, "end_seconds": 3, "text": "Busy beginners, start smaller."},
            {"role": "body", "start_seconds": 3, "end_seconds": 16, "text": "Build the habit before adding intensity."},
            {"role": "cta", "start_seconds": 16, "end_seconds": 20, "text": "Walk today."},
        ],
        "on_screen_text_suggestions": [],
        "delivery_notes": [],
        "visual_cues": [],
        "cta": "Walk today.",
        "claims_requiring_verification": [],
        "creative_mechanisms": ["identity_callout"],
        "retrieved_evidence": [
            {
                "record_id": "sir:source:1.0.0",
                "report_id": "source-report",
                "source_content_hash": "source",
                "relevance_score": 0.8,
            }
        ],
        "rationale": [],
    }


def source_record() -> dict:
    return {
        "record_id": "sir:source:1.0.0",
        "content": {
            "clean_transcript": {
                "value": "Start with a ten minute walk every morning and keep the promise to yourself."
            }
        },
    }


class EvaluationTests(unittest.TestCase):
    def test_phrase_and_ngram_overlap_are_deterministic(self) -> None:
        candidate = "Take a ten minute walk every morning and then add more."
        source = "Start with a ten minute walk every morning and keep going."
        phrase = longest_common_token_phrase(candidate, source)
        self.assertEqual(phrase["token_count"], 7)
        self.assertGreater(ngram_overlap_share(candidate, source), 0)

    def test_copying_source_fails_similarity_guard(self) -> None:
        evaluation = OfflineEvaluator().evaluate(
            request(),
            result("Start with a ten minute walk every morning and keep the promise to yourself."),
            source_records=[source_record()],
            candidate_version="candidate-copy",
            fixture_set_version="test-v1",
        )
        metrics = {item["name"]: item for item in evaluation["deterministic_metrics"]}
        self.assertEqual(metrics["corpus_similarity_guard"]["status"], "fail")
        self.assertGreaterEqual(
            metrics["corpus_similarity_guard"]["value"]["max_contiguous_tokens"], 8
        )

    def test_human_dimensions_remain_explicitly_unscored(self) -> None:
        evaluation = OfflineEvaluator().evaluate(
            request(),
            result("Busy beginners can begin small. Build consistency gently. Walk today."),
            source_records=[source_record()],
            candidate_version="candidate-original",
            fixture_set_version="test-v1",
        )
        self.assertEqual(evaluation["judgment_dimensions"]["hook_quality"]["status"], "not_evaluated")
        self.assertIsNone(evaluation["judgment_dimensions"]["hook_quality"]["score"])
        self.assertEqual(evaluation["summary"]["promotion_decision"], "not_available_offline")

    def test_regression_comparison_has_no_fake_aggregate(self) -> None:
        evaluator = OfflineEvaluator()
        baseline = evaluator.evaluate(
            request(),
            result("Busy beginners can begin small. Build consistency gently. Walk today."),
            source_records=[source_record()],
            candidate_version="baseline",
            fixture_set_version="test-v1",
        )
        candidate_result = result("Use this secret hack. Walk today.")
        candidate = evaluator.evaluate(
            request(),
            candidate_result,
            source_records=[source_record()],
            candidate_version="candidate",
            fixture_set_version="test-v1",
            baseline_version="baseline",
        )
        comparison = evaluator.compare(candidate, baseline)
        self.assertIsNone(comparison["aggregate_score"])
        self.assertIn(
            "banned_pattern_compliance",
            {change["metric"] for change in comparison["changes"]},
        )

    def test_evaluation_storage_is_idempotent(self) -> None:
        evaluation = OfflineEvaluator().evaluate(
            request(),
            result("Busy beginners can begin small. Build consistency gently. Walk today."),
            source_records=[source_record()],
            candidate_version="candidate",
            fixture_set_version="test-v1",
        )
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(Path(directory) / "registry.sqlite3")
            registry.initialize()
            self.assertTrue(registry.save_evaluation(evaluation))
            self.assertFalse(registry.save_evaluation(evaluation))
            registry.close()

    def test_regression_fixture_requests_validate(self) -> None:
        fixture = json.loads((ROOT / "fixtures/evaluation/requests.v1.json").read_text())
        self.assertEqual(fixture["fixture_set_version"], "requests-v1")
        for item in fixture["fixtures"]:
            request_data = {key: value for key, value in item.items() if key != "id"}
            validate_generation_request(request_data)
