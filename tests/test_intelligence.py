from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from script_writer.contracts import validate_script_intelligence_record
from script_writer.intelligence import ScriptIntelligenceCompiler
from script_writer.validation import parse_and_validate_report


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "86c1671e8a3a7b46-The-Godfather-of-AI-on-his-Feud-with-Elon-Musk---TVO-Today--1080p.json"


def minimal_report() -> dict:
    return {
        "report_id": "minimal-1",
        "source": {"content_hash": "source-minimal", "duration_seconds": 10.0},
        "processing": {"status": "complete", "extractor_version": "test-v1"},
        "transcript": {
            "status": "complete",
            "language": "en",
            "full_text": "What if this simple habit changed everything? Try it today.",
        },
    }


class FailingAnalyzer:
    version = "always-fails-v1"

    def analyze(self, _context: object) -> object:
        raise RuntimeError("offline analyzer unavailable")


class IntelligenceCompilerTests(unittest.TestCase):
    def test_minimal_partial_report_compiles_with_explicit_unknowns(self) -> None:
        result = ScriptIntelligenceCompiler().compile(
            minimal_report(), artifact_sha256="a" * 64
        )
        validate_script_intelligence_record(result.record)

        self.assertEqual(result.record["identity"]["platform"]["evidence_type"], "unknown")
        self.assertEqual(result.record["content"]["topic"]["evidence_type"], "unknown")
        self.assertEqual(result.record["delivery"]["pauses"]["evidence_type"], "unknown")
        self.assertEqual(result.record["script_edit_relationships"], [])

    def test_compilation_is_byte_deterministic(self) -> None:
        compiler = ScriptIntelligenceCompiler()
        first = compiler.compile(minimal_report(), artifact_sha256="a" * 64)
        second = compiler.compile(minimal_report(), artifact_sha256="a" * 64)
        self.assertEqual(first.canonical_json, second.canonical_json)
        self.assertEqual(first.sha256, second.sha256)

    def test_semantic_failure_falls_back_without_losing_deterministic_features(self) -> None:
        result = ScriptIntelligenceCompiler(FailingAnalyzer()).compile(
            minimal_report(), artifact_sha256="b" * 64
        )
        self.assertEqual(result.record["content"]["spoken_word_count"]["value"], 10)
        self.assertEqual(result.record["content"]["topic"]["evidence_type"], "unknown")
        self.assertTrue(
            any("failed gracefully" in warning for warning in result.record["quality"]["warnings"])
        )

    def test_real_extractor_report_compiles_and_preserves_uncertainty(self) -> None:
        raw = SAMPLE.read_bytes()
        report, _ = parse_and_validate_report(raw, split_salt="integration")
        result = ScriptIntelligenceCompiler().compile(
            report, artifact_sha256=hashlib.sha256(raw).hexdigest()
        )
        mechanisms = {
            item["mechanism"] for item in result.record["hook_intelligence"]["mechanisms"]
        }
        devices = {item["device"] for item in result.record["retention_devices"]}

        self.assertEqual(result.record["content"]["spoken_word_count"]["value"], 166)
        self.assertAlmostEqual(
            result.record["content"]["words_per_second"]["value"], 2.8079, places=4
        )
        self.assertIn("question", mechanisms)
        self.assertIn("contrast", devices)
        self.assertEqual(
            result.record["script_structure"]["hook"]["evidence_type"],
            "heuristic_inference",
        )
        self.assertFalse(result.record["quality"]["outcome_evidence_available"])
        self.assertLess(len(result.canonical_json), len(raw) // 10)

    def test_normalized_transcript_is_not_the_raw_report(self) -> None:
        result = ScriptIntelligenceCompiler().compile(
            minimal_report(), artifact_sha256="c" * 64
        )
        projection = result.record["index_projections"]
        self.assertIn("simple habit", projection["semantic_text"].lower())
        self.assertNotIn("training_features", projection["semantic_text"])
