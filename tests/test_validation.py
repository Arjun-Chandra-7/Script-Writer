from __future__ import annotations

import json
import unittest

from script_writer.validation import ReportValidationError, assign_split, validate_report


def valid_report() -> dict:
    return {
        "report_id": "report-1",
        "source": {
            "content_hash": "video-hash-1",
            "duration_seconds": 42.5,
            "fps": 30,
            "resolution": "1080x1920",
        },
        "processing": {"status": "complete", "extractor_version": "extractor-v1"},
        "transcript": {
            "status": "complete",
            "full_text": "This is a complete short video transcript.",
            "delivery": {"word_count": 7, "overall_words_per_minute": 155},
        },
        "semantic": {"sections": [{"type": "hook", "verification_status": "unverified"}]},
        "training_features": {
            "values": {"duration": 42.5},
            "provenance": {"editing": {"verification_status": "measured"}},
            "excluded": {"semantic_sections": "unverified"},
        },
    }


class ValidationTests(unittest.TestCase):
    def test_projection_is_small_and_does_not_promote_unverified_semantics(self) -> None:
        raw = json.dumps(valid_report()).encode()
        result = validate_report(raw, split_salt="test-salt")
        projection = json.loads(result.canonical_json)

        self.assertEqual(result.quality_status, "observation_only")
        self.assertEqual(result.group_key, "video-hash-1")
        self.assertTrue(projection["transcript"]["text"].startswith("This is"))
        self.assertNotIn("semantic", projection)
        self.assertIsNone(projection["outcome"])
        self.assertIsNone(projection["rights"])

    def test_split_is_deterministic(self) -> None:
        self.assertEqual(
            assign_split("same-video", "same-salt"),
            assign_split("same-video", "same-salt"),
        )

    def test_incomplete_reports_are_rejected(self) -> None:
        report = valid_report()
        report["processing"]["status"] = "failed"

        with self.assertRaisesRegex(ReportValidationError, "processing.status"):
            validate_report(json.dumps(report).encode(), split_salt="test-salt")
