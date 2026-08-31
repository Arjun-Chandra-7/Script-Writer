from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from script_writer.database import Registry
from script_writer.outcomes import (
    OutcomeValidationError,
    canonical_outcome_json,
    validate_outcome_record,
)


def outcome() -> dict:
    return {
        "schema_version": "1.0.0",
        "report_id": "report-1",
        "platform": "instagram_reels",
        "account_context_id": "client-account-1",
        "published_at": "2026-08-01T12:00:00Z",
        "measured_at": "2026-08-08T12:00:00Z",
        "measurement_window_hours": 168,
        "metrics": {
            "views": 10000,
            "reach": 8000,
            "average_watch_time_seconds": 18.2,
            "shares": 220,
            "saves": 130,
        },
        "cohort_normalization": None,
        "rights": {"training_allowed": True, "basis": "client_owned"},
    }


class OutcomeTests(unittest.TestCase):
    def test_instagram_outcome_without_normalization_is_valid_content_attachment(self) -> None:
        record = outcome()
        self.assertIs(validate_outcome_record(record), record)
        self.assertIn('"instagram_reels"', canonical_outcome_json(record))

    def test_raw_views_do_not_create_an_implicit_performance_score(self) -> None:
        canonical = json.loads(canonical_outcome_json(outcome()))
        self.assertNotIn("performance_score", canonical)
        self.assertIsNone(canonical["cohort_normalization"])

    def test_invalid_rate_and_time_order_are_rejected(self) -> None:
        record = outcome()
        record["metrics"]["completion_rate"] = 1.2
        with self.assertRaisesRegex(OutcomeValidationError, "between 0 and 1"):
            validate_outcome_record(record)

        record = outcome()
        record["measured_at"] = "2026-07-01T00:00:00Z"
        with self.assertRaisesRegex(OutcomeValidationError, "cannot precede"):
            validate_outcome_record(record)

    def test_outcome_storage_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            registry = Registry(Path(directory) / "registry.sqlite3")
            registry.initialize()
            record = outcome()
            canonical = canonical_outcome_json(record)
            self.assertTrue(registry.save_outcome(record, canonical))
            self.assertFalse(registry.save_outcome(record, canonical))
            self.assertEqual(
                registry.connection.execute("SELECT COUNT(*) FROM outcome_records").fetchone()[0],
                1,
            )
            registry.close()
