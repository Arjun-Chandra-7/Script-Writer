from __future__ import annotations

import json
import unittest
from pathlib import Path

from script_writer.contracts import Evidence, EvidenceType, SourceRef, unknown


ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    def test_evidence_contract_rejects_fabricated_unknown_value(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown evidence"):
            Evidence("invented", EvidenceType.UNKNOWN)

    def test_evidence_serializes_provenance_span(self) -> None:
        value = Evidence(
            "hook",
            EvidenceType.HEURISTIC,
            confidence=0.6,
            method="test",
            sources=(SourceRef("$.semantic.sections[0]", 0, 2.1),),
        ).to_dict()
        self.assertEqual(value["sources"][0]["end_seconds"], 2.1)
        self.assertEqual(value["evidence_type"], "heuristic_inference")

    def test_unknown_has_reason_and_no_sources(self) -> None:
        value = unknown("not supplied")
        self.assertIsNone(value["value"])
        self.assertEqual(value["reason"], "not supplied")
        self.assertEqual(value["sources"], [])

    def test_all_json_schema_documents_parse(self) -> None:
        schema_files = sorted((ROOT / "schemas").glob("*.json"))
        self.assertGreaterEqual(len(schema_files), 2)
        for path in schema_files:
            schema = json.loads(path.read_text())
            self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
