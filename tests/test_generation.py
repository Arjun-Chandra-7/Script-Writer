from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from script_writer.config import Settings
from script_writer.contracts import validate_generation_result
from script_writer.corpus import CorpusIndex
from script_writer.database import Registry
from script_writer.domain import RemoteFile
from script_writer.generation import (
    DeterministicOutlineGenerator,
    GenerationRequest,
    RetrievalFirstScriptWriter,
)
from script_writer.ingestion import IngestionService, MemorySource


class GenerationPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        settings = Settings(
            folder_id="folder",
            credentials_file=root / "unused.json",
            state_dir=root / "state",
        )
        self.registry = Registry(settings.database_path)
        self.registry.initialize()
        body = json.dumps(
            {
                "report_id": "fitness-example",
                "source": {"content_hash": "fitness-video", "duration_seconds": 30},
                "processing": {"status": "complete", "extractor_version": "test-v1"},
                "context": {"platform": "instagram_reels", "topic": "fitness"},
                "transcript": {
                    "status": "complete",
                    "language": "en",
                    "full_text": "What if one consistent habit made fitness simpler? Start with a daily walk.",
                },
            }
        ).encode()
        source = MemorySource(
            [
                (
                    RemoteFile(
                        file_id="drive-1",
                        name="one.json",
                        mime_type="application/json",
                        modified_time="2026-01-01T00:00:00Z",
                        size=len(body),
                        md5_checksum="one",
                    ),
                    body,
                )
            ]
        )
        IngestionService(settings, self.registry, source).sync_once()
        self.corpus = CorpusIndex(self.registry)
        self.corpus.rebuild_all()
        self.writer = RetrievalFirstScriptWriter(
            self.corpus, DeterministicOutlineGenerator(), retrieval_count=3
        )

    def tearDown(self) -> None:
        self.registry.close()
        self.temporary.cleanup()

    def request(self, **changes: object) -> GenerationRequest:
        values = {
            "client_context_id": "client-1",
            "niche": "fitness",
            "topic": "daily walking",
            "objective": "help beginners start exercising consistently",
            "audience": "busy beginners",
            "desired_duration_seconds": 30,
            "platform": "instagram_reels",
            "factual_context": ("The client recommends starting with ten minutes.",),
        }
        values.update(changes)
        return GenerationRequest(**values)  # type: ignore[arg-type]

    def test_retrieval_first_result_is_structured_and_cites_evidence(self) -> None:
        result = self.writer.generate(self.request())
        validate_generation_result(result)

        self.assertEqual(result["generator_version"], "deterministic-outline-1.0.0")
        self.assertEqual(len(result["retrieved_evidence"]), 1)
        self.assertEqual(result["retrieved_evidence"][0]["report_id"], "fitness-example")
        self.assertEqual(len(result["sections"]), 4)
        self.assertEqual(
            result["claims_requiring_verification"],
            ["The client recommends starting with ten minutes."],
        )

    def test_context_contains_analysis_only_evidence_and_anti_copy_instruction(self) -> None:
        request = self.request()
        query = self.writer.build_query(request)
        hits = self.corpus.search(query)
        context = self.writer.build_context(request, hits, query)

        self.assertIn("never as copy templates", context.instructions[0])
        self.assertEqual(context.evidence[0].source_content_hash, "fitness-video")
        self.assertIn("question", context.evidence[0].hook_mechanisms)

    def test_inspiration_record_shapes_structural_query(self) -> None:
        record_id = self.corpus.registry.connection.execute(
            "SELECT json_extract(record_json, '$.record_id') FROM intelligence_records"
        ).fetchone()[0]
        query = self.writer.build_query(
            self.request(inspiration_record_ids=(record_id,))
        )
        self.assertTrue(query.structural_fingerprint)

    def test_banned_pattern_is_enforced_after_generation(self) -> None:
        with self.assertRaisesRegex(ValueError, "banned patterns"):
            self.writer.generate(self.request(banned_patterns=("here is a focused way",)))

    def test_invalid_request_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive"):
            self.request(desired_duration_seconds=0)
