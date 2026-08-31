from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from script_writer.config import Settings
from script_writer.corpus import CorpusIndex, SearchQuery
from script_writer.database import Registry
from script_writer.domain import RemoteFile
from script_writer.embeddings import HashingEmbeddingProvider
from script_writer.ingestion import IngestionService, MemorySource


def make_report(index: int, text: str, topic: str) -> bytes:
    return json.dumps(
        {
            "report_id": f"report-{index}",
            "source": {"content_hash": f"video-{index}", "duration_seconds": 20 + index},
            "processing": {"status": "complete", "extractor_version": "test-v1"},
            "context": {
                "platform": "instagram_reels",
                "topic": topic,
                "content_format": "talking_head",
                "audience_intent": "learn",
            },
            "transcript": {"status": "complete", "language": "en", "full_text": text},
        },
        sort_keys=True,
    ).encode()


class CorpusIndexTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.settings = Settings(
            folder_id="folder",
            credentials_file=root / "unused.json",
            state_dir=root / "state",
        )
        self.registry = Registry(self.settings.database_path)
        self.registry.initialize()
        bodies = [
            make_report(1, "What if one simple fitness habit changed your body? Try it today.", "fitness"),
            make_report(2, "Stop believing this fitness myth. Consistency beats extreme workouts.", "fitness"),
            make_report(3, "The richest founder in technology explains why focus matters.", "technology"),
        ]
        entries = [
            (
                RemoteFile(
                    file_id=f"drive-{index}",
                    name=f"report-{index}.json",
                    mime_type="application/json",
                    modified_time="2026-01-01T00:00:00Z",
                    size=len(body),
                    md5_checksum=f"md5-{index}",
                ),
                body,
            )
            for index, body in enumerate(bodies, start=1)
        ]
        IngestionService(self.settings, self.registry, MemorySource(entries)).sync_once()
        self.index = CorpusIndex(self.registry, HashingEmbeddingProvider(64))

    def tearDown(self) -> None:
        self.registry.close()
        self.temporary.cleanup()

    def test_incremental_and_deterministic_index_rebuild(self) -> None:
        first = self.index.rebuild_all()
        vectors_before = [
            row[0]
            for row in self.registry.connection.execute(
                "SELECT vector_json FROM intelligence_embeddings ORDER BY intelligence_id"
            )
        ]
        second = self.index.rebuild_all()
        forced = self.index.rebuild_all(force=True)
        vectors_after = [
            row[0]
            for row in self.registry.connection.execute(
                "SELECT vector_json FROM intelligence_embeddings ORDER BY intelligence_id"
            )
        ]

        self.assertEqual(first, {"indexed": 3, "remaining": 0})
        self.assertEqual(second, {"indexed": 0, "remaining": 0})
        self.assertEqual(forced, {"indexed": 3, "remaining": 0})
        self.assertEqual(vectors_before, vectors_after)

    def test_lexical_and_semantic_search(self) -> None:
        self.index.rebuild_all()
        hits = self.index.search(SearchQuery(text="fitness habit workout", top_k=2))
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].record["content"]["topic"]["value"], "fitness")
        self.assertGreater(hits[0].score, 0)

    def test_metadata_and_mechanism_filtering(self) -> None:
        self.index.rebuild_all()
        hits = self.index.search(
            SearchQuery(
                platform="instagram_reels",
                topics=("fitness",),
                hook_mechanisms=("question",),
                max_duration_seconds=25,
                top_k=5,
            )
        )
        self.assertEqual([hit.report_id for hit in hits], ["report-1"])

    def test_structural_similarity_excludes_source_record(self) -> None:
        self.index.rebuild_all()
        source_record = self.registry.connection.execute(
            "SELECT record_json FROM intelligence_records ORDER BY id LIMIT 1"
        ).fetchone()[0]
        record_id = json.loads(source_record)["record_id"]
        hits = self.index.structurally_similar(record_id, top_k=2)

        self.assertEqual(len(hits), 2)
        self.assertTrue(all(hit.record_id != record_id for hit in hits))
        self.assertTrue(all(hit.structural_score >= 0 for hit in hits))

    def test_exclusion_prevents_source_from_being_returned(self) -> None:
        self.index.rebuild_all()
        hits = self.index.search(
            SearchQuery(text="fitness", exclude_source_hashes=("video-1",), top_k=5)
        )
        self.assertNotIn("video-1", {hit.source_content_hash for hit in hits})
