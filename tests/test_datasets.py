from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from script_writer.config import Settings
from script_writer.database import Registry
from script_writer.datasets import DatasetBuilder, DatasetNotReadyError
from script_writer.domain import RemoteFile
from script_writer.ingestion import IngestionService, MemorySource


def settings(root: Path, *, rights: bool, minimum: int = 2, maximum: int = 2) -> Settings:
    return Settings(
        folder_id="folder",
        credentials_file=root / "unused.json",
        state_dir=root / "state",
        min_new_examples=minimum,
        max_new_examples=maximum,
        rights_attested=rights,
    )


def ingest_reports(registry: Registry, config: Settings, count: int) -> None:
    entries = []
    for index in range(count):
        body = json.dumps(
            {
                "report_id": f"report-{index}",
                "source": {
                    "content_hash": f"video-{index}",
                    "duration_seconds": 40,
                },
                "processing": {
                    "status": "complete",
                    "extractor_version": "test-v1",
                },
                "transcript": {
                    "status": "complete",
                    "full_text": f"This is useful transcript number {index} for a video.",
                },
            }
        ).encode()
        item = RemoteFile(
            file_id=f"drive-{index}",
            name=f"report-{index}.json",
            mime_type="application/json",
            modified_time="2026-01-01T00:00:00Z",
            size=len(body),
            md5_checksum=f"md5-{index}",
        )
        entries.append((item, body))
    IngestionService(config, registry, MemorySource(entries)).sync_once()


class DatasetBuilderTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_rights_gate_blocks_manifest_creation(self) -> None:
        config = settings(self.root, rights=False)
        registry = Registry(config.database_path)
        registry.initialize()
        ingest_reports(registry, config, 3)

        with self.assertRaisesRegex(DatasetNotReadyError, "rights"):
            DatasetBuilder(config, registry).propose_run()
        self.assertFalse(config.manifest_dir.exists())
        registry.close()

    def test_proposal_reserves_exact_batch_and_disables_training(self) -> None:
        config = settings(self.root, rights=True)
        registry = Registry(config.database_path)
        registry.initialize()
        ingest_reports(registry, config, 3)

        proposal = DatasetBuilder(config, registry).propose_run()

        self.assertEqual(proposal.new_examples, 2)
        self.assertFalse(proposal.training_enabled)
        self.assertEqual(registry.counts()["unreserved_train_reports"], 1)
        self.assertEqual(registry.counts()["active_training_runs"], 1)
        manifest = Path(proposal.manifest_path).read_text().splitlines()
        self.assertEqual(json.loads(manifest[0])["manifest"]["new_count"], 2)
        self.assertTrue(all(json.loads(line)["role"] == "new" for line in manifest[1:]))

        with self.assertRaisesRegex(RuntimeError, "active training run"):
            DatasetBuilder(config, registry).propose_run()
        self.assertEqual(registry.counts()["unreserved_train_reports"], 1)
        registry.close()

    def test_threshold_leaves_all_examples_unreserved(self) -> None:
        config = settings(self.root, rights=True, minimum=4, maximum=5)
        registry = Registry(config.database_path)
        registry.initialize()
        ingest_reports(registry, config, 3)

        with self.assertRaisesRegex(DatasetNotReadyError, "need 4"):
            DatasetBuilder(config, registry).propose_run()
        self.assertEqual(registry.counts()["unreserved_train_reports"], 3)
        registry.close()
