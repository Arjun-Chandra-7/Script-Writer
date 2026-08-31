from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from script_writer.config import Settings
from script_writer.database import Registry
from script_writer.domain import RemoteFile
from script_writer.ingestion import IngestionService, MemorySource


def make_report(report_id: str = "report-1", content_hash: str = "video-1") -> bytes:
    return json.dumps(
        {
            "report_id": report_id,
            "source": {"content_hash": content_hash, "duration_seconds": 59.0},
            "processing": {"status": "complete", "extractor_version": "test-v1"},
            "transcript": {
                "status": "complete",
                "full_text": "A valid transcript with enough useful words.",
            },
            "training_features": {"values": {}, "provenance": {}, "excluded": {}},
        },
        sort_keys=True,
    ).encode()


def remote(file_id: str, body: bytes, name: str = "report.json") -> RemoteFile:
    return RemoteFile(
        file_id=file_id,
        name=name,
        mime_type="application/json",
        modified_time="2026-08-31T00:00:00Z",
        size=len(body),
        md5_checksum=hashlib.md5(body, usedforsecurity=False).hexdigest(),
    )


def settings(tmp_path: Path, max_bytes: int = 1024 * 1024) -> Settings:
    return Settings(
        folder_id="folder",
        credentials_file=tmp_path / "unused.json",
        state_dir=tmp_path / "state",
        max_file_bytes=max_bytes,
    )


class IngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_repeated_scan_does_not_redownload_or_readmit(self) -> None:
        body = make_report()
        source = MemorySource([(remote("drive-1", body), body)])
        registry = Registry(settings(self.tmp_path).database_path)
        registry.initialize()
        service = IngestionService(settings(self.tmp_path), registry, source)

        first = service.sync_once()
        second = service.sync_once()

        self.assertEqual(first.admitted, 1)
        self.assertEqual(second.skipped, 1)
        self.assertEqual(source.download_count, 1)
        self.assertEqual(registry.counts()["unique_reports"], 1)
        registry.close()

    def test_reupload_of_identical_bytes_is_one_report(self) -> None:
        body = make_report()
        source = MemorySource(
            [(remote("drive-1", body), body), (remote("drive-2", body, "copy.json"), body)]
        )
        registry = Registry(settings(self.tmp_path).database_path)
        registry.initialize()

        summary = IngestionService(settings(self.tmp_path), registry, source).sync_once()

        self.assertEqual(summary.admitted, 2)
        self.assertEqual(registry.counts()["unique_reports"], 1)
        registry.close()

    def test_reextracted_same_video_is_one_semantic_report(self) -> None:
        first = make_report(report_id="first", content_hash="same-video")
        changed = json.loads(first)
        changed["report_id"] = "second"
        changed["processing"]["extractor_version"] = "test-v2"
        second = json.dumps(changed, sort_keys=True).encode()
        source = MemorySource(
            [
                (remote("drive-1", first, "first.json"), first),
                (remote("drive-2", second, "second.json"), second),
            ]
        )
        registry = Registry(settings(self.tmp_path).database_path)
        registry.initialize()

        summary = IngestionService(settings(self.tmp_path), registry, source).sync_once()

        self.assertEqual(summary.admitted, 2)
        self.assertEqual(registry.counts()["unique_reports"], 1)
        links = registry.connection.execute(
            "SELECT DISTINCT report_pk FROM source_revisions"
        ).fetchall()
        self.assertEqual(len(links), 1)
        registry.close()

    def test_invalid_report_is_quarantined_and_not_retried_forever(self) -> None:
        body = b"not-json"
        source = MemorySource([(remote("drive-bad", body), body)])
        registry = Registry(settings(self.tmp_path).database_path)
        registry.initialize()
        service = IngestionService(settings(self.tmp_path), registry, source)

        self.assertEqual(service.sync_once().quarantined, 1)
        self.assertEqual(service.sync_once().skipped, 1)
        self.assertEqual(source.download_count, 1)
        registry.close()

    def test_declared_oversize_file_is_never_downloaded(self) -> None:
        body = make_report()
        item = remote("drive-large", body)
        item = RemoteFile(**{**item.__dict__, "size": 9999})
        source = MemorySource([(item, body)])
        registry = Registry(settings(self.tmp_path, max_bytes=100).database_path)
        registry.initialize()

        summary = IngestionService(
            settings(self.tmp_path, max_bytes=100), registry, source
        ).sync_once()

        self.assertEqual(summary.quarantined, 1)
        self.assertEqual(source.download_count, 0)
        registry.close()
