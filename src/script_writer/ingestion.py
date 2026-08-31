from __future__ import annotations

import logging
from dataclasses import dataclass

from .config import Settings
from .database import Registry
from .domain import RemoteFile, RemoteSource
from .storage import FileTooLargeError, RawStore
from .validation import ReportValidationError, validate_report


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyncSummary:
    discovered: int = 0
    admitted: int = 0
    skipped: int = 0
    quarantined: int = 0
    retry: int = 0


class IngestionService:
    def __init__(self, settings: Settings, registry: Registry, source: RemoteSource):
        self.settings = settings
        self.registry = registry
        self.source = source
        self.store = RawStore(settings.raw_dir, settings.max_file_bytes)

    def sync_once(self) -> SyncSummary:
        counters = {key: 0 for key in SyncSummary.__dataclass_fields__}
        for item in self.source.list_files():
            counters["discovered"] += 1
            revision_id = self.registry.discover(item)
            if item.size is not None and item.size > self.settings.max_file_bytes:
                self.registry.mark_quarantined(
                    revision_id,
                    f"remote size {item.size} exceeds limit {self.settings.max_file_bytes}",
                )
                counters["quarantined"] += 1
                continue
            if not self.registry.claim(revision_id, self.settings.lease_seconds):
                counters["skipped"] += 1
                continue
            try:
                digest, size, path, raw = self.store.receive(
                    lambda destination, file_id=item.file_id: self.source.download(
                        file_id, destination
                    )
                )
                result = validate_report(raw, split_salt=self.settings.split_salt)
                self.registry.admit(
                    revision_id,
                    content_sha256=digest,
                    byte_size=size,
                    artifact_path=str(path),
                    result=result,
                )
            except (ReportValidationError, FileTooLargeError) as exc:
                self.registry.mark_quarantined(revision_id, str(exc))
                counters["quarantined"] += 1
                LOGGER.warning("quarantined Drive item %s: %s", item.file_id, exc)
            except Exception as exc:
                self.registry.mark_retry(revision_id, f"{type(exc).__name__}: {exc}")
                counters["retry"] += 1
                LOGGER.exception("transient ingestion failure for Drive item %s", item.file_id)
            else:
                counters["admitted"] += 1
        return SyncSummary(**counters)


class MemorySource(RemoteSource):
    """Deterministic source adapter used by tests and local dry-runs."""

    def __init__(self, entries: list[tuple[RemoteFile, bytes]]):
        self._items = {item.file_id: (item, content) for item, content in entries}
        self.download_count = 0

    def list_files(self) -> list[RemoteFile]:
        return [entry[0] for entry in self._items.values()]

    def download(self, file_id: str, destination: object) -> None:
        self.download_count += 1
        _item, content = self._items[file_id]
        destination.write(content)  # type: ignore[attr-defined]
