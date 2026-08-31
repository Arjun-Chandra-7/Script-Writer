from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO


@dataclass(frozen=True)
class RemoteFile:
    file_id: str
    name: str
    mime_type: str
    modified_time: str
    size: int | None
    md5_checksum: str | None

    @property
    def revision_key(self) -> str:
        if self.md5_checksum:
            return f"md5:{self.md5_checksum}"
        return f"meta:{self.modified_time}:{self.size if self.size is not None else 'unknown'}"


@dataclass(frozen=True)
class ValidationResult:
    report_id: str
    source_content_hash: str
    group_key: str
    split: str
    quality_status: str
    extractor_version: str
    transcript_sha256: str
    canonical_json: str


class RemoteSource:
    """Protocol-like base class kept dependency-free for tests and adapters."""

    def list_files(self) -> list[RemoteFile]:
        raise NotImplementedError

    def download(self, file_id: str, destination: BinaryIO) -> None:
        raise NotImplementedError
