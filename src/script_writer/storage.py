from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import BinaryIO, Callable


class FileTooLargeError(ValueError):
    pass


class _BoundedHashWriter:
    def __init__(self, target: BinaryIO, max_bytes: int):
        self.target = target
        self.max_bytes = max_bytes
        self.size = 0
        self.hasher = hashlib.sha256()

    def write(self, data: bytes) -> int:
        if self.size + len(data) > self.max_bytes:
            raise FileTooLargeError(f"download exceeds {self.max_bytes} bytes")
        written = self.target.write(data)
        self.hasher.update(data[:written])
        self.size += written
        return written

    def flush(self) -> None:
        self.target.flush()


class RawStore:
    def __init__(self, root: Path, max_file_bytes: int):
        self.root = root
        self.max_file_bytes = max_file_bytes
        root.mkdir(parents=True, exist_ok=True)

    def receive(self, producer: Callable[[BinaryIO], None]) -> tuple[str, int, Path, bytes]:
        descriptor, temporary_name = tempfile.mkstemp(prefix="download-", suffix=".part", dir=self.root)
        temporary_path = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as raw_file:
                writer = _BoundedHashWriter(raw_file, self.max_file_bytes)
                producer(writer)  # type: ignore[arg-type]
                writer.flush()
                os.fsync(raw_file.fileno())
                digest = writer.hasher.hexdigest()
                size = writer.size

            final_dir = self.root / digest[:2]
            final_dir.mkdir(parents=True, exist_ok=True)
            final_path = final_dir / f"{digest}.json"
            if final_path.exists():
                temporary_path.unlink()
            else:
                os.replace(temporary_path, final_path)
            raw = final_path.read_bytes()
            if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
                raise OSError("stored artifact failed post-write integrity check")
            return digest, size, final_path, raw
        except BaseException:
            temporary_path.unlink(missing_ok=True)
            raise
