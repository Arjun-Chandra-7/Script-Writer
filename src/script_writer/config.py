from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_FOLDER_ID = "1loe1nchN4PFqkTzujZmbkB31E_DpCwqE"


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


@dataclass(frozen=True)
class Settings:
    folder_id: str
    credentials_file: Path
    state_dir: Path
    poll_seconds: int = 60
    max_file_bytes: int = 16 * 1024 * 1024
    lease_seconds: int = 15 * 60
    split_salt: str = "viralyst-script-writer-v1"

    @property
    def database_path(self) -> Path:
        return self.state_dir / "registry.sqlite3"

    @property
    def raw_dir(self) -> Path:
        return self.state_dir / "raw"

    @property
    def manifest_dir(self) -> Path:
        return self.state_dir / "manifests"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            folder_id=os.getenv("SCRIPT_WRITER_DRIVE_FOLDER_ID", DEFAULT_FOLDER_ID),
            credentials_file=Path(
                os.getenv(
                    "SCRIPT_WRITER_GOOGLE_CREDENTIALS",
                    "credentials/google-service-account.json",
                )
            ),
            state_dir=Path(os.getenv("SCRIPT_WRITER_STATE_DIR", "state")),
            poll_seconds=_positive_int("SCRIPT_WRITER_POLL_SECONDS", 60),
            max_file_bytes=_positive_int(
                "SCRIPT_WRITER_MAX_FILE_BYTES", 16 * 1024 * 1024
            ),
            lease_seconds=_positive_int("SCRIPT_WRITER_LEASE_SECONDS", 15 * 60),
            split_salt=os.getenv(
                "SCRIPT_WRITER_SPLIT_SALT", "viralyst-script-writer-v1"
            ),
        )
