from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .training_compiler import CompilationResult


CACHE_SCHEMA_VERSION = 1


class TrainingCompilationCache:
    """Versioned local cache for expensive per-record intent reconstruction.

    Group/split assignments are deliberately applied after cache reads because
    near-duplicate cluster membership can change with a new corpus snapshot.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS cache_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS compiled_sources (
                cache_key TEXT PRIMARY KEY,
                source_content_hash TEXT NOT NULL,
                payload_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS cache_quarantine (
                cache_key TEXT PRIMARY KEY,
                reason TEXT NOT NULL
            );
            """
        )
        row = self.connection.execute(
            "SELECT value FROM cache_meta WHERE key='schema_version'"
        ).fetchone()
        if row is not None and int(row[0]) > CACHE_SCHEMA_VERSION:
            raise RuntimeError("training cache is newer than this compiler")
        self.connection.execute(
            "INSERT OR REPLACE INTO cache_meta(key,value) VALUES('schema_version',?)",
            (str(CACHE_SCHEMA_VERSION),),
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get(self, cache_key: str) -> CompilationResult | None:
        row = self.connection.execute(
            "SELECT payload_json FROM compiled_sources WHERE cache_key=?", (cache_key,)
        ).fetchone()
        if row is None:
            return None
        try:
            value = json.loads(row[0])
            return CompilationResult(tuple(value["examples"]), tuple(value["rejections"]))
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            self.connection.execute(
                "INSERT OR REPLACE INTO cache_quarantine(cache_key,reason) VALUES(?,?)",
                (cache_key, f"corrupt cache payload: {exc}"),
            )
            self.connection.execute("DELETE FROM compiled_sources WHERE cache_key=?", (cache_key,))
            self.connection.commit()
            return None

    def put(self, cache_key: str, source_hash: str, result: CompilationResult) -> None:
        payload = json.dumps(
            {"examples": result.examples, "rejections": result.rejections},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        self.connection.execute(
            "INSERT OR REPLACE INTO compiled_sources(cache_key,source_content_hash,payload_json) VALUES(?,?,?)",
            (cache_key, source_hash, payload),
        )
        self.connection.commit()
