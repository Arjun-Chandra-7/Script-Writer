from __future__ import annotations

import sqlite3


CURRENT_DATABASE_VERSION = 2


MIGRATION_2 = """
CREATE TABLE IF NOT EXISTS intelligence_records (
    id INTEGER PRIMARY KEY,
    report_pk INTEGER NOT NULL UNIQUE,
    schema_version TEXT NOT NULL,
    compiler_version TEXT NOT NULL,
    analyzer_version TEXT NOT NULL,
    source_artifact_sha256 TEXT NOT NULL,
    record_sha256 TEXT NOT NULL,
    record_json TEXT NOT NULL,
    search_text TEXT NOT NULL,
    structural_fingerprint TEXT NOT NULL,
    platform TEXT,
    content_format TEXT,
    topic TEXT,
    duration_seconds REAL NOT NULL,
    compile_status TEXT NOT NULL CHECK (compile_status IN ('ready', 'failed')),
    last_error TEXT,
    compiled_at TEXT NOT NULL,
    FOREIGN KEY(report_pk) REFERENCES reports(id)
);

CREATE INDEX IF NOT EXISTS idx_intelligence_filter
ON intelligence_records(platform, content_format, topic, duration_seconds);
CREATE INDEX IF NOT EXISTS idx_intelligence_source
ON intelligence_records(source_artifact_sha256, compiler_version, analyzer_version);

CREATE TABLE IF NOT EXISTS intelligence_mechanisms (
    intelligence_id INTEGER NOT NULL,
    category TEXT NOT NULL,
    mechanism TEXT NOT NULL,
    start_seconds REAL,
    end_seconds REAL,
    confidence REAL,
    evidence_type TEXT NOT NULL,
    PRIMARY KEY(intelligence_id, category, mechanism, start_seconds, end_seconds),
    FOREIGN KEY(intelligence_id) REFERENCES intelligence_records(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_mechanism_lookup
ON intelligence_mechanisms(category, mechanism);

CREATE TABLE IF NOT EXISTS intelligence_embeddings (
    intelligence_id INTEGER NOT NULL,
    projection TEXT NOT NULL,
    provider_version TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector_json TEXT NOT NULL,
    PRIMARY KEY(intelligence_id, projection, provider_version),
    FOREIGN KEY(intelligence_id) REFERENCES intelligence_records(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS intelligence_compile_attempts (
    report_pk INTEGER PRIMARY KEY,
    status TEXT NOT NULL CHECK (status IN ('pending', 'ready', 'failed')),
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(report_pk) REFERENCES reports(id) ON DELETE CASCADE
);

CREATE VIRTUAL TABLE IF NOT EXISTS intelligence_fts USING fts5(
    intelligence_id UNINDEXED,
    search_text,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS outcome_records (
    id INTEGER PRIMARY KEY,
    report_id TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    platform TEXT NOT NULL,
    account_context_id TEXT NOT NULL,
    cohort_key TEXT,
    measured_at TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    rights_allowed INTEGER NOT NULL CHECK (rights_allowed IN (0, 1)),
    created_at TEXT NOT NULL,
    UNIQUE(report_id, platform, account_context_id, measured_at)
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    id INTEGER PRIMARY KEY,
    evaluation_id TEXT NOT NULL UNIQUE,
    evaluator_version TEXT NOT NULL,
    candidate_version TEXT NOT NULL,
    baseline_version TEXT,
    fixture_set_version TEXT NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def migrate(connection: sqlite3.Connection) -> None:
    row = connection.execute(
        "SELECT value FROM schema_meta WHERE key = 'version'"
    ).fetchone()
    version = int(row[0]) if row else 1
    if version > CURRENT_DATABASE_VERSION:
        raise RuntimeError(
            f"database version {version} is newer than supported {CURRENT_DATABASE_VERSION}"
        )
    if version < 2:
        connection.executescript(MIGRATION_2)
        connection.execute(
            """
            INSERT INTO schema_meta(key, value) VALUES ('version', '2')
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """
        )
