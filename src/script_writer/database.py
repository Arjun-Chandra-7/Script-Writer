from __future__ import annotations

import contextlib
import hashlib
import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

from .domain import RemoteFile, ValidationResult
from .migrations import migrate


SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS source_revisions (
    id INTEGER PRIMARY KEY,
    drive_file_id TEXT NOT NULL,
    revision_key TEXT NOT NULL,
    name TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    modified_time TEXT NOT NULL,
    remote_size INTEGER,
    remote_md5 TEXT,
    state TEXT NOT NULL CHECK (state IN (
        'discovered', 'downloading', 'downloaded', 'admitted', 'quarantined', 'retry'
    )),
    attempts INTEGER NOT NULL DEFAULT 0,
    lease_until TEXT,
    content_sha256 TEXT,
    artifact_path TEXT,
    report_pk INTEGER,
    last_error TEXT,
    discovered_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(drive_file_id, revision_key),
    FOREIGN KEY(report_pk) REFERENCES reports(id)
);

CREATE INDEX IF NOT EXISTS idx_source_revisions_state
ON source_revisions(state, lease_until);

CREATE TABLE IF NOT EXISTS artifacts (
    sha256 TEXT PRIMARY KEY,
    byte_size INTEGER NOT NULL,
    path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY,
    artifact_sha256 TEXT NOT NULL UNIQUE,
    report_id TEXT NOT NULL,
    source_content_hash TEXT NOT NULL,
    group_key TEXT NOT NULL,
    split TEXT NOT NULL CHECK (split IN ('train', 'validation', 'test')),
    quality_status TEXT NOT NULL,
    extractor_version TEXT NOT NULL,
    transcript_sha256 TEXT NOT NULL,
    canonical_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY(artifact_sha256) REFERENCES artifacts(sha256)
);

CREATE INDEX IF NOT EXISTS idx_reports_group ON reports(group_key);
CREATE INDEX IF NOT EXISTS idx_reports_split_quality ON reports(split, quality_status);
CREATE UNIQUE INDEX IF NOT EXISTS unique_report_per_source_content
ON reports(source_content_hash);

CREATE TABLE IF NOT EXISTS outcomes (
    report_id TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    rights_allowed INTEGER NOT NULL CHECK (rights_allowed IN (0, 1)),
    measured_at TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_versions (
    id INTEGER PRIMARY KEY,
    version TEXT NOT NULL UNIQUE,
    manifest_sha256 TEXT NOT NULL UNIQUE,
    manifest_path TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('building', 'ready', 'failed')),
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS dataset_members (
    dataset_id INTEGER NOT NULL,
    report_pk INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('new', 'replay', 'evaluation')),
    PRIMARY KEY(dataset_id, report_pk),
    FOREIGN KEY(dataset_id) REFERENCES dataset_versions(id),
    FOREIGN KEY(report_pk) REFERENCES reports(id)
);

CREATE TABLE IF NOT EXISTS training_runs (
    id INTEGER PRIMARY KEY,
    run_id TEXT NOT NULL UNIQUE,
    dataset_id INTEGER NOT NULL UNIQUE,
    state TEXT NOT NULL CHECK (state IN (
        'queued', 'preparing', 'running', 'evaluating', 'promotable',
        'promoted', 'failed', 'abandoned'
    )),
    training_enabled INTEGER NOT NULL DEFAULT 0 CHECK (training_enabled = 0),
    base_model TEXT,
    base_revision TEXT,
    config_sha256 TEXT NOT NULL,
    checkpoint_path TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(dataset_id) REFERENCES dataset_versions(id)
);

CREATE UNIQUE INDEX IF NOT EXISTS one_active_training_run
ON training_runs((1))
WHERE state IN ('queued', 'preparing', 'running', 'evaluating');

-- Failed runs retain their reservations for checkpoint resume; promotable runs
-- retain theirs until the evaluated model is promoted or explicitly rejected.
CREATE UNIQUE INDEX IF NOT EXISTS one_unresolved_training_run
ON training_runs((1))
WHERE state IN (
    'queued', 'preparing', 'running', 'evaluating', 'promotable', 'failed'
);

CREATE TABLE IF NOT EXISTS example_reservations (
    report_pk INTEGER PRIMARY KEY,
    run_pk INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('reserved', 'consumed')),
    created_at TEXT NOT NULL,
    FOREIGN KEY(report_pk) REFERENCES reports(id),
    FOREIGN KEY(run_pk) REFERENCES training_runs(id)
);
"""


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


class Registry:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=30, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")

    def close(self) -> None:
        self.connection.close()

    def initialize(self) -> None:
        self.connection.executescript(SCHEMA)
        self.connection.execute(
            "INSERT OR IGNORE INTO schema_meta(key, value) VALUES ('version', '1')"
        )
        migrate(self.connection)

    @contextlib.contextmanager
    def transaction(self, *, immediate: bool = False) -> Iterator[sqlite3.Connection]:
        self.connection.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
        try:
            yield self.connection
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def discover(self, item: RemoteFile) -> int:
        now = utc_now()
        self.connection.execute(
            """
            INSERT INTO source_revisions(
                drive_file_id, revision_key, name, mime_type, modified_time,
                remote_size, remote_md5, state, discovered_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'discovered', ?, ?)
            ON CONFLICT(drive_file_id, revision_key) DO UPDATE SET
                name = excluded.name,
                modified_time = excluded.modified_time,
                updated_at = excluded.updated_at
            """,
            (
                item.file_id,
                item.revision_key,
                item.name,
                item.mime_type,
                item.modified_time,
                item.size,
                item.md5_checksum,
                now,
                now,
            ),
        )
        row = self.connection.execute(
            "SELECT id FROM source_revisions WHERE drive_file_id = ? AND revision_key = ?",
            (item.file_id, item.revision_key),
        ).fetchone()
        assert row is not None
        return int(row["id"])

    def claim(self, revision_id: int, lease_seconds: int) -> bool:
        now_dt = datetime.now(UTC)
        now = now_dt.isoformat()
        lease = (now_dt + timedelta(seconds=lease_seconds)).isoformat()
        with self.transaction(immediate=True) as connection:
            changed = connection.execute(
                """
                UPDATE source_revisions
                SET state = 'downloading', attempts = attempts + 1,
                    lease_until = ?, updated_at = ?, last_error = NULL
                WHERE id = ? AND (
                    state IN ('discovered', 'retry')
                    OR (state = 'downloading' AND lease_until < ?)
                )
                """,
                (lease, now, revision_id, now),
            ).rowcount
        return changed == 1

    def mark_retry(self, revision_id: int, error: str) -> None:
        self.connection.execute(
            """
            UPDATE source_revisions
            SET state = 'retry', lease_until = NULL, last_error = ?, updated_at = ?
            WHERE id = ?
            """,
            (error[:2000], utc_now(), revision_id),
        )

    def mark_quarantined(self, revision_id: int, error: str) -> None:
        self.connection.execute(
            """
            UPDATE source_revisions
            SET state = 'quarantined', lease_until = NULL, last_error = ?, updated_at = ?
            WHERE id = ?
            """,
            (error[:2000], utc_now(), revision_id),
        )

    def admit(
        self,
        revision_id: int,
        *,
        content_sha256: str,
        byte_size: int,
        artifact_path: str,
        result: ValidationResult,
    ) -> int:
        now = utc_now()
        with self.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO artifacts(sha256, byte_size, path, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (content_sha256, byte_size, artifact_path, now),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO reports(
                    artifact_sha256, report_id, source_content_hash, group_key,
                    split, quality_status, extractor_version, transcript_sha256,
                    canonical_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    content_sha256,
                    result.report_id,
                    result.source_content_hash,
                    result.group_key,
                    result.split,
                    result.quality_status,
                    result.extractor_version,
                    result.transcript_sha256,
                    result.canonical_json,
                    now,
                ),
            )
            report = connection.execute(
                """
                SELECT id FROM reports
                WHERE artifact_sha256 = ? OR source_content_hash = ?
                ORDER BY CASE WHEN artifact_sha256 = ? THEN 0 ELSE 1 END
                LIMIT 1
                """,
                (content_sha256, result.source_content_hash, content_sha256),
            ).fetchone()
            assert report is not None
            report_pk = int(report["id"])
            connection.execute(
                """
                UPDATE source_revisions
                SET state = 'admitted', lease_until = NULL, content_sha256 = ?,
                    artifact_path = ?, report_pk = ?, updated_at = ?
                WHERE id = ?
                """,
                (content_sha256, artifact_path, report_pk, now, revision_id),
            )
        return report_pk

    def revision_state(self, revision_id: int) -> str:
        row = self.connection.execute(
            "SELECT state FROM source_revisions WHERE id = ?", (revision_id,)
        ).fetchone()
        if row is None:
            raise KeyError(revision_id)
        return str(row["state"])

    def report_artifact_sha256(self, report_pk: int) -> str:
        row = self.connection.execute(
            "SELECT artifact_sha256 FROM reports WHERE id = ?", (report_pk,)
        ).fetchone()
        if row is None:
            raise KeyError(report_pk)
        return str(row["artifact_sha256"])

    def counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT state, COUNT(*) AS count FROM source_revisions GROUP BY state"
        ).fetchall()
        result = {str(row["state"]): int(row["count"]) for row in rows}
        result["unique_reports"] = int(
            self.connection.execute("SELECT COUNT(*) FROM reports").fetchone()[0]
        )
        result["unreserved_train_reports"] = int(
            self.connection.execute(
                """
                SELECT COUNT(*) FROM reports r
                LEFT JOIN example_reservations er ON er.report_pk = r.id
                WHERE r.split = 'train' AND er.report_pk IS NULL
                """
            ).fetchone()[0]
        )
        result["active_training_runs"] = int(
            self.connection.execute(
                """
                SELECT COUNT(*) FROM training_runs
                WHERE state IN (
                    'queued', 'preparing', 'running', 'evaluating', 'promotable', 'failed'
                )
                """
            ).fetchone()[0]
        )
        return result

    def select_new_training_reports(self, limit: int) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT r.* FROM reports r
            LEFT JOIN example_reservations er ON er.report_pk = r.id
            WHERE r.split = 'train' AND er.report_pk IS NULL
            ORDER BY r.id
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    def active_run_id(self) -> str | None:
        row = self.connection.execute(
            """
            SELECT run_id FROM training_runs
            WHERE state IN (
                'queued', 'preparing', 'running', 'evaluating', 'promotable', 'failed'
            )
            ORDER BY id DESC LIMIT 1
            """
        ).fetchone()
        return None if row is None else str(row["run_id"])

    def select_replay_reports(self, limit: int, seed: str) -> list[sqlite3.Row]:
        if limit <= 0:
            return []
        rows = self.connection.execute(
            """
            SELECT DISTINCT r.* FROM reports r
            JOIN example_reservations er ON er.report_pk = r.id
            WHERE r.split = 'train' AND er.status = 'consumed'
            """
        ).fetchall()
        return sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{seed}:{row['artifact_sha256']}".encode()
            ).digest(),
        )[:limit]

    def create_queued_run(
        self,
        *,
        version: str,
        manifest_sha256: str,
        manifest_path: str,
        config_sha256: str,
        new_report_ids: list[int],
        replay_report_ids: list[int],
        evaluation_report_ids: list[int],
    ) -> str:
        """Atomically register an immutable snapshot and reserve its new examples."""
        now = utc_now()
        run_id = f"run-{version}"
        with self.transaction(immediate=True) as connection:
            active = connection.execute(
                """
                SELECT run_id FROM training_runs
                WHERE state IN (
                    'queued', 'preparing', 'running', 'evaluating', 'promotable', 'failed'
                )
                """
            ).fetchone()
            if active is not None:
                raise RuntimeError(f"active training run already exists: {active['run_id']}")
            connection.execute(
                """
                INSERT INTO dataset_versions(
                    version, manifest_sha256, manifest_path, status, created_at
                ) VALUES (?, ?, ?, 'ready', ?)
                """,
                (version, manifest_sha256, manifest_path, now),
            )
            dataset_id = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
            connection.execute(
                """
                INSERT INTO training_runs(
                    run_id, dataset_id, state, training_enabled, config_sha256,
                    created_at, updated_at
                ) VALUES (?, ?, 'queued', 0, ?, ?, ?)
                """,
                (run_id, dataset_id, config_sha256, now, now),
            )
            run_pk = int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])
            members = [
                *((dataset_id, report_id, "new") for report_id in new_report_ids),
                *((dataset_id, report_id, "replay") for report_id in replay_report_ids),
                *((dataset_id, report_id, "evaluation") for report_id in evaluation_report_ids),
            ]
            connection.executemany(
                "INSERT INTO dataset_members(dataset_id, report_pk, role) VALUES (?, ?, ?)",
                members,
            )
            connection.executemany(
                """
                INSERT INTO example_reservations(report_pk, run_pk, status, created_at)
                VALUES (?, ?, 'reserved', ?)
                """,
                [(report_id, run_pk, now) for report_id in new_report_ids],
            )
        return run_id

    def run_details(self) -> list[dict[str, object]]:
        rows = self.connection.execute(
            """
            SELECT tr.run_id, tr.state, tr.training_enabled, tr.created_at,
                   dv.version, dv.manifest_sha256, dv.manifest_path,
                   SUM(CASE WHEN dm.role = 'new' THEN 1 ELSE 0 END) AS new_count,
                   SUM(CASE WHEN dm.role = 'replay' THEN 1 ELSE 0 END) AS replay_count,
                   SUM(CASE WHEN dm.role = 'evaluation' THEN 1 ELSE 0 END) AS eval_count
            FROM training_runs tr
            JOIN dataset_versions dv ON dv.id = tr.dataset_id
            JOIN dataset_members dm ON dm.dataset_id = dv.id
            GROUP BY tr.id
            ORDER BY tr.id DESC
            """
        ).fetchall()
        return [dict(row) for row in rows]

    def pending_intelligence_reports(
        self, *, compiler_version: str, analyzer_version: str, limit: int
    ) -> list[sqlite3.Row]:
        return self.connection.execute(
            """
            SELECT r.id AS report_pk, r.artifact_sha256, a.path AS artifact_path
            FROM reports r
            JOIN artifacts a ON a.sha256 = r.artifact_sha256
            LEFT JOIN intelligence_records ir ON ir.report_pk = r.id
            WHERE ir.id IS NULL
               OR ir.compiler_version != ?
               OR ir.analyzer_version != ?
               OR ir.source_artifact_sha256 != r.artifact_sha256
            ORDER BY r.id
            LIMIT ?
            """,
            (compiler_version, analyzer_version, limit),
        ).fetchall()

    def save_intelligence(
        self,
        *,
        report_pk: int,
        record: dict[str, object],
        canonical_json: str,
        record_sha256: str,
    ) -> bool:
        compiler = record["compiler"]
        identity = record["identity"]
        content = record["content"]
        projections = record["index_projections"]
        assert isinstance(compiler, dict)
        assert isinstance(identity, dict)
        assert isinstance(content, dict)
        assert isinstance(projections, dict)
        compiler_version = str(compiler["version"])
        analyzer_version = str(compiler["semantic_analyzer_version"])
        source_sha = str(identity["source_artifact_sha256"]["value"])
        existing = self.connection.execute(
            "SELECT record_sha256 FROM intelligence_records WHERE report_pk = ?",
            (report_pk,),
        ).fetchone()
        if existing is not None and existing["record_sha256"] == record_sha256:
            return False

        def evidence_value(node: object) -> object | None:
            return node.get("value") if isinstance(node, dict) else None

        platform = evidence_value(identity.get("platform"))
        content_format = evidence_value(content.get("content_format"))
        topic = evidence_value(content.get("topic"))
        duration = evidence_value(content.get("video_duration_seconds"))
        search_text = str(projections["semantic_text"])
        fingerprint = json.dumps(projections["structural_fingerprint"], separators=(",", ":"))
        now = utc_now()
        with self.transaction(immediate=True) as connection:
            connection.execute(
                """
                INSERT INTO intelligence_records(
                    report_pk, schema_version, compiler_version, analyzer_version,
                    source_artifact_sha256, record_sha256, record_json, search_text,
                    structural_fingerprint, platform, content_format, topic,
                    duration_seconds, compile_status, compiled_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'ready', ?)
                ON CONFLICT(report_pk) DO UPDATE SET
                    schema_version = excluded.schema_version,
                    compiler_version = excluded.compiler_version,
                    analyzer_version = excluded.analyzer_version,
                    source_artifact_sha256 = excluded.source_artifact_sha256,
                    record_sha256 = excluded.record_sha256,
                    record_json = excluded.record_json,
                    search_text = excluded.search_text,
                    structural_fingerprint = excluded.structural_fingerprint,
                    platform = excluded.platform,
                    content_format = excluded.content_format,
                    topic = excluded.topic,
                    duration_seconds = excluded.duration_seconds,
                    compile_status = 'ready', last_error = NULL,
                    compiled_at = excluded.compiled_at
                """,
                (
                    report_pk,
                    record["schema_version"],
                    compiler_version,
                    analyzer_version,
                    source_sha,
                    record_sha256,
                    canonical_json,
                    search_text,
                    fingerprint,
                    platform,
                    content_format,
                    topic,
                    duration,
                    now,
                ),
            )
            intelligence_id = int(
                connection.execute(
                    "SELECT id FROM intelligence_records WHERE report_pk = ?", (report_pk,)
                ).fetchone()[0]
            )
            connection.execute(
                "DELETE FROM intelligence_fts WHERE intelligence_id = ?", (intelligence_id,)
            )
            connection.execute(
                "INSERT INTO intelligence_fts(intelligence_id, search_text) VALUES (?, ?)",
                (intelligence_id, search_text),
            )
            connection.execute(
                "DELETE FROM intelligence_mechanisms WHERE intelligence_id = ?",
                (intelligence_id,),
            )
            mechanism_rows: list[tuple[object, ...]] = []
            hook = record.get("hook_intelligence", {})
            if isinstance(hook, dict):
                for item in hook.get("mechanisms", []):
                    if isinstance(item, dict):
                        evidence = item.get("evidence", {})
                        mechanism_rows.append(
                            (
                                intelligence_id,
                                "hook",
                                item.get("mechanism"),
                                item.get("start_seconds"),
                                item.get("end_seconds"),
                                evidence.get("confidence") if isinstance(evidence, dict) else None,
                                evidence.get("evidence_type", "unknown") if isinstance(evidence, dict) else "unknown",
                            )
                        )
            for item in record.get("retention_devices", []):
                if isinstance(item, dict):
                    evidence = item.get("evidence", {})
                    mechanism_rows.append(
                        (
                            intelligence_id,
                            "retention",
                            item.get("device"),
                            item.get("start_seconds"),
                            item.get("end_seconds"),
                            evidence.get("confidence") if isinstance(evidence, dict) else None,
                            evidence.get("evidence_type", "unknown") if isinstance(evidence, dict) else "unknown",
                        )
                    )
            connection.executemany(
                """
                INSERT OR IGNORE INTO intelligence_mechanisms(
                    intelligence_id, category, mechanism, start_seconds,
                    end_seconds, confidence, evidence_type
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                mechanism_rows,
            )
            connection.execute(
                """
                INSERT INTO intelligence_compile_attempts(
                    report_pk, status, attempts, last_error, updated_at
                ) VALUES (?, 'ready', 1, NULL, ?)
                ON CONFLICT(report_pk) DO UPDATE SET
                    status = 'ready', attempts = attempts + 1,
                    last_error = NULL, updated_at = excluded.updated_at
                """,
                (report_pk, now),
            )
        return True

    def mark_intelligence_failed(self, report_pk: int, error: str) -> None:
        self.connection.execute(
            """
            INSERT INTO intelligence_compile_attempts(
                report_pk, status, attempts, last_error, updated_at
            ) VALUES (?, 'failed', 1, ?, ?)
            ON CONFLICT(report_pk) DO UPDATE SET
                status = 'failed', attempts = attempts + 1,
                last_error = excluded.last_error, updated_at = excluded.updated_at
            """,
            (report_pk, error[:2000], utc_now()),
        )

    def intelligence_counts(self) -> dict[str, int]:
        return {
            "ready": int(self.connection.execute("SELECT COUNT(*) FROM intelligence_records WHERE compile_status = 'ready'").fetchone()[0]),
            "failed": int(self.connection.execute("SELECT COUNT(*) FROM intelligence_compile_attempts WHERE status = 'failed'").fetchone()[0]),
            "pending": int(self.connection.execute("SELECT COUNT(*) FROM reports r LEFT JOIN intelligence_records ir ON ir.report_pk = r.id WHERE ir.id IS NULL").fetchone()[0]),
        }

    def save_outcome(self, record: dict[str, object], canonical_json: str) -> bool:
        rights = record["rights"]
        normalization = record.get("cohort_normalization")
        assert isinstance(rights, dict)
        cohort_key = (
            normalization.get("cohort_key")
            if isinstance(normalization, dict)
            else record.get("cohort_key")
        )
        before = self.connection.total_changes
        self.connection.execute(
            """
            INSERT OR IGNORE INTO outcome_records(
                report_id, schema_version, platform, account_context_id,
                cohort_key, measured_at, payload_json, rights_allowed, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["report_id"],
                record["schema_version"],
                record["platform"],
                record["account_context_id"],
                cohort_key,
                record["measured_at"],
                canonical_json,
                1 if rights["training_allowed"] else 0,
                utc_now(),
            ),
        )
        return self.connection.total_changes > before
