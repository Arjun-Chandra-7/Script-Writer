from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .database import Registry


class DatasetNotReadyError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProposedRun:
    run_id: str
    dataset_version: str
    manifest_path: str
    manifest_sha256: str
    new_examples: int
    replay_examples: int
    evaluation_examples: int
    training_enabled: bool = False


class DatasetBuilder:
    def __init__(self, settings: Settings, registry: Registry):
        self.settings = settings
        self.registry = registry

    def propose_run(self) -> ProposedRun:
        if not self.settings.rights_attested:
            raise DatasetNotReadyError(
                "training-rights gate is closed; set SCRIPT_WRITER_RIGHTS_ATTESTED=true "
                "only after confirming the corpus is legally usable for training"
            )
        active_run = self.registry.active_run_id()
        if active_run is not None:
            raise RuntimeError(f"active training run already exists: {active_run}")
        new_rows = self.registry.select_new_training_reports(
            self.settings.max_new_examples
        )
        if len(new_rows) < self.settings.min_new_examples:
            raise DatasetNotReadyError(
                f"need {self.settings.min_new_examples} new train examples; "
                f"only {len(new_rows)} are eligible"
            )
        self.settings.manifest_dir.mkdir(parents=True, exist_ok=True)

        identity = "\n".join(str(row["artifact_sha256"]) for row in new_rows)
        identity_digest = hashlib.sha256(identity.encode()).hexdigest()
        version = identity_digest[:16]
        replay_limit = (
            len(new_rows) * self.settings.replay_ratio_percent + 99
        ) // 100
        replay_rows = self.registry.select_replay_reports(replay_limit, version)
        evaluation_rows = self.registry.connection.execute(
            """
            SELECT * FROM reports
            WHERE split IN ('validation', 'test')
            ORDER BY id
            """
        ).fetchall()

        header = {
            "format": "viralyst-dataset-manifest-v1",
            "version": version,
            "training_execution_enabled": False,
            "rights_attested": True,
            "new_count": len(new_rows),
            "replay_count": len(replay_rows),
            "evaluation_count": len(evaluation_rows),
        }
        lines = [json.dumps({"manifest": header}, sort_keys=True)]
        for role, rows in (
            ("new", new_rows),
            ("replay", replay_rows),
            ("evaluation", evaluation_rows),
        ):
            for row in rows:
                lines.append(
                    json.dumps(
                        {
                            "role": role,
                            "report_pk": row["id"],
                            "artifact_sha256": row["artifact_sha256"],
                            "split": row["split"],
                            "example": json.loads(row["canonical_json"]),
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
        payload = ("\n".join(lines) + "\n").encode()
        manifest_sha256 = hashlib.sha256(payload).hexdigest()
        final_path = self.settings.manifest_dir / f"dataset-{version}.jsonl"
        self._write_once(final_path, payload)
        config = {
            "min_new_examples": self.settings.min_new_examples,
            "max_new_examples": self.settings.max_new_examples,
            "replay_ratio_percent": self.settings.replay_ratio_percent,
            "split_salt": self.settings.split_salt,
            "training_execution_enabled": False,
        }
        config_sha256 = hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        try:
            run_id = self.registry.create_queued_run(
                version=version,
                manifest_sha256=manifest_sha256,
                manifest_path=str(final_path),
                config_sha256=config_sha256,
                new_report_ids=[int(row["id"]) for row in new_rows],
                replay_report_ids=[int(row["id"]) for row in replay_rows],
                evaluation_report_ids=[int(row["id"]) for row in evaluation_rows],
            )
        except Exception:
            # The manifest is content-addressed and harmless if DB registration
            # loses a race. A later integrity sweep can remove unattached files.
            raise
        return ProposedRun(
            run_id=run_id,
            dataset_version=version,
            manifest_path=str(final_path),
            manifest_sha256=manifest_sha256,
            new_examples=len(new_rows),
            replay_examples=len(replay_rows),
            evaluation_examples=len(evaluation_rows),
        )

    @staticmethod
    def _write_once(path: Path, payload: bytes) -> None:
        if path.exists():
            if path.read_bytes() != payload:
                raise OSError(f"immutable manifest collision at {path}")
            return
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}-", suffix=".part", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
