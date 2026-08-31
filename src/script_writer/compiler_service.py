from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from .database import Registry
from .intelligence import ScriptIntelligenceCompiler
from .validation import parse_and_validate_report


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CompileSummary:
    examined: int = 0
    compiled: int = 0
    unchanged: int = 0
    failed: int = 0


class IntelligenceCompilationService:
    def __init__(self, registry: Registry, compiler: ScriptIntelligenceCompiler, split_salt: str):
        self.registry = registry
        self.compiler = compiler
        self.split_salt = split_salt

    def compile_pending(self, *, limit: int = 100) -> CompileSummary:
        rows = self.registry.pending_intelligence_reports(
            compiler_version=self.compiler.version,
            analyzer_version=self.compiler.semantic_analyzer.version,
            limit=limit,
        )
        counters = {"examined": 0, "compiled": 0, "unchanged": 0, "failed": 0}
        for row in rows:
            counters["examined"] += 1
            report_pk = int(row["report_pk"])
            try:
                raw = Path(row["artifact_path"]).read_bytes()
                report, _validation = parse_and_validate_report(raw, split_salt=self.split_salt)
                compiled = self.compiler.compile(
                    report, artifact_sha256=str(row["artifact_sha256"])
                )
                changed = self.registry.save_intelligence(
                    report_pk=report_pk,
                    record=compiled.record,
                    canonical_json=compiled.canonical_json,
                    record_sha256=compiled.sha256,
                )
            except Exception as exc:
                counters["failed"] += 1
                self.registry.mark_intelligence_failed(
                    report_pk, f"{type(exc).__name__}: {exc}"
                )
                LOGGER.exception("script intelligence compilation failed for report %s", report_pk)
            else:
                counters["compiled" if changed else "unchanged"] += 1
        return CompileSummary(**counters)
