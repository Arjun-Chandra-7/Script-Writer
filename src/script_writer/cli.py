from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .config import Settings
from .database import Registry
from .domain import RemoteFile
from .drive import GoogleDriveSource
from .ingestion import IngestionService, MemorySource


LOGGER = logging.getLogger(__name__)


def _registry(settings: Settings) -> Registry:
    registry = Registry(settings.database_path)
    registry.initialize()
    return registry


def _sync_once(settings: Settings) -> dict[str, int]:
    registry = _registry(settings)
    try:
        source = GoogleDriveSource(settings.folder_id, settings.credentials_file)
        summary = IngestionService(settings, registry, source).sync_once()
        return asdict(summary)
    finally:
        registry.close()


def _dry_run_sample(settings: Settings, sample: Path) -> dict[str, int]:
    body = sample.read_bytes()
    item = RemoteFile(
        file_id=f"local:{sample.resolve()}",
        name=sample.name,
        mime_type="application/json",
        modified_time=str(sample.stat().st_mtime_ns),
        size=len(body),
        md5_checksum=None,
    )
    registry = _registry(settings)
    try:
        summary = IngestionService(
            settings, registry, MemorySource([(item, body)])
        ).sync_once()
        return asdict(summary)
    finally:
        registry.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="script-writer")
    parser.add_argument("--verbose", action="store_true")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="initialize the durable local registry")
    commands.add_parser("sync", help="perform one Drive reconciliation scan")
    commands.add_parser("status", help="show ingestion and training queue state")
    watch = commands.add_parser("watch", help="continuously reconcile the Drive folder")
    watch.add_argument("--once", action="store_true", help="run once and exit")
    dry_run = commands.add_parser(
        "dry-run-sample", help="ingest one local report without Drive access"
    )
    dry_run.add_argument("path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings.from_env()

    if args.command == "init":
        registry = _registry(settings)
        registry.close()
        print(json.dumps({"database": str(settings.database_path), "initialized": True}))
        return 0
    if args.command == "status":
        registry = _registry(settings)
        try:
            print(json.dumps(registry.counts(), sort_keys=True))
        finally:
            registry.close()
        return 0
    if args.command == "sync":
        print(json.dumps(_sync_once(settings), sort_keys=True))
        return 0
    if args.command == "dry-run-sample":
        print(json.dumps(_dry_run_sample(settings, args.path), sort_keys=True))
        return 0
    if args.command == "watch":
        stop = False

        def request_stop(_signum: int, _frame: object) -> None:
            nonlocal stop
            stop = True

        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
        while not stop:
            try:
                print(json.dumps(_sync_once(settings), sort_keys=True), flush=True)
            except Exception:
                LOGGER.exception("Drive reconciliation failed; next scan will retry")
            if args.once:
                break
            deadline = time.monotonic() + settings.poll_seconds
            while not stop and time.monotonic() < deadline:
                time.sleep(min(1.0, deadline - time.monotonic()))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
