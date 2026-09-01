from __future__ import annotations

import argparse
import hashlib
import json
import logging
import signal
import sys
import time
from dataclasses import asdict
from pathlib import Path

from .config import Settings
from .compiler_service import IntelligenceCompilationService
from .corpus import CorpusIndex, SearchQuery
from .database import Registry
from .datasets import DatasetBuilder, DatasetNotReadyError
from .domain import RemoteFile
from .drive import GoogleDriveSource
from .evaluation import OfflineEvaluator
from .generation import DeterministicOutlineGenerator, GenerationRequest, RetrievalFirstScriptWriter
from .ingestion import IngestionService, MemorySource
from .intelligence import ScriptIntelligenceCompiler
from .training_dataset import ReviewStore, SamplingPolicy
from .training_workflow import build_training_artifacts, inspect_example
from .semantic_reconstruction import (
    RuleBasedSemanticIntentAdapter, SemanticInferenceCache, SemanticReconstructionService,
    estimate_corpus, field_leakage_report,
)
from .semantic_evaluation import evaluate_gold
from .sharded_assembly import build_shards
from .training_contracts import ClientTrainingContext
from .validation import parse_and_validate_report


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


def _watch_cycle(settings: Settings) -> dict[str, object]:
    registry = _registry(settings)
    try:
        source = GoogleDriveSource(settings.folder_id, settings.credentials_file)
        summary = IngestionService(settings, registry, source).sync_once()
        index_result = CorpusIndex(registry).rebuild()
        result: dict[str, object] = {
            "ingestion": asdict(summary),
            "intelligence": registry.intelligence_counts(),
            "index": index_result,
        }
        return result
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
    commands.add_parser(
        "propose-run",
        help="legacy inert training-manifest proposal; never executes training",
    )
    compile_command = commands.add_parser(
        "compile", help="compile pending admitted reports into ScriptIntelligenceRecords"
    )
    compile_command.add_argument("--limit", type=int, default=500)
    compile_command.add_argument("--all", action="store_true")
    compile_record = commands.add_parser(
        "compile-record", help="compile one local extractor report to canonical JSON"
    )
    compile_record.add_argument("path", type=Path)
    compile_record.add_argument("--output", type=Path)
    index_command = commands.add_parser("index", help="build missing corpus embeddings")
    index_command.add_argument("--force", action="store_true")
    query_command = commands.add_parser("query", help="search compiled script intelligence")
    query_command.add_argument("text", nargs="?", default="")
    query_command.add_argument("--platform")
    query_command.add_argument("--topic", action="append", default=[])
    query_command.add_argument("--format", dest="content_formats", action="append", default=[])
    query_command.add_argument("--hook", action="append", default=[])
    query_command.add_argument("--retention", action="append", default=[])
    query_command.add_argument("--max-duration", type=float)
    query_command.add_argument("--top-k", type=int, default=10)
    structural = commands.add_parser("structural-similar", help="find structurally similar scripts")
    structural.add_argument("record_id")
    structural.add_argument("--top-k", type=int, default=10)
    draft = commands.add_parser(
        "draft-baseline", help="run the deterministic retrieval-first contract demonstrator"
    )
    draft.add_argument("request", type=Path)
    evaluate = commands.add_parser("evaluate", help="run deterministic offline evaluation")
    evaluate.add_argument("request", type=Path)
    evaluate.add_argument("result", type=Path)
    evaluate.add_argument("--candidate-version", required=True)
    evaluate.add_argument("--fixture-version", default="manual-v1")
    watch = commands.add_parser("watch", help="continuously reconcile the Drive folder")
    watch.add_argument("--once", action="store_true", help="run once and exit")
    dry_run = commands.add_parser(
        "dry-run-sample", help="ingest one local report without Drive access"
    )
    dry_run.add_argument("path", type=Path)
    dataset = commands.add_parser("dataset", help="compile, audit, and inspect training data")
    dataset_commands = dataset.add_subparsers(dest="dataset_command", required=True)
    dataset_build = dataset_commands.add_parser("build", help="build immutable objective datasets")
    dataset_build.add_argument("--client", type=Path, required=True)
    dataset_build.add_argument("--intelligence", type=Path, action="append", required=True)
    dataset_build.add_argument("--output", type=Path, required=True)
    dataset_build.add_argument("--cluster-cap", type=int)
    dataset_build.add_argument("--source-cap", type=int)
    dataset_build.add_argument("--minimum-reviews", type=int, default=25)
    dataset_build.add_argument("--minimum-examples", type=int, default=100)
    dataset_show = dataset_commands.add_parser("show", help="show one compact review view")
    dataset_show.add_argument("directory", type=Path)
    dataset_show.add_argument("example_id")
    dataset_review = dataset_commands.add_parser("review", help="accept, reject, or flag an example")
    dataset_review.add_argument("directory", type=Path)
    dataset_review.add_argument("example_id")
    dataset_review.add_argument("decision", choices=("accept", "reject", "flag"))
    dataset_review.add_argument("--note", default="")
    dataset_audit_command = dataset_commands.add_parser("audit", help="show the latest immutable dataset audit")
    dataset_audit_command.add_argument("directory", type=Path)
    shard = dataset_commands.add_parser("shard", help="stream a compiled example JSONL into immutable shards")
    shard.add_argument("input", type=Path)
    shard.add_argument("--output", type=Path, required=True)
    shard.add_argument("--size", type=int, default=1000)
    semantic = commands.add_parser("semantic", help="semantic intent reconstruction workflows")
    semantic_commands = semantic.add_subparsers(dest="semantic_command", required=True)
    semantic_infer = semantic_commands.add_parser("infer", help="reconstruct one MinimumSufficientTrainingBrief")
    semantic_infer.add_argument("--intelligence", type=Path, required=True)
    semantic_infer.add_argument("--client", type=Path, required=True)
    semantic_infer.add_argument("--cache", type=Path)
    semantic_evaluate = semantic_commands.add_parser("evaluate", help="evaluate briefs against human gold annotations")
    semantic_evaluate.add_argument("--briefs", type=Path, required=True)
    semantic_evaluate.add_argument("--annotations", type=Path, required=True)
    semantic_estimate = semantic_commands.add_parser("estimate-corpus", help="estimate semantic inference scale without requests")
    semantic_estimate.add_argument("--records", type=int, required=True)
    semantic_estimate.add_argument("--average-words", type=int, default=160)
    semantic_estimate.add_argument("--input-price-per-million", type=float, default=0.0)
    semantic_estimate.add_argument("--output-price-per-million", type=float, default=0.0)
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
            print(
                json.dumps(
                    {
                        "counts": registry.counts(),
                        "intelligence": registry.intelligence_counts(),
                        "runs": registry.run_details(),
                    },
                    sort_keys=True,
                )
            )
        finally:
            registry.close()
        return 0
    if args.command == "propose-run":
        registry = _registry(settings)
        try:
            try:
                proposal = DatasetBuilder(settings, registry).propose_run()
            except DatasetNotReadyError as exc:
                print(json.dumps({"queued": False, "reason": str(exc)}))
                return 3
            print(json.dumps(asdict(proposal), sort_keys=True))
        finally:
            registry.close()
        return 0
    if args.command == "sync":
        print(json.dumps(_sync_once(settings), sort_keys=True))
        return 0
    if args.command == "compile":
        registry = _registry(settings)
        try:
            service = IntelligenceCompilationService(
                registry, ScriptIntelligenceCompiler(), settings.split_salt
            )
            total = {"examined": 0, "compiled": 0, "unchanged": 0, "failed": 0}
            while True:
                summary = asdict(service.compile_pending(limit=args.limit))
                for key, value in summary.items():
                    total[key] += value
                if not args.all or summary["examined"] == 0 or summary["compiled"] == 0:
                    break
            print(json.dumps(total, sort_keys=True))
        finally:
            registry.close()
        return 0
    if args.command == "compile-record":
        raw = args.path.read_bytes()
        report, _ = parse_and_validate_report(raw, split_salt=settings.split_salt)
        compiled = ScriptIntelligenceCompiler().compile(
            report, artifact_sha256=hashlib.sha256(raw).hexdigest()
        )
        rendered = json.dumps(compiled.record, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered)
            print(
                json.dumps(
                    {"output": str(args.output), "record_sha256": compiled.sha256, "bytes": len(rendered.encode())},
                    sort_keys=True,
                )
            )
        else:
            print(rendered, end="")
        return 0
    if args.command == "index":
        registry = _registry(settings)
        try:
            print(json.dumps(CorpusIndex(registry).rebuild_all(force=args.force), sort_keys=True))
        finally:
            registry.close()
        return 0
    if args.command == "query":
        registry = _registry(settings)
        try:
            index = CorpusIndex(registry)
            index.rebuild_all()
            hits = index.search(
                SearchQuery(
                    text=args.text,
                    platform=args.platform,
                    topics=tuple(args.topic),
                    content_formats=tuple(args.content_formats),
                    hook_mechanisms=tuple(args.hook),
                    retention_devices=tuple(args.retention),
                    max_duration_seconds=args.max_duration,
                    top_k=args.top_k,
                )
            )
            print(json.dumps([hit.summary() for hit in hits], indent=2, ensure_ascii=False))
        finally:
            registry.close()
        return 0
    if args.command == "structural-similar":
        registry = _registry(settings)
        try:
            hits = CorpusIndex(registry).structurally_similar(args.record_id, top_k=args.top_k)
            print(json.dumps([hit.summary() for hit in hits], indent=2, ensure_ascii=False))
        finally:
            registry.close()
        return 0
    if args.command == "draft-baseline":
        registry = _registry(settings)
        try:
            request_data = json.loads(args.request.read_text())
            request_data.pop("id", None)
            request = GenerationRequest(**request_data)
            index = CorpusIndex(registry)
            index.rebuild_all()
            result = RetrievalFirstScriptWriter(
                index, DeterministicOutlineGenerator()
            ).generate(request)
            print(json.dumps(result, indent=2, ensure_ascii=False))
        finally:
            registry.close()
        return 0
    if args.command == "evaluate":
        registry = _registry(settings)
        try:
            request_data = json.loads(args.request.read_text())
            request_data.pop("id", None)
            result_data = json.loads(args.result.read_text())
            source_records = []
            index = CorpusIndex(registry)
            for item in result_data.get("retrieved_evidence", []):
                try:
                    source_records.append(index.get_record(str(item["record_id"])))
                except KeyError:
                    pass
            evaluation = OfflineEvaluator().evaluate(
                request_data,
                result_data,
                source_records=source_records,
                candidate_version=args.candidate_version,
                fixture_set_version=args.fixture_version,
            )
            registry.save_evaluation(evaluation)
            print(json.dumps(evaluation, indent=2, ensure_ascii=False))
        finally:
            registry.close()
        return 0
    if args.command == "dry-run-sample":
        print(json.dumps(_dry_run_sample(settings, args.path), sort_keys=True))
        return 0
    if args.command == "dataset":
        if args.dataset_command == "build":
            result = build_training_artifacts(
                args.intelligence,
                args.client,
                args.output,
                split_salt=settings.split_salt,
                sampling_policy=SamplingPolicy(
                    max_examples_per_cluster_per_objective=args.cluster_cap,
                    max_examples_per_source_per_objective=args.source_cap,
                ),
                minimum_reviewed_examples=args.minimum_reviews,
                minimum_exported_examples=args.minimum_examples,
            )
            print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
            return 0
        if args.dataset_command == "show":
            print(json.dumps(inspect_example(args.directory, args.example_id), indent=2, ensure_ascii=False, sort_keys=True))
            return 0
        if args.dataset_command == "review":
            decision = ReviewStore(args.directory / "reviews.json").record(
                args.example_id, args.decision, note=args.note
            )
            print(json.dumps({"example_id": args.example_id, **decision}, sort_keys=True))
            return 0
        if args.dataset_command == "audit":
            paths = sorted(args.directory.glob("dataset-audit-*.json"), reverse=True)
            if not paths:
                raise FileNotFoundError("no dataset audit found")
            print(paths[0].read_text(), end="")
            return 0
        if args.dataset_command == "shard":
            def examples() -> object:
                for line in args.input.read_text().splitlines():
                    if line.strip():
                        yield json.loads(line)
            print(json.dumps(build_shards(examples(), args.output, shard_size=args.size), indent=2, sort_keys=True))
            return 0
    if args.command == "semantic":
        if args.semantic_command == "infer":
            cache = SemanticInferenceCache(args.cache) if args.cache else None
            try:
                record = json.loads(args.intelligence.read_text())
                client = ClientTrainingContext.from_path(args.client).projection
                brief = SemanticReconstructionService(RuleBasedSemanticIntentAdapter(), cache).reconstruct(record, client)
                target = str(record["content"]["clean_transcript"]["value"])
                print(json.dumps({"record_id": record["record_id"], "brief": brief, "field_leakage": field_leakage_report(brief, target)}, indent=2, ensure_ascii=False))
            finally:
                if cache: cache.close()
            return 0
        if args.semantic_command == "evaluate":
            raw = json.loads(args.briefs.read_text())
            values = raw if isinstance(raw, list) else raw.get("briefs", [raw])
            briefs = {
                item["record_id"]: item.get("brief", item)
                for item in values if isinstance(item, dict) and "record_id" in item
            }
            annotations = json.loads(args.annotations.read_text())
            print(json.dumps(evaluate_gold(briefs, annotations), indent=2, ensure_ascii=False))
            return 0
        if args.semantic_command == "estimate-corpus":
            print(json.dumps(estimate_corpus(args.records, args.average_words, adapter=RuleBasedSemanticIntentAdapter(), input_price_per_million=args.input_price_per_million, output_price_per_million=args.output_price_per_million), indent=2, sort_keys=True))
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
                print(json.dumps(_watch_cycle(settings), sort_keys=True), flush=True)
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
