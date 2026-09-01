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
from typing import Any

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
from .gold_validation import (
    adjudicate, benchmark_adapter, contamination_test, freeze_gold_set, full_corpus_projection, new_annotation,
    pilot, review_payload, run_ablation, semantic_quality_gate_report, stratified_gold_sample, verified_quality_report,
)
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


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
    dataset_build.add_argument("--semantic-rules", action="store_true", help="use the conservative local semantic intent adapter")
    dataset_build.add_argument("--gold-manifest", type=Path, help="frozen evaluation-only sources to exclude from training")
    dataset_build.add_argument("--semantic-quality-report", type=Path, help="frozen quality-gate report; only a passing report can satisfy that readiness gate")
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
    semantic_infer.add_argument("--output", type=Path, help="optionally persist the compact review artifact")
    semantic_evaluate = semantic_commands.add_parser("evaluate", help="evaluate briefs against human gold annotations")
    semantic_evaluate.add_argument("--briefs", type=Path, required=True)
    semantic_evaluate.add_argument("--annotations", type=Path, required=True)
    semantic_errors = semantic_commands.add_parser("error-analysis", help="show semantic reconstruction failures from gold annotations")
    semantic_errors.add_argument("--briefs", type=Path, required=True)
    semantic_errors.add_argument("--annotations", type=Path, required=True)
    semantic_estimate = semantic_commands.add_parser("estimate-corpus", help="estimate semantic inference scale without requests")
    semantic_estimate.add_argument("--records", type=int, required=True)
    semantic_estimate.add_argument("--average-words", type=int, default=160)
    semantic_estimate.add_argument("--input-price-per-million", type=float, default=0.0)
    semantic_estimate.add_argument("--output-price-per-million", type=float, default=0.0)
    semantic_review = semantic_commands.add_parser("review", help="append one human semantic annotation")
    semantic_review.add_argument("--record-id", required=True)
    semantic_review.add_argument("--output", type=Path, required=True)
    semantic_review.add_argument("--field", required=True)
    semantic_review.add_argument("--status", choices=("value", "unknown", "ambiguous", "not_inferable"), required=True)
    semantic_review.add_argument("--value", action="append", default=[])
    semantic_review.add_argument("--reviewer", required=True)
    semantic_review.add_argument("--note", default="")
    gold = commands.add_parser("gold", help="human semantic-quality validation workflows")
    gold_commands = gold.add_subparsers(dest="gold_command", required=True)
    gold_sample = gold_commands.add_parser("sample", help="create a deterministic stratified evaluation-only selection")
    gold_sample.add_argument("--intelligence", type=Path, action="append", required=True)
    gold_sample.add_argument("--output", type=Path, required=True)
    gold_sample.add_argument("--count", type=int, default=150)
    gold_sample.add_argument("--seed", default="gold-v1")
    gold_review = gold_commands.add_parser("review", help="emit a blind or assisted compact review artifact")
    gold_review.add_argument("--intelligence", type=Path, required=True)
    gold_review.add_argument("--client", type=Path, required=True)
    gold_review.add_argument("--mode", choices=("blind", "assisted"), required=True)
    gold_review.add_argument("--output", type=Path, required=True)
    gold_annotate = gold_commands.add_parser("annotate", help="append one reviewer field annotation")
    gold_annotate.add_argument("--record-id", required=True)
    gold_annotate.add_argument("--reviewer", required=True)
    gold_annotate.add_argument("--mode", choices=("blind", "assisted"), required=True)
    gold_annotate.add_argument("--field", required=True)
    gold_annotate.add_argument(
        "--status",
        choices=(
            "value", "accepted", "partial", "wrong",
            "too_broad", "too_narrow", "too_vague", "too_detailed",
            "leakage", "unknown", "ambiguous", "not_inferable", "reject"
        ),
        required=True,
    )
    gold_annotate.add_argument("--value", action="append", default=[])
    gold_annotate.add_argument("--proposed-value")
    gold_annotate.add_argument("--confidence", type=float)
    gold_annotate.add_argument("--inferability")
    gold_annotate.add_argument("--ambiguity")
    gold_annotate.add_argument("--alternatives", action="append", default=[])
    gold_annotate.add_argument("--notes", "--note", dest="notes", default="")
    gold_annotate.add_argument("--output", type=Path, required=True)
    gold_adjudicate = gold_commands.add_parser("adjudicate", help="append a resolved field without deleting reviewer disagreement")
    gold_adjudicate.add_argument("--annotations", type=Path, required=True)
    gold_adjudicate.add_argument("--record-id", required=True)
    gold_adjudicate.add_argument("--field", required=True)
    gold_adjudicate.add_argument("--resolver", required=True)
    gold_adjudicate.add_argument(
        "--status",
        choices=(
            "value", "accepted", "partial", "wrong",
            "too_broad", "too_narrow", "too_vague", "too_detailed",
            "leakage", "unknown", "ambiguous", "not_inferable", "reject"
        ),
        required=True,
    )
    gold_adjudicate.add_argument("--value", action="append", default=[])
    gold_adjudicate.add_argument("--notes", "--note", dest="notes", default="")
    gold_adjudicate.add_argument("--output", type=Path, required=True)
    gold_freeze = gold_commands.add_parser("freeze", help="create immutable evaluation-only gold manifest")
    gold_freeze.add_argument("--selection", type=Path, required=True)
    gold_freeze.add_argument("--annotations", type=Path, required=True)
    gold_freeze.add_argument("--adjudications", type=Path, required=True)
    gold_freeze.add_argument("--output", type=Path, required=True)
    gold_benchmark = gold_commands.add_parser("benchmark", help="compare local single-pass and staged semantic configurations")
    gold_benchmark.add_argument("--intelligence", type=Path, action="append", required=True)
    gold_benchmark.add_argument("--client", type=Path, required=True)
    gold_benchmark.add_argument("--annotations", type=Path, default=None)
    gold_benchmark.add_argument("--output", type=Path, required=True)
    gold_ablation = gold_commands.add_parser("ablation", help="compare canonical semantic input variants")
    gold_ablation.add_argument("--intelligence", type=Path, action="append", required=True)
    gold_ablation.add_argument("--client", type=Path, required=True)
    gold_ablation.add_argument("--annotations", type=Path, default=None)
    gold_ablation.add_argument("--output", type=Path, required=True)
    gold_contaminate = gold_commands.add_parser("contamination-test", help="verify irrelevant client contexts do not rewrite source meaning")
    gold_contaminate.add_argument("--intelligence", type=Path, required=True)
    gold_contaminate.add_argument("--clients", type=Path, action="append", required=True)
    gold_contaminate.add_argument("--output", type=Path, required=True)
    gold_pilot = gold_commands.add_parser("pilot", help="run a bounded local semantic corpus pilot")
    gold_pilot.add_argument("--intelligence", type=Path, action="append", required=True)
    gold_pilot.add_argument("--client", type=Path, required=True)
    gold_pilot.add_argument("--limit", type=int, default=500)
    gold_pilot.add_argument("--output", type=Path, required=True)
    gold_report = gold_commands.add_parser("report", help="apply frozen semantic quality gates to a benchmark artifact")
    gold_report.add_argument("--benchmark", type=Path, required=True)
    gold_report.add_argument("--reviewed-sources", type=int, required=True)
    gold_report.add_argument("--gates", type=Path, required=True)
    gold_report.add_argument("--output", type=Path, required=True)
    gold_estimate = gold_commands.add_parser("estimate", help="project full-corpus semantic cost/runtime from a measured pilot")
    gold_estimate.add_argument("--pilot", type=Path, required=True)
    gold_estimate.add_argument("--sources", type=int, required=True)
    gold_estimate.add_argument("--input-price-per-million", type=float, default=0.0)
    gold_estimate.add_argument("--output-price-per-million", type=float, default=0.0)
    gold_estimate.add_argument("--output", type=Path, required=True)
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
                semantic_rules=args.semantic_rules,
                gold_manifest=json.loads(args.gold_manifest.read_text()) if args.gold_manifest else None,
                semantic_quality_gold_evaluated=verified_quality_report(json.loads(args.semantic_quality_report.read_text())) if args.semantic_quality_report else False,
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
                with args.input.open() as handle:
                    for line in handle:
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
                artifact = {"record_id": record["record_id"], "brief": brief, "field_leakage": field_leakage_report(brief, target)}
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
                print(json.dumps(artifact, indent=2, ensure_ascii=False))
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
        if args.semantic_command == "error-analysis":
            raw = json.loads(args.briefs.read_text())
            values = raw if isinstance(raw, list) else raw.get("briefs", [raw])
            briefs = {
                item["record_id"]: item.get("brief", item)
                for item in values if isinstance(item, dict) and "record_id" in item
            }
            report = evaluate_gold(briefs, json.loads(args.annotations.read_text()))
            print(json.dumps({"error_count": report["error_count"], "errors": report["errors"]}, indent=2, ensure_ascii=False))
            return 0
        if args.semantic_command == "estimate-corpus":
            print(json.dumps(estimate_corpus(args.records, args.average_words, adapter=RuleBasedSemanticIntentAdapter(), input_price_per_million=args.input_price_per_million, output_price_per_million=args.output_price_per_million), indent=2, sort_keys=True))
            return 0
        if args.semantic_command == "review":
            annotations = json.loads(args.output.read_text()) if args.output.exists() else []
            if not isinstance(annotations, list): raise ValueError("annotation file must be an array")
            item = next((value for value in annotations if value.get("record_id") == args.record_id and value.get("reviewer_id") == args.reviewer), None)
            if item is None:
                item = {"record_id": args.record_id, "reviewer_id": args.reviewer, "fields": {}}
                annotations.append(item)
            field = {"status": args.status, "note": args.note}
            if args.status == "value": field["acceptable_values"] = args.value
            item["fields"][args.field] = field
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(annotations, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
            print(json.dumps({"record_id": args.record_id, "field": args.field, "status": args.status}))
            return 0
    if args.command == "gold":
        if args.gold_command == "estimate":
            result = full_corpus_projection(json.loads(args.pilot.read_text()), args.sources, input_price_per_million=args.input_price_per_million, output_price_per_million=args.output_price_per_million)
            _write_json(args.output, result)
            print(json.dumps({"sources": args.sources, "pricing_configured": result["pricing_configured"], "expected": result["expected"]}, sort_keys=True))
            return 0
        if args.gold_command == "report":
            benchmark = json.loads(args.benchmark.read_text())
            # A comparison artifact contains the staged result; direct benchmark artifacts are also accepted.
            candidate = benchmark.get("staged", benchmark)
            result = semantic_quality_gate_report(candidate, reviewed_sources=args.reviewed_sources, config=json.loads(args.gates.read_text()))
            _write_json(args.output, result)
            print(json.dumps({"semantic_reconstruction_gold_quality_verified": result["semantic_reconstruction_gold_quality_verified"], "failed_gates": result["failed_gates"]}, sort_keys=True))
            return 0
        if args.gold_command == "sample":
            selection = stratified_gold_sample([json.loads(path.read_text()) for path in args.intelligence], args.count, seed=args.seed)
            _write_json(args.output, selection)
            print(json.dumps({"selection_id": selection["selection_id"], "available_records": selection["available_records"], "sampled_records": len(selection["entries"]), "evaluation_only": True}, sort_keys=True))
            return 0
        if args.gold_command == "review":
            record = json.loads(args.intelligence.read_text())
            client = ClientTrainingContext.from_path(args.client).projection
            proposal = None
            if args.mode == "assisted": proposal = SemanticReconstructionService(RuleBasedSemanticIntentAdapter()).reconstruct(record, client)
            payload = review_payload(record, mode=args.mode, client=client, proposal=proposal)
            _write_json(args.output, payload)
            print(json.dumps({"record_id": record["record_id"], "mode": args.mode, "output": str(args.output)}, sort_keys=True))
            return 0
        if args.gold_command == "annotate":
            annotations = json.loads(args.output.read_text()) if args.output.exists() else []
            if not isinstance(annotations, list): raise ValueError("annotations output must be an array")
            annotation = next((item for item in annotations if item.get("record_id") == args.record_id and item.get("reviewer_id") == args.reviewer and item.get("mode") == args.mode), None)
            if annotation is None:
                annotation = new_annotation(args.record_id, args.reviewer, args.mode, {})
                annotations.append(annotation)
            field: dict[str, Any] = {"status": args.status}
            if args.status == "value" or args.value: field["acceptable_values"] = args.value
            if getattr(args, "proposed_value", None) is not None: field["proposed_value"] = args.proposed_value
            if getattr(args, "confidence", None) is not None: field["confidence"] = args.confidence
            if getattr(args, "inferability", None) is not None: field["inferability"] = args.inferability
            if getattr(args, "ambiguity", None) is not None: field["ambiguity"] = args.ambiguity
            if getattr(args, "alternatives", None): field["alternatives"] = args.alternatives
            if getattr(args, "notes", None): field["notes"] = args.notes
            annotation["fields"][args.field] = field
            from .gold_validation import validate_gold_annotation
            validate_gold_annotation(annotation)
            _write_json(args.output, annotations)
            print(json.dumps({"record_id": args.record_id, "reviewer": args.reviewer, "field": args.field, "status": args.status}, sort_keys=True))
            return 0
        if args.gold_command == "adjudicate":
            annotations = json.loads(args.annotations.read_text())
            resolved: dict[str, Any] = {"status": args.status}
            if args.status == "value" or args.value: resolved["acceptable_values"] = args.value
            note = getattr(args, "notes", "")
            if note: resolved["notes"] = note
            item = adjudicate(args.record_id, args.field, args.resolver, resolved, annotations, note=note)
            adjudications = json.loads(args.output.read_text()) if args.output.exists() else []
            adjudications.append(item)
            _write_json(args.output, adjudications)
            print(json.dumps({"record_id": args.record_id, "field": args.field, "reviewer_count": len(item["source_reviewers"])}, sort_keys=True))
            return 0
        if args.gold_command == "freeze":
            result = freeze_gold_set(json.loads(args.selection.read_text()), json.loads(args.annotations.read_text()), json.loads(args.adjudications.read_text()), args.output)
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        if args.gold_command in {"benchmark", "ablation", "pilot"}:
            records = [json.loads(path.read_text()) for path in args.intelligence]
            client = ClientTrainingContext.from_path(args.client).projection
            labels = []
            if getattr(args, "annotations", None) and args.annotations.exists():
                labels = json.loads(args.annotations.read_text())
            factory = lambda: SemanticReconstructionService(RuleBasedSemanticIntentAdapter())
            if args.gold_command == "benchmark":
                result = {
                    "baseline_local": benchmark_adapter(records, client, labels, factory, mode="single_pass", input_variant="full", name="rule-local-single-pass"),
                    "staged": benchmark_adapter(records, client, labels, factory, mode="staged", input_variant="full", name="rule-local-staged"),
                }
            elif args.gold_command == "ablation":
                result = run_ablation(records, client, labels, factory)
            else:
                result = pilot(records, client, factory, limit=args.limit)
            _write_json(args.output, result)
            print(json.dumps({"command": args.gold_command, "records": len(records), "output": str(args.output)}, sort_keys=True))
            return 0
        if args.gold_command == "contamination-test":
            record = json.loads(args.intelligence.read_text())
            contexts = {path.stem: ClientTrainingContext.from_path(path).projection for path in args.clients}
            result = contamination_test(record, contexts, SemanticReconstructionService(RuleBasedSemanticIntentAdapter()))
            _write_json(args.output, result)
            print(json.dumps({"pass": result["pass"], "client_context_contamination_rate": result["client_context_contamination_rate"]}, sort_keys=True))
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
