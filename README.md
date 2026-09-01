# VIRALYST Script Writer

VIRALYST Script Writer is the reusable intelligence layer between detailed
short-form video extraction and future script generation. It is Instagram/Reels
first, platform-aware, and scoped to one client/account context at a time. The
wider system's three creative agents can call the same corpus and contracts;
they are parallel creative producers, not separate channel specialists.

**No model training, fine-tuning, GPU work, or external model API is implemented
or invoked.** The current system now ends at audited, immutable training data.

## What exists today

```text
Extractor report (large, immutable)
        |
        v
Drive/local ingestion -> validation -> exact/source dedupe -> raw archive
        |
        v
Script Intelligence compiler
  deterministic text/delivery statistics
  conservative semantic hypotheses
  evidence type + confidence + source spans
        |
        v
ScriptIntelligenceRecord v1 (~tens of KB, not ~800 KB)
        |
        v
SQLite metadata + FTS5 + structural fingerprints + embedding interface
        |
        v
Retrieval-first generation context -> generator contract -> structured result
        |
        v
Deterministic offline checks + pending human/model-judgment dimensions
        |
        v
ClientTrainingContext + intent reconstruction -> leakage-safe objective datasets
        |
        v
Dataset audit + TrainingReadinessReport -> FUTURE training only
```

Implemented capabilities:

- read-only, paginated Google Drive discovery;
- atomic downloads, retry leases, quarantine, SHA-256 and source-level dedupe;
- database migrations and cached, incrementally rebuildable intelligence;
- versioned `ScriptIntelligenceRecord`, outcome, generation, and evaluation schemas;
- observed/deterministic/heuristic/model/unknown evidence distinctions;
- deterministic transcript, language, structure, linguistic, density, delivery,
  and script/edit relationship projections;
- local semantic-rule adapter plus a graceful null adapter and protocol for a
  future versioned semantic model;
- metadata, lexical, mechanism, structural, and pluggable semantic retrieval;
- deterministic local hashing embeddings for tests and offline operation;
- retrieval-first generation context and structured result contracts;
- anti-copy instructions, banned-pattern enforcement, and corpus-overlap checks;
- Instagram/Reels-compatible outcome records kept separate from content evidence;
- objective-specific dataset policies and exact/derived/near-duplicate leakage guards;
- regression fixtures and offline evaluation without fabricated quality scores.
- compact, source-grounded `ClientTrainingContext` projections;
- seven objective-specific `ScriptTrainingExample` contracts;
- target-leakage metrics, quality filtering, duplicate clustering, universal
  source splits, immutable JSONL manifests, and explicit rejection files;
- incremental per-record compilation cache, corpus audit, readiness gates, and
  compact CLI review workflow.
- staged semantic intent reconstruction with fail-closed adapters, field-level
  leakage/compression checks, gold-set evaluation helpers, and shard export.
- real-corpus semantic quality-validation workflow: stratified, evaluation-only
  gold selection; blind/assisted review; adjudication; frozen gold exclusions;
  adapter, ablation, contamination, pilot, calibration, and anomaly reports.

## What does not exist yet

- a production semantic inference model for topic, story, persuasion, claims, or CTA;
- a production creative language generator;
- human/editor judgment collection;
- Instagram Insights ingestion and cohort normalization;
- a production embedding provider (the interface exists; local hashing is a baseline);
- production semantic intent reconstruction, which is required before most
  real full-script candidates become eligible;
- model training, fine-tuning, preference optimization, or model promotion.

Unknown is deliberately emitted when evidence is absent. The extractor report
describes content, not performance; analytics later add ranking/weighting signals
without blocking content understanding today.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env
.venv/bin/script-writer init
.venv/bin/python -m unittest discover -s tests -v
```

Compile the supplied real extractor report without Drive or network access:

```bash
.venv/bin/script-writer compile-record \
  86c1671e8a3a7b46-The-Godfather-of-AI-on-his-Feud-with-Elon-Musk---TVO-Today--1080p.json \
  --output /tmp/script-intelligence.json
```

A validated compiler output from that report is committed as
[`examples/compiled/86c1671e8a3a7b46.script-intelligence.v1.json`](examples/compiled/86c1671e8a3a7b46.script-intelligence.v1.json).

Ingest it into a disposable local registry, build the index, and query it:

```bash
SCRIPT_WRITER_STATE_DIR=/tmp/viralyst-demo .venv/bin/script-writer dry-run-sample \
  86c1671e8a3a7b46-The-Godfather-of-AI-on-his-Feud-with-Elon-Musk---TVO-Today--1080p.json
SCRIPT_WRITER_STATE_DIR=/tmp/viralyst-demo .venv/bin/script-writer index
SCRIPT_WRITER_STATE_DIR=/tmp/viralyst-demo .venv/bin/script-writer query \
  'government workers Elon consequences'
SCRIPT_WRITER_STATE_DIR=/tmp/viralyst-demo .venv/bin/script-writer query \
  --hook question --retention contrast
```

Build the committed real training-data demonstration:

```bash
.venv/bin/script-writer dataset build \
  --client fixtures/client.example.json \
  --intelligence examples/compiled/86c1671e8a3a7b46.script-intelligence.v1.json \
  --output /tmp/viralyst-training-data
.venv/bin/script-writer dataset audit /tmp/viralyst-training-data
```

The real sample intentionally remains `not_training_ready`: its topic and
central idea are not reliably present, so 14 candidates are rejected and only
one continuation example is exported. See the committed
[`examples/training`](examples/training/) artifacts.

See [architecture](docs/architecture.md), [operations](docs/operations.md),
[training-data pipeline](docs/training-data.md), [research](docs/research.md),
[semantic reconstruction](docs/semantic-reconstruction.md), and the
[semantic validation](docs/semantic-validation.md), and the machine-readable
[schemas](schemas/).
