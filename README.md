# VIRALYST Script Writer

VIRALYST Script Writer is the reusable intelligence layer between detailed
short-form video extraction and future script generation. It is Instagram/Reels
first, platform-aware, and scoped to one client/account context at a time. The
wider system's three creative agents can call the same corpus and contracts;
they are parallel creative producers, not separate channel specialists.

**No model training, fine-tuning, GPU work, or external model API is implemented
or invoked.** The current baseline is deterministic compilation + local
retrieval + a versioned generator interface.

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

## What does not exist yet

- a production semantic inference model for topic, story, persuasion, claims, or CTA;
- a production creative language generator;
- human/editor judgment collection;
- Instagram Insights ingestion and cohort normalization;
- a production embedding provider (the interface exists; local hashing is a baseline);
- SFT/preference datasets, model training, fine-tuning, or model promotion.

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

See [architecture](docs/architecture.md), [operations](docs/operations.md),
[research](docs/research.md), and the machine-readable [schemas](schemas/).
