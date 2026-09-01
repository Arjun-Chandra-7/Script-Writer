# Operations guide

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env
.venv/bin/script-writer init
.venv/bin/python -m unittest discover -s tests -v
```

No model, GPU runtime, embedding download, or external API is required for tests.

## Compile the real sample locally

```bash
.venv/bin/script-writer compile-record \
  86c1671e8a3a7b46-The-Godfather-of-AI-on-his-Feud-with-Elon-Musk---TVO-Today--1080p.json \
  --output /tmp/script-intelligence.json
```

This validates and compiles without mutating the registry. `compile-record` is
also useful when developing a new extractor adapter.

## Local ingestion and retrieval demonstration

Use an empty state path:

```bash
export SCRIPT_WRITER_STATE_DIR=/tmp/viralyst-script-writer-demo
.venv/bin/script-writer dry-run-sample \
  86c1671e8a3a7b46-The-Godfather-of-AI-on-his-Feud-with-Elon-Musk---TVO-Today--1080p.json
.venv/bin/script-writer index
.venv/bin/script-writer status
```

Example query operations:

```bash
# Hybrid lexical/local-semantic retrieval
.venv/bin/script-writer query 'Elon government workers consequences'

# Multi-label mechanism filtering
.venv/bin/script-writer query --hook question --retention contrast

# Delivery/length filter
.venv/bin/script-writer query 'criticism consequences' --max-duration 65

# Exact structural comparison against another record (empty with a one-record corpus)
.venv/bin/script-writer structural-similar \
  'sir:277de2b59f2be8f88047b631ed1e5de4:1.0.0'
```

`query` returns record/report/source IDs, component scores, matched mechanisms,
hook, and a short excerpt. It does not return raw extractor arrays.

## Google Drive setup

1. Create a Google Cloud project and enable Drive API.
2. Create a service account and download its JSON key.
3. Share folder `1loe1nchN4PFqkTzujZmbkB31E_DpCwqE` with the service-account
   email as Viewer.
4. Store the key at `credentials/google-service-account.json`.

The watcher requests read-only Drive scope. It never changes source files.

```bash
.venv/bin/script-writer sync
.venv/bin/script-writer watch --once
.venv/bin/script-writer watch
```

Each watch cycle reconciles Drive, validates/deduplicates, compiles intelligence,
and builds missing local index vectors. It does **not** propose or execute model
training.

The included user-service unit can be linked to `~/.config/systemd/user/`:

```bash
systemctl --user daemon-reload
systemctl --user enable --now viralyst-script-writer.service
journalctl --user -u viralyst-script-writer.service -f
```

## Incremental rebuilds and migrations

```bash
# Compile legacy reports missing the current compiler/analyzer version
.venv/bin/script-writer compile --all --limit 500

# Add only missing vectors
.venv/bin/script-writer index

# Deterministically rebuild current-provider vectors and FTS rows
.venv/bin/script-writer index --force
```

Registry initialization applies ordered migrations. Database v1 migrates to v2
with intelligence, mechanism, embedding, outcome, and evaluation tables. Never
edit `schema_meta` manually.

## Generation and evaluation contracts

`draft-baseline` accepts one `GenerationRequest` JSON object. The fixture file
contains a set, so copy one fixture object without its `id` to `request.json`:

```bash
.venv/bin/script-writer draft-baseline request.json
.venv/bin/script-writer evaluate request.json result.json \
  --candidate-version local-baseline-v1 \
  --fixture-version requests-v1
```

The draft command is a pipeline/contract demonstrator, not the production
creative writer. Evaluation persists deterministic results and leaves
subjective dimensions unscored.

## Training-data compilation

This command consumes compact `ScriptIntelligenceRecord` files, never the giant
raw extractor report:

```bash
.venv/bin/script-writer dataset build \
  --client fixtures/client.example.json \
  --intelligence examples/compiled/86c1671e8a3a7b46.script-intelligence.v1.json \
  --output /tmp/viralyst-training-data \
  --minimum-reviews 25 \
  --minimum-examples 100
.venv/bin/script-writer dataset audit /tmp/viralyst-training-data
```

Repeat `--intelligence PATH` for additional records. Exact reruns reuse
`.training-cache/compilations.sqlite3`; the cache is rebuildable and excluded
from source control. Dataset JSONL and manifests are content-addressed and
write-once. Optional `--source-cap` and `--cluster-cap` sampling limits are
recorded in every manifest; no silent balancing occurs.

Inspect and review without opening source reports:

```bash
.venv/bin/script-writer dataset show /tmp/viralyst-training-data EXAMPLE_ID
.venv/bin/script-writer dataset review /tmp/viralyst-training-data EXAMPLE_ID flag \
  --note 'topic reconstruction needs review'
```

Decisions `reject` and `flag` hold an example out of subsequent manifests;
`accept` records inspection. Re-run `dataset build` after decisions to produce
new immutable artifacts and a new readiness report. Old manifests remain
unchanged.

Do not lower readiness thresholds to make a tiny demonstration pass. A report
must show no cross-split or target leakage, sufficient eligible examples,
frozen validation/test membership, and sufficient human inspection before its
status can become `training_ready`.

## Semantic brief reconstruction

```bash
.venv/bin/script-writer semantic infer \
  --intelligence examples/compiled/86c1671e8a3a7b46.script-intelligence.v1.json \
  --client fixtures/client.example.json \
  --cache /tmp/semantic-cache.sqlite3
.venv/bin/script-writer semantic estimate-corpus --records 7500 --average-words 160
```

The local rule adapter is an inspectable baseline. The provider-neutral HTTP
adapter is intentionally not selected by CLI and has no implicit network path.
Use `dataset build --semantic-rules` only for a dry-run candidate audit; it does
not prove semantic quality or open the training gate.

## Human semantic validation

Run `gold sample`, then use blind review before assisted review for the core
fields. Preserve reviewer annotations and append adjudications; freeze the
result before benchmarking. The full reproducible command sequence and its
current one-real-record limitation are in `docs/semantic-validation.md`.

## Monitoring and recovery

```bash
.venv/bin/script-writer status
sqlite3 state/registry.sqlite3 'PRAGMA integrity_check;'
```

Status separates ingestion and intelligence counts. Back up
`state/registry.sqlite3`, `state/raw/`, and `state/manifests/` together. Expired
download leases become retryable; `.part` files are never admitted. Raw-report
validation failures use source quarantine. Intelligence compiler failures keep
the raw report admitted and record an independently retryable error.

## Current safety boundary

- `SCRIPT_WRITER_AUTO_PROPOSE_RUN` defaults false and is not used by the watcher.
- The legacy database still enforces `training_enabled = 0`.
- `propose-run` is retained only for compatibility/audit and produces an inert
  legacy manifest; it is not a Script Intelligence workflow.
- No training runner, model weights, checkpoints, or promotion operation exists.
- `dataset build` only writes training data; it cannot launch a training job.
