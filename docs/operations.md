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
