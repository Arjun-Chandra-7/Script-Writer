# Operations guide

## 1. One-time Google setup

1. Create a Google Cloud project and enable the Google Drive API.
2. Create a service account and download its JSON key.
3. Share Drive folder `1loe1nchN4PFqkTzujZmbkB31E_DpCwqE` with the service
   account's email as **Viewer**.
4. Store the key as `credentials/google-service-account.json`. This directory is
   ignored by Git.

The link being viewable in a browser is not a durable machine-authentication
mechanism. A service account gives the watcher a revocable identity and lets it
use the supported Drive API. The watcher requests only the read-only Drive
scope and never deletes, renames, or uploads source files.

## 2. Install and test

```bash
python3 -m venv .venv
.venv/bin/pip install -e '.[dev]'
cp .env.example .env
.venv/bin/script-writer init
.venv/bin/python -m unittest discover -s tests -v
```

Before using Drive, exercise the full local path with the supplied report:

```bash
.venv/bin/script-writer dry-run-sample \
  86c1671e8a3a7b46-The-Godfather-of-AI-on-his-Feud-with-Elon-Musk---TVO-Today--1080p.json
.venv/bin/script-writer status
```

The sample should be admitted once. Repeating the command should report it as
skipped, proving revision idempotency.

## 3. Run the watcher

For foreground verification:

```bash
.venv/bin/script-writer watch --once
.venv/bin/script-writer watch
```

For continuous operation, copy or link `deploy/viralyst-script-writer.service`
to `~/.config/systemd/user/`, then run:

```bash
systemctl --user daemon-reload
systemctl --user enable --now viralyst-script-writer.service
journalctl --user -u viralyst-script-writer.service -f
```

The unit is pinned to this workspace path and grants write access only to the
runtime `state` directory. If the repository moves, update all three paths in
the unit.

## 4. Rights and proposal gates

Leave `SCRIPT_WRITER_RIGHTS_ATTESTED=false` while testing. Once the owner has
verified training rights for the complete corpus, set it to `true` and restart
the watcher. At the configured threshold, the watcher automatically writes an
immutable manifest and queues exactly one run.

This revision always writes `training_execution_enabled: false` and the
database enforces `training_enabled = 0`. A queued run is therefore a proposal,
not a GPU job. This satisfies the current instruction not to train yet while
making the ingest-to-training handoff explicit and auditable.

## 5. Batch behavior

With the default 500/500 settings:

- batch 1 reserves up to 500 previously unreserved train-split reports;
- repeated Drive scans skip unchanged files;
- batch 2 may ingest while batch 1 is queued/running, but remains unreserved;
- no second run can be queued while an active run exists;
- a future approved runner must mark a successful run consumed before the next
  500 are proposed, or abandon it explicitly to release reservations;
- validation/test items are permanently split by source hash and included only
  as evaluation members.

The 90/5/5 split means a Drive upload of exactly 500 reports usually produces
about 450 train examples, not 500. If the intended trigger is 500 uploaded files
rather than 500 train examples, lower `SCRIPT_WRITER_MIN_NEW_EXAMPLES` after
review. Do not change `SCRIPT_WRITER_SPLIT_SALT` after the first run.

## 6. Monitoring and recovery

```bash
.venv/bin/script-writer status
sqlite3 state/registry.sqlite3 'PRAGMA integrity_check;'
```

Back up `state/registry.sqlite3`, `state/raw/`, and `state/manifests/` together.
On an unclean shutdown, a download lease expires after 15 minutes and the same
revision becomes claimable. Partial `.part` files are never admitted. A Drive
reconciliation discovers anything uploaded during downtime.

Quarantined reports retain a reason in `source_revisions.last_error`. Fix the
upstream file and upload it as a new revision; do not edit the registry by hand.

## 7. What must exist before training is enabled

- approved base model and exact immutable revision;
- objective-specific prompt/target builder (observation imitation versus
  performance/preference learning must not be conflated);
- outcome sidecars and cohort-normalization rules for claims about virality;
- near-duplicate transcript detection before split finalization;
- fixed editorial, factuality, memorization, and safety evaluations;
- checkpoint/resume implementation with incomplete-checkpoint detection;
- explicit owner approval and a separate model-promotion gate.

Until these are implemented and reviewed, queued manifests must remain inert.
