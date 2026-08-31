# Viralyst Script Writer

Training-ready architecture for the Script Writer sub-agent in the Agentic
YouTube Evolution System. It continuously discovers evidence reports in a
Google Drive folder, downloads and validates each unique report exactly once,
builds immutable dataset snapshots, and queues resumable training work.

**Training is intentionally disabled in this repository revision.** The system
can ingest data and create training manifests, but it cannot launch a trainer or
change model weights until an owner explicitly enables a future training
runner.

## Why the extra gates?

The extractor report describes what happened in a video. It does not, by
itself, prove that the video was viral. Performance-aware training also needs an
auditable outcome record (retention, viewed-versus-swiped, shares, audience,
publish context, and measurement window). Unverified semantic hypotheses in an
extractor report are retained as context but never silently promoted to labels.

Read [the architecture](docs/architecture.md) and
[the research notes](docs/research.md) before enabling production ingestion.
Deployment and recovery instructions are in the
[operations guide](docs/operations.md).

## Planned operator flow

```text
Google Drive folder
        |
        v
read-only discovery -> atomic download -> SHA-256 dedupe -> validation
        |                                      |
        |                                      +-> quarantine + reason
        v
durable example registry -> fixed split assignment -> dataset snapshot
        |                                              |
        v                                              v
new examples + replay                         immutable manifest
        \______________________________________________/
                               |
                               v
                      queued training run
                    (execution disabled now)
```

The Drive folder ID from the provided URL is
`1loe1nchN4PFqkTzujZmbkB31E_DpCwqE`. Production access should use a Google
service account whose email has Viewer access to that folder. The watcher never
deletes or modifies Drive files.

## Safety properties

- Remote file revisions and downloaded bytes are tracked separately.
- Content SHA-256 is the final deduplication key; renames and re-uploads do not
  create duplicate examples.
- Downloads use temporary files and atomic rename, so interruption cannot admit
  a partial report.
- Split assignment is deterministic and grouped by source content, preventing a
  re-upload from crossing train/evaluation boundaries.
- Dataset versions and their membership are immutable.
- Only one training run may be active; newly arriving batches remain eligible
  for the next run.
- Failed/interrupted runs resume the same run and checkpoint. They do not create
  a second run over the same "new" examples.
- Model promotion is a separate, evaluation-gated operation.

## Repository status

The source context document and supplied extractor sample are preserved at the
repository root. Application code, tests, and deployment configuration are
added in subsequent commits so each logical change is independently auditable.
