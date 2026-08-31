# Script Writer architecture

## 1. Scope and non-goals

This component is a narrowly scoped Script Writer sub-agent. It receives
evidence reports from an upstream video extractor and eventually produces
short-form scripts plus explicit creative rationale. It does not publish,
change channel policy, self-approve fine-tuning, or infer success from a video's
existence.

This phase builds ingestion, provenance, dataset, and job-control foundations.
It does **not** download a base model, tokenize a corpus, allocate a GPU, run a
training step, or promote a model.

## 2. Evidence contract

### Extractor report (required)

The supplied report format contains useful observed evidence:

- stable report/source identity and extraction version;
- transcript text, word timing, pauses, delivery, and confidence;
- measured editing, audio, color, overlay, and cross-modal events;
- provenance, verification state, exclusions, and known deferred analysis.

The adapter admits a report only when JSON is parseable, extraction and
transcript states are complete, identity fields exist, duration and transcript
are plausible, and the report is not an exact content duplicate. Values marked
unverified, inferred, or excluded remain contextual features and cannot become
ground-truth labels.

### Outcome sidecar (required for performance weighting)

An extractor report is not a virality label. A companion outcome record should
eventually be joined by `report_id` and include at least:

```json
{
  "schema_version": "1.0",
  "report_id": "86c1671e8a3a7b46",
  "platform": "youtube_shorts",
  "channel_id_hash": "...",
  "niche": "technology",
  "published_at": "2026-08-01T12:00:00Z",
  "measured_at": "2026-08-08T12:00:00Z",
  "impressions": 100000,
  "views": 64000,
  "viewed_vs_swiped": 0.71,
  "average_percentage_viewed": 0.88,
  "retention_curve": [[0, 1.0], [3, 0.82], [30, 0.61]],
  "shares": 2100,
  "saves": 900,
  "likes": 7800,
  "comments": 430,
  "rights": {"training_allowed": true, "basis": "owned"}
}
```

Raw counts must not be treated as directly comparable across channel size,
niche, platform, geography, publication time, and video age. The future label
builder must normalize within an appropriate cohort and retain the original
measurements. Rights are a hard gate, not a score.

## 3. Ingestion and idempotency

The service performs a paginated, read-only listing of JSON files whose parent
is the configured folder. A full reconciliation scan is the correctness path;
notifications may later reduce latency but are never the source of truth.

For every Drive item it records:

1. Drive file ID (remote identity).
2. revision key (`md5Checksum`, or modified time and size when unavailable).
3. SHA-256 of the downloaded bytes (content identity).
4. report ID and upstream source content hash (semantic/group identity).

Only one trainable report row is admitted per upstream source content hash. A
re-extraction with changed JSON bytes remains preserved as a source revision but
links to the existing semantic example, so it cannot be silently trained twice.

This handles four distinct cases safely:

| Situation | Result |
|---|---|
| Same file appears in every scan | No download when revision is unchanged |
| File renamed | Same Drive ID/revision; no new example |
| Same bytes uploaded under a new ID | SHA-256 collision links to existing artifact |
| File replaced in place | New immutable revision is downloaded and validated |

Download goes to a unique `.part` path, is flushed and hashed, then atomically
renamed. Database transitions happen in short transactions. Startup reclaims
expired leases and reconciles temporary files, making power loss and process
restart safe.

## 4. Durable state machines

SQLite in WAL mode is sufficient for one host and 10,000 reports. The schema is
designed so PostgreSQL can replace it if multiple writer hosts are introduced.

```text
source revision: discovered -> downloading -> downloaded
                                      |             |
                                      +-> retry     v
                                                validated -> admitted
                                                     |
                                                     +-> quarantined

training run: queued -> preparing -> running -> evaluating -> promotable
                          |           |              |
                          +-----------+-> failed/resumable
```

Unique constraints enforce content identity, report revision identity, dataset
membership, and "new example consumed by run" identity. States are updated only
after the corresponding durable artifact exists.

## 5. Dataset snapshots

Each admitted report produces a canonical example projection rather than
feeding the 800 KB raw JSON directly to the model. The raw file remains
immutable for audit/reprocessing. The projection contains:

- clean transcript and timing-derived delivery features;
- only measured/verified creative features with provenance;
- source/extractor schema versions and confidence policy;
- outcome and cohort-normalized scores when available;
- rights and exclusion decisions;
- a target script representation only when the training objective supports it.

Split assignment uses a keyed hash of the strongest stable group identity
(upstream video content hash, falling back to report ID). Assignment is written
once. All revisions or duplicates of a source stay in the same split.

Snapshots are immutable JSONL manifests with a SHA-256 digest. A training run
references one exact snapshot, base model revision, adapter revision,
configuration digest, code commit, and random seed. Reconstructing a run never
depends on the current contents of Drive.

## 6. Batch arrival and continual updates

The orchestrator permits ingestion while training but permits only one active
training run. When 500 files arrive:

1. all visible files are reconciled and admitted or quarantined;
2. eligible, never-consumed training examples are reserved transactionally;
3. the run snapshot combines those new examples with a stratified replay sample
   of older examples;
4. files arriving during the run remain unconsumed for the next run;
5. interruption resumes the same run from its last complete checkpoint;
6. only a successful, evaluation-gated run marks its new-example reservation
   consumed;
7. a failed run can be retried or explicitly abandoned, releasing its
   reservation without losing the examples.

Failed/resumable and evaluated-but-not-promoted runs count as unresolved. They
block creation of a later run so adapter lineage cannot jump past a failure or
an unreviewed candidate while newer files continue to ingest safely.

This is at-least-once discovery with effectively-once dataset admission. It
does not claim impossible exactly-once execution across GPU/process failure;
instead it makes every repeated operation idempotent.

## 7. Training strategy (future, approval required)

Do not train a language model from scratch on 10,000 short videos. Start with a
strong instruction-tuned base whose license permits the intended use, freeze
the backbone, and compare prompt/RAG baselines against LoRA/QLoRA supervised
fine-tuning. Select the exact base only after hardware, languages, commercial
license, context length, and latency requirements are known.

Recommended staged objectives:

1. **Structure baseline:** retrieval plus a carefully specified generation
   contract, before weight changes.
2. **Supervised adaptation:** high-quality, rights-cleared script examples;
   loss should apply to the assistant target, not source metadata.
3. **Preference optimization:** human/editor preference pairs and reliable
   performance-matched pairs; never raw view-count ranking.
4. **Continual updates:** new high-quality data plus replay, conservative
   learning rate, fixed regression suite, and adapter lineage.

The generated contract should separate spoken script, on-screen text, beat
timing, visual suggestions, claims needing verification, and rationale. That
keeps downstream agents from parsing an unstructured blob.

## 8. Evaluation and promotion

"Banger" is a product outcome, not a single offline metric. Promotion requires:

- schema/constraint pass rate and factuality/safety checks;
- fixed, never-trained-on challenge sets by niche and format;
- novelty and memorization checks against the corpus;
- blinded pairwise ratings from qualified human editors;
- regression against the currently promoted model;
- later, randomized online experiments using comparable channel cohorts.

Perplexity alone is not an acceptance metric. A candidate can be lower-loss and
still write worse hooks, hallucinate claims, or imitate source scripts too
closely. Promotion writes a new registry pointer atomically; rollback only
changes that pointer. Checkpoints and evaluation evidence remain immutable.

## 9. Security and operations

- Use a service account with Viewer access to only the input folder.
- Mount credentials as a secret; never commit them or copy them into logs.
- Validate maximum byte size before and during download.
- Treat report text as untrusted data, not instructions to the orchestrator.
- Log redacted IDs, state transitions, hashes, counts, and reasons.
- Export backlog age, scan lag, retry count, quarantine count, active run,
  checkpoint age, and evaluation/promotion status.
- Back up the state database and immutable manifests; test restoration.
- Keep raw reports because future schema adapters must be reproducible.

## 10. Explicit owner gates

The main context classifies model fine-tuning as approval-required. Therefore:

- ingestion and validation may run automatically;
- manifest creation may run automatically under configured thresholds;
- launching a training process requires an explicit owner-controlled enable
  switch and an approved run manifest;
- promotion requires an evaluation report and a separate approval decision.
