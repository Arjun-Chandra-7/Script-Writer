# Script Intelligence architecture

## Product scope

VIRALYST currently targets short-form content for one client/account context at
a time, with Instagram/Reels as the primary platform. Platform is a metadata
dimension rather than a hardcoded dependency. Three creative agents may use the
same intelligence, retrieval, and generation contracts and independently
produce different creative candidates.

The historical Agentic YouTube Evolution System document is retained only as
project history. Portfolio/channel evolution, different specialist agents,
YouTube-only metrics, and automatic training are not current assumptions.

## Current data flow

```text
Extractor
  raw multimodal evidence report
        |
        v
Ingestion
  Drive revision -> atomic raw artifact -> validation -> dedupe/quarantine
        |
        v
Canonical Evidence
  validated identity/transcript + immutable raw provenance
        |
        v
Script Intelligence compiler v1
  deterministic features + versioned semantic adapter + uncertainty envelopes
        |
        v
Corpus Index
  metadata | FTS5 | mechanisms | fingerprints | embedding projections
        |
        v
Retrieval
  filters + lexical + semantic interface + structural similarity + diversity
        |
        v
Generation Context
  bounded evidence summaries, explicit anti-copy and grounding instructions
        |
        v
Script Writer interface
  structured ScriptGenerationResult (deterministic demonstrator today)
        |
        v
Offline evaluation
        |
        v
Client context projection -> intent reconstruction -> training examples
        |
        v
quality/leakage filters -> universal grouped splits -> immutable manifests
        |
        v
TrainingReadinessReport -> future training only
```

## Ingestion invariants retained

The existing ingestion design remains the durable boundary:

- Drive file ID + revision key track remote identity;
- report-byte SHA-256 tracks exact content;
- `source.content_hash` tracks the semantic source video;
- temporary downloads are size-bounded, flushed, hashed, and atomically moved;
- SQLite WAL transactions, leases, and retry/quarantine states survive restart;
- re-uploads and re-extractions preserve raw revisions but map to one semantic report.

Compilation occurs from the already-parsed report during new ingestion. Legacy
admitted reports use `script-writer compile`; compiled records are cached by
source artifact, compiler version, analyzer version, and record SHA-256. A
compiler failure never destroys or rejects an otherwise valid raw report.

## ScriptIntelligenceRecord v1

The authoritative machine contract is
`schemas/script-intelligence-record.v1.schema.json`. Every interpretive value
uses one of five evidence classes:

| Evidence type | Meaning |
|---|---|
| `observed` | Directly supplied/measured by the extractor or enrichment context |
| `deterministic_derivation` | Reproducible calculation from cited source fields |
| `heuristic_inference` | Versioned rule/hypothesis, never ground truth |
| `model_inference` | Output of a named/versioned future model |
| `unknown` | Evidence is absent or reliability is insufficient |

Source references use JSON-style paths plus timestamps where available.
Confidence is included only when meaningful; unknown values are JSON `null` and
must include a reason.

### Deterministic compiler output

- cleaned and normalized transcript;
- compiler word count, duration-normalized rate, sentence distributions;
- vocabulary proxies, person usage, question/numerical density, transitions,
  repetition, short-sentence share, and explicitly labeled lexical proxies;
- aligned cadence, extractor pace, pauses, candidate emphasis, silence, and
  large rolling-pace transitions;
- 10-second lexical information-density windows;
- verified edit proximity to sentence boundaries;
- preserved upstream caption/cross-modal relationships when evidence exists.

### Semantic output

`SemanticAnalyzer` is an explicit local protocol. The included conservative
rule implementation detects only supported surface mechanisms (for example,
question punctuation, contrast markers, explicit examples, direct address,
specificity markers, and stakes language). All are labeled heuristic. If an
analyzer raises, the compiler uses `NullSemanticAnalyzer`, preserves
deterministic fields, records the failure, and emits unknown semantic fields.

Topic, subtopic, content format, audience intent, full progression, storytelling
roles, persuasion roles, factual claims, and CTA are not reliably available in
the real sample. Optional `report.context` enrichment can provide platform,
topic, subtopic, content format, and audience intent as observed metadata.

## Index projections

Raw ~800 KB reports are never embedded. The record defines three bounded views:

1. `lexical_text`: cleaned spoken transcript for FTS5 retrieval.
2. `semantic_text`: hook + detected mechanisms/devices + transcript, for a
   pluggable semantic embedding provider.
3. `structural_fingerprint`: symbolic section, hook, retention, and script/edit
   tokens for exact Jaccard structural similarity.

The current `HashingEmbeddingProvider` is deterministic signed unigram/bigram
feature hashing. It supports tests, reproducibility, and a useful local lexical-
semantic baseline; it is not presented as a learned semantic model. A future
provider must be explicitly versioned and can rebuild alongside old vectors.

SQLite FTS5 handles lexical candidate generation. Metadata and mechanism tables
support platform, topic, format, duration, hook, and retention filters. Semantic
cosine and structural Jaccard rerank candidates, followed by bounded
diversification. Exact structural search scans only compact fingerprints, not
record JSON, which is appropriate for 5,000–50,000 records.

## Content evidence versus outcome evidence

`ScriptIntelligenceRecord` answers: **what creative/script mechanisms exist?**
It is useful without analytics.

`ShortFormOutcomeRecord` answers: **how did this post perform within a defined
account/platform/time cohort?** It supports Instagram/Reels metrics including
views, reach, plays, total/average watch time, likes, comments, shares, saves,
interactions, follows, profile visits, and optional retention/rate fields.

Raw counts are stored, never converted into an invented virality score. Any
normalization must name its cohort, method, sample size, measurement window, and
provenance. Outcome records later supply ranking, preference, and weighting
signals; they are not required for intelligence compilation or retrieval.

## Retrieval-first generation contract

`GenerationRequest` contains client context, niche, topic, objective, audience,
duration, platform, tone, format, constraints, factual context, inspirations,
banned patterns, CTA, and experimentation controls.

The pipeline constructs a corpus query, retrieves and diversifies evidence,
builds a bounded `GenerationContext`, and invokes a `ScriptGenerator` protocol.
`ScriptGenerationResult` separates spoken script, hook, timed sections,
on-screen text, delivery notes, useful visual cues, CTA, claims needing review,
mechanisms, evidence references, and rationale.

Retrieved text is analysis-only. Instructions prohibit copying; banned patterns
are checked after generation; offline evaluation measures contiguous and
5-gram overlap. The included deterministic outline generator demonstrates the
contract only. It is not the final creative agent.

## Training-data compiler

`ClientTrainingContext` projects only allowlisted script-relevant fields from
`client.json` and retains the source path/hash and field provenance. A versioned
`IntentReconstructor` contract supplies the minimum sufficient conditioning.
The local implementation preserves known fields and conservative surface forms,
but refuses to invent a topic, central idea, creator prompt, or business goal.

Each source may yield full-script, hook, continuation, abstract structure,
section, CTA, and measurable-style candidates. Unsupported candidates are
omitted; insufficient candidates remain reviewable but are excluded from
manifests with explicit reasons. Low-level color, spectral, and frame features
never enter script conditioning.

Conditioning is compared against its target using token-set overlap, five-gram
overlap, longest contiguous token sequence, and exact long-sentence matching.
High-severity matches are rejected. `training_evidence_quality` is a decomposed
measure of evidence readiness only and is never called creative quality,
performance, or virality.

## Objective-specific datasets and leakage

Training objectives have independent manifests, but every example derived from
one source/near-duplicate cluster receives one **universal split**. Objective
names are deliberately excluded from the split hash, preventing a hook in train
while its full script appears in validation.

| Objective | Eligibility | Split policy |
|---|---|---|
| Corpus understanding | Compiled content evidence | 100% searchable corpus; not training |
| Script generation SFT | Explicit prompt + approved target | 90/5/5 baseline |
| Preference/ranking | Comparable chosen/rejected pair | 80/10/10 baseline |
| Retrieval evaluation | Labeled query/relevance judgments | development/test |
| Challenge/regression | Owner-reviewed frozen fixtures | 100% challenge, never training |
| Performance learning | Cohort-aware outcome evidence | 80/10/10 baseline, later temporal review |

Split percentages are policies, not universal truths. `LeakageGuard` clusters by
source hash, exact transcript, derived-source linkage, and banded 64-bit
transcript SimHash before deterministic universal assignment. Manifest
validation rejects a cluster crossing splits. The legacy inert training-manifest
code remains for audit compatibility but is not invoked by the watcher and does
not constitute a valid SFT dataset.

## Offline evaluation

Deterministic checks implemented today:

- generation schema and required sections;
- approximate duration budget (explicit 2.5 words/second planning proxy);
- section timestamp monotonicity/bounds;
- banned-pattern and desired-CTA contract compliance;
- retrieval reference integrity and diversity reporting;
- longest contiguous phrase and five-gram overlap against retrieved sources;
- claims-review queue presence.

Hook quality, coherence, progression, originality, relevance, audience/client
fit, factual grounding, unsupported claims, CTA quality, and retrieval quality
remain `not_evaluated` until a human or named/versioned evaluator supplies
judgment. No aggregate quality or virality score is fabricated.

## Scale and recovery

For 5,000–50,000 reports:

- each giant raw JSON is parsed once during new ingestion;
- compact intelligence JSON and projections are cached;
- compilation and embedding rebuilds are incremental and batch-bounded;
- training compilation reads compact intelligence rather than giant extractor
  reports and caches unchanged record/client/compiler combinations in a
  versioned SQLite cache;
- FTS and metadata filter candidates before record loading;
- full structural comparison reads compact fingerprints only;
- the registry migrates deterministically to v2; the separate training cache
  carries and checks its own schema version;
- raw corruption is quarantined and compiler failures are separately observable;
- compiler, analyzer, embedding, schema, and fixture versions make rebuilds reproducible.

SQLite remains appropriate for one local writer/service at this scale. Multiple
concurrent writer hosts would require a transactional server database.

The intelligence/index path is batch-bounded. Training manifest assembly still
materializes a fixed compact snapshot in one process; disk-backed staging and a
full 7,500-source benchmark remain required before claiming bounded-memory
training-data builds at the top of the target range.

## Planned, not implemented

- learned semantic analyzer and production embeddings;
- Instagram Insights connector and cohort-normalization jobs;
- production semantic intent reconstructor and reviewed topic/brief labels;
- richer browser-based editor interface (compact CLI review exists today);
- production creative generator used by the three agents;
- frozen retrieval judgments and broader challenge fixtures;
- any training, fine-tuning, checkpoint, or promotion workflow.

The authoritative training-data design is in `docs/training-data.md`.
