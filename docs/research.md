# Research and extractor audit

Accessed and updated 2026-09-01. Primary/official sources are preferred.

## Real extractor report findings

The supplied 811,349-byte report was inspected directly rather than treated as
a generic JSON blob. It contains:

- report ID, upstream content hash, duration (59.118 s), resolution, FPS, and
  extractor version;
- 166 aligned transcript words, 11 timed sentences, language probability,
  rolling delivery windows, 16 pauses, and six emphasis candidates;
- five verified edit boundaries and editing summary;
- 65 OCR/caption events plus transcript alignment and explicit OCR errors;
- audio measurements/events, silence ranges, color/visual evidence, five
  cross-modal edit events, and a 217-item master timeline;
- confidence/provenance policy and deferred speaker diarization;
- ten upstream semantic sections whose status is explicitly
  `structural_hypothesis_unverified`.

Important absences: platform, publication timestamp, client/account, niche,
topic, audience, reliable content format, outcome analytics, CTA truth, factual
claim labels, speaker diarization, and verified story/persuasion roles. The
compiler emits unknown for these unless an enrichment context or future
versioned semantic analyzer supplies them.

The upstream semantic labels are useful hypotheses but cannot be upgraded to
observations. Caption OCR confidence also does not imply transcript alignment;
the sample contains high-confidence OCR strings with zero lexical alignment.

## Migration from historical assumptions

| Historical assumption | Current decision |
|---|---|
| Agentic YouTube portfolio/channels | One client/account context; Instagram/Reels first |
| Agents are niche specialists | Three peers share the same intelligence and may vary outputs |
| YouTube Shorts outcome fields | Platform-aware outcome envelope with Instagram/Reels metrics |
| Raw reports become training examples | Raw -> canonical intelligence -> filtered training candidates; no model training |
| Automatic 500-file training proposal | Watcher compiles/indexes only; auto-proposal disabled |
| Universal 90/5/5 | Objective-specific policies with grouped leakage protection |
| Fine-tuning architecture is the near-term center | Retrieval-first generation and evaluation are current |

The current training-data phase changes the fourth row further: canonical
intelligence may now produce supervised candidates, but only after compact
client conditioning, intent reconstruction, target-leakage checks, quality
filtering, and universal source-cluster splitting. Candidate existence does not
mean eligibility.

## Training-intent reconstruction findings

Supervision requires an intent-to-output mapping, not merely an analyzed target.
The real sample demonstrates the hard case: language, duration, measurable
style, hook hypotheses, and beat hypotheses exist, while topic, central idea,
audience problem, creator objective, format, and CTA do not. Surface entity
hints cannot safely replace a reviewed topic. The conservative compiler
therefore generated 15 inspectable candidates but exported only the continuation
candidate, whose observed hook is valid conditioning for a non-overlapping
remainder target. Fourteen candidates were rejected for
`missing_reliable_topic_or_central_idea`.

This is the expected quality behavior. The next research requirement is a
versioned semantic intent adapter evaluated against human-reconstructed minimum
briefs—not a generic request to "write a viral script."

## Semantic reconstruction baseline

The rule baseline reconstructs the supplied sample as topic “DOGE government
workforce cuts,” central idea “arbitrary workforce cuts harm public workers,”
objective “persuade,” and format “commentary.” These are explicitly
`model_inference` envelopes, not ground truth. It abstains on target audience,
audience problem, and CTA. The fixture gold annotation accepts this output, but
one fixture is not a human quality study; a 100–200-record stratified review is
required before claiming calibrated reconstruction quality.

The historical `.docx` remains unchanged for provenance.

## Retrieval choices

SQLite FTS5 provides built-in full-text indexing and BM25 ranking, which is a
good operational baseline for 5,000–50,000 compact records without another
service. [SQLite FTS5 documentation](https://www.sqlite.org/fts5.html)

Dense-vector search is abstracted behind `EmbeddingProvider`. The current local
signed hashing vector is deterministic and dependency-free, not a production
semantic model. At this corpus size, exact reranking of a bounded lexical/
metadata candidate set is practical. Faiss remains an optional future backend;
its own guidance notes that simple/exact approaches can be preferable when
query volume or corpus size does not justify index-building complexity.
[Faiss project](https://github.com/facebookresearch/faiss),
[index-selection guidance](https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index)

Only the semantic projection is embedded: hook, mechanisms/devices, and clean
script. Frame arrays, word acoustic arrays, full captions, and the raw 800 KB
report are excluded because they add cost/noise and are already available via
provenance when needed.

## Outcomes and Instagram

Meta's official Instagram API workspace documents media insights and describes
Reels shares, while Meta has also described total and average watch-time
metrics for Reels. Metric availability and names can change, so the outcome
schema permits platform-native extensions and never zero-fills unavailable
fields. [Official Meta Instagram API workspace](https://www.postman.com/meta/instagram/overview),
[Instagram Insights collection](https://www.postman.com/meta/instagram/folder/23987686-f659d7d1-d74c-44e4-9192-9b1e8694c511),
[Meta Reels watch-time announcement](https://about.fb.com/news/2023/04/instagram-reels-trending-audio-and-gifts-updates/)

Views, reach, plays, watch time, and interactions are not comparable across
accounts, follower sizes, niches, paid/organic distribution, geography, age, or
measurement windows. The repository stores raw measurements and accepts an
explicit externally computed cohort-normalization object; it does not invent a
virality score.

## Leakage and memorization

Exact/source dedupe alone does not prevent a rewritten or clipped transcript
from crossing evaluation boundaries. The objective-layer leakage guard adds
exact transcript, derived-source, and banded SimHash grouping. The dataset
deduplication literature also reports reduced memorization and more reliable
validation after near-duplicate removal.
[Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)

Generation context marks source text analysis-only. Offline evaluation measures
contiguous and five-gram overlap. These are deterministic warning/gate signals,
not a complete originality judgment.

Training conditioning receives stricter checks: exact long sentences, longest
contiguous token sequences, target five-gram coverage, and token-set Jaccard.
The real full-script candidate had Jaccard 0.057325, longest match 1 token, and
zero shared target five-grams. It passed leakage but failed brief sufficiency.

## Evaluation boundary

Constraint, schema, timing, reference-integrity, and overlap checks can be
deterministic. Hook quality, coherence, client fit, factual grounding, and
originality require human or named/versioned model judgment. The current
evaluator explicitly records them as not evaluated and does not collapse unlike
dimensions into a fake aggregate score.

## Open research before a production generator

1. Human-reconstruct minimum sufficient briefs for a stratified source sample,
   with explicit unknowns, then evaluate a versioned intent adapter.
2. Establish inter-rater agreement and uncertainty policy for editor judgments.
3. Compare learned embedding providers against labeled retrieval queries.
4. Define client-context storage, factual source provenance, and privacy rules.
5. Ingest Instagram Insights with stable account/time cohorts and metric-version
   tracking.
6. Benchmark retrieval-first production generators using frozen fixtures and
   blinded editor comparisons before considering any weight changes.
