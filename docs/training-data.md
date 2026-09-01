# Training-data compiler

## Boundary

This phase ends at verified datasets. It contains no base-model choice, trainer,
LoRA/QLoRA implementation, GPU dependency, checkpoint, or weight update.

```text
Client Brain -> client.json -> RAG Searcher -> relevant videos -> Extractor
     -> ScriptIntelligenceRecord -> intent reconstruction
     -> ScriptTrainingExample candidates -> quality/leakage filtering
     -> duplicate clustering -> universal grouped split
     -> objective JSONL + immutable manifests -> TrainingReadinessReport
     -> FUTURE MODEL TRAINING
```

## Contracts

`ClientTrainingContext` is a versioned, content-addressed projection of only
script-relevant client facts: positioning, niche/subniche, audience, pillars,
voice/tone, offers, beliefs, vocabulary, CTA rules, factual constraints,
prohibitions, recurring narratives, and differentiation. Missing values are
explicit unknowns. Examples reference the projection ID and hash instead of
duplicating a giant client file.

`ScriptTrainingExample` contains source identity, universal group/split,
objective, client reference, reconstructed brief, abstract creative plan,
target, provenance, leakage metrics, decomposed evidence quality, eligibility,
reasons, warnings, and review state. Core JSONL is provider-neutral; a later
adapter may render system/user/assistant messages without changing this schema.

## Intent reconstruction

`IntentReconstructor` is a versioned protocol. The dependency-free
`MinimumConditioningReconstructor` copies observed client facts and canonical
measurements, preserves hook/beat hypotheses as heuristics, extracts only
surface-form concept hints, and emits unknown for semantic facts it cannot
support. It never claims to recover the original creator prompt, business goal,
demographics, or unstated facts.

Full script, hook, structure, section, CTA, and style objectives require a
reliable topic or central idea. Continuation can instead use a non-overlapping
observed opening. A production semantic adapter must improve coverage without
weakening provenance or uncertainty.

## Objectives

| Objective | Conditioning | Target | Required evidence |
|---|---|---|---|
| `full_script_sft` | client + brief + plan | cleaned spoken script | sufficient brief, usable script |
| `hook_generation` | client + topic/audience/intent | bounded hook | reliable hook boundary |
| `continuation` | brief + observed opening | remaining script | non-empty non-overlapping remainder |
| `structure_planning` | client + brief, target progression removed | abstract role sequence | at least three aligned roles |
| `section_generation` | brief + abstract role | section text | usable aligned section |
| `cta_generation` | brief + CTA intent | CTA text | reliably detected CTA |
| `style_conditioned` | brief + measured statistics | script | concrete measurable style only |

One source may create many candidates, but all inherit one source-cluster split.
Meaningless paraphrase multiplication is not implemented.

## Leakage and duplicate controls

Conditioning/target comparison records token-set Jaccard, target five-gram
coverage, longest contiguous match, and exact target sentences of eight or more
tokens. Thresholds are stored with every result. High severity excludes the
candidate; warnings remain visible.

Exact source/transcript duplicates are suppressed before objective expansion.
Near duplicates use banded 64-bit SimHash and union-find clustering. Derived
examples use the same `source_group_id`; a universal salted hash assigns one
90/5/5 split independent of objective. Optional source/cluster caps retain the
highest evidence-quality candidates and are written into manifest metadata.
No automatic balancing or translated-copy detector is claimed.

## Evidence quality and eligibility

`training_evidence_quality` is a documented weighted composition:

- transcript integrity: 25%;
- brief completeness: 25%;
- provenance confidence: 20%;
- plan completeness: 15%;
- leakage safety: 15%.

It predicts neither creativity nor performance. Eligibility is separately
classified as `eligible`, `eligible_with_warning`, or `ineligible`, with reasons
such as short script, low transcript confidence when available, malformed
timing, weak hook/section boundary, missing brief subject, leakage, duplicate,
sampling holdout, or human review hold.

## Build, audit, and review

`script-writer dataset build` performs deterministic fixed-snapshot clustering,
uses a versioned SQLite cache for unchanged record/client/compiler combinations,
and writes content-addressed JSONL, rejection JSONL, objective manifests, compact
review sources, audit, and readiness report. It consumes canonical intelligence,
not raw extractor arrays.

The audit reports sources, eligibility, objectives, splits, leakage, exact and
near duplicates, language, duration, word count, speaking rate, topic, format,
hooks, structures, CTA presence, unknowns, provenance, evidence quality,
rejections, sampling, imbalance warnings, and cache hits/misses.

`dataset show` renders one source transcript, client projection, brief, plan,
target, provenance, leakage, eligibility, and review status. `dataset review`
records accept/reject/flag decisions. Rejects and flags are excluded on rebuild.

## Readiness gates

The machine report cannot become `training_ready` unless all gates pass:

- every candidate validates;
- no source or duplicate cluster crosses splits;
- no exported high-severity target leakage exists;
- objective manifests exist and are immutable;
- compiler and client context are frozen;
- all rejections have reasons;
- fixed-input regeneration is deterministic;
- validation and test sets exist and their manifest membership is frozen;
- configurable minimum eligible examples and human inspections are met.

The one-record committed demonstration correctly fails the last three practical
corpus gates. Passing gates is necessary, not sufficient, for choosing a model.

## Scale

The large extractor report is parsed upstream once. Training compilation reads
compact `ScriptIntelligenceRecord` files, caches per-record transforms, stores
immutable outputs, and can resume unchanged transforms after interruption.
Near-duplicate identity state is compact. The current CLI still materializes the
fixed snapshot's compact records and candidate examples while assembling
manifests. That is verified for the supplied sample but is not yet a proven
bounded-memory 50,000-source build. Disk-backed staging/sharded export and a
7,500-source benchmark are required before a production corpus run. Multi-host
concurrent builds would additionally require a server database and distributed
snapshot lock.

Future work: human gold intent reconstruction, a validated semantic adapter,
translated-copy detection, outcome-aware sampling interfaces, preference/ranking
examples, provider-specific export adapters, and only then model selection and
training experiments.
