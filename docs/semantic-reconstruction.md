# Semantic intent reconstruction

## Purpose

This layer reconstructs a `MinimumSufficientTrainingBrief`, not a creator's literal prompt. It answers only what a future Script Writer minimally needs to produce a similar kind of script. It is an evidence-quality experiment, not a training system.

```text
ScriptIntelligenceRecord + ClientTrainingContext
  -> staged SemanticIntentAdapter
  -> MinimumSufficientTrainingBrief
  -> per-field leakage/compression validation
  -> objective-specific training eligibility
  -> human gold review/evaluation
```

## Implemented

- versioned brief schema with eight core and twelve optional evidence envelopes;
- three stages: topic/central idea, objective/format, then conditional context;
- strict adapter contract, retries, persistent request cache, and fail-closed invalid response behavior;
- deterministic rule adapter, mock adapter, and provider-neutral HTTP structured adapter boundary; no remote request is made without an explicit caller;
- field-level lexical leakage checks and 32-word central-idea compression cap;
- compact JSON gold annotations supporting value, multiple acceptable values, unknown, ambiguous, and not inferable;
- field metrics, abstention/error reporting, agreement helper, estimate command, and immutable streaming shard writer.

The rule adapter is an unvalidated baseline. It can make candidates eligible but cannot make a dataset training-ready; only gold-set evidence can support that.

## Client context rule

Client context is supplied as an explicitly labeled prior. The prompt forbids using it as proof. Audience remains unknown in the real example even though the fixture client has an audience; the source transcript does not establish it.

## Commands

```bash
script-writer semantic infer --intelligence RECORD.json --client client.json --cache state/semantic-cache.sqlite3 \
  --output review/brief.json
script-writer semantic estimate-corpus --records 7500 --average-words 160
script-writer semantic review --record-id RECORD_ID --output annotations.json \
  --field target_audience --status unknown --reviewer reviewer-a
script-writer semantic evaluate --briefs review/brief.json --annotations annotations.json
script-writer semantic error-analysis --briefs review/brief.json --annotations annotations.json
script-writer dataset shard compiled-examples.jsonl --output shards --size 1000
```

`semantic evaluate` consumes a list of `{record_id, brief}` objects plus gold annotations. Reviewers should label only what the source supports and select unknown/ambiguous freely. There is no live-provider credential configured in this repository.

## Known boundaries

The production HTTP adapter architecture is tested with mocks, not live API traffic. Rule-based semantic quality has only fixture evaluation, not a 100–200-record human gold study. Cross-language duplicate detection is still an interface-level future extension. Shard writing is bounded-memory, but source compilation itself still needs disk-backed staging before a 50,000-source claim.
