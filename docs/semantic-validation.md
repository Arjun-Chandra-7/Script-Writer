# Semantic quality validation

## Purpose and current reality

The semantic gold validation architecture is **fully implemented** and its end-to-end workflow is **verified with fixtures**.
However, semantic reconstruction quality is **NOT YET HUMAN VALIDATED** on real production data: the repository currently contains only **one real compiled ScriptIntelligenceRecord**, not the 100–200 diverse real records required for a human gold study.

```text
Client Brain -> client.json -> acquisition -> Extractor -> Script Intelligence
  -> Semantic Reconstruction -> Human-Validated Training Brief
  -> Training Example Compiler -> immutable dataset -> future model training
```

Status breakdown:
- **IMPLEMENTED**: Stratified gold sampling, blind/assisted review payloads, multi-reviewer annotation schemas, adjudication, immutable gold manifest freeze, gold training exclusion (exact + SimHash near-duplicate), semantic adapter benchmarking, field-level classification metrics, confusion matrices, confidence calibration & ECE, leakage audits, anomaly detection, adversarial client contamination tests, input ablation (5 variants), staged vs single-pass comparisons, pilot workflow, full corpus projection (7,500 sources), fail-closed quality gates, training dataset build integration.
- **WORKFLOW VERIFIED WITH FIXTURES**: All CLI commands, data structures, exclusions, and gate evaluations pass in automated test suites (95 tests passing).
- **NOT YET HUMAN VALIDATED**: 100–200 real human gold reviews have not taken place.
- **NOT YET RUN AT FULL CORPUS SCALE**: Full 7,500-source semantic inference has not occurred.

Fixtures or synthetic reviewer data **CAN NEVER** satisfy `semantic_reconstruction_gold_quality_verified`. Training readiness remains fail-closed.

---

## Gold workflow

### 1. Stratified Gold Selection
```bash
script-writer gold sample \
  --intelligence records/*.json \
  --count 150 \
  --seed gold-v1 \
  --output review/selection.json
```
Selection balances duration, transcript length, speaking rate, hook mechanism, beat pattern, CTA status, transcript confidence, and semantic complexity. The selection manifest is permanently marked `evaluation_only: true`.

### 2. Blind and Assisted Review Payloads
```bash
# Blind review: reviewer cannot see semantic model proposal
script-writer gold review \
  --intelligence RECORD.json \
  --client client.json \
  --mode blind \
  --output review/blind-RECORD.json

# Assisted review: reviewer sees model proposal for verification
script-writer gold review \
  --intelligence RECORD.json \
  --client client.json \
  --mode assisted \
  --output review/assisted-RECORD.json
```

### 3. Reviewer Annotations
Reviewers record judgments per field:
```bash
# Value annotation (controlled or free-text)
script-writer gold annotate \
  --record-id RECORD_ID \
  --reviewer reviewer-a \
  --mode blind \
  --field topic \
  --status value \
  --value "DOGE government workforce cuts" \
  --output review/annotations.json

# Qualitative / abstention annotation
script-writer gold annotate \
  --record-id RECORD_ID \
  --reviewer reviewer-b \
  --mode blind \
  --field topic \
  --status too_broad \
  --notes "Needs to specify public workforce impact" \
  --output review/annotations.json
```

Supported annotation statuses:
- `value`: provides human ground truth in `acceptable_values` or `human_value`
- `accepted`: accepts model proposal
- `partial`: proposal is partially correct
- `wrong`: proposal is incorrect
- `too_broad`: proposal is overly general
- `too_narrow`: proposal is overly specific
- `too_vague`: proposal lacks concrete substance
- `too_detailed`: proposal contains unnecessary micro-detail
- `leakage`: proposal copies target script phrases
- `unknown`: cannot be determined from evidence
- `ambiguous`: source has multiple plausible interpretations
- `not_inferable`: cannot be inferred from source evidence
- `reject`: explicitly rejected proposal

### 4. Multi-Reviewer Adjudication
Disagreements between reviewers are resolved without deleting or overwriting the original reviewer data:
```bash
script-writer gold adjudicate \
  --annotations review/annotations.json \
  --record-id RECORD_ID \
  --field topic \
  --resolver lead-reviewer \
  --status value \
  --value "DOGE government workforce cuts" \
  --notes "Canonical synthesis of reviewer-a and reviewer-b" \
  --output review/adjudications.json
```

### 5. Immutable Gold Freeze
```bash
script-writer gold freeze \
  --selection review/selection.json \
  --annotations review/annotations.json \
  --adjudications review/adjudications.json \
  --output review/frozen
```
Generates a content-addressed manifest `gold-manifest-<sha256>.json` containing:
- Selection records, hashes, and strata
- Complete review and adjudication histories
- Resolved gold annotations
- Reviewer metadata
- `evaluation_only: true`
- Training exclusions: `source_content_hashes`, `transcript_simhash64`, and `transcript_sha256`

---

## Gold training exclusion

All gold records, exact duplicates, and SimHash lexical near duplicates (Hamming distance $\le 3$) are permanently excluded from future SFT training builds.

When compiling training datasets:
```bash
script-writer dataset build \
  --client client.json \
  --intelligence records/*.json \
  --output dataset \
  --semantic-rules \
  --gold-manifest review/frozen/gold-manifest-<sha256>.json
```
The compiler verifies `evaluation_only: true` and rejects any gold source or near duplicate before generating training examples, recording `gold_evaluation_source_or_near_duplicate_excluded` in the rejections log.

---

## Field-level evaluation metrics & benchmarks

### Comparative Benchmark
```bash
script-writer gold benchmark \
  --intelligence records/*.json \
  --client client.json \
  --annotations review/annotations.json \
  --output reports/benchmark.json
```

Evaluation outputs:
- **Controlled fields** (`content_objective`, `content_format`, `cta_intent`):
  - Truth counts, predicted counts, true positives
  - Precision, Recall, F1 per class
  - Macro F1 across classes
  - Confusion matrix pairs (`prediction -> truth`) and structured matrix grid
- **Abstention & Unsupported Inference**:
  - Correct abstention count and rate
  - Unnecessary abstention count and rate
  - Unsupported inference count and rate (false inference)
- **Confidence Calibration**:
  - Categorical bands: `high` ($\ge 0.8$), `medium` ($0.6 \le c < 0.8$), `low` ($< 0.6$), `unknown`
  - Numeric buckets: `0.0-0.2`, `0.2-0.4`, `0.4-0.6`, `0.6-0.8`, `0.8-1.0`
  - Calibration gap per bucket and Expected Calibration Error (ECE)
- **Lexical Target Leakage**:
  - Field-level token Jaccard, 5-gram overlap, longest common sequence, high-risk flag count
- **Anomaly Detection**:
  - Central idea length $> 32$ words
  - Repeated central ideas across distinct sources
  - Generic topic collapse ($> 50\%$ frequency)
  - Identical audience everywhere ($> 50\%$ frequency)
  - Objective / format collapse ($> 80\%$ frequency)
  - 100% core field completion (zero abstention)
  - Suspiciously high confidence ($\ge 0.98$)
  - Inferred fields with empty evidence sources

---

## Comparative experiments

### Staged vs Single-Pass
Compares 3-stage reconstruction (`topic_central_idea`, `objective_format`, `conditional_context`) against single-pass inference under the identical canonical input contract:
- Request counts ($3\times$ vs $1\times$)
- Token consumption (input/output proxies)
- Latency and serial runtime
- Field acceptance and abstention rates

### Input Ablation
Benchmarks 5 canonical input variants:
1. `transcript`: clean transcript text only
2. `structure`: transcript + hook mechanisms + major beat roles
3. `delivery`: transcript + structure + delivery metrics
4. `full`: transcript + structure + delivery + persuasion + linguistic characteristics
5. `full_with_client`: full evidence + client context prior

```bash
script-writer gold ablation \
  --intelligence records/*.json \
  --client client.json \
  --annotations review/annotations.json \
  --output reports/ablation.json
```

Identifies the smallest input representation that preserves semantic fidelity.

### Adversarial Client Contamination Test
Verifies that supplying irrelevant client contexts (e.g. fitness vs finance vs technology) does not mutate protected source meaning (`topic`, `central_idea`, `content_objective`, `content_format`):
```bash
script-writer gold contamination-test \
  --intelligence RECORD.json \
  --clients client-fitness.json --clients client-finance.json --clients client-tech.json \
  --output reports/contamination.json
```

---

## Pilot & full-corpus projection

### Bounded Pilot
Runs a limited pilot (target 250–500 real sources) to measure real execution metrics:
```bash
script-writer gold pilot \
  --intelligence records/*.json \
  --client client.json \
  --limit 500 \
  --output reports/pilot.json
```
Reports:
- Source counts, success count, failure count, failure rate
- Eligibility for all 7 training objectives:
  1. Full script
  2. Hook
  3. Continuation
  4. Structure
  5. Section
  6. CTA
  7. Measurable style
- Abstention breakdown across all fields
- Cache hits, misses, and hit rate
- Measured input and output token proxies
- Latency and serial runtime

### Full Corpus Projection (7,500 Sources)
Extrapolates measured pilot statistics to full corpus scale:
```bash
script-writer gold estimate \
  --pilot reports/pilot.json \
  --sources 7500 \
  --input-price-per-million 2.50 \
  --output-price-per-million 10.00 \
  --output reports/projection.json
```
Generates `best` (0.8x), `expected` (1.0x), and `worst` (1.3x) scenarios for requests, tokens, serial latency, and cost. If pricing is omitted, cost remains null without fake estimates.

---

## Quality gate evaluation

Apply the frozen precision-first thresholds from `fixtures/semantic/quality-gates.v1.json`:
```bash
script-writer gold report \
  --benchmark reports/benchmark.json \
  --reviewed-sources 150 \
  --gates fixtures/semantic/quality-gates.v1.json \
  --output reports/readiness.json
```

Gate criteria:
1. `minimum_human_reviewed_sources`: $\ge 100$ human-reviewed sources (fixtures rejected)
2. `topic_acceptance`: $\ge 0.90$
3. `central_idea_acceptance`: $\ge 0.85$
4. `objective_acceptance`: $\ge 0.80$
5. `format_acceptance`: $\ge 0.80$
6. `unsupported_audience_inference`: $\le 0.05$
7. `false_cta_inference`: $\le 0.02$
8. `zero_high_severity_leakage`: 0 high-risk leakage instances

Missing metrics fail closed. Only when all gates pass does `semantic_reconstruction_gold_quality_verified` become true.

---

## Training readiness integration

When (and only when) real human validation passes:
```bash
script-writer dataset build \
  --client client.json \
  --intelligence records/*.json \
  --output dataset \
  --semantic-rules \
  --gold-manifest review/frozen/gold-manifest-<sha256>.json \
  --semantic-quality-report reports/readiness.json
```

If the report is missing, invalid, or fails any gate, `semantic_reconstruction_gold_quality_verified` remains false, blocking training readiness.
