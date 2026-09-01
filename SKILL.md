---
name: script-writer-control
description: Autonomous controller skill for the VIRALYST Script Writer subsystem. Ingests raw video extractor reports, compiles ScriptIntelligenceRecords, audits subsystem corpus suitability, runs semantic intent reconstruction, executes gold set validation workflows, and exports training datasets without external API dependencies.
---

# VIRALYST Script Writer Autonomous Agent Skill

This skill allows an autonomous AI agent to operate, validate, and query the **VIRALYST Script Writer** subsystem.

## Subsystem Purpose & Architecture

VIRALYST Script Writer is the intermediate intelligence layer between raw video extractor reports and training dataset compilation. It enforces strict separation of concerns:
- **Client Relevance $\neq$ Script Learning Value**: A video may be relevant to a client's niche but contain no spoken dialogue or script structure. It must not be discarded globally; it routes to Editing, Audio, or Visual subsystems.
- **Fail-Closed Evidence**: Unknown/absent values are preserved as `unknown`, never hallucinated.
- **Objective-Specific Eligibility**: Assesses suitability across 7 explicit objectives: `FULL_SCRIPT`, `HOOK`, `CONTINUATION`, `STRUCTURE`, `SECTION`, `CTA`, and `MEASURABLE_STYLE`.

---

## Agent Operations Playbook

### 1. Ingestion & Compilation
To compile raw extractor reports into canonical `ScriptIntelligenceRecord` (SIR) schemas:

```bash
# Compile a single extractor report JSON
script-writer compile-record <path_to_extractor_report.json> --output <output_sir.json>

# Ingest a sample into local state registry and index for search
SCRIPT_WRITER_STATE_DIR=state script-writer dry-run-sample <path_to_extractor_report.json>
SCRIPT_WRITER_STATE_DIR=state script-writer index
```

### 2. Subsystem Suitability & RAG Feedback
Evaluate objective-specific suitability on a batch of extractor reports and generate feedback for upstream acquisition:

```python
import json, glob
from script_writer.intelligence import ScriptIntelligenceCompiler
from script_writer.validation import parse_and_validate_report
from script_writer.corpus_suitability import (
    assess_from_sir, assess_silent_valid,
    corpus_suitability_report, corpus_feedback_report
)

compiler = ScriptIntelligenceCompiler()
records = []

for report_path in glob.glob("path/to/reports/*.json"):
    raw = open(report_path, "rb").read()
    raw_doc = json.loads(raw)
    try:
        report, _ = parse_and_validate_report(raw, split_salt="viralyst-script-writer-v1")
        compiled = compiler.compile(report, artifact_sha256="...")
        records.append(assess_from_sir(compiled.record, raw_report=raw_doc))
    except Exception:
        records.append(assess_silent_valid(raw_doc))

# Generate reports
suitability_summary = corpus_suitability_report(records)
feedback = corpus_feedback_report(suitability_summary, target_total=7500)
```

### 3. Gold Validation & Semantic Reconstruction
To run stratified sampling, review creation, or adapter benchmarking:

```bash
# Generate deterministic stratified gold sample from compiled SIR directory
script-writer gold sample --inputs state/compiled/ --count 25 --output state/gold/selection.json

# Run semantic reconstruction benchmark with local baseline adapter
script-writer gold benchmark \
  --selection state/gold/selection.json \
  --client fixtures/client.example.json \
  --mode staged \
  --output state/gold/benchmark.json

# Check semantic quality gate (fails closed if thresholds/human review counts aren't met)
script-writer gold report --benchmark state/gold/benchmark.json --gate fixtures/semantic/quality-gates.v1.json
```

### 4. Dataset Compilation & Audit
Compile objective-specific training datasets from audited SIRs:

```bash
script-writer dataset build \
  --client fixtures/client.example.json \
  --intelligence examples/compiled/86c1671e8a3a7b46.script-intelligence.v1.json \
  --output state/training-data

script-writer dataset audit state/training-data
```

---

## Key Constraints for Autonomous Execution
1. **Never Train Models**: Do not add LoRA, QLoRA, SFT trainers, or GPU code.
2. **Never Fabricate Annotations**: The production quality gate requires genuine human-reviewed sources.
3. **Preserve Subsystem Routing**: Silent videos must retain valid status and be routed to editing/audio/visual tags.
4. **All Tests Must Pass**: Validate test suite anytime changes occur with `pytest tests/ -q`.
