# VIRALYST Subsystem Corpus Suitability & Acquisition Architecture

## Core Architectural Principle

> **NOT EVERY CLIENT-RELEVANT VIDEO SHOULD TRAIN EVERY MODEL.**

A video retrieved by the upstream RAG Searcher may be 100% relevant to the client (e.g. Basecamp workflows, productivity culture, or competitor comparisons) while having near-zero supervisory value for the **Script Writer** (e.g., a silent UI screen recording with background music, or a 5-second aesthetic loop).

Discarding such videos globally would damage downstream subsystems that thrive on non-verbal signals:
* **Video Editing Intelligence**: cut pace, b-roll selection, screen occupancy, overlay timing.
* **Audio Intelligence**: music genre, ducking, speech rhythm, acoustic energy transitions.
* **Visual / Color Intelligence**: LUT grading, color palette, caption typography, motion dynamics.

---

## Subsystem Pipeline & Routing

```
                  Upstream RAG Searcher
                 (Client-Relevant Content)
                            │
                            ▼
                    Extractor Pipeline
                            │
                            ▼
               Corpus Suitability Layer
             (CorpusSuitabilityRecord)
           ┌────────────────┼────────────────┐
           │                │                │
           ▼                ▼                ▼
     Script Writer    Video Editing     Audio/Visual
      Suitability      Suitability      Suitability
    (Obj-Specific)          │                │
           │          Editing Model    Audio/Visual
           │            Datasets         Datasets
           ▼
ScriptIntelligenceRecords
           │
Semantic Reconstruction
(Topic / Briefs / Intent)
           │
Training Data Compiler
(Full Script, Hook, CTA...)
```

---

## Objective-Specific Script Suitability

Rather than a coarse global boolean, the `corpus_suitability` layer evaluates source videos across seven specific script-learning objectives into three tiers (`ELIGIBLE`, `MARGINAL`, `INELIGIBLE`):

1. **`FULL_SCRIPT`**: Substantial speech ($\ge 80$ words, $\ge 3$ sentences, multi-beat structure or verified hook).
2. **`HOOK`**: Clear opening hook mechanism ($\ge 0.5$ confidence, $\ge 10$ words; short 15-word reels can qualify).
3. **`CONTINUATION`**: Substantial narrative ($\ge 60$ words, $\ge 3$ sentences) allowing prefix/continuation pairs.
4. **`STRUCTURE`**: Multi-beat structure ($\ge 3$ beats or $\ge 40$ words with distinct structural phases).
5. **`SECTION`**: At least one focused topical section ($\ge 20$ words).
6. **`CTA`**: Evidenced call-to-action in spoken transcript.
7. **`MEASURABLE_STYLE`**: Sufficient linguistic density ($\ge 50$ words) for stylistic/tone conditioning.

---

## Silent vs. Corrupt Distinction

- **Corrupt Report**: Broken JSON, missing schema envelopes, unreadable byte encodings. (Quarantined by ingestion).
- **Valid Silent Report**: Valid extractor document with duration, visual features, audio tracks, but no spoken dialogue. (Marked `source_valid=True`, routed to `editing`, `audio`, `visual_color`, but flagged `INELIGIBLE` for `script` with reason `no_spoken_transcript`).

---

## Feedback Loop to Upstream RAG Searcher

The `CorpusFeedbackReport` contract exports machine-readable gap analysis to guide future acquisition waves:
- Tracks transcript rate, word-count percentiles (P10, P25, P50, P75, P90), hook count, structure count, and CTA counts.
- Recommends adaptive acquisition adjustments (e.g. targeting educational/commentary creators, increasing query depth for scripted Shorts) without contaminating retrieval with post-hoc virality or view metrics.
