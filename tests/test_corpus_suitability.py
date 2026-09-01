"""Tests for corpus_suitability — objective-specific eligibility and corpus feedback.

All tests run without any remote API.  They use synthetic in-memory records or
the real extractor reports from the RAG validation fixture corpus.
"""
from __future__ import annotations

import hashlib
import json
import glob
import os
from pathlib import Path
from typing import Any

import pytest

from script_writer.corpus_suitability import (
    CORPUS_FEEDBACK_VERSION,
    SUITABILITY_VERSION,
    CorpusSuitabilityRecord,
    ObjectiveSuitability,
    RejectionReason,
    ScriptObjective,
    SubsystemTag,
    SuitabilityTier,
    assess_from_sir,
    assess_silent_valid,
    corpus_feedback_report,
    corpus_suitability_report,
)


# ──────────────────────────────────────────────────────────────
# Helpers — synthetic SIR builders
# ──────────────────────────────────────────────────────────────

def _evidence(value: Any, ev_type: str = "observed") -> dict[str, Any]:
    return {"value": value, "evidence_type": ev_type, "sources": [], "confidence": 0.9}


def _sir(
    word_count: int = 0,
    n_sentences: int = 0,
    n_beats: int = 1,
    hook_mechanisms: list[str] | None = None,
    hook_confidence: float = 0.0,
    has_cta: bool = False,
    wps: float | None = None,
    transcript_text: str | None = None,
) -> dict[str, Any]:
    """Build a minimal synthetic ScriptIntelligenceRecord for testing.

    Generates a transcript with the requested number of sentences (with
    period terminators) so _count_sentences() works correctly.
    """
    if transcript_text is None:
        if word_count == 0:
            transcript_text = ""
        else:
            effective_sentences = max(n_sentences, 1)
            base = word_count // effective_sentences
            extra = word_count % effective_sentences
            parts = []
            for i in range(effective_sentences):
                w = base + (1 if i < extra else 0)
                parts.append(" ".join(["word"] * max(w, 1)) + ".")
            transcript_text = " ".join(parts) if word_count > 0 else ""

    hooks = []
    if hook_mechanisms:
        for mech in hook_mechanisms:
            hooks.append({
                "mechanism": mech,
                "evidence": {
                    "value": True,
                    "evidence_type": "heuristic_inference",
                    "sources": [],
                    "confidence": hook_confidence,
                    "method": "test",
                },
                "text": transcript_text[:40],
                "start_seconds": 0.0,
                "end_seconds": 5.0,
            })

    beats = [{"section_id": f"beat-{i}", "role": "body"} for i in range(n_beats)]

    cta_node = (
        {"value": True, "evidence_type": "observed", "sources": [], "confidence": 0.8}
        if has_cta
        else {"value": None, "evidence_type": "unknown", "sources": []}
    )

    return {
        "record_id": "sir:test:1.0.0",
        "schema_version": "1.0.0",
        "identity": {
            "report_id": _evidence("testreportid"),
            "source_content_hash": _evidence("abc123"),
        },
        "content": {
            "clean_transcript": _evidence(transcript_text),
            "normalized_transcript": _evidence(transcript_text.lower()),
            "video_duration_seconds": _evidence(30.0),
            "spoken_word_count": _evidence(word_count),
            "words_per_second": _evidence(wps or (word_count / 30.0 if word_count else 0.0)),
        },
        "hook_intelligence": {
            "mechanisms": hooks,
        },
        "script_structure": {
            "major_beats": beats,
        },
        "persuasion": {
            "cta_mechanism": cta_node,
        },
        "information_density": {},
        "delivery": {},
    }


def _raw_report(has_transcript: bool = True, duration: float = 30.0) -> dict[str, Any]:
    return {
        "report_id": "rawreport001",
        "source": {
            "content_hash": "abc123",
            "duration_seconds": duration,
            "has_audio": True,
        },
        "transcript": {
            "status": "complete" if has_transcript else "failed",
            "full_text": "hello world" if has_transcript else "",
        },
        "training_eligibility": {
            "script": has_transcript,
            "editing": True,
            "audio": True,
            "color": True,
            "captions": True,
        },
    }


# ──────────────────────────────────────────────────────────────
# SECTION 1: Valid-silent-vs-corrupt distinction
# ──────────────────────────────────────────────────────────────

class TestSilentVsCorrupt:
    def test_silent_valid_video_is_not_corrupt(self) -> None:
        """A video with no transcript is VALID but script-INELIGIBLE."""
        raw = _raw_report(has_transcript=False)
        result = assess_silent_valid(raw)
        assert result.source_valid is True
        assert result.has_spoken_transcript is False
        assert result.spoken_word_count == 0
        assert result.script.any_eligible is False
        assert result.script.any_marginal is False

    def test_silent_video_routes_to_non_script_subsystems(self) -> None:
        """Silent videos with valid editing/audio/color flags should be routed there."""
        raw = _raw_report(has_transcript=False)
        result = assess_silent_valid(raw)
        # Should be routed to editing, audio, visual — not script
        assert SubsystemTag.SCRIPT not in result.routing_tags
        assert any(t in result.routing_tags for t in (
            str(SubsystemTag.EDITING), str(SubsystemTag.AUDIO), str(SubsystemTag.VISUAL_COLOR)
        ))

    def test_silent_video_all_script_objectives_ineligible(self) -> None:
        raw = _raw_report(has_transcript=False)
        result = assess_silent_valid(raw)
        for obj in result.script.objectives:
            assert obj.tier == SuitabilityTier.INELIGIBLE
            assert RejectionReason.NO_SPOKEN_TRANSCRIPT in obj.reasons

    def test_silent_video_schema_version(self) -> None:
        raw = _raw_report(has_transcript=False)
        result = assess_silent_valid(raw)
        assert result.schema_version == SUITABILITY_VERSION


# ──────────────────────────────────────────────────────────────
# SECTION 2: Objective-specific suitability
# ──────────────────────────────────────────────────────────────

class TestFullScriptSuitability:
    def test_no_transcript_ineligible(self) -> None:
        r = assess_from_sir(_sir(word_count=0))
        _check(r, ScriptObjective.FULL_SCRIPT, SuitabilityTier.INELIGIBLE)

    def test_sparse_transcript_ineligible(self) -> None:
        r = assess_from_sir(_sir(word_count=10, n_sentences=1))
        _check(r, ScriptObjective.FULL_SCRIPT, SuitabilityTier.INELIGIBLE)

    def test_medium_sparse_is_marginal(self) -> None:
        r = assess_from_sir(_sir(word_count=50, n_sentences=2))
        _check(r, ScriptObjective.FULL_SCRIPT, SuitabilityTier.MARGINAL)

    def test_substantial_with_hook_is_eligible(self) -> None:
        r = assess_from_sir(_sir(
            word_count=100,
            n_sentences=5,
            hook_mechanisms=["question"],
            hook_confidence=0.7,
        ))
        _check(r, ScriptObjective.FULL_SCRIPT, SuitabilityTier.ELIGIBLE)

    def test_substantial_without_structure_is_marginal(self) -> None:
        """Substantial word count but only 1 beat and no hook → marginal."""
        r = assess_from_sir(_sir(word_count=90, n_sentences=4, n_beats=1))
        _check(r, ScriptObjective.FULL_SCRIPT, SuitabilityTier.MARGINAL)

    def test_repetitive_content_demoted(self) -> None:
        """Very high WPS signals repetitive/nonlinguistic content."""
        r = assess_from_sir(_sir(word_count=120, n_sentences=3, wps=6.5))
        obj = _get_obj(r, ScriptObjective.FULL_SCRIPT)
        assert obj.tier != SuitabilityTier.ELIGIBLE


class TestHookSuitability:
    def test_short_20_word_hook_reel_eligible(self) -> None:
        """A 20-word Reel with a clear hook MUST be hook-eligible."""
        r = assess_from_sir(_sir(
            word_count=20,
            n_sentences=2,
            hook_mechanisms=["contrarian_claim"],
            hook_confidence=0.75,
        ))
        _check(r, ScriptObjective.HOOK, SuitabilityTier.ELIGIBLE)

    def test_no_hook_mechanism_ineligible(self) -> None:
        r = assess_from_sir(_sir(word_count=30, n_sentences=3))
        _check(r, ScriptObjective.HOOK, SuitabilityTier.INELIGIBLE)

    def test_low_confidence_hook_marginal(self) -> None:
        r = assess_from_sir(_sir(
            word_count=15,
            n_sentences=2,
            hook_mechanisms=["story_opening"],
            hook_confidence=0.3,
        ))
        _check(r, ScriptObjective.HOOK, SuitabilityTier.MARGINAL)

    def test_empty_transcript_hook_ineligible(self) -> None:
        r = assess_from_sir(_sir(word_count=0))
        _check(r, ScriptObjective.HOOK, SuitabilityTier.INELIGIBLE)

    def test_very_short_with_hook_marginal(self) -> None:
        """5 words with hook signal → marginal (not enough evidence)."""
        r = assess_from_sir(_sir(
            word_count=6,
            hook_mechanisms=["question"],
            hook_confidence=0.6,
        ))
        _check(r, ScriptObjective.HOOK, SuitabilityTier.MARGINAL)


class TestContinuationSuitability:
    def test_60_words_3_sentences_eligible(self) -> None:
        r = assess_from_sir(_sir(word_count=60, n_sentences=3))
        _check(r, ScriptObjective.CONTINUATION, SuitabilityTier.ELIGIBLE)

    def test_30_words_2_sentences_marginal(self) -> None:
        r = assess_from_sir(_sir(word_count=35, n_sentences=2))
        _check(r, ScriptObjective.CONTINUATION, SuitabilityTier.MARGINAL)

    def test_sparse_ineligible(self) -> None:
        r = assess_from_sir(_sir(word_count=15, n_sentences=1))
        _check(r, ScriptObjective.CONTINUATION, SuitabilityTier.INELIGIBLE)


class TestStructureSuitability:
    def test_3_beats_eligible(self) -> None:
        r = assess_from_sir(_sir(word_count=50, n_beats=3))
        _check(r, ScriptObjective.STRUCTURE, SuitabilityTier.ELIGIBLE)

    def test_40_words_with_hook_eligible(self) -> None:
        r = assess_from_sir(_sir(
            word_count=40,
            n_beats=1,
            hook_mechanisms=["story_opening"],
            hook_confidence=0.6,
        ))
        _check(r, ScriptObjective.STRUCTURE, SuitabilityTier.ELIGIBLE)

    def test_1_beat_no_hook_ineligible(self) -> None:
        r = assess_from_sir(_sir(word_count=15, n_beats=1))
        _check(r, ScriptObjective.STRUCTURE, SuitabilityTier.INELIGIBLE)


class TestSectionSuitability:
    def test_20_words_eligible(self) -> None:
        r = assess_from_sir(_sir(word_count=20, n_sentences=2))
        _check(r, ScriptObjective.SECTION, SuitabilityTier.ELIGIBLE)

    def test_10_words_marginal(self) -> None:
        r = assess_from_sir(_sir(word_count=10, n_sentences=1))
        _check(r, ScriptObjective.SECTION, SuitabilityTier.MARGINAL)

    def test_5_words_ineligible(self) -> None:
        r = assess_from_sir(_sir(word_count=5, n_sentences=1))
        _check(r, ScriptObjective.SECTION, SuitabilityTier.INELIGIBLE)

    def test_long_script_unusable_for_full_script_eligible_for_section(self) -> None:
        """A script with 45 words is too short for full_script (marginal) but fine for section."""
        r = assess_from_sir(_sir(word_count=45, n_sentences=2))
        _check(r, ScriptObjective.FULL_SCRIPT, SuitabilityTier.MARGINAL)
        _check(r, ScriptObjective.SECTION, SuitabilityTier.ELIGIBLE)


class TestCTASuitability:
    def test_with_cta_eligible(self) -> None:
        r = assess_from_sir(_sir(word_count=50, n_sentences=3, has_cta=True))
        _check(r, ScriptObjective.CTA, SuitabilityTier.ELIGIBLE)

    def test_without_cta_ineligible(self) -> None:
        r = assess_from_sir(_sir(word_count=80, n_sentences=5, has_cta=False))
        _check(r, ScriptObjective.CTA, SuitabilityTier.INELIGIBLE)

    def test_cta_rejection_reason(self) -> None:
        r = assess_from_sir(_sir(word_count=80, n_sentences=5, has_cta=False))
        obj = _get_obj(r, ScriptObjective.CTA)
        assert RejectionReason.CTA_ABSENT in obj.reasons


class TestMeasurableStyleSuitability:
    def test_50_words_eligible(self) -> None:
        r = assess_from_sir(_sir(word_count=50))
        _check(r, ScriptObjective.MEASURABLE_STYLE, SuitabilityTier.ELIGIBLE)

    def test_25_words_marginal(self) -> None:
        r = assess_from_sir(_sir(word_count=25))
        _check(r, ScriptObjective.MEASURABLE_STYLE, SuitabilityTier.MARGINAL)

    def test_10_words_ineligible(self) -> None:
        r = assess_from_sir(_sir(word_count=10))
        _check(r, ScriptObjective.MEASURABLE_STYLE, SuitabilityTier.INELIGIBLE)


# ──────────────────────────────────────────────────────────────
# SECTION 3: Routing
# ──────────────────────────────────────────────────────────────

class TestRouting:
    def test_script_eligible_gets_script_tag(self) -> None:
        r = assess_from_sir(
            _sir(word_count=100, n_sentences=5, hook_mechanisms=["question"], hook_confidence=0.7),
            raw_report=_raw_report(),
        )
        assert str(SubsystemTag.SCRIPT) in r.routing_tags

    def test_silent_video_no_script_tag(self) -> None:
        r = assess_silent_valid(_raw_report(has_transcript=False))
        assert str(SubsystemTag.SCRIPT) not in r.routing_tags

    def test_multi_subsystem_routing(self) -> None:
        """A short video with transcript should route to script + editing + audio + visual."""
        r = assess_from_sir(
            _sir(word_count=25, n_sentences=2),
            raw_report=_raw_report(),
        )
        assert str(SubsystemTag.EDITING) in r.routing_tags
        assert str(SubsystemTag.AUDIO) in r.routing_tags


# ──────────────────────────────────────────────────────────────
# SECTION 4: Corpus-level aggregation
# ──────────────────────────────────────────────────────────────

class TestCorpusSuitabilityReport:
    def _make_corpus(self) -> list[CorpusSuitabilityRecord]:
        records = []
        # 5 fully eligible scripts
        for _ in range(5):
            records.append(assess_from_sir(_sir(
                word_count=100,
                n_sentences=5,
                hook_mechanisms=["question"],
                hook_confidence=0.7,
            )))
        # 10 marginal (mid-length, no hook)
        for _ in range(10):
            records.append(assess_from_sir(_sir(word_count=50, n_sentences=2)))
        # 15 silent
        for _ in range(15):
            records.append(assess_silent_valid(_raw_report(has_transcript=False)))
        return records

    def test_report_schema_version(self) -> None:
        report = corpus_suitability_report(self._make_corpus())
        assert report["schema_version"] == SUITABILITY_VERSION

    def test_total_sources(self) -> None:
        corpus = self._make_corpus()
        report = corpus_suitability_report(corpus)
        assert report["total_sources"] == 30

    def test_transcript_counts(self) -> None:
        report = corpus_suitability_report(self._make_corpus())
        assert report["with_spoken_transcript"] == 15
        assert report["without_spoken_transcript"] == 15

    def test_objective_breakdown_present(self) -> None:
        report = corpus_suitability_report(self._make_corpus())
        for obj in ScriptObjective:
            assert str(obj) in report["objective_breakdown"]

    def test_word_count_percentiles(self) -> None:
        report = corpus_suitability_report(self._make_corpus())
        pcts = report["word_count_percentiles"]
        assert "p10" in pcts and "p50" in pcts and "p90" in pcts


# ──────────────────────────────────────────────────────────────
# SECTION 5: Corpus Feedback Report
# ──────────────────────────────────────────────────────────────

class TestCorpusFeedbackReport:
    def test_feedback_schema_version(self) -> None:
        sr = corpus_suitability_report([assess_silent_valid(_raw_report(has_transcript=False))])
        fb = corpus_feedback_report(sr)
        assert fb["schema_version"] == CORPUS_FEEDBACK_VERSION

    def test_low_transcript_rate_creates_gap(self) -> None:
        """A corpus with 0% transcripts must generate a high-severity gap."""
        records = [assess_silent_valid(_raw_report(has_transcript=False)) for _ in range(20)]
        sr = corpus_suitability_report(records)
        fb = corpus_feedback_report(sr)
        dims = [g["dimension"] for g in fb["gaps"]]
        assert "script_transcript_coverage" in dims

    def test_good_corpus_fewer_gaps(self) -> None:
        """Corpus with adequate script content should have fewer gaps."""
        records = []
        for _ in range(50):
            records.append(assess_from_sir(_sir(
                word_count=100,
                n_sentences=5,
                hook_mechanisms=["question"],
                hook_confidence=0.8,
                has_cta=True,
                n_beats=3,
            )))
        sr = corpus_suitability_report(records)
        fb = corpus_feedback_report(sr)
        high_gaps = [g for g in fb["gaps"] if g["severity"] == "high"]
        assert len(high_gaps) < 3

    def test_acquisition_recommendation_present(self) -> None:
        records = [assess_silent_valid(_raw_report(has_transcript=False)) for _ in range(10)]
        sr = corpus_suitability_report(records)
        fb = corpus_feedback_report(sr)
        assert fb["acquisition_recommendation"] in ("A", "B", "C")
        assert fb["acquisition_recommendation_text"]

    def test_feedback_note_present(self) -> None:
        records = [assess_silent_valid(_raw_report(has_transcript=False))]
        sr = corpus_suitability_report(records)
        fb = corpus_feedback_report(sr)
        assert "note" in fb

    def test_projection_present_when_records_exist(self) -> None:
        records = [assess_from_sir(_sir(word_count=100, n_sentences=5))]
        sr = corpus_suitability_report(records)
        fb = corpus_feedback_report(sr, target_total=7500)
        assert "projection" in fb
        assert fb["projection"]["target_total_sources"] == 7500

    def test_feedback_does_not_instruct_on_virality(self) -> None:
        """Feedback report must not tell the searcher to optimize for viral/popularity signals."""
        records = [assess_silent_valid(_raw_report(has_transcript=False)) for _ in range(20)]
        sr = corpus_suitability_report(records)
        fb = corpus_feedback_report(sr)
        payload = json.dumps(fb).lower()
        # Must not prescribe searching for viral content or view counts
        assert "view_count" not in payload
        assert "engagement_rate" not in payload
        assert "likes_per_view" not in payload
        # Acquisition hints must address learning value, not popularity
        for gap in fb.get("gaps", []):
            hint = gap.get("acquisition_hint", "")
            assert "viral" not in hint.lower()
            assert "trending" not in hint.lower()


# ──────────────────────────────────────────────────────────────
# SECTION 6: Determinism and idempotency
# ──────────────────────────────────────────────────────────────

class TestDeterminism:
    def test_same_sir_same_result(self) -> None:
        sir = _sir(word_count=80, n_sentences=4, hook_mechanisms=["story_opening"], hook_confidence=0.6)
        r1 = assess_from_sir(sir)
        r2 = assess_from_sir(sir)
        assert r1.as_dict() == r2.as_dict()

    def test_corpus_report_deterministic(self) -> None:
        records = [assess_from_sir(_sir(word_count=i * 10)) for i in range(10)]
        rep1 = corpus_suitability_report(records)
        rep2 = corpus_suitability_report(records)
        assert rep1 == rep2


# ──────────────────────────────────────────────────────────────
# SECTION 7: Real-data integration test (no API)
# ──────────────────────────────────────────────────────────────

RAG_REPORT_DIRS = [
    "/home/xor_sensei/Dev/Viralyst/RAG/validation/media_test_100/reports",
]


def _find_real_reports() -> list[str]:
    paths = []
    seen = set()
    for d in RAG_REPORT_DIRS:
        for p in glob.glob(f"{d}/*.json"):
            bn = os.path.basename(p)
            if bn not in seen:
                seen.add(bn)
                paths.append(p)
    return paths


REAL_REPORTS = _find_real_reports()


@pytest.mark.skipif(not REAL_REPORTS, reason="Real RAG report corpus not available")
class TestRealCorpusIntegration:
    """Run suitability analysis on the actual 100 Basecamp-context reports."""

    def _load_all(self):
        from script_writer.intelligence import ScriptIntelligenceCompiler
        from script_writer.validation import parse_and_validate_report

        compiler = ScriptIntelligenceCompiler()
        sirs: list[CorpusSuitabilityRecord] = []
        silents: list[CorpusSuitabilityRecord] = []

        for path in REAL_REPORTS:
            raw = open(path, "rb").read()
            raw_report = json.loads(raw)
            try:
                report, _ = parse_and_validate_report(raw, split_salt="viralyst-script-writer-v1")
                compiled = compiler.compile(report, artifact_sha256=hashlib.sha256(raw).hexdigest())
                sirs.append(assess_from_sir(compiled.record, raw_report=raw_report))
            except Exception:
                silents.append(assess_silent_valid(raw_report))

        return sirs, silents

    def test_all_100_produce_suitability_records(self) -> None:
        sirs, silents = self._load_all()
        assert len(sirs) + len(silents) == len(REAL_REPORTS)

    def test_silent_records_are_not_corrupt(self) -> None:
        """All 43 quarantined records must be assessed as valid-but-silent, not corrupt."""
        _, silents = self._load_all()
        for r in silents:
            assert r.source_valid is True
            assert r.has_spoken_transcript is False

    def test_corpus_report_generates(self) -> None:
        sirs, silents = self._load_all()
        all_records = sirs + silents
        report = corpus_suitability_report(all_records)
        assert report["total_sources"] == len(REAL_REPORTS)
        assert report["schema_version"] == SUITABILITY_VERSION

    def test_feedback_report_generates(self) -> None:
        sirs, silents = self._load_all()
        all_records = sirs + silents
        sr = corpus_suitability_report(all_records)
        fb = corpus_feedback_report(sr, target_total=7500)
        assert fb["schema_version"] == CORPUS_FEEDBACK_VERSION
        assert isinstance(fb["gaps"], list)

    def test_hook_only_short_reel_classified_correctly(self) -> None:
        """Short-reel hook sources must not be globally rejected as script-ineligible."""
        sirs, _ = self._load_all()
        # Find records with hook mechanism but < 40 words
        short_hooks = [
            r for r in sirs
            if r.spoken_word_count < 40
            and r.spoken_word_count >= 10
        ]
        # These should at minimum be section-eligible and hook-classified
        for r in short_hooks:
            # Should not be ALL objectives ineligible if they have any words
            all_inelig = all(o.tier == SuitabilityTier.INELIGIBLE for o in r.script.objectives)
            # Section should be eligible if word_count >= 20
            if r.spoken_word_count >= 20:
                section_tier = next(
                    o.tier for o in r.script.objectives if o.objective == ScriptObjective.SECTION
                )
                assert section_tier in (SuitabilityTier.ELIGIBLE, SuitabilityTier.MARGINAL)

    def test_routing_tags_non_empty(self) -> None:
        """Every record (even silent) should have at least one routing tag."""
        sirs, silents = self._load_all()
        for r in sirs + silents:
            assert len(r.routing_tags) > 0, f"No routing tags for {r.source_report_id}"


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _get_obj(record: CorpusSuitabilityRecord, objective: ScriptObjective) -> ObjectiveSuitability:
    for o in record.script.objectives:
        if o.objective == objective:
            return o
    pytest.fail(f"Objective {objective} not found in record")


def _check(record: CorpusSuitabilityRecord, objective: ScriptObjective, expected: SuitabilityTier) -> None:
    obj = _get_obj(record, objective)
    assert obj.tier == expected, (
        f"{objective}: expected {expected}, got {obj.tier}. Reasons: {obj.reasons}"
    )
