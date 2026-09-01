"""Script Training Corpus Suitability Layer.

Distinguishes CLIENT RELEVANCE from SCRIPT-LEARNING VALUE.
A video can be client-relevant while having negligible script-training value —
and vice versa. This module determines subsystem-specific eligibility.

Architecture:

    RAG-retrieved videos
    ↓
    Extractor / ScriptIntelligenceRecords
    ↓
    CorpusSuitabilityRecord  ← this module
    ├── SCRIPT_SUITABILITY     (objective-specific)
    ├── EDITING_SUITABILITY
    ├── AUDIO_SUITABILITY
    └── VISUAL_COLOR_SUITABILITY
    ↓
    Script Writer Training Data Compiler

Key principle: NOT EVERY CLIENT-RELEVANT VIDEO SHOULD TRAIN EVERY MODEL.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .training_contracts import evidence_value


SUITABILITY_VERSION = "1.0.0"


# ──────────────────────────────────────────────────────────────
# Enumerations
# ──────────────────────────────────────────────────────────────

class SuitabilityTier(StrEnum):
    ELIGIBLE = "eligible"
    MARGINAL = "marginal"
    INELIGIBLE = "ineligible"


class ScriptObjective(StrEnum):
    FULL_SCRIPT = "full_script"
    HOOK = "hook"
    CONTINUATION = "continuation"
    STRUCTURE = "structure"
    SECTION = "section"
    CTA = "cta"
    MEASURABLE_STYLE = "measurable_style"


class SubsystemTag(StrEnum):
    SCRIPT = "script"
    EDITING = "editing"
    AUDIO = "audio"
    VISUAL_COLOR = "visual_color"


# ──────────────────────────────────────────────────────────────
# Rejection reason catalogue
# ──────────────────────────────────────────────────────────────

class RejectionReason(StrEnum):
    NO_SPOKEN_TRANSCRIPT = "no_spoken_transcript"
    TRANSCRIPT_TOO_SPARSE_FOR_FULL_SCRIPT = "transcript_too_sparse_for_full_script"
    COHERENT_SHORT_HOOK_ONLY = "coherent_short_hook_only"
    NO_RELIABLE_HOOK = "no_reliable_hook"
    INSUFFICIENT_STRUCTURE = "insufficient_structure"
    SEMANTIC_BRIEF_NOT_INFERABLE = "semantic_brief_not_inferable"
    TRANSCRIPT_LOW_CONFIDENCE = "transcript_low_confidence"
    REPETITIVE_OR_NONLINGUISTIC = "repetitive_or_nonlinguistic"
    USEFUL_FOR_SECTION_ONLY = "useful_for_section_only"
    CTA_ABSENT = "cta_absent"
    CONTINUATION_OPENING_INSUFFICIENT = "continuation_opening_insufficient"
    INSUFFICIENT_SENTENCE_COUNT = "insufficient_sentence_count"
    STYLE_EVIDENCE_INSUFFICIENT = "style_evidence_insufficient"
    TRANSCRIPT_TOO_SHORT_FOR_CONTINUATION = "transcript_too_short_for_continuation"


# ──────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ObjectiveSuitability:
    """Suitability assessment for one Script Writer objective."""
    objective: ScriptObjective
    tier: SuitabilityTier
    reasons: tuple[str, ...]
    evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "objective": str(self.objective),
            "tier": str(self.tier),
            "reasons": list(self.reasons),
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class ScriptSuitabilitySummary:
    """Aggregated script suitability across all objectives."""
    objectives: tuple[ObjectiveSuitability, ...]
    any_eligible: bool
    any_marginal: bool
    primary_rejection_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        eligible = [o for o in self.objectives if o.tier == SuitabilityTier.ELIGIBLE]
        marginal = [o for o in self.objectives if o.tier == SuitabilityTier.MARGINAL]
        ineligible = [o for o in self.objectives if o.tier == SuitabilityTier.INELIGIBLE]
        return {
            "any_eligible": self.any_eligible,
            "any_marginal": self.any_marginal,
            "eligible_objectives": [str(o.objective) for o in eligible],
            "marginal_objectives": [str(o.objective) for o in marginal],
            "ineligible_objectives": [str(o.objective) for o in ineligible],
            "primary_rejection_reasons": list(self.primary_rejection_reasons),
            "objectives": [o.as_dict() for o in self.objectives],
        }


@dataclass(frozen=True)
class SubsystemSuitability:
    """Suitability for a non-script VIRALYST subsystem."""
    subsystem: SubsystemTag
    tier: SuitabilityTier
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "subsystem": str(self.subsystem),
            "tier": str(self.tier),
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class CorpusSuitabilityRecord:
    """
    Versioned suitability assessment for one source video.

    This record sits between the extractor report and the training compiler.
    It answers per-subsystem, per-objective eligibility rather than a single
    global 'is_good' flag.

    Critically, a 'silent' (no-transcript) video that is a valid extractor
    report is NOT corrupt — it is script-ineligible but may be fully eligible
    for editing/audio/visual subsystems.
    """
    schema_version: str
    source_report_id: str
    source_content_hash: str
    source_valid: bool
    has_spoken_transcript: bool
    spoken_word_count: int
    duration_seconds: float | None
    script: ScriptSuitabilitySummary
    subsystems: tuple[SubsystemSuitability, ...]
    routing_tags: tuple[str, ...]
    raw_evidence: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_report_id": self.source_report_id,
            "source_content_hash": self.source_content_hash,
            "source_valid": self.source_valid,
            "has_spoken_transcript": self.has_spoken_transcript,
            "spoken_word_count": self.spoken_word_count,
            "duration_seconds": self.duration_seconds,
            "script": self.script.as_dict(),
            "subsystems": [s.as_dict() for s in self.subsystems],
            "routing_tags": list(self.routing_tags),
            "raw_evidence": self.raw_evidence,
        }


# ──────────────────────────────────────────────────────────────
# Evidence extraction helpers
# ──────────────────────────────────────────────────────────────

def _ev(record: dict[str, Any], *path: str) -> Any:
    """Navigate evidence envelope chain and unwrap value."""
    cur: Any = record
    for k in path:
        cur = cur.get(k, {}) if isinstance(cur, dict) else {}
    return evidence_value(cur)


def _extract_script_evidence(record: dict[str, Any]) -> dict[str, Any]:
    """Extract all script-relevant signals from a compiled ScriptIntelligenceRecord."""
    transcript = str(_ev(record, "content", "clean_transcript") or "")
    word_count: int = len(transcript.split()) if transcript else 0
    sentence_count_val = _ev(record, "content", "sentence_count")
    sentence_count: int = int(sentence_count_val) if isinstance(sentence_count_val, (int, float)) else _count_sentences(transcript)
    words_per_second = _ev(record, "content", "words_per_second")
    duration = _ev(record, "content", "video_duration_seconds")
    tx_conf = _ev(record, "content", "transcript_confidence")

    hooks = record.get("hook_intelligence", {}).get("mechanisms", [])
    hook_mechanisms: list[str] = [str(h.get("mechanism", "")) for h in hooks if isinstance(h, dict) and h.get("mechanism")]
    hook_confidence = max(
        (h.get("evidence", {}).get("confidence", 0.0) for h in hooks if isinstance(h, dict)),
        default=0.0,
    )

    beats = record.get("script_structure", {}).get("major_beats", [])
    n_beats: int = len(beats)

    cta_node = record.get("persuasion", {}).get("cta_mechanism", {})
    has_cta = isinstance(cta_node, dict) and cta_node.get("evidence_type") not in (None, "unknown")

    info_density_node = record.get("information_density", {})
    unique_concept_density = _ev(info_density_node, "unique_concept_density") if isinstance(info_density_node, dict) else None

    # Repetition / non-linguistic signal: very high WPS suggests looping/noise
    repetition_risk = isinstance(words_per_second, (int, float)) and float(words_per_second) > 5.0

    return {
        "transcript": transcript,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "words_per_second": words_per_second,
        "duration_seconds": duration,
        "transcript_confidence": tx_conf,
        "hook_mechanisms": hook_mechanisms,
        "hook_confidence": hook_confidence,
        "n_beats": n_beats,
        "has_cta": has_cta,
        "unique_concept_density": unique_concept_density,
        "repetition_risk": repetition_risk,
    }


def _count_sentences(text: str) -> int:
    if not text:
        return 0
    import re
    return max(1, len(re.split(r"(?<=[.!?])\s+", text.strip())))


# ──────────────────────────────────────────────────────────────
# Per-objective suitability classifiers
# ──────────────────────────────────────────────────────────────

def _assess_full_script(ev: dict[str, Any]) -> ObjectiveSuitability:
    """
    FULL_SCRIPT requires substantial coherent spoken content.

    Thresholds (evidence-based, not arbitrary):
    - eligible:  >= 80 words, >= 3 sentences, hook mechanism or >= 2 beats
    - marginal:  >= 40 words, >= 2 sentences
    - ineligible: otherwise
    """
    wc = ev["word_count"]
    sents = ev["sentence_count"]
    hooks = ev["hook_mechanisms"]
    n_beats = ev["n_beats"]
    rep = ev["repetition_risk"]

    reasons: list[str] = []

    if wc == 0:
        return ObjectiveSuitability(
            objective=ScriptObjective.FULL_SCRIPT,
            tier=SuitabilityTier.INELIGIBLE,
            reasons=(RejectionReason.NO_SPOKEN_TRANSCRIPT,),
            evidence={"word_count": wc},
        )

    if rep:
        reasons.append(RejectionReason.REPETITIVE_OR_NONLINGUISTIC)

    if wc >= 80 and sents >= 3 and not rep:
        has_structure = bool(hooks) or n_beats >= 2
        tier = SuitabilityTier.ELIGIBLE if has_structure else SuitabilityTier.MARGINAL
        if not has_structure:
            reasons.append(RejectionReason.INSUFFICIENT_STRUCTURE)
        return ObjectiveSuitability(
            objective=ScriptObjective.FULL_SCRIPT,
            tier=tier,
            reasons=tuple(reasons),
            evidence={"word_count": wc, "sentence_count": sents, "hooks": hooks, "beats": n_beats},
        )

    if wc >= 40 and sents >= 2 and not rep:
        reasons.append(RejectionReason.TRANSCRIPT_TOO_SPARSE_FOR_FULL_SCRIPT)
        return ObjectiveSuitability(
            objective=ScriptObjective.FULL_SCRIPT,
            tier=SuitabilityTier.MARGINAL,
            reasons=tuple(reasons),
            evidence={"word_count": wc, "sentence_count": sents},
        )

    if wc < 40:
        reasons.append(RejectionReason.TRANSCRIPT_TOO_SPARSE_FOR_FULL_SCRIPT)
    if sents < 2:
        reasons.append(RejectionReason.INSUFFICIENT_SENTENCE_COUNT)
    return ObjectiveSuitability(
        objective=ScriptObjective.FULL_SCRIPT,
        tier=SuitabilityTier.INELIGIBLE,
        reasons=tuple(reasons),
        evidence={"word_count": wc, "sentence_count": sents},
    )


def _assess_hook(ev: dict[str, Any]) -> ObjectiveSuitability:
    """
    HOOK: A 20-word focused Reel can qualify — short scripts are fine if hook is clear.

    - eligible:   has hook mechanism with confidence >= 0.5, >= 10 words
    - marginal:   has hook mechanism confidence < 0.5, or 5-9 words with any hook signal
    - ineligible: no hook evidence or < 5 words
    """
    wc = ev["word_count"]
    hooks = ev["hook_mechanisms"]
    conf = ev["hook_confidence"]

    if wc == 0:
        return ObjectiveSuitability(
            objective=ScriptObjective.HOOK,
            tier=SuitabilityTier.INELIGIBLE,
            reasons=(RejectionReason.NO_SPOKEN_TRANSCRIPT,),
            evidence={"word_count": wc},
        )

    has_hook = bool(hooks) and any(h and h != "none" for h in hooks)

    if wc >= 10 and has_hook and conf >= 0.5:
        return ObjectiveSuitability(
            objective=ScriptObjective.HOOK,
            tier=SuitabilityTier.ELIGIBLE,
            reasons=(),
            evidence={"word_count": wc, "hook_mechanisms": hooks, "hook_confidence": conf},
        )

    if wc >= 5 and has_hook:
        return ObjectiveSuitability(
            objective=ScriptObjective.HOOK,
            tier=SuitabilityTier.MARGINAL,
            reasons=(RejectionReason.NO_RELIABLE_HOOK if conf < 0.5 else RejectionReason.COHERENT_SHORT_HOOK_ONLY,),
            evidence={"word_count": wc, "hook_mechanisms": hooks, "hook_confidence": conf},
        )

    reasons: list[str] = []
    if not has_hook:
        reasons.append(RejectionReason.NO_RELIABLE_HOOK)
    if wc < 5:
        reasons.append(RejectionReason.TRANSCRIPT_TOO_SPARSE_FOR_FULL_SCRIPT)
    return ObjectiveSuitability(
        objective=ScriptObjective.HOOK,
        tier=SuitabilityTier.INELIGIBLE,
        reasons=tuple(reasons),
        evidence={"word_count": wc, "hook_mechanisms": hooks},
    )


def _assess_continuation(ev: dict[str, Any]) -> ObjectiveSuitability:
    """
    CONTINUATION: Needs a coherent opening AND sufficient remaining content.

    - eligible:   >= 60 words, >= 3 sentences (can construct prefix+suffix pairs)
    - marginal:   >= 30 words, >= 2 sentences
    - ineligible: < 30 words or < 2 sentences
    """
    wc = ev["word_count"]
    sents = ev["sentence_count"]

    if wc == 0:
        return ObjectiveSuitability(
            objective=ScriptObjective.CONTINUATION,
            tier=SuitabilityTier.INELIGIBLE,
            reasons=(RejectionReason.NO_SPOKEN_TRANSCRIPT,),
            evidence={"word_count": wc},
        )

    if wc >= 60 and sents >= 3:
        return ObjectiveSuitability(
            objective=ScriptObjective.CONTINUATION,
            tier=SuitabilityTier.ELIGIBLE,
            reasons=(),
            evidence={"word_count": wc, "sentence_count": sents},
        )

    if wc >= 30 and sents >= 2:
        return ObjectiveSuitability(
            objective=ScriptObjective.CONTINUATION,
            tier=SuitabilityTier.MARGINAL,
            reasons=(RejectionReason.TRANSCRIPT_TOO_SHORT_FOR_CONTINUATION,),
            evidence={"word_count": wc, "sentence_count": sents},
        )

    reasons: list[str] = []
    if wc < 30:
        reasons.append(RejectionReason.CONTINUATION_OPENING_INSUFFICIENT)
    if sents < 2:
        reasons.append(RejectionReason.INSUFFICIENT_SENTENCE_COUNT)
    return ObjectiveSuitability(
        objective=ScriptObjective.CONTINUATION,
        tier=SuitabilityTier.INELIGIBLE,
        reasons=tuple(reasons),
        evidence={"word_count": wc, "sentence_count": sents},
    )


def _assess_structure(ev: dict[str, Any]) -> ObjectiveSuitability:
    """
    STRUCTURE: Needs multiple meaningful beats/sections to teach structural planning.

    - eligible:   >= 3 beats OR >= 40 words with hooks
    - marginal:   >= 2 beats OR >= 20 words with some structure signal
    - ineligible: only 1 beat and < 20 words
    """
    wc = ev["word_count"]
    n_beats = ev["n_beats"]
    hooks = ev["hook_mechanisms"]

    if wc == 0:
        return ObjectiveSuitability(
            objective=ScriptObjective.STRUCTURE,
            tier=SuitabilityTier.INELIGIBLE,
            reasons=(RejectionReason.NO_SPOKEN_TRANSCRIPT,),
            evidence={"word_count": wc},
        )

    if n_beats >= 3 or (wc >= 40 and bool(hooks)):
        return ObjectiveSuitability(
            objective=ScriptObjective.STRUCTURE,
            tier=SuitabilityTier.ELIGIBLE,
            reasons=(),
            evidence={"n_beats": n_beats, "word_count": wc, "hooks": hooks},
        )

    if n_beats >= 2 or (wc >= 20 and bool(hooks)):
        return ObjectiveSuitability(
            objective=ScriptObjective.STRUCTURE,
            tier=SuitabilityTier.MARGINAL,
            reasons=(RejectionReason.INSUFFICIENT_STRUCTURE,),
            evidence={"n_beats": n_beats, "word_count": wc},
        )

    return ObjectiveSuitability(
        objective=ScriptObjective.STRUCTURE,
        tier=SuitabilityTier.INELIGIBLE,
        reasons=(RejectionReason.INSUFFICIENT_STRUCTURE,),
        evidence={"n_beats": n_beats, "word_count": wc},
    )


def _assess_section(ev: dict[str, Any]) -> ObjectiveSuitability:
    """
    SECTION: One strong section qualifies — more lenient than STRUCTURE.

    - eligible:   >= 20 words
    - marginal:   >= 8 words
    - ineligible: < 8 words
    """
    wc = ev["word_count"]

    if wc == 0:
        return ObjectiveSuitability(
            objective=ScriptObjective.SECTION,
            tier=SuitabilityTier.INELIGIBLE,
            reasons=(RejectionReason.NO_SPOKEN_TRANSCRIPT,),
            evidence={"word_count": wc},
        )
    if wc >= 20:
        return ObjectiveSuitability(
            objective=ScriptObjective.SECTION,
            tier=SuitabilityTier.ELIGIBLE,
            reasons=(),
            evidence={"word_count": wc},
        )
    if wc >= 8:
        return ObjectiveSuitability(
            objective=ScriptObjective.SECTION,
            tier=SuitabilityTier.MARGINAL,
            reasons=(RejectionReason.USEFUL_FOR_SECTION_ONLY,),
            evidence={"word_count": wc},
        )
    return ObjectiveSuitability(
        objective=ScriptObjective.SECTION,
        tier=SuitabilityTier.INELIGIBLE,
        reasons=(RejectionReason.TRANSCRIPT_TOO_SPARSE_FOR_FULL_SCRIPT,),
        evidence={"word_count": wc},
    )


def _assess_cta(ev: dict[str, Any]) -> ObjectiveSuitability:
    """
    CTA: Requires actual evidenced CTA in the script.

    - eligible:   has_cta = True
    - marginal:   not classified as marginal currently (CTA is binary)
    - ineligible: no CTA evidence
    """
    if ev["word_count"] == 0:
        return ObjectiveSuitability(
            objective=ScriptObjective.CTA,
            tier=SuitabilityTier.INELIGIBLE,
            reasons=(RejectionReason.NO_SPOKEN_TRANSCRIPT, RejectionReason.CTA_ABSENT),
            evidence={},
        )
    if ev["has_cta"]:
        return ObjectiveSuitability(
            objective=ScriptObjective.CTA,
            tier=SuitabilityTier.ELIGIBLE,
            reasons=(),
            evidence={"has_cta": True},
        )
    return ObjectiveSuitability(
        objective=ScriptObjective.CTA,
        tier=SuitabilityTier.INELIGIBLE,
        reasons=(RejectionReason.CTA_ABSENT,),
        evidence={"has_cta": False},
    )


def _assess_measurable_style(ev: dict[str, Any]) -> ObjectiveSuitability:
    """
    MEASURABLE_STYLE: Enough language to measure style characteristics reliably.

    - eligible:   >= 50 words (enough for linguistic analysis)
    - marginal:   >= 20 words
    - ineligible: < 20 words
    """
    wc = ev["word_count"]

    if wc == 0:
        return ObjectiveSuitability(
            objective=ScriptObjective.MEASURABLE_STYLE,
            tier=SuitabilityTier.INELIGIBLE,
            reasons=(RejectionReason.NO_SPOKEN_TRANSCRIPT,),
            evidence={"word_count": wc},
        )
    if wc >= 50:
        return ObjectiveSuitability(
            objective=ScriptObjective.MEASURABLE_STYLE,
            tier=SuitabilityTier.ELIGIBLE,
            reasons=(),
            evidence={"word_count": wc},
        )
    if wc >= 20:
        return ObjectiveSuitability(
            objective=ScriptObjective.MEASURABLE_STYLE,
            tier=SuitabilityTier.MARGINAL,
            reasons=(RejectionReason.STYLE_EVIDENCE_INSUFFICIENT,),
            evidence={"word_count": wc},
        )
    return ObjectiveSuitability(
        objective=ScriptObjective.MEASURABLE_STYLE,
        tier=SuitabilityTier.INELIGIBLE,
        reasons=(RejectionReason.STYLE_EVIDENCE_INSUFFICIENT,),
        evidence={"word_count": wc},
    )


# ──────────────────────────────────────────────────────────────
# Non-script subsystem assessors
# ──────────────────────────────────────────────────────────────

def _assess_editing(raw_report: dict[str, Any]) -> SubsystemSuitability:
    """Editing suitability: any structurally valid video with adequate duration."""
    dur = raw_report.get("source", {}).get("duration_seconds", 0)
    elig = raw_report.get("training_eligibility", {})
    edit_flag = elig.get("editing", False)
    tier = SuitabilityTier.ELIGIBLE if edit_flag and dur >= 3 else (
        SuitabilityTier.MARGINAL if dur >= 3 else SuitabilityTier.INELIGIBLE
    )
    reasons = () if tier == SuitabilityTier.ELIGIBLE else ("duration_too_short",)
    return SubsystemSuitability(SubsystemTag.EDITING, tier, reasons)


def _assess_audio(raw_report: dict[str, Any]) -> SubsystemSuitability:
    """Audio suitability: any video with audio track present."""
    elig = raw_report.get("training_eligibility", {})
    audio_flag = elig.get("audio", False)
    dur = raw_report.get("source", {}).get("duration_seconds", 0)
    has_audio = raw_report.get("source", {}).get("has_audio", True)
    tier = SuitabilityTier.ELIGIBLE if audio_flag and has_audio and dur >= 3 else (
        SuitabilityTier.INELIGIBLE if not has_audio else SuitabilityTier.MARGINAL
    )
    reasons = () if tier == SuitabilityTier.ELIGIBLE else ("no_audio_track" if not has_audio else "marginal_duration",)
    return SubsystemSuitability(SubsystemTag.AUDIO, tier, reasons)


def _assess_visual_color(raw_report: dict[str, Any]) -> SubsystemSuitability:
    """Visual/color suitability: any video with sufficient visual content."""
    elig = raw_report.get("training_eligibility", {})
    color_flag = elig.get("color", False)
    caption_flag = elig.get("captions", False)
    tier = SuitabilityTier.ELIGIBLE if (color_flag or caption_flag) else SuitabilityTier.INELIGIBLE
    reasons = () if tier == SuitabilityTier.ELIGIBLE else ("insufficient_visual_evidence",)
    return SubsystemSuitability(SubsystemTag.VISUAL_COLOR, tier, reasons)


# ──────────────────────────────────────────────────────────────
# Main public API
# ──────────────────────────────────────────────────────────────

def assess_from_sir(
    record: dict[str, Any],
    raw_report: dict[str, Any] | None = None,
) -> CorpusSuitabilityRecord:
    """
    Assess corpus suitability from a compiled ScriptIntelligenceRecord.

    Args:
        record:      Compiled ScriptIntelligenceRecord dict.
        raw_report:  Original extractor JSON (for non-script subsystems).
                     When None, non-script subsystems default to MARGINAL.
    """
    ev = _extract_script_evidence(record)
    report_id = str(_ev(record, "identity", "report_id") or record.get("record_id", "unknown"))
    content_hash = str(_ev(record, "identity", "source_content_hash") or "")
    dur = ev.get("duration_seconds")

    # Script objectives
    obj_results = (
        _assess_full_script(ev),
        _assess_hook(ev),
        _assess_continuation(ev),
        _assess_structure(ev),
        _assess_section(ev),
        _assess_cta(ev),
        _assess_measurable_style(ev),
    )

    any_eligible = any(o.tier == SuitabilityTier.ELIGIBLE for o in obj_results)
    any_marginal = any(o.tier == SuitabilityTier.MARGINAL for o in obj_results)

    all_reasons: list[str] = []
    for o in obj_results:
        if o.tier == SuitabilityTier.INELIGIBLE:
            all_reasons.extend(o.reasons)
    from collections import Counter
    primary_reasons = tuple(r for r, _ in Counter(all_reasons).most_common(3))

    script_summary = ScriptSuitabilitySummary(
        objectives=obj_results,
        any_eligible=any_eligible,
        any_marginal=any_marginal,
        primary_rejection_reasons=primary_reasons,
    )

    # Subsystems
    if raw_report is not None:
        subsystems: tuple[SubsystemSuitability, ...] = (
            _assess_editing(raw_report),
            _assess_audio(raw_report),
            _assess_visual_color(raw_report),
        )
    else:
        subsystems = (
            SubsystemSuitability(SubsystemTag.EDITING, SuitabilityTier.MARGINAL, ("raw_report_unavailable",)),
            SubsystemSuitability(SubsystemTag.AUDIO, SuitabilityTier.MARGINAL, ("raw_report_unavailable",)),
            SubsystemSuitability(SubsystemTag.VISUAL_COLOR, SuitabilityTier.MARGINAL, ("raw_report_unavailable",)),
        )

    # Routing tags
    tags: list[str] = []
    if any_eligible or any_marginal:
        tags.append(SubsystemTag.SCRIPT)
    for sub in subsystems:
        if sub.tier in (SuitabilityTier.ELIGIBLE, SuitabilityTier.MARGINAL):
            tags.append(str(sub.subsystem))

    return CorpusSuitabilityRecord(
        schema_version=SUITABILITY_VERSION,
        source_report_id=report_id,
        source_content_hash=content_hash,
        source_valid=True,
        has_spoken_transcript=ev["word_count"] > 0,
        spoken_word_count=ev["word_count"],
        duration_seconds=float(dur) if isinstance(dur, (int, float)) else None,
        script=script_summary,
        subsystems=subsystems,
        routing_tags=tuple(sorted(set(tags))),
        raw_evidence=ev,
    )


def assess_silent_valid(
    raw_report: dict[str, Any],
) -> CorpusSuitabilityRecord:
    """
    Assess a structurally valid extractor report that has no usable transcript.

    These are NOT corrupt — they are valid videos unsuitable for script training
    but potentially valuable for editing/audio/visual subsystems.
    """
    src = raw_report.get("source", {})
    report_id = raw_report.get("report_id", "unknown")
    content_hash = src.get("content_hash", "")
    dur = src.get("duration_seconds")

    all_ineligible = tuple(
        ObjectiveSuitability(
            objective=script_obj,
            tier=SuitabilityTier.INELIGIBLE,
            reasons=(RejectionReason.NO_SPOKEN_TRANSCRIPT,),
            evidence={},
        )
        for script_obj in ScriptObjective
    )

    script_summary = ScriptSuitabilitySummary(
        objectives=all_ineligible,
        any_eligible=False,
        any_marginal=False,
        primary_rejection_reasons=(RejectionReason.NO_SPOKEN_TRANSCRIPT,),
    )

    subsystems = (
        _assess_editing(raw_report),
        _assess_audio(raw_report),
        _assess_visual_color(raw_report),
    )

    tags: list[str] = [str(sub.subsystem) for sub in subsystems if sub.tier != SuitabilityTier.INELIGIBLE]

    return CorpusSuitabilityRecord(
        schema_version=SUITABILITY_VERSION,
        source_report_id=report_id,
        source_content_hash=content_hash,
        source_valid=True,
        has_spoken_transcript=False,
        spoken_word_count=0,
        duration_seconds=float(dur) if isinstance(dur, (int, float)) else None,
        script=script_summary,
        subsystems=subsystems,
        routing_tags=tuple(sorted(set(tags))),
        raw_evidence={"reason": "no_usable_transcript"},
    )


# ──────────────────────────────────────────────────────────────
# Corpus-level analysis
# ──────────────────────────────────────────────────────────────

def corpus_suitability_report(
    records: list[CorpusSuitabilityRecord],
) -> dict[str, Any]:
    """Aggregate suitability counts across a corpus."""
    total = len(records)
    with_transcript = sum(1 for r in records if r.has_spoken_transcript)
    without_transcript = total - with_transcript

    objective_stats: dict[str, dict[str, int]] = {}
    for obj in ScriptObjective:
        eligible = marginal = ineligible = 0
        for r in records:
            for o in r.script.objectives:
                if o.objective == obj:
                    if o.tier == SuitabilityTier.ELIGIBLE:
                        eligible += 1
                    elif o.tier == SuitabilityTier.MARGINAL:
                        marginal += 1
                    else:
                        ineligible += 1
        objective_stats[str(obj)] = {
            "eligible": eligible,
            "marginal": marginal,
            "ineligible": ineligible,
        }

    # Rejection reason frequency
    from collections import Counter
    reason_counter: Counter[str] = Counter()
    for r in records:
        for o in r.script.objectives:
            reason_counter.update(o.reasons)

    # Routing distribution
    routing_counter: Counter[str] = Counter()
    for r in records:
        for tag in r.routing_tags:
            routing_counter[tag] += 1

    # Word count percentiles
    wcs = sorted(r.spoken_word_count for r in records)

    def _pct(lst: list[int], p: float) -> int:
        if not lst:
            return 0
        idx = int(p / 100 * (len(lst) - 1))
        return lst[idx]

    return {
        "schema_version": SUITABILITY_VERSION,
        "total_sources": total,
        "with_spoken_transcript": with_transcript,
        "without_spoken_transcript": without_transcript,
        "script_transcript_rate": round(with_transcript / total, 4) if total else 0.0,
        "any_script_eligible": sum(1 for r in records if r.script.any_eligible),
        "any_script_marginal_or_eligible": sum(
            1 for r in records if r.script.any_eligible or r.script.any_marginal
        ),
        "objective_breakdown": objective_stats,
        "rejection_reason_distribution": dict(reason_counter.most_common(20)),
        "routing_distribution": dict(routing_counter),
        "word_count_percentiles": {
            "p10": _pct(wcs, 10),
            "p25": _pct(wcs, 25),
            "p50": _pct(wcs, 50),
            "p75": _pct(wcs, 75),
            "p90": _pct(wcs, 90),
            "p95": _pct(wcs, 95),
        },
        "subsystem_eligibility": {
            str(tag): sum(
                1 for r in records
                for s in r.subsystems
                if str(s.subsystem) == tag and s.tier == SuitabilityTier.ELIGIBLE
            )
            for tag in SubsystemTag
            if tag != SubsystemTag.SCRIPT
        },
    }


# ──────────────────────────────────────────────────────────────
# Corpus Feedback Report (for RAG Searcher consumption)
# ──────────────────────────────────────────────────────────────

CORPUS_FEEDBACK_VERSION = "1.0.0"

# Recommended thresholds for a healthy script corpus
SCRIPT_CORPUS_TARGETS = {
    "min_script_transcript_rate": 0.40,       # >= 40% should have spoken transcript
    "min_full_script_eligible_rate": 0.20,    # >= 20% eligible for full_script
    "min_hook_eligible_count": 30,            # absolute: at least 30 hook examples per 100
    "min_cta_eligible_count": 10,             # absolute: at least 10 CTA examples per 100
    "min_structure_eligible_count": 15,       # at least 15 structure examples per 100
    "max_format_concentration": 0.60,         # no single archetype > 60%
    "max_topic_concentration": 0.50,          # no single topic > 50% of script-bearing
    "max_talking_head_share": 0.80,           # not all talking-head
    "min_word_count_p50": 30,                 # median transcript >= 30 words
    "min_word_count_p75": 80,                 # 75th percentile >= 80 words
}


def corpus_feedback_report(
    suitability_report: dict[str, Any],
    *,
    target_total: int = 7500,
    archetype_distribution: dict[str, int] | None = None,
) -> dict[str, Any]:
    """
    Generate a machine-readable CorpusFeedbackReport for the RAG Searcher.

    This report identifies gaps that should drive the next acquisition wave.
    It does NOT prescribe exact queries — that is the Searcher's job.
    """
    total = suitability_report["total_sources"]
    gaps: list[dict[str, Any]] = []
    satisfied: list[str] = []

    def _rate(key: str) -> float:
        return suitability_report.get(key, 0) / total if total else 0.0

    def _obj_rate(obj: str, tier: str) -> float:
        return suitability_report["objective_breakdown"].get(obj, {}).get(tier, 0) / total if total else 0.0

    # 1. Transcript rate
    tx_rate = _rate("with_spoken_transcript")
    if tx_rate < SCRIPT_CORPUS_TARGETS["min_script_transcript_rate"]:
        gaps.append({
            "dimension": "script_transcript_coverage",
            "severity": "high",
            "measured": round(tx_rate, 4),
            "target": SCRIPT_CORPUS_TARGETS["min_script_transcript_rate"],
            "message": (
                f"Only {tx_rate:.0%} of sources have a spoken transcript. "
                "Acquisition should prioritize content from creators with consistent voiceover."
            ),
            "acquisition_hint": "prefer_voice_over_broll_creators",
        })
    else:
        satisfied.append("script_transcript_coverage")

    # 2. Full script eligible rate
    full_script_rate = _obj_rate("full_script", "eligible")
    if full_script_rate < SCRIPT_CORPUS_TARGETS["min_full_script_eligible_rate"]:
        gaps.append({
            "dimension": "full_script_eligible_rate",
            "severity": "high",
            "measured": round(full_script_rate, 4),
            "target": SCRIPT_CORPUS_TARGETS["min_full_script_eligible_rate"],
            "message": (
                f"Only {full_script_rate:.0%} of sources are eligible for full_script training. "
                "Target creators who produce longer scripted Shorts (60+ seconds, educational/commentary)."
            ),
            "acquisition_hint": "prefer_scripted_educational_commentary",
        })
    else:
        satisfied.append("full_script_eligible_rate")

    # 3. Hook examples
    hook_count = suitability_report["objective_breakdown"].get("hook", {}).get("eligible", 0)
    hook_target = SCRIPT_CORPUS_TARGETS["min_hook_eligible_count"]
    if hook_count < hook_target:
        gaps.append({
            "dimension": "hook_example_count",
            "severity": "medium",
            "measured": hook_count,
            "target": hook_target,
            "message": (
                f"Only {hook_count} hook-eligible sources. "
                "Acquire more content with strong opening hooks (question, contrarian claim, story opening)."
            ),
            "acquisition_hint": "target_strong_hook_formats",
        })
    else:
        satisfied.append("hook_example_count")

    # 4. CTA examples
    cta_count = suitability_report["objective_breakdown"].get("cta", {}).get("eligible", 0)
    cta_target = SCRIPT_CORPUS_TARGETS["min_cta_eligible_count"]
    if cta_count < cta_target:
        gaps.append({
            "dimension": "cta_example_count",
            "severity": "medium",
            "measured": cta_count,
            "target": cta_target,
            "message": (
                f"Only {cta_count} CTA-eligible sources. "
                "Acquire more content where creators explicitly direct audience action."
            ),
            "acquisition_hint": "target_cta_bearing_creators",
        })
    else:
        satisfied.append("cta_example_count")

    # 5. Structure examples
    struct_count = suitability_report["objective_breakdown"].get("structure", {}).get("eligible", 0)
    struct_target = SCRIPT_CORPUS_TARGETS["min_structure_eligible_count"]
    if struct_count < struct_target:
        gaps.append({
            "dimension": "structure_example_count",
            "severity": "medium",
            "measured": struct_count,
            "target": struct_target,
            "message": (
                f"Only {struct_count} structure-eligible sources. "
                "Target creators who use clear multi-beat scripts (problem/solution, list formats, tutorials)."
            ),
            "acquisition_hint": "target_structured_scripted_content",
        })
    else:
        satisfied.append("structure_example_count")

    # 6. Median word count
    wc_p50 = suitability_report.get("word_count_percentiles", {}).get("p50", 0)
    if wc_p50 < SCRIPT_CORPUS_TARGETS["min_word_count_p50"]:
        gaps.append({
            "dimension": "median_transcript_length",
            "severity": "high",
            "measured": wc_p50,
            "target": SCRIPT_CORPUS_TARGETS["min_word_count_p50"],
            "message": (
                f"Median transcript length is only {wc_p50} words. "
                "Corpus is dominated by very short/silent content. "
                "Prioritize Shorts with >= 30 seconds of substantive speech."
            ),
            "acquisition_hint": "prefer_longer_spoken_content",
        })
    else:
        satisfied.append("median_transcript_length")

    # 7. P75 word count
    wc_p75 = suitability_report.get("word_count_percentiles", {}).get("p75", 0)
    if wc_p75 < SCRIPT_CORPUS_TARGETS["min_word_count_p75"]:
        gaps.append({
            "dimension": "p75_transcript_length",
            "severity": "medium",
            "measured": wc_p75,
            "target": SCRIPT_CORPUS_TARGETS["min_word_count_p75"],
            "message": (
                f"Only top 25% of sources have >= {wc_p75} words — target is 80. "
                "Insufficient long-form scripted Shorts for structure/full-script training."
            ),
            "acquisition_hint": "increase_scripted_content_percentage",
        })
    else:
        satisfied.append("p75_transcript_length")

    # Archetype concentration check
    if archetype_distribution:
        arch_total = sum(archetype_distribution.values())
        for arch, cnt in archetype_distribution.items():
            rate = cnt / arch_total if arch_total else 0
            if rate > SCRIPT_CORPUS_TARGETS["max_format_concentration"]:
                gaps.append({
                    "dimension": "archetype_concentration",
                    "severity": "medium",
                    "archetype": arch,
                    "measured": round(rate, 4),
                    "target": SCRIPT_CORPUS_TARGETS["max_format_concentration"],
                    "message": (
                        f"Archetype '{arch}' accounts for {rate:.0%} of sources. "
                        "Diversify across talking_head, educational, storytelling formats."
                    ),
                    "acquisition_hint": "diversify_content_archetypes",
                })

    # Projection to 7,500 sources
    if total > 0:
        script_eligible_rate = _rate("any_script_eligible")
        projected_script_eligible = round(script_eligible_rate * target_total)
        projected_full_script = round(full_script_rate * target_total)
        projected_hook = round(_obj_rate("hook", "eligible") * target_total)
        projected_cta = round(_obj_rate("cta", "eligible") * target_total)
        projection = {
            "target_total_sources": target_total,
            "projected_script_eligible": projected_script_eligible,
            "projected_full_script_eligible": projected_full_script,
            "projected_hook_eligible": projected_hook,
            "projected_cta_eligible": projected_cta,
            "note": "Projection assumes current sample is representative of production corpus.",
        }
    else:
        projection = {}

    # Overall acquisition recommendation
    high_severity = sum(1 for g in gaps if g["severity"] == "high")
    if high_severity >= 2:
        recommendation = "A"
        recommendation_text = (
            "IMPROVE ACQUISITION FIRST. "
            "Current corpus has fundamental gaps in script-bearing content that cannot be resolved "
            "by volume alone. The RAG acquisition strategy should be adjusted before scaling."
        )
    elif gaps:
        recommendation = "C"
        recommendation_text = (
            "BOTH. "
            "Current acquisition direction is partially suitable but has measurable gaps. "
            "Improve acquisition targeting AND increase volume in parallel."
        )
    else:
        recommendation = "B"
        recommendation_text = (
            "SCALE VOLUME. "
            "Current acquisition mix is acceptable. "
            "Increasing volume with the same strategy should yield adequate corpus coverage."
        )

    return {
        "schema_version": CORPUS_FEEDBACK_VERSION,
        "total_sources_analyzed": total,
        "target_total_sources": target_total,
        "gaps": gaps,
        "gap_count": len(gaps),
        "high_severity_gap_count": high_severity,
        "satisfied_dimensions": satisfied,
        "projection": projection,
        "acquisition_recommendation": recommendation,
        "acquisition_recommendation_text": recommendation_text,
        "corpus_targets": SCRIPT_CORPUS_TARGETS,
        "note": (
            "This report is consumed by the RAG acquisition layer. "
            "It does NOT instruct the Searcher to add performance or popularity metrics to queries. "
            "Gaps describe LEARNING VALUE deficits, not popularity signals."
        ),
    }
