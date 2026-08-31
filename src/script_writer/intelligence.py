from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .contracts import (
    SCRIPT_INTELLIGENCE_SCHEMA_VERSION,
    SourceRef,
    derived,
    heuristic,
    observed,
    unknown,
    validate_script_intelligence_record,
)
from .semantic import NullSemanticAnalyzer, RuleBasedSemanticAnalyzer, SemanticAnalyzer, SemanticContext
from .text_analysis import (
    NUMBER_RE,
    clean_transcript,
    linguistic_statistics,
    normalize_transcript,
    sentence_cadence,
    words,
)


COMPILER_VERSION = "1.0.0"


@dataclass(frozen=True)
class CompiledIntelligence:
    record: dict[str, Any]
    canonical_json: str
    sha256: str


def _confidence(value: Any) -> float | None:
    if isinstance(value, (int, float)) and 0 <= value <= 1:
        return float(value)
    return None


def _mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _sentences(report: dict[str, Any], clean_text: str) -> list[dict[str, Any]]:
    raw = _mapping(report.get("transcript")).get("sentences", [])
    result = []
    if isinstance(raw, list):
        for index, sentence in enumerate(raw):
            if not isinstance(sentence, dict) or not str(sentence.get("text", "")).strip():
                continue
            result.append(
                {
                    "sentence_id": sentence.get("sentence_id", index),
                    "text": clean_transcript(str(sentence["text"])),
                    "start": sentence.get("start"),
                    "end": sentence.get("end"),
                    "confidence": sentence.get("confidence"),
                }
            )
    if result:
        return result
    chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", clean_text) if chunk.strip()]
    return [
        {"sentence_id": index, "text": text, "start": None, "end": None, "confidence": None}
        for index, text in enumerate(chunks)
    ]


def _upstream_sections(report: dict[str, Any], sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    semantic_root = _mapping(report.get("semantic"))
    raw_sections = semantic_root.get("sections", [])
    sections = []
    if isinstance(raw_sections, list):
        for index, section in enumerate(raw_sections):
            if not isinstance(section, dict) or not str(section.get("text", "")).strip():
                continue
            start = section.get("start")
            end = section.get("end")
            confidence = _confidence(section.get("confidence")) or 0.5
            sections.append(
                {
                    "section_id": str(section.get("section_id", index)),
                    "role": str(section.get("type", "unknown")),
                    "text": clean_transcript(str(section["text"])),
                    "start_seconds": start,
                    "end_seconds": end,
                    "evidence": heuristic(
                        True,
                        [SourceRef(f"$.semantic.sections[{index}]", start, end)],
                        str(semantic_root.get("method", "upstream_semantic_hypothesis")),
                        confidence,
                    ),
                    "upstream_verification_status": section.get("verification_status"),
                }
            )
    if sections:
        return sections
    if not sentences:
        return []
    first = sentences[0]
    return [
        {
            "section_id": "fallback-0",
            "role": "hook_candidate",
            "text": first["text"],
            "start_seconds": first.get("start"),
            "end_seconds": first.get("end"),
            "evidence": heuristic(
                True,
                [SourceRef("$.transcript.sentences[0]", first.get("start"), first.get("end"))],
                "first_sentence_hook_candidate_fallback",
                0.45,
            ),
            "upstream_verification_status": "not_available",
        }
    ]


def _field_from_sections(
    sections: list[dict[str, Any]], roles: set[str], *, multiple: bool = False
) -> dict[str, Any] | list[dict[str, Any]]:
    matches = [section for section in sections if section["role"] in roles]
    if multiple:
        return matches
    if not matches:
        return unknown(f"no section with role in {sorted(roles)} was available")
    section = matches[0]
    return heuristic(
        {
            "text": section["text"],
            "start_seconds": section["start_seconds"],
            "end_seconds": section["end_seconds"],
            "role": section["role"],
        },
        [SourceRef("$.semantic.sections", section["start_seconds"], section["end_seconds"])],
        "preserved_upstream_structural_hypothesis",
        section["evidence"].get("confidence", 0.5),
    )


def _delivery(report: dict[str, Any], duration: float, word_count: int) -> dict[str, Any]:
    transcript = _mapping(report.get("transcript"))
    delivery = transcript.get("delivery", {}) if isinstance(transcript.get("delivery"), dict) else {}
    observed_wpm = delivery.get("overall_words_per_minute")
    speaking_span = delivery.get("speaking_span_seconds")
    denominator = float(speaking_span) if isinstance(speaking_span, (int, float)) and speaking_span > 0 else duration
    computed_wps = round(word_count / denominator, 4) if denominator > 0 else 0
    pauses = transcript.get("pauses", []) if isinstance(transcript.get("pauses"), list) else []
    emphasis = transcript.get("emphasized_words", []) if isinstance(transcript.get("emphasized_words"), list) else []
    rolling = delivery.get("rolling_windows", []) if isinstance(delivery.get("rolling_windows"), list) else []
    transitions = []
    previous = None
    for window in rolling:
        pace = window.get("words_per_minute") if isinstance(window, dict) else None
        if not isinstance(pace, (int, float)):
            continue
        if previous is not None and abs(float(pace) - previous[1]) >= 60:
            transitions.append(
                {
                    "at_seconds": window.get("start"),
                    "from_wpm": previous[1],
                    "to_wpm": float(pace),
                    "direction": "faster" if pace > previous[1] else "slower",
                    "evidence": derived(
                        True,
                        [SourceRef("$.transcript.delivery.rolling_windows", window.get("start"), window.get("end"))],
                        "adjacent_rolling_window_delta_at_least_60_wpm",
                    ),
                }
            )
        previous = (window.get("start"), float(pace))
    return {
        "pace_words_per_minute": (
            observed(observed_wpm, "$.transcript.delivery.overall_words_per_minute", method="extractor_measured")
            if isinstance(observed_wpm, (int, float))
            else derived(computed_wps * 60, [SourceRef("$.transcript.full_text"), SourceRef("$.source.duration_seconds")], "word_count_divided_by_speaking_duration")
        ),
        "pace_words_per_second": derived(
            computed_wps,
            [SourceRef("$.transcript.full_text"), SourceRef("$.transcript.delivery.speaking_span_seconds")],
            "token_count_divided_by_speaking_span",
        ),
        "pauses": observed(pauses, "$.transcript.pauses", method="extractor_word_timing_gaps") if pauses else unknown("extractor provided no pause events"),
        "emphasis_candidates": (
            heuristic(
                emphasis,
                [SourceRef("$.transcript.emphasized_words")],
                "upstream_pitch_energy_duration_candidates",
                min((_confidence(item.get("confidence")) or 0.5 for item in emphasis), default=0.5),
            )
            if emphasis else unknown("extractor provided no emphasis candidates")
        ),
        "silence_ranges": observed(_mapping(_mapping(report.get("audio")).get("events")).get("silence_ranges", []), "$.audio.events.silence_ranges", method="extractor_audio_measurement"),
        "pace_transitions": transitions,
        "sentence_cadence": derived(
            sentence_cadence(_sentences(report, clean_transcript(str(transcript.get("full_text", ""))))),
            [SourceRef("$.transcript.sentences")],
            "sentence_word_count_divided_by_aligned_duration",
        ),
        "energy_changes": unknown("extractor provides word-level energy but no validated section-level energy-change events"),
        "speech_rhythm": unknown("a stable speech-rhythm taxonomy has not been defined"),
    }


def _script_edit_relationships(report: dict[str, Any], sentences: list[dict[str, Any]]) -> list[dict[str, Any]]:
    relationships: list[dict[str, Any]] = []
    edits = _mapping(report.get("editing")).get("verified_events", [])
    if isinstance(edits, list):
        for index, edit in enumerate(edits):
            if not isinstance(edit, dict) or edit.get("verification_status") != "verified":
                continue
            timestamp = edit.get("timestamp")
            if not isinstance(timestamp, (int, float)):
                continue
            nearest = None
            nearest_distance = float("inf")
            for sentence in sentences:
                for boundary in (sentence.get("start"), sentence.get("end")):
                    if isinstance(boundary, (int, float)) and abs(timestamp - boundary) < nearest_distance:
                        nearest, nearest_distance = sentence, abs(timestamp - boundary)
            if nearest is not None and nearest_distance <= 0.4:
                relationships.append(
                    {
                        "relationship": "verified_edit_near_sentence_boundary",
                        "timestamp_seconds": round(float(timestamp), 4),
                        "script_span": {
                            "start_seconds": nearest.get("start"),
                            "end_seconds": nearest.get("end"),
                            "text": nearest.get("text"),
                        },
                        "edit_type": edit.get("type"),
                        "evidence": derived(
                            {"boundary_distance_seconds": round(nearest_distance, 4)},
                            [SourceRef(f"$.editing.verified_events[{index}]", timestamp, timestamp), SourceRef("$.transcript.sentences", nearest.get("start"), nearest.get("end"))],
                            "nearest_verified_edit_to_sentence_boundary_within_0.4_seconds",
                        ),
                    }
                )
    cross_modal = report.get("cross_modal_events", [])
    if isinstance(cross_modal, list):
        for index, event in enumerate(cross_modal):
            if not isinstance(event, dict) or not event.get("interpretations"):
                continue
            relationships.append(
                {
                    "relationship": "upstream_cross_modal_interpretation",
                    "timestamp_seconds": event.get("start"),
                    "script_span": {
                        "start_seconds": event.get("start"),
                        "end_seconds": event.get("end"),
                        "text": None,
                    },
                    "interpretations": event.get("interpretations"),
                    "evidence": heuristic(
                        True,
                        [SourceRef(f"$.cross_modal_events[{index}]", event.get("start"), event.get("end"))],
                        "preserved_upstream_cross_modal_interpretation",
                        _confidence(event.get("observed", {}).get("edit", {}).get("confidence")) or 0.5,
                    ),
                }
            )
    captions = _mapping(_mapping(report.get("text_overlay")).get("caption_analysis")).get("captions", [])
    if isinstance(captions, list):
        for index, caption in enumerate(captions):
            if not isinstance(caption, dict):
                continue
            highlighting = caption.get("word_highlighting", {})
            highlighted = highlighting.get("highlighted_words", []) if isinstance(highlighting, dict) else []
            if not highlighted:
                highlighted = caption.get("transcript_alignment", {}).get("highlighted_words", [])
            if highlighted:
                relationships.append(
                    {
                        "relationship": "overlay_highlights_spoken_words",
                        "timestamp_seconds": caption.get("start"),
                        "script_span": {
                            "start_seconds": caption.get("start"),
                            "end_seconds": caption.get("end"),
                            "text": caption.get("text"),
                        },
                        "highlighted_words": highlighted,
                        "evidence": observed(
                            True,
                            f"$.text_overlay.caption_analysis.captions[{index}]",
                            confidence=_confidence(caption.get("alignment_confidence")),
                            start=caption.get("start"),
                            end=caption.get("end"),
                            method="extractor_observed_overlay_and_alignment",
                        ),
                    }
                )
    return relationships


def _density_windows(sentences: list[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    windows = []
    start = 0.0
    while start < duration:
        end = min(duration, start + 10.0)
        selected = [
            sentence for sentence in sentences
            if isinstance(sentence.get("start"), (int, float)) and start <= float(sentence["start"]) < end
        ]
        text = " ".join(str(sentence.get("text", "")) for sentence in selected)
        windows.append(
            {
                "start_seconds": round(start, 3),
                "end_seconds": round(end, 3),
                "word_count": len(words(text)),
                "sentence_count": len(selected),
                "question_count": text.count("?"),
                "numerical_reference_count": len(NUMBER_RE.findall(text)),
                "example_marker_count": len(re.findall(r"\b(?:for example|for instance|such as)\b", text, re.I)),
            }
        )
        start = end
    return windows


class ScriptIntelligenceCompiler:
    def __init__(self, semantic_analyzer: SemanticAnalyzer | None = None):
        self.semantic_analyzer = semantic_analyzer or RuleBasedSemanticAnalyzer()

    @property
    def version(self) -> str:
        return COMPILER_VERSION

    def compile(self, report: dict[str, Any], *, artifact_sha256: str) -> CompiledIntelligence:
        source = report["source"]
        transcript = report["transcript"]
        context_metadata = _mapping(report.get("context"))
        raw_text = str(transcript["full_text"])
        clean_text = clean_transcript(raw_text)
        normalized_text = normalize_transcript(clean_text)
        sentences = _sentences(report, clean_text)
        token_count = len(words(clean_text))
        duration = float(source["duration_seconds"])
        sections = _upstream_sections(report, sentences)
        hook_section = next((section for section in sections if section["role"] == "hook"), sections[0] if sections else None)
        hook_text = hook_section["text"] if hook_section else sentences[0]["text"] if sentences else ""
        hook_start = hook_section.get("start_seconds") if hook_section else None
        hook_end = hook_section.get("end_seconds") if hook_section else None
        semantic_error = None
        analyzer_version = self.semantic_analyzer.version
        try:
            semantic = self.semantic_analyzer.analyze(
                SemanticContext(clean_text, sentences, hook_text, hook_start, hook_end, transcript.get("language"))
            )
        except Exception as exc:
            semantic_error = f"{type(exc).__name__}: {exc}"
            fallback = NullSemanticAnalyzer()
            semantic = fallback.analyze(
                SemanticContext(clean_text, sentences, hook_text, hook_start, hook_end, transcript.get("language"))
            )
            analyzer_version = f"{analyzer_version}->fallback:{fallback.version}"
        linguistic = linguistic_statistics(clean_text, sentences)
        relationships = _script_edit_relationships(report, sentences)

        structure_tokens = sorted(
            {
                *(f"section:{section['role']}" for section in sections),
                *(f"hook:{item['mechanism']}" for item in semantic.hook_mechanisms),
                *(f"retention:{item['device']}" for item in semantic.retention_devices),
                *(f"edit:{item['relationship']}" for item in relationships),
            }
        )
        mechanism_text = " ".join(item["mechanism"].replace("_", " ") for item in semantic.hook_mechanisms)
        device_text = " ".join(item["device"].replace("_", " ") for item in semantic.retention_devices)
        semantic_projection = clean_transcript(
            f"Hook: {hook_text}. Mechanisms: {mechanism_text}. Devices: {device_text}. Script: {clean_text}"
        )
        word_confidences = [
            float(item["confidence"])
            for item in transcript.get("words", [])
            if isinstance(item, dict) and isinstance(item.get("confidence"), (int, float))
        ]
        asr_confidence = (
            round(sum(word_confidences) / len(word_confidences), 4)
            if word_confidences
            else None
        )
        platform_evidence = (
            observed(context_metadata["platform"], "$.context.platform")
            if context_metadata.get("platform")
            else unknown("platform is not present in the extractor report or enrichment context")
        )
        topic_evidence = observed(context_metadata["topic"], "$.context.topic") if context_metadata.get("topic") else semantic.topic
        subtopic_evidence = observed(context_metadata["subtopic"], "$.context.subtopic") if context_metadata.get("subtopic") else semantic.subtopic
        format_evidence = observed(context_metadata["content_format"], "$.context.content_format") if context_metadata.get("content_format") else semantic.content_format
        audience_evidence = observed(context_metadata["audience_intent"], "$.context.audience_intent") if context_metadata.get("audience_intent") else semantic.audience_intent
        record_id = f"sir:{source['content_hash']}:{SCRIPT_INTELLIGENCE_SCHEMA_VERSION}"
        unknown_fields = [
            "script_structure.progression", "script_structure.climax", "script_structure.cta",
            "storytelling.*", "persuasion.*", "information_density.claims/facts/anecdotes/actionable_steps",
            "delivery.energy_changes", "delivery.speech_rhythm",
        ]
        for name, value in (
            ("identity.platform", platform_evidence),
            ("content.topic", topic_evidence),
            ("content.subtopic", subtopic_evidence),
            ("content.content_format", format_evidence),
            ("content.audience_intent", audience_evidence),
        ):
            if value["evidence_type"] == "unknown":
                unknown_fields.append(name)
        record: dict[str, Any] = {
            "schema_version": SCRIPT_INTELLIGENCE_SCHEMA_VERSION,
            "record_id": record_id,
            "compiler": {
                "name": "viralyst-script-intelligence-compiler",
                "version": COMPILER_VERSION,
                "semantic_analyzer_version": analyzer_version,
                "deterministic": True,
                "network_access": False,
            },
            "identity": {
                "report_id": observed(report["report_id"], "$.report_id"),
                "source_content_hash": observed(source["content_hash"], "$.source.content_hash"),
                "source_artifact_sha256": derived(artifact_sha256, [SourceRef("$raw_bytes")], "sha256_of_raw_report_bytes"),
                "extractor_version": observed(report["processing"]["extractor_version"], "$.processing.extractor_version"),
                "source_filename": observed(source.get("filename"), "$.source.filename") if source.get("filename") else unknown("source filename absent"),
                "platform": platform_evidence,
                "source_created_at": unknown("source publication timestamp is not present"),
                "source_modified_at": unknown("source modification timestamp is not present inside the report"),
            },
            "content": {
                "clean_transcript": observed(
                    clean_text,
                    "$.transcript.full_text",
                    confidence=asr_confidence,
                    method="extractor_ASR_with_mean_word_confidence" if transcript.get("words") else "extractor_ASR",
                ),
                "normalized_transcript": derived(normalized_text, [SourceRef("$.transcript.full_text")], "NFKC_whitespace_lowercase_word_normalization_v1"),
                "language": observed(transcript.get("language"), "$.transcript.language", confidence=_confidence(transcript.get("language_probability"))) if transcript.get("language") else unknown("language absent"),
                "topic": topic_evidence,
                "subtopic": subtopic_evidence,
                "content_format": format_evidence,
                "estimated_audience_intent": audience_evidence,
                "video_duration_seconds": observed(duration, "$.source.duration_seconds"),
                "spoken_word_count": derived(token_count, [SourceRef("$.transcript.full_text")], "compiler_tokenizer_v1"),
                "words_per_second": derived(round(token_count / duration, 4), [SourceRef("$.transcript.full_text"), SourceRef("$.source.duration_seconds")], "word_count_divided_by_video_duration"),
            },
            "script_structure": {
                "hook": _field_from_sections(sections, {"hook", "hook_candidate"}),
                "setup_context": _field_from_sections(sections, {"setup", "context"}),
                "major_beats": sections,
                "progression": semantic.progression,
                "climax": unknown("no verified climax label exists in the extractor"),
                "payoff": _field_from_sections(sections, {"payoff", "conclusion"}, multiple=True),
                "cta": _field_from_sections(sections, {"cta", "call_to_action"}),
                "ending_type": semantic.ending_type,
            },
            "hook_intelligence": {
                "text": hook_text,
                "start_seconds": hook_start,
                "end_seconds": hook_end,
                "mechanisms": semantic.hook_mechanisms,
                "multi_label": True,
            },
            "retention_devices": semantic.retention_devices,
            "linguistic_characteristics": {
                key: derived(value, [SourceRef("$.transcript.full_text"), SourceRef("$.transcript.sentences")], f"deterministic_{key}_v1")
                for key, value in linguistic.items()
            } | {
                "imperative_density_note": heuristic(
                    "candidate_only",
                    [SourceRef("$.transcript.sentences")],
                    "sentence_initial_imperative_lexicon_without_POS_parser",
                    0.55,
                ),
                "emotional_language_note": heuristic(
                    "lexicon_matches_only",
                    [SourceRef("$.transcript.full_text")],
                    "small_versioned_emotion_lexicon_v1",
                    0.55,
                ),
                "punchiness": heuristic(
                    {"short_sentence_share": linguistic["short_sentence_share"]},
                    [SourceRef("$.transcript.sentences")],
                    "sentences_at_most_8_words_proxy",
                    0.6,
                ),
                "parallelism": unknown("requires syntactic or semantic analysis"),
            },
            "storytelling": semantic.storytelling,
            "persuasion": semantic.persuasion,
            "information_density": {
                **semantic.information_units,
                "examples": heuristic(
                    [item for item in semantic.retention_devices if item["device"] == "example_or_proof"],
                    [SourceRef("$.transcript.sentences")],
                    "explicit_example_marker_rules_v1",
                    0.76,
                ) if any(item["device"] == "example_or_proof" for item in semantic.retention_devices) else unknown("no explicit example markers detected"),
                "redundancy_proxy": derived(
                    round(1 - linguistic["vocabulary"]["type_token_ratio"], 4),
                    [SourceRef("$.transcript.full_text")],
                    "one_minus_type_token_ratio_not_semantic_redundancy",
                ),
                "density_over_time": derived(
                    _density_windows(sentences, duration),
                    [SourceRef("$.transcript.sentences"), SourceRef("$.source.duration_seconds")],
                    "non_overlapping_10_second_lexical_windows_v1",
                ),
            },
            "delivery": _delivery(report, duration, token_count),
            "script_edit_relationships": relationships,
            "quality": {
                "content_evidence_available": True,
                "outcome_evidence_available": False,
                "performance_claims_allowed": False,
                "upstream_confidence_policy": report.get("confidence"),
                "deferred_upstream_features": report.get("deferred", []),
                "unknown_or_unreliable_fields": unknown_fields,
                "warnings": [
                    "upstream semantic section labels are preserved as unverified hypotheses",
                    "hook and retention mechanism rules are heuristic, not ground truth",
                    "no publication timestamp or outcome metadata was supplied",
                ] + (
                    ["platform/topic/audience metadata was not supplied"]
                    if not context_metadata else []
                ) + ([f"semantic analyzer failed gracefully: {semantic_error}"] if semantic_error else []),
            },
            "index_projections": {
                "lexical_text": clean_text,
                "semantic_text": semantic_projection,
                "structural_fingerprint": structure_tokens,
                "embedding_policy": {
                    "embed": ["semantic_text", "hook text and mechanisms", "section/beat summaries"],
                    "do_not_embed": ["raw extractor JSON", "frame arrays", "word-level acoustic arrays", "unverified hidden metadata"],
                },
            },
        }
        validate_script_intelligence_record(record)
        canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return CompiledIntelligence(record, canonical, hashlib.sha256(canonical.encode()).hexdigest())
