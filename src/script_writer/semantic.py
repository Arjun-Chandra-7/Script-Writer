from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import SourceRef, heuristic, unknown


HOOK_MECHANISMS = {
    "curiosity", "contrarian_claim", "pain_problem", "direct_promise", "transformation",
    "shocking_fact", "identity_callout", "challenge", "authority", "story_opening",
    "question", "result_first", "demonstration", "urgency", "social_proof",
}


@dataclass(frozen=True)
class SemanticContext:
    transcript: str
    sentences: list[dict[str, Any]]
    hook_text: str
    hook_start: float | None
    hook_end: float | None
    language: str | None


@dataclass(frozen=True)
class SemanticAnalysis:
    topic: dict[str, Any]
    subtopic: dict[str, Any]
    content_format: dict[str, Any]
    audience_intent: dict[str, Any]
    progression: dict[str, Any]
    ending_type: dict[str, Any]
    hook_mechanisms: list[dict[str, Any]]
    retention_devices: list[dict[str, Any]]
    storytelling: dict[str, Any]
    persuasion: dict[str, Any]
    information_units: dict[str, Any]


class SemanticAnalyzer(Protocol):
    @property
    def version(self) -> str: ...

    def analyze(self, context: SemanticContext) -> SemanticAnalysis: ...


def _mechanism(name: str, context: SemanticContext, confidence: float, method: str) -> dict[str, Any]:
    return {
        "mechanism": name,
        "evidence": heuristic(
            True,
            [SourceRef("$.transcript.full_text", context.hook_start, context.hook_end)],
            method,
            confidence,
        ),
        "text": context.hook_text,
        "start_seconds": context.hook_start,
        "end_seconds": context.hook_end,
    }


def _device(
    name: str,
    text: str,
    start: float | None,
    end: float | None,
    confidence: float,
    method: str,
) -> dict[str, Any]:
    return {
        "device": name,
        "evidence": heuristic(
            True,
            [SourceRef("$.transcript.sentences", start, end)],
            method,
            confidence,
        ),
        "text": text,
        "start_seconds": start,
        "end_seconds": end,
    }


class RuleBasedSemanticAnalyzer:
    """Conservative, local heuristics; never upgrades its output to observed fact."""

    version = "local-rules-1.0.0"

    def analyze(self, context: SemanticContext) -> SemanticAnalysis:
        hook = context.hook_text.lower()
        mechanisms: list[dict[str, Any]] = []
        if "?" in context.hook_text:
            mechanisms.append(_mechanism("question", context, 0.97, "hook_contains_question_mark"))
        if re.search(r"\b(did you know|what if|here's why|the secret|you won't believe)\b", hook):
            mechanisms.append(_mechanism("curiosity", context, 0.72, "curiosity_phrase_rules_v1"))
        if re.search(r"\b(how to|you can|you will|i'll show|this will)\b", hook):
            mechanisms.append(_mechanism("direct_promise", context, 0.7, "promise_phrase_rules_v1"))
        if re.search(r"\b(if you|for (?:every|any|all)|people who|creators|founders|parents)\b", hook):
            mechanisms.append(_mechanism("identity_callout", context, 0.68, "identity_phrase_rules_v1"))
        if re.search(r"\b(stop|never|wrong|myth|actually|truth)\b", hook):
            mechanisms.append(_mechanism("contrarian_claim", context, 0.64, "contrarian_phrase_rules_v1"))
        if re.search(r"\b(richest|expert|professor|doctor|nobel|award|founder|ceo)\b", hook):
            mechanisms.append(_mechanism("authority", context, 0.62, "authority_reference_rules_v1"))
        if re.search(r"\b(terrible|problem|pain|struggle|mistake|danger|threat)\b", hook):
            mechanisms.append(_mechanism("pain_problem", context, 0.66, "problem_phrase_rules_v1"))
        if re.search(r"\b(i was|when i|years ago|one day|this happened)\b", hook):
            mechanisms.append(_mechanism("story_opening", context, 0.7, "story_opening_rules_v1"))

        devices: list[dict[str, Any]] = []
        for sentence in context.sentences:
            text = str(sentence.get("text", ""))
            lowered = text.lower()
            start = sentence.get("start")
            end = sentence.get("end")
            if "?" in text:
                devices.append(_device("rhetorical_or_prompt_question", text, start, end, 0.68, "question_mark_rule"))
            if re.search(r"\b(but|however|yet|instead|although)\b", lowered):
                devices.append(_device("contrast", text, start, end, 0.82, "contrast_marker_rules_v1"))
            if re.search(r"\b(for example|for instance|such as|like)\b", lowered):
                devices.append(_device("example_or_proof", text, start, end, 0.76, "example_marker_rules_v1"))
            if re.search(r"\b(you|your|you're)\b", lowered):
                devices.append(_device("direct_address", text, start, end, 0.9, "second_person_pronoun_rule"))
            if re.search(r"\b(the only|the one|exactly|specifically|\d+)\b", lowered):
                devices.append(_device("specificity", text, start, end, 0.73, "specificity_marker_rules_v1"))
            if re.search(r"\b(worse|terrible|consequences|risk|lose|danger|threat)\b", lowered):
                devices.append(_device("stakes_language", text, start, end, 0.67, "stakes_lexicon_rules_v1"))

        unknown_story = {
            name: unknown("requires semantic inference or explicit source metadata")
            for name in (
                "protagonist", "goal", "conflict", "stakes", "turning_point", "resolution",
                "transformation", "chronology", "narrative_perspective",
            )
        }
        unknown_persuasion = {
            name: unknown("requires semantic inference beyond reliable local rules")
            for name in (
                "promise", "pain", "desired_outcome", "authority", "evidence", "proof",
                "objection_handling", "urgency", "scarcity", "social_proof", "risk_reversal",
                "cta_mechanism",
            )
        }
        unknown_units = {
            name: unknown("claim/fact classification requires a versioned semantic analyzer")
            for name in ("claims", "facts", "anecdotes", "actionable_steps")
        }
        return SemanticAnalysis(
            topic=unknown("topic is not provided by the extractor and local keyword inference is disabled"),
            subtopic=unknown("subtopic is not provided by the extractor"),
            content_format=unknown("speaker diarization and source format metadata are unavailable"),
            audience_intent=unknown("audience intent is not observed in the extractor report"),
            progression=unknown("reliable progression needs semantic inference"),
            ending_type=unknown("ending type needs semantic inference"),
            hook_mechanisms=mechanisms,
            retention_devices=devices,
            storytelling=unknown_story,
            persuasion=unknown_persuasion,
            information_units=unknown_units,
        )


class NullSemanticAnalyzer:
    version = "null-1.0.0"

    def analyze(self, context: SemanticContext) -> SemanticAnalysis:
        reason = "semantic analysis intentionally unavailable"
        return SemanticAnalysis(
            topic=unknown(reason),
            subtopic=unknown(reason),
            content_format=unknown(reason),
            audience_intent=unknown(reason),
            progression=unknown(reason),
            ending_type=unknown(reason),
            hook_mechanisms=[],
            retention_devices=[],
            storytelling={name: unknown(reason) for name in (
                "protagonist", "goal", "conflict", "stakes", "turning_point", "resolution",
                "transformation", "chronology", "narrative_perspective",
            )},
            persuasion={name: unknown(reason) for name in (
                "promise", "pain", "desired_outcome", "authority", "evidence", "proof",
                "objection_handling", "urgency", "scarcity", "social_proof", "risk_reversal",
                "cta_mechanism",
            )},
            information_units={name: unknown(reason) for name in (
                "claims", "facts", "anecdotes", "actionable_steps",
            )},
        )
