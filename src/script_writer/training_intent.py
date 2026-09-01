from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Protocol

from .contracts import SourceRef, derived, heuristic, unknown
from .training_contracts import evidence_value, known_evidence


INTENT_RECONSTRUCTOR_VERSION = "minimum-conditioning-rules-v1"


class IntentReconstructor(Protocol):
    version: str

    def reconstruct(
        self, intelligence: dict[str, Any], client_context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]: ...


def _client_field(context: dict[str, Any], name: str) -> dict[str, Any]:
    value = context.get("fields", {}).get(name)
    return value if isinstance(value, dict) else unknown(f"client field {name} unavailable")


def _named_concepts(text: str) -> list[str]:
    # Conservative surface-form extraction: no entity type or unstated meaning is inferred.
    candidates = re.findall(r"\b(?:[A-Z][A-Za-z]+(?:\s+[A-Z][A-Za-z]+){0,2}|[A-Z]{2,})\b", text)
    stop = {
        "Let", "Well", "Okay", "What", "So", "And", "But", "The", "I",
        "You", "He", "She", "It", "They", "We", "This", "That",
    }
    result: list[str] = []
    for candidate in candidates:
        if candidate in stop or candidate.lower() in {item.lower() for item in result}:
            continue
        result.append(candidate)
    return result[:8]


@dataclass(frozen=True)
class MinimumConditioningReconstructor:
    """Reconstructs only conditioning supported by the canonical evidence.

    It deliberately does not summarize or guess an original creator prompt.
    A future semantic implementation must use the same contract and identify
    itself with a stable version.
    """

    version: str = INTENT_RECONSTRUCTOR_VERSION

    def reconstruct(
        self, intelligence: dict[str, Any], client_context: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        content = intelligence["content"]
        structure = intelligence["script_structure"]
        transcript = str(evidence_value(content["clean_transcript"]) or "")
        concepts = _named_concepts(transcript)
        concept_evidence = (
            heuristic(
                concepts,
                [SourceRef("$.content.clean_transcript")],
                "capitalized_surface_forms_without_entity_typing_v1",
                0.72,
            )
            if concepts
            else unknown("no conservative required concepts could be extracted")
        )
        topic = content.get("topic")
        brief = {
            "language": content.get("language", unknown("language unavailable")),
            "topic": topic if known_evidence(topic) else unknown("topic requires reviewed semantic reconstruction"),
            "subtopic": content.get("subtopic", unknown("subtopic unavailable")),
            "central_idea": unknown("a central idea cannot be recovered reliably without semantic review"),
            "audience": content.get("audience_intent") if known_evidence(content.get("audience_intent")) else _client_field(client_context, "audience"),
            "audience_problem_desire": unknown("audience problem/desire is not explicit"),
            "content_objective": unknown("private creator objective cannot be recovered from the script"),
            "content_format": content.get("content_format", unknown("format unavailable")),
            "desired_outcome": unknown("desired outcome is not explicit"),
            "target_duration_seconds": content["video_duration_seconds"],
            "tone": _client_field(client_context, "tone"),
            "perspective": derived(
                intelligence["linguistic_characteristics"]["person_usage"]["value"],
                [SourceRef("$.linguistic_characteristics.person_usage")],
                "copy_measured_person_usage_v1",
            ),
            "sophistication_level": intelligence["linguistic_characteristics"].get("vocabulary", unknown("not measured")),
            "required_concepts": concept_evidence,
            "constraints": _client_field(client_context, "factual_constraints"),
            "prohibited_concepts": _client_field(client_context, "prohibited_topics_claims"),
        }
        roles = [str(item.get("role", "unknown")) for item in structure.get("major_beats", [])]
        hook_mechanisms = [item["mechanism"] for item in intelligence["hook_intelligence"].get("mechanisms", [])]
        plan = {
            "hook_mechanisms": heuristic(
                hook_mechanisms,
                [SourceRef("$.hook_intelligence.mechanisms")],
                "preserve_canonical_hook_hypotheses_v1",
                min([item.get("evidence", {}).get("confidence", 0.5) for item in intelligence["hook_intelligence"].get("mechanisms", [])] or [0.5]),
            ) if hook_mechanisms else unknown("no hook mechanism detected"),
            "progression": heuristic(
                roles,
                [SourceRef("$.script_structure.major_beats")],
                "abstract_roles_only_no_section_text_v1",
                0.58,
            ) if roles else unknown("no usable beat roles"),
            "open_loops": unknown("open-loop intent requires semantic reconstruction"),
            "payoff_strategy": unknown("payoff strategy requires semantic reconstruction"),
            "persuasion_strategy": unknown("persuasion strategy requires semantic reconstruction"),
            "information_density_strategy": derived(
                evidence_value(intelligence["information_density"].get("density_over_time")) or [],
                [SourceRef("$.information_density.density_over_time")],
                "preserve_measured_density_windows_v1",
            ),
            "storytelling_strategy": unknown("storytelling strategy is not verified"),
            "cta_mechanism": intelligence["persuasion"].get("cta_mechanism", unknown("CTA unavailable")),
        }
        return brief, plan
