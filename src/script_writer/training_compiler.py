from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .contracts import SourceRef, derived, heuristic, unknown
from .text_analysis import normalize_transcript
from .training_contracts import (
    Eligibility,
    TRAINING_COMPILER_VERSION,
    TRAINING_EXAMPLE_SCHEMA_VERSION,
    TrainingObjective,
    canonical_json,
    evidence_value,
    known_evidence,
    validate_client_training_context,
    validate_training_example,
)
from .training_intent import IntentReconstructor, MinimumConditioningReconstructor
from .training_leakage import LeakagePolicy, leakage_metrics


@dataclass(frozen=True)
class CompilationResult:
    examples: tuple[dict[str, Any], ...]
    rejections: tuple[dict[str, Any], ...]


def _confidence_values(value: Any) -> list[float]:
    result: list[float] = []
    if isinstance(value, dict):
        if value.get("evidence_type") and value.get("value") is not None:
            result.append(float(value.get("confidence", 1.0)))
        for child in value.values():
            result.extend(_confidence_values(child))
    elif isinstance(value, list):
        for child in value:
            result.extend(_confidence_values(child))
    return result


def _known_fraction(value: dict[str, Any]) -> float:
    envelopes = [item for item in value.values() if isinstance(item, dict) and "evidence_type" in item]
    return sum(known_evidence(item) for item in envelopes) / len(envelopes) if envelopes else 0.0


def _evidence_quality(
    intelligence: dict[str, Any], brief: dict[str, Any], plan: dict[str, Any], leakage: dict[str, Any]
) -> dict[str, Any]:
    transcript = intelligence["content"]["clean_transcript"]
    transcript_component = 1.0 if known_evidence(transcript) else 0.0
    brief_component = _known_fraction(brief)
    plan_component = _known_fraction(plan)
    confidences = _confidence_values({"brief": brief, "plan": plan})
    provenance_component = sum(confidences) / len(confidences) if confidences else 0.0
    leakage_component = 0.0 if leakage["severity"] == "high" else 0.6 if leakage["severity"] == "warning" else 1.0
    components = {
        "transcript_integrity": round(transcript_component, 4),
        "brief_completeness": round(brief_component, 4),
        "plan_completeness": round(plan_component, 4),
        "provenance_confidence": round(provenance_component, 4),
        "leakage_safety": round(leakage_component, 4),
    }
    weights = {
        "transcript_integrity": 0.25,
        "brief_completeness": 0.25,
        "plan_completeness": 0.15,
        "provenance_confidence": 0.20,
        "leakage_safety": 0.15,
    }
    return {
        "name": "training_evidence_quality",
        "value": round(sum(components[key] * weights[key] for key in weights), 4),
        "components": components,
        "weights": weights,
        "meaning": "evidence/readiness only; not creative quality or predicted performance",
    }


class TrainingExampleCompiler:
    def __init__(
        self,
        reconstructor: IntentReconstructor | None = None,
        leakage_policy: LeakagePolicy = LeakagePolicy(),
    ):
        self.reconstructor = reconstructor or MinimumConditioningReconstructor()
        self.leakage_policy = leakage_policy

    def compile(
        self,
        intelligence: dict[str, Any],
        client_context: dict[str, Any],
        *,
        group_id: str,
        split: str,
    ) -> CompilationResult:
        validate_client_training_context(client_context)
        transcript = str(evidence_value(intelligence["content"]["clean_transcript"]) or "").strip()
        source_hash = str(evidence_value(intelligence["identity"]["source_content_hash"]) or "")
        if not transcript or not source_hash:
            return CompilationResult((), ({"objective": "all", "reasons": ["missing_script_or_source_identity"]},))
        brief, plan = self.reconstructor.reconstruct(intelligence, client_context)
        variants = self._objective_variants(intelligence, transcript, brief, plan)
        examples: list[dict[str, Any]] = []
        rejections: list[dict[str, Any]] = []
        for objective, variant_id, target, target_metadata, objective_reasons in variants:
            conditioning_plan = dict(plan)
            if objective is TrainingObjective.STRUCTURE:
                # The abstract role sequence is the target for this objective.
                conditioning_plan.pop("progression", None)
            conditioning = {"content_brief": brief, "creative_plan": conditioning_plan}
            conditioning.update(target_metadata.pop("conditioning", {}))
            leakage = leakage_metrics(conditioning, target, self.leakage_policy)
            reasons = list(objective_reasons)
            warnings: list[str] = []
            if leakage["rejected"]:
                reasons.append("high_severity_target_leakage")
            elif leakage["severity"] == "warning":
                warnings.append("conditioning_target_similarity_warning")
            if _known_fraction(brief) < 0.20:
                reasons.append("insufficient_brief_reconstruction")
            has_subject = known_evidence(brief.get("topic")) or known_evidence(brief.get("central_idea"))
            if not has_subject and objective is not TrainingObjective.CONTINUATION:
                reasons.append("missing_reliable_topic_or_central_idea")
            if any(
                item.get("upstream_verification_status") == "structural_hypothesis_unverified"
                for item in intelligence["script_structure"].get("major_beats", [])
            ) and objective in {TrainingObjective.HOOK, TrainingObjective.STRUCTURE, TrainingObjective.SECTION}:
                warnings.append("uses_unverified_structural_hypothesis")
            eligibility = Eligibility.INELIGIBLE if reasons else Eligibility.WARNING if warnings else Eligibility.ELIGIBLE
            identity_seed = ":".join(
                [source_hash, objective.value, variant_id, client_context["context_id"], TRAINING_COMPILER_VERSION]
            )
            example = {
                "schema_version": TRAINING_EXAMPLE_SCHEMA_VERSION,
                "example_id": f"ste:{hashlib.sha256(identity_seed.encode()).hexdigest()[:24]}",
                "identity": {
                    "source_video_id": source_hash,
                    "source_report_id": evidence_value(intelligence["identity"]["report_id"]),
                    "script_intelligence_record_id": intelligence["record_id"],
                    "script_intelligence_schema_version": intelligence["schema_version"],
                    "compiler_version": TRAINING_COMPILER_VERSION,
                    "intent_reconstructor_version": self.reconstructor.version,
                    "client_id": client_context["client_id"],
                    "source_content_hash": source_hash,
                    "dataset_objective": objective.value,
                    "source_group_id": group_id,
                    "split": split,
                    "variant_id": variant_id,
                    "source_attributes": {
                        "language": evidence_value(intelligence["content"].get("language")),
                        "duration_seconds": evidence_value(intelligence["content"].get("video_duration_seconds")),
                        "word_count": evidence_value(intelligence["content"].get("spoken_word_count")),
                        "words_per_second": evidence_value(intelligence["content"].get("words_per_second")),
                        "topic": evidence_value(intelligence["content"].get("topic")),
                        "content_format": evidence_value(intelligence["content"].get("content_format")),
                    },
                },
                "client_context_ref": {
                    "context_id": client_context["context_id"],
                    "schema_version": client_context["schema_version"],
                    "source_sha256": client_context["source"]["sha256"],
                },
                "training_input": conditioning,
                "creative_plan": plan,
                "target_output": {"text": target, **target_metadata},
                "provenance": {
                    "source_record_id": intelligence["record_id"],
                    "target": [
                        {"path": "$.content.clean_transcript"}
                        if objective in {TrainingObjective.FULL_SCRIPT, TrainingObjective.STYLE}
                        else {"path": f"$.script_structure.{variant_id}"}
                    ],
                    "reconstruction_evidence_types": sorted(
                        {item for item in self._evidence_types({"brief": brief, "plan": plan})}
                    ),
                },
                "quality": {
                    "eligibility": eligibility.value,
                    "exclusion_reasons": sorted(set(reasons)),
                    "warnings": sorted(set(warnings)),
                    "leakage": leakage,
                    "training_evidence_quality": _evidence_quality(intelligence, brief, plan, leakage),
                    "performance_signal_used": False,
                },
                "review": {"status": "unreviewed", "decision": None},
            }
            validate_training_example(example)
            if eligibility is Eligibility.INELIGIBLE:
                rejections.append(
                    {"example_id": example["example_id"], "objective": objective.value, "reasons": example["quality"]["exclusion_reasons"]}
                )
            examples.append(example)
        return CompilationResult(tuple(examples), tuple(rejections))

    @staticmethod
    def _evidence_types(value: Any) -> list[str]:
        found: list[str] = []
        if isinstance(value, dict):
            if "evidence_type" in value:
                found.append(str(value["evidence_type"]))
            for child in value.values():
                found.extend(TrainingExampleCompiler._evidence_types(child))
        elif isinstance(value, list):
            for child in value:
                found.extend(TrainingExampleCompiler._evidence_types(child))
        return found

    def _objective_variants(
        self,
        intelligence: dict[str, Any],
        transcript: str,
        brief: dict[str, Any],
        plan: dict[str, Any],
    ) -> list[tuple[TrainingObjective, str, str, dict[str, Any], list[str]]]:
        variants: list[tuple[TrainingObjective, str, str, dict[str, Any], list[str]]] = []
        word_count = len(normalize_transcript(transcript).split())
        full_reasons = [] if word_count >= 20 else ["script_too_short"]
        transcript_confidence = intelligence["content"].get("transcript_confidence")
        if known_evidence(transcript_confidence) and float(evidence_value(transcript_confidence)) < 0.7:
            full_reasons.append("transcript_confidence_too_low")
        variants.append((TrainingObjective.FULL_SCRIPT, "full_script", transcript, {"target_type": "spoken_script"}, full_reasons))

        hook = intelligence["script_structure"].get("hook", {})
        hook_value = evidence_value(hook)
        hook_text = hook_value.get("text", "") if isinstance(hook_value, dict) else ""
        hook_confidence = float(hook.get("confidence", 0)) if isinstance(hook, dict) else 0.0
        if hook_text:
            reasons = [] if hook_confidence >= 0.6 else ["hook_boundary_confidence_too_low"]
            variants.append((TrainingObjective.HOOK, "hook", hook_text, {"target_type": "hook"}, reasons))
            if transcript.startswith(hook_text):
                remainder = transcript[len(hook_text):].strip()
            else:
                remainder = " ".join(
                    str(item.get("text", "")) for item in intelligence["script_structure"].get("major_beats", [])[1:]
                ).strip()
            if remainder:
                variants.append(
                    (
                        TrainingObjective.CONTINUATION,
                        "continuation",
                        remainder,
                        {
                            "target_type": "remaining_spoken_script",
                            "conditioning": {"opening": heuristic(hook_text, [SourceRef("$.script_structure.hook")], "observed_opening_is_not_part_of_continuation_target_v1", hook_confidence)},
                        },
                        [],
                    )
                )
        roles = [str(item.get("role", "unknown")) for item in intelligence["script_structure"].get("major_beats", [])]
        malformed_timing = any(
            not isinstance(item.get("start_seconds"), (int, float))
            or not isinstance(item.get("end_seconds"), (int, float))
            or float(item["start_seconds"]) < 0
            or float(item["end_seconds"]) <= float(item["start_seconds"])
            for item in intelligence["script_structure"].get("major_beats", [])
        )
        if roles:
            variants.append(
                (
                    TrainingObjective.STRUCTURE,
                    "major_beats",
                    " -> ".join(roles),
                    {"target_type": "abstract_beat_roles", "roles": roles},
                    (["incomplete_structure"] if len(roles) < 3 else [])
                    + (["malformed_section_timing"] if malformed_timing else []),
                )
            )
        for index, section in enumerate(intelligence["script_structure"].get("major_beats", [])):
            text = str(section.get("text", "")).strip()
            role = str(section.get("role", "unknown"))
            if not text:
                continue
            confidence = float(section.get("evidence", {}).get("confidence", 0))
            variants.append(
                (
                    TrainingObjective.SECTION,
                    f"section_{index}",
                    text,
                    {
                        "target_type": "script_section",
                        "section_role": role,
                        "start_seconds": section.get("start_seconds"),
                        "end_seconds": section.get("end_seconds"),
                        "conditioning": {"section_role": derived(role, [SourceRef(f"$.script_structure.major_beats[{index}].role")], "copy_abstract_section_role_v1")},
                    },
                    ([] if confidence >= 0.55 else ["section_alignment_confidence_too_low"])
                    + (["malformed_section_timing"] if (
                        not isinstance(section.get("start_seconds"), (int, float))
                        or not isinstance(section.get("end_seconds"), (int, float))
                        or float(section["start_seconds"]) < 0
                        or float(section["end_seconds"]) <= float(section["start_seconds"])
                    ) else []),
                )
            )
        cta = intelligence["script_structure"].get("cta", {})
        cta_value = evidence_value(cta)
        if isinstance(cta_value, dict) and cta_value.get("text"):
            variants.append((TrainingObjective.CTA, "cta", str(cta_value["text"]), {"target_type": "cta"}, []))
        linguistic = intelligence["linguistic_characteristics"]
        style = {
            "words_per_second": intelligence["content"]["words_per_second"],
            "sentence_length_distribution": linguistic["sentence_length_distribution"],
            "question_density": linguistic["question_density_per_sentence"],
            "imperative_density": linguistic["imperative_candidate_density"],
            "short_sentence_share": linguistic["short_sentence_share"],
            "person_usage": linguistic["person_usage"],
        }
        variants.append(
            (
                TrainingObjective.STYLE,
                "measured_style",
                transcript,
                {"target_type": "style_conditioned_spoken_script", "conditioning": {"measurable_style": style}},
                full_reasons,
            )
        )
        return variants
