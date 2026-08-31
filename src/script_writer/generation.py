from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from .contracts import (
    GENERATION_CONTRACT_VERSION,
    validate_generation_request,
    validate_generation_result,
)
from .corpus import CorpusHit, CorpusIndex, SearchQuery


@dataclass(frozen=True)
class GenerationRequest:
    client_context_id: str
    niche: str
    topic: str
    objective: str
    audience: str
    desired_duration_seconds: float
    platform: str = "instagram_reels"
    tone: tuple[str, ...] = ()
    content_format: str | None = None
    constraints: tuple[str, ...] = ()
    factual_context: tuple[str, ...] = ()
    inspiration_record_ids: tuple[str, ...] = ()
    banned_patterns: tuple[str, ...] = ()
    desired_cta: str | None = None
    experimentation: dict[str, Any] = field(default_factory=dict)
    contract_version: str = GENERATION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not self.client_context_id.strip():
            raise ValueError("client_context_id is required")
        if not self.topic.strip() or not self.objective.strip() or not self.audience.strip():
            raise ValueError("topic, objective, and audience are required")
        if self.desired_duration_seconds <= 0:
            raise ValueError("desired_duration_seconds must be positive")
        validate_generation_request(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedEvidence:
    record_id: str
    report_id: str
    source_content_hash: str
    relevance_score: float
    hook_mechanisms: tuple[str, ...]
    retention_devices: tuple[str, ...]
    structural_fingerprint: tuple[str, ...]
    hook_text_for_analysis_only: str
    excerpt_for_analysis_only: str


@dataclass(frozen=True)
class GenerationContext:
    contract_version: str
    request: dict[str, Any]
    query: dict[str, Any]
    evidence: tuple[RetrievedEvidence, ...]
    instructions: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ScriptGenerator(Protocol):
    @property
    def version(self) -> str: ...

    def generate(self, context: GenerationContext) -> dict[str, Any]: ...


class DeterministicOutlineGenerator:
    """Local contract demonstration, not a claim of production creative quality."""

    version = "deterministic-outline-1.0.0"

    def generate(self, context: GenerationContext) -> dict[str, Any]:
        request = context.request
        audience = str(request["audience"])
        topic = str(request["topic"])
        objective = str(request["objective"])
        duration = float(request["desired_duration_seconds"])
        hook = f"{audience}: here is a focused way to think about {topic}."
        setup = f"The goal is {objective}."
        body = "Use the supplied factual context to develop one clear point, one example, and one practical next step."
        cta = str(request.get("desired_cta") or "Choose one next step and act on it today.")
        references = [
            {
                "record_id": item.record_id,
                "report_id": item.report_id,
                "source_content_hash": item.source_content_hash,
                "relevance_score": item.relevance_score,
            }
            for item in context.evidence
        ]
        mechanisms = sorted(
            {
                mechanism
                for item in context.evidence
                for mechanism in (*item.hook_mechanisms, *item.retention_devices)
            }
        )
        result = {
            "contract_version": GENERATION_CONTRACT_VERSION,
            "generator_version": self.version,
            "spoken_script": " ".join((hook, setup, body, cta)),
            "hook": hook,
            "sections": [
                {"role": "hook", "start_seconds": 0.0, "end_seconds": min(3.0, duration), "text": hook},
                {"role": "setup", "start_seconds": min(3.0, duration), "end_seconds": min(duration * 0.25, duration), "text": setup},
                {"role": "body", "start_seconds": min(duration * 0.25, duration), "end_seconds": min(duration * 0.85, duration), "text": body},
                {"role": "cta", "start_seconds": min(duration * 0.85, duration), "end_seconds": duration, "text": cta},
            ],
            "on_screen_text_suggestions": [{"at_seconds": 0.0, "text": topic}],
            "delivery_notes": [{"at_seconds": 0.0, "note": "Deliver the opening clearly; do not imitate a retrieved speaker."}],
            "visual_cues": [],
            "cta": cta,
            "claims_requiring_verification": list(request.get("factual_context", [])),
            "creative_mechanisms": mechanisms,
            "retrieved_evidence": references,
            "rationale": [
                {
                    "decision": "Use a direct audience callout and a single progression.",
                    "why": "The request names a specific audience and objective; retrieved records are structural evidence only.",
                    "evidence_record_ids": [item.record_id for item in context.evidence],
                },
                {
                    "decision": "Do not reuse retrieved wording.",
                    "why": "Corpus examples are evidence for mechanisms and structure, not copy templates.",
                    "evidence_record_ids": [],
                },
            ],
        }
        validate_generation_result(result)
        return result


class RetrievalFirstScriptWriter:
    def __init__(
        self,
        corpus: CorpusIndex,
        generator: ScriptGenerator,
        *,
        retrieval_count: int = 8,
    ):
        if retrieval_count <= 0:
            raise ValueError("retrieval_count must be positive")
        self.corpus = corpus
        self.generator = generator
        self.retrieval_count = retrieval_count

    def build_query(self, request: GenerationRequest) -> SearchQuery:
        desired_hooks = tuple(request.experimentation.get("hook_mechanisms", ()))
        desired_retention = tuple(request.experimentation.get("retention_devices", ()))
        inspiration_structure: set[str] = set()
        for record_id in request.inspiration_record_ids:
            record = self.corpus.get_record(record_id)
            inspiration_structure.update(
                record["index_projections"]["structural_fingerprint"]
            )
        text = " ".join(
            part for part in (request.niche, request.topic, request.objective, request.audience) if part
        )
        return SearchQuery(
            text=text,
            platform=request.platform or None,
            content_formats=(request.content_format,) if request.content_format else (),
            hook_mechanisms=desired_hooks,
            retention_devices=desired_retention,
            structural_fingerprint=tuple(sorted(inspiration_structure)),
            max_duration_seconds=max(request.desired_duration_seconds * 1.75, request.desired_duration_seconds + 10),
            top_k=self.retrieval_count,
            candidate_limit=max(500, self.retrieval_count * 25),
            diversity_strength=float(request.experimentation.get("diversity_strength", 0.15)),
        )

    def build_context(
        self, request: GenerationRequest, hits: list[CorpusHit], query: SearchQuery
    ) -> GenerationContext:
        evidence = tuple(
            RetrievedEvidence(
                record_id=hit.record_id,
                report_id=hit.report_id,
                source_content_hash=hit.source_content_hash,
                relevance_score=round(hit.score, 6),
                hook_mechanisms=tuple(
                    item["mechanism"] for item in hit.record["hook_intelligence"]["mechanisms"]
                ),
                retention_devices=tuple(
                    item["device"] for item in hit.record["retention_devices"]
                ),
                structural_fingerprint=tuple(
                    hit.record["index_projections"]["structural_fingerprint"]
                ),
                hook_text_for_analysis_only=hit.hook_text,
                excerpt_for_analysis_only=hit.excerpt,
            )
            for hit in hits
        )
        return GenerationContext(
            contract_version=GENERATION_CONTRACT_VERSION,
            request=request.to_dict(),
            query={
                "text": query.text,
                "platform": query.platform,
                "content_formats": list(query.content_formats),
                "hook_mechanisms": list(query.hook_mechanisms),
                "retention_devices": list(query.retention_devices),
                "max_duration_seconds": query.max_duration_seconds,
            },
            evidence=evidence,
            instructions=(
                "Use retrieved examples as evidence for structure and mechanisms, never as copy templates.",
                "Do not reproduce a contiguous phrase from a source script.",
                "Ground factual claims only in request.factual_context and flag every claim needing verification.",
                "Return the versioned structured result contract.",
                "Explain major decisions with retrieved record IDs where relevant.",
            ),
        )

    def generate(self, request: GenerationRequest) -> dict[str, Any]:
        query = self.build_query(request)
        hits = self.corpus.search(query)
        context = self.build_context(request, hits, query)
        result = self.generator.generate(context)
        validate_generation_result(result)
        generated_text = " ".join(
            str(result.get(field, "")) for field in ("spoken_script", "hook", "cta")
        ).casefold()
        violated = [pattern for pattern in request.banned_patterns if pattern.casefold() in generated_text]
        if violated:
            raise ValueError(f"generator used banned patterns: {violated}")
        return result
