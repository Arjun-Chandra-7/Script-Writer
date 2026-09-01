from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .text_analysis import normalize_transcript


LEAKAGE_DETECTOR_VERSION = "lexical-leakage-v1"


def _tokens(value: str) -> list[str]:
    return normalize_transcript(value).split()


def _flatten_conditioning(value: Any) -> str:
    parts: list[str] = []
    if isinstance(value, dict):
        if "evidence_type" in value:
            return _flatten_conditioning(value.get("value"))
        for key, child in value.items():
            if key not in {"sources", "method", "reason", "evidence_type", "confidence"}:
                parts.append(_flatten_conditioning(child))
    elif isinstance(value, list):
        parts.extend(_flatten_conditioning(item) for item in value)
    elif isinstance(value, (str, int, float)):
        parts.append(str(value))
    return " ".join(part for part in parts if part)


@dataclass(frozen=True)
class LeakagePolicy:
    high_common_sequence_tokens: int = 12
    reject_target_5gram_fraction: float = 0.20
    warn_token_jaccard: float = 0.35
    warn_sequence_tokens: int = 8


def leakage_metrics(conditioning: Any, target: str, policy: LeakagePolicy = LeakagePolicy()) -> dict[str, Any]:
    source_tokens = _tokens(_flatten_conditioning(conditioning))
    target_tokens = _tokens(target)
    source_set, target_set = set(source_tokens), set(target_tokens)
    union = source_set | target_set
    jaccard = len(source_set & target_set) / len(union) if union else 0.0
    matcher = SequenceMatcher(a=source_tokens, b=target_tokens, autojunk=False)
    longest = matcher.find_longest_match().size
    source_5 = {tuple(source_tokens[index:index + 5]) for index in range(max(0, len(source_tokens) - 4))}
    target_5 = {tuple(target_tokens[index:index + 5]) for index in range(max(0, len(target_tokens) - 4))}
    shared_5 = source_5 & target_5
    fraction = len(shared_5) / len(target_5) if target_5 else 0.0
    exact_sentences = []
    normalized_source = " ".join(source_tokens)
    for sentence in target.replace("?", ".").replace("!", ".").split("."):
        normalized = " ".join(_tokens(sentence))
        if len(normalized.split()) >= 8 and normalized in normalized_source:
            exact_sentences.append(normalized)
    high = bool(
        exact_sentences
        or longest >= policy.high_common_sequence_tokens
        or fraction >= policy.reject_target_5gram_fraction
    )
    warning = high or jaccard >= policy.warn_token_jaccard or longest >= policy.warn_sequence_tokens
    return {
        "detector_version": LEAKAGE_DETECTOR_VERSION,
        "conditioning_token_count": len(source_tokens),
        "target_token_count": len(target_tokens),
        "token_set_jaccard": round(jaccard, 6),
        "longest_common_contiguous_tokens": longest,
        "shared_target_5gram_fraction": round(fraction, 6),
        "exact_target_sentences": exact_sentences,
        "severity": "high" if high else "warning" if warning else "none",
        "rejected": high,
        "thresholds": {
            "high_common_sequence_tokens": policy.high_common_sequence_tokens,
            "reject_target_5gram_fraction": policy.reject_target_5gram_fraction,
            "warn_token_jaccard": policy.warn_token_jaccard,
            "warn_sequence_tokens": policy.warn_sequence_tokens,
        },
    }
