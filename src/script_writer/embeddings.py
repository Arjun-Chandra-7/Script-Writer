from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Protocol

from .text_analysis import normalize_transcript


class EmbeddingProvider(Protocol):
    @property
    def version(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, text: str) -> list[float]: ...


class HashingEmbeddingProvider:
    """Dependency-free signed feature hashing for deterministic local tests/baselines."""

    def __init__(self, dimensions: int = 192):
        if dimensions < 32:
            raise ValueError("dimensions must be at least 32")
        self._dimensions = dimensions

    @property
    def version(self) -> str:
        return f"signed-hashing-unigram-bigram-v1:{self._dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed(self, text: str) -> list[float]:
        tokens = normalize_transcript(text).split()
        features = [f"u:{token}" for token in tokens]
        features.extend(f"b:{left}_{right}" for left, right in zip(tokens, tokens[1:]))
        counts = Counter(features)
        vector = [0.0] * self._dimensions
        for feature, count in counts.items():
            digest = hashlib.blake2b(feature.encode(), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "big") % self._dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        norm = math.sqrt(sum(value * value for value in vector))
        if norm:
            vector = [round(value / norm, 8) for value in vector]
        return vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have equal dimensions")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)
