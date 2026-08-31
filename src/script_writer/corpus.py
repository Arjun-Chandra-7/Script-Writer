from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any

from .database import Registry
from .embeddings import EmbeddingProvider, HashingEmbeddingProvider, cosine_similarity


@dataclass(frozen=True)
class SearchQuery:
    text: str = ""
    platform: str | None = None
    content_formats: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    hook_mechanisms: tuple[str, ...] = ()
    retention_devices: tuple[str, ...] = ()
    structural_fingerprint: tuple[str, ...] = ()
    min_duration_seconds: float | None = None
    max_duration_seconds: float | None = None
    exclude_source_hashes: tuple[str, ...] = ()
    top_k: int = 10
    candidate_limit: int = 500
    diversity_strength: float = 0.15

    def __post_init__(self) -> None:
        if not 1 <= self.top_k <= 100:
            raise ValueError("top_k must be between 1 and 100")
        if self.candidate_limit < self.top_k:
            raise ValueError("candidate_limit must be at least top_k")
        if not 0 <= self.diversity_strength <= 1:
            raise ValueError("diversity_strength must be between 0 and 1")


@dataclass(frozen=True)
class CorpusHit:
    record_id: str
    report_id: str
    source_content_hash: str
    score: float
    lexical_score: float
    semantic_score: float
    structural_score: float
    matched_mechanisms: tuple[str, ...]
    hook_text: str
    excerpt: str
    record: dict[str, Any] = field(compare=False)

    def summary(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "report_id": self.report_id,
            "source_content_hash": self.source_content_hash,
            "score": round(self.score, 6),
            "lexical_score": round(self.lexical_score, 6),
            "semantic_score": round(self.semantic_score, 6),
            "structural_score": round(self.structural_score, 6),
            "matched_mechanisms": list(self.matched_mechanisms),
            "hook_text": self.hook_text,
            "excerpt": self.excerpt,
        }


def _fts_query(text: str) -> str | None:
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    if not tokens:
        return None
    return " OR ".join(f'"{token}"' for token in tokens[:32])


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


class CorpusIndex:
    """SQLite metadata/FTS index plus a pluggable embedding projection."""

    def __init__(
        self,
        registry: Registry,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        self.registry = registry
        self.embedding_provider = embedding_provider or HashingEmbeddingProvider()

    def rebuild(self, *, force: bool = False, batch_size: int = 500) -> dict[str, int]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if force:
            with self.registry.transaction(immediate=True) as connection:
                connection.execute(
                    "DELETE FROM intelligence_embeddings WHERE provider_version = ?",
                    (self.embedding_provider.version,),
                )
                connection.execute("DELETE FROM intelligence_fts")
                connection.execute(
                    """
                    INSERT INTO intelligence_fts(intelligence_id, search_text)
                    SELECT id, search_text FROM intelligence_records
                    WHERE compile_status = 'ready'
                    ORDER BY id
                    """
                )
        rows = self.registry.connection.execute(
            """
            SELECT ir.id, ir.record_json
            FROM intelligence_records ir
            LEFT JOIN intelligence_embeddings ie
              ON ie.intelligence_id = ir.id
             AND ie.projection = 'semantic_text'
             AND ie.provider_version = ?
            WHERE ir.compile_status = 'ready' AND ie.intelligence_id IS NULL
            ORDER BY ir.id
            LIMIT ?
            """,
            (self.embedding_provider.version, batch_size),
        ).fetchall()
        indexed = 0
        with self.registry.transaction(immediate=True) as connection:
            for row in rows:
                record = json.loads(row["record_json"])
                text = str(record["index_projections"]["semantic_text"])
                vector = self.embedding_provider.embed(text)
                connection.execute(
                    """
                    INSERT INTO intelligence_embeddings(
                        intelligence_id, projection, provider_version, dimensions, vector_json
                    ) VALUES (?, 'semantic_text', ?, ?, ?)
                    ON CONFLICT(intelligence_id, projection, provider_version) DO UPDATE SET
                        dimensions = excluded.dimensions, vector_json = excluded.vector_json
                    """,
                    (
                        row["id"],
                        self.embedding_provider.version,
                        self.embedding_provider.dimensions,
                        json.dumps(vector, separators=(",", ":")),
                    ),
                )
                indexed += 1
        remaining = int(
            self.registry.connection.execute(
                """
                SELECT COUNT(*) FROM intelligence_records ir
                LEFT JOIN intelligence_embeddings ie
                  ON ie.intelligence_id = ir.id
                 AND ie.projection = 'semantic_text'
                 AND ie.provider_version = ?
                WHERE ir.compile_status = 'ready' AND ie.intelligence_id IS NULL
                """,
                (self.embedding_provider.version,),
            ).fetchone()[0]
        )
        return {"indexed": indexed, "remaining": remaining}

    def get_record(self, record_id: str) -> dict[str, Any]:
        row = self.registry.connection.execute(
            "SELECT record_json FROM intelligence_records WHERE json_extract(record_json, '$.record_id') = ?",
            (record_id,),
        ).fetchone()
        if row is None:
            raise KeyError(record_id)
        return json.loads(row["record_json"])

    def rebuild_all(self, *, force: bool = False, batch_size: int = 500) -> dict[str, int]:
        total = 0
        first = True
        while True:
            result = self.rebuild(force=force and first, batch_size=batch_size)
            first = False
            total += result["indexed"]
            if result["remaining"] == 0 or result["indexed"] == 0:
                return {"indexed": total, "remaining": result["remaining"]}

    def search(self, query: SearchQuery) -> list[CorpusHit]:
        fts = _fts_query(query.text)
        parameters: list[object] = []
        conditions = ["ir.compile_status = 'ready'"]
        if query.platform:
            conditions.append("ir.platform = ?")
            parameters.append(query.platform)
        if query.content_formats:
            placeholders = ",".join("?" for _ in query.content_formats)
            conditions.append(f"ir.content_format IN ({placeholders})")
            parameters.extend(query.content_formats)
        if query.topics:
            placeholders = ",".join("?" for _ in query.topics)
            conditions.append(f"ir.topic IN ({placeholders})")
            parameters.extend(query.topics)
        if query.min_duration_seconds is not None:
            conditions.append("ir.duration_seconds >= ?")
            parameters.append(query.min_duration_seconds)
        if query.max_duration_seconds is not None:
            conditions.append("ir.duration_seconds <= ?")
            parameters.append(query.max_duration_seconds)

        if fts:
            sql = f"""
                SELECT ir.id, ir.record_json, ir.structural_fingerprint,
                       bm25(intelligence_fts) AS lexical_rank, ie.vector_json
                FROM intelligence_fts
                JOIN intelligence_records ir
                  ON ir.id = intelligence_fts.intelligence_id
                LEFT JOIN intelligence_embeddings ie
                  ON ie.intelligence_id = ir.id
                 AND ie.projection = 'semantic_text'
                 AND ie.provider_version = ?
                WHERE intelligence_fts MATCH ? AND {' AND '.join(conditions)}
                ORDER BY lexical_rank
                LIMIT ?
            """
            rows = self.registry.connection.execute(
                sql,
                [self.embedding_provider.version, fts, *parameters, query.candidate_limit],
            ).fetchall()
        else:
            sql = f"""
                SELECT ir.id, ir.record_json, ir.structural_fingerprint,
                       NULL AS lexical_rank, ie.vector_json
                FROM intelligence_records ir
                LEFT JOIN intelligence_embeddings ie
                  ON ie.intelligence_id = ir.id
                 AND ie.projection = 'semantic_text'
                 AND ie.provider_version = ?
                WHERE {' AND '.join(conditions)}
                ORDER BY ir.id
                LIMIT ?
            """
            rows = self.registry.connection.execute(
                sql, [self.embedding_provider.version, *parameters, query.candidate_limit]
            ).fetchall()

        query_vector = self.embedding_provider.embed(query.text) if query.text else None
        required_hook = set(query.hook_mechanisms)
        required_retention = set(query.retention_devices)
        target_structure = set(query.structural_fingerprint)
        excluded = set(query.exclude_source_hashes)
        scored: list[tuple[CorpusHit, list[float] | None]] = []
        total_rows = max(1, len(rows))
        for rank, row in enumerate(rows):
            record = json.loads(row["record_json"])
            source_hash = str(record["identity"]["source_content_hash"]["value"])
            if source_hash in excluded:
                continue
            hook_mechanisms = {
                item["mechanism"] for item in record["hook_intelligence"]["mechanisms"]
            }
            retention = {item["device"] for item in record["retention_devices"]}
            if not required_hook.issubset(hook_mechanisms):
                continue
            if not required_retention.issubset(retention):
                continue
            vector = json.loads(row["vector_json"]) if row["vector_json"] else None
            semantic_score = (
                max(0.0, cosine_similarity(query_vector, vector))
                if query_vector is not None and vector is not None else 0.0
            )
            lexical_score = (1.0 - rank / total_rows) if fts else 0.0
            record_structure = set(json.loads(row["structural_fingerprint"]))
            structural_score = _jaccard(target_structure, record_structure)
            if fts:
                score = 0.5 * semantic_score + 0.4 * lexical_score + 0.1 * structural_score
            elif target_structure:
                score = structural_score
            else:
                score = 1.0
            matched = tuple(sorted((required_hook & hook_mechanisms) | (required_retention & retention)))
            transcript = str(record["content"]["clean_transcript"]["value"])
            hit = CorpusHit(
                record_id=str(record["record_id"]),
                report_id=str(record["identity"]["report_id"]["value"]),
                source_content_hash=source_hash,
                score=score,
                lexical_score=lexical_score,
                semantic_score=semantic_score,
                structural_score=structural_score,
                matched_mechanisms=matched,
                hook_text=str(record["hook_intelligence"]["text"]),
                excerpt=transcript[:280],
                record=record,
            )
            scored.append((hit, vector))
        scored.sort(key=lambda pair: (-pair[0].score, pair[0].record_id))
        return self._diversify(scored, query.top_k, query.diversity_strength)

    @staticmethod
    def _diversify(
        candidates: list[tuple[CorpusHit, list[float] | None]],
        top_k: int,
        strength: float,
    ) -> list[CorpusHit]:
        selected: list[tuple[CorpusHit, list[float] | None]] = []
        remaining = list(candidates)
        while remaining and len(selected) < top_k:
            best_index = 0
            best_value = float("-inf")
            for index, (hit, vector) in enumerate(remaining):
                redundancy = 0.0
                if vector is not None:
                    redundancy = max(
                        (
                            max(0.0, cosine_similarity(vector, chosen_vector))
                            for _chosen, chosen_vector in selected
                            if chosen_vector is not None
                        ),
                        default=0.0,
                    )
                value = hit.score - strength * redundancy
                if value > best_value:
                    best_value, best_index = value, index
            selected.append(remaining.pop(best_index))
        return [hit for hit, _vector in selected]

    def structurally_similar(self, record_id: str, *, top_k: int = 10) -> list[CorpusHit]:
        record = self.get_record(record_id)
        fingerprint = set(record["index_projections"]["structural_fingerprint"])
        source_hash = str(record["identity"]["source_content_hash"]["value"])
        current_id = int(
            self.registry.connection.execute(
                "SELECT id FROM intelligence_records WHERE json_extract(record_json, '$.record_id') = ?",
                (record_id,),
            ).fetchone()[0]
        )
        # Fingerprints are compact, so exact Jaccard over 50k rows is bounded and
        # avoids loading the much larger record JSON for non-finalists.
        rows = self.registry.connection.execute(
            "SELECT id, structural_fingerprint FROM intelligence_records WHERE compile_status = 'ready'"
        ).fetchall()
        ranked = sorted(
            (
                (_jaccard(fingerprint, set(json.loads(item["structural_fingerprint"]))), int(item["id"]))
                for item in rows
                if int(item["id"]) != current_id
            ),
            key=lambda pair: (-pair[0], pair[1]),
        )[:top_k]
        hits = []
        for score, intelligence_id in ranked:
            item = self.registry.connection.execute(
                "SELECT record_json FROM intelligence_records WHERE id = ?", (intelligence_id,)
            ).fetchone()
            if item is None:
                continue
            candidate = json.loads(item["record_json"])
            candidate_hash = str(candidate["identity"]["source_content_hash"]["value"])
            if candidate_hash == source_hash:
                continue
            transcript = str(candidate["content"]["clean_transcript"]["value"])
            hits.append(
                CorpusHit(
                    record_id=str(candidate["record_id"]),
                    report_id=str(candidate["identity"]["report_id"]["value"]),
                    source_content_hash=candidate_hash,
                    score=score,
                    lexical_score=0.0,
                    semantic_score=0.0,
                    structural_score=score,
                    matched_mechanisms=(),
                    hook_text=str(candidate["hook_intelligence"]["text"]),
                    excerpt=transcript[:280],
                    record=candidate,
                )
            )
        return hits
