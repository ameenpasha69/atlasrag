"""Reciprocal Rank Fusion.

    RRF(d) = sum over retrievers r of  weight_r / (k + rank_r(d))

Rank, never score. A BM25 score and a cosine similarity are different units, so combining
them numerically is meaningless; discarding the magnitudes and keeping only the positions
is what makes the two channels comparable at all.

This module is deliberately pure — no store, no index, no I/O — so its arithmetic can be
checked against fixtures computed by hand in tests/unit/test_fusion.py.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from atlasrag.domain.models import RankContribution, RetrieverName

DEFAULT_RRF_K = 60


@dataclass(frozen=True, slots=True)
class RetrieverResult:
    name: RetrieverName
    ranked: Sequence[tuple[str, float]]
    """(chunk_id, raw_score) ordered best-first. Ranks are derived from position."""


@dataclass(frozen=True, slots=True)
class FusedResult:
    chunk_id: str
    fused_score: float
    fused_rank: int
    contributions: list[RankContribution] = field(default_factory=list)


def reciprocal_rank_fusion(
    results: Sequence[RetrieverResult],
    *,
    k: int = DEFAULT_RRF_K,
    top_k: int = 10,
    weights: Mapping[str, float] | None = None,
) -> list[FusedResult]:
    if k < 1:
        raise ValueError("rrf k must be >= 1")

    accumulated: dict[str, float] = {}
    contributions: dict[str, list[RankContribution]] = {}

    for result in results:
        weight = float((weights or {}).get(result.name, 1.0))
        seen: set[str] = set()
        rank = 0
        for chunk_id, raw_score in result.ranked:
            # A retriever returning the same chunk twice must not be counted twice.
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            rank += 1
            term = 1.0 / (k + rank)
            accumulated[chunk_id] = accumulated.get(chunk_id, 0.0) + weight * term
            contributions.setdefault(chunk_id, []).append(
                RankContribution(
                    retriever=result.name,
                    rank=rank,
                    raw_score=float(raw_score),
                    rrf_term=term,
                    weight=weight,
                )
            )

    ordered = sorted(accumulated.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        FusedResult(
            chunk_id=chunk_id,
            fused_score=score,
            fused_rank=position,
            contributions=sorted(contributions[chunk_id], key=lambda c: c.retriever),
        )
        for position, (chunk_id, score) in enumerate(ordered[:top_k], start=1)
    ]
