"""Evidence selection: which retrieved hits are allowed to support an answer."""

from __future__ import annotations

from atlasrag.domain.models import EvidenceSet
from atlasrag.retrieval.engine import SearchOutcome


def build_evidence(query_text: str, outcome: SearchOutcome, *, top_n: int) -> EvidenceSet:
    selected = outcome.hits[:top_n]
    return EvidenceSet(
        query=query_text,
        hits=outcome.hits,
        selected=selected,
        max_fused_score=outcome.hits[0].fused_score if outcome.hits else 0.0,
        max_bm25_score=outcome.max_bm25_score,
        max_cosine_score=outcome.max_cosine_score,
    )
