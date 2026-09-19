"""Search orchestration: filters, both channels, fusion, hydration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from atlasrag.config import Settings
from atlasrag.domain.models import SearchHit, SearchQuery
from atlasrag.domain.protocols import EmbeddingModel
from atlasrag.errors import ModelUnavailableError
from atlasrag.indexing.manager import IndexManager
from atlasrag.retrieval.fusion import RetrieverResult, reciprocal_rank_fusion
from atlasrag.storage.sqlite_store import SqliteDocumentStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SearchOutcome:
    hits: list[SearchHit] = field(default_factory=list)
    max_bm25_score: float | None = None
    max_cosine_score: float | None = None


class SearchEngine:
    def __init__(
        self,
        store: SqliteDocumentStore,
        indexes: IndexManager,
        settings: Settings,
        embedder: EmbeddingModel | None = None,
    ) -> None:
        self._store = store
        self._indexes = indexes
        self._settings = settings
        self._embedder = embedder

    def search(self, query: SearchQuery) -> SearchOutcome:
        self._indexes.ensure_ready()

        allowed = self._store.document_ids_matching(query.filters)
        if allowed is not None and not allowed:
            return SearchOutcome()

        results: list[RetrieverResult] = []
        max_bm25: float | None = None
        max_cosine: float | None = None

        if query.mode in ("bm25", "hybrid"):
            ranked = self._indexes.bm25.search(
                query.text, top_k=query.per_retriever_k, allowed_document_ids=allowed
            )
            results.append(RetrieverResult(name="bm25", ranked=ranked))
            max_bm25 = ranked[0][1] if ranked else 0.0

        if query.mode in ("dense", "hybrid"):
            # Raises ModelUnavailableError rather than quietly degrading to lexical-only:
            # a silent fallback would make a hybrid result indistinguishable from a BM25 one.
            if self._embedder is None:
                raise ModelUnavailableError(
                    "dense retrieval was requested but no embedding model is configured"
                )
            vectors = self._indexes.vectors
            ranked = vectors.search(
                self._embedder.embed_query(query.text),
                top_k=query.per_retriever_k,
                allowed_document_ids=allowed,
            )
            results.append(RetrieverResult(name="dense", ranked=ranked))
            max_cosine = ranked[0][1] if ranked else 0.0

        fused = reciprocal_rank_fusion(results, k=query.rrf_k, top_k=query.top_k)
        if not fused:
            return SearchOutcome(max_bm25_score=max_bm25, max_cosine_score=max_cosine)

        chunks = self._store.get_chunks([f.chunk_id for f in fused])
        titles = self._store.chunk_titles()

        hits: list[SearchHit] = []
        rank = 0
        for result in fused:
            chunk = chunks.get(result.chunk_id)
            if chunk is None:
                logger.warning("index referenced missing chunk %s; skipping", result.chunk_id)
                continue
            rank += 1
            hits.append(
                SearchHit(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    title=titles.get(chunk.chunk_id, ""),
                    text=chunk.text,
                    start_offset=chunk.start_offset,
                    end_offset=chunk.end_offset,
                    fused_score=result.fused_score,
                    fused_rank=rank,
                    contributions=result.contributions,
                )
            )
        return SearchOutcome(hits=hits, max_bm25_score=max_bm25, max_cosine_score=max_cosine)
