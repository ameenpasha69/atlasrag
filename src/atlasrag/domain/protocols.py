"""Interfaces the domain depends on. Concrete adapters are injected at the composition root."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from atlasrag.domain.models import (
    AnswerResponse,
    Chunk,
    Document,
    DocumentSummary,
    EvidenceSet,
    RankContribution,
    SearchFilters,
    SearchQuery,
)


@runtime_checkable
class DocumentStore(Protocol):
    def upsert_document(self, document: Document, chunks: Sequence[Chunk]) -> None: ...

    def get_document(self, document_id: str) -> Document | None: ...

    def get_document_by_content(self, content_sha256: str) -> Document | None: ...

    def list_documents(self) -> list[DocumentSummary]: ...

    def delete_document(self, document_id: str) -> bool: ...

    def get_chunk(self, chunk_id: str) -> Chunk | None: ...

    def iter_chunks(self) -> list[Chunk]: ...

    def chunks_for_document(self, document_id: str) -> list[Chunk]: ...

    def document_ids_matching(self, filters: SearchFilters | None) -> set[str] | None: ...


@runtime_checkable
class EmbeddingModel(Protocol):
    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str: ...

    @property
    def dimension(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


@runtime_checkable
class Retriever(Protocol):
    @property
    def name(self) -> str: ...

    def retrieve(
        self, query: str, *, top_k: int, allowed_document_ids: set[str] | None
    ) -> list[tuple[str, float]]:
        """Return (chunk_id, raw_score) ordered best-first."""
        ...


@runtime_checkable
class AnswerProvider(Protocol):
    @property
    def name(self) -> str: ...

    def answer(self, query: SearchQuery, evidence: EvidenceSet) -> AnswerResponse: ...


__all__ = [
    "AnswerProvider",
    "DocumentStore",
    "EmbeddingModel",
    "RankContribution",
    "Retriever",
]
