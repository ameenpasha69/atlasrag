"""Composition root.

Everything above this line is injected; nothing below the domain layer constructs its own
dependencies. Swapping the embedder, the store or the answer provider happens here and
nowhere else.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from atlasrag.answering.abstention import AbstentionThresholds
from atlasrag.answering.evidence import build_evidence
from atlasrag.answering.extractive import ExtractiveAnswerProvider
from atlasrag.config import Settings
from atlasrag.domain.models import (
    AnswerResponse,
    Chunk,
    Document,
    DocumentSummary,
    IngestionResult,
    SearchQuery,
)
from atlasrag.domain.protocols import EmbeddingModel
from atlasrag.indexing.dense.embedder import BgeEmbedder
from atlasrag.indexing.manager import IndexManager
from atlasrag.ingestion.pipeline import IngestionService
from atlasrag.retrieval.engine import SearchEngine, SearchOutcome
from atlasrag.storage.sqlite_store import SqliteDocumentStore


def build_embedder(settings: Settings) -> EmbeddingModel:
    return BgeEmbedder(
        model_id=settings.embedding_model,
        revision=settings.embedding_revision,
        batch_size=settings.embedding_batch_size,
        max_tokens=settings.embedding_max_tokens,
        device=settings.embedding_device,
    )


class AtlasRagService:
    def __init__(
        self,
        settings: Settings,
        *,
        store: SqliteDocumentStore | None = None,
        embedder: EmbeddingModel | None = None,
    ) -> None:
        self.settings = settings
        self.store = store or SqliteDocumentStore(settings.db_path)
        self.embedder = embedder if embedder is not None else build_embedder(settings)
        self.indexes = IndexManager(self.store, settings, self.embedder)
        self.ingestion = IngestionService(self.store, settings)
        self.engine = SearchEngine(self.store, self.indexes, settings, self.embedder)
        self.thresholds = AbstentionThresholds(
            min_support=settings.abstain_min_support,
            min_bm25=settings.abstain_min_bm25_score,
            min_cosine=settings.abstain_min_cosine,
        )
        self.answerer = ExtractiveAnswerProvider(
            self.store,
            idf=lambda term: self.indexes.bm25.idf(term),
            oov_idf=lambda: self.indexes.bm25.oov_idf,
            thresholds=self.thresholds,
        )

    def ingest_bytes(
        self, raw: bytes, *, filename: str, now: datetime | None = None
    ) -> IngestionResult:
        result = self.ingestion.ingest_bytes(raw, filename=filename, now=now)
        if result.status == "ingested":
            self.indexes.ensure_ready()
        return result

    def ingest_path(self, path: Path, *, now: datetime | None = None) -> IngestionResult:
        result = self.ingestion.ingest_path(path, now=now)
        if result.status == "ingested":
            self.indexes.ensure_ready()
        return result

    def list_documents(self) -> list[DocumentSummary]:
        return self.store.list_documents()

    def get_document(self, document_id: str) -> Document | None:
        return self.store.get_document(document_id)

    def chunks_for_document(self, document_id: str) -> list[Chunk]:
        return self.store.chunks_for_document(document_id)

    def delete_document(self, document_id: str) -> bool:
        deleted = self.store.delete_document(document_id)
        if deleted:
            self.indexes.ensure_ready()
        return deleted

    def search(self, query: SearchQuery) -> SearchOutcome:
        return self.engine.search(query)

    def answer(self, query: SearchQuery) -> AnswerResponse:
        outcome = self.search(query)
        evidence = build_evidence(query.text, outcome, top_n=self.settings.evidence_top_n)
        return self.answerer.answer(query, evidence)

    def close(self) -> None:
        self.store.close()
