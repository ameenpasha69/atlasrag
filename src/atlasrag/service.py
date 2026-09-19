"""Composition root.

Everything above this line is injected; nothing below the domain layer constructs its own
dependencies. Swapping the embedder, the store or the answer provider happens here and
nowhere else.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from atlasrag.answering.abstention import AbstentionThresholds
from atlasrag.answering.evidence import build_evidence
from atlasrag.answering.extractive import ExtractiveAnswerProvider
from atlasrag.answering.llm import HttpChatClient, LlmAnswerProvider
from atlasrag.config import Settings
from atlasrag.domain.models import (
    AnswerResponse,
    Chunk,
    Document,
    DocumentSummary,
    IngestionResult,
    SearchQuery,
)
from atlasrag.domain.protocols import AnswerProvider, EmbeddingModel
from atlasrag.indexing.dense.embedder import BgeEmbedder
from atlasrag.indexing.manager import IndexManager
from atlasrag.ingestion.pipeline import IngestionService
from atlasrag.retrieval.engine import SearchEngine, SearchOutcome
from atlasrag.storage.sqlite_store import SqliteDocumentStore

logger = logging.getLogger(__name__)


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
        self.extractive = ExtractiveAnswerProvider(
            self.store,
            idf=lambda term: self.indexes.bm25.idf(term),
            oov_idf=lambda: self.indexes.bm25.oov_idf,
            thresholds=self.thresholds,
        )
        self.answer_provider_requested = settings.answer_provider
        self.answer_provider_note: str | None = None
        self.answerer: AnswerProvider = self._select_provider()

    def _select_provider(self) -> AnswerProvider:
        """Pick the answer provider, and say so when the pick is not what was asked for.

        A request-time substitution would be a silent fallback. Choosing at construction and
        reporting both the requested and the active provider through /ready keeps the
        no-API-key path usable without ever making a degraded answer look like a normal one.
        """
        if self.settings.answer_provider != "llm":
            return self.extractive

        base_url = self.settings.llm_base_url
        model = self.settings.llm_model
        if not base_url or not model:
            self.answer_provider_note = (
                "answer_provider=llm was requested but ATLASRAG_LLM_BASE_URL and "
                "ATLASRAG_LLM_MODEL are not both set; using the extractive provider"
            )
            logger.warning(self.answer_provider_note)
            return self.extractive

        return LlmAnswerProvider(
            self.store,
            HttpChatClient(
                base_url=base_url,
                model=model,
                api_key=self.settings.llm_api_key,
                timeout_seconds=self.settings.llm_timeout_seconds,
            ),
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
