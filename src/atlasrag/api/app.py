"""FastAPI application.

The service is constructed once at startup and injected into routes. Routes contain no
retrieval logic — they translate HTTP to the service layer and back, so the CLI, the tests and
the API all exercise the same code path.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from atlasrag import __version__
from atlasrag.api.schemas import (
    AnswerRequest,
    AnswerResponseOut,
    DocumentDetailOut,
    DocumentOut,
    ErrorResponse,
    HealthResponse,
    IngestionErrorOut,
    IngestionResultOut,
    IngestResponse,
    ReadyResponse,
    ReindexResponse,
    SearchHitOut,
    SearchRequest,
    SearchResponse,
)
from atlasrag.config import Settings
from atlasrag.domain.models import SearchQuery
from atlasrag.errors import AtlasRagError, ModelUnavailableError
from atlasrag.service import AtlasRagService

logger = logging.getLogger(__name__)

UI_DIR = Path(__file__).resolve().parent.parent / "ui"


def get_service(request: Request) -> AtlasRagService:
    service: AtlasRagService = request.app.state.service
    return service


ServiceDep = Annotated[AtlasRagService, Depends(get_service)]


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        service = AtlasRagService(resolved)
        app.state.service = service
        try:
            service.indexes.ensure_ready()
        except ModelUnavailableError as exc:
            # Startup must not fail because weights are missing: lexical search still works and
            # /ready reports the dense channel as unavailable with the reason.
            logger.warning("dense index unavailable at startup: %s", exc)
        yield
        service.close()

    app = FastAPI(
        title="AtlasRAG",
        version=__version__,
        description=(
            "Hybrid lexical + dense retrieval with Reciprocal Rank Fusion, span-verified "
            "citations and mandatory abstention."
        ),
        lifespan=lifespan,
    )

    @app.exception_handler(ModelUnavailableError)
    async def _model_unavailable(_r: Request, exc: ModelUnavailableError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content=ErrorResponse(error="model_unavailable", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(AtlasRagError)
    async def _atlas_error(_r: Request, exc: AtlasRagError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(error=type(exc).__name__, detail=str(exc)).model_dump(),
        )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    def health() -> HealthResponse:
        """Liveness only: is the process running. Says nothing about the indexes."""
        return HealthResponse(status="ok", version=__version__)

    @app.get("/ready", response_model=ReadyResponse, tags=["ops"])
    def ready(service: ServiceDep) -> ReadyResponse:
        """Readiness: are the indexes actually usable, and if not, why not."""
        service.indexes.ensure_ready()
        documents, chunks = service.store.counts()
        dense_ready = service.indexes.dense_ready
        return ReadyResponse(
            ready=service.indexes.bm25.doc_count == chunks,
            documents=documents,
            chunks=chunks,
            lexical_chunks=service.indexes.bm25.doc_count,
            lexical_terms=service.indexes.bm25.vocabulary_size,
            dense_ready=dense_ready,
            dense_vectors=service.indexes.vectors.size if dense_ready else None,
            dense_dimension=service.indexes.vectors.dimension if dense_ready else None,
            dense_error=service.indexes.dense_error,
            embedding_model=service.settings.embedding_model,
            embedding_revision=service.settings.embedding_revision,
            config_fingerprint=service.settings.fingerprint(),
            answer_provider=service.answerer.name,
            answer_provider_requested=service.answer_provider_requested,
            answer_provider_note=service.answer_provider_note,
        )

    @app.post("/documents", response_model=IngestResponse, tags=["documents"])
    async def ingest(
        service: ServiceDep, files: Annotated[list[UploadFile], File()]
    ) -> IngestResponse:
        results: list[IngestionResultOut] = []
        for upload in files:
            raw = await upload.read()
            outcome = service.ingest_bytes(raw, filename=upload.filename or "unnamed")
            results.append(
                IngestionResultOut(
                    filename=upload.filename or "unnamed",
                    status=outcome.status,
                    document_id=outcome.document_id,
                    title=outcome.title,
                    chunk_count=outcome.chunk_count,
                    duplicate_of=outcome.duplicate_of,
                    errors=[
                        IngestionErrorOut(code=e.code, message=e.message) for e in outcome.errors
                    ],
                )
            )
        documents, chunks = service.store.counts()
        return IngestResponse(results=results, documents=documents, chunks=chunks)

    @app.get("/documents", response_model=list[DocumentOut], tags=["documents"])
    def list_documents(service: ServiceDep) -> list[DocumentOut]:
        return [DocumentOut.from_summary(d) for d in service.list_documents()]

    @app.get("/documents/{document_id}", response_model=DocumentDetailOut, tags=["documents"])
    def get_document(service: ServiceDep, document_id: str) -> DocumentDetailOut:
        document = service.get_document(document_id)
        if document is None:
            raise HTTPException(status_code=404, detail="document not found")
        chunks = service.chunks_for_document(document_id)
        return DocumentDetailOut.build(document, chunks, len(chunks))

    @app.delete("/documents/{document_id}", status_code=204, tags=["documents"])
    def delete_document(service: ServiceDep, document_id: str) -> None:
        if not service.delete_document(document_id):
            raise HTTPException(status_code=404, detail="document not found")

    @app.post("/search", response_model=SearchResponse, tags=["retrieval"])
    def search(service: ServiceDep, request: SearchRequest) -> SearchResponse:
        query = SearchQuery(
            text=request.text,
            mode=request.mode,
            top_k=request.top_k,
            rrf_k=request.rrf_k,
            filters=request.filters,
        )
        outcome = service.search(query)
        return SearchResponse(
            query=request.text,
            mode=request.mode,
            rrf_k=request.rrf_k,
            hit_count=len(outcome.hits),
            max_bm25_score=outcome.max_bm25_score,
            max_cosine_score=outcome.max_cosine_score,
            hits=[SearchHitOut.from_hit(h) for h in outcome.hits],
        )

    @app.post("/answer", response_model=AnswerResponseOut, tags=["answering"])
    def answer(service: ServiceDep, request: AnswerRequest) -> AnswerResponseOut:
        query = SearchQuery(
            text=request.text,
            mode=request.mode,
            top_k=request.top_k,
            rrf_k=request.rrf_k,
            filters=request.filters,
        )
        return AnswerResponseOut.from_response(service.answer(query))

    @app.post("/reindex", response_model=ReindexResponse, tags=["ops"])
    def reindex(service: ServiceDep) -> ReindexResponse:
        """Drop every derived index and rebuild from the registry."""
        service.indexes.drop_persisted()
        service.indexes.rebuild_all()
        dense_ready = service.indexes.dense_ready
        return ReindexResponse(
            lexical_chunks=service.indexes.bm25.doc_count,
            lexical_terms=service.indexes.bm25.vocabulary_size,
            dense_vectors=service.indexes.vectors.size if dense_ready else None,
            dense_dimension=service.indexes.vectors.dimension if dense_ready else None,
        )

    @app.get("/", response_class=HTMLResponse, include_in_schema=False)
    def index() -> HTMLResponse:
        return HTMLResponse((UI_DIR / "templates" / "index.html").read_text(encoding="utf-8"))

    app.mount("/static", StaticFiles(directory=UI_DIR / "static"), name="static")
    return app


app = create_app()
