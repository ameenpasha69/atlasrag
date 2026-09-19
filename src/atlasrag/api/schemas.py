"""HTTP request and response schemas.

Deliberately separate from the domain models. The wire format can carry presentation concerns
(such as flagging instruction-like text for the UI) without those leaking into the retrieval
core, and the domain can change shape without silently breaking clients.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from atlasrag.answering.injection import looks_like_injected_instruction
from atlasrag.domain.models import (
    AnswerResponse,
    Chunk,
    Citation,
    Document,
    DocumentSummary,
    RankContribution,
    SearchFilters,
    SearchHit,
    SearchMode,
)


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    ready: bool
    documents: int
    chunks: int
    lexical_chunks: int
    lexical_terms: int
    dense_ready: bool
    dense_vectors: int | None = None
    dense_dimension: int | None = None
    dense_error: str | None = None
    embedding_model: str
    embedding_revision: str
    config_fingerprint: str
    answer_provider: str
    answer_provider_requested: str
    answer_provider_note: str | None = None


class IngestionErrorOut(BaseModel):
    code: str
    message: str


class IngestionResultOut(BaseModel):
    filename: str
    status: str
    document_id: str | None = None
    title: str | None = None
    chunk_count: int = 0
    duplicate_of: str | None = None
    errors: list[IngestionErrorOut] = Field(default_factory=list)


class IngestResponse(BaseModel):
    results: list[IngestionResultOut]
    documents: int
    chunks: int


class DocumentOut(BaseModel):
    document_id: str
    title: str
    original_filename: str
    media_type: str
    content_sha256: str
    char_count: int
    chunk_count: int
    ingested_at_utc: datetime

    @classmethod
    def from_summary(cls, summary: DocumentSummary) -> DocumentOut:
        return cls(**summary.model_dump())


class ChunkOut(BaseModel):
    chunk_id: str
    chunk_index: int
    start_offset: int
    end_offset: int
    text: str
    contains_instruction_like_text: bool

    @classmethod
    def from_chunk(cls, chunk: Chunk) -> ChunkOut:
        return cls(
            chunk_id=chunk.chunk_id,
            chunk_index=chunk.chunk_index,
            start_offset=chunk.start_offset,
            end_offset=chunk.end_offset,
            text=chunk.text,
            contains_instruction_like_text=looks_like_injected_instruction(chunk.text),
        )


class DocumentDetailOut(BaseModel):
    document: DocumentOut
    normalized_text: str
    chunks: list[ChunkOut]

    @classmethod
    def build(cls, document: Document, chunks: list[Chunk], chunk_count: int) -> DocumentDetailOut:
        return cls(
            document=DocumentOut(
                document_id=document.document_id,
                title=document.title,
                original_filename=document.source.original_filename,
                media_type=document.source.media_type,
                content_sha256=document.content_sha256,
                char_count=document.char_count,
                chunk_count=chunk_count,
                ingested_at_utc=document.ingested_at_utc,
            ),
            normalized_text=document.normalized_text,
            chunks=[ChunkOut.from_chunk(c) for c in chunks],
        )


class SearchRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)
    mode: SearchMode = "hybrid"
    top_k: int = Field(default=10, ge=1, le=50)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    filters: SearchFilters | None = None


class ContributionOut(BaseModel):
    retriever: str
    rank: int
    raw_score: float
    rrf_term: float
    weight: float

    @classmethod
    def from_contribution(cls, contribution: RankContribution) -> ContributionOut:
        return cls(**contribution.model_dump())


class SearchHitOut(BaseModel):
    chunk_id: str
    document_id: str
    title: str
    text: str
    start_offset: int
    end_offset: int
    fused_score: float
    fused_rank: int
    retrieved_by: list[str]
    contributions: list[ContributionOut]
    contains_instruction_like_text: bool

    @classmethod
    def from_hit(cls, hit: SearchHit) -> SearchHitOut:
        return cls(
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            title=hit.title,
            text=hit.text,
            start_offset=hit.start_offset,
            end_offset=hit.end_offset,
            fused_score=hit.fused_score,
            fused_rank=hit.fused_rank,
            retrieved_by=hit.retrieved_by,
            contributions=[ContributionOut.from_contribution(c) for c in hit.contributions],
            contains_instruction_like_text=looks_like_injected_instruction(hit.text),
        )


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    rrf_k: int
    hit_count: int
    max_bm25_score: float | None
    max_cosine_score: float | None
    hits: list[SearchHitOut]


class CitationOut(BaseModel):
    document_id: str
    chunk_id: str
    title: str
    start_offset: int
    end_offset: int
    snippet: str
    validated: bool

    @classmethod
    def from_citation(cls, citation: Citation) -> CitationOut:
        return cls(
            document_id=citation.document_id,
            chunk_id=citation.chunk_id,
            title=citation.title,
            start_offset=citation.start_offset,
            end_offset=citation.end_offset,
            snippet=citation.snippet,
            validated=citation.validated,
        )


class AbstentionOut(BaseModel):
    reason: str
    explanation: str
    evidence_considered: int


class AnswerRequest(SearchRequest):
    pass


class AnswerResponseOut(BaseModel):
    answered: bool
    text: str | None
    provider: str
    mode: SearchMode
    citations: list[CitationOut]
    abstention: AbstentionOut | None
    conflict_detected: bool
    conflict_note: str | None
    evidence: list[SearchHitOut]

    @classmethod
    def from_response(cls, response: AnswerResponse) -> AnswerResponseOut:
        return cls(
            answered=response.answered,
            text=response.text,
            provider=response.provider,
            mode=response.mode,
            citations=[CitationOut.from_citation(c) for c in response.citations],
            abstention=(
                AbstentionOut(
                    reason=response.abstention.reason.value,
                    explanation=response.abstention.explanation,
                    evidence_considered=response.abstention.evidence_considered,
                )
                if response.abstention
                else None
            ),
            conflict_detected=response.evidence.conflict_detected,
            conflict_note=response.evidence.conflict_note,
            evidence=[SearchHitOut.from_hit(h) for h in response.evidence.selected],
        )


class ReindexResponse(BaseModel):
    lexical_chunks: int
    lexical_terms: int
    dense_vectors: int | None
    dense_dimension: int | None


class ErrorResponse(BaseModel):
    error: str
    detail: str
