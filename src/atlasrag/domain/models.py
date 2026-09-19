"""Typed core contracts.

These models are the vocabulary of the whole system. They import nothing that touches
disk, network or torch, so the domain layer stays testable in isolation.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SearchMode = Literal["bm25", "dense", "hybrid"]
RetrieverName = Literal["bm25", "dense"]

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class DocumentSource(BaseModel):
    model_config = _FROZEN

    uri: str
    scheme: Literal["file", "upload", "inline"] = "file"
    media_type: str
    original_filename: str
    size_bytes: int = Field(ge=0)
    mtime_utc: datetime | None = None


class Document(BaseModel):
    model_config = _FROZEN

    document_id: str
    title: str
    source: DocumentSource
    content_sha256: str
    normalized_text: str
    char_count: int = Field(ge=0)
    ingestion_version: str
    ingested_at_utc: datetime

    @model_validator(mode="after")
    def _char_count_matches_text(self) -> Document:
        if self.char_count != len(self.normalized_text):
            raise ValueError("char_count does not match normalized_text length")
        return self


class DocumentSummary(BaseModel):
    """Document without its full text, for listings and API responses."""

    model_config = _FROZEN

    document_id: str
    title: str
    original_filename: str
    media_type: str
    content_sha256: str
    char_count: int
    chunk_count: int
    ingested_at_utc: datetime


class Chunk(BaseModel):
    model_config = _FROZEN

    chunk_id: str
    document_id: str
    chunk_index: int = Field(ge=0)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)
    text: str
    content_sha256: str
    chunking_version: str

    @model_validator(mode="after")
    def _offsets_consistent(self) -> Chunk:
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offset must be greater than start_offset")
        if self.end_offset - self.start_offset != len(self.text):
            raise ValueError("offset span does not match text length")
        return self


class IngestionErrorDetail(BaseModel):
    model_config = _FROZEN

    code: str
    message: str


IngestionStatus = Literal["ingested", "unchanged", "duplicate_content", "rejected"]


class IngestionResult(BaseModel):
    model_config = _FROZEN

    status: IngestionStatus
    document_id: str | None = None
    title: str | None = None
    chunk_count: int = 0
    duplicate_of: str | None = None
    errors: list[IngestionErrorDetail] = Field(default_factory=list)


class SearchFilters(BaseModel):
    model_config = _FROZEN

    document_ids: list[str] | None = None
    title_contains: str | None = None
    media_types: list[str] | None = None
    ingested_after: datetime | None = None
    ingested_before: datetime | None = None

    def is_empty(self) -> bool:
        return not any(
            (
                self.document_ids,
                self.title_contains,
                self.media_types,
                self.ingested_after,
                self.ingested_before,
            )
        )


class SearchQuery(BaseModel):
    model_config = _FROZEN

    text: str = Field(min_length=1)
    mode: SearchMode = "hybrid"
    top_k: int = Field(default=10, ge=1, le=100)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    per_retriever_k: int = Field(default=50, ge=1, le=500)
    filters: SearchFilters | None = None


class RankContribution(BaseModel):
    """How one retriever contributed to one fused result.

    `raw_score` lives only here and is never summed across retrievers: a BM25 score and a
    cosine similarity are different units, and fusion deliberately uses rank alone.
    """

    model_config = _FROZEN

    retriever: RetrieverName
    rank: int = Field(ge=1)
    raw_score: float
    rrf_term: float
    weight: float = 1.0


class SearchHit(BaseModel):
    model_config = _FROZEN

    chunk_id: str
    document_id: str
    title: str
    text: str
    start_offset: int
    end_offset: int
    fused_score: float
    fused_rank: int = Field(ge=1)
    contributions: list[RankContribution]

    @property
    def retrieved_by(self) -> list[str]:
        return [c.retriever for c in self.contributions]


class Citation(BaseModel):
    model_config = _FROZEN

    document_id: str
    chunk_id: str
    title: str
    start_offset: int
    end_offset: int
    snippet: str
    validated: bool
    retrieval_provenance: list[RankContribution] = Field(default_factory=list)


class EvidenceSet(BaseModel):
    model_config = _FROZEN

    query: str
    hits: list[SearchHit]
    selected: list[SearchHit]
    max_fused_score: float = 0.0
    max_bm25_score: float | None = None
    max_cosine_score: float | None = None
    conflict_detected: bool = False
    conflict_note: str | None = None


class AbstentionReason(StrEnum):
    NO_EVIDENCE = "no_evidence"
    BELOW_THRESHOLD = "below_threshold"
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    OUT_OF_SCOPE = "out_of_scope"
    CITATION_VALIDATION_FAILED = "citation_validation_failed"


class Abstention(BaseModel):
    model_config = _FROZEN

    reason: AbstentionReason
    explanation: str
    evidence_considered: int = 0


class AnswerResponse(BaseModel):
    model_config = _FROZEN

    answered: bool
    text: str | None
    citations: list[Citation] = Field(default_factory=list)
    abstention: Abstention | None = None
    evidence: EvidenceSet
    provider: str
    mode: SearchMode
    unsupported_sentences: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _answer_shape_is_coherent(self) -> AnswerResponse:
        if self.answered:
            if not self.text:
                raise ValueError("an answered response must carry text")
            if not self.citations:
                raise ValueError("an answered response must carry at least one citation")
            if self.abstention is not None:
                raise ValueError("an answered response must not carry an abstention")
        else:
            if self.abstention is None:
                raise ValueError("an unanswered response must carry an abstention")
            if self.text is not None:
                raise ValueError("an abstaining response must not carry answer text")
        return self


EvaluationCategory = Literal[
    "exact_keyword",
    "rare_identifier",
    "semantic_paraphrase",
    "multi_concept",
    "metadata_filtered",
    "ambiguous",
    "unanswerable",
    "contradictory",
    "adversarial",
]


class EvaluationQuery(BaseModel):
    model_config = _FROZEN

    query_id: str
    text: str
    category: EvaluationCategory
    expected_answerable: bool
    filters: SearchFilters | None = None
    note: str | None = None


class RelevanceJudgment(BaseModel):
    model_config = _FROZEN

    query_id: str
    document_id: str
    chunk_id: str | None = None
    grade: int = Field(ge=0, le=3)
    rationale: str


class MetricBundle(BaseModel):
    model_config = _FROZEN

    query_count: int
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int


class FailureRecord(BaseModel):
    model_config = _FROZEN

    query_id: str
    category: EvaluationCategory
    query_text: str
    expected_document_ids: list[str]
    retrieved_document_ids: list[str]
    likely_cause: str


class EvaluationRun(BaseModel):
    model_config = _FROZEN

    run_id: str
    started_at_utc: datetime
    dataset_version: str
    mode: SearchMode
    config_fingerprint: str
    overall: MetricBundle
    per_category: dict[str, MetricBundle]
    abstention_precision: float | None = None
    abstention_recall: float | None = None
    citation_validity_rate: float | None = None
    unsupported_claim_rate: float | None = None
    failures: list[FailureRecord] = Field(default_factory=list)
