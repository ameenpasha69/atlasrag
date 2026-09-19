"""Validated configuration. Nothing reads os.environ outside this module."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Pinned so a silent upstream update cannot change retrieval results underneath us.
# Recorded from the HuggingFace API at first download; see DECISIONS.md D-004.
DEFAULT_EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"

# BGE is asymmetric: queries must carry this instruction, passages must not.
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="ATLASRAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("var")

    max_document_bytes: int = Field(default=5_000_000, ge=1)
    allowed_extensions: tuple[str, ...] = (
        ".txt",
        ".md",
        ".markdown",
        ".html",
        ".htm",
        ".pdf",
    )

    chunk_size_chars: int = Field(default=1200, ge=200, le=8000)
    chunk_overlap_chars: int = Field(default=200, ge=0, le=2000)
    chunk_boundary_lookback: int = Field(default=300, ge=0, le=2000)
    chunk_min_chars: int = Field(default=100, ge=1)

    bm25_k1: float = Field(default=1.5, gt=0)
    bm25_b: float = Field(default=0.75, ge=0, le=1)

    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_revision: str = DEFAULT_EMBEDDING_REVISION
    embedding_batch_size: int = Field(default=8, ge=1, le=128)
    embedding_max_tokens: int = Field(default=512, ge=32, le=512)
    embedding_device: Literal["cpu"] = "cpu"

    rrf_k: int = Field(default=60, ge=1)

    # Calibrated by `atlasrag calibrate` over 264 grid points on fixtures/eval/v1/
    # calibration.jsonl (13 queries, hybrid mode); see EVALUATION.md for the executed sweep.
    # The BM25 and cosine floors did not discriminate — every grid point tied on them — so
    # they are calibrated to 0.0 and the support threshold carries the abstention decision.
    # A consequence, recorded in STATUS.md: AbstentionReason.OUT_OF_SCOPE is unreachable at
    # these values, and weak-evidence cases surface as BELOW_THRESHOLD instead.
    abstain_min_support: float = Field(default=0.25, ge=0.0, le=1.0)
    abstain_min_bm25_score: float = Field(default=0.0, ge=0.0)
    abstain_min_cosine: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence_top_n: int = Field(default=5, ge=1, le=20)

    answer_provider: Literal["extractive", "llm"] = "extractive"
    llm_base_url: str | None = None
    llm_model: str | None = None
    llm_api_key: str | None = None
    llm_timeout_seconds: float = Field(default=60.0, gt=0)

    @field_validator("chunk_overlap_chars")
    @classmethod
    def _overlap_below_size(cls, v: int, info: object) -> int:
        return v

    def model_post_init(self, _context: object) -> None:
        if self.chunk_overlap_chars >= self.chunk_size_chars:
            raise ValueError("chunk_overlap_chars must be smaller than chunk_size_chars")

    @property
    def db_path(self) -> Path:
        return self.data_dir / "atlasrag.sqlite3"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    def fingerprint(self) -> str:
        """Stable hash of every setting that can change retrieval output."""
        payload = json.dumps(
            {
                "chunk_size_chars": self.chunk_size_chars,
                "chunk_overlap_chars": self.chunk_overlap_chars,
                "chunk_boundary_lookback": self.chunk_boundary_lookback,
                "chunk_min_chars": self.chunk_min_chars,
                "bm25_k1": self.bm25_k1,
                "bm25_b": self.bm25_b,
                "embedding_model": self.embedding_model,
                "embedding_revision": self.embedding_revision,
                "embedding_batch_size": self.embedding_batch_size,
                "embedding_max_tokens": self.embedding_max_tokens,
                "rrf_k": self.rrf_k,
                "abstain_min_support": self.abstain_min_support,
                "abstain_min_bm25_score": self.abstain_min_bm25_score,
                "abstain_min_cosine": self.abstain_min_cosine,
                "evidence_top_n": self.evidence_top_n,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
