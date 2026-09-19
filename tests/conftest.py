from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from atlasrag.config import Settings
from atlasrag.indexing.lexical.bm25 import tokenize
from atlasrag.service import AtlasRagService

FIXTURE_CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "corpus" / "v1"
EVAL_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "eval" / "v1"


class HashEmbedder:
    """Deterministic stand-in for the real model.

    It is a hashed bag-of-words projection, so it exercises every dense code path — batching,
    dimension checks, persistence, cosine ranking — without downloading weights or spending
    seconds per test. It has no semantic ability whatsoever, which is exactly why the
    paraphrase tests are marked `requires_model` and use the real embedder instead.
    """

    def __init__(self, dimension: int = 64) -> None:
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return "test/hash-embedder"

    @property
    def revision(self) -> str:
        return "0" * 40

    @property
    def dimension(self) -> int:
        return self._dimension

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dimension, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            vector[digest[0] % self._dimension] += 1.0
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector /= norm
        return vector

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dimension), dtype=np.float32)
        return np.vstack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self._vector(text)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "var")


@pytest.fixture
def service(settings: Settings) -> AtlasRagService:
    svc = AtlasRagService(settings, embedder=HashEmbedder())
    yield svc
    svc.close()


@pytest.fixture
def corpus_service(service: AtlasRagService) -> AtlasRagService:
    for path in sorted(FIXTURE_CORPUS.iterdir()):
        if path.is_file():
            service.ingest_path(path)
    service.indexes.ensure_ready()
    return service


@pytest.fixture
def real_service(settings: Settings) -> AtlasRagService:
    """Service backed by the pinned embedding model. Slow; used only where semantics matter."""
    svc = AtlasRagService(settings)
    yield svc
    svc.close()
