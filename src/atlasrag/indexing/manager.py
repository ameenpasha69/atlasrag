"""Builds and persists the derived indexes, keeping them consistent with the registry.

Because chunk ids are content-addressed, a chunk that survives an edit keeps its id, so
embeddings can be reused by id across rebuilds. That makes the safe strategy — rebuild from
the store whenever the chunk set changes — cheap enough to always use, which in turn is why
deletion propagates everywhere without any invalidation bookkeeping.
"""

from __future__ import annotations

import logging
import shutil
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from atlasrag.config import Settings
from atlasrag.domain.models import Chunk
from atlasrag.domain.protocols import EmbeddingModel
from atlasrag.errors import IndexCorruptError, IndexIncompatibleError, ModelUnavailableError
from atlasrag.indexing.dense.vector_store import VectorIndex
from atlasrag.indexing.lexical.bm25 import BM25Index
from atlasrag.storage.sqlite_store import SqliteDocumentStore

logger = logging.getLogger(__name__)

_BM25_FILE = "bm25.json"


class IndexManager:
    def __init__(
        self,
        store: SqliteDocumentStore,
        settings: Settings,
        embedder: EmbeddingModel | None = None,
    ) -> None:
        self._store = store
        self._settings = settings
        self._embedder = embedder
        self._lock = threading.RLock()
        self._bm25: BM25Index | None = None
        self._vectors: VectorIndex | None = None
        self._dense_error: str | None = None

    @property
    def index_dir(self) -> Path:
        return self._settings.index_dir

    @property
    def dense_ready(self) -> bool:
        return self._vectors is not None and self._vectors.size > 0

    @property
    def dense_error(self) -> str | None:
        return self._dense_error

    @property
    def bm25(self) -> BM25Index:
        if self._bm25 is None:
            raise IndexCorruptError("lexical index has not been built")
        return self._bm25

    @property
    def vectors(self) -> VectorIndex:
        if self._vectors is None:
            raise ModelUnavailableError(
                self._dense_error or "dense index is unavailable; embedding model not loaded"
            )
        return self._vectors

    def ensure_ready(self) -> None:
        """Load persisted indexes, rebuilding any that disagree with the registry."""
        with self._lock:
            store_chunks = self._store.iter_chunks()
            store_ids = [c.chunk_id for c in store_chunks]

            if self._bm25 is None:
                self._bm25 = self._load_bm25()
            if self._bm25 is None or sorted(self._bm25.chunk_ids) != sorted(store_ids):
                self._rebuild_lexical(store_chunks)

            if self._embedder is None:
                self._dense_error = "no embedding model configured"
                return
            if self._vectors is None:
                self._vectors = self._load_vectors()
            if self._vectors is None or sorted(self._vectors.chunk_ids) != sorted(store_ids):
                self._rebuild_dense(store_chunks)

    def rebuild_all(self) -> None:
        with self._lock:
            self._bm25 = None
            self._vectors = None
            chunks = self._store.iter_chunks()
            self._rebuild_lexical(chunks)
            if self._embedder is not None:
                self._rebuild_dense(chunks)

    def drop_persisted(self) -> None:
        """Delete index files from disk. The registry is untouched."""
        with self._lock:
            if self.index_dir.exists():
                shutil.rmtree(self.index_dir)
            self._bm25 = None
            self._vectors = None

    def _load_bm25(self) -> BM25Index | None:
        path = self.index_dir / _BM25_FILE
        if not path.exists():
            return None
        try:
            return BM25Index.load(path)
        except (IndexCorruptError, IndexIncompatibleError) as exc:
            logger.warning("discarding unusable lexical index: %s", exc)
            return None

    def _load_vectors(self) -> VectorIndex | None:
        if self._embedder is None or not (self.index_dir / "vectors.manifest.json").exists():
            return None
        try:
            return VectorIndex.load(
                self.index_dir,
                expected_model=self._embedder.model_id,
                expected_revision=self._embedder.revision,
            )
        except (IndexCorruptError, IndexIncompatibleError) as exc:
            logger.warning("discarding unusable vector index: %s", exc)
            return None

    def _rebuild_lexical(self, chunks: Sequence[Chunk]) -> None:
        index = BM25Index(k1=self._settings.bm25_k1, b=self._settings.bm25_b)
        index.build(chunks)
        index.save(self.index_dir / _BM25_FILE)
        self._bm25 = index
        logger.info(
            "lexical index rebuilt: %d chunks, %d terms", index.doc_count, index.vocabulary_size
        )

    def _rebuild_dense(self, chunks: Sequence[Chunk]) -> None:
        assert self._embedder is not None
        try:
            dimension = self._embedder.dimension
        except ModelUnavailableError as exc:
            self._dense_error = str(exc)
            self._vectors = None
            logger.warning("dense index unavailable: %s", exc)
            return

        reusable: dict[str, np.ndarray] = {}
        if self._vectors is not None and self._vectors.size:
            reusable = {
                cid: self._vectors.matrix[i] for i, cid in enumerate(self._vectors.chunk_ids)
            }

        chunk_ids = [c.chunk_id for c in chunks]
        missing = [c for c in chunks if c.chunk_id not in reusable]
        if missing:
            try:
                fresh = self._embedder.embed_documents([c.text for c in missing])
            except ModelUnavailableError as exc:
                self._dense_error = str(exc)
                self._vectors = None
                logger.warning("dense index unavailable: %s", exc)
                return
            for chunk, row in zip(missing, fresh, strict=True):
                reusable[chunk.chunk_id] = row

        matrix = (
            np.vstack([reusable[cid] for cid in chunk_ids])
            if chunk_ids
            else np.zeros((0, dimension), dtype=np.float32)
        )
        index = VectorIndex(
            model_id=self._embedder.model_id,
            revision=self._embedder.revision,
            dimension=dimension,
        )
        index.replace(
            chunk_ids,
            {c.chunk_id: c.document_id for c in chunks},
            matrix,
        )
        index.save(self.index_dir)
        self._vectors = index
        self._dense_error = None
        logger.info("dense index rebuilt: %d vectors, dim %d", index.size, dimension)
