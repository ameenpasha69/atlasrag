"""Exact dense index: an L2-normalized float32 matrix plus an id map.

Exact cosine search, not an approximate nearest-neighbour structure. For the corpus sizes
this system targets a single matrix multiply is fast, and — more importantly for a project
whose point is measurement — it has zero recall error, so a missing result is always a
retrieval or chunking problem and never an ANN artifact. See DECISIONS.md D-002.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from atlasrag.errors import IndexCorruptError, IndexIncompatibleError

FORMAT_VERSION = 1
_VECTORS_FILE = "vectors.npy"
_MANIFEST_FILE = "vectors.manifest.json"


class VectorIndex:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        dimension: int,
        chunk_ids: list[str] | None = None,
        chunk_documents: dict[str, str] | None = None,
        matrix: np.ndarray | None = None,
    ) -> None:
        self.model_id = model_id
        self.revision = revision
        self.dimension = dimension
        self.chunk_ids: list[str] = chunk_ids or []
        self.chunk_documents: dict[str, str] = chunk_documents or {}
        self.matrix: np.ndarray = (
            matrix if matrix is not None else np.zeros((0, dimension), dtype=np.float32)
        )

    @property
    def size(self) -> int:
        return len(self.chunk_ids)

    def replace(
        self, chunk_ids: list[str], chunk_documents: dict[str, str], matrix: np.ndarray
    ) -> None:
        if matrix.shape[0] != len(chunk_ids):
            raise IndexCorruptError(
                f"vector rows {matrix.shape[0]} != chunk id count {len(chunk_ids)}"
            )
        if matrix.size and matrix.shape[1] != self.dimension:
            raise IndexIncompatibleError(
                f"vector dimension {matrix.shape[1]} != expected {self.dimension}"
            )
        self.chunk_ids = chunk_ids
        self.chunk_documents = chunk_documents
        self.matrix = matrix.astype(np.float32, copy=False)

    def search(
        self,
        query_vector: np.ndarray,
        *,
        top_k: int,
        allowed_document_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        if self.size == 0 or self.matrix.size == 0:
            return []
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.matrix.shape[1]:
            raise IndexIncompatibleError(
                f"query dimension {query.shape[0]} != index dimension {self.matrix.shape[1]}"
            )
        scores = self.matrix @ query

        candidates = range(self.size)
        if allowed_document_ids is not None:
            candidates = [  # type: ignore[assignment]
                i
                for i in range(self.size)
                if self.chunk_documents.get(self.chunk_ids[i]) in allowed_document_ids
            ]
            if not candidates:
                return []

        # Tie-break on chunk_id so equal cosines never reorder between runs.
        ranked = sorted(
            ((self.chunk_ids[i], float(scores[i])) for i in candidates),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return ranked[:top_k]

    def manifest(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "model_id": self.model_id,
            "revision": self.revision,
            "dimension": self.dimension,
            "count": self.size,
            "chunk_ids": self.chunk_ids,
            "chunk_documents": self.chunk_documents,
        }

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        vectors_path = directory / _VECTORS_FILE
        manifest_path = directory / _MANIFEST_FILE

        tmp_vectors = vectors_path.with_suffix(".npy.tmp")
        with tmp_vectors.open("wb") as handle:
            np.save(handle, self.matrix, allow_pickle=False)
        tmp_vectors.replace(vectors_path)

        tmp_manifest = manifest_path.with_suffix(".json.tmp")
        tmp_manifest.write_text(
            json.dumps(self.manifest(), sort_keys=True, ensure_ascii=False), encoding="utf-8"
        )
        tmp_manifest.replace(manifest_path)

    @classmethod
    def load(cls, directory: Path, *, expected_model: str, expected_revision: str) -> VectorIndex:
        manifest_path = directory / _MANIFEST_FILE
        vectors_path = directory / _VECTORS_FILE
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexCorruptError(f"cannot read vector manifest at {manifest_path}: {exc}") from exc

        if manifest.get("format_version") != FORMAT_VERSION:
            raise IndexIncompatibleError(
                f"vector index format {manifest.get('format_version')} != {FORMAT_VERSION}"
            )
        if manifest.get("model_id") != expected_model or manifest.get("revision") != expected_revision:
            raise IndexIncompatibleError(
                f"vector index was built with {manifest.get('model_id')}@"
                f"{str(manifest.get('revision'))[:12]} but configuration expects "
                f"{expected_model}@{expected_revision[:12]}; rebuild the index"
            )

        try:
            with vectors_path.open("rb") as handle:
                matrix = np.load(handle, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise IndexCorruptError(f"cannot read vectors at {vectors_path}: {exc}") from exc

        chunk_ids = list(manifest["chunk_ids"])
        if matrix.shape[0] != len(chunk_ids):
            raise IndexCorruptError(
                f"vector rows {matrix.shape[0]} != manifest chunk ids {len(chunk_ids)}"
            )
        index = cls(
            model_id=manifest["model_id"],
            revision=manifest["revision"],
            dimension=int(manifest["dimension"]),
        )
        index.chunk_ids = chunk_ids
        index.chunk_documents = dict(manifest["chunk_documents"])
        index.matrix = matrix.astype(np.float32, copy=False)
        return index
