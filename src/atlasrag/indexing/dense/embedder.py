"""Local embedding model wrapper.

Weights are loaded lazily and pinned to an exact revision. Nothing downloads at import
time, so importing the package stays cheap and offline-safe; the first call that actually
needs vectors is the one that pays.

BGE models are asymmetric: the query side carries an instruction prefix and the passage
side does not. Applying that inconsistently silently degrades retrieval, so both paths
live here rather than at the call sites.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from atlasrag.config import BGE_QUERY_INSTRUCTION
from atlasrag.errors import EmbeddingError, ModelUnavailableError


class BgeEmbedder:
    def __init__(
        self,
        *,
        model_id: str,
        revision: str,
        batch_size: int = 8,
        max_tokens: int = 512,
        device: str = "cpu",
        query_instruction: str = BGE_QUERY_INSTRUCTION,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._batch_size = batch_size
        self._max_tokens = max_tokens
        self._device = device
        self._query_instruction = query_instruction
        self._tokenizer: Any = None
        self._model: Any = None
        self._dimension: int | None = None

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str:
        return self._revision

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._ensure_loaded()
        assert self._dimension is not None
        return self._dimension

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise ModelUnavailableError(f"torch/transformers unavailable: {exc}") from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_id, revision=self._revision
            )
            model = AutoModel.from_pretrained(self._model_id, revision=self._revision)
        except Exception as exc:  # noqa: BLE001 - surfaced as a typed error below
            raise ModelUnavailableError(
                f"could not load {self._model_id}@{self._revision[:12]}: {exc}"
            ) from exc

        model.eval()
        model.to(self._device)
        torch.set_grad_enabled(False)
        self._model = model
        self._dimension = int(model.config.hidden_size)

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        self._ensure_loaded()
        import torch

        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            encoded = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_tokens,
                return_tensors="pt",
            ).to(self._device)
            with torch.inference_mode():
                output = self._model(**encoded)
            # BGE pools the [CLS] token, then L2-normalizes so dot product == cosine.
            cls = output.last_hidden_state[:, 0]
            cls = torch.nn.functional.normalize(cls, p=2, dim=1)
            vectors.append(cls.cpu().numpy().astype(np.float32, copy=False))

        if not vectors:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.vstack(vectors)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        matrix = self._encode(texts)
        if matrix.shape[1] != self.dimension:
            raise EmbeddingError(
                f"embedding dimension {matrix.shape[1]} != expected {self.dimension}"
            )
        return matrix

    def embed_query(self, text: str) -> np.ndarray:
        matrix = self._encode([self._query_instruction + text])
        return matrix[0]
