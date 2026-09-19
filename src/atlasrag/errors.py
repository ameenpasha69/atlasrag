"""Explicit exception hierarchy. No layer raises bare Exception."""

from __future__ import annotations


class AtlasRagError(Exception):
    """Base class for every error this package raises deliberately."""


class IngestionError(AtlasRagError):
    def __init__(self, message: str, *, code: str) -> None:
        super().__init__(message)
        self.code = code


class UnsupportedMediaTypeError(IngestionError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="unsupported_media_type")


class EmptyDocumentError(IngestionError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="empty_document")


class DocumentTooLargeError(IngestionError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="document_too_large")


class MalformedDocumentError(IngestionError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="malformed_document")


class UnsafePathError(IngestionError):
    def __init__(self, message: str) -> None:
        super().__init__(message, code="unsafe_path")


class StorageError(AtlasRagError):
    pass


class IndexLayerError(AtlasRagError):
    """Index-layer failure. Named to avoid shadowing builtins.IndexError."""


class IndexIncompatibleError(IndexLayerError):
    """Persisted index was built with a different model, dimension or version."""


class IndexCorruptError(IndexLayerError):
    pass


class EmbeddingError(AtlasRagError):
    pass


class ModelUnavailableError(EmbeddingError):
    """Weights are not present locally and could not be fetched."""


class AnswerProviderError(AtlasRagError):
    """A generation backend failed.

    Deliberately distinct from abstention: an abstention is a correct, evidence-driven
    outcome, while this is a malfunction and must never be reported as one.
    """


class CitationValidationError(AtlasRagError):
    pass


class EvaluationError(AtlasRagError):
    pass
