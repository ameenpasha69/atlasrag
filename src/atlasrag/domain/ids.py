"""Deterministic identifier and checksum derivation.

Every identifier in AtlasRAG is a pure function of content plus explicit version tags.
Re-ingesting identical bytes under identical configuration must reproduce identical ids,
and changing a version tag must change every id it touches.
"""

from __future__ import annotations

import hashlib
import unicodedata

INGESTION_VERSION = "1"
CHUNKING_VERSION = "1"

_DOCUMENT_ID_NAMESPACE = "atlasrag.document.v1"
_CHUNK_ID_NAMESPACE = "atlasrag.chunk.v1"
_FIELD_SEPARATOR = "\x00"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def text_checksum(text: str) -> str:
    return sha256_hex(text.encode("utf-8"))


def normalize_source_key(original_filename: str) -> str:
    """Reduce a filename to a stable identity component.

    Case-folded because Windows filesystems are case-insensitive: `Notes.md` and
    `notes.md` are the same file there, and an id scheme that disagrees with the
    filesystem would report a phantom second document.
    """
    cleaned = unicodedata.normalize("NFC", original_filename.strip())
    cleaned = cleaned.replace("\\", "/").rsplit("/", 1)[-1]
    return cleaned.casefold()


def _digest(namespace: str, parts: tuple[str, ...]) -> str:
    payload = _FIELD_SEPARATOR.join((namespace, *parts))
    return sha256_hex(payload.encode("utf-8"))


def derive_document_id(
    *, source_key: str, content_sha256: str, ingestion_version: str = INGESTION_VERSION
) -> str:
    return _digest(_DOCUMENT_ID_NAMESPACE, (ingestion_version, source_key, content_sha256))


def derive_chunk_id(
    *,
    document_id: str,
    chunk_index: int,
    start_offset: int,
    end_offset: int,
    content_sha256: str,
    chunking_version: str = CHUNKING_VERSION,
) -> str:
    return _digest(
        _CHUNK_ID_NAMESPACE,
        (
            chunking_version,
            document_id,
            str(chunk_index),
            str(start_offset),
            str(end_offset),
            content_sha256,
        ),
    )
