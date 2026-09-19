"""Ingestion orchestration: bytes in, registry rows out."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from atlasrag.chunking.fixed_window import ChunkingConfig, chunk_document
from atlasrag.config import Settings
from atlasrag.domain.ids import (
    INGESTION_VERSION,
    derive_document_id,
    normalize_source_key,
    text_checksum,
)
from atlasrag.domain.models import (
    Document,
    DocumentSource,
    IngestionErrorDetail,
    IngestionResult,
)
from atlasrag.errors import IngestionError
from atlasrag.ingestion.loaders.registry import loader_for
from atlasrag.ingestion.normalize import derive_title
from atlasrag.ingestion.validate import (
    media_type_for,
    safe_filename,
    validate_normalized,
    validate_size,
)
from atlasrag.storage.sqlite_store import SqliteDocumentStore


class IngestionService:
    def __init__(self, store: SqliteDocumentStore, settings: Settings) -> None:
        self._store = store
        self._settings = settings
        self._chunking = ChunkingConfig(
            size_chars=settings.chunk_size_chars,
            overlap_chars=settings.chunk_overlap_chars,
            boundary_lookback=settings.chunk_boundary_lookback,
            min_chars=settings.chunk_min_chars,
        )

    def ingest_bytes(
        self,
        raw: bytes,
        *,
        filename: str,
        uri: str | None = None,
        scheme: str = "upload",
        mtime_utc: datetime | None = None,
        now: datetime | None = None,
    ) -> IngestionResult:
        try:
            clean_name = safe_filename(filename)
            media_type = media_type_for(clean_name, self._settings.allowed_extensions)
            validate_size(len(raw), self._settings.max_document_bytes, clean_name)
            loaded = loader_for(media_type).load(raw, filename=clean_name)
            normalized = loaded.text
            validate_normalized(normalized, clean_name)
        except IngestionError as exc:
            return IngestionResult(
                status="rejected",
                errors=[IngestionErrorDetail(code=exc.code, message=str(exc))],
            )

        content_sha = text_checksum(normalized)
        document_id = derive_document_id(
            source_key=normalize_source_key(clean_name), content_sha256=content_sha
        )

        if self._store.get_document(document_id) is not None:
            return IngestionResult(
                status="unchanged",
                document_id=document_id,
                chunk_count=len(self._store.chunks_for_document(document_id)),
            )

        existing = self._store.get_document_by_content(content_sha)
        if existing is not None:
            return IngestionResult(
                status="duplicate_content",
                document_id=None,
                duplicate_of=existing.document_id,
                errors=[
                    IngestionErrorDetail(
                        code="duplicate_content",
                        message=(
                            f"identical text is already indexed as '{existing.title}' "
                            f"({existing.document_id[:12]}); not indexed again"
                        ),
                    )
                ],
            )

        document = Document(
            document_id=document_id,
            title=loaded.title_hint or derive_title(normalized, clean_name),
            source=DocumentSource(
                uri=uri or clean_name,
                scheme=scheme,  # type: ignore[arg-type]
                media_type=media_type,
                original_filename=clean_name,
                size_bytes=len(raw),
                mtime_utc=mtime_utc,
            ),
            content_sha256=content_sha,
            normalized_text=normalized,
            char_count=len(normalized),
            ingestion_version=INGESTION_VERSION,
            ingested_at_utc=now or datetime.now(UTC),
            locators=loaded.locators,
        )
        chunks = chunk_document(document, self._chunking)
        self._store.upsert_document(document, chunks)
        return IngestionResult(
            status="ingested",
            document_id=document_id,
            title=document.title,
            chunk_count=len(chunks),
        )

    def ingest_path(self, path: Path, *, now: datetime | None = None) -> IngestionResult:
        if not path.is_file():
            return IngestionResult(
                status="rejected",
                errors=[
                    IngestionErrorDetail(code="not_a_file", message=f"{path} is not a regular file")
                ],
            )
        stat = path.stat()
        return self.ingest_bytes(
            path.read_bytes(),
            filename=path.name,
            uri=path.as_posix(),
            scheme="file",
            mtime_utc=datetime.fromtimestamp(stat.st_mtime, tz=UTC),
            now=now,
        )
