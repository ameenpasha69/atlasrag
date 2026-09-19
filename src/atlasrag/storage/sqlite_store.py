"""SQLite document registry.

This is the single source of truth. The BM25 postings and the dense vector matrix are
derived artifacts that can be deleted and rebuilt from these tables, which is what makes
"delete removes the document from every index" and "survives a clean restart" structural
properties rather than synchronization chores.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from atlasrag.domain.models import (
    Chunk,
    Document,
    DocumentSource,
    DocumentSummary,
    SearchFilters,
)
from atlasrag.errors import StorageError

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    document_id       TEXT PRIMARY KEY,
    title             TEXT NOT NULL,
    uri               TEXT NOT NULL,
    scheme            TEXT NOT NULL,
    media_type        TEXT NOT NULL,
    original_filename TEXT NOT NULL,
    size_bytes        INTEGER NOT NULL,
    mtime_utc         TEXT,
    content_sha256    TEXT NOT NULL,
    normalized_text   TEXT NOT NULL,
    char_count        INTEGER NOT NULL,
    ingestion_version TEXT NOT NULL,
    ingested_at_utc   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_content ON documents(content_sha256);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id         TEXT PRIMARY KEY,
    document_id      TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
    chunk_index      INTEGER NOT NULL,
    start_offset     INTEGER NOT NULL,
    end_offset       INTEGER NOT NULL,
    text             TEXT NOT NULL,
    content_sha256   TEXT NOT NULL,
    chunking_version TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
"""


def _to_iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _from_iso(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


class SqliteDocumentStore:
    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA synchronous=FULL")
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR IGNORE INTO meta(key, value) VALUES ('schema_version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()
        self._assert_schema_version()

    def _assert_schema_version(self) -> None:
        row = self._conn.execute("SELECT value FROM meta WHERE key='schema_version'").fetchone()
        if row is None or int(row["value"]) != SCHEMA_VERSION:
            found = row["value"] if row else "missing"
            raise StorageError(
                f"database schema version {found} does not match expected {SCHEMA_VERSION}"
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> SqliteDocumentStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def upsert_document(self, document: Document, chunks: Sequence[Chunk]) -> None:
        """Replace a document and its chunks atomically."""
        with self._lock, self._conn:
            self._conn.execute("DELETE FROM chunks WHERE document_id = ?", (document.document_id,))
            self._conn.execute(
                """
                INSERT INTO documents (
                    document_id, title, uri, scheme, media_type, original_filename,
                    size_bytes, mtime_utc, content_sha256, normalized_text, char_count,
                    ingestion_version, ingested_at_utc
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(document_id) DO UPDATE SET
                    title=excluded.title,
                    uri=excluded.uri,
                    scheme=excluded.scheme,
                    media_type=excluded.media_type,
                    original_filename=excluded.original_filename,
                    size_bytes=excluded.size_bytes,
                    mtime_utc=excluded.mtime_utc,
                    content_sha256=excluded.content_sha256,
                    normalized_text=excluded.normalized_text,
                    char_count=excluded.char_count,
                    ingestion_version=excluded.ingestion_version,
                    ingested_at_utc=excluded.ingested_at_utc
                """,
                (
                    document.document_id,
                    document.title,
                    document.source.uri,
                    document.source.scheme,
                    document.source.media_type,
                    document.source.original_filename,
                    document.source.size_bytes,
                    _to_iso(document.source.mtime_utc),
                    document.content_sha256,
                    document.normalized_text,
                    document.char_count,
                    document.ingestion_version,
                    _to_iso(document.ingested_at_utc),
                ),
            )
            self._conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, chunk_index, start_offset, end_offset,
                    text, content_sha256, chunking_version
                ) VALUES (?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        c.chunk_id,
                        c.document_id,
                        c.chunk_index,
                        c.start_offset,
                        c.end_offset,
                        c.text,
                        c.content_sha256,
                        c.chunking_version,
                    )
                    for c in chunks
                ],
            )

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> Document:
        return Document(
            document_id=row["document_id"],
            title=row["title"],
            source=DocumentSource(
                uri=row["uri"],
                scheme=row["scheme"],
                media_type=row["media_type"],
                original_filename=row["original_filename"],
                size_bytes=row["size_bytes"],
                mtime_utc=_from_iso(row["mtime_utc"]),
            ),
            content_sha256=row["content_sha256"],
            normalized_text=row["normalized_text"],
            char_count=row["char_count"],
            ingestion_version=row["ingestion_version"],
            ingested_at_utc=_from_iso(row["ingested_at_utc"]),  # type: ignore[arg-type]
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            start_offset=row["start_offset"],
            end_offset=row["end_offset"],
            text=row["text"],
            content_sha256=row["content_sha256"],
            chunking_version=row["chunking_version"],
        )

    def get_document(self, document_id: str) -> Document | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        return self._row_to_document(row) if row else None

    def get_document_by_content(self, content_sha256: str) -> Document | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE content_sha256 = ? ORDER BY document_id LIMIT 1",
                (content_sha256,),
            ).fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(self) -> list[DocumentSummary]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT d.document_id, d.title, d.original_filename, d.media_type,
                       d.content_sha256, d.char_count, d.ingested_at_utc,
                       COUNT(c.chunk_id) AS chunk_count
                FROM documents d
                LEFT JOIN chunks c ON c.document_id = d.document_id
                GROUP BY d.document_id
                ORDER BY d.ingested_at_utc DESC, d.document_id ASC
                """
            ).fetchall()
        return [
            DocumentSummary(
                document_id=r["document_id"],
                title=r["title"],
                original_filename=r["original_filename"],
                media_type=r["media_type"],
                content_sha256=r["content_sha256"],
                char_count=r["char_count"],
                chunk_count=r["chunk_count"],
                ingested_at_utc=_from_iso(r["ingested_at_utc"]),  # type: ignore[arg-type]
            )
            for r in rows
        ]

    def delete_document(self, document_id: str) -> bool:
        with self._lock, self._conn:
            cursor = self._conn.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )
            return cursor.rowcount > 0

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        return self._row_to_chunk(row) if row else None

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",  # noqa: S608
                tuple(chunk_ids),
            ).fetchall()
        return {r["chunk_id"]: self._row_to_chunk(r) for r in rows}

    def iter_chunks(self) -> list[Chunk]:
        """Every chunk in a stable order. Index builds depend on this ordering."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks ORDER BY document_id ASC, chunk_index ASC"
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def chunks_for_document(self, document_id: str) -> list[Chunk]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index ASC",
                (document_id,),
            ).fetchall()
        return [self._row_to_chunk(r) for r in rows]

    def chunk_titles(self) -> dict[str, str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT c.chunk_id, d.title FROM chunks c JOIN documents d"
                " ON d.document_id = c.document_id"
            ).fetchall()
        return {r["chunk_id"]: r["title"] for r in rows}

    def document_ids_matching(self, filters: SearchFilters | None) -> set[str] | None:
        """Return allowed document ids, or None when no filter constrains the search."""
        if filters is None or filters.is_empty():
            return None
        clauses: list[str] = []
        params: list[Any] = []
        if filters.document_ids:
            placeholders = ",".join("?" * len(filters.document_ids))
            clauses.append(f"document_id IN ({placeholders})")
            params.extend(filters.document_ids)
        if filters.title_contains:
            clauses.append("LOWER(title) LIKE ?")
            params.append(f"%{filters.title_contains.casefold()}%")
        if filters.media_types:
            placeholders = ",".join("?" * len(filters.media_types))
            clauses.append(f"media_type IN ({placeholders})")
            params.extend(filters.media_types)
        if filters.ingested_after:
            clauses.append("ingested_at_utc >= ?")
            params.append(_to_iso(filters.ingested_after))
        if filters.ingested_before:
            clauses.append("ingested_at_utc <= ?")
            params.append(_to_iso(filters.ingested_before))
        where = " AND ".join(clauses)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT document_id FROM documents WHERE {where}",  # noqa: S608
                tuple(params),
            ).fetchall()
        return {r["document_id"] for r in rows}

    def counts(self) -> tuple[int, int]:
        with self._lock:
            docs = self._conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
            chunks = self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        return int(docs), int(chunks)
