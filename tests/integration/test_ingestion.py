"""Acceptance criteria 1, 2, 3: id stability, idempotency, loud failure."""

from __future__ import annotations

from pathlib import Path

from atlasrag.config import Settings
from atlasrag.service import AtlasRagService
from tests.conftest import FIXTURE_CORPUS, HashEmbedder

SAMPLE = FIXTURE_CORPUS / "atlas-gateway-runbook.md"


def test_reingesting_the_same_file_is_idempotent(service: AtlasRagService) -> None:
    first = service.ingest_path(SAMPLE)
    assert first.status == "ingested"

    second = service.ingest_path(SAMPLE)
    assert second.status == "unchanged"
    assert second.document_id == first.document_id

    documents, chunks = service.store.counts()
    assert documents == 1
    assert chunks == first.chunk_count


def test_ids_are_identical_across_two_independent_databases(tmp_path: Path) -> None:
    """Acceptance criterion 1: same input and configuration, same ids."""
    ids: list[tuple[str, tuple[str, ...]]] = []
    for name in ("a", "b"):
        service = AtlasRagService(Settings(data_dir=tmp_path / name), embedder=HashEmbedder())
        result = service.ingest_path(SAMPLE)
        assert result.document_id is not None
        chunks = service.store.chunks_for_document(result.document_id)
        ids.append((result.document_id, tuple(c.chunk_id for c in chunks)))
        service.close()

    assert ids[0] == ids[1]


def test_identical_content_under_a_different_name_is_reported_as_duplicate(
    service: AtlasRagService, tmp_path: Path
) -> None:
    original = service.ingest_path(SAMPLE)
    copy = tmp_path / "renamed-copy.md"
    copy.write_bytes(SAMPLE.read_bytes())

    result = service.ingest_path(copy)
    assert result.status == "duplicate_content"
    assert result.duplicate_of == original.document_id
    assert service.store.counts()[0] == 1


def test_editing_a_file_creates_a_new_document_id(service: AtlasRagService, tmp_path: Path) -> None:
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\nOriginal body sentence here.", encoding="utf-8")
    first = service.ingest_path(path)

    path.write_text("# Notes\n\nEdited body sentence here.", encoding="utf-8")
    second = service.ingest_path(path)

    assert first.document_id != second.document_id
    assert service.store.counts()[0] == 2


class TestRejections:
    def test_empty_file(self, service: AtlasRagService, tmp_path: Path) -> None:
        path = tmp_path / "empty.txt"
        path.write_bytes(b"")
        result = service.ingest_path(path)
        assert result.status == "rejected"
        assert result.errors[0].code == "empty_document"

    def test_whitespace_only_file(self, service: AtlasRagService, tmp_path: Path) -> None:
        path = tmp_path / "blank.txt"
        path.write_text("   \n\n\t\n", encoding="utf-8")
        result = service.ingest_path(path)
        assert result.status == "rejected"
        assert result.errors[0].code == "empty_document"

    def test_unsupported_extension(self, service: AtlasRagService, tmp_path: Path) -> None:
        path = tmp_path / "program.exe"
        path.write_bytes(b"MZ binary content")
        result = service.ingest_path(path)
        assert result.status == "rejected"
        assert result.errors[0].code == "unsupported_media_type"

    def test_oversize_file(self, tmp_path: Path) -> None:
        service = AtlasRagService(
            Settings(data_dir=tmp_path / "var", max_document_bytes=64), embedder=HashEmbedder()
        )
        path = tmp_path / "big.txt"
        path.write_text("x" * 500, encoding="utf-8")
        result = service.ingest_path(path)
        assert result.status == "rejected"
        assert result.errors[0].code == "document_too_large"
        service.close()

    def test_invalid_utf8(self, service: AtlasRagService, tmp_path: Path) -> None:
        path = tmp_path / "broken.txt"
        path.write_bytes(b"valid start \xff\xfe then broken")
        result = service.ingest_path(path)
        assert result.status == "rejected"
        assert result.errors[0].code == "malformed_document"

    def test_rejection_leaves_no_partial_rows(
        self, service: AtlasRagService, tmp_path: Path
    ) -> None:
        path = tmp_path / "program.exe"
        path.write_bytes(b"binary")
        service.ingest_path(path)
        assert service.store.counts() == (0, 0)

    def test_missing_file(self, service: AtlasRagService, tmp_path: Path) -> None:
        result = service.ingest_path(tmp_path / "does-not-exist.md")
        assert result.status == "rejected"
        assert result.errors[0].code == "not_a_file"


def test_stored_chunks_match_the_document_text_exactly(service: AtlasRagService) -> None:
    result = service.ingest_path(SAMPLE)
    assert result.document_id is not None
    document = service.get_document(result.document_id)
    assert document is not None
    for chunk in service.chunks_for_document(result.document_id):
        assert chunk.text == document.normalized_text[chunk.start_offset : chunk.end_offset]
