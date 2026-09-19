"""Acceptance criteria 1 and 11 at the unit level: id stability and offset fidelity."""

from __future__ import annotations

from datetime import UTC, datetime
from itertools import pairwise

import pytest

from atlasrag.chunking.fixed_window import ChunkingConfig, chunk_document
from atlasrag.domain.ids import (
    derive_chunk_id,
    derive_document_id,
    normalize_source_key,
    text_checksum,
)
from atlasrag.domain.models import Document, DocumentSource

LONG_TEXT = (
    "Alpha paragraph about the staging belt and its behaviour under load.\n\n"
    "Beta paragraph describing the sorting line and the exception lane in detail. "
    "It continues for a while so that the chunker has something to divide. "
    "Gamma sentence adds more length to force a second window boundary here.\n\n"
    "Delta paragraph closes the document with a final observation about throughput."
) * 6


def make_document(text: str = LONG_TEXT, filename: str = "sample.md") -> Document:
    checksum = text_checksum(text)
    return Document(
        document_id=derive_document_id(
            source_key=normalize_source_key(filename), content_sha256=checksum
        ),
        title="Sample",
        source=DocumentSource(
            uri=filename,
            scheme="file",
            media_type="text/markdown",
            original_filename=filename,
            size_bytes=len(text.encode()),
        ),
        content_sha256=checksum,
        normalized_text=text,
        char_count=len(text),
        ingestion_version="1",
        ingested_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
    )


class TestDocumentIds:
    def test_identical_input_yields_identical_id(self) -> None:
        assert make_document().document_id == make_document().document_id

    def test_different_content_yields_different_id(self) -> None:
        assert make_document().document_id != make_document(LONG_TEXT + " extra").document_id

    def test_different_filename_yields_different_id(self) -> None:
        assert make_document().document_id != make_document(filename="other.md").document_id

    def test_source_key_is_case_folded_and_stripped_of_directories(self) -> None:
        assert normalize_source_key("A/B/Notes.MD") == "notes.md"
        assert normalize_source_key("notes.md") == normalize_source_key("NOTES.MD")

    def test_ingestion_version_participates(self) -> None:
        checksum = text_checksum("x")
        assert derive_document_id(
            source_key="a.md", content_sha256=checksum, ingestion_version="1"
        ) != derive_document_id(source_key="a.md", content_sha256=checksum, ingestion_version="2")


class TestChunkIds:
    def test_every_component_changes_the_id(self) -> None:
        base = {
            "document_id": "d",
            "chunk_index": 0,
            "start_offset": 0,
            "end_offset": 10,
            "content_sha256": "c",
        }
        baseline = derive_chunk_id(**base)  # type: ignore[arg-type]
        for field, value in (
            ("document_id", "d2"),
            ("chunk_index", 1),
            ("start_offset", 1),
            ("end_offset", 11),
            ("content_sha256", "c2"),
        ):
            assert derive_chunk_id(**{**base, field: value}) != baseline  # type: ignore[arg-type]

    def test_chunking_version_participates(self) -> None:
        base = {
            "document_id": "d",
            "chunk_index": 0,
            "start_offset": 0,
            "end_offset": 10,
            "content_sha256": "c",
        }
        assert derive_chunk_id(**base, chunking_version="1") != derive_chunk_id(  # type: ignore[arg-type]
            **base, chunking_version="2"
        )


class TestChunking:
    @pytest.fixture
    def cfg(self) -> ChunkingConfig:
        return ChunkingConfig(size_chars=400, overlap_chars=80, boundary_lookback=120, min_chars=50)

    def test_produces_multiple_chunks(self, cfg: ChunkingConfig) -> None:
        assert len(chunk_document(make_document(), cfg)) > 1

    def test_offsets_reproduce_the_source_text_exactly(self, cfg: ChunkingConfig) -> None:
        """Acceptance criterion 11, at its root: a citation locator must be exact."""
        document = make_document()
        for chunk in chunk_document(document, cfg):
            assert chunk.text == document.normalized_text[chunk.start_offset : chunk.end_offset]

    def test_chunks_are_identical_across_runs(self, cfg: ChunkingConfig) -> None:
        first = chunk_document(make_document(), cfg)
        second = chunk_document(make_document(), cfg)
        assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
        assert [(c.start_offset, c.end_offset) for c in first] == [
            (c.start_offset, c.end_offset) for c in second
        ]

    def test_indices_are_contiguous_from_zero(self, cfg: ChunkingConfig) -> None:
        chunks = chunk_document(make_document(), cfg)
        assert [c.chunk_index for c in chunks] == list(range(len(chunks)))

    def test_whole_document_is_covered(self, cfg: ChunkingConfig) -> None:
        document = make_document()
        chunks = chunk_document(document, cfg)
        assert chunks[0].start_offset == 0
        assert chunks[-1].end_offset == len(document.normalized_text)

    def test_consecutive_chunks_overlap_or_touch(self, cfg: ChunkingConfig) -> None:
        chunks = chunk_document(make_document(), cfg)
        for previous, current in pairwise(chunks):
            assert current.start_offset <= previous.end_offset

    def test_short_document_is_a_single_chunk(self, cfg: ChunkingConfig) -> None:
        chunks = chunk_document(make_document("Just one short sentence here."), cfg)
        assert len(chunks) == 1
        assert chunks[0].start_offset == 0

    def test_changing_config_changes_chunk_ids(self) -> None:
        document = make_document()
        a = chunk_document(document, ChunkingConfig(size_chars=400, overlap_chars=80))
        b = chunk_document(document, ChunkingConfig(size_chars=500, overlap_chars=80))
        assert [c.chunk_id for c in a] != [c.chunk_id for c in b]

    def test_no_chunk_starts_or_ends_with_whitespace(self, cfg: ChunkingConfig) -> None:
        for chunk in chunk_document(make_document(), cfg):
            assert chunk.text == chunk.text.strip()

    def test_overlap_must_be_smaller_than_size(self) -> None:
        with pytest.raises(ValueError, match="overlap_chars"):
            ChunkingConfig(size_chars=100, overlap_chars=100)
