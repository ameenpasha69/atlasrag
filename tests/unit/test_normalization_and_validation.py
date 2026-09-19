"""Normalization, validation, safe filenames and sentence spans."""

from __future__ import annotations

from pathlib import Path

import pytest

from atlasrag.answering.text_spans import sentence_spans
from atlasrag.errors import (
    DocumentTooLargeError,
    EmptyDocumentError,
    MalformedDocumentError,
    UnsafePathError,
    UnsupportedMediaTypeError,
)
from atlasrag.ingestion.normalize import decode_bytes, derive_title, normalize_text
from atlasrag.ingestion.validate import (
    media_type_for,
    resolve_within,
    safe_filename,
    validate_normalized,
    validate_size,
)

ALLOWED = (".txt", ".md", ".markdown")


class TestNormalize:
    def test_line_endings_converge(self) -> None:
        assert normalize_text("a\r\nb\rc") == normalize_text("a\nb\nc")

    def test_unicode_forms_converge(self) -> None:
        assert normalize_text("café") == normalize_text("café")

    def test_bom_is_stripped(self) -> None:
        assert decode_bytes("﻿hello".encode()) == "hello"

    def test_zero_width_characters_removed(self) -> None:
        assert normalize_text("a​b") == "ab"

    def test_trailing_whitespace_per_line_removed(self) -> None:
        assert normalize_text("a   \nb\t\n") == "a\nb"

    def test_runs_of_blank_lines_collapse(self) -> None:
        assert normalize_text("a\n\n\n\n\nb") == "a\n\nb"

    def test_is_idempotent(self) -> None:
        once = normalize_text("  a\r\n\r\n\r\nb  \n")
        assert normalize_text(once) == once

    def test_invalid_utf8_is_rejected(self) -> None:
        with pytest.raises(MalformedDocumentError):
            decode_bytes(b"\xff\xfe\x00bad")


class TestTitle:
    def test_markdown_heading_wins(self) -> None:
        assert derive_title("# Real Title\n\nbody", "file.md") == "Real Title"

    def test_falls_back_to_filename(self) -> None:
        assert derive_title("no heading here", "my_notes-file.txt") == "my notes file"

    def test_ignores_heading_deeper_in_text_only_if_none_earlier(self) -> None:
        assert derive_title("intro line\n\n## Second\n\nbody", "f.md") == "Second"


class TestValidation:
    def test_unsupported_extension(self) -> None:
        with pytest.raises(UnsupportedMediaTypeError):
            media_type_for("payload.exe", ALLOWED)

    def test_no_extension(self) -> None:
        with pytest.raises(UnsupportedMediaTypeError):
            media_type_for("README", ALLOWED)

    def test_supported_extensions(self) -> None:
        assert media_type_for("a.txt", ALLOWED) == "text/plain"
        assert media_type_for("a.MD", ALLOWED) == "text/markdown"

    def test_zero_bytes_rejected(self) -> None:
        with pytest.raises(EmptyDocumentError):
            validate_size(0, 100, "a.txt")

    def test_oversize_rejected(self) -> None:
        with pytest.raises(DocumentTooLargeError):
            validate_size(101, 100, "a.txt")

    def test_whitespace_only_rejected_after_normalization(self) -> None:
        with pytest.raises(EmptyDocumentError):
            validate_normalized("   \n\n  ", "a.txt")


class TestSafeFilename:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("../../etc/passwd", "passwd"),
            ("..\\..\\windows\\system32\\cfg.txt", "cfg.txt"),
            ("normal.md", "normal.md"),
            ("with space.md", "with space.md"),
            ("a<b>c:d.md", "a_b_c_d.md"),
        ],
    )
    def test_traversal_and_unsafe_characters_removed(self, raw: str, expected: str) -> None:
        assert safe_filename(raw) == expected

    def test_windows_reserved_names_are_escaped(self) -> None:
        assert safe_filename("CON.txt") == "_CON.txt"
        assert safe_filename("lpt1.md") == "_lpt1.md"

    def test_empty_after_sanitization_is_refused(self) -> None:
        with pytest.raises(UnsafePathError):
            safe_filename("../")

    def test_null_byte_removed(self) -> None:
        assert safe_filename("a\x00b.md") == "a_b.md"


class TestResolveWithin:
    def test_escape_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(UnsafePathError):
            resolve_within(tmp_path, Path("../outside.txt"))

    def test_inside_is_allowed(self, tmp_path: Path) -> None:
        assert resolve_within(tmp_path, Path("inside.txt")).parent == tmp_path.resolve()


class TestSentenceSpans:
    def test_splits_on_terminators_and_preserves_offsets(self) -> None:
        text = "First one. Second two! Third three?"
        spans = sentence_spans(text)
        assert [text[s:e] for s, e in spans] == ["First one.", "Second two!", "Third three?"]

    def test_trailing_fragment_is_kept(self) -> None:
        text = "Complete. Fragment without terminator"
        assert [text[s:e] for s, e in sentence_spans(text)][-1] == "Fragment without terminator"

    def test_blank_line_is_a_boundary(self) -> None:
        text = "Heading line\n\nBody sentence."
        assert [text[s:e] for s, e in sentence_spans(text)] == ["Heading line", "Body sentence."]

    def test_no_empty_spans(self) -> None:
        for start, end in sentence_spans("a.  \n\n  b."):
            assert end > start

    def test_empty_text(self) -> None:
        assert sentence_spans("") == []
