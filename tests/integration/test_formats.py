"""Milestone 4: HTML and PDF adapters, each with its own fixtures and its own failure modes.

Per format, the specification asks for reading order, encoding, repeated headers and footers,
page or section locators, empty extraction, malformed inputs and citation accuracy. Each has a
test below, and the things that are *not* supported have tests asserting they fail loudly
rather than quietly producing something plausible.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from atlasrag.domain.models import SearchQuery
from atlasrag.errors import MalformedDocumentError
from atlasrag.ingestion.loaders.base import locator_for
from atlasrag.ingestion.loaders.html import HtmlLoader
from atlasrag.ingestion.loaders.pdf import PdfLoader
from atlasrag.ingestion.loaders.registry import loader_for, supported_media_types
from atlasrag.ingestion.loaders.text import MarkdownLoader
from atlasrag.service import AtlasRagService

FORMATS = Path(__file__).resolve().parents[2] / "fixtures" / "corpus" / "formats"
HTML_FIXTURE = FORMATS / "hub-status-page.html"
PDF_FIXTURE = FORMATS / "vehicle-inspection.pdf"
SCANNED_FIXTURE = FORMATS / "scanned-no-text.pdf"


class TestRegistry:
    def test_every_supported_media_type_resolves(self) -> None:
        for media_type in supported_media_types():
            assert loader_for(media_type) is not None

    def test_expected_formats_are_registered(self) -> None:
        assert set(supported_media_types()) == {
            "text/plain",
            "text/markdown",
            "text/html",
            "application/pdf",
        }


class TestHtml:
    @pytest.fixture
    def loaded(self):
        return HtmlLoader().load(HTML_FIXTURE.read_bytes(), filename="hub-status-page.html")

    def test_script_and_style_content_is_excluded(self, loaded) -> None:
        assert "hubStatus" not in loaded.text
        assert "console.log" not in loaded.text
        assert "font-family" not in loaded.text

    def test_navigation_chrome_is_excluded(self, loaded) -> None:
        """nav / header / footer / aside are boilerplate repeated across a site's pages."""
        assert "navigation chrome" not in loaded.text
        assert "Reports" not in loaded.text
        assert "vehicle maintenance schedule" not in loaded.text

    def test_body_prose_survives(self, loaded) -> None:
        assert "exception lane depth" in loaded.text
        assert "HUB-ESC-2291" in loaded.text

    def test_entities_are_decoded(self, loaded) -> None:
        assert "<angle brackets>" in loaded.text
        assert "&amp;" not in loaded.text
        assert "&mdash;" not in loaded.text

    def test_reading_order_follows_the_document(self, loaded) -> None:
        colours = loaded.text.index("Status colours")
        gaps = loaded.text.index("Reporting gaps")
        assert colours < gaps

    def test_source_line_wrapping_does_not_survive(self, loaded) -> None:
        """A newline inside a paragraph is insignificant in HTML and must not reach a snippet."""
        assert "derived from the exception lane depth" in loaded.text

    def test_section_locators_point_at_their_heading(self, loaded) -> None:
        assert [loc.label for loc in loaded.locators] == [
            "Hub Status Reference",
            "Status colours",
            "Reporting gaps",
        ]
        for locator in loaded.locators:
            assert loaded.text.startswith(locator.label, locator.start_offset)

    def test_title_comes_from_the_title_element(self, loaded) -> None:
        assert loaded.title_hint == "Hub Status Reference"

    def test_locator_lookup_resolves_an_offset_to_its_section(self, loaded) -> None:
        offset = loaded.text.index("A missing report")
        assert locator_for(loaded.locators, offset) == "Reporting gaps"

    def test_unclosed_tags_do_not_crash(self) -> None:
        loaded = HtmlLoader().load(
            b"<html><body><p>First paragraph<p>Second paragraph<div>Third", filename="broken.html"
        )
        assert "First paragraph" in loaded.text
        assert "Third" in loaded.text

    def test_invalid_encoding_is_rejected(self) -> None:
        with pytest.raises(MalformedDocumentError):
            HtmlLoader().load(b"<p>\xff\xfe broken</p>", filename="bad.html")

    def test_empty_body_extracts_nothing(self) -> None:
        loaded = HtmlLoader().load(
            b"<html><head><title>T</title></head><body></body></html>", filename="empty.html"
        )
        assert loaded.text.strip() == ""


class TestPdf:
    @pytest.fixture
    def loaded(self):
        return PdfLoader().load(PDF_FIXTURE.read_bytes(), filename="vehicle-inspection.pdf")

    def test_page_locators_are_emitted_in_order(self, loaded) -> None:
        assert [loc.label for loc in loaded.locators] == [
            "page 1",
            "page 2",
            "page 3",
            "page 4",
        ]
        offsets = [loc.start_offset for loc in loaded.locators]
        assert offsets == sorted(offsets)

    def test_each_page_locator_points_at_that_page_text(self, loaded) -> None:
        expected = {
            "page 1": "Vehicle Inspection Procedure",
            "page 2": "Tyres and brakes",
            "page 3": "Load restraint",
            "page 4": "Recording and escalation",
        }
        for locator in loaded.locators:
            assert loaded.text.startswith(expected[locator.label], locator.start_offset)

    def test_running_header_and_footer_are_removed(self, loaded) -> None:
        assert "Confidential" not in loaded.text
        assert "revision 7" not in loaded.text

    def test_body_text_survives_boilerplate_removal(self, loaded) -> None:
        assert "Tread depth below 2.4 millimetres" in loaded.text
        assert "BR-118" in loaded.text
        assert "LR-204" in loaded.text

    def test_reading_order_is_page_order(self, loaded) -> None:
        positions = [
            loaded.text.index(marker)
            for marker in ("Vehicle Inspection", "Tyres and brakes", "Load restraint", "Recording")
        ]
        assert positions == sorted(positions)

    def test_locator_lookup_resolves_an_offset_to_its_page(self, loaded) -> None:
        assert locator_for(loaded.locators, loaded.text.index("LR-204")) == "page 3"
        assert locator_for(loaded.locators, loaded.text.index("BR-118")) == "page 2"

    def test_a_scan_with_no_text_layer_extracts_nothing(self) -> None:
        """There is no OCR. A scan must extract nothing, not a plausible blank document."""
        loaded = PdfLoader().load(SCANNED_FIXTURE.read_bytes(), filename="scanned-no-text.pdf")
        assert loaded.text.strip() == ""

    def test_malformed_pdf_is_rejected(self) -> None:
        with pytest.raises(MalformedDocumentError):
            PdfLoader().load(b"%PDF-1.7\nnot actually a pdf", filename="broken.pdf")

    def test_empty_bytes_are_rejected(self) -> None:
        with pytest.raises(MalformedDocumentError):
            PdfLoader().load(b"", filename="empty.pdf")

    def test_boilerplate_stripping_needs_enough_pages(self) -> None:
        """With too few pages a repeated line is not evidence of boilerplate."""
        from atlasrag.ingestion.loaders.pdf import _boilerplate_lines

        assert _boilerplate_lines(["Header\nA", "Header\nB"]) == set()


class TestMarkdownLocators:
    def test_headings_become_section_locators(self) -> None:
        raw = b"# Title\n\nIntro text here.\n\n## Retry semantics\n\nBody about retries.\n"
        loaded = MarkdownLoader().load(raw, filename="doc.md")
        assert [loc.label for loc in loaded.locators] == ["Title", "Retry semantics"]
        assert locator_for(loaded.locators, loaded.text.index("Body about retries")) == (
            "Retry semantics"
        )

    def test_plain_text_reports_no_structure_it_does_not_have(self) -> None:
        from atlasrag.ingestion.loaders.text import TextLoader

        loaded = TextLoader().load(b"Just prose, no headings.", filename="notes.txt")
        assert loaded.locators == []


class TestEndToEndAcrossFormats:
    """Citation accuracy is the property that must hold identically for every format."""

    @pytest.fixture
    def loaded_service(self, service: AtlasRagService) -> AtlasRagService:
        for path in (HTML_FIXTURE, PDF_FIXTURE):
            result = service.ingest_path(path)
            assert result.status == "ingested", result
        service.indexes.ensure_ready()
        return service

    def test_both_formats_are_searchable(self, loaded_service: AtlasRagService) -> None:
        html_hit = loaded_service.search(SearchQuery(text="HUB-ESC-2291", mode="bm25", top_k=3))
        assert html_hit.hits and "escalation" in html_hit.hits[0].text.lower()

        pdf_hit = loaded_service.search(SearchQuery(text="BR-118", mode="bm25", top_k=3))
        assert pdf_hit.hits and "brake" in pdf_hit.hits[0].text.lower()

    def test_chunk_offsets_reproduce_source_text_for_every_format(
        self, loaded_service: AtlasRagService
    ) -> None:
        for summary in loaded_service.list_documents():
            document = loaded_service.get_document(summary.document_id)
            assert document is not None
            for chunk in loaded_service.chunks_for_document(summary.document_id):
                assert chunk.text == document.normalized_text[chunk.start_offset : chunk.end_offset]

    def test_citations_carry_a_locator(self, loaded_service: AtlasRagService) -> None:
        response = loaded_service.answer(
            SearchQuery(text="what tread depth fails the inspection", mode="bm25", top_k=5)
        )
        assert response.answered, response.abstention
        assert any(c.locator and c.locator.startswith("page") for c in response.citations)

    def test_locators_survive_a_storage_round_trip(self, loaded_service: AtlasRagService) -> None:
        pdf_doc = next(
            d for d in loaded_service.list_documents() if d.media_type == "application/pdf"
        )
        document = loaded_service.get_document(pdf_doc.document_id)
        assert document is not None
        assert [loc.label for loc in document.locators] == ["page 1", "page 2", "page 3", "page 4"]

    def test_a_scanned_pdf_is_rejected_rather_than_indexed_blank(
        self, service: AtlasRagService
    ) -> None:
        result = service.ingest_path(SCANNED_FIXTURE)
        assert result.status == "rejected"
        assert result.errors[0].code == "empty_document"
        assert service.store.counts() == (0, 0)
