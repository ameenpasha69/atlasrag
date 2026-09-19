"""PDF text extraction with page locators.

Scope, stated before the code so it is not mistaken for more than it is:

- Extracts the text layer only. **There is no OCR.** A scanned PDF has no text layer, so it
  extracts nothing and is rejected as empty rather than silently indexed as a blank document.
- **No table extraction and no layout understanding.** A table's cells arrive in whatever order
  the text layer stores them, which may not be reading order.
- Repeated headers and footers are detected by appearing in the same position on most pages and
  removed, because otherwise every page of a long document shares the same boilerplate and that
  boilerplate competes with body text in retrieval.

Page boundaries become locators, so a citation can be reported as "page 4" while the offsets
still index exactly into the extracted text.
"""

from __future__ import annotations

import io
from collections import Counter

from atlasrag.domain.models import SourceLocator
from atlasrag.errors import MalformedDocumentError
from atlasrag.ingestion.loaders.base import LoadedDocument
from atlasrag.ingestion.normalize import normalize_text

# A line must appear on at least this proportion of pages, and the document must have at least
# MIN_PAGES_FOR_BOILERPLATE pages, before it is treated as running head or foot. Below that the
# evidence is too thin and a genuine repeated sentence would be destroyed.
BOILERPLATE_PAGE_RATIO = 0.6
MIN_PAGES_FOR_BOILERPLATE = 3
EDGE_LINES = 2


def _page_texts(raw: bytes) -> list[str]:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise MalformedDocumentError(f"pypdf unavailable: {exc}") from exc

    try:
        reader = PdfReader(io.BytesIO(raw))
        if reader.is_encrypted:
            raise MalformedDocumentError("PDF is encrypted; decrypt it before ingesting")
        return [page.extract_text() or "" for page in reader.pages]
    except MalformedDocumentError:
        raise
    except (PdfReadError, ValueError, OSError, KeyError, TypeError) as exc:
        raise MalformedDocumentError(f"could not read PDF: {exc}") from exc


def _boilerplate_lines(pages: list[str]) -> set[str]:
    """Lines appearing at the top or bottom of most pages."""
    if len(pages) < MIN_PAGES_FOR_BOILERPLATE:
        return set()

    counts: Counter[str] = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        edges = lines[:EDGE_LINES] + lines[-EDGE_LINES:]
        counts.update(set(edges))

    threshold = max(2, int(len(pages) * BOILERPLATE_PAGE_RATIO))
    return {line for line, count in counts.items() if count >= threshold}


def _strip_boilerplate(page: str, boilerplate: set[str]) -> str:
    kept = [ln for ln in page.splitlines() if ln.strip() not in boilerplate]
    return "\n".join(kept)


class PdfLoader:
    @property
    def media_types(self) -> tuple[str, ...]:
        return ("application/pdf",)

    def load(self, raw: bytes, *, filename: str) -> LoadedDocument:
        pages = _page_texts(raw)
        if not pages:
            raise MalformedDocumentError(f"{filename} contains no pages")

        boilerplate = _boilerplate_lines(pages)

        parts: list[str] = []
        locators: list[SourceLocator] = []
        cursor = 0
        for number, page in enumerate(pages, start=1):
            body = normalize_text(_strip_boilerplate(page, boilerplate))
            if not body:
                continue
            if parts:
                parts.append("\n\n")
                cursor += 2
            locators.append(SourceLocator(label=f"page {number}", start_offset=cursor))
            parts.append(body)
            cursor += len(body)

        text = "".join(parts)
        # `text` is already the concatenation of normalised pages; normalising again would shift
        # every offset recorded above and silently invalidate the locators.
        return LoadedDocument(text=text, locators=locators, title_hint=None)
