"""HTML extraction using the standard library only.

No parsing dependency: `html.parser` handles entities and malformed markup, and adding a
scraping library for this would be a dependency to justify at every future audit.

Two decisions worth stating plainly, because both lose information:

- `script`, `style`, `template` and `noscript` contents are dropped entirely. They are not
  prose, and indexing them pollutes retrieval with minified JavaScript.
- `nav`, `header`, `footer` and `aside` are dropped as boilerplate. This is what makes repeated
  navigation chrome stop competing with body text across every page of a site. It also means a
  fact stated *only* in a footer is lost, which is a real trade-off, tested and documented.

Offsets index into the extracted text, not into the original HTML bytes. A citation therefore
reproduces the extracted text exactly, and cannot be mapped back to a byte range in the source
file. That is a stated limitation, not an oversight.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from atlasrag.domain.models import SourceLocator
from atlasrag.errors import MalformedDocumentError
from atlasrag.ingestion.loaders.base import LoadedDocument
from atlasrag.ingestion.normalize import decode_bytes, normalize_text

_DROP_CONTENT = {"script", "style", "template", "noscript"}
_DROP_SECTIONS = {"nav", "header", "footer", "aside"}
_BLOCK = {
    "address",
    "article",
    "blockquote",
    "div",
    "dd",
    "dl",
    "dt",
    "fieldset",
    "figcaption",
    "figure",
    "form",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "li",
    "main",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
}
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_PREFORMATTED = {"pre", "code", "textarea"}
_WHITESPACE = re.compile(r"\s+")


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.headings: list[tuple[int, str]] = []
        self.title: str | None = None
        self._suppress_depth = 0
        self._suppressed_tags: list[str] = []
        self._capture: str | None = None
        self._captured: list[str] = []
        self._pre_depth = 0

    def _emit(self, text: str) -> None:
        self.parts.append(text)

    @property
    def _length(self) -> int:
        return sum(len(p) for p in self.parts)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _DROP_CONTENT or tag in _DROP_SECTIONS:
            self._suppress_depth += 1
            self._suppressed_tags.append(tag)
            return
        if self._suppress_depth:
            return
        if tag in _BLOCK:
            self._emit("\n\n")
        if tag == "br":
            self._emit("\n")
        if tag in _HEADINGS or tag == "title":
            self._capture = tag
            self._captured = []

    def handle_endtag(self, tag: str) -> None:
        if self._suppressed_tags and self._suppressed_tags[-1] == tag:
            self._suppressed_tags.pop()
            self._suppress_depth -= 1
            return
        if self._suppress_depth:
            return
        if tag in _PREFORMATTED and self._pre_depth:
            self._pre_depth -= 1
        if self._capture == tag:
            label = " ".join("".join(self._captured).split())
            if tag == "title":
                self.title = label or None
            elif label:
                # Offset of the heading text itself, which the block break has already opened.
                self.headings.append((self._length, label))
                self._emit(label)
            self._capture = None
            self._captured = []
        if tag in _BLOCK:
            self._emit("\n\n")

    def handle_data(self, data: str) -> None:
        if self._suppress_depth:
            return
        if self._capture is not None:
            self._captured.append(data)
            return
        # HTML treats a newline inside a paragraph as a space. Preserving the source's line
        # wrapping would put arbitrary line breaks inside every cited snippet.
        self._emit(data if self._pre_depth else _WHITESPACE.sub(" ", data))


class HtmlLoader:
    @property
    def media_types(self) -> tuple[str, ...]:
        return ("text/html",)

    def load(self, raw: bytes, *, filename: str) -> LoadedDocument:
        source = decode_bytes(raw)
        parser = _Extractor()
        try:
            parser.feed(source)
            parser.close()
        except Exception as exc:
            raise MalformedDocumentError(f"could not parse HTML in {filename}: {exc}") from exc

        raw_text = "".join(parser.parts)
        text = normalize_text(raw_text)

        # Normalisation collapses whitespace, so heading offsets recorded against the raw
        # extraction no longer line up. Re-locate each heading in the normalised text instead of
        # trusting a stale offset — a locator pointing at the wrong place is worse than none.
        locators: list[SourceLocator] = []
        cursor = 0
        for _, label in parser.headings:
            found = text.find(label, cursor)
            if found == -1:
                continue
            locators.append(SourceLocator(label=label, start_offset=found))
            cursor = found + len(label)

        title = parser.title or (locators[0].label if locators else None)
        return LoadedDocument(text=text, locators=locators, title_hint=title)
