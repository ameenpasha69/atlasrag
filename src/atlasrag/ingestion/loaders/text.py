"""Plain text and Markdown.

Markdown headings become section locators, so a citation into a `.md` file can be described as
"Retry semantics" rather than only as characters 1840-1962. Plain text has no structure to
report, so it reports none rather than inventing one.
"""

from __future__ import annotations

import re

from atlasrag.domain.models import SourceLocator
from atlasrag.ingestion.loaders.base import LoadedDocument
from atlasrag.ingestion.normalize import decode_bytes, normalize_text

_HEADING_LINE = re.compile(r"^(#{1,6})[ \t]+(?P<title>\S.*?)[ \t]*$", re.MULTILINE)


class TextLoader:
    @property
    def media_types(self) -> tuple[str, ...]:
        return ("text/plain",)

    def load(self, raw: bytes, *, filename: str) -> LoadedDocument:
        return LoadedDocument(text=normalize_text(decode_bytes(raw)))


class MarkdownLoader:
    @property
    def media_types(self) -> tuple[str, ...]:
        return ("text/markdown",)

    def load(self, raw: bytes, *, filename: str) -> LoadedDocument:
        text = normalize_text(decode_bytes(raw))
        locators = [
            SourceLocator(label=match.group("title").strip(), start_offset=match.start())
            for match in _HEADING_LINE.finditer(text)
        ]
        title = locators[0].label if locators else None
        return LoadedDocument(text=text, locators=locators, title_hint=title)
