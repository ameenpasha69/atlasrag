"""Loader contract.

A loader turns raw bytes into normalized text plus the locators that let a character offset be
described in the source's own terms — "page 4" for a PDF, a heading for Markdown or HTML. Only
the loader knows which of those the source even has, which is why locators are produced here
and not inferred downstream.

Every loader upholds one invariant, checked per format in the tests: the offsets it reports
index into the text it returns, so a citation span extracted later reproduces the source text
exactly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from atlasrag.domain.models import SourceLocator


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    text: str
    locators: list[SourceLocator] = field(default_factory=list)
    title_hint: str | None = None


class Loader(Protocol):
    @property
    def media_types(self) -> tuple[str, ...]: ...

    def load(self, raw: bytes, *, filename: str) -> LoadedDocument: ...


def locator_for(locators: list[SourceLocator], offset: int) -> str | None:
    """The last locator at or before `offset`.

    Linear scan: locator lists are one entry per page or heading, so they are short, and a
    bisect here would trade clarity for nothing measurable.
    """
    found: str | None = None
    for locator in locators:
        if locator.start_offset <= offset:
            found = locator.label
        else:
            break
    return found
