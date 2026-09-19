"""Deterministic sentence segmentation that preserves exact offsets.

Offsets are the whole point: an extracted sentence must be locatable in the source text
character-for-character, or it cannot be cited.
"""

from __future__ import annotations

import re

_BOUNDARY = re.compile(r"[.!?]+[\"')\]]*(?=\s|$)|\n{2,}")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end) spans of sentences, whitespace-trimmed, in document order."""
    raw: list[tuple[int, int]] = []
    cursor = 0
    for match in _BOUNDARY.finditer(text):
        raw.append((cursor, match.end()))
        cursor = match.end()
    if cursor < len(text):
        raw.append((cursor, len(text)))

    spans: list[tuple[int, int]] = []
    for start, end in raw:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end > start:
            spans.append((start, end))
    return spans
