"""Deterministic character-window chunker with boundary preference.

Character offsets (not token offsets) are the citation locator, because a character span
can be checked against the stored source text by anyone, with no tokenizer and no model.
The invariant every chunk upholds is:

    chunk.text == document.normalized_text[chunk.start_offset : chunk.end_offset]

`tests/unit/test_chunking.py` asserts this for every chunk of every fixture.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from atlasrag.domain.ids import CHUNKING_VERSION, derive_chunk_id, text_checksum
from atlasrag.domain.models import Chunk, Document

_SENTENCE_END = re.compile(r"[.!?][\"')\]]?\s")


@dataclass(frozen=True, slots=True)
class ChunkingConfig:
    size_chars: int = 1200
    overlap_chars: int = 200
    boundary_lookback: int = 300
    min_chars: int = 100

    def __post_init__(self) -> None:
        if self.overlap_chars >= self.size_chars:
            raise ValueError("overlap_chars must be smaller than size_chars")
        if self.size_chars <= 0:
            raise ValueError("size_chars must be positive")


def _find_boundary(text: str, start: int, hard_end: int, cfg: ChunkingConfig) -> int:
    """Pick the most natural break at or before `hard_end`, preferring larger structures."""
    window_start = max(start + cfg.min_chars, hard_end - cfg.boundary_lookback)
    if window_start >= hard_end:
        return hard_end
    segment = text[window_start:hard_end]

    paragraph = segment.rfind("\n\n")
    if paragraph != -1:
        return window_start + paragraph + 2

    last_sentence = -1
    for match in _SENTENCE_END.finditer(segment):
        last_sentence = match.end()
    if last_sentence != -1:
        return window_start + last_sentence

    newline = segment.rfind("\n")
    if newline != -1:
        return window_start + newline + 1

    space = segment.rfind(" ")
    if space != -1:
        return window_start + space + 1

    return hard_end


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def chunk_document(document: Document, cfg: ChunkingConfig) -> list[Chunk]:
    text = document.normalized_text
    length = len(text)
    chunks: list[Chunk] = []
    start = 0
    index = 0

    while start < length:
        hard_end = min(start + cfg.size_chars, length)
        end = hard_end if hard_end >= length else _find_boundary(text, start, hard_end, cfg)

        trimmed_start, trimmed_end = _trim(text, start, end)
        if trimmed_end > trimmed_start:
            body = text[trimmed_start:trimmed_end]
            checksum = text_checksum(body)
            chunks.append(
                Chunk(
                    chunk_id=derive_chunk_id(
                        document_id=document.document_id,
                        chunk_index=index,
                        start_offset=trimmed_start,
                        end_offset=trimmed_end,
                        content_sha256=checksum,
                    ),
                    document_id=document.document_id,
                    chunk_index=index,
                    start_offset=trimmed_start,
                    end_offset=trimmed_end,
                    text=body,
                    content_sha256=checksum,
                    chunking_version=CHUNKING_VERSION,
                )
            )
            index += 1

        if end >= length:
            break
        next_start = end - cfg.overlap_chars
        start = next_start if next_start > start else start + 1

    return chunks
