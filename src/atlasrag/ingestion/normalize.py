"""Text normalization.

Normalization runs once, before hashing and chunking, so that the same logical document
produces the same id regardless of the editor, line-ending convention or Unicode form it
arrived in. `normalized_text` is thereafter the only text the system knows about, and all
citation offsets refer to it.
"""

from __future__ import annotations

import re
import unicodedata

from atlasrag.errors import MalformedDocumentError

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Built from code points so this source file stays pure ASCII: the literal
# characters are invisible and easy to corrupt in an editor.
_ZERO_WIDTH = re.compile(
    "[" + "".join(chr(c) for c in (*range(0x200B, 0x2010), 0x2028, 0x2029, 0xFEFF)) + "]"
)
_TRAILING_WS = re.compile(r"[ \t]+$", re.MULTILINE)
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

_MD_HEADING = re.compile(r"^#{1,6}[ \t]+(?P<title>\S.*?)[ \t]*$", re.MULTILINE)


def decode_bytes(raw: bytes) -> str:
    """Decode as UTF-8 (tolerating a BOM). Anything else is a malformed input, not a guess."""
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise MalformedDocumentError(
            f"file is not valid UTF-8 (byte {exc.start}): {exc.reason}"
        ) from exc


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = _ZERO_WIDTH.sub("", text)
    text = _CONTROL_CHARS.sub("", text)
    text = text.expandtabs(4)
    text = _TRAILING_WS.sub("", text)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def derive_title(normalized_text: str, original_filename: str) -> str:
    match = _MD_HEADING.search(normalized_text)
    if match:
        title = match.group("title").strip().strip("#").strip()
        if title:
            return title[:200]
    stem = original_filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    return (stem.replace("_", " ").replace("-", " ").strip() or original_filename)[:200]
