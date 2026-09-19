"""Input validation and safe filename handling."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

from atlasrag.errors import (
    DocumentTooLargeError,
    EmptyDocumentError,
    UnsafePathError,
    UnsupportedMediaTypeError,
)

_MEDIA_TYPES = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".html": "text/html",
    ".htm": "text/html",
    ".pdf": "application/pdf",
}

_UNSAFE_NAME = re.compile(r"[\x00-\x1f<>:\"/\\|?*]")
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def safe_filename(candidate: str) -> str:
    """Reduce an untrusted filename to a plain, traversal-free basename.

    Applied to every upload before the name is stored or echoed back, so that neither the
    filesystem nor the UI ever sees `../`, a device name, or a control character.
    """
    name = unicodedata.normalize("NFC", candidate).strip()
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = _UNSAFE_NAME.sub("_", name).strip(". ")
    if not name:
        raise UnsafePathError("filename is empty after sanitization")
    stem = name.rsplit(".", 1)[0].casefold()
    if stem in _WINDOWS_RESERVED:
        name = f"_{name}"
    return name[:255]


def resolve_within(base: Path, candidate: Path) -> Path:
    """Resolve `candidate` and refuse it if it escapes `base`."""
    base_resolved = base.resolve()
    target = (
        (base_resolved / candidate).resolve()
        if not candidate.is_absolute()
        else candidate.resolve()
    )
    if base_resolved != target and base_resolved not in target.parents:
        raise UnsafePathError(f"path escapes the permitted directory: {candidate}")
    return target


def media_type_for(filename: str, allowed_extensions: tuple[str, ...]) -> str:
    suffix = Path(filename).suffix.casefold()
    if suffix not in allowed_extensions:
        raise UnsupportedMediaTypeError(
            f"unsupported file extension {suffix or '(none)'}; "
            f"allowed: {', '.join(allowed_extensions)}"
        )
    media_type = _MEDIA_TYPES.get(suffix)
    if media_type is None:
        raise UnsupportedMediaTypeError(f"no media type registered for {suffix}")
    return media_type


def validate_size(size_bytes: int, max_bytes: int, filename: str) -> None:
    if size_bytes == 0:
        raise EmptyDocumentError(f"{filename} is empty (0 bytes)")
    if size_bytes > max_bytes:
        raise DocumentTooLargeError(
            f"{filename} is {size_bytes} bytes, exceeding the {max_bytes} byte limit"
        )


def validate_normalized(text: str, filename: str) -> None:
    if not text.strip():
        raise EmptyDocumentError(f"{filename} contains no text after normalization")
