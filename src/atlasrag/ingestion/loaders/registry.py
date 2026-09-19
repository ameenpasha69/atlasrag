"""Maps a media type to the loader that understands it."""

from __future__ import annotations

from atlasrag.errors import UnsupportedMediaTypeError
from atlasrag.ingestion.loaders.base import Loader
from atlasrag.ingestion.loaders.html import HtmlLoader
from atlasrag.ingestion.loaders.pdf import PdfLoader
from atlasrag.ingestion.loaders.text import MarkdownLoader, TextLoader

_LOADERS: tuple[Loader, ...] = (TextLoader(), MarkdownLoader(), HtmlLoader(), PdfLoader())

_BY_MEDIA_TYPE: dict[str, Loader] = {
    media_type: loader for loader in _LOADERS for media_type in loader.media_types
}


def loader_for(media_type: str) -> Loader:
    loader = _BY_MEDIA_TYPE.get(media_type)
    if loader is None:
        raise UnsupportedMediaTypeError(f"no loader registered for {media_type}")
    return loader


def supported_media_types() -> tuple[str, ...]:
    return tuple(sorted(_BY_MEDIA_TYPE))
