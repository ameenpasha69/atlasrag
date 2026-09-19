"""Citation construction and validation.

A citation is only allowed to exist if the exact snippet can be found at the exact offsets
of the cited document, read back from the registry. Validation deliberately re-reads the
*document* rather than the chunk row, so a corrupted or stale index cannot vouch for itself.
"""

from __future__ import annotations

from atlasrag.domain.models import Citation, RankContribution
from atlasrag.ingestion.loaders.base import locator_for
from atlasrag.storage.sqlite_store import SqliteDocumentStore


def verify_span(
    store: SqliteDocumentStore, *, document_id: str, start: int, end: int, snippet: str
) -> bool:
    document = store.get_document(document_id)
    if document is None:
        return False
    text = document.normalized_text
    if start < 0 or end > len(text) or end <= start:
        return False
    return text[start:end] == snippet


def build_citation(
    store: SqliteDocumentStore,
    *,
    document_id: str,
    chunk_id: str,
    title: str,
    start: int,
    end: int,
    snippet: str,
    provenance: list[RankContribution] | None = None,
) -> Citation:
    validated = verify_span(store, document_id=document_id, start=start, end=end, snippet=snippet)
    document = store.get_document(document_id)
    return Citation(
        document_id=document_id,
        chunk_id=chunk_id,
        title=title,
        start_offset=start,
        end_offset=end,
        snippet=snippet,
        validated=validated,
        locator=locator_for(document.locators, start) if document else None,
        retrieval_provenance=provenance or [],
    )


def all_validated(citations: list[Citation]) -> bool:
    return bool(citations) and all(c.validated for c in citations)
