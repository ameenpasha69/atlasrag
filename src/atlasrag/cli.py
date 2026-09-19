"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from atlasrag.config import Settings
from atlasrag.domain.models import SearchFilters, SearchQuery
from atlasrag.service import AtlasRagService


def _service(args: argparse.Namespace) -> AtlasRagService:
    settings = Settings(data_dir=Path(args.data_dir))
    return AtlasRagService(settings)


def _filters(args: argparse.Namespace) -> SearchFilters | None:
    if not (args.document_id or args.title_contains or args.media_type):
        return None
    return SearchFilters(
        document_ids=list(args.document_id) or None,
        title_contains=args.title_contains,
        media_types=list(args.media_type) or None,
    )


def cmd_ingest(args: argparse.Namespace) -> int:
    service = _service(args)
    paths: list[Path] = []
    for raw in args.paths:
        path = Path(raw)
        paths.extend(sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path])
    for path in paths:
        result = service.ingest_path(path)
        detail = result.errors[0].message if result.errors else ""
        print(f"{result.status:<18} {path.name:<34} chunks={result.chunk_count} {detail}")
    docs, chunks = service.store.counts()
    print(f"\nregistry: {docs} documents, {chunks} chunks")
    service.close()
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    service = _service(args)
    for doc in service.list_documents():
        print(
            f"{doc.document_id[:12]}  {doc.title[:38]:<38} "
            f"{doc.media_type:<15} chunks={doc.chunk_count:<4} chars={doc.char_count}"
        )
    service.close()
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    service = _service(args)
    deleted = service.delete_document(args.document_id)
    print("deleted" if deleted else "not found")
    service.close()
    return 0 if deleted else 1


def cmd_search(args: argparse.Namespace) -> int:
    service = _service(args)
    query = SearchQuery(
        text=args.query, mode=args.mode, top_k=args.top_k, filters=_filters(args)
    )
    outcome = service.search(query)
    print(f"mode={query.mode}  hits={len(outcome.hits)}  "
          f"best_bm25={outcome.max_bm25_score}  best_cosine={outcome.max_cosine_score}\n")
    for hit in outcome.hits:
        contributions = "  ".join(
            f"{c.retriever}#{c.rank} raw={c.raw_score:.4f} rrf={c.rrf_term:.6f}"
            for c in hit.contributions
        )
        print(f"[{hit.fused_rank}] {hit.title}  fused={hit.fused_score:.6f}")
        print(f"     {contributions}")
        print(f"     {hit.chunk_id[:12]} [{hit.start_offset}:{hit.end_offset}]")
        print(f"     {hit.text[:160].replace(chr(10), ' ')}...\n")
    service.close()
    return 0


def cmd_answer(args: argparse.Namespace) -> int:
    service = _service(args)
    query = SearchQuery(
        text=args.query, mode=args.mode, top_k=args.top_k, filters=_filters(args)
    )
    response = service.answer(query)
    if response.answered:
        print(f"ANSWER ({response.provider}, mode={response.mode}):\n{response.text}\n")
        for citation in response.citations:
            mark = "valid" if citation.validated else "INVALID"
            print(
                f"  [{mark}] {citation.title} "
                f"[{citation.start_offset}:{citation.end_offset}] {citation.document_id[:12]}"
            )
    else:
        assert response.abstention is not None
        print(f"ABSTAINED: {response.abstention.reason.value}")
        print(f"  {response.abstention.explanation}")
        print(f"  evidence considered: {response.abstention.evidence_considered}")
    service.close()
    return 0


def cmd_rebuild(args: argparse.Namespace) -> int:
    service = _service(args)
    service.indexes.drop_persisted()
    service.indexes.rebuild_all()
    print(f"lexical: {service.indexes.bm25.doc_count} chunks, "
          f"{service.indexes.bm25.vocabulary_size} terms")
    print(f"dense:   {service.indexes.vectors.size} vectors, dim {service.indexes.vectors.dimension}")
    service.close()
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    service = _service(args)
    service.indexes.ensure_ready()
    docs, chunks = service.store.counts()
    payload = {
        "documents": docs,
        "chunks": chunks,
        "bm25_chunks": service.indexes.bm25.doc_count,
        "bm25_terms": service.indexes.bm25.vocabulary_size,
        "dense_ready": service.indexes.dense_ready,
        "dense_error": service.indexes.dense_error,
        "config_fingerprint": service.settings.fingerprint(),
        "embedding_model": service.settings.embedding_model,
        "embedding_revision": service.settings.embedding_revision,
    }
    if service.indexes.dense_ready:
        payload["dense_vectors"] = service.indexes.vectors.size
        payload["dense_dimension"] = service.indexes.vectors.dimension
    print(json.dumps(payload, indent=2))
    service.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="atlasrag", description="AtlasRAG command line")
    parser.add_argument("--data-dir", default="var", help="directory for the registry and indexes")
    parser.add_argument("--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="ingest files or directories")
    ingest.add_argument("paths", nargs="+")
    ingest.set_defaults(func=cmd_ingest)

    listing = sub.add_parser("list", help="list indexed documents")
    listing.set_defaults(func=cmd_list)

    delete = sub.add_parser("delete", help="delete a document by id")
    delete.add_argument("document_id")
    delete.set_defaults(func=cmd_delete)

    for name, handler, help_text in (
        ("search", cmd_search, "retrieve evidence"),
        ("answer", cmd_answer, "answer from evidence, or abstain"),
    ):
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("query")
        cmd.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
        cmd.add_argument("--top-k", type=int, default=5)
        cmd.add_argument("--document-id", action="append", default=[])
        cmd.add_argument("--title-contains", default=None)
        cmd.add_argument("--media-type", action="append", default=[])
        cmd.set_defaults(func=handler)

    rebuild = sub.add_parser("rebuild", help="drop and rebuild every index from the registry")
    rebuild.set_defaults(func=cmd_rebuild)

    stats = sub.add_parser("stats", help="print registry and index statistics")
    stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
