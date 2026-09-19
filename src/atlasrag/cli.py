"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from atlasrag.config import Settings
from atlasrag.domain.models import SearchFilters, SearchMode, SearchQuery
from atlasrag.service import AtlasRagService

if TYPE_CHECKING:
    from atlasrag.evaluation.dataset import EvaluationDataset


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
    query = SearchQuery(text=args.query, mode=args.mode, top_k=args.top_k, filters=_filters(args))
    outcome = service.search(query)
    print(
        f"mode={query.mode}  hits={len(outcome.hits)}  "
        f"best_bm25={outcome.max_bm25_score}  best_cosine={outcome.max_cosine_score}\n"
    )
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
    query = SearchQuery(text=args.query, mode=args.mode, top_k=args.top_k, filters=_filters(args))
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
    print(
        f"lexical: {service.indexes.bm25.doc_count} chunks, "
        f"{service.indexes.bm25.vocabulary_size} terms"
    )
    print(
        f"dense:   {service.indexes.vectors.size} vectors, dim {service.indexes.vectors.dimension}"
    )
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


def _load_dataset(service: AtlasRagService, split: str, version: str = "v2") -> EvaluationDataset:
    from atlasrag.evaluation.dataset import load_dataset

    mapping = {d.original_filename: d.document_id for d in service.store.list_documents()}
    return load_dataset(
        Path("fixtures/eval") / version, split, source_to_document_id=mapping, version=version
    )


def cmd_evaluate(args: argparse.Namespace) -> int:
    from atlasrag.evaluation.report import (
        answering_table,
        comparison_table,
        failure_table,
        per_category_table,
    )
    from atlasrag.evaluation.runner import build_run

    service = _service(args)
    service.indexes.ensure_ready()
    dataset = _load_dataset(service, args.split, args.dataset_version)
    titles = {d.document_id: d.title for d in service.store.list_documents()}

    modes: tuple[SearchMode, ...] = ("bm25", "dense", "hybrid")
    runs = [build_run(service, dataset, mode, k=args.k) for mode in modes]

    print(f"dataset: {dataset.version}/{dataset.split}  queries: {len(dataset.queries)}")
    print(f"config fingerprint: {service.settings.fingerprint()}")
    print(
        f"thresholds: support>={service.thresholds.min_support} "
        f"bm25>={service.thresholds.min_bm25} cosine>={service.thresholds.min_cosine}\n"
    )
    print("## Retrieval: mode comparison\n")
    print(comparison_table(runs))
    print("\n## Answering\n")
    print(answering_table(runs))
    for run in runs:
        print(f"\n## Per category: {run.mode}\n")
        print(per_category_table(run))
    hybrid = next(r for r in runs if r.mode == "hybrid")
    print("\n## Retrieval failures (hybrid)\n")
    print(failure_table(hybrid, titles))

    from atlasrag.evaluation.runner import run_answering

    decisions = run_answering(service, dataset, "hybrid", k=args.k).decisions
    print("\n## Abstention decisions (hybrid)\n")
    print("| query | category | expected | actual | verdict |")
    print("|---|---|---|---|---|")
    for query in dataset.queries:
        abstained = decisions[query.query_id]
        expected_abstain = not query.expected_answerable
        verdict = (
            "correct"
            if abstained == expected_abstain
            else ("FALSE ABSTENTION" if abstained else "MISSED ABSTENTION")
        )
        print(
            f"| {query.query_id}: {query.text[:44]} | {query.category} | "
            f"{'abstain' if expected_abstain else 'answer'} | "
            f"{'abstain' if abstained else 'answer'} | {verdict} |"
        )

    if args.json:
        Path(args.json).write_text(
            json.dumps([r.model_dump(mode="json") for r in runs], indent=2), encoding="utf-8"
        )
        print(f"\nwrote {args.json}")
    service.close()
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from atlasrag.evaluation.calibration import calibrate

    service = _service(args)
    service.indexes.ensure_ready()
    dataset = _load_dataset(service, "calibration", args.dataset_version)
    result = calibrate(service, dataset, args.mode, k=args.k)

    print(
        f"swept {result.evaluated} threshold combinations on "
        f"{dataset.version}/calibration ({len(dataset.queries)} queries), mode={args.mode}\n"
    )
    print(
        "| rank | min_support | min_bm25 | min_cosine | precision | recall | F1 | "
        "false abstentions | missed abstentions |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for rank, point in enumerate(result.top, start=1):
        t = point.thresholds
        print(
            f"| {rank} | {t.min_support:.2f} | {t.min_bm25:.2f} | {t.min_cosine:.2f} | "
            f"{point.precision:.3f} | {point.recall:.3f} | {point.f1:.3f} | "
            f"{point.false_abstentions} | {point.missed_abstentions} |"
        )
    best = result.best.thresholds
    print("\nselected:")
    print(f"  ATLASRAG_ABSTAIN_MIN_SUPPORT={best.min_support}")
    print(f"  ATLASRAG_ABSTAIN_MIN_BM25_SCORE={best.min_bm25}")
    print(f"  ATLASRAG_ABSTAIN_MIN_COSINE={best.min_cosine}")
    service.close()
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Per-query win/loss table across retrieval modes.

    Aggregate metrics can hide the fact that two modes fail *different* queries while failing
    the same *number* of them, which looks like equivalence and is not.
    """
    from atlasrag.evaluation.metrics import dedupe_documents, reciprocal_rank

    service = _service(args)
    service.indexes.ensure_ready()
    dataset = _load_dataset(service, args.split, args.dataset_version)
    modes: tuple[SearchMode, ...] = ("bm25", "dense", "hybrid")

    print(f"dataset: {dataset.version}/{dataset.split}  k={args.k}")
    print("\nreciprocal rank per query (1.000 = correct document at rank 1, 0 = missed)\n")
    print("| query | category | bm25 | dense | hybrid | note |")
    print("|---|---|---|---|---|---|")

    totals = dict.fromkeys(modes, 0.0)
    counted = 0
    for query in dataset.queries:
        relevant = dataset.relevant(query.query_id)
        if not relevant:
            continue
        counted += 1
        scores: dict[str, float] = {}
        for mode in modes:
            outcome = service.search(
                SearchQuery(text=query.text, mode=mode, top_k=args.k, filters=query.filters)
            )
            ranked = dedupe_documents([h.document_id for h in outcome.hits])
            scores[mode] = reciprocal_rank(ranked, relevant)
            totals[mode] += scores[mode]

        best = max(scores.values())
        worst = min(scores.values())
        note = ""
        if best > worst:
            winners = [m for m, v in scores.items() if v == best and m != "hybrid"]
            note = "split: " + ", ".join(winners) + " ahead"
        print(
            f"| {query.query_id}: {query.text[:40]} | {query.category} | "
            + " | ".join(f"{scores[m]:.3f}" for m in modes)
            + f" | {note} |"
        )

    print(
        "| **mean MRR** | | " + " | ".join(f"**{totals[m] / counted:.3f}**" for m in modes) + " | |"
    )
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

    evaluate = sub.add_parser("evaluate", help="run the evaluation dataset and report metrics")
    evaluate.add_argument("--split", choices=["test", "calibration"], default="test")
    evaluate.add_argument("--dataset-version", default="v2")
    evaluate.add_argument("--k", type=int, default=5)
    evaluate.add_argument("--json", default=None, help="also write raw runs to this path")
    evaluate.set_defaults(func=cmd_evaluate)

    calibrate = sub.add_parser(
        "calibrate", help="sweep abstention thresholds on the calibration split"
    )
    calibrate.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="hybrid")
    calibrate.add_argument("--dataset-version", default="v2")
    calibrate.add_argument("--k", type=int, default=5)
    calibrate.set_defaults(func=cmd_calibrate)

    compare = sub.add_parser("compare", help="per-query win/loss table across modes")
    compare.add_argument("--split", choices=["test", "calibration"], default="test")
    compare.add_argument("--dataset-version", default="v2")
    compare.add_argument("--k", type=int, default=3)
    compare.set_defaults(func=cmd_compare)

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
