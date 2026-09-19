"""Markdown rendering of executed evaluation runs."""

from __future__ import annotations

from collections.abc import Sequence

from atlasrag.domain.models import EvaluationRun, MetricBundle


def _row(name: str, bundle: MetricBundle) -> str:
    return (
        f"| {name} | {bundle.query_count} | {bundle.recall_at_k:.3f} | "
        f"{bundle.precision_at_k:.3f} | {bundle.mrr:.3f} | {bundle.ndcg_at_k:.3f} |"
    )


def comparison_table(runs: Sequence[EvaluationRun]) -> str:
    k = runs[0].overall.k if runs else 5
    lines = [
        f"| mode | queries | Recall@{k} | Precision@{k} | MRR | nDCG@{k} |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(_row(run.mode, run.overall) for run in runs)
    return "\n".join(lines)


def per_category_table(run: EvaluationRun) -> str:
    k = run.overall.k
    lines = [
        f"| category | queries | Recall@{k} | Precision@{k} | MRR | nDCG@{k} |",
        "|---|---|---|---|---|---|",
    ]
    lines.extend(_row(name, bundle) for name, bundle in run.per_category.items())
    lines.append(_row("**overall**", run.overall))
    return "\n".join(lines)


def answering_table(runs: Sequence[EvaluationRun]) -> str:
    lines = [
        "| mode | abstention precision | abstention recall | citation validity | "
        "unsupported claims |",
        "|---|---|---|---|---|",
    ]
    for run in runs:
        lines.append(
            f"| {run.mode} | {run.abstention_precision:.3f} | {run.abstention_recall:.3f} | "
            f"{run.citation_validity_rate:.3f} | {run.unsupported_claim_rate:.3f} |"
        )
    return "\n".join(lines)


def failure_table(run: EvaluationRun, id_to_title: dict[str, str]) -> str:
    if not run.failures:
        return "_No retrieval failures in this run._"
    lines = [
        "| query | category | expected | retrieved | likely cause |",
        "|---|---|---|---|---|",
    ]
    for failure in run.failures:
        expected = ", ".join(id_to_title.get(d, d[:8]) for d in failure.expected_document_ids)
        retrieved = (
            ", ".join(id_to_title.get(d, d[:8]) for d in failure.retrieved_document_ids[:3])
            or "(nothing)"
        )
        lines.append(
            f"| {failure.query_id}: {failure.query_text} | {failure.category} | "
            f"{expected} | {retrieved} | {failure.likely_cause} |"
        )
    return "\n".join(lines)
