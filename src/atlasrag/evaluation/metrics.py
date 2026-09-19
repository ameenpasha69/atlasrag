"""Retrieval and answering metrics.

Metrics are computed at document granularity: a query's judgments name documents, and a
ranked list of chunks is reduced to its first occurrence of each document. Chunk-level
judgments would be more precise but far more brittle to maintain, because they change
whenever the chunker configuration changes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def dedupe_documents(document_ids: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for document_id in document_ids:
        if document_id not in seen:
            seen.add(document_id)
            ordered.append(document_id)
    return ordered


def recall_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    positives = {d for d, g in relevant.items() if g > 0}
    if not positives:
        return 0.0
    return len(set(ranked[:k]) & positives) / len(positives)


def precision_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    positives = {d for d, g in relevant.items() if g > 0}
    return len(set(ranked[:k]) & positives) / k


def reciprocal_rank(ranked: Sequence[str], relevant: Mapping[str, int]) -> float:
    positives = {d for d, g in relevant.items() if g > 0}
    for position, document_id in enumerate(ranked, start=1):
        if document_id in positives:
            return 1.0 / position
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    def gain(grade: int) -> float:
        return (2.0**grade) - 1.0

    dcg = sum(
        gain(relevant.get(document_id, 0)) / math.log2(position + 1)
        for position, document_id in enumerate(ranked[:k], start=1)
    )
    ideal_grades = sorted((g for g in relevant.values() if g > 0), reverse=True)[:k]
    idcg = sum(
        gain(grade) / math.log2(position + 1)
        for position, grade in enumerate(ideal_grades, start=1)
    )
    return (dcg / idcg) if idcg > 0 else 0.0


def binary_classification(
    *, true_positive: int, false_positive: int, false_negative: int
) -> tuple[float, float, float]:
    """Return (precision, recall, f1). Undefined ratios are reported as 0.0."""
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive) > 0
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
