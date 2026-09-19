"""Abstention threshold calibration.

Thresholds are swept on the calibration split only. Reporting them on the same queries that
chose them would be self-congratulation, so `atlasrag evaluate` reports on the test split,
which the sweep never sees.

Retrieval output does not depend on the thresholds, so evidence is computed once per query
and reused across the whole grid. That makes an exhaustive sweep cheap and, more importantly,
guarantees every grid point is scored against byte-identical evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from atlasrag.answering.abstention import AbstentionThresholds
from atlasrag.answering.evidence import build_evidence
from atlasrag.answering.extractive import ExtractiveAnswerProvider
from atlasrag.domain.models import EvidenceSet, SearchMode, SearchQuery
from atlasrag.evaluation.dataset import EvaluationDataset
from atlasrag.evaluation.metrics import binary_classification
from atlasrag.service import AtlasRagService

SUPPORT_GRID = (0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
BM25_GRID = (0.0, 0.5, 1.0, 2.0)
COSINE_GRID = (0.0, 0.45, 0.50, 0.55, 0.60, 0.65)


@dataclass(frozen=True, slots=True)
class GridPoint:
    thresholds: AbstentionThresholds
    precision: float
    recall: float
    f1: float
    false_abstentions: int
    missed_abstentions: int


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    best: GridPoint
    evaluated: int
    top: list[GridPoint]


def _cache_evidence(
    service: AtlasRagService, dataset: EvaluationDataset, mode: SearchMode, k: int
) -> list[tuple[SearchQuery, EvidenceSet, bool]]:
    cached: list[tuple[SearchQuery, EvidenceSet, bool]] = []
    for query in dataset.queries:
        search_query = SearchQuery(text=query.text, mode=mode, top_k=k, filters=query.filters)
        outcome = service.search(search_query)
        evidence = build_evidence(query.text, outcome, top_n=service.settings.evidence_top_n)
        cached.append((search_query, evidence, query.expected_answerable))
    return cached


def _score(
    service: AtlasRagService,
    cached: list[tuple[SearchQuery, EvidenceSet, bool]],
    thresholds: AbstentionThresholds,
) -> GridPoint:
    provider = ExtractiveAnswerProvider(
        service.store,
        idf=lambda term: service.indexes.bm25.idf(term),
        oov_idf=lambda: service.indexes.bm25.oov_idf,
        thresholds=thresholds,
    )
    true_positive = false_positive = false_negative = 0
    for search_query, evidence, expected_answerable in cached:
        abstained = not provider.answer(search_query, evidence).answered
        if abstained and not expected_answerable:
            true_positive += 1
        elif abstained and expected_answerable:
            false_positive += 1
        elif not abstained and not expected_answerable:
            false_negative += 1

    precision, recall, f1 = binary_classification(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )
    return GridPoint(
        thresholds=thresholds,
        precision=precision,
        recall=recall,
        f1=f1,
        false_abstentions=false_positive,
        missed_abstentions=false_negative,
    )


def calibrate(
    service: AtlasRagService,
    dataset: EvaluationDataset,
    mode: SearchMode = "hybrid",
    *,
    k: int = 5,
) -> CalibrationResult:
    cached = _cache_evidence(service, dataset, mode, k)
    points: list[GridPoint] = [
        _score(
            service,
            cached,
            AbstentionThresholds(min_support=support, min_bm25=bm25, min_cosine=cosine),
        )
        for support in SUPPORT_GRID
        for bm25 in BM25_GRID
        for cosine in COSINE_GRID
    ]

    # Highest F1 wins. Ties break toward the most permissive thresholds, because an
    # unnecessary abstention is a silent failure to answer a question the corpus covers.
    points.sort(
        key=lambda p: (
            -p.f1,
            p.thresholds.min_support,
            p.thresholds.min_bm25,
            p.thresholds.min_cosine,
        )
    )
    return CalibrationResult(best=points[0], evaluated=len(points), top=points[:10])
