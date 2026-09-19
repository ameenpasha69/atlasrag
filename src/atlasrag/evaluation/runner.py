"""Executes evaluation runs. Every number this module produces comes from a live query."""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime

from atlasrag.answering.abstention import AbstentionThresholds
from atlasrag.answering.evidence import build_evidence
from atlasrag.answering.extractive import ExtractiveAnswerProvider
from atlasrag.answering.text_spans import sentence_spans
from atlasrag.domain.models import (
    EvaluationQuery,
    EvaluationRun,
    FailureRecord,
    MetricBundle,
    SearchMode,
    SearchQuery,
)
from atlasrag.evaluation.dataset import EvaluationDataset
from atlasrag.evaluation.metrics import (
    binary_classification,
    dedupe_documents,
    mean,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from atlasrag.service import AtlasRagService


@dataclass(slots=True)
class _Accumulator:
    recall: list[float] = field(default_factory=list)
    precision: list[float] = field(default_factory=list)
    rr: list[float] = field(default_factory=list)
    ndcg: list[float] = field(default_factory=list)

    def add(self, recall: float, precision: float, rr: float, ndcg: float) -> None:
        self.recall.append(recall)
        self.precision.append(precision)
        self.rr.append(rr)
        self.ndcg.append(ndcg)

    def bundle(self, k: int) -> MetricBundle:
        return MetricBundle(
            query_count=len(self.recall),
            recall_at_k=mean(self.recall),
            precision_at_k=mean(self.precision),
            mrr=mean(self.rr),
            ndcg_at_k=mean(self.ndcg),
            k=k,
        )


@dataclass(frozen=True, slots=True)
class AnsweringOutcome:
    abstention_precision: float
    abstention_recall: float
    abstention_f1: float
    citation_validity_rate: float
    unsupported_claim_rate: float
    decisions: dict[str, bool]
    """query_id -> abstained"""


def _to_search_query(query: EvaluationQuery, mode: SearchMode, k: int) -> SearchQuery:
    return SearchQuery(text=query.text, mode=mode, top_k=k, filters=query.filters)


def run_retrieval(
    service: AtlasRagService, dataset: EvaluationDataset, mode: SearchMode, k: int = 5
) -> tuple[MetricBundle, dict[str, MetricBundle], list[FailureRecord]]:
    overall = _Accumulator()
    by_category: dict[str, _Accumulator] = defaultdict(_Accumulator)
    failures: list[FailureRecord] = []

    for query in dataset.queries:
        relevant = dataset.relevant(query.query_id)
        if not relevant:
            continue  # unanswerable queries have no retrieval ground truth

        outcome = service.search(_to_search_query(query, mode, k))
        ranked = dedupe_documents([hit.document_id for hit in outcome.hits])

        recall = recall_at_k(ranked, relevant, k)
        precision = precision_at_k(ranked, relevant, k)
        rr = reciprocal_rank(ranked, relevant)
        ndcg = ndcg_at_k(ranked, relevant, k)

        overall.add(recall, precision, rr, ndcg)
        by_category[query.category].add(recall, precision, rr, ndcg)

        if recall == 0.0:
            failures.append(
                FailureRecord(
                    query_id=query.query_id,
                    category=query.category,
                    query_text=query.text,
                    expected_document_ids=sorted(relevant),
                    retrieved_document_ids=ranked,
                    likely_cause=(
                        "no results returned"
                        if not ranked
                        else "retrieved other documents; relevant document ranked below k"
                    ),
                )
            )

    return (
        overall.bundle(k),
        {name: acc.bundle(k) for name, acc in sorted(by_category.items())},
        failures,
    )


def run_answering(
    service: AtlasRagService,
    dataset: EvaluationDataset,
    mode: SearchMode,
    *,
    k: int = 5,
    thresholds: AbstentionThresholds | None = None,
) -> AnsweringOutcome:
    provider = (
        ExtractiveAnswerProvider(
            service.store,
            idf=lambda term: service.indexes.bm25.idf(term),
            oov_idf=lambda: service.indexes.bm25.oov_idf,
            thresholds=thresholds,
        )
        if thresholds is not None
        else service.answerer
    )

    true_positive = false_positive = false_negative = 0
    citations_total = citations_valid = 0
    sentences_total = sentences_unsupported = 0
    decisions: dict[str, bool] = {}

    for query in dataset.queries:
        search_query = _to_search_query(query, mode, k)
        outcome = service.search(search_query)
        evidence = build_evidence(query.text, outcome, top_n=service.settings.evidence_top_n)
        response = provider.answer(search_query, evidence)

        abstained = not response.answered
        decisions[query.query_id] = abstained

        if abstained and not query.expected_answerable:
            true_positive += 1
        elif abstained and query.expected_answerable:
            false_positive += 1
        elif not abstained and not query.expected_answerable:
            false_negative += 1

        citations_total += len(response.citations)
        citations_valid += sum(1 for c in response.citations if c.validated)

        if response.answered and response.text:
            snippets = [c.snippet for c in response.citations if c.validated]
            for start, end in sentence_spans(response.text):
                sentence = response.text[start:end]
                sentences_total += 1
                if not any(sentence in snippet or snippet in sentence for snippet in snippets):
                    sentences_unsupported += 1

    precision, recall, f1 = binary_classification(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
    )
    return AnsweringOutcome(
        abstention_precision=precision,
        abstention_recall=recall,
        abstention_f1=f1,
        citation_validity_rate=(citations_valid / citations_total) if citations_total else 1.0,
        unsupported_claim_rate=(
            sentences_unsupported / sentences_total if sentences_total else 0.0
        ),
        decisions=decisions,
    )


def build_run(
    service: AtlasRagService,
    dataset: EvaluationDataset,
    mode: SearchMode,
    *,
    k: int = 5,
) -> EvaluationRun:
    overall, per_category, failures = run_retrieval(service, dataset, mode, k)
    answering = run_answering(service, dataset, mode, k=k)
    return EvaluationRun(
        run_id=uuid.uuid4().hex[:12],
        started_at_utc=datetime.now(UTC),
        dataset_version=f"{dataset.version}/{dataset.split}",
        mode=mode,
        config_fingerprint=service.settings.fingerprint(),
        overall=overall,
        per_category=per_category,
        abstention_precision=answering.abstention_precision,
        abstention_recall=answering.abstention_recall,
        citation_validity_rate=answering.citation_validity_rate,
        unsupported_claim_rate=answering.unsupported_claim_rate,
        failures=failures,
    )
