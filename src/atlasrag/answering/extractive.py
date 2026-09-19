"""Deterministic extractive answering.

This is the baseline that proves retrieval, citation validation and abstention work with no
language model involved at all. It never writes a new sentence: every sentence it returns is
copied verbatim from a source span, which is why its citations are exact by construction.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from atlasrag.answering import abstention as abstain
from atlasrag.answering.citations import build_citation
from atlasrag.answering.injection import looks_like_injected_instruction
from atlasrag.answering.stopwords import content_terms
from atlasrag.answering.text_spans import sentence_spans
from atlasrag.domain.models import (
    AnswerResponse,
    Citation,
    EvidenceSet,
    SearchHit,
    SearchQuery,
)
from atlasrag.indexing.lexical.bm25 import tokenize
from atlasrag.storage.sqlite_store import SqliteDocumentStore

_VALUE = re.compile(
    r"\b\d[\d,]*(?:\.\d+)?\s*"
    r"(?:%|percent|ms|milliseconds?|secs?|seconds?|mins?|minutes?|hours?|days?|"
    r"weeks?|months?|years?|[kmgt]?b)?",
    re.IGNORECASE,
)

MAX_ANSWER_SENTENCES = 3
CONFLICT_SCAN_DEPTH = 8
SUPPORT_COMPANION_MARGIN = 0.15
MIN_SENTENCE_TOKENS = 5

_HEADING = re.compile(r"^#{1,6}\s")


def _is_quotable(sentence: str) -> bool:
    """Only factual assertions may be quoted as an answer.

    Headings and fragments are navigational. Instruction-shaped text is an imperative aimed
    at an assistant, not a claim about the world, and quoting it would let an untrusted
    document put words in the system's mouth.
    """
    if _HEADING.match(sentence):
        return False
    if looks_like_injected_instruction(sentence):
        return False
    return len(tokenize(sentence)) >= MIN_SENTENCE_TOKENS


@dataclass(frozen=True, slots=True)
class _Candidate:
    hit: SearchHit
    doc_start: int
    doc_end: int
    sentence: str
    support: float
    values: frozenset[str]


def _normalize_value(raw: str) -> str:
    return re.sub(r"[\s,]", "", raw).casefold()


def _extract_values(text: str, *, ignore: frozenset[str]) -> frozenset[str]:
    found = {_normalize_value(m.group(0)) for m in _VALUE.finditer(text)}
    return frozenset(v for v in found if v and v not in ignore)


class ExtractiveAnswerProvider:
    """Answers by selecting supporting sentences from retrieved chunks."""

    def __init__(
        self,
        store: SqliteDocumentStore,
        *,
        idf: Callable[[str], float],
        oov_idf: Callable[[], float],
        thresholds: abstain.AbstentionThresholds,
        max_sentences: int = MAX_ANSWER_SENTENCES,
    ) -> None:
        self._store = store
        self._idf = idf
        self._oov_idf = oov_idf
        self._thresholds = thresholds
        self._max_sentences = max_sentences

    @property
    def name(self) -> str:
        return "extractive"

    def _term_weights(self, query_text: str) -> dict[str, float]:
        """IDF weight per content term, with out-of-vocabulary terms pinned at the ceiling.

        Stopwords are dropped so a sentence cannot look responsive merely by containing
        "how" and "are". Unknown terms keep their full weight in the denominator, so a
        question about something the corpus never mentions cannot reach a high support
        score no matter which sentence is compared against it.
        """
        terms = content_terms(set(tokenize(query_text)))
        ceiling = self._oov_idf()
        weights: dict[str, float] = {}
        for term in terms:
            observed = self._idf(term)
            weights[term] = observed if observed > 0.0 else ceiling
        return weights

    def _candidates(self, query: SearchQuery, hits: list[SearchHit]) -> list[_Candidate]:
        weights = self._term_weights(query.text)
        total = sum(weights.values())
        query_values = _extract_values(query.text, ignore=frozenset())

        candidates: list[_Candidate] = []
        for hit in hits:
            for start, end in sentence_spans(hit.text):
                sentence = hit.text[start:end]
                if not _is_quotable(sentence):
                    continue
                present = set(tokenize(sentence))
                matched = sum(w for term, w in weights.items() if term in present)
                support = (matched / total) if total > 0 else 0.0
                candidates.append(
                    _Candidate(
                        hit=hit,
                        doc_start=hit.start_offset + start,
                        doc_end=hit.start_offset + end,
                        sentence=sentence,
                        support=support,
                        values=_extract_values(sentence, ignore=query_values),
                    )
                )
        # Deterministic: best support first, then earliest fused rank, then document position.
        candidates.sort(key=lambda c: (-c.support, c.hit.fused_rank, c.doc_start, c.hit.chunk_id))
        return candidates

    def _detect_conflict(self, candidates: list[_Candidate]) -> tuple[bool, str | None]:
        """Look for two comparably-supported sentences from different documents that assert
        different values. Scans the whole credible band, not just the single best sentence:
        the top sentence is often a framing sentence carrying no value at all."""
        credible = [
            c for c in candidates if c.support >= self._thresholds.min_support and c.values
        ][:CONFLICT_SCAN_DEPTH]
        if len(credible) < 2:
            return False, None

        anchor = credible[0]
        for other in credible[1:]:
            if other.hit.document_id == anchor.hit.document_id:
                continue
            if anchor.support - other.support > self._thresholds.conflict_margin:
                break
            if other.values.isdisjoint(anchor.values):
                detail = (
                    f"'{anchor.hit.title}' states {sorted(anchor.values)} while "
                    f"'{other.hit.title}' states {sorted(other.values)}."
                )
                return True, detail
        return False, None

    def answer(self, query: SearchQuery, evidence: EvidenceSet) -> AnswerResponse:
        hits = evidence.selected
        considered = len(hits)

        if not hits:
            return self._abstain(query, evidence, abstain.no_evidence(considered))

        best_bm25 = evidence.max_bm25_score
        best_cosine = evidence.max_cosine_score
        lexically_empty = best_bm25 is None or best_bm25 < self._thresholds.min_bm25
        semantically_empty = best_cosine is None or best_cosine < self._thresholds.min_cosine
        if lexically_empty and semantically_empty:
            return self._abstain(
                query, evidence, abstain.out_of_scope(considered, best_bm25, best_cosine)
            )

        candidates = self._candidates(query, hits)
        if not candidates:
            return self._abstain(query, evidence, abstain.no_evidence(considered))

        conflict, detail = self._detect_conflict(candidates)
        if conflict and detail is not None:
            return self._abstain(
                query,
                evidence,
                abstain.conflicting(considered, detail),
                conflict=True,
                conflict_note=detail,
            )

        best = candidates[0]
        if best.support < self._thresholds.min_support:
            return self._abstain(
                query,
                evidence,
                abstain.below_threshold(considered, best.support, self._thresholds.min_support),
            )

        selected: list[_Candidate] = [best]
        seen_spans = {(best.hit.document_id, best.doc_start)}
        for candidate in candidates[1:]:
            if len(selected) >= self._max_sentences:
                break
            # Only sentences nearly as well supported as the best one may join it; otherwise
            # the answer becomes a grab-bag of loosely related text.
            if candidate.support < self._thresholds.min_support:
                break
            if best.support - candidate.support > SUPPORT_COMPANION_MARGIN:
                break
            key = (candidate.hit.document_id, candidate.doc_start)
            if key in seen_spans:
                continue
            seen_spans.add(key)
            selected.append(candidate)

        citations: list[Citation] = [
            build_citation(
                self._store,
                document_id=c.hit.document_id,
                chunk_id=c.hit.chunk_id,
                title=c.hit.title,
                start=c.doc_start,
                end=c.doc_end,
                snippet=c.sentence,
                provenance=c.hit.contributions,
            )
            for c in selected
        ]

        invalid = [c for c in citations if not c.validated]
        if invalid:
            detail = (
                f"{len(invalid)} of {len(citations)} citations failed span verification "
                f"(first: {invalid[0].document_id[:12]} "
                f"[{invalid[0].start_offset}:{invalid[0].end_offset}])."
            )
            return self._abstain(query, evidence, abstain.citation_failed(considered, detail))

        text = " ".join(c.sentence for c in selected)
        return AnswerResponse(
            answered=True,
            text=text,
            citations=citations,
            abstention=None,
            evidence=evidence,
            provider=self.name,
            mode=query.mode,
            unsupported_sentences=[],
        )

    def _abstain(
        self,
        query: SearchQuery,
        evidence: EvidenceSet,
        abstention: object,
        *,
        conflict: bool = False,
        conflict_note: str | None = None,
    ) -> AnswerResponse:
        updated = (
            evidence.model_copy(update={"conflict_detected": True, "conflict_note": conflict_note})
            if conflict
            else evidence
        )
        return AnswerResponse(
            answered=False,
            text=None,
            citations=[],
            abstention=abstention,  # type: ignore[arg-type]
            evidence=updated,
            provider=self.name,
            mode=query.mode,
        )
