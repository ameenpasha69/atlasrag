"""Abstention policy.

Abstaining is a correct output, not a failure path, so the conditions are explicit,
thresholded, and calibrated on a held-out split (fixtures/eval/v1/calibration.jsonl)
rather than chosen by feel. See EVALUATION.md for the executed sweep.
"""

from __future__ import annotations

from dataclasses import dataclass

from atlasrag.domain.models import Abstention, AbstentionReason


@dataclass(frozen=True, slots=True)
class AbstentionThresholds:
    min_support: float
    """Lowest normalized query-term support a sentence may have and still be answerable."""

    min_bm25: float
    """Below this best-BM25 score, nothing lexically relevant was found at all."""

    min_cosine: float
    """Below this best cosine, nothing semantically relevant was found at all."""

    conflict_margin: float = 0.25
    """Two candidate answers this close in support are treated as equally credible."""


def no_evidence(considered: int = 0) -> Abstention:
    return Abstention(
        reason=AbstentionReason.NO_EVIDENCE,
        explanation="No indexed passage matched this question.",
        evidence_considered=considered,
    )


def out_of_scope(considered: int, best_bm25: float | None, best_cosine: float | None) -> Abstention:
    parts = []
    if best_bm25 is not None:
        parts.append(f"best BM25 {best_bm25:.3f}")
    if best_cosine is not None:
        parts.append(f"best cosine {best_cosine:.3f}")
    detail = f" ({', '.join(parts)})" if parts else ""
    return Abstention(
        reason=AbstentionReason.OUT_OF_SCOPE,
        explanation=(
            "The corpus does not appear to cover this question; every retrieved passage "
            f"scored below the relevance floor{detail}."
        ),
        evidence_considered=considered,
    )


def below_threshold(considered: int, best_support: float, min_support: float) -> Abstention:
    return Abstention(
        reason=AbstentionReason.BELOW_THRESHOLD,
        explanation=(
            f"Retrieved passages were related but too weakly supporting to answer from "
            f"(support {best_support:.2f} < {min_support:.2f})."
        ),
        evidence_considered=considered,
    )


def conflicting(considered: int, detail: str) -> Abstention:
    return Abstention(
        reason=AbstentionReason.CONFLICTING_EVIDENCE,
        explanation=("Sources disagree, so no single answer is reported. " + detail),
        evidence_considered=considered,
    )


def citation_failed(considered: int, detail: str) -> Abstention:
    return Abstention(
        reason=AbstentionReason.CITATION_VALIDATION_FAILED,
        explanation=(
            "A candidate answer was produced but its citation did not match the stored "
            f"source text, so it was discarded. {detail}"
        ),
        evidence_considered=considered,
    )
