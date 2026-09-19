"""RRF arithmetic checked against values computed by hand.

The expected numbers below were derived on paper from RRF(d) = sum 1/(k + rank_r(d)) with
k = 60, not copied from program output.
"""

from __future__ import annotations

import random

import pytest

from atlasrag.retrieval.fusion import RetrieverResult, reciprocal_rank_fusion

# 1/61, 1/62, 1/63 written out so the arithmetic below is inspectable.
R61 = 1.0 / 61.0
R62 = 1.0 / 62.0
R63 = 1.0 / 63.0

TOL = 1e-12


def _bm25() -> RetrieverResult:
    return RetrieverResult(name="bm25", ranked=[("c1", 9.0), ("c2", 5.0), ("c3", 2.0)])


def _dense() -> RetrieverResult:
    return RetrieverResult(name="dense", ranked=[("c3", 0.9), ("c1", 0.8), ("c4", 0.7)])


def test_rrf_scores_match_hand_computation() -> None:
    fused = reciprocal_rank_fusion([_bm25(), _dense()], k=60, top_k=10)
    scores = {f.chunk_id: f.fused_score for f in fused}

    assert scores["c1"] == pytest.approx(R61 + R62, abs=TOL)  # bm25 #1, dense #2
    assert scores["c3"] == pytest.approx(R63 + R61, abs=TOL)  # bm25 #3, dense #1
    assert scores["c2"] == pytest.approx(R62, abs=TOL)  # bm25 #2 only
    assert scores["c4"] == pytest.approx(R63, abs=TOL)  # dense #3 only


def test_rrf_ordering_and_ranks() -> None:
    fused = reciprocal_rank_fusion([_bm25(), _dense()], k=60, top_k=10)
    assert [f.chunk_id for f in fused] == ["c1", "c3", "c2", "c4"]
    assert [f.fused_rank for f in fused] == [1, 2, 3, 4]


def test_a_chunk_found_by_both_beats_a_chunk_found_once_at_a_better_rank() -> None:
    """c1 (ranks 1 and 2) must beat c2, which BM25 alone put at rank 2."""
    fused = reciprocal_rank_fusion([_bm25(), _dense()], k=60, top_k=10)
    order = [f.chunk_id for f in fused]
    assert order.index("c1") < order.index("c2")


def test_contributions_expose_rank_raw_score_and_term() -> None:
    fused = reciprocal_rank_fusion([_bm25(), _dense()], k=60, top_k=10)
    c1 = next(f for f in fused if f.chunk_id == "c1")
    by_retriever = {c.retriever: c for c in c1.contributions}

    assert by_retriever["bm25"].rank == 1
    assert by_retriever["bm25"].raw_score == 9.0
    assert by_retriever["bm25"].rrf_term == pytest.approx(R61, abs=TOL)
    assert by_retriever["dense"].rank == 2
    assert by_retriever["dense"].raw_score == 0.8
    assert by_retriever["dense"].rrf_term == pytest.approx(R62, abs=TOL)
    assert sum(c.rrf_term for c in c1.contributions) == pytest.approx(c1.fused_score, abs=TOL)


def test_ties_break_on_chunk_id() -> None:
    fused = reciprocal_rank_fusion(
        [
            RetrieverResult(name="bm25", ranked=[("zz", 5.0)]),
            RetrieverResult(name="dense", ranked=[("aa", 0.5)]),
        ],
        k=60,
        top_k=10,
    )
    assert [f.fused_score for f in fused] == pytest.approx([R61, R61], abs=TOL)
    assert [f.chunk_id for f in fused] == ["aa", "zz"]


def test_ordering_is_stable_across_retriever_argument_order() -> None:
    baseline = [f.chunk_id for f in reciprocal_rank_fusion([_bm25(), _dense()], k=60, top_k=10)]
    rng = random.Random(1234)
    for _ in range(50):
        pair = [_bm25(), _dense()]
        rng.shuffle(pair)
        assert [f.chunk_id for f in reciprocal_rank_fusion(pair, k=60, top_k=10)] == baseline


def test_k_changes_the_flattening_not_the_single_list_order() -> None:
    for k in (1, 10, 60, 500):
        fused = reciprocal_rank_fusion([_bm25()], k=k, top_k=10)
        assert [f.chunk_id for f in fused] == ["c1", "c2", "c3"]
        assert fused[0].fused_score == pytest.approx(1.0 / (k + 1), abs=TOL)


def test_duplicate_chunk_from_one_retriever_is_counted_once() -> None:
    fused = reciprocal_rank_fusion(
        [RetrieverResult(name="bm25", ranked=[("c1", 9.0), ("c1", 8.0), ("c2", 1.0)])],
        k=60,
        top_k=10,
    )
    assert [f.chunk_id for f in fused] == ["c1", "c2"]
    assert fused[0].fused_score == pytest.approx(R61, abs=TOL)
    assert fused[1].fused_score == pytest.approx(R62, abs=TOL)


def test_weights_scale_contributions() -> None:
    fused = reciprocal_rank_fusion([_bm25(), _dense()], k=60, top_k=10, weights={"dense": 2.0})
    scores = {f.chunk_id: f.fused_score for f in fused}
    assert scores["c3"] == pytest.approx(R63 + 2.0 * R61, abs=TOL)
    assert scores["c1"] == pytest.approx(R61 + 2.0 * R62, abs=TOL)


def test_rejects_invalid_k() -> None:
    with pytest.raises(ValueError, match="rrf k"):
        reciprocal_rank_fusion([_bm25()], k=0, top_k=5)
