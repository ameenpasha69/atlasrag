"""BM25 scoring checked against values computed by hand.

Worked example, derived on paper, for the three-chunk corpus built below:

    c1 = "the quick brown fox"  -> 4 tokens
    c2 = "the lazy dog"         -> 3 tokens
    c3 = "quick quick fox"      -> 3 tokens
    N  = 3,  avgdl = 10 / 3

Query "quick": df = 2, so idf = ln(1 + (3 - 2 + 0.5) / (2 + 0.5)) = ln(1.6).

    c1: tf=1, dl=4 -> denom = 1 + 1.5 * (0.25 + 0.75 * 1.2) = 2.725
        score = ln(1.6) * (1 * 2.5) / 2.725
    c3: tf=2, dl=3 -> denom = 2 + 1.5 * (0.25 + 0.75 * 0.9) = 3.3875
        score = ln(1.6) * (2 * 2.5) / 3.3875
"""

from __future__ import annotations

import math

import pytest

from atlasrag.domain.models import Chunk
from atlasrag.indexing.lexical.bm25 import BM25Index, tokenize

TOL = 1e-12


def make_chunk(chunk_id: str, document_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        chunk_index=0,
        start_offset=0,
        end_offset=len(text),
        text=text,
        content_sha256="0" * 64,
        chunking_version="1",
    )


@pytest.fixture
def index() -> BM25Index:
    idx = BM25Index(k1=1.5, b=0.75)
    idx.build(
        [
            make_chunk("c1", "d1", "the quick brown fox"),
            make_chunk("c2", "d2", "the lazy dog"),
            make_chunk("c3", "d3", "quick quick fox"),
        ]
    )
    return idx


def test_idf_matches_hand_computation(index: BM25Index) -> None:
    assert index.idf("quick") == pytest.approx(math.log(1.6), abs=TOL)
    assert index.idf("dog") == pytest.approx(math.log(1 + 2.5 / 1.5), abs=TOL)
    assert index.idf("absent") == 0.0


def test_scores_match_hand_computation(index: BM25Index) -> None:
    scores = index.score_terms(["quick"])
    expected_c1 = math.log(1.6) * 2.5 / 2.725
    expected_c3 = math.log(1.6) * 5.0 / 3.3875

    assert scores["c1"] == pytest.approx(expected_c1, abs=TOL)
    assert scores["c3"] == pytest.approx(expected_c3, abs=TOL)
    assert "c2" not in scores


def test_higher_term_frequency_in_a_shorter_chunk_ranks_first(index: BM25Index) -> None:
    assert [cid for cid, _ in index.search("quick", top_k=10)] == ["c3", "c1"]


def test_idf_stays_positive_for_a_term_in_every_chunk() -> None:
    idx = BM25Index()
    idx.build([make_chunk("a", "d", "alpha beta"), make_chunk("b", "d", "alpha gamma")])
    assert idx.idf("alpha") == pytest.approx(math.log(1 + 0.5 / 2.5), abs=TOL)
    assert idx.idf("alpha") > 0.0


def test_zero_scores_are_not_returned(index: BM25Index) -> None:
    assert index.search("nonexistentterm", top_k=10) == []


def test_filter_restricts_to_allowed_documents(index: BM25Index) -> None:
    assert [cid for cid, _ in index.search("quick", top_k=10, allowed_document_ids={"d1"})] == [
        "c1"
    ]
    assert index.search("quick", top_k=10, allowed_document_ids=set()) == []


def test_tie_breaks_on_chunk_id() -> None:
    idx = BM25Index()
    idx.build([make_chunk("zz", "d", "alpha beta"), make_chunk("aa", "d", "alpha beta")])
    assert [cid for cid, _ in idx.search("alpha", top_k=10)] == ["aa", "zz"]


class TestTokenizer:
    def test_compound_identifier_is_kept_whole_and_split(self) -> None:
        assert tokenize("CVE-2023-4863") == ["cve-2023-4863", "cve", "2023", "4863"]

    def test_case_folded(self) -> None:
        assert tokenize("Quick FOX") == ["quick", "fox"]

    def test_punctuation_is_dropped(self) -> None:
        assert tokenize("hello, world! (yes)") == ["hello", "world", "yes"]

    def test_empty_text(self) -> None:
        assert tokenize("") == []


class TestPersistence:
    def test_round_trip_preserves_scores(self, index: BM25Index, tmp_path) -> None:
        path = tmp_path / "bm25.json"
        index.save(path)
        reloaded = BM25Index.load(path)
        assert reloaded.search("quick", top_k=10) == index.search("quick", top_k=10)

    def test_rebuilding_the_same_corpus_produces_identical_bytes(
        self, index: BM25Index, tmp_path
    ) -> None:
        first = tmp_path / "a.json"
        second = tmp_path / "b.json"
        index.save(first)

        rebuilt = BM25Index(k1=1.5, b=0.75)
        rebuilt.build(
            [
                make_chunk("c3", "d3", "quick quick fox"),
                make_chunk("c1", "d1", "the quick brown fox"),
                make_chunk("c2", "d2", "the lazy dog"),
            ]
        )
        rebuilt.save(second)
        assert first.read_bytes() == second.read_bytes()

    def test_incompatible_format_is_refused(self, index: BM25Index, tmp_path) -> None:
        import json

        from atlasrag.errors import IndexIncompatibleError

        path = tmp_path / "bm25.json"
        index.save(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["format_version"] = 999
        path.write_text(json.dumps(payload), encoding="utf-8")

        with pytest.raises(IndexIncompatibleError):
            BM25Index.load(path)

    def test_corrupt_file_is_refused(self, tmp_path) -> None:
        from atlasrag.errors import IndexCorruptError

        path = tmp_path / "bm25.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(IndexCorruptError):
            BM25Index.load(path)
