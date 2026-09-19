"""Acceptance criteria 4, 9, 10, 15: lexical precision, filters, deletion, restart."""

from __future__ import annotations

from pathlib import Path

import pytest

from atlasrag.config import Settings
from atlasrag.domain.models import SearchFilters, SearchQuery
from atlasrag.errors import ModelUnavailableError
from atlasrag.service import AtlasRagService
from tests.conftest import FIXTURE_CORPUS, HashEmbedder

RARE_IDENTIFIER = "CAL-X7-4421B"
GATEWAY_ASSET_TAG = "MG-GW-7741"


def test_rare_identifier_is_found_by_bm25_at_rank_one(corpus_service: AtlasRagService) -> None:
    """Acceptance criterion 4."""
    outcome = corpus_service.search(SearchQuery(text=RARE_IDENTIFIER, mode="bm25", top_k=5))
    assert outcome.hits
    assert outcome.hits[0].title == "Conveyor Sensor Calibration"
    assert RARE_IDENTIFIER.lower() in outcome.hits[0].text.lower()


def test_a_second_rare_identifier_resolves_to_its_own_document(
    corpus_service: AtlasRagService,
) -> None:
    outcome = corpus_service.search(SearchQuery(text=GATEWAY_ASSET_TAG, mode="bm25", top_k=5))
    assert outcome.hits[0].title == "Atlas Gateway Runbook"


def test_nonsense_query_returns_nothing_from_bm25(corpus_service: AtlasRagService) -> None:
    outcome = corpus_service.search(
        SearchQuery(text="zzzzqqqq nonexistentterm", mode="bm25", top_k=5)
    )
    assert outcome.hits == []


def test_contributions_expose_full_provenance(corpus_service: AtlasRagService) -> None:
    outcome = corpus_service.search(SearchQuery(text="calibration deviation", mode="hybrid"))
    assert outcome.hits
    for hit in outcome.hits:
        assert hit.contributions
        for contribution in hit.contributions:
            assert contribution.retriever in {"bm25", "dense"}
            assert contribution.rank >= 1
            assert contribution.rrf_term == pytest.approx(
                1.0 / (60 + contribution.rank), abs=1e-12
            )
        assert hit.fused_score == pytest.approx(
            sum(c.rrf_term * c.weight for c in hit.contributions), abs=1e-12
        )


def test_hybrid_returns_at_least_as_many_documents_as_either_channel(
    corpus_service: AtlasRagService,
) -> None:
    def docs(mode: str) -> set[str]:
        outcome = corpus_service.search(SearchQuery(text="belt sensor", mode=mode, top_k=10))
        return {h.document_id for h in outcome.hits}

    assert docs("bm25") <= docs("hybrid")
    assert docs("dense") <= docs("hybrid")


class TestFilters:
    """Acceptance criterion 9: filters must apply identically in every mode."""

    @pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
    def test_document_id_filter(self, corpus_service: AtlasRagService, mode: str) -> None:
        target = next(
            d for d in corpus_service.list_documents() if d.title == "Data Retention Policy"
        )
        outcome = corpus_service.search(
            SearchQuery(
                text="retention archival logs",
                mode=mode,
                top_k=10,
                filters=SearchFilters(document_ids=[target.document_id]),
            )
        )
        assert outcome.hits
        assert {h.document_id for h in outcome.hits} == {target.document_id}

    @pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
    def test_media_type_filter(self, corpus_service: AtlasRagService, mode: str) -> None:
        outcome = corpus_service.search(
            SearchQuery(
                text="belt supervisor shift",
                mode=mode,
                top_k=10,
                filters=SearchFilters(media_types=["text/plain"]),
            )
        )
        assert outcome.hits
        titles = {h.title for h in outcome.hits}
        assert titles == {"onboarding guide"}

    @pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
    def test_title_filter(self, corpus_service: AtlasRagService, mode: str) -> None:
        outcome = corpus_service.search(
            SearchQuery(
                text="timeout retry",
                mode=mode,
                top_k=10,
                filters=SearchFilters(title_contains="atlas gateway"),
            )
        )
        assert outcome.hits
        assert {h.title for h in outcome.hits} == {"Atlas Gateway Runbook"}

    @pytest.mark.parametrize("mode", ["bm25", "dense", "hybrid"])
    def test_filter_matching_nothing_returns_nothing(
        self, corpus_service: AtlasRagService, mode: str
    ) -> None:
        outcome = corpus_service.search(
            SearchQuery(
                text="belt",
                mode=mode,
                top_k=10,
                filters=SearchFilters(title_contains="no such document anywhere"),
            )
        )
        assert outcome.hits == []


class TestDeletion:
    """Acceptance criterion 10: deletion must reach every index."""

    def test_deleted_document_disappears_from_all_modes(
        self, corpus_service: AtlasRagService
    ) -> None:
        target = next(
            d for d in corpus_service.list_documents() if d.title == "Conveyor Sensor Calibration"
        )
        before = corpus_service.search(SearchQuery(text=RARE_IDENTIFIER, mode="bm25", top_k=5))
        assert before.hits

        assert corpus_service.delete_document(target.document_id) is True

        for mode in ("bm25", "dense", "hybrid"):
            outcome = corpus_service.search(
                SearchQuery(text=RARE_IDENTIFIER, mode=mode, top_k=10)
            )
            assert target.document_id not in {h.document_id for h in outcome.hits}

        assert corpus_service.get_document(target.document_id) is None
        assert corpus_service.chunks_for_document(target.document_id) == []

    def test_deletion_removes_rows_from_every_index_structure(
        self, corpus_service: AtlasRagService
    ) -> None:
        target = corpus_service.list_documents()[0]
        chunk_ids = {c.chunk_id for c in corpus_service.chunks_for_document(target.document_id)}
        corpus_service.delete_document(target.document_id)
        corpus_service.indexes.ensure_ready()

        assert chunk_ids.isdisjoint(corpus_service.indexes.bm25.chunk_ids)
        assert chunk_ids.isdisjoint(corpus_service.indexes.vectors.chunk_ids)
        assert corpus_service.indexes.vectors.matrix.shape[0] == len(
            corpus_service.indexes.vectors.chunk_ids
        )

    def test_deleting_an_unknown_id_reports_false(self, corpus_service: AtlasRagService) -> None:
        assert corpus_service.delete_document("0" * 64) is False


class TestRestartAndRebuild:
    """Acceptance criterion 15."""

    def test_results_survive_a_process_restart(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "var"
        first = AtlasRagService(Settings(data_dir=data_dir), embedder=HashEmbedder())
        for path in sorted(FIXTURE_CORPUS.iterdir()):
            first.ingest_path(path)
        first.indexes.ensure_ready()
        before = first.search(SearchQuery(text=RARE_IDENTIFIER, mode="hybrid", top_k=5))
        first.close()

        second = AtlasRagService(Settings(data_dir=data_dir), embedder=HashEmbedder())
        after = second.search(SearchQuery(text=RARE_IDENTIFIER, mode="hybrid", top_k=5))
        second.close()

        assert [h.chunk_id for h in before.hits] == [h.chunk_id for h in after.hits]
        assert [h.fused_score for h in before.hits] == pytest.approx(
            [h.fused_score for h in after.hits]
        )

    def test_dropping_indexes_rebuilds_identical_lexical_bytes(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "var"
        service = AtlasRagService(Settings(data_dir=data_dir), embedder=HashEmbedder())
        for path in sorted(FIXTURE_CORPUS.iterdir()):
            service.ingest_path(path)
        service.indexes.ensure_ready()

        index_file = service.indexes.index_dir / "bm25.json"
        original = index_file.read_bytes()

        service.indexes.drop_persisted()
        assert not index_file.exists()

        service.indexes.rebuild_all()
        assert index_file.read_bytes() == original
        service.close()

    def test_corrupt_lexical_index_is_discarded_and_rebuilt(self, tmp_path: Path) -> None:
        data_dir = tmp_path / "var"
        service = AtlasRagService(Settings(data_dir=data_dir), embedder=HashEmbedder())
        service.ingest_path(FIXTURE_CORPUS / "sensor-calibration.md")
        service.indexes.ensure_ready()

        (service.indexes.index_dir / "bm25.json").write_text("{corrupt", encoding="utf-8")
        service.indexes._bm25 = None  # force a reload from disk

        outcome = service.search(SearchQuery(text=RARE_IDENTIFIER, mode="bm25", top_k=5))
        assert outcome.hits
        service.close()


def test_dense_mode_without_an_embedder_raises_rather_than_silently_degrading(
    tmp_path: Path,
) -> None:
    """No silent fallbacks: a hybrid result must never be a lexical-only result in disguise."""
    from atlasrag.indexing.manager import IndexManager
    from atlasrag.retrieval.engine import SearchEngine
    from atlasrag.storage.sqlite_store import SqliteDocumentStore

    settings = Settings(data_dir=tmp_path / "var")
    store = SqliteDocumentStore(settings.db_path)
    engine = SearchEngine(store, IndexManager(store, settings, None), settings, None)

    with pytest.raises(ModelUnavailableError):
        engine.search(SearchQuery(text="anything", mode="hybrid"))
    store.close()
