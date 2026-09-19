"""Milestone 7: reliability and safety properties that were previously asserted but untested."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from atlasrag.config import Settings
from atlasrag.domain.models import SearchQuery
from atlasrag.errors import IndexCorruptError, IndexIncompatibleError
from atlasrag.indexing.dense.vector_store import VectorIndex
from atlasrag.service import AtlasRagService
from tests.conftest import FIXTURE_CORPUS, HashEmbedder

RARE_IDENTIFIER = "CAL-X7-4421B"


class TestConcurrency:
    def test_reads_during_writes_never_error_or_tear(self, service: AtlasRagService) -> None:
        """Searches against a registry being written must not error or see a torn document."""
        service.ingest_path(FIXTURE_CORPUS / "sensor-calibration.md")
        service.indexes.ensure_ready()

        errors: list[BaseException] = []
        observed_counts: list[int] = []
        stop = threading.Event()

        def reader() -> None:
            try:
                while not stop.is_set():
                    outcome = service.search(
                        SearchQuery(text="calibration sensor belt", mode="bm25", top_k=5)
                    )
                    for hit in outcome.hits:
                        # A torn read would surface as a hit whose chunk no longer resolves.
                        assert service.store.get_chunk(hit.chunk_id) is not None
                    observed_counts.append(len(outcome.hits))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for thread in threads:
            thread.start()
        try:
            for path in sorted(FIXTURE_CORPUS.iterdir()):
                if path.is_file():
                    service.ingest_path(path)
        finally:
            stop.set()
            for thread in threads:
                thread.join(timeout=30)

        assert not errors, f"concurrent reads raised: {errors[0]!r}"
        assert observed_counts, "readers never completed a search"

    def test_concurrent_ingestion_of_the_same_file_yields_one_document(
        self, service: AtlasRagService
    ) -> None:
        """Content-addressed ids make a race a no-op rather than a duplicate."""
        barrier = threading.Barrier(4)
        errors: list[BaseException] = []

        def worker() -> None:
            try:
                barrier.wait(timeout=30)
                service.ingest_path(FIXTURE_CORPUS / "sensor-calibration.md")
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=30)

        assert not errors, f"concurrent ingestion raised: {errors[0]!r}"
        documents, _ = service.store.counts()
        assert documents == 1


class TestAtomicity:
    def test_index_writes_leave_no_temporary_files(self, corpus_service: AtlasRagService) -> None:
        corpus_service.indexes.rebuild_all()
        leftovers = [p.name for p in corpus_service.indexes.index_dir.iterdir() if ".tmp" in p.name]
        assert leftovers == []

    def test_rejected_ingestion_is_fully_rolled_back(
        self, service: AtlasRagService, tmp_path: Path
    ) -> None:
        service.ingest_path(FIXTURE_CORPUS / "sensor-calibration.md")
        before = service.store.counts()

        bad = tmp_path / "broken.txt"
        bad.write_bytes(b"prefix \xff\xfe suffix")
        assert service.ingest_path(bad).status == "rejected"

        assert service.store.counts() == before

    def test_deleting_a_document_leaves_no_orphan_chunks(
        self, corpus_service: AtlasRagService
    ) -> None:
        target = corpus_service.list_documents()[0]
        corpus_service.delete_document(target.document_id)

        remaining = {d.document_id for d in corpus_service.list_documents()}
        for chunk in corpus_service.store.iter_chunks():
            assert chunk.document_id in remaining


class TestCorruptionRecovery:
    def test_corrupt_vector_manifest_is_discarded_and_rebuilt(
        self, corpus_service: AtlasRagService
    ) -> None:
        manifest = corpus_service.indexes.index_dir / "vectors.manifest.json"
        manifest.write_text("{ truncated", encoding="utf-8")
        corpus_service.indexes._vectors = None

        outcome = corpus_service.search(SearchQuery(text=RARE_IDENTIFIER, mode="dense", top_k=3))
        assert outcome.hits

    def test_truncated_vector_matrix_is_detected(self, corpus_service: AtlasRagService) -> None:
        index_dir = corpus_service.indexes.index_dir
        manifest = json.loads((index_dir / "vectors.manifest.json").read_text(encoding="utf-8"))
        manifest["chunk_ids"] = manifest["chunk_ids"][:-1]
        (index_dir / "vectors.manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        with pytest.raises(IndexCorruptError):
            VectorIndex.load(
                index_dir,
                expected_model=manifest["model_id"],
                expected_revision=manifest["revision"],
            )

    def test_index_built_with_another_model_is_refused(
        self, corpus_service: AtlasRagService
    ) -> None:
        index_dir = corpus_service.indexes.index_dir
        manifest = json.loads((index_dir / "vectors.manifest.json").read_text(encoding="utf-8"))

        with pytest.raises(IndexIncompatibleError, match="rebuild the index"):
            VectorIndex.load(
                index_dir, expected_model="some/other-model", expected_revision=manifest["revision"]
            )

    def test_changing_the_model_rebuilds_rather_than_mixing_vector_spaces(
        self, tmp_path: Path
    ) -> None:
        settings = Settings(data_dir=tmp_path / "var")
        first = AtlasRagService(settings, embedder=HashEmbedder(dimension=64))
        first.ingest_path(FIXTURE_CORPUS / "sensor-calibration.md")
        first.indexes.ensure_ready()
        assert first.indexes.vectors.dimension == 64
        first.close()

        class OtherEmbedder(HashEmbedder):
            @property
            def model_id(self) -> str:
                return "test/other-embedder"

        second = AtlasRagService(settings, embedder=OtherEmbedder(dimension=32))
        second.indexes.ensure_ready()
        assert second.indexes.vectors.dimension == 32
        assert second.indexes.vectors.model_id == "test/other-embedder"
        assert second.search(SearchQuery(text="calibration", mode="dense", top_k=3)).hits
        second.close()

    def test_query_with_wrong_dimension_is_refused(self, corpus_service: AtlasRagService) -> None:
        with pytest.raises(IndexIncompatibleError):
            corpus_service.indexes.vectors.search(np.zeros(7, dtype=np.float32), top_k=3)


class TestHostileContent:
    """Document text is attacker-controlled. It must be stored and returned as inert data."""

    def test_markup_in_a_document_is_stored_verbatim_and_never_interpreted(
        self, service: AtlasRagService, tmp_path: Path
    ) -> None:
        payload = (
            "# Report\n\n"
            "<script>alert('xss')</script> and an <img src=x onerror=alert(1)> tag "
            "inside otherwise ordinary prose about conveyor belts.\n"
        )
        path = tmp_path / "hostile.md"
        path.write_text(payload, encoding="utf-8")

        result = service.ingest_path(path)
        assert result.status == "ingested"
        assert result.document_id is not None

        document = service.get_document(result.document_id)
        assert document is not None
        assert "<script>" in document.normalized_text, "text must be preserved, not sanitised away"

        for chunk in service.chunks_for_document(result.document_id):
            assert chunk.text == document.normalized_text[chunk.start_offset : chunk.end_offset]

    def test_citation_offsets_survive_hostile_text(
        self, service: AtlasRagService, tmp_path: Path
    ) -> None:
        path = tmp_path / "hostile2.md"
        path.write_text(
            "# Calibration\n\nThe reference unit <b>XYZ-9</b> is stored in the cabinet. "
            "A deviation above 1.5 millimetres requires replacement.\n",
            encoding="utf-8",
        )
        service.ingest_path(path)
        service.indexes.ensure_ready()

        response = service.answer(
            SearchQuery(text="what deviation requires replacement", mode="bm25", top_k=3)
        )
        if response.answered:
            for citation in response.citations:
                document = service.get_document(citation.document_id)
                assert document is not None
                assert (
                    document.normalized_text[citation.start_offset : citation.end_offset]
                    == citation.snippet
                )

    def test_null_bytes_and_control_characters_are_stripped(
        self, service: AtlasRagService, tmp_path: Path
    ) -> None:
        path = tmp_path / "control.txt"
        path.write_bytes(b"Before\x00\x07\x1bAfter the control characters.")
        result = service.ingest_path(path)
        assert result.document_id is not None
        document = service.get_document(result.document_id)
        assert document is not None
        assert "\x00" not in document.normalized_text
        assert "\x1b" not in document.normalized_text


class TestRestartSemantics:
    def test_two_services_on_one_directory_agree(self, tmp_path: Path) -> None:
        settings = Settings(data_dir=tmp_path / "var")
        writer = AtlasRagService(settings, embedder=HashEmbedder())
        writer.ingest_path(FIXTURE_CORPUS / "sensor-calibration.md")
        writer.indexes.ensure_ready()

        reader = AtlasRagService(settings, embedder=HashEmbedder())
        assert reader.store.counts() == writer.store.counts()
        assert (
            reader.search(SearchQuery(text=RARE_IDENTIFIER, mode="hybrid", top_k=3))
            .hits[0]
            .chunk_id
            == writer.search(SearchQuery(text=RARE_IDENTIFIER, mode="hybrid", top_k=3))
            .hits[0]
            .chunk_id
        )
        reader.close()
        writer.close()

    def test_missing_index_directory_is_rebuilt_on_demand(
        self, corpus_service: AtlasRagService
    ) -> None:
        corpus_service.indexes.drop_persisted()
        assert not corpus_service.indexes.index_dir.exists()

        outcome = corpus_service.search(SearchQuery(text=RARE_IDENTIFIER, mode="hybrid", top_k=3))
        assert outcome.hits
        assert corpus_service.indexes.index_dir.exists()
