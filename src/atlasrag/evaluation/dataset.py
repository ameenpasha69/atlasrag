"""Loading of the version-controlled evaluation dataset.

Judgments name documents by source filename rather than by document id. Ids are content
hashes, so a one-character edit to a fixture would invalidate a hand-written judgment file
and make the dataset unreviewable. Filenames are resolved to ids against the live registry
at load time, and an unresolvable filename is an error rather than a silently dropped row.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from atlasrag.domain.models import EvaluationQuery, SearchFilters
from atlasrag.errors import EvaluationError

DATASET_VERSION = "v1"


@dataclass(frozen=True, slots=True)
class Judgment:
    query_id: str
    source: str
    grade: int
    rationale: str


@dataclass(frozen=True, slots=True)
class EvaluationDataset:
    version: str
    split: str
    queries: list[EvaluationQuery]
    judgments_by_query: dict[str, dict[str, int]]
    """query_id -> {document_id: grade}"""

    def relevant(self, query_id: str) -> dict[str, int]:
        return self.judgments_by_query.get(query_id, {})


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        raise EvaluationError(f"evaluation file not found: {path}")
    rows: list[dict[str, object]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            rows.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise EvaluationError(f"{path}:{number} is not valid JSON: {exc}") from exc
    return rows


def load_queries(path: Path) -> list[EvaluationQuery]:
    queries: list[EvaluationQuery] = []
    for row in _read_jsonl(path):
        raw_filters = row.pop("filters", None)
        filters = SearchFilters(**raw_filters) if isinstance(raw_filters, dict) else None
        queries.append(EvaluationQuery(**row, filters=filters))  # type: ignore[arg-type]
    return queries


def load_judgments(path: Path) -> list[Judgment]:
    return [Judgment(**row) for row in _read_jsonl(path)]  # type: ignore[arg-type]


def load_dataset(
    directory: Path,
    split: str,
    *,
    source_to_document_id: dict[str, str],
    version: str = DATASET_VERSION,
) -> EvaluationDataset:
    queries = load_queries(directory / f"{split}.jsonl")
    judgments = load_judgments(directory / "judgments.jsonl")
    query_ids = {q.query_id for q in queries}

    grouped: dict[str, dict[str, int]] = defaultdict(dict)
    for judgment in judgments:
        if judgment.query_id not in query_ids:
            continue
        document_id = source_to_document_id.get(judgment.source)
        if document_id is None:
            raise EvaluationError(
                f"judgment for {judgment.query_id} names '{judgment.source}', which is not "
                f"in the registry; ingest the fixture corpus before evaluating"
            )
        grouped[judgment.query_id][document_id] = judgment.grade

    for query in queries:
        if query.expected_answerable and not grouped.get(query.query_id):
            raise EvaluationError(
                f"query {query.query_id} is marked answerable but has no relevance judgments"
            )

    return EvaluationDataset(
        version=version,
        split=split,
        queries=queries,
        judgments_by_query=dict(grouped),
    )
