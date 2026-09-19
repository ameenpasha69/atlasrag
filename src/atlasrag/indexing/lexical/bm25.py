"""BM25 Okapi lexical index.

Implemented in-repo rather than taken from a library so that the scoring can be checked
against fixtures computed by hand (tests/unit/test_bm25.py) and so that tokenization is
pinned and inspectable. The formula is the standard Okapi BM25 with Lucene's
always-positive IDF variant:

    idf(t)   = ln(1 + (N - df + 0.5) / (df + 0.5))
    score(t) = idf(t) * (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / avgdl))
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from atlasrag.domain.models import Chunk
from atlasrag.errors import IndexCorruptError, IndexIncompatibleError

FORMAT_VERSION = 1
TOKENIZER_VERSION = "1"

_WORD = re.compile(r"[A-Za-z0-9]+(?:[._\-/][A-Za-z0-9]+)*")
_SUBTOKEN_SPLIT = re.compile(r"[._\-/]")


def tokenize(text: str) -> list[str]:
    """Case-folded alphanumeric tokens, keeping compound identifiers whole.

    A compound such as `CVE-2023-4863` is emitted both whole and as its parts, so an exact
    query matches the strong, rare whole token while a partial query can still land.
    """
    tokens: list[str] = []
    for match in _WORD.finditer(text):
        word = match.group(0).casefold()
        tokens.append(word)
        if _SUBTOKEN_SPLIT.search(word):
            tokens.extend(part for part in _SUBTOKEN_SPLIT.split(word) if part)
    return tokens


class BM25Index:
    def __init__(self, *, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._postings: dict[str, dict[str, int]] = {}
        self._doc_lengths: dict[str, int] = {}
        self._chunk_documents: dict[str, str] = {}
        self._avg_length: float = 0.0

    @property
    def doc_count(self) -> int:
        return len(self._doc_lengths)

    @property
    def vocabulary_size(self) -> int:
        return len(self._postings)

    @property
    def chunk_ids(self) -> list[str]:
        return list(self._doc_lengths)

    def build(self, chunks: Sequence[Chunk]) -> None:
        postings: dict[str, dict[str, int]] = defaultdict(dict)
        doc_lengths: dict[str, int] = {}
        chunk_documents: dict[str, str] = {}

        for chunk in chunks:
            tokens = tokenize(chunk.text)
            doc_lengths[chunk.chunk_id] = len(tokens)
            chunk_documents[chunk.chunk_id] = chunk.document_id
            counts: dict[str, int] = defaultdict(int)
            for token in tokens:
                counts[token] += 1
            for term, tf in counts.items():
                postings[term][chunk.chunk_id] = tf

        self._postings = {term: dict(sorted(v.items())) for term, v in sorted(postings.items())}
        self._doc_lengths = dict(sorted(doc_lengths.items()))
        self._chunk_documents = dict(sorted(chunk_documents.items()))
        total = sum(self._doc_lengths.values())
        self._avg_length = (total / len(self._doc_lengths)) if self._doc_lengths else 0.0

    def idf(self, term: str) -> float:
        """Inverse document frequency for a single term, for callers that weight by rarity."""
        return self._idf(term)

    @property
    def oov_idf(self) -> float:
        """IDF a term would have at df = 0, i.e. the ceiling of the IDF curve.

        Scoring callers use this for query terms absent from the corpus. Treating them as
        weightless would make a question about a topic the corpus has never heard of look
        just as answerable as one it covers; treating them as maximally informative and
        permanently unmatched is what lets support scores drive abstention.
        """
        n = len(self._doc_lengths)
        return math.log(1.0 + (n + 0.5) / 0.5) if n else 0.0

    def _idf(self, term: str) -> float:
        df = len(self._postings.get(term, ()))
        if df == 0:
            return 0.0
        n = len(self._doc_lengths)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score_terms(self, terms: Sequence[str]) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        if not self._doc_lengths or self._avg_length <= 0:
            return {}
        for term in terms:
            posting = self._postings.get(term)
            if not posting:
                continue
            idf = self._idf(term)
            if idf <= 0.0:
                continue
            for chunk_id, tf in posting.items():
                dl = self._doc_lengths[chunk_id]
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avg_length)
                scores[chunk_id] += idf * (tf * (self.k1 + 1.0)) / denom
        return dict(scores)

    def search(
        self,
        query: str,
        *,
        top_k: int,
        allowed_document_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        scores = self.score_terms(tokenize(query))
        if allowed_document_ids is not None:
            scores = {
                cid: s
                for cid, s in scores.items()
                if self._chunk_documents.get(cid) in allowed_document_ids
            }
        # Tie-break on chunk_id so equal scores never reorder between runs.
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(cid, score) for cid, score in ranked[:top_k] if score > 0.0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": FORMAT_VERSION,
            "tokenizer_version": TOKENIZER_VERSION,
            "k1": self.k1,
            "b": self.b,
            "avg_length": self._avg_length,
            "doc_lengths": self._doc_lengths,
            "chunk_documents": self._chunk_documents,
            "postings": self._postings,
        }

    def save(self, path: Path) -> None:
        """Atomic write: a crash mid-save must not leave a half-written index."""
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        payload = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False, indent=None)
        tmp.write_text(payload, encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> BM25Index:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexCorruptError(f"cannot read BM25 index at {path}: {exc}") from exc
        if data.get("format_version") != FORMAT_VERSION:
            raise IndexIncompatibleError(
                f"BM25 index format {data.get('format_version')} != expected {FORMAT_VERSION}"
            )
        if data.get("tokenizer_version") != TOKENIZER_VERSION:
            raise IndexIncompatibleError(
                f"BM25 tokenizer {data.get('tokenizer_version')} != expected {TOKENIZER_VERSION}"
            )
        index = cls(k1=float(data["k1"]), b=float(data["b"]))
        index._postings = data["postings"]
        index._doc_lengths = data["doc_lengths"]
        index._chunk_documents = data["chunk_documents"]
        index._avg_length = float(data["avg_length"])
        return index
