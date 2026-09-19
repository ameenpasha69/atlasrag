# AtlasRAG

Hybrid lexical + dense retrieval with Reciprocal Rank Fusion, span-verified citations, and
mandatory abstention. Local-first, no paid API required.

This is a **search and retrieval engineering project**, not an LLM wrapper. The default answer
provider contains no language model at all.

> Nothing in this repository is described as working unless a command in
> [EVIDENCE.md](EVIDENCE.md) produced that result. Current capabilities and — just as
> importantly — current limitations are in [STATUS.md](STATUS.md).

**Status: Milestones 1, 2 and 6 executed.** CLI, HTTP API and web interface all work against
the same service layer. Not yet done: optional LLM answering, PDF/HTML ingestion, container
packaging, and any performance measurement.

---

## What it does

1. Ingests `.txt` / `.md` deterministically — same bytes in, same document and chunk ids out.
2. Chunks with exact character offsets, so every citation is checkable against source text.
3. Retrieves with **BM25** (exact terms, rare identifiers) and **dense vectors** (paraphrase).
4. Fuses the two rankings with **RRF**, exposing every rank, raw score and contribution.
5. Answers by quoting verbatim source spans with validated citations — or **abstains**.

Abstention is a first-class output, not an error path. The system refuses to answer when
evidence is missing, too weak, contradictory, or consists of instruction-shaped text planted in
a document.

## Why both retrievers

Measured on the fixture corpus, query *"why do packages pile up before they are sorted"*:

| mode | result |
|---|---|
| dense | **rank 1**, cosine 0.632 |
| bm25 | **not in the top 5** |

The target sentence reads "parcels accumulate at the staging belt" — no content word in common.
Conversely, BM25 returns `CAL-X7-4421B` at rank 1 with a single hit, which is exactly the case
vector search handles badly.

## An honest headline result

On the current evaluation set, **hybrid fusion does not beat dense retrieval alone**
(nDCG@3 0.943 vs 1.000). Hybrid does beat BM25 on recall. The corpus is small and
paraphrase-heavy, which structurally favours dense, and the case for fusion rests on query
types this dataset is too small to contain. That analysis — and the reasons not to
over-read it — is in [EVALUATION.md](EVALUATION.md).

Reporting this rather than burying it is the point of the project.

## Quick start

```bash
uv sync --extra dev
uv run atlasrag ingest fixtures/corpus/v1
uv run atlasrag stats
```

First run downloads the pinned embedding model (~130 MB). After that it works offline.

```bash
# exact identifier -> lexical channel
uv run atlasrag search "CAL-X7-4421B" --mode bm25

# paraphrase -> dense channel
uv run atlasrag search "why do packages pile up before they are sorted" --mode dense

# fusion, with per-retriever contributions printed
uv run atlasrag search "calibration deviation" --mode hybrid

# grounded answer, or a structured abstention
uv run atlasrag answer "what serial is stamped on the calibration reference unit"
uv run atlasrag answer "how long are dispatch logs retained"   # abstains: sources disagree
uv run atlasrag answer "what is the melting point of tungsten" # abstains: not in corpus
```

## API and web interface

```bash
uv run uvicorn atlasrag.api.app:app --port 8077
```

Open <http://127.0.0.1:8077/> for the interface, or `/docs` for the generated OpenAPI schema.

| Route | Purpose |
|---|---|
| `POST /documents` | upload one or more files; per-file status and rejection reason |
| `GET /documents` · `GET /documents/{id}` · `DELETE /documents/{id}` | browse, inspect, remove |
| `POST /search` | retrieve with full per-retriever rank, raw score and RRF contribution |
| `POST /answer` | grounded answer with validated citations, or a structured abstention |
| `POST /reindex` | drop and rebuild every derived index from the registry |
| `GET /health` · `GET /ready` | liveness (is the process up) vs readiness (are the indexes usable) |

The interface has no decorative controls: every button calls a real endpoint. It shows the RRF
contribution table per hit, opens a citation to the highlighted source span, flags passages
containing instruction-like text, and displays abstentions with their reason.

> The API has no authentication, rate limiting or request-size cap. Bind it to localhost.
> See [SECURITY.md](SECURITY.md).

## Evaluation

```bash
uv run atlasrag calibrate                                        # sweep on the calibration split
uv run atlasrag evaluate --split test --k 5 --dataset-version v2 # report on the held-out split
```

Thresholds are fitted on `fixtures/eval/v2/calibration.jsonl` and reported on
`test.jsonl`, which the sweep never sees.

## Verification

```bash
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy
uv run pytest -q
```

Last executed: ruff clean · mypy strict clean (49 files) · 135 tests passed
(unit, integration and end-to-end HTTP).

## Architecture

```
ingestion -> chunking -> SQLite registry (source of truth)
                              |
                   +----------+----------+
                   v                     v
            BM25 index            vector index      (both derived, rebuildable)
                   \                     /
                    +----- RRF fusion --+
                              |
                    evidence -> citations -> answer | abstention
```

SQLite is authoritative; both indexes are derived artifacts rebuilt from it. That is what makes
"deletion reaches every index" and "survives a restart" structural properties rather than
bookkeeping. Because chunk ids are content-addressed, embeddings are reused by id across
rebuilds, which keeps always-rebuild affordable.

The domain layer defines `Protocol` interfaces and imports nothing that touches disk, network or
torch. Concrete adapters are wired in `service.py` and nowhere else.

| Document | Contents |
|---|---|
| [STATUS.md](STATUS.md) | what is verified, what is not, known limitations, failed experiments |
| [DECISIONS.md](DECISIONS.md) | 13 decisions with trade-offs and revisit conditions |
| [EVALUATION.md](EVALUATION.md) | dataset design, metrics, mode comparison, failure analysis, threats to validity |
| [EVIDENCE.md](EVIDENCE.md) | append-only log of executed commands and their real output |
| [SECURITY.md](SECURITY.md) | threat model, prompt-injection handling, and gaps |

## Not supported

No PDF, HTML, OCR, scanned documents, table extraction or layout understanding. No
authentication, multi-tenancy or compliance claim. No container image. No performance claim —
no latency, throughput or memory measurement has been taken.

The prompt-injection filter is a documented heuristic, not a guarantee; what is structural is
that the default provider has no generation step and therefore cannot follow an instruction at
all. [SECURITY.md](SECURITY.md) is explicit about the difference.

## License

MIT.
