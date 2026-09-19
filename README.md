# AtlasRAG

Hybrid lexical + dense retrieval with Reciprocal Rank Fusion, span-verified citations, and
mandatory abstention. Local-first, no paid API required.

This is a **search and retrieval engineering project**, not an LLM wrapper. The default answer
provider contains no language model at all.

> Nothing in this repository is described as working unless a command in
> [EVIDENCE.md](EVIDENCE.md) produced that result. Current capabilities and — just as
> importantly — current limitations are in [STATUS.md](STATUS.md).

**Status: Milestones 1, 2, 3, 5 (experiment 1), 6 and 7 executed.** CLI, HTTP API and web
interface all work against the same service layer. Not yet done: PDF/HTML ingestion, and a
container build verified end to end. The generative provider's safety machinery is tested
against a stub model but has **never been run against a real one** — see
[STATUS.md](STATUS.md) for exactly what that does and does not establish.

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

**Hybrid fusion does not beat dense retrieval alone on any dataset measured here.**

On the first dataset it was behind (nDCG@3 0.943 vs 1.000). The obvious objection was that the
corpus contained no query type where dense should fail — so a second dataset was built
specifically to contain them: asset tags differing by one character, error codes differing by a
digit transposition, each pair split across different documents.

The predicted failure happened exactly as stated. Asked for `ERR-4471`, dense returns the
document containing `ERR-4417`; BM25 gets it right at rank 1. The reverse also happened: on a
paraphrased question BM25 missed the correct document entirely (reciprocal rank 0.000) while
dense scored 1.000. **The two channels genuinely fail on disjoint query types.**

And hybrid still only reached *parity* — mean MRR 0.842 against dense's 0.842. RRF gives both
channels an equal vote, so a query one channel gets right and the other gets wrong tends to land
the answer at rank 2 rather than rank 1.

What fusion did buy is narrower than the usual claim: **robustness, not accuracy.** It is the
only mode with no catastrophic miss on either query family. That is a real reason to keep it,
and it is not the reason hybrid retrieval is normally sold on.

The experiment, including the follow-up deliberately *not* run because its validating queries
are in the test split, is in [EVALUATION.md §5a](EVALUATION.md). Reporting this rather than
burying it is the point of the project.

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
uv run atlasrag compare  --split test --k 3 --dataset-version v2 # per-query win/loss by mode
```

Thresholds are fitted on `fixtures/eval/v2/calibration.jsonl` and reported on `test.jsonl`,
which the sweep never sees. `compare` exists because aggregate metrics hid the fact that two
modes were failing *different* queries while failing the same *number* of them.

To reproduce the fusion experiment on its own corpus, in its own data directory:

```bash
uv run atlasrag --data-dir var_exp ingest fixtures/corpus/v2
uv run atlasrag --data-dir var_exp compare --split test --dataset-version v3 --k 3
```

Datasets are versioned and never edited in place; every change is justified in
[`fixtures/eval/CHANGES.md`](fixtures/eval/CHANGES.md).

## Verification

```bash
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy
uv run pytest -q
```

Last executed: ruff clean · mypy strict clean (50 files) · **167 tests passed**
(unit, integration, reliability, generative-provider and end-to-end HTTP).

```bash
uv run atlasrag bench --repeats 5    # query latency percentiles on your machine
```

Measured here: BM25 p50 **0.3 ms**, dense **30.2 ms**, hybrid **29.3 ms** on an 11-chunk
corpus, single-threaded. The dense cost is the query-side forward pass, not the similarity
search — so hybrid costs the same as dense rather than the sum of both. No throughput or
capacity claim follows from this; see [EVALUATION.md §6](EVALUATION.md).

## Answering

The default provider is **extractive** and contains no language model: it selects verbatim
source spans, so its citations are exact by construction and it cannot follow an instruction
found in a document, because it cannot produce a token that is not already in the corpus.

An optional generative provider exists because the extractive one has a *measured* limit — it
fails paraphrased questions, since its support score is lexical. Its guarantee lives in code,
not in the prompt:

- evidence is passed inside a delimited block and the prompt states it is untrusted data;
- **every generated sentence must cite a supplied passage and survive a post-generation support
  check against it**, or the sentence is discarded;
- if nothing survives, the response abstains rather than answering;
- a provider timeout raises an error and is **never** reported as an abstention.

Enable it with `ATLASRAG_ANSWER_PROVIDER=llm` plus `ATLASRAG_LLM_BASE_URL` and
`ATLASRAG_LLM_MODEL` (any OpenAI-compatible endpoint). Without those, the service keeps the
extractive provider and says so: `/ready` reports `answer_provider`,
`answer_provider_requested` and a note explaining the difference. The fallback is part of the
API contract, not a surprise.

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
authentication, rate limiting, multi-tenancy or compliance claim — bind the API to localhost.
Latency is measured; **throughput, concurrency under load and memory use are not**. Answer
quality with a generative provider is unmeasured, because no real model endpoint was
available here.

The prompt-injection filter is a documented heuristic, not a guarantee; what is structural is
that the default provider has no generation step and therefore cannot follow an instruction at
all. [SECURITY.md](SECURITY.md) is explicit about the difference.

## License

MIT.
