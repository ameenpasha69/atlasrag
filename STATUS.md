# AtlasRAG — Status

Updated: 2026-09-19 · Milestones 1–2 executed · Milestone 6 (API/UI) not started

A capability is `VERIFIED` only if a command in [EVIDENCE.md](EVIDENCE.md) produced that
result on this machine. Nothing here is aspirational.

## Current state

Local retrieval, citation and abstention pipeline runs end to end from the CLI. Retrieval has
been evaluated against version-controlled judgments. There is **no HTTP API and no web UI yet**,
so the project does not yet meet its own definition of done.

## Feature status

| Feature | Status | Evidence |
|---|---|---|
| Deterministic ids (document + chunk) | VERIFIED | 110-test suite; two independent DBs produce identical ids |
| Offset-exact chunking | VERIFIED | every chunk asserts `text == doc[start:end]` |
| Idempotent re-ingestion | VERIFIED | second ingest returns `unchanged`, counts unmoved |
| Duplicate-content detection | VERIFIED | renamed copy returns `duplicate_content` |
| Input rejection (empty/oversize/type/UTF-8) | VERIFIED | 7 rejection tests, no partial rows |
| BM25 lexical retrieval | VERIFIED | scored against hand-computed fixtures |
| Dense retrieval | VERIFIED | paraphrase query dense rank 1, absent from BM25 top-5 |
| RRF fusion | VERIFIED | matches hand-computed 1/(60+rank) fixtures |
| Deterministic tie-breaking | VERIFIED | stable across 50 shuffled permutations |
| Metadata filters in all 3 modes | VERIFIED | 12 parametrised filter tests |
| Deletion across every index | VERIFIED | absent from SQLite, BM25 postings, vector matrix |
| Restart / rebuild | VERIFIED | byte-identical BM25 index after drop + rebuild |
| Corrupt-index recovery | VERIFIED | unusable index discarded and rebuilt |
| Citation span validation | VERIFIED | citation validity 1.000; negative test rejects bad offsets |
| Abstention (no evidence / weak / conflicting) | VERIFIED | 12 of 15 test queries correct |
| Injected-instruction filtering | VERIFIED | both adversarial queries abstain; held-out q22 generalised |
| Evaluation harness + metrics | VERIFIED | executed; see EVALUATION.md |
| Abstention calibration | VERIFIED | 264-point sweep, F1 0.833 on calibration |
| Lint / format / strict types | VERIFIED | ruff clean, mypy strict clean on 47 files |
| Extractive answering | VERIFIED *with a measured limitation* | fails paraphrased questions — see below |
| `OUT_OF_SCOPE` abstention reason | **UNREACHABLE** | calibrated floors are 0.0; see EVALUATION.md §2 |
| HTTP API | NOT STARTED | — |
| Web UI | NOT STARTED | — |
| LLM answer provider | NOT STARTED | — |
| PDF / HTML ingestion | NOT STARTED | — |
| Container packaging | NOT STARTED | — |
| Concurrency / latency benchmark | NOT STARTED | — |

## Verified capabilities, stated precisely

- Ingests `.txt`, `.md`, `.markdown`. **No other format is supported or claimed.**
- 7 documents → 11 chunks → 646 BM25 terms → 11 dense vectors at dimension 384.
- Searches in `bm25`, `dense`, `hybrid`, each exposing per-retriever rank, raw score and RRF
  contribution for every hit.
- Answers extractively with span-verified citations, or abstains with a structured reason.
- Deletion and restart are tested, not assumed.

## Known limitations

1. **Hybrid fusion does not beat dense alone on the current dataset** (nDCG@3 0.943 vs 1.000).
   Measured, not suspected. EVALUATION.md §3.
2. **The extractive answerer fails paraphrased questions** — 2 of 15 test queries. Support
   scoring is lexical, so it is weakest exactly where dense retrieval is strongest.
3. **Ambiguous questions are not detected.** Conflict detection is cross-document only; three
   different timeout values inside one document are not a contradiction, so q14 is answered
   when it should abstain.
4. **`OUT_OF_SCOPE` is unreachable at the calibrated thresholds.** Out-of-corpus questions
   still abstain, but report `BELOW_THRESHOLD`.
5. **Evaluation rests on 7 documents and 28 queries, all author-written.** See EVALUATION.md §6
   before quoting any metric.
6. **Injection filtering is a pattern heuristic** — English-only and bypassable by rephrasing.
   SECURITY.md states what is and is not guaranteed.
7. **Dense embeddings are not bit-reproducible** across BLAS or batch-size changes. Dense tests
   use tolerance-based assertions; only ids and BM25 are bit-exact.
8. **No concurrency, latency or memory measurements exist.** No performance claim is made.

## Failed experiments and corrections

| What | Outcome |
|---|---|
| Support scoring over all query terms | Stopwords let irrelevant sentences look responsive; out-of-corpus questions were answered. Fixed by excluding stopwords and pinning unseen terms at the IDF ceiling. |
| Conflict detection on the top candidate only | Missed the 30-vs-90-day contradiction, because the top sentence was a framing sentence with no value. Fixed by scanning the credible band. |
| Quoting any sentence | Markdown headings and injected instructions were quoted as answers. Fixed by a quotability filter. |
| `min_bm25` / `min_cosine` thresholds | Did not discriminate at all. Kept at 0.0 and documented rather than tuned to look active. |
| Adding `long` to the stopword list | **Not done.** It would likely fix q14, which is in the test split. Declined as tuning on test. |

## Next smallest step

Milestone 6: FastAPI routes (`/documents`, `/search`, `/answer`, `/health`, `/ready`) over the
existing service, then the UI. The service layer already exposes everything the routes need.

## Verification commands

```bash
uv sync --extra dev
uv run ruff check src tests && uv run ruff format --check src tests
uv run mypy
uv run pytest -q
rm -rf var && uv run atlasrag ingest fixtures/corpus/v1
uv run atlasrag stats
uv run atlasrag calibrate
uv run atlasrag evaluate --split test --k 5 --dataset-version v2
```
