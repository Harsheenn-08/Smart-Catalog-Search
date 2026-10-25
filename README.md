# Smart Catalog Search Service

A hybrid e-commerce search engine combining SQL-based structured filtering with semantic vector search, plus an LLM query-understanding layer. Built with FastAPI, SQLite, SentenceTransformers, and FAISS.

## Overview

Instead of plain keyword matching, this service lets users search naturally (e.g., "gaming laptop with high performance") while enforcing hard constraints like price, stock availability, and category — the same pattern used by real e-commerce search systems.

The `/api/search/smart` endpoint goes one step further: users can type constraints in plain language ("furniture under 500 dollars in stock") and an LLM converts them into the structured filters the pipeline already supports.

**Pipeline:**
1. *(Smart endpoint only)* An LLM extracts `semantic_query`, `category`, `max_price` and `in_stock_only` from the sentence, validated with Pydantic
2. Query text is embedded into a 384-dimensional vector using `all-MiniLM-L6-v2` (SentenceTransformers)
3. FAISS (`IndexFlatIP`) retrieves a wide candidate pool of semantically similar products
4. SQLite applies hard filters (price, category, stock) on that candidate pool, using B-tree indexes
5. Remaining candidates are ranked by cosine similarity and returned as JSON

## Tech Stack

- **Backend:** Python, FastAPI, Pydantic
- **Database:** SQLite (B-tree indexes on `price`, `category`, `stock_count`)
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`, local model, no external API calls)
- **Vector Search:** FAISS (`IndexFlatIP`)
- **LLM query parsing:** Groq API (`openai/gpt-oss-20b`), JSON output validated with Pydantic
- **Testing:** pytest
- **Deployment:** Docker

## API Endpoints

### `POST /api/products`
Creates a product and computes/stores its embedding.

### `GET /api/search`
Searches the catalog. Query params: `query`, `category`, `max_price`, `in_stock_only`, `top_k`.

### `GET /api/search/smart`
Same search, but takes a natural-language sentence. Query params: `query`, `top_k`. The response includes the filters the LLM extracted (`parsed_filters`) and `used_llm`, which is `false` if the LLM call failed and the endpoint fell back to the raw query.

Interactive docs available at `/docs` once the server is running.

## Setup

```bash
pip install -r requirements.txt
python database.py        # initializes catalog.db
python main.py             # starts the server at http://127.0.0.1:8000
```

The smart endpoint needs an API key. Copy `.env.example` to `.env` and fill in `GROQ_API_KEY` (and optionally `LLM_MODEL`). The rest of the service works without it, and the smart endpoint falls back to plain search if the key is missing or the call fails.

## Testing

```bash
python -m pytest -v
```

12 tests, none of which need network access or an API key:

- Empty query handling
- Category filter correctness
- Price filter correctness
- Stock filter correctness
- **Candidate-pool regression test** — see "Known Limitations & Design Decisions" below
- LLM parser (7 tests): valid output, API errors, invalid JSON, negative price rejected, invented category dropped, category case normalized, empty query handled. The LLM call is isolated in one function and replaced in tests.

## LLM Query Understanding (`/api/search/smart`)

The plain `/api/search` endpoint embeds the whole sentence, so words like "under 500" and "in stock" are ignored by the embedding. The smart endpoint adds an LLM step in front of the existing pipeline. Retrieval and filtering are unchanged.

**Design decisions**
- **The LLM only extracts filters.** It never selects products, so results always come from the real catalog.
- **Schema validation.** Output is validated with Pydantic (for example, price must be positive).
- **Category safety.** The prompt lists the real categories from the database, and the result is mapped back to an exact database value or dropped, since the SQL filter is an exact match. A category is only set when the user names one, not inferred from the product type.
- **Graceful fallback.** If the LLM call fails or returns invalid JSON, the endpoint falls back to the original raw-query behaviour.
- **Transient errors.** Temporary server errors (503) are retried with backoff; quota errors are not.

**Evaluation** (`eval_llm.py`, synthetic 5,000-product catalog, top-5 results). Score = share of returned results that satisfy every constraint in the query.

| | Baseline | With LLM layer |
|---|---|---|
| Constraint satisfaction (30 queries) | 34.0% | 94.7% |
| Constraint satisfaction (5 queries added after prompt tuning) | 32.0% | 100% |
| Filter extraction accuracy (30 queries) | n/a | 100% |
| LLM fallback rate | n/a | 0% |
| Avg latency per query | 36 ms | 798 ms |

The prompt was refined once after a first run, where a category was sometimes missed, and 5 fresh queries were added afterwards to check the fix held. The query set is small and hand-written, and the metric measures constraint satisfaction, not real-world relevance. The LLM step adds roughly 0.7 s per query.

To reproduce: `python eval_llm.py` (results are cached in `parse_cache.json`; delete it after changing the prompt or model).

## Performance Benchmarks

Measured on synthetic catalogs of increasing size, averaged over 20 runs per size (see `benchmark.py`):

| Catalog Size | Embed (ms) | FAISS Search (ms) | SQL Filter (ms) | Avg Total (ms) |
|---|---|---|---|---|
| 30 | 8.66 | 0.102 | 0.664 | 9.43 |
| 500 | 8.37 | 0.231 | 1.328 | 9.93 |
| 5,000 | 11.51 | 0.931 | 2.964 | 15.40 |

**Observations:**
- Embedding time stays roughly flat (~8-11ms) regardless of catalog size, as expected — it only depends on the query string length, not catalog size.
- FAISS search time grows with catalog size (~9x from 30 to 5,000 products), consistent with `IndexFlatIP` performing an exact, linear-scan (O(N)) search rather than an approximate sub-linear one.
- At the tested scale, **embedding generation is the dominant cost**, not vector search — meaning the first optimization target for scaling further would be batching/caching embeddings, not switching FAISS index types.

## Known Limitations & Design Decisions

- **`IndexFlatIP` performs exact brute-force search — O(N), not sub-linear.** This is a deliberate choice at this catalog scale: it guarantees exact (not approximate) nearest-neighbor results with simple implementation. At millions of products, I'd switch to `IndexHNSWFlat` or `IndexIVFFlat` for approximate, sub-linear search.
- **Candidate-pool bug (fixed):** an earlier version retrieved only the top-20 FAISS matches before applying SQL filters, which could return too few (or zero) results if most of those 20 candidates were filtered out by price/stock, even when better matches existed further down the ranking. Fixed by widening the candidate pool (`max(200, top_k * 10)`) before filtering. This is directly covered by a regression test in `test_search.py`.
- **Candidate pool still limits tight filters.** With 5,000 products, a strict limit such as "under $80" can leave fewer than 5 results, because only a few of the top 200 FAISS matches are that cheap. This is why constraint satisfaction on the smart endpoint is 94.7%, not 100%, even though filter extraction was correct. Enlarging the pool when filters are active would address it.
- **LLM adds latency and an external dependency.** About 0.7 s per smart query, and a third-party API with rate limits. The fallback keeps search working when it is unavailable.
- **Single `products` table**, not normalized into separate inventory/warehouse tables — a deliberate scope decision for this project size. In production, I'd split stock/warehouse data into a separate table with a foreign key to support multiple warehouses per product.
- **FAISS index is rebuilt on every product insert.** Fine at this scale (hundreds–low thousands of products); would need incremental indexing or batched rebuilds at larger scale.
- **No authentication yet** — planned as a follow-up addition (JWT-based).