# Smart Catalog Search Service

A hybrid e-commerce search engine combining SQL-based structured filtering with semantic vector search, built with FastAPI, SQLite, and SentenceTransformers.

## Overview

Instead of plain keyword matching, this service lets users search naturally (e.g., "gaming laptop with high performance") while enforcing hard constraints like price, stock availability, and category — the same pattern used by real e-commerce search systems.

**Pipeline:**
1. Query text is embedded into a 384-dimensional vector using `all-MiniLM-L6-v2` (SentenceTransformers)
2. FAISS (`IndexFlatIP`) retrieves a wide candidate pool of semantically similar products
3. SQLite applies hard filters (price, category, stock) on that candidate pool, using B-tree indexes
4. Remaining candidates are ranked by cosine similarity and returned as JSON

## Tech Stack

- **Backend:** Python, FastAPI, Pydantic
- **Database:** SQLite (B-tree indexes on `price`, `category`, `stock_count`)
- **Embeddings:** SentenceTransformers (`all-MiniLM-L6-v2`, local model, no external API calls)
- **Vector Search:** FAISS (`IndexFlatIP`)
- **Testing:** pytest
- **Deployment:** Docker

## API Endpoints

### `POST /api/products`
Creates a product and computes/stores its embedding.

### `GET /api/search`
Searches the catalog. Query params: `query`, `category`, `max_price`, `in_stock_only`, `top_k`.

Interactive docs available at `/docs` once the server is running.

## Setup

```bash
pip install -r requirements.txt
python database.py        # initializes catalog.db
python main.py             # starts the server at http://127.0.0.1:8000
```

## Testing

```bash
python -m pytest -v
```

5 tests covering:
- Empty query handling
- Category filter correctness
- Price filter correctness
- Stock filter correctness
- **Candidate-pool regression test** — see "Known Limitations & Design Decisions" below

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
- **Single `products` table**, not normalized into separate inventory/warehouse tables — a deliberate scope decision for this project size. In production, I'd split stock/warehouse data into a separate table with a foreign key to support multiple warehouses per product.
- **FAISS index is rebuilt on every product insert.** Fine at this scale (hundreds–low thousands of products); would need incremental indexing or batched rebuilds at larger scale.
- **No authentication yet** — planned as a follow-up addition (JWT-based).
