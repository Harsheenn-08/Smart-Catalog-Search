"""Compares baseline search (raw query, no filters) with LLM-assisted search.

Usage:
    python eval_llm.py --catalog     # prints categories and sample products
    python eval_llm.py               # runs the comparison on eval_queries.json

Successful LLM parses are cached in parse_cache.json, so re-running costs no API quota.
DELETE parse_cache.json whenever you change the prompt or the model.

Each entry in eval_queries.json has a "query" and the ground-truth constraints you expect:
    {"query": "...", "max_price": 150, "in_stock_only": true, "category": "Furniture"}
(max_price, in_stock_only and category are all optional.)

Per-query score = (top-5 results that satisfy ALL ground-truth constraints) / 5.
"""

import json
import os
import statistics
import sys
import time

from sentence_transformers import SentenceTransformer

import main
from database import get_database_connection, init_db
from faiss_index import VectorIndex
from query_parser import SearchFilters, parse_query

TOP_K = 5
QUERIES_FILE = "eval_queries.json"
CACHE_FILE = "parse_cache.json"
SLEEP_BETWEEN_QUERIES = float(os.getenv("EVAL_SLEEP", "4"))  # only used after a real API call


def setup():
    init_db()
    main.model = SentenceTransformer("all-MiniLM-L6-v2")
    main.vector_db = VectorIndex()
    main.vector_db.build_index_from_db()


def print_catalog():
    with get_database_connection() as conn:
        cats = conn.execute(
            "SELECT category, COUNT(*) AS n, MIN(price) AS lo, MAX(price) AS hi FROM products GROUP BY category"
        ).fetchall()
        for c in cats:
            print(f"\n{c['category']}  ({c['n']} products, ${c['lo']:.2f} - ${c['hi']:.2f})")
            samples = conn.execute(
                "SELECT title, price, stock_count FROM products WHERE category = ? LIMIT 4", (c["category"],)
            ).fetchall()
            for s in samples:
                print(f"   - {s['title']}  ${s['price']:.2f}  stock={s['stock_count']}")


def satisfies(item, truth):
    if truth.get("category") and item["category"] != truth["category"]:
        return False
    if truth.get("max_price") is not None and item["price"] > truth["max_price"]:
        return False
    if truth.get("in_stock_only") and item["stock_count"] <= 0:
        return False
    return True


def score(results, truth):
    return sum(satisfies(r, truth) for r in results) / TOP_K


def filters_correct(filters, truth):
    """Did the LLM extract the same constraints we wrote as ground truth?"""
    return (
        filters.category == truth.get("category")
        and filters.max_price == truth.get("max_price")
        and filters.in_stock_only == bool(truth.get("in_stock_only", False))
    )


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def get_filters(query, cache):
    """Returns (filters, used_llm, llm_ms). Only successful LLM parses are cached (with their latency)."""
    if query in cache:
        entry = cache[query]
        return SearchFilters(**entry["filters"]), True, entry["llm_ms"]
    t0 = time.perf_counter()
    filters, used_llm = parse_query(query)
    llm_ms = (time.perf_counter() - t0) * 1000
    if used_llm:
        cache[query] = {"filters": filters.model_dump(), "llm_ms": llm_ms}
        save_cache(cache)
    return filters, used_llm, llm_ms


def run():
    with open(QUERIES_FILE, encoding="utf-8") as f:
        cases = json.load(f)

    setup()
    main.search_products(query="warm up", top_k=1)  # first call is slower; keep it out of the timings
    cache = load_cache()

    rows = []
    for case in cases:
        q = case["query"]
        was_cached = q in cache

        t0 = time.perf_counter()
        base_results = main.search_products(query=q, top_k=TOP_K)
        base_ms = (time.perf_counter() - t0) * 1000

        filters, used_llm, llm_ms = get_filters(q, cache)
        t0 = time.perf_counter()
        smart_results = main.search_products(
            query=filters.semantic_query,
            category=filters.category,
            max_price=filters.max_price,
            in_stock_only=filters.in_stock_only,
            top_k=TOP_K,
        )
        smart_ms = llm_ms + (time.perf_counter() - t0) * 1000

        rows.append({
            "query": q,
            "baseline_score": score(base_results, case),
            "smart_score": score(smart_results, case),
            "baseline_ms": base_ms,
            "smart_ms": smart_ms,
            "used_llm": used_llm,
            "filters_correct": filters_correct(filters, case) if used_llm else False,
            "parsed": filters.model_dump(),
        })
        tag = "cached" if was_cached else ("llm" if used_llm else "FALLBACK")
        print(f"{q[:55]:55s} base={rows[-1]['baseline_score']:.2f} smart={rows[-1]['smart_score']:.2f} [{tag}]")
        if not was_cached:
            time.sleep(SLEEP_BETWEEN_QUERIES)

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    n = len(rows)
    ok = [r for r in rows if r["used_llm"]]
    print("\n=== Summary ===")
    print(f"Queries:                            {n}")
    print(f"LLM parsed successfully:            {len(ok)} / {n}  (fallback rate {1 - len(ok) / n:.1%})")
    if not ok:
        print("No successful LLM parses, so there is nothing to compare. Check the errors above.")
        return
    print(f"Compared on the {len(ok)} queries where the LLM worked:")
    print(f"   Constraint satisfaction, baseline: {statistics.mean(r['baseline_score'] for r in ok):.1%}")
    print(f"   Constraint satisfaction, LLM:      {statistics.mean(r['smart_score'] for r in ok):.1%}")
    print(f"   Filter extraction accuracy:        {sum(r['filters_correct'] for r in ok) / len(ok):.1%}")
    print(f"   Avg latency baseline / LLM:        {statistics.mean(r['baseline_ms'] for r in ok):.0f} ms / "
          f"{statistics.mean(r['smart_ms'] for r in ok):.0f} ms")
    print("\nPer-query details saved to eval_results.json")


if __name__ == "__main__":
    if "--catalog" in sys.argv:
        print_catalog()
    else:
        run()