from main import search_products
from conftest import seed_product


def test_empty_query_does_not_crash(test_db):
    """
    WHY THIS TEST EXISTS:
    A real user will eventually submit an empty search box. This proves
    the system handles it gracefully (returns something, doesn't throw
    an unhandled exception) instead of crashing the API.
    """
    seed_product("Wireless Mouse", "Electronics", price=25.0, stock_count=10)

    results = search_products(query="", top_k=5)

    assert isinstance(results, list)


def test_category_filter_excludes_non_matching_products(test_db):
    """
    WHY THIS TEST EXISTS:
    Proves the category filter is actually applied in the SQL query,
    not silently ignored.
    """
    seed_product("Office Chair", "Furniture", price=150.0, stock_count=5)
    seed_product("Wireless Mouse", "Electronics", price=25.0, stock_count=10)

    results = search_products(query="office chair", category="Electronics", top_k=5)

    for item in results:
        assert item["category"] == "Electronics"


def test_max_price_excludes_expensive_products(test_db):
    """
    WHY THIS TEST EXISTS:
    Proves the hard SQL price constraint is really enforced.
    """
    seed_product("Budget Desk Lamp", "Furniture", price=20.0, stock_count=5)
    seed_product("Luxury Desk Lamp", "Furniture", price=999.0, stock_count=5)

    results = search_products(query="desk lamp", max_price=50.0, top_k=5)

    for item in results:
        assert item["price"] <= 50.0


def test_in_stock_only_excludes_zero_stock_products(test_db):
    """
    WHY THIS TEST EXISTS:
    Out-of-stock items showing up in search results is a real bug —
    a customer shouldn't be shown something they can't buy.
    """
    seed_product("Sold Out Keyboard", "Electronics", price=40.0, stock_count=0)
    seed_product("Available Keyboard", "Electronics", price=45.0, stock_count=8)

    results = search_products(query="keyboard", in_stock_only=True, top_k=5)

    titles = [item["title"] for item in results]
    assert "Sold Out Keyboard" not in titles
    assert "Available Keyboard" in titles


def test_candidate_pool_regression(test_db):
    """
    WHY THIS TEST EXISTS — the most important test in the file.
    Recreates the exact bug scenario we fixed: many cheap "decoy"
    products that are semantically similar but filtered out by price,
    plus ONE valid product that should still be found because the
    candidate pool is now wide enough to reach it.
    """
    for i in range(30):
        seed_product(f"Gaming Laptop Model {i}", "Electronics", price=2000.0, stock_count=5)

    seed_product("Budget Gaming Laptop", "Electronics", price=499.0, stock_count=5)

    results = search_products(
        query="gaming laptop with high performance",
        max_price=500.0,
        top_k=5,
    )

    titles = [item["title"] for item in results]
    assert "Budget Gaming Laptop" in titles