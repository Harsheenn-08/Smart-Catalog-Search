import os
import json
import pytest
import database
import main
from sentence_transformers import SentenceTransformer
from faiss_index import VectorIndex


@pytest.fixture(scope="session")
def real_model():
    """
    Loads the real embedding model ONCE for the whole test session.
    Loading it fresh for every single test would be very slow (a few
    seconds each time), so 'scope=session' means pytest reuses this
    across all tests instead of reloading it per test.
    """
    return SentenceTransformer("all-MiniLM-L6-v2")


@pytest.fixture
def test_db(tmp_path, monkeypatch, real_model):
    """
    Runs before (and cleans up after) EVERY test that asks for it as an argument.

    - tmp_path: a pytest built-in fixture that gives us a fresh temporary
      folder, automatically deleted after the test.
    - monkeypatch: a pytest built-in fixture that lets us temporarily
      override a variable (here, database.database_name) and automatically
      restores the original value after the test, even if the test fails.
    """
    temp_db_path = str(tmp_path / "test_catalog.db")

    # Point database.py's connection function at our throwaway file instead
    # of the real catalog.db, for the duration of this test only.
    monkeypatch.setattr(database, "database_name", temp_db_path)

    database.init_db()

    # Wire up the real model + a fresh empty FAISS index for this test
    main.model = real_model
    main.vector_db = VectorIndex()

    yield temp_db_path  # <-- the test function runs at this point

    # (tmp_path cleans up the file automatically, nothing else needed here)


def seed_product(title, category, price, stock_count, warehouse="US-East"):
    """
    Helper (not a fixture, just a plain function) to insert one product
    with a real computed embedding, then rebuild the FAISS index so it's
    searchable. Tests call this directly to set up their specific scenario.
    """
    description = f"{title} in the {category} category, a great product."
    text_to_embed = f"{title} {category} {description}"
    vector = main.model.encode(text_to_embed).tolist()

    with database.get_database_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO products (title, category, description, price, stock_count, warehouse_location, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (title, category, description, price, stock_count, warehouse, json.dumps(vector)),
        )
        conn.commit()

    main.vector_db.build_index_from_db()