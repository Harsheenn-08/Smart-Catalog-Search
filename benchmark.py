import time
import json
import random
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

# Import modules from main
import main
from main import search_products, vector_db
from database import get_database_connection

CATALOG_SIZES = [30, 500, 5000]
NUM_RUNS = 20

ADJECTIVES = ["Ergonomic", "Wireless", "Smart", "Compact", "Heavy-Duty", "Ultra-Fast", "Eco-Friendly"]
NOUNS = ["Desk", "Chair", "Monitor", "Headset", "Keyboard", "Laptop", "Mouse"]
CATEGORIES = ["Furniture", "Electronics", "Office Supplies", "Accessories"]
LOCATIONS = ["US-East", "US-West", "EU-Central", "APAC-South"]

def init_services():
    """Ensures model and vector_db are initialized on main before running benchmarks."""
    if main.model is None:
        print("Initializing SentenceTransformer model ('all-MiniLM-L6-v2')...")
        main.model = SentenceTransformer("all-MiniLM-L6-v2")
    if main.vector_db is None:
        from faiss_index import VectorIndex
        main.vector_db = VectorIndex(dimension=384)

def seed_database(num_products: int):
    """Populates catalog.db with synthetic data and pre-computed embeddings."""
    products = []
    for _ in range(num_products):
        title = f"{random.choice(ADJECTIVES)} {random.choice(NOUNS)}"
        category = random.choice(CATEGORIES)
        description = f"High performance {title.lower()} engineered for daily use."
        price = round(random.uniform(50.0, 2000.0), 2)
        stock_count = random.randint(0, 50)
        warehouse = random.choice(LOCATIONS)
        
        text = f"{title} {category} {description}"
        # Access model via main.model to guarantee it's initialized
        vec = main.model.encode(text).tolist()
        products.append((title, category, description, price, stock_count, warehouse, json.dumps(vec)))

    with get_database_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("DROP TABLE IF EXISTS products;")
        cursor.execute("""
            CREATE TABLE products (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                price REAL NOT NULL,
                stock_count INTEGER NOT NULL,
                warehouse_location TEXT NOT NULL,
                embedding TEXT NOT NULL
            );
        """)
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_price ON products(price);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock ON products(stock_count);")
        
        cursor.executemany("""
            INSERT INTO products (title, category, description, price, stock_count, warehouse_location, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, products)
        conn.commit()

def run_benchmarks():
    # Initialize services first
    init_services()
    
    test_query = "gaming laptop with high performance"
    
    print("\n=================================== HYBRID ENGINE SCALE BENCHMARK ===================================")
    print(f"{'Catalog Size':<12} | {'Embed (ms)':<12} | {'FAISS (ms)':<12} | {'SQL Filter (ms)':<15} | {'Avg Total (ms)':<14} | {'P95 Total (ms)':<14}")
    print("-" * 95)

    for size in CATALOG_SIZES:
        seed_database(size)
        main.vector_db.build_index_from_db()
        
        # Warm-up run (discarded)
        _ = search_products(
            query=test_query,
            max_price=1500.0,
            in_stock_only=True,
            top_k=5,
            return_timings=True
        )

        embed_times, faiss_times, sql_times, total_times = [], [], [], []

        for _ in range(NUM_RUNS):
            res = search_products(
                query=test_query,
                max_price=1500.0,
                in_stock_only=True,
                top_k=5,
                return_timings=True
            )
            timings = res["timings"]
            embed_times.append(timings["embed_ms"])
            faiss_times.append(timings["faiss_ms"])
            sql_times.append(timings["sql_ms"])
            total_times.append(timings["total_ms"])

        avg_embed = np.mean(embed_times)
        avg_faiss = np.mean(faiss_times)
        avg_sql = np.mean(sql_times)
        avg_total = np.mean(total_times)
        p95_total = np.percentile(total_times, 95)

        print(f"{size:<12} | {avg_embed:<12.2f} | {avg_faiss:<12.3f} | {avg_sql:<15.3f} | {avg_total:<14.2f} | {p95_total:<14.2f}")

    print("=====================================================================================================\n")

if __name__ == "__main__":
    run_benchmarks()