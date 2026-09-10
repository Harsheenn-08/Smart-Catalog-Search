import json
import sqlite3
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

# 2. Define sample multi-category products
SAMPLE_PRODUCTS = [
    # Electronics
    {
        "title": "Noise-Canceling Wireless Headphones",
        "description": "Over-ear Bluetooth headphones with active noise cancellation, deep bass, and 30-hour battery life.",
        "category": "Electronics",
        "price": 199.99,
        "stock_count": 45,
        "warehouse_location": "US-East"
    },
    {
        "title": "Mechanical Gaming Keyboard",
        "description": "RGB backlit wired mechanical keyboard with tactile blue switches and custom macro support.",
        "category": "Electronics",
        "price": 89.50,
        "stock_count": 120,
        "warehouse_location": "US-West"
    },
    {
        "title": "Ultra-Wide Curved Monitor 34-inch",
        "description": "144Hz WQHD curved display designed for immersive gaming and productivity multi-tasking.",
        "category": "Electronics",
        "price": 449.00,
        "stock_count": 15,
        "warehouse_location": "US-Central"
    },
    
    # Home & Office
    {
        "title": "Ergonomic Mesh Executive Chair",
        "description": "High-back desk chair with adjustable lumbar support, 3D armrests, and breathable mesh for long work sessions.",
        "category": "Home & Office",
        "price": 249.99,
        "stock_count": 30,
        "warehouse_location": "US-East"
    },
    {
        "title": "Electric Standing Desk Frame",
        "description": "Dual-motor height adjustable desk base with digital memory presets and solid steel construction.",
        "category": "Home & Office",
        "price": 299.00,
        "stock_count": 0,  # Testing out-of-stock scenario
        "warehouse_location": "US-West"
    },

    # Kitchen & Appliances
    {
        "title": "Programmable Espresso & Coffee Machine",
        "description": "15-bar Italian pump coffee maker with integrated milk frother for lattes and cappuccinos.",
        "category": "Kitchen",
        "price": 179.95,
        "stock_count": 60,
        "warehouse_location": "US-Central"
    },
    {
        "title": "Air Fryer XL 6-Quart",
        "description": "Rapid air circulation technology for healthy oil-free cooking with digital touchscreen controls.",
        "category": "Kitchen",
        "price": 99.99,
        "stock_count": 85,
        "warehouse_location": "US-East"
    },

    # Fitness & Outdoors
    {
        "title": "Adjustable Select Dumbbell Set",
        "description": "All-in-one weight dumbbell replacing 15 sets of weights, adjustable from 5 to 52 lbs.",
        "category": "Fitness",
        "price": 329.00,
        "stock_count": 20,
        "warehouse_location": "US-West"
    },
    {
        "title": "Non-Slip Eco-Friendly Yoga Mat",
        "description": "Extra thick natural rubber workout mat with alignment markings and carrying strap.",
        "category": "Fitness",
        "price": 39.99,
        "stock_count": 110,
        "warehouse_location": "US-East"
    }
]

def generate_batch_embeddings(products):
    print(f"Generating vector embeddings for {len(products)} products...")
    text_corpus = [f"{p['title']}: {p['description']}" for p in products]
    vectors = model.encode(text_corpus)
    return vectors

def seed_database(products):
  # 1. Compute vectors
  vectors = generate_batch_embeddings(products)

  # 2. Format records into tuples matching SQL columns
  db_records = []
  for product, vector in zip(products, vectors):
    vector_json = json.dumps(vector.tolist())
    record = (
        product["title"],
        product["description"],
        product["category"],
        product["price"],
        product["stock_count"],
        product["warehouse_location"],
        vector_json,
    )
    db_records.append(record)

  # 3. Insert into catalog.db using SQLite
  conn = sqlite3.connect("catalog.db")
  cursor = conn.cursor()

  # Wipe old table entries first
  cursor.execute("DELETE FROM products;")

  # Query matching our database schema
  insert_query = """
    INSERT INTO products (title, description, category, price, stock_count, warehouse_location, embedding)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """

  cursor.executemany(insert_query, db_records)
  conn.commit()
  conn.close()

  print(f"Successfully seeded {len(db_records)} products into catalog.db!")

if __name__ == "__main__":
    seed_database(SAMPLE_PRODUCTS)