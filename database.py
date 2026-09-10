import sqlite3

database_name = "catalog.db"
def get_database_connection():
    """Establishes and returns a connection to the catalog database with row factory set"""
    conn = sqlite3.connect(database_name)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn 

def init_db():
    conn = get_database_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            category TEXT NOT NULL,
            description TEXT NOT NULL,
            price REAL NOT NULL,
            stock_count INTEGER NOT NULL DEFAULT 0,
            warehouse_location TEXT NOT NULL,
            embedding TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """)

    # Phase 5: B-tree indexes on metadata filtering columns
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_category ON products(category);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_price ON products(price);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_stock ON products(stock_count);")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
    print("Database initialized sccessfully! with updated schema and metadata indexes")