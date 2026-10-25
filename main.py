import json
import time
from contextlib import asynccontextmanager
from typing import List, Optional, Dict , Any
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from faiss_index import VectorIndex
from database import get_database_connection, init_db
from query_parser import parse_query

# Global resources initialized during application startup
model: Optional[SentenceTransformer] = None
vector_db: Optional[VectorIndex] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager replacing deprecated @app.on_event handlers."""
    global model, vector_db
    print("Initializing database, embedding model, and FAISS vector index...")
    init_db()
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vector_db = VectorIndex()
    vector_db.build_index_from_db()
    yield
    print("Shutting down search service resources...")

app = FastAPI(
    title="Smart Catalog Search Service",
    description="E-commerce search service with pre-computed vector embeddings and hybrid SQL filtering",
    version="0.2.0",
    lifespan=lifespan
)

# Pydantic V2 Schemas using json_schema_extra
class ProductCreate(BaseModel):
    title: str = Field(..., json_schema_extra={"example": "Ergonomic Mesh Office Chair"})
    category: str = Field(..., json_schema_extra={"example": "Furniture"})
    description: str = Field(..., json_schema_extra={"example": "High back mesh chair with lumbar support"})
    price: float = Field(..., gt=0, json_schema_extra={"example": 199.99})
    stock_count: int = Field(..., ge=0, json_schema_extra={"example": 45})
    warehouse_location: str = Field(..., json_schema_extra={"example": "US-East"})

class ProductResponse(BaseModel):
    id: int
    title: str
    category: str
    description: str
    price: float
    stock_count: int
    warehouse_location: str
    similarity_score: Optional[float] = None

def compute_embedding(text: str) -> List[float]:
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding.tolist()

@app.get("/")
def home():
    return {"message": "Smart Catalog Search API is up and running"}

@app.post("/api/products", response_model=ProductResponse, status_code=201)
def create_product(product: ProductCreate):
    # Include description in embedding generation for higher semantic quality
    text_to_embed = f"{product.title} {product.category} {product.description}"
    vector = compute_embedding(text_to_embed)
    vector_json = json.dumps(vector)

    with get_database_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO products (title, category, description, price, stock_count, warehouse_location, embedding)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                product.title,
                product.category,
                product.description,
                product.price,
                product.stock_count,
                product.warehouse_location,
                vector_json,
            ),
        )
        conn.commit()
        new_id = cursor.lastrowid

    vector_db.build_index_from_db()

    return ProductResponse(
        id=new_id,
        title=product.title,
        category=product.category,
        description=product.description,
        price=product.price,
        stock_count=product.stock_count,
        warehouse_location=product.warehouse_location,
    )

from typing import List, Optional, Dict, Any
from fastapi import Query, HTTPException

@app.get("/api/search")
def search_products(
    query: str,
    category: Optional[str] = None,
    max_price: Optional[float] = None,
    in_stock_only: bool = False,
    top_k: int = 5,
    return_timings: bool = False
):
    """
    Integrates FAISS vector retrieval with SQLite metadata filtering.
    Plain Python defaults are used (no Query(...) wrapper) so this function
    works correctly both as a FastAPI route AND when called directly from
    Python (e.g. benchmark scripts, pytest) without any special handling.

    If return_timings=True, returns a payload containing both results and
    step-by-step latency metrics.
    """
    if model is None or vector_db is None:
        raise HTTPException(status_code=500, detail="Search engine services not initialized")

    # Phase 1: Query Embedding
    t0 = time.perf_counter()
    query_vector = model.encode(query)
    t1 = time.perf_counter()

    # Phase 2: FAISS Candidate Retrieval (Expanded Candidate Pool = max(200, top_k * 10))
    candidate_pool_size = max(200, top_k * 10)
    matched_ids, similarity_scores = vector_db.search(query_vector, top_k=candidate_pool_size)
    t2 = time.perf_counter()

    if not matched_ids:
        results = []
        t3 = time.perf_counter()
    else:
        score_map = dict(zip(matched_ids, similarity_scores))

        placeholders = ",".join(["?"] * len(matched_ids))
        conditions = [f"id IN ({placeholders})"]
        params = list(matched_ids)

        if category:
            conditions.append("category = ?")
            params.append(category)
        if max_price is not None:
            conditions.append("price <= ?")
            params.append(max_price)
        if in_stock_only:
            conditions.append("stock_count > 0")

        where_clause = "WHERE " + " AND ".join(conditions)
        sql_query = f"SELECT id, title, category, description, price, stock_count, warehouse_location FROM products {where_clause}"

        with get_database_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(sql_query, params)
            rows = cursor.fetchall()

        scored_results = []
        for row in rows:
            item = dict(row)
            item["similarity_score"] = round(score_map[item["id"]], 4)
            scored_results.append(item)

        scored_results.sort(key=lambda x: x["similarity_score"], reverse=True)
        results = scored_results[:top_k]
        t3 = time.perf_counter()

    if return_timings:
        return {
            "results": results,
            "timings": {
                "embed_ms": (t1 - t0) * 1000,
                "faiss_ms": (t2 - t1) * 1000,
                "sql_ms": (t3 - t2) * 1000,
                "total_ms": (t3 - t0) * 1000,
            }
        }

    return results

@app.get("/api/search/smart")
def smart_search(query: str, top_k: int = 5):
    """LLM extracts filters from the sentence, then the existing FAISS + SQLite pipeline runs."""
    filters, used_llm = parse_query(query)
    results = search_products(
        query=filters.semantic_query,
        category=filters.category,
        max_price=filters.max_price,
        in_stock_only=filters.in_stock_only,
        top_k=top_k,
    )
    return {
        "parsed_filters": filters.model_dump(),
        "used_llm": used_llm,  # False means the LLM call failed and we fell back to the raw query
        "results": results,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)