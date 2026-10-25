"""Turns a natural-language shopping query into structured search filters using an LLM.

The LLM only *extracts* filters. Retrieval (FAISS) and filtering (SQLite) stay
exactly as they were in main.search_products().
"""

import os
import time
from typing import List, Optional, Tuple

from groq import Groq
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

from database import get_database_connection

load_dotenv()

# Check Google's docs for the current free-tier model name and override via .env if needed.
MODEL_NAME = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")

class SearchFilters(BaseModel):
    """Mirrors the parameters that search_products() already accepts."""

    semantic_query: str = Field(description="What the user wants, without price/stock words.")
    category: Optional[str] = Field(default=None, description="Exactly one allowed category, or null.")
    max_price: Optional[float] = Field(default=None, description="Upper price limit, or null.")
    in_stock_only: bool = Field(default=False, description="True only if the user asks for in-stock items.")

    @field_validator("max_price")
    @classmethod
    def price_must_be_positive(cls, v):
        # Enforced here instead of Field(gt=0): gt=0 puts "exclusiveMinimum" in the JSON schema, which Gemini rejects.
        if v is not None and v <= 0:
            raise ValueError("max_price must be positive")
        return v


_client = None


def _get_client():
    global _client
    if _client is None:
        _client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _client


def get_categories() -> List[str]:
    """Real category values from the DB. The SQL filter is an exact match, so the LLM must use these."""
    with get_database_connection() as conn:
        rows = conn.execute("SELECT DISTINCT category FROM products ORDER BY category").fetchall()
    return [row["category"] for row in rows]


def build_prompt(user_query: str, categories: List[str]) -> str:
    return f"""You extract search filters from an e-commerce shopping query.
Return JSON with these fields:
- semantic_query: what the user wants, with price and stock wording removed. Never empty.
- category: set ONLY if the user's wording contains one of these category names: {categories}. Match it in any capitalization and in singular or plural form (a plural or lowercase mention still counts), and return the exact spelling from the list. Do NOT infer a category from the product type (for example "chair" or "monitor" alone means null). Otherwise null.
- max_price: a number if the user states an upper price limit (e.g. "under 150", "below $200", "max 80"), else null.
- in_stock_only: true only if the user asks for items that are available / in stock, else false.

Rules: never invent a category that is not in the list. Do not guess a price the user did not state.
"cheap" or "affordable" with no number means max_price is null.

Examples:
Query: grey office chair under 150 dollars, in stock
JSON: {{"semantic_query": "grey office chair", "category": null, "max_price": 150, "in_stock_only": true}}
Query: something nice for a small bedroom
JSON: {{"semantic_query": "something nice for a small bedroom", "category": null, "max_price": null, "in_stock_only": false}}

Query: {user_query}
JSON:"""


def _call_llm(user_query: str, categories: List[str]) -> str:
    """Isolated so tests can replace it without touching the network."""
    response = _get_client().chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": build_prompt(user_query, categories)}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    return response.choices[0].message.content


def _normalize_category(category: Optional[str], categories: List[str]) -> Optional[str]:
    """Map the LLM's category onto a real DB value (case-insensitive), or drop it."""
    if not category:
        return None
    lookup = {c.lower(): c for c in categories}
    return lookup.get(category.strip().lower())

def _call_with_retry(user_query: str, categories: List[str], attempts: int = 3) -> str:
    """Retry only temporary server errors (503). Quota errors (429) fail immediately, since retrying can't help."""
    for attempt in range(attempts):
        try:
            return _call_llm(user_query, categories)
        except Exception as exc:
            transient = "503" in str(exc) or "UNAVAILABLE" in str(exc)
            if not transient or attempt == attempts - 1:
                raise
            time.sleep(2 ** (attempt + 1))

def parse_query(user_query: str) -> Tuple[SearchFilters, bool]:
    """Returns (filters, used_llm). On any failure, falls back to the raw query with no filters."""
    fallback = SearchFilters(semantic_query=user_query)
    try:
        categories = get_categories()
        raw = _call_with_retry(user_query, categories)
        filters = SearchFilters.model_validate_json(raw)  # validates types and constraints
    except Exception as exc:  # network error, quota, bad JSON, validation error
        print(f"[query_parser] LLM parsing failed, using fallback: {exc}")
        return fallback, False

    filters.category = _normalize_category(filters.category, categories)
    if not filters.semantic_query.strip():
        filters.semantic_query = user_query
    return filters, True