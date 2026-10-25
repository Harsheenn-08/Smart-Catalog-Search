"""Tests for the LLM query parser. No network or API key needed: the LLM call is replaced."""

import json

import pytest

import query_parser
from query_parser import parse_query

QUERY = "grey office chair under 150 in stock"


@pytest.fixture(autouse=True)
def fake_categories(monkeypatch):
    monkeypatch.setattr(query_parser, "get_categories", lambda: ["Furniture", "Electronics"])


def fake_llm(payload):
    def _call(user_query, categories):
        return payload if isinstance(payload, str) else json.dumps(payload)

    return _call


def test_valid_response_is_parsed(monkeypatch):
    monkeypatch.setattr(query_parser, "_call_llm", fake_llm(
        {"semantic_query": "grey office chair", "category": "Furniture", "max_price": 150, "in_stock_only": True}))
    filters, used_llm = parse_query(QUERY)
    assert used_llm is True
    assert filters.semantic_query == "grey office chair"
    assert filters.category == "Furniture"
    assert filters.max_price == 150
    assert filters.in_stock_only is True


def test_llm_exception_falls_back_to_raw_query(monkeypatch):
    def boom(user_query, categories):
        raise RuntimeError("quota exceeded")

    monkeypatch.setattr(query_parser, "_call_llm", boom)
    filters, used_llm = parse_query(QUERY)
    assert used_llm is False
    assert filters.semantic_query == QUERY
    assert filters.category is None and filters.max_price is None and filters.in_stock_only is False


def test_invalid_json_falls_back(monkeypatch):
    monkeypatch.setattr(query_parser, "_call_llm", fake_llm("this is not json"))
    filters, used_llm = parse_query(QUERY)
    assert used_llm is False
    assert filters.semantic_query == QUERY


def test_negative_price_fails_validation_and_falls_back(monkeypatch):
    monkeypatch.setattr(query_parser, "_call_llm", fake_llm(
        {"semantic_query": "chair", "category": None, "max_price": -5, "in_stock_only": False}))
    _, used_llm = parse_query(QUERY)
    assert used_llm is False


def test_invented_category_is_dropped(monkeypatch):
    monkeypatch.setattr(query_parser, "_call_llm", fake_llm(
        {"semantic_query": "chair", "category": "Spaceships", "max_price": None, "in_stock_only": False}))
    filters, used_llm = parse_query(QUERY)
    assert used_llm is True
    assert filters.category is None


def test_category_case_is_normalized(monkeypatch):
    monkeypatch.setattr(query_parser, "_call_llm", fake_llm(
        {"semantic_query": "chair", "category": "furniture", "max_price": None, "in_stock_only": False}))
    filters, _ = parse_query(QUERY)
    assert filters.category == "Furniture"


def test_empty_semantic_query_uses_original(monkeypatch):
    monkeypatch.setattr(query_parser, "_call_llm", fake_llm(
        {"semantic_query": "  ", "category": None, "max_price": 100, "in_stock_only": False}))
    filters, _ = parse_query(QUERY)
    assert filters.semantic_query == QUERY