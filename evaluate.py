from main import search_products_logic, lifespan, app

def evaluate_search_engine():
    # Test queries paired with relevant expected product IDs
    test_cases = [
        {"query": "gaming monitor", "relevant_ids": [2, 3]},
        {"query": "ergonomic office chair", "relevant_ids": [4]}
    ]

    precisions, recalls, reciprocal_ranks = [], [], []

    for test in test_cases:
        results = search_products_logic(query=test["query"], top_k=5)
        retrieved_ids = [item["id"] for item in results]

        relevant = set(test["relevant_ids"])
        retrieved = set(retrieved_ids)
        hits = len(relevant.intersection(retrieved))

        precision = hits / len(retrieved) if retrieved else 0.0
        recall = hits / len(relevant) if relevant else 0.0

        # Calculate Reciprocal Rank
        rr = 0.0
        for rank, item_id in enumerate(retrieved_ids, start=1):
            if item_id in relevant:
                rr = 1.0 / rank
                break

        precisions.append(precision)
        recalls.append(recall)
        reciprocal_ranks.append(rr)

    print("\n=== SEARCH RELEVANCE EVALUATION ===")
    print(f"Mean Precision@5 : {sum(precisions)/len(precisions):.4f}")
    print(f"Mean Recall@5    : {sum(recalls)/len(recalls):.4f}")
    print(f"MRR              : {sum(reciprocal_ranks)/len(reciprocal_ranks):.4f}")
    print("===================================")

if __name__ == "__main__":
    import asyncio
    async def run_eval():
        async with lifespan(app):
            evaluate_search_engine()
    asyncio.run(run_eval())