import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_PATH = (
    PROJECT_ROOT / "evals" / "retrieval_benchmark.json"
)
REPORT_PATH = PROJECT_ROOT / "evals" / "retrieval_report.json"

TOP_K = 3


def load_benchmark():
    """Load and validate the evaluation dataset."""

    with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
        benchmark = json.load(file)

    corpus = benchmark["corpus"]
    queries = benchmark["queries"]

    if not corpus or not queries:
        raise ValueError("Corpus and queries must not be empty.")

    corpus_ids = [item["id"] for item in corpus]

    if len(corpus_ids) != len(set(corpus_ids)):
        raise ValueError("Corpus IDs must be unique.")

    known_ids = set(corpus_ids)

    for query in queries:
        missing_ids = (
            set(query["expected_source_ids"]) - known_ids
        )

        if missing_ids:
            raise ValueError(
                f"Query {query['id']} references unknown sources: "
                f"{sorted(missing_ids)}"
            )

        if query["answerable"] and not query["expected_source_ids"]:
            raise ValueError(
                f"Answerable query {query['id']} needs an expected source."
            )

    return benchmark


def calculate_metrics(
    corpus,
    queries,
    model,
    threshold,
):
    """Measure retrieval ranking and threshold-based abstention."""

    texts = [item["text"] for item in corpus]

    print("\nGenerating corpus embeddings...")

    corpus_embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    corpus_embeddings = np.asarray(
        corpus_embeddings,
        dtype=np.float32,
    )

    answerable_queries = []
    unanswerable_queries = []
    query_results = []

    print("\nEvaluating questions...")

    for query in queries:
        question_embedding = model.encode(
            query["question"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        question_embedding = np.asarray(
            question_embedding,
            dtype=np.float32,
        )

        # Normalized vectors allow cosine similarity
        # to be calculated using the dot product.
        scores = corpus_embeddings @ question_embedding
        ranked_indices = np.argsort(-scores)

        ranked_corpus = [
            {
                "source_id": corpus[int(index)]["id"],
                "filename": corpus[int(index)]["filename"],
                "similarity": float(scores[int(index)]),
            }
            for index in ranked_indices
        ]

        top_results = ranked_corpus[:TOP_K]
        best_score = ranked_corpus[0]["similarity"]

        expected_ids = set(query["expected_source_ids"])

        relevant_ranks = [
            rank
            for rank, item in enumerate(ranked_corpus, start=1)
            if item["source_id"] in expected_ids
        ]

        first_relevant_rank = (
            relevant_ranks[0] if relevant_ranks else None
        )

        recall_at_k_hit = (
            first_relevant_rank is not None
            and first_relevant_rank <= TOP_K
        )

        reciprocal_rank = (
            1.0 / first_relevant_rank
            if first_relevant_rank is not None
            else 0.0
        )

        accepted_by_threshold = best_score >= threshold

        result = {
            "query_id": query["id"],
            "question": query["question"],
            "answerable": query["answerable"],
            "expected_source_ids": sorted(expected_ids),
            "top_results": top_results,
            "best_similarity": round(best_score, 4),
            "accepted_by_threshold": accepted_by_threshold,
            "expected_source_rank": first_relevant_rank,
            "recall_at_3_hit": recall_at_k_hit,
        }

        query_results.append(result)

        if query["answerable"]:
            answerable_queries.append({
                "recall_at_3_hit": recall_at_k_hit,
                "reciprocal_rank": reciprocal_rank,
                "accepted": accepted_by_threshold,
            })
        else:
            unanswerable_queries.append({
                "abstained": not accepted_by_threshold,
            })

        status = "ANSWERABLE" if query["answerable"] else "UNANSWERABLE"

        print(
            f"{query['id']} | {status:<12} | "
            f"best={best_score:.3f} | "
            f"expected_rank={first_relevant_rank}"
        )

    answerable_count = len(answerable_queries)
    unanswerable_count = len(unanswerable_queries)

    recall_at_3 = (
        sum(item["recall_at_3_hit"] for item in answerable_queries)
        / answerable_count
        if answerable_count
        else 0.0
    )

    mean_reciprocal_rank = (
        sum(item["reciprocal_rank"] for item in answerable_queries)
        / answerable_count
        if answerable_count
        else 0.0
    )

    answerable_acceptance_rate = (
        sum(item["accepted"] for item in answerable_queries)
        / answerable_count
        if answerable_count
        else 0.0
    )

    unanswerable_abstention_rate = (
        sum(item["abstained"] for item in unanswerable_queries)
        / unanswerable_count
        if unanswerable_count
        else 0.0
    )

    metrics = {
        "corpus_size": len(corpus),
        "query_count": len(queries),
        "answerable_query_count": answerable_count,
        "unanswerable_query_count": unanswerable_count,
        "recall_at_3": round(recall_at_3, 4),
        "mean_reciprocal_rank": round(
            mean_reciprocal_rank, 4
        ),
        "answerable_acceptance_rate": round(
            answerable_acceptance_rate, 4
        ),
        "unanswerable_abstention_rate": round(
            unanswerable_abstention_rate, 4
        ),
        "threshold": threshold,
    }

    return metrics, query_results


def main():
    print("=" * 55)
    print("       DocuMind AI - Retrieval Evaluation")
    print("=" * 55)

    benchmark = load_benchmark()

    model_name = benchmark["embedding_model"]
    threshold = float(
        benchmark["initial_abstention_threshold"]
    )

    print(f"\nBenchmark: {benchmark['benchmark_name']}")
    print(f"Corpus documents: {len(benchmark['corpus'])}")
    print(f"Evaluation queries: {len(benchmark['queries'])}")
    print(f"Embedding model: {model_name}")
    print(f"Abstention threshold: {threshold}")

    print("\nLoading embedding model...")
    model = SentenceTransformer(model_name)

    metrics, query_results = calculate_metrics(
        corpus=benchmark["corpus"],
        queries=benchmark["queries"],
        model=model,
        threshold=threshold,
    )

    report = {
        "benchmark_name": benchmark["benchmark_name"],
        "embedding_model": model_name,
        "metrics": metrics,
        "query_results": query_results,
        "limitations": [
            "This is a small synthetic benchmark.",
            "Recall@3 and MRR measure retrieval, not answer correctness.",
            "Threshold acceptance does not prove that an answer is supported.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 55)
    print("EVALUATION SUMMARY")
    print("=" * 55)

    for name, value in metrics.items():
        print(f"{name}: {value}")

    print(f"\nDetailed report saved to: {REPORT_PATH}")
    print("Evaluation completed.")


if __name__ == "__main__":
    main()