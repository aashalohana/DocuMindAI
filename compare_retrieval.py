

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parent
BENCHMARK_PATH = (
    PROJECT_ROOT / "evals" / "hard_negative_benchmark.json"
)
REPORT_PATH = (
    PROJECT_ROOT / "evals" / "retrieval_comparison_report.json"
)

TOP_K = 3


def load_benchmark():
    """Load and validate the hard-negative benchmark."""

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
        unknown_ids = (
            set(query["expected_source_ids"]) - known_ids
        )

        if unknown_ids:
            raise ValueError(
                f"{query['id']} refers to unknown sources: "
                f"{sorted(unknown_ids)}"
            )

        if query["answerable"] and not query["expected_source_ids"]:
            raise ValueError(
                f"{query['id']} needs at least one expected source."
            )

        if (
            not query["answerable"]
            and query["expected_source_ids"]
        ):
            raise ValueError(
                f"{query['id']} is unanswerable but has expected sources."
            )

    return benchmark


def evaluate_rankings(
    method_name,
    benchmark,
    score_matrix,
):
    """Calculate ranking metrics for one retrieval method."""

    corpus = benchmark["corpus"]
    queries = benchmark["queries"]

    query_results = []
    answerable_metrics = []
    unanswerable_scores = []

    for query_index, query in enumerate(queries):
        
        scores = score_matrix[query_index]
        all_ranked_indices = np.argsort(-scores, kind="stable")

        # TF-IDF with zero score means there was no lexical match.
        # Do not count zero-score documents as retrieved results.
        if method_name == "tfidf_word_and_bigram":
            ranked_indices = [
                int(index)
                for index in all_ranked_indices
                if float(scores[int(index)]) > 0.0
            ]
        else:
            # Semantic search still ranks all documents.
            ranked_indices = [
                int(index) for index in all_ranked_indices
            ]

        expected_ids = set(query["expected_source_ids"])

        ranked_sources = [
            corpus[index]["id"]
            for index in ranked_indices
        ]

        top_sources = []

        for index in ranked_indices[:TOP_K]:
            corpus_item = corpus[index]

            top_sources.append({
                "source_id": corpus_item["id"],
                "filename": corpus_item["filename"],
                "score": round(float(scores[index]), 4),
            })

        matching_ranks = [
            rank
            for rank, source_id in enumerate(
                ranked_sources, start=1
            )
            if source_id in expected_ids
        ]

        expected_rank = (
            matching_ranks[0] if matching_ranks else None
        )

        best_score = (
            float(scores[ranked_indices[0]])
            if ranked_indices
            else 0.0
        )

        if query["answerable"]:
            reciprocal_rank = (
                1.0 / expected_rank
                if expected_rank is not None
                else 0.0
            )

            recall_at_k = (
                len(expected_ids & set(ranked_sources[:TOP_K]))
                / len(expected_ids)
            )

            answerable_metrics.append({
                "hit_at_1": bool(
                    expected_rank is not None and expected_rank == 1
                ),
                "recall_at_3": recall_at_k,
                "reciprocal_rank": reciprocal_rank,
                "expected_source_rank": expected_rank,
            })
        else:
            reciprocal_rank = None
            recall_at_k = None

            unanswerable_scores.append(best_score)

        query_result = {
            "query_id": query["id"],
            "question": query["question"],
            "answerable": query["answerable"],
            "expected_source_ids": sorted(expected_ids),
            "expected_source_rank": expected_rank,
            "best_score": round(best_score, 4),
            "top_results": top_sources,
        }

        query_results.append(query_result)

        status = (
            "ANSWERABLE"
            if query["answerable"]
            else "UNANSWERABLE"
        )

        rank_text = (
            str(expected_rank)
            if expected_rank is not None
            else "not found"
        )

        print(
            f"{query['id']} | {status:<12} | "
            f"best={best_score:.3f} | expected_rank={rank_text}"
        )

    count = len(answerable_metrics)

    if count == 0:
        raise ValueError("No answerable queries are available.")

    unanswerable_count = len(unanswerable_scores)

    metrics = {
        "method": method_name,
        "answerable_query_count": count,
        "unanswerable_query_count": unanswerable_count,
        "hit_at_1": round(
            sum(item["hit_at_1"] for item in answerable_metrics)
            / count,
            4,
        ),
        "recall_at_3": round(
            sum(item["recall_at_3"] for item in answerable_metrics)
            / count,
            4,
        ),
        "mean_reciprocal_rank": round(
            sum(item["reciprocal_rank"] for item in answerable_metrics)
            / count,
            4,
        ),
        "average_best_score_unanswerable": (
            round(
                sum(unanswerable_scores) / unanswerable_count,
                4,
            )
            if unanswerable_count
            else None
        ),
        "max_best_score_unanswerable": (
            round(max(unanswerable_scores), 4)
            if unanswerable_count
            else None
        ),
    }

    return {
        "metrics": metrics,
        "query_results": query_results,
    }


def main():
    print("=" * 65)
    print("DOCUMIND AI - TF-IDF VS SEMANTIC RETRIEVAL")
    print("=" * 65)

    benchmark = load_benchmark()

    corpus = benchmark["corpus"]
    queries = benchmark["queries"]

    corpus_texts = [item["text"] for item in corpus]
    questions = [item["question"] for item in queries]

    print(f"\nCorpus documents: {len(corpus)}")
    print(f"Evaluation queries: {len(queries)}")
    print(f"Ranking cutoff: Top {TOP_K}")

    # Method 1: lexical retrieval using words and word pairs.
    print("\n[1/2] Building TF-IDF keyword retrieval...")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )

    corpus_tfidf = vectorizer.fit_transform(corpus_texts)
    query_tfidf = vectorizer.transform(questions)

    tfidf_scores = cosine_similarity(
        query_tfidf,
        corpus_tfidf,
    )

    # Method 2: semantic retrieval using dense embeddings.
    model_name = benchmark["embedding_model"]

    print(f"\n[2/2] Loading embedding model: {model_name}")

    model = SentenceTransformer(model_name)

    print("Generating corpus embeddings...")

    corpus_embeddings = model.encode(
        corpus_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    print("Generating question embeddings...")

    question_embeddings = model.encode(
        questions,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    semantic_scores = np.asarray(
        question_embeddings,
        dtype=np.float32,
    ) @ np.asarray(
        corpus_embeddings,
        dtype=np.float32,
    ).T

    print("\n" + "=" * 65)
    print("TF-IDF RESULTS")
    print("=" * 65)

    tfidf_report = evaluate_rankings(
        "tfidf_word_and_bigram",
        benchmark,
        tfidf_scores,
    )

    print("\n" + "=" * 65)
    print("SEMANTIC EMBEDDING RESULTS")
    print("=" * 65)

    semantic_report = evaluate_rankings(
        "sentence_transformer",
        benchmark,
        semantic_scores,
    )

    comparison = {
        "benchmark_name": benchmark["benchmark_name"],
        "corpus_size": len(corpus),
        "query_count": len(queries),
        "top_k": TOP_K,
        "models": {
            "tfidf": tfidf_report,
            "semantic": semantic_report,
        },
        "interpretation_notes": [
            "Metrics are measured on a small synthetic benchmark.",
            "Recall@3 measures source retrieval, not answer correctness.",
            "MRR rewards the expected source appearing higher in the ranking.",
            "TF-IDF and embedding similarity scores use different scales.",
            "A shared abstention threshold should not be applied to both methods.",
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2, ensure_ascii=False)

    print("\n" + "=" * 65)
    print("FINAL COMPARISON")
    print("=" * 65)

    for key in ("tfidf", "semantic"):
        metrics = comparison["models"][key]["metrics"]

        print(f"\n{metrics['method']}")
        print(f"  Hit@1: {metrics['hit_at_1']:.3f}")
        print(f"  Recall@3: {metrics['recall_at_3']:.3f}")
        print(
            "  MRR: "
            f"{metrics['mean_reciprocal_rank']:.3f}"
        )
        print(
            "  Average best score for unanswerable queries: "
            f"{metrics['average_best_score_unanswerable']}"
        )

    print(f"\nDetailed report saved to: {REPORT_PATH}")
    print("Comparison completed.")


if __name__ == "__main__":
    main()
