
import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


BENCHMARK_PATH = Path("evals/hard_negative_benchmark.json")
TOP_K = 3
CURRENT_THRESHOLD = 0.30


def load_benchmark():
    if not BENCHMARK_PATH.exists():
        raise FileNotFoundError(
            f"Benchmark file not found: {BENCHMARK_PATH}"
        )

    with BENCHMARK_PATH.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not data.get("corpus") or not data.get("queries"):
        raise ValueError("Benchmark corpus or queries are empty.")

    return data


def calculate_results(records, threshold):
    answerable = [r for r in records if r["answerable"]]
    unanswerable = [r for r in records if not r["answerable"]]

    # An answerable question counts as correctly accepted only if
    # its expected source is in the top results and its best score
    # meets the selected threshold.
    correct_accepts = sum(
        r["answerable"]
        and r["expected_source_found"]
        and r["best_score"] >= threshold
        for r in records
    )

    false_accepts = sum(
        not r["answerable"] and r["best_score"] >= threshold
        for r in records
    )

    answerable_rate = correct_accepts / len(answerable)
    abstention_rate = (
        len(unanswerable) - false_accepts
    ) / len(unanswerable)

    balanced_accuracy = (answerable_rate + abstention_rate) / 2

    return {
        "threshold": threshold,
        "correct_answerable_acceptance": answerable_rate,
        "unanswerable_abstention": abstention_rate,
        "false_accepts": false_accepts,
        "balanced_accuracy": balanced_accuracy,
    }


def main():
    data = load_benchmark()

    corpus = data["corpus"]
    queries = data["queries"]
    model_name = data.get(
        "embedding_model",
        "sentence-transformers/all-MiniLM-L6-v2",
    )

    print("=" * 62)
    print("DOCUMIND AI - THRESHOLD CALIBRATION")
    print("=" * 62)
    print(f"Documents: {len(corpus)}")
    print(f"Queries: {len(queries)}")
    print(f"Current threshold: {CURRENT_THRESHOLD:.2f}")
    print(f"Embedding model: {model_name}")

    model = SentenceTransformer(model_name)

    document_embeddings = model.encode(
        [item["text"] for item in corpus],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embeddings = model.encode(
        [item["question"] for item in queries],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    similarity_matrix = query_embeddings @ document_embeddings.T
    records = []

    for query_index, query in enumerate(queries):
        scores = similarity_matrix[query_index]
        ranked_indices = np.argsort(-scores, kind="stable")

        top_indices = ranked_indices[:TOP_K]
        retrieved_ids = {
            corpus[int(index)]["id"] for index in top_indices
        }
        expected_ids = set(query["expected_source_ids"])

        records.append({
            "id": query["id"],
            "answerable": query["answerable"],
            "best_score": float(scores[int(ranked_indices[0])]),
            "expected_source_found": bool(
                expected_ids.intersection(retrieved_ids)
            ),
        })

    answerable_records = [r for r in records if r["answerable"]]
    retrieval_hits = sum(
        r["expected_source_found"] for r in answerable_records
    )

    print(
        f"\nAnswerable source retrieval Hit@{TOP_K}: "
        f"{retrieval_hits}/{len(answerable_records)} "
        f"({retrieval_hits / len(answerable_records):.1%})"
    )

    # Test the current threshold and useful comparison points.
    thresholds_to_show = [
        0.30, 0.40, 0.45, 0.49, 0.525, 0.55
    ]

    # Also test boundaries immediately above observed scores.
    candidate_thresholds = {0.0, 1.0, CURRENT_THRESHOLD}

    for record in records:
        score = record["best_score"]
        candidate_thresholds.add(round(score, 6))
        candidate_thresholds.add(round(min(1.0, score + 0.00001), 6))

    all_results = [
        calculate_results(records, threshold)
        for threshold in sorted(candidate_thresholds)
    ]

    best_result = max(
        all_results,
        key=lambda item: (
            item["balanced_accuracy"],
            -item["threshold"],
        ),
    )

    print("\nTHRESHOLD COMPARISON")
    print("-" * 75)
    print(
        f"{'Threshold':>9} | {'Correct answerable':>18} | "
        f"{'Unanswerable rejected':>21} | {'Balanced':>9}"
    )
    print("-" * 75)

    for threshold in thresholds_to_show:
        result = calculate_results(records, threshold)
        print(
            f"{threshold:>9.3f} | "
            f"{result['correct_answerable_acceptance']:>17.1%} | "
            f"{result['unanswerable_abstention']:>20.1%} | "
            f"{result['balanced_accuracy']:>8.1%}"
        )

    print("\nBEST THRESHOLD ON THIS BENCHMARK")
    print("-" * 40)
    print(f"Threshold: {best_result['threshold']:.5f}")
    print(
        "Correct answerable acceptance: "
        f"{best_result['correct_answerable_acceptance']:.1%}"
    )
    print(
        "Unanswerable abstention: "
        f"{best_result['unanswerable_abstention']:.1%}"
    )
    print(
        "Balanced accuracy: "
        f"{best_result['balanced_accuracy']:.1%}"
    )
    print(
        "\nCalibration complete. The API threshold has NOT been changed."
    )


if __name__ == "__main__":
    main()
