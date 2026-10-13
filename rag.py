
from app import read_document
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def chunk_text(
    text: str,
    chunk_size: int = 80,
    overlap: int = 15,
) -> list[str]:
    """Split document text into overlapping word chunks."""

    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "Overlap must be smaller than chunk size."
        )

    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for start in range(0, len(words), step):
        chunk = words[start:start + chunk_size]

        if chunk:
            chunks.append(" ".join(chunk))

    return chunks


def retrieve_semantic_chunks(
    query: str,
    chunks: list[str],
    model: SentenceTransformer,
    top_k: int = 3,
) -> list[tuple[str, float]]:
    """Find document chunks that are semantically similar to a query."""

    if not query.strip() or not chunks or top_k <= 0:
        return []

    # Convert document chunks into AI embeddings.
    chunk_embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Convert the user's question into an embedding.
    query_embedding = model.encode(
        query,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    # Compare vectors to rank the most relevant chunks.
    scores = chunk_embeddings @ query_embedding

    ranked_indices = scores.argsort()[::-1][:top_k]

    return [
        (chunks[index], float(scores[index]))
        for index in ranked_indices
    ]


def main():
    print("=" * 45)
    print("       DocuMind AI - Semantic Search")
    print("=" * 45)

    file_path = input(
        "\nEnter your PDF or DOCX file path: "
    ).strip().strip('"')

    try:
        document_text = read_document(file_path)
        chunks = chunk_text(document_text)

        print(f"\nDocument processed into {len(chunks)} chunk(s).")
        print("\nLoading the AI embedding model...")

        # Downloads the model the first time it is needed.
        # Later runs normally reuse the local cached copy.
        model = SentenceTransformer(MODEL_NAME)

        print("AI embedding model ready!")

        query = input(
            "\nAsk a question about your document: "
        ).strip()

        if not query:
            print("Please enter a question.")
            return

        results = retrieve_semantic_chunks(
            query=query,
            chunks=chunks,
            model=model,
        )

        if not results:
            print("No searchable text or question was provided.")
            return

        print("\nMost relevant passages:")
        print("-" * 45)

        for rank, (chunk, score) in enumerate(results, start=1):
            print(f"\nResult {rank} | Similarity score: {score:.3f}")
            print(chunk)

        print("\n" + "-" * 45)
        print("Semantic search completed successfully.")

    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")

    except Exception as error:
        print(
            "\nThe semantic search could not complete."
            "\nCheck your internet connection if the model is "
            "being downloaded for the first time."
            f"\nDetails: {error}"
        )


if __name__ == "__main__":
    main()