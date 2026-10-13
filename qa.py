
from app import read_document
from rag import MODEL_NAME, chunk_text, retrieve_semantic_chunks

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


EMBEDDING_MODEL = MODEL_NAME
ANSWER_MODEL = "google/flan-t5-small"


def generate_answer(
    question: str,
    context: str,
    tokenizer,
    answer_model,
) -> str:
    """Generate an answer using the retrieved document context."""

    prompt = (
        "Answer the question using only the information "
        "in the context below. Do not invent facts. "
        "If the answer is not available in the context, "
        "say that you could not find the answer in the document.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        "Answer:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    output = answer_model.generate(
        **inputs,
        max_new_tokens=80,
        num_beams=2,
        do_sample=False,
    )

    answer = tokenizer.decode(
        output[0],
        skip_special_tokens=True,
    )

    return answer.strip()


def main():
    print("=" * 45)
    print("       DocuMind AI - Question Answering")
    print("=" * 45)

    file_path = input(
        "\nEnter your PDF or DOCX file path: "
    ).strip().strip('"')

    try:
        document_text = read_document(file_path)
        chunks = chunk_text(document_text)

        if not chunks:
            print("No readable text was found.")
            return

        print("\nLoading the embedding model...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL)

        print("Loading the local answer-generation model...")
        print("The first run may download model files.")

        tokenizer = AutoTokenizer.from_pretrained(ANSWER_MODEL)
        answer_model = AutoModelForSeq2SeqLM.from_pretrained(
            ANSWER_MODEL
        )
        answer_model.eval()

        question = input(
            "\nAsk a question about your document: "
        ).strip()

        if not question:
            print("Please enter a question.")
            return

        results = retrieve_semantic_chunks(
            query=question,
            chunks=chunks,
            model=embedding_model,
            top_k=3,
        )

        if not results:
            print("\nNo relevant passage was retrieved.")
            return

        context = "\n\n".join(
            chunk for chunk, score in results
        )

        print("\nGenerating an answer...")

        answer = generate_answer(
            question=question,
            context=context,
            tokenizer=tokenizer,
            answer_model=answer_model,
        )

        print("\n" + "=" * 45)
        print("ANSWER")
        print("=" * 45)
        print(answer)

        print("\nSOURCE PASSAGE(S)")
        print("-" * 45)

        for index, (chunk, score) in enumerate(results, start=1):
            print(f"\nSource {index} (score: {score:.3f})")
            print(chunk)

    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")

    except Exception as error:
        print(f"\nSomething went wrong: {error}")
        print(
            "Check the error details before changing or "
            "reinstalling any packages."
        )


if __name__ == "__main__":
    main()