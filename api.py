from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app import read_document
from rag import MODEL_NAME, chunk_text, retrieve_semantic_chunks

import numpy as np

from storage import (
    save_document,
    list_documents,
    list_chunks,
    replace_chunks,
    delete_document,
)

ANSWER_MODEL = "google/flan-t5-small"
MAX_FILE_SIZE = 10 * 1024 * 1024
MIN_SIMILARITY_SCORE = 0.30

ALLOWED_EXTENSIONS = {".pdf", ".docx"}

app = FastAPI(
    title="DocuMind AI",
    description="Upload documents and ask questions using RAG.",
    version="1.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5175",
        "http://127.0.0.1:5175",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load and cache the embedding model."""
    return SentenceTransformer(MODEL_NAME)


@lru_cache(maxsize=1)
def get_answer_model():
    """Load and cache the local answer-generation model."""
    tokenizer = AutoTokenizer.from_pretrained(ANSWER_MODEL)

    model = AutoModelForSeq2SeqLM.from_pretrained(
        ANSWER_MODEL
    )
    model.eval()

    return tokenizer, model


async def extract_uploaded_text(file: UploadFile):
    """Validate an upload and extract its text."""

    filename = Path(file.filename or "uploaded_file").name
    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported.",
        )

    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="Maximum file size is 10 MB.",
        )

    temporary_path = None

    try:
        with NamedTemporaryFile(
            mode="wb",
            suffix=extension,
            delete=False,
        ) as temporary_file:
            temporary_file.write(content)
            temporary_path = Path(temporary_file.name)

        text = read_document(str(temporary_path))

        return filename, text

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Could not read document: {error}",
        ) from error

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


@app.get("/")
def home():
    return {
        "message": "DocuMind AI API is running!",
        "docs": "/docs",
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document and return its extracted text."""

    filename, text = await extract_uploaded_text(file)

    return {
        "filename": filename,
        "message": "Document uploaded and read successfully.",
        "total_characters": len(text),
        "text": text[:5000],
        "truncated": len(text) > 5000,
    }


@app.post("/documents")
async def create_document(file: UploadFile = File(...)):
    """Save a document and index its chunks and embeddings."""

    filename, document_text = await extract_uploaded_text(file)

    try:
        # Save the document text in SQLite.
        saved_document = save_document(
            filename=filename,
            text_content=document_text,
        )

        document_id = saved_document["id"]

        # Avoid re-embedding content that is already indexed.
        stored_chunks = list_chunks(document_id=document_id)

        already_indexed = (
            bool(stored_chunks)
            and all(
                chunk["embedding"] is not None
                and chunk["embedding_model"] == MODEL_NAME
                for chunk in stored_chunks
            )
        )

        if already_indexed:
            chunk_count = len(stored_chunks)
        else:
            # Split the document and generate reusable embeddings.
            chunks = chunk_text(document_text)

            if not chunks:
                raise HTTPException(
                    status_code=400,
                    detail="No readable text was found.",
                )

            embedding_model = get_embedding_model()

            vectors = embedding_model.encode(
                chunks,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            chunk_count = replace_chunks(
                document_id=document_id,
                chunks=chunks,
                embeddings=vectors.tolist(),
                embedding_model=MODEL_NAME,
            )

        return {
            "message": "Document saved and indexed successfully.",
            "document_id": document_id,
            "filename": saved_document["filename"],
            "already_exists": saved_document["already_exists"],
            "chunk_count": chunk_count,
            "embedding_model": MODEL_NAME,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document indexing failed: {error}",
        ) from error


@app.get("/documents")
def get_documents():
    """Return the saved document library."""

    return {
        "documents": list_documents(),
    }


@app.delete("/documents/{document_id}")
def remove_document(document_id: int):
    """Delete a document and its associated chunks."""

    deleted = delete_document(document_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail="Document not found.",
        )

    return {
        "message": "Document deleted successfully.",
        "document_id": document_id,
    }

@app.post("/ask")
async def ask_question(
    file: UploadFile = File(...),
    question: str = Form(...),
):
    """Answer a question using information retrieved from an upload."""

    if not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    filename, document_text = await extract_uploaded_text(file)
    chunks = chunk_text(document_text)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No readable text was found.",
        )

    try:
        embedding_model = get_embedding_model()

        results = retrieve_semantic_chunks(
            query=question,
            chunks=chunks,
            model=embedding_model,
            top_k=3,
        )

        
        if not results:
            return {
                "filename": filename,
                "question": question,
                "answer": "I could not find relevant information in the document.",
                "sources": [],
            }

        # Check the relevance of the best matching passage.
        best_score = max(
            score for chunk, score in results
        )

        # Do not generate an answer if the match is too weak.
        if best_score < MIN_SIMILARITY_SCORE:
            return {
                "filename": filename,
                "question": question,
                "answer": (
                    "I could not find an answer to that question "
                    "in the uploaded document."
                ),
                "sources": [],
                "message": (
                    "The retrieved passages were not relevant "
                    "enough to answer this question reliably."
                ),
            }

        # Continue to answer generation only if the passage
        # passes the relevance threshold.
        context = "\n\n".join(
            chunk for chunk, score in results
        )


        tokenizer, answer_model = get_answer_model()

        prompt = (
            "Answer the question using only the context below. "
            "If the answer is not present, say you could not "
            "find it in the document.\n\n"
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

        generated = answer_model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=2,
            do_sample=False,
        )

        answer = tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        ).strip()

        sources = [
            {
                "text": chunk,
                "similarity": round(score, 3),
            }
            for chunk, score in results
        ]

        return {
            "filename": filename,
            "question": question,
            "answer": answer,
            "sources": sources,
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Question answering failed: {error}",
        ) from error

        
@app.post("/knowledge/ask")
async def ask_knowledge_base(question: str = Form(...)):
    """Answer questions using all saved and indexed documents."""

    if not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    documents = list_documents()

    if not documents:
        raise HTTPException(
            status_code=404,
            detail="No saved documents found. Upload a document first.",
        )

    # Only use vectors created by our current embedding model.
    indexed_chunks = [
        chunk
        for chunk in list_chunks()
        if chunk["embedding"] is not None
        and chunk["embedding_model"] == MODEL_NAME
    ]

    if not indexed_chunks:
        raise HTTPException(
            status_code=404,
            detail="No indexed chunks found. Upload and index a document first.",
        )

    try:
        embedding_model = get_embedding_model()

        # The question is embedded once. Stored document vectors
        # are reused instead of being recalculated on every query.
        query_vector = embedding_model.encode(
            question,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )

        document_vectors = np.asarray(
            [chunk["embedding"] for chunk in indexed_chunks],
            dtype=np.float32,
        )

        query_vector = np.asarray(query_vector, dtype=np.float32)

        # Stored vectors and the query vector are normalized,
        # so their dot product gives cosine similarity.
        scores = document_vectors @ query_vector

        document_names = {
            document["id"]: document["filename"]
            for document in documents
        }

        ranked_indices = scores.argsort()[::-1]

        results = []

        for index in ranked_indices:
            score = float(scores[index])

            # Filter out low-relevance passages.
            if score < MIN_SIMILARITY_SCORE:
                continue

            chunk = indexed_chunks[int(index)]

            results.append({
                "document_id": chunk["document_id"],
                "filename": document_names.get(
                    chunk["document_id"],
                    "Unknown document",
                ),
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text_content"],
                "similarity": round(score, 3),
            })

            if len(results) == 3:
                break

        if not results:
            return {
                "question": question,
                "answer": (
                    "I could not find an answer to that question "
                    "in the saved documents."
                ),
                "sources": [],
            }

        # Give the generator source-labelled evidence.
        context = "\n\n".join(
            (
                f"Source: {source['filename']}, "
                f"chunk {source['chunk_index'] + 1}\n"
                f"{source['text']}"
            )
            for source in results
        )

        tokenizer, answer_model = get_answer_model()

        prompt = (
            "Answer the question using only the context below. "
            "Do not invent facts. If the context does not contain "
            "the answer, say you could not find it in the documents.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            "Answer:"
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )

        generated = answer_model.generate(
            **inputs,
            max_new_tokens=80,
            num_beams=2,
            do_sample=False,
        )

        answer = tokenizer.decode(
            generated[0],
            skip_special_tokens=True,
        ).strip()

        return {
            "question": question,
            "answer": answer,
            "sources": results,
            "retrieval_count": len(results),
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge-base question failed: {error}",
        ) from error
