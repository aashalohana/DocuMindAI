"""DocuMind AI FastAPI backend using the Gemini API for lightweight RAG."""
from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai

from app import read_document
from storage import (
    delete_document,
    get_document,
    list_chunks,
    list_documents,
    replace_chunks,
    save_document,
)

# Load .env for local development. Hosting providers should use secret env vars.
load_dotenv()

GEMINI_TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.5-flash-lite")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
# Version the storage label so vectors made by the old local model are not mixed in.
EMBEDDING_MODEL_ID = f"{GEMINI_EMBEDDING_MODEL}:question-answering-v1"
MAX_FILE_SIZE = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf", ".docx"}
# Provisional baseline; calibrate this for the selected embedding model before production use.
MIN_SIMILARITY_SCORE = float(os.getenv("MIN_SIMILARITY_SCORE", "0.35"))
CHUNK_SIZE_WORDS = 80
CHUNK_OVERLAP_WORDS = 15

app = FastAPI(
    title="DocuMind AI",
    description="Upload PDF/DOCX documents and ask grounded questions with RAG.",
    version="1.1.0",
)

_default_origins = (
    "http://localhost:5175,http://127.0.0.1:5175,"
    "http://localhost:5173,http://127.0.0.1:5173"
)
ALLOWED_ORIGINS = [
    item.strip()
    for item in os.getenv("FRONTEND_ORIGINS", _default_origins).split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    # Allows Vercel preview/production URLs; set FRONTEND_ORIGINS explicitly in Render.
allow_origin_regex=r"^(https?://(localhost|127\.0\.0\.1)(:\d+)?|https://[a-zA-Z0-9-]+\.vercel\.app)$",    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE_WORDS,
               overlap: int = CHUNK_OVERLAP_WORDS) -> list[str]:
    """Split extracted text into overlapping word chunks."""
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")
    words = text.split()
    step = chunk_size - overlap
    return [" ".join(words[start:start + chunk_size])
            for start in range(0, len(words), step)
            if words[start:start + chunk_size]]


@lru_cache(maxsize=1)
def get_genai_client() -> genai.Client:
    """Create the Gemini client from a secret environment variable."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Gemini API key is not configured on the backend.",
        )
    return genai.Client(api_key=api_key)


def _normalise(vector: list[float]) -> list[float]:
    values = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError("The embedding API returned an empty vector.")
    return (values / norm).astype(float).tolist()


def embed_text(text: str, purpose: Literal["document", "query"],
               title: str | None = None) -> list[float]:
    """Create a normalized vector using Gemini Embedding 2's recommended task prompts."""
    if purpose == "document":
        embedding_input = f"title: {title or 'none'} | text: {text}"
    else:
        embedding_input = f"task: question answering | query: {text}"

    response = get_genai_client().models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=embedding_input,
    )
    if not response.embeddings or not response.embeddings[0].values:
        raise ValueError("Gemini did not return an embedding vector.")
    return _normalise(list(response.embeddings[0].values))


def make_embeddings(chunks: list[str], title: str) -> list[list[float]]:
    """Embed chunks one at a time; this supports Gemini Embedding 2's input behavior."""
    return [embed_text(chunk, "document", title=title) for chunk in chunks]


def generate_grounded_answer(question: str, sources: list[dict]) -> str:
    """Generate an answer based only on retrieved source passages."""
    if not sources:
        return "I could not find an answer to that question in the saved documents."

    context = "\n\n".join(
        f"Source: {source.get('filename', 'Unknown document')}, "
        f"chunk {int(source.get('chunk_index', 0)) + 1}\n{source['text']}"
        for source in sources
    )
    prompt = (
        "You are DocuMind AI, a retrieval-augmented document assistant.\n"
        "Answer the user's question using ONLY the source passages below.\n"
        "Do not use outside knowledge or make assumptions. If the sources do not "
        "explicitly support an answer, say: 'I could not find an answer to that "
        "question in the saved documents.' Keep the answer concise and factual.\n\n"
        f"SOURCE PASSAGES:\n{context}\n\nQUESTION:\n{question}\n\nANSWER:"
    )
    response = get_genai_client().models.generate_content(
        model=GEMINI_TEXT_MODEL,
        contents=prompt,
    )
    answer = (response.text or "").strip()
    return answer or "I could not generate an answer. Please try again."


async def extract_uploaded_text(file: UploadFile) -> tuple[str, str]:
    """Validate a PDF/DOCX upload and extract text safely via a temporary file."""
    filename = Path(file.filename or "uploaded_file").name
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    content = await file.read(MAX_FILE_SIZE + 1)
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="Maximum file size is 10 MB.")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", suffix=extension, delete=False) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        text = read_document(str(temporary_path))
        if not text.strip():
            raise HTTPException(status_code=400, detail="No readable text was found in this file.")
        return filename, text
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=f"Could not read document: {error}") from error
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def index_document(document_id: int, filename: str, text: str) -> int:
    """Create/re-create chunks and Gemini embeddings for a saved document."""
    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No readable text was found.")
    vectors = make_embeddings(chunks, filename)
    return replace_chunks(
        document_id=document_id,
        chunks=chunks,
        embeddings=vectors,
        embedding_model=EMBEDDING_MODEL_ID,
    )


def ensure_current_index() -> list[dict]:
    """Re-index previously saved documents whose vectors use the old local model."""
    documents = list_documents()
    all_chunks: list[dict] = []
    for document in documents:
        chunks = list_chunks(document_id=int(document["id"]))
        is_current = bool(chunks) and all(
            chunk["embedding"] is not None
            and chunk["embedding_model"] == EMBEDDING_MODEL_ID
            for chunk in chunks
        )
        if not is_current:
            full_document = get_document(int(document["id"]))
            if full_document and full_document.get("text_content", "").strip():
                index_document(
                    int(document["id"]),
                    str(full_document["filename"]),
                    str(full_document["text_content"]),
                )
        all_chunks.extend(
            chunk for chunk in list_chunks(document_id=int(document["id"]))
            if chunk["embedding"] is not None
            and chunk["embedding_model"] == EMBEDDING_MODEL_ID
        )
    return all_chunks


@app.get("/")
def home() -> dict:
    return {
        "message": "DocuMind AI API is running!",
        "provider": "Gemini API",
        "docs": "/docs",
    }


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    filename, text = await extract_uploaded_text(file)
    return {
        "filename": filename,
        "message": "Document uploaded and read successfully.",
        "total_characters": len(text),
        "text": text[:5000],
        "truncated": len(text) > 5000,
    }


@app.post("/documents")
async def create_document(file: UploadFile = File(...)) -> dict:
    filename, document_text = await extract_uploaded_text(file)
    try:
        saved_document = save_document(filename=filename, text_content=document_text)
        document_id = int(saved_document["id"])
        stored_chunks = list_chunks(document_id=document_id)
        already_indexed = bool(stored_chunks) and all(
            chunk["embedding"] is not None
            and chunk["embedding_model"] == EMBEDDING_MODEL_ID
            for chunk in stored_chunks
        )
        if already_indexed:
            chunk_count = len(stored_chunks)
        else:
            chunk_count = index_document(document_id, filename, document_text)
        return {
            "message": "Document saved and indexed successfully.",
            "document_id": document_id,
            "filename": saved_document["filename"],
            "already_exists": saved_document["already_exists"],
            "chunk_count": chunk_count,
            "embedding_model": EMBEDDING_MODEL_ID,
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Document indexing failed. Check the Gemini API key, quota, and model access. Details: {error}",
        ) from error


@app.get("/documents")
def get_documents() -> dict:
    return {"documents": list_documents()}


@app.delete("/documents/{document_id}")
def remove_document(document_id: int) -> dict:
    deleted = delete_document(document_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"message": "Document deleted successfully.", "document_id": document_id}


@app.post("/ask")
async def ask_question(file: UploadFile = File(...), question: str = Form(...)) -> dict:
    if not question.strip():
        raise HTTPException(status_code=400, detail="Please enter a question.")
    filename, document_text = await extract_uploaded_text(file)
    chunks = chunk_text(document_text)
    try:
        if not chunks:
            raise HTTPException(status_code=400, detail="No readable text was found.")
        query_vector = np.asarray(embed_text(question, "query"), dtype=np.float32)
        document_vectors = np.asarray(make_embeddings(chunks, filename), dtype=np.float32)
        scores = document_vectors @ query_vector
        ranked_indices = scores.argsort()[::-1][:3]
        sources = [
            {"text": chunks[int(index)], "similarity": round(float(scores[int(index)]), 3),
             "filename": filename, "chunk_index": int(index)}
            for index in ranked_indices if float(scores[int(index)]) >= MIN_SIMILARITY_SCORE
        ]
        if not sources:
            return {
                "filename": filename,
                "question": question,
                "answer": "I could not find an answer to that question in the uploaded document.",
                "sources": [],
            }
        answer = generate_grounded_answer(question, sources)
        return {"filename": filename, "question": question, "answer": answer, "sources": sources}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Question answering failed. Check Gemini API key, quota, and model access. Details: {error}",
        ) from error


@app.post("/knowledge/ask")
async def ask_knowledge_base(question: str = Form(...)) -> dict:
    if not question.strip():
        raise HTTPException(status_code=400, detail="Please enter a question.")
    documents = list_documents()
    if not documents:
        raise HTTPException(status_code=404, detail="No saved documents found. Upload a document first.")
    try:
        indexed_chunks = ensure_current_index()
        if not indexed_chunks:
            raise HTTPException(
                status_code=404,
                detail="No indexed chunks found. Upload and index a document first.",
            )
        query_vector = np.asarray(embed_text(question, "query"), dtype=np.float32)
        document_vectors = np.asarray([chunk["embedding"] for chunk in indexed_chunks], dtype=np.float32)
        scores = document_vectors @ query_vector
        document_names = {int(document["id"]): document["filename"] for document in documents}
        ranked_indices = scores.argsort()[::-1]
        sources: list[dict] = []
        for index in ranked_indices:
            score = float(scores[int(index)])
            if score < MIN_SIMILARITY_SCORE:
                continue
            chunk = indexed_chunks[int(index)]
            sources.append({
                "document_id": int(chunk["document_id"]),
                "filename": document_names.get(int(chunk["document_id"]), "Unknown document"),
                "chunk_index": int(chunk["chunk_index"]),
                "text": chunk["text_content"],
                "similarity": round(score, 3),
            })
            if len(sources) == 3:
                break

        if not sources:
            return {
                "question": question,
                "answer": "I could not find an answer to that question in the saved documents.",
                "sources": [],
            }
        answer = generate_grounded_answer(question, sources)
        return {"question": question, "answer": answer, "sources": sources}
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail=f"Question answering failed. Check Gemini API key, quota, and model access. Details: {error}",
        ) from error
