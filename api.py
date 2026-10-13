from fastapi.middleware.cors import CORSMiddleware
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from app import read_document
from rag import MODEL_NAME, chunk_text, retrieve_semantic_chunks


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