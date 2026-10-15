# DocuMind AI — RAG Document Assistant

DocuMind AI is a local-first document question-answering application. Upload PDF or DOCX files, save them in a persistent document library, and ask questions across the saved documents. Answers are generated from retrieved passages, with source information returned by the API.

## Features

- PDF and DOCX text extraction
- Persistent document library backed by SQLite
- Semantic chunk retrieval using `sentence-transformers/all-MiniLM-L6-v2`
- Local answer generation using `google/flan-t5-small`
- Source passages returned with answers
- Duplicate-content detection and document deletion
- Automated tests and retrieval evaluation scripts
- React + Vite frontend and FastAPI backend

## Architecture

```text
React + Vite frontend
        |
        | HTTP requests
        v
FastAPI backend (api.py)
        |
        +-- PDF/DOCX text extraction
        +-- Sentence Transformers embeddings and retrieval
        +-- FLAN-T5 answer generation
        +-- SQLite document/chunk persistence
```

## Run locally on Windows

Use two terminals from the project directory.

### 1. Backend

Activate your existing virtual environment, then run:

```cmd
python -m pip install -r requirements.txt
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

The API health route is `http://127.0.0.1:8000/` and interactive API documentation is at `http://127.0.0.1:8000/docs`.

### 2. Frontend

In a second terminal:

```cmd
cd frontend
npm ci
npm run dev
```

Open the local address printed by Vite (the current project uses port 5175).

### First startup

The embedding model and answer-generation model are downloaded from Hugging Face the first time they are loaded. This can take time and disk space. Later starts normally reuse the local model cache. Internet access is needed for the initial download unless the models have already been cached.

## Run tests and evaluation

From the project root with the virtual environment activated:

```cmd
python -m pytest -v
python evaluate_retrieval.py
python compare_retrieval.py
python calibrate_threshold.py
```

The retrieval benchmarks are small, synthetic test sets. The results are useful for regression checks but do **not** prove real-world accuracy. In particular, semantically similar passages can score highly even when they do not answer the question. Treat the threshold report as an experiment, not as production validation.

## API routes

- `GET /` — API health check
- `POST /upload` — extract text from a temporary uploaded PDF/DOCX
- `POST /documents` — save and index a document
- `GET /documents` — list saved documents
- `DELETE /documents/{document_id}` — delete a saved document and its chunks
- `POST /ask` — ask a question about an uploaded document
- `POST /knowledge/ask` — ask a question across the saved document library

For exact request/response schemas, start the backend and open `/docs`.

## Data and privacy

The SQLite database is stored under `data/` and is ignored by Git. Document contents are stored locally by this prototype. The local transformer models run on your machine; their first download requires access to Hugging Face. Do not upload confidential documents to a hosted version without reviewing the hosting provider, storage, logging, access control, and deletion behavior.

## Docker (local containerized setup)

With Docker Desktop and Docker Compose installed, run from the project root:

```cmd
docker compose up --build
```

Then open `http://localhost:5175` for the frontend and `http://localhost:8000/docs` for API documentation. The database and downloaded models use named Docker volumes so they can persist between container restarts. Initial build/model download may take time.

Stop the containers with `Ctrl+C`; to stop and remove containers while keeping named volumes, run:

```cmd
docker compose down
```

This Compose configuration is intended for local reproducible testing, not as a hardened public production deployment. A cloud deployment needs the frontend API base URL and backend CORS origins configured for the actual hosted domains, plus appropriate authentication, upload limits, persistent storage, and operational monitoring.

## Known limitations

- Scanned/image-only PDFs may require OCR, which is not implemented in this version.
- Retrieval thresholds are not yet validated on a large, real-world dataset.
- This prototype does not provide multi-user authentication or tenant-level document isolation.
- FLAN-T5 Small is a lightweight local model and can produce incomplete answers; source verification and abstention behavior should be tested before relying on answers for important decisions.

## Main technology stack

- Frontend: React, Vite
- Backend: Python, FastAPI
- Persistence: SQLite
- Retrieval: Sentence Transformers and cosine similarity
- Answer generation: Transformers (`google/flan-t5-small`)
- Tests: pytest
