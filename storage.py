from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
import json
import os
import sqlite3
from pathlib import Path
from typing import Iterator, Sequence


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "documind.db"


def _resolve_database_path(
    db_path: str | Path | None = None,
) -> Path:
    """Resolve the database path, with an optional deployment override."""

    if db_path is not None:
        return Path(db_path)

    configured_path = os.getenv("DOCUMIND_DB_PATH")

    if configured_path:
        return Path(configured_path)

    return DEFAULT_DATABASE_PATH


@contextmanager
def _connect(
    db_path: str | Path | None = None,
) -> Iterator[sqlite3.Connection]:
    """Open a database connection and safely commit or roll back."""

    path = _resolve_database_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(str(path), timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_db(db_path: str | Path | None = None) -> None:
    """Create the document and chunk tables if they do not exist."""

    with _connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                content_hash TEXT NOT NULL UNIQUE,
                text_content TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )

        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
                text_content TEXT NOT NULL,
                embedding_json TEXT,
                embedding_model TEXT,
                FOREIGN KEY (document_id)
                    REFERENCES documents(id)
                    ON DELETE CASCADE,
                UNIQUE (document_id, chunk_index)
            )
            """
        )

        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_document_id
            ON chunks(document_id)
            """
        )


def save_document(
    filename: str,
    text_content: str,
    db_path: str | Path | None = None,
) -> dict:
    """Save extracted document text, avoiding duplicate content."""

    safe_filename = str(filename).replace("\\", "/").rsplit("/", 1)[-1]
    cleaned_text = text_content.strip()

    if not safe_filename.strip():
        raise ValueError("A document filename is required.")

    if not cleaned_text:
        raise ValueError("Document text cannot be empty.")

    content_hash = sha256(
        cleaned_text.encode("utf-8")
    ).hexdigest()

    init_db(db_path)

    with _connect(db_path) as connection:
        existing = connection.execute(
            """
            SELECT id, filename, content_hash, created_at
            FROM documents
            WHERE content_hash = ?
            """,
            (content_hash,),
        ).fetchone()

        if existing:
            result = dict(existing)
            result["already_exists"] = True
            return result

        created_at = datetime.now(timezone.utc).isoformat()

        cursor = connection.execute(
            """
            INSERT INTO documents (
                filename,
                content_hash,
                text_content,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                safe_filename,
                content_hash,
                cleaned_text,
                created_at,
            ),
        )

        return {
            "id": cursor.lastrowid,
            "filename": safe_filename,
            "content_hash": content_hash,
            "created_at": created_at,
            "already_exists": False,
        }


def list_documents(
    db_path: str | Path | None = None,
) -> list[dict]:
    """List saved documents without returning their full text."""

    init_db(db_path)

    with _connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT
                d.id,
                d.filename,
                d.content_hash,
                d.created_at,
                COUNT(c.id) AS chunk_count
            FROM documents AS d
            LEFT JOIN chunks AS c ON c.document_id = d.id
            GROUP BY d.id
            ORDER BY d.id DESC
            """
        ).fetchall()

        return [dict(row) for row in rows]


def get_document(
    document_id: int,
    db_path: str | Path | None = None,
) -> dict | None:
    """Get a saved document, including its extracted text."""

    init_db(db_path)

    with _connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT id, filename, content_hash, text_content, created_at
            FROM documents
            WHERE id = ?
            """,
            (document_id,),
        ).fetchone()

        return dict(row) if row else None


def replace_chunks(
    document_id: int,
    chunks: Sequence[str],
    embeddings: Sequence[Sequence[float]] | None = None,
    embedding_model: str | None = None,
    db_path: str | Path | None = None,
) -> int:
    """Replace a document's chunks and optionally save their embeddings."""

    cleaned_chunks = [chunk.strip() for chunk in chunks]

    if any(not chunk for chunk in cleaned_chunks):
        raise ValueError("Chunks cannot contain empty text.")

    if embeddings is not None and len(embeddings) != len(cleaned_chunks):
        raise ValueError(
            "The number of embeddings must match the number of chunks."
        )

    if embeddings is not None and not embedding_model:
        raise ValueError(
            "Specify the embedding model when storing embeddings."
        )

    prepared_rows = []

    for index, chunk in enumerate(cleaned_chunks):
        embedding_json = None

        if embeddings is not None:
            vector = [float(value) for value in embeddings[index]]

            if not vector:
                raise ValueError("An embedding vector cannot be empty.")

            embedding_json = json.dumps(vector)

        prepared_rows.append(
            (
                document_id,
                index,
                chunk,
                embedding_json,
                embedding_model,
            )
        )

    init_db(db_path)

    with _connect(db_path) as connection:
        document = connection.execute(
            "SELECT id FROM documents WHERE id = ?",
            (document_id,),
        ).fetchone()

        if document is None:
            raise ValueError(f"Document {document_id} does not exist.")

        connection.execute(
            "DELETE FROM chunks WHERE document_id = ?",
            (document_id,),
        )

        connection.executemany(
            """
            INSERT INTO chunks (
                document_id,
                chunk_index,
                text_content,
                embedding_json,
                embedding_model
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            prepared_rows,
        )

    return len(cleaned_chunks)


def list_chunks(
    document_id: int | None = None,
    db_path: str | Path | None = None,
) -> list[dict]:
    """List stored chunks and decode any saved embedding vectors."""

    init_db(db_path)

    with _connect(db_path) as connection:
        if document_id is None:
            rows = connection.execute(
                """
                SELECT id, document_id, chunk_index, text_content,
                       embedding_json, embedding_model
                FROM chunks
                ORDER BY document_id, chunk_index
                """
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT id, document_id, chunk_index, text_content,
                       embedding_json, embedding_model
                FROM chunks
                WHERE document_id = ?
                ORDER BY chunk_index
                """,
                (document_id,),
            ).fetchall()

        results = []

        for row in rows:
            item = dict(row)
            stored_embedding = item.pop("embedding_json")

            item["embedding"] = (
                json.loads(stored_embedding)
                if stored_embedding is not None
                else None
            )

            results.append(item)

        return results


def delete_document(
    document_id: int,
    db_path: str | Path | None = None,
) -> bool:
    """Delete a document and automatically delete its chunks."""

    init_db(db_path)

    with _connect(db_path) as connection:
        cursor = connection.execute(
            "DELETE FROM documents WHERE id = ?",
            (document_id,),
        )

        return cursor.rowcount > 0