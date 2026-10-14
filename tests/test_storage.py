import pytest

from storage import (
    delete_document,
    get_document,
    list_chunks,
    list_documents,
    replace_chunks,
    save_document,
)


def test_document_is_saved_and_persists(tmp_path):
    db_path = tmp_path / "test.db"

    saved = save_document(
        "guide.docx",
        "Refunds are available within 14 days.",
        db_path=db_path,
    )

    assert saved["already_exists"] is False

    document = get_document(saved["id"], db_path=db_path)

    assert document is not None
    assert document["filename"] == "guide.docx"
    assert "14 days" in document["text_content"]


def test_duplicate_content_is_not_saved_twice(tmp_path):
    db_path = tmp_path / "test.db"

    first = save_document(
        "first.docx",
        "The same document text.",
        db_path=db_path,
    )
    second = save_document(
        "renamed.docx",
        "The same document text.",
        db_path=db_path,
    )

    assert first["id"] == second["id"]
    assert second["already_exists"] is True
    assert len(list_documents(db_path=db_path)) == 1


def test_empty_filename_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_document(
            "",
            "Some content",
            db_path=tmp_path / "test.db",
        )


def test_empty_text_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        save_document(
            "empty.docx",
            "   ",
            db_path=tmp_path / "test.db",
        )


def test_chunks_and_embeddings_are_saved(tmp_path):
    db_path = tmp_path / "test.db"

    document = save_document(
        "guide.docx",
        "Refund information.",
        db_path=db_path,
    )

    count = replace_chunks(
        document_id=document["id"],
        chunks=[
            "Refunds within 14 days.",
            "Support is available on weekdays.",
        ],
        embeddings=[
            [0.1, 0.2],
            [0.3, 0.4],
        ],
        embedding_model="test-model",
        db_path=db_path,
    )

    chunks = list_chunks(
        document_id=document["id"],
        db_path=db_path,
    )

    assert count == 2
    assert chunks[0]["chunk_index"] == 0
    assert chunks[0]["text_content"] == "Refunds within 14 days."
    assert chunks[0]["embedding"] == [0.1, 0.2]
    assert chunks[1]["embedding_model"] == "test-model"

    documents = list_documents(db_path=db_path)
    assert documents[0]["chunk_count"] == 2


def test_embedding_count_must_match_chunks(tmp_path):
    db_path = tmp_path / "test.db"

    document = save_document(
        "guide.docx",
        "Refund information.",
        db_path=db_path,
    )

    with pytest.raises(ValueError, match="number of embeddings"):
        replace_chunks(
            document_id=document["id"],
            chunks=["First chunk", "Second chunk"],
            embeddings=[[0.1, 0.2]],
            embedding_model="test-model",
            db_path=db_path,
        )


def test_deleting_document_also_deletes_chunks(tmp_path):
    db_path = tmp_path / "test.db"

    document = save_document(
        "guide.docx",
        "Refund information.",
        db_path=db_path,
    )

    replace_chunks(
        document_id=document["id"],
        chunks=["Refunds within 14 days."],
        db_path=db_path,
    )

    assert delete_document(document["id"], db_path=db_path) is True
    assert get_document(document["id"], db_path=db_path) is None
    assert list_chunks(document_id=document["id"], db_path=db_path) == []
    assert delete_document(document["id"], db_path=db_path) is False