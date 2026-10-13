
from docx import Document
import pytest

from app import read_document
from rag import chunk_text


def test_read_docx_file(tmp_path):
    """The document reader should extract Word document text."""

    document_path = tmp_path / "sample.docx"

    document = Document()
    document.add_paragraph("Refunds are available within 14 days.")
    document.save(str(document_path))

    text = read_document(str(document_path))

    assert "14 days" in text
    assert "Refunds" in text


def test_chunk_text_splits_words():
    """Text should be divided into overlapping chunks."""

    text = "one two three four five six"

    chunks = chunk_text(
        text,
        chunk_size=4,
        overlap=1,
    )

    assert chunks == [
        "one two three four",
        "four five six",
    ]


def test_chunk_text_rejects_invalid_overlap():
    """Overlap must be smaller than chunk size."""

    with pytest.raises(ValueError):
        chunk_text(
            "one two three",
            chunk_size=3,
            overlap=3,
        )


def test_missing_document_raises_error(tmp_path):
    """A missing document should produce a clear error."""

    missing_path = tmp_path / "missing.docx"

    with pytest.raises(FileNotFoundError):
        read_document(str(missing_path))