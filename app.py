from pathlib import Path

from docx import Document
from pypdf import PdfReader


def read_document(file_path: str) -> str:
    """Read text from a PDF or DOCX document."""

    path = Path(file_path)

    # Check whether the file exists.
    if not path.is_file():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    file_extension = path.suffix.lower()

    # Extract text from a PDF file.
    if file_extension == ".pdf":
        reader = PdfReader(str(path))

        text = "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    # Extract text from a Word document.
    elif file_extension == ".docx":
        document = Document(str(path))

        text = "\n".join(
            paragraph.text
            for paragraph in document.paragraphs
        )

    else:
        raise ValueError(
            "Unsupported file type. Please use PDF or DOCX."
        )

    # Check whether readable text was extracted.
    if not text.strip():
        raise ValueError(
            "No readable text found. "
            "The file may be empty or contain scanned images."
        )

    return text


def main():
    print("=" * 40)
    print("       DocuMind AI Document Reader")
    print("=" * 40)

    file_path = input(
        "\nEnter your PDF or DOCX file path: "
    ).strip().strip('"')

    try:
        extracted_text = read_document(file_path)

        print("\nDocument read successfully!")
        print("-" * 40)

        # Display the first 2000 characters.
        print(extracted_text[:2000])

        print("\n" + "-" * 40)
        print(
            f"Total characters: {len(extracted_text)}"
        )

    except (FileNotFoundError, ValueError) as error:
        print(f"\nError: {error}")

    except Exception as error:
        print(
            "\nThe document could not be read. "
            f"Details: {error}"
        )


if __name__ == "__main__":
    main()