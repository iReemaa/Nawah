from io import BytesIO

from pypdf import PdfWriter

from app.ingestion import (
    extract_pdf_pages,
    PDFExtractorError
)


def test_invalid_pdf_content():
    fake_pdf = b"This is not a PDF"

    try:
        extract_pdf_pages(fake_pdf)

    except PDFExtractorError:
        print(
            "Invalid PDF test passed."
        )

    else:
        raise AssertionError(
            "Invalid PDF should have failed validation."
        )


if __name__ == "__main__":
    test_invalid_pdf_content()