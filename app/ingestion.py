"""Nawah source ingestion, extraction, validation, and content audit."""
from app.schemas import SourceConfig, DocumentMetadata, ExtractedPDFPage


# ======================== SCRAPER ========================
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


# Shared bounded network policy: retry only transient failures, never 401/403/404.
import time

_TRANSIENT_STATUSES = {429, 500, 502, 503, 504}

def _get_with_retries(url: str, *, headers: dict, timeout: int, params=None):
    last_error = None
    for attempt in range(3):
        try:
            kwargs = {"headers": headers, "timeout": timeout}
            if params is not None:
                kwargs["params"] = params
            response = requests.get(url, **kwargs)
            if response.status_code not in _TRANSIENT_STATUSES or attempt == 2:
                return response
            last_error = requests.exceptions.HTTPError(
                f"Transient HTTP {response.status_code} for {url}"
            )
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as error:
            last_error = error
            if attempt == 2:
                raise
        if attempt < 2:
            time.sleep(0.5 * (2 ** attempt))
    raise last_error  # pragma: no cover; defensive


class ScraperError(Exception):
    """Custom exception used for scraping-related failures."""
    pass


def validate_url(url: str) -> None:
    """Validate URL before making a request."""

    if not isinstance(url, str):
        raise ScraperError("URL must be a string.")

    if not url.strip():
        raise ScraperError("URL cannot be empty.")

    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise ScraperError(
            f"Unsupported URL scheme: {parsed.scheme}"
        )

    if not parsed.netloc:
        raise ScraperError(
            f"Invalid URL: {url}"
        )


def scrape_webpage(url: str) -> str:
    """
    Download and extract readable text from an HTML webpage.

    Validation includes:
    - URL validation
    - network errors
    - HTTP status
    - content type
    - HTML existence
    - encoding integrity
    - extracted text quality
    - common error-page detection
    """

    validate_url(url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140 Safari/537.36"
        ),
        "Accept-Language": "ar,en;q=0.9"
    }

    try:
        response = _get_with_retries(
            url, headers=headers, timeout=20
        )

    except requests.exceptions.Timeout as error:
        raise ScraperError(
            f"Request timed out for: {url}"
        ) from error

    except requests.exceptions.ConnectionError as error:
        raise ScraperError(
            f"Connection failed for: {url}"
        ) from error

    except requests.exceptions.RequestException as error:
        raise ScraperError(
            f"Request failed for: {url}. Error: {error}"
        ) from error

    if response.status_code != 200:
        raise ScraperError(
            f"Unexpected HTTP status "
            f"{response.status_code} for: {url}"
        )

    content_type = response.headers.get(
        "Content-Type",
        ""
    ).lower()

    if "text/html" not in content_type:
        raise ScraperError(
            f"Expected HTML but received "
            f"'{content_type}' from: {url}"
        )

    raw_html = response.content

    if not raw_html:
        raise ScraperError(
            f"Received empty HTML from: {url}"
        )

    try:
        soup = BeautifulSoup(
            raw_html,
            "html.parser"
        )

    except Exception as error:
        raise ScraperError(
            f"Failed to parse HTML from: {url}"
        ) from error

    # Preserve the original body before destructive HTML cleanup.
    original_body_text = (
        soup.body.get_text(separator=" ", strip=True) if soup.body else ""
    )

    # Prefer the article/main content; removing navigation alone is insufficient
    # for government service portals with sidebars and filter panels.
    for element in soup(["script", "style", "nav", "header", "footer", "noscript", "svg", "form"]):
        element.decompose()
    for element in soup.select(
        '[role="navigation"], [role="banner"], [role="contentinfo"], '
        'aside, .breadcrumb, .breadcrumbs, .sidebar, .pagination, '
        '.cookie-banner, .cookie-consent, .social-share'
    ):
        element.decompose()

    # Only select a content container if it holds enough actual text; otherwise
    # retain the full cleaned body rather than accidentally dropping requirements.
    candidates = soup.select(
        'main, article, [role="main"], #main-content, #content'
    )

    meaningful = [
        node for node in candidates
        if len(node.get_text(" ", strip=True)) >= 200
    ]

    root = (
        max(
            meaningful,
            key=lambda node: len(node.get_text(" ", strip=True))
        )
        if meaningful
        else (soup.body or soup)
    )

    text = root.get_text(separator=" ", strip=True)

    # Some government pages place their service content outside
    # the selected content container.
    if len(text) < 200 and original_body_text:
        text = original_body_text

    text = " ".join(
            text.split()
        )

    if not text:
            raise ScraperError(
                f"No readable text was extracted from: {url}"
            )

    if len(text) < 200:
            raise ScraperError(
                f"Extracted content is suspiciously short "
                f"({len(text)} characters) for: {url}"
            )

    blocked_phrases = [
        "access denied",
        "403 forbidden",
        "404 not found",
        "page not found",
        "service unavailable",
        "internal server error"
    ]

    lowered_text = text.lower()

    for phrase in blocked_phrases:
        if phrase in lowered_text:
            raise ScraperError(
                f"Page appears to contain an error page. "
                f"Detected phrase: '{phrase}'"
            )

    mojibake_markers = [
        "Ø",
        "Ù",
        "Ã",
        "Â",
        "�"
    ]

    mojibake_count = sum(
        text.count(marker)
        for marker in mojibake_markers
    )

    if mojibake_count > 5:
        raise ScraperError(
            f"Character-encoding corruption detected "
            f"for: {url}"
        )

    return text

# ======================== PDF_EXTRACTOR ========================
from io import BytesIO
from urllib.parse import urlparse

import requests
from pydantic import ValidationError
from pypdf import PdfReader
from pypdf.errors import PdfReadError



class PDFExtractorError(Exception):
    """Raised when downloading or extracting a PDF fails."""
    pass


def validate_pdf_url(url: str) -> None:
    """
    Validate the PDF URL before requesting it.
    """

    if not isinstance(url, str):
        raise PDFExtractorError(
            "PDF URL must be a string."
        )

    url = url.strip()

    if not url:
        raise PDFExtractorError(
            "PDF URL cannot be empty."
        )

    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise PDFExtractorError(
            f"Unsupported URL scheme: {parsed.scheme}"
        )

    if not parsed.netloc:
        raise PDFExtractorError(
            f"Invalid PDF URL: {url}"
        )


def download_pdf(url: str) -> bytes:
    """
    Download and validate an official PDF source.
    """

    validate_pdf_url(url)

    headers = {
        "User-Agent": (
            "Mozilla/5.0 "
            "(Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140 Safari/537.36"
        ),
        "Accept": "application/pdf,*/*",
        "Accept-Language": "ar,en;q=0.9"
    }

    try:
        response = _get_with_retries(
            url, headers=headers, timeout=30
        )

    except requests.exceptions.Timeout as error:
        raise PDFExtractorError(
            f"PDF request timed out for: {url}"
        ) from error

    except requests.exceptions.ConnectionError as error:
        raise PDFExtractorError(
            f"Connection failed while downloading PDF: {url}"
        ) from error

    except requests.exceptions.RequestException as error:
        raise PDFExtractorError(
            f"PDF request failed for: {url}. "
            f"Error: {error}"
        ) from error

    if response.status_code != 200:
        raise PDFExtractorError(
            f"Unexpected HTTP status "
            f"{response.status_code} for PDF: {url}"
        )

    pdf_bytes = response.content

    if not pdf_bytes:
        raise PDFExtractorError(
            f"Downloaded PDF is empty: {url}"
        )

    # A valid PDF normally begins with the PDF signature.
    if not pdf_bytes.startswith(b"%PDF"):
        raise PDFExtractorError(
            f"Downloaded content does not appear "
            f"to be a valid PDF: {url}"
        )

    # Prevent accidentally processing tiny/error files.
    if len(pdf_bytes) < 500:
        raise PDFExtractorError(
            f"Downloaded PDF is suspiciously small "
            f"({len(pdf_bytes)} bytes): {url}"
        )

    return pdf_bytes


# Page-level extraction diagnostics are reset for each PDF. These are surfaced
# in source_failures.txt, and all surviving pages from an affected PDF are
# marked incomplete so the review/indexing gate can reject them.
_PDF_PAGE_ISSUES: list[str] = []


def _extract_misa_registration_layout(page, fitz_module) -> str:
    """OCR the known two-column MISA registration page by visual reading regions.

    Fail closed if the expected content cannot be recovered; never fabricate text.
    """
    import subprocess
    import shutil
    if not shutil.which("tesseract"):
        raise PDFExtractorError("Tesseract required for MISA registration layout")

    def region(bounds):
        image = page.get_pixmap(
            matrix=fitz_module.Matrix(3, 3),
            clip=fitz_module.Rect(bounds), alpha=False
        ).tobytes("png")
        result = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", "eng"],
            input=image, capture_output=True, timeout=90, check=True
        )
        return result.stdout.decode("utf-8", errors="replace").strip()

    # The visual layout has a top-right service description and two lower columns.
    # OCR the regions separately so the fee, document list and conditions do not mix.
    description = region((265, 85, 594, 345))
    fees_and_timing = region((10, 345, 265, 805))
    documents_and_conditions = region((275, 345, 594, 805))
    combined = ("3.1.1 Registering for investment\n\n"
                "SERVICE DESCRIPTION\n" + description + "\n\n"
                "SERVICE FEES, APPLICATION PORTAL AND PROCESSING TIME\n"
                + fees_and_timing + "\n\n"
                "REQUIRED DOCUMENTS AND SERVICE CONDITIONS\n"
                + documents_and_conditions)
    checks = ("investment", "registration", "fifteen", "financial statements",
              "conditions", "working day")
    if not all(item in combined.lower() for item in checks):
        raise PDFExtractorError("MISA layout OCR failed expected-content checks")
    return combined


def _reorder_misa_real_estate_page(text: str) -> str:
    """Put PDF page 7's right-hand service before its left-hand details.

    Reorder existing OCR verbatim; do not synthesize or silently correct words.
    Fail closed when the expected page structure is absent.
    """
    import re

    heading = re.search(
        r"(?im)^\s*3\s*\.\s*1\s*\.\s*2\s+Registration of Non-Saudi",
        text,
    )
    if heading is None:
        raise PDFExtractorError("MISA page 7: expected 3.1.2 heading not found")
    before = text[:heading.start()].strip()
    after = text[heading.start():].strip()
    if not before.lower().startswith("service conditions"):
        raise PDFExtractorError("MISA page 7: unexpected leading section; manual review required")
    if not all(term in after.lower() for term in ("service description", "required documents")):
        raise PDFExtractorError("MISA page 7: required right-column sections missing")
    if not all(term in before.lower() for term in ("service fees", "processing time")):
        raise PDFExtractorError("MISA page 7: required left-column sections missing")
    return after + "\n\n" + before


def extract_pdf_pages(pdf_bytes: bytes, source_url: str = "") -> list[ExtractedPDFPage]:
    """Extract pages independently; preserve gaps and report every unreadable page."""
    import re
    import shutil
    import subprocess

    _PDF_PAGE_ISSUES.clear()
    if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
        raise PDFExtractorError("PDF content must be nonempty bytes.")

    try:
        reader = PdfReader(BytesIO(pdf_bytes))
        if reader.is_encrypted and reader.decrypt("") == 0:
            raise PDFExtractorError("PDF is password-protected.")
        if not reader.pages:
            raise PDFExtractorError("PDF contains no pages.")
    except PDFExtractorError:
        raise
    except Exception as error:
        raise PDFExtractorError(f"Failed to open PDF: {error}") from error

    try:
        import fitz
        alternate = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as error:
        raise PDFExtractorError(
            "PyMuPDF is required: python -m pip install pymupdf. "
            + str(error)
        ) from error

    extracted_pages = []
    try:
        if len(alternate) != len(reader.pages):
            raise PDFExtractorError("PDF extractors disagree on page count.")

        for page_number in range(1, len(reader.pages) + 1):
            try:
                try:
                    primary = reader.pages[page_number - 1].extract_text() or ""
                except Exception:
                    primary = ""
                secondary = alternate[page_number - 1].get_text("text", sort=True) or ""
                primary, secondary = primary.strip(), secondary.strip()
                text = secondary if len(secondary) > len(primary) else primary
                if ("misa.gov.sa/" in source_url
                        and "Investor-Guide_12-05-compressed.pdf" in source_url
                        and page_number == 6):
                    try:
                        text = _extract_misa_registration_layout(
                            alternate[page_number - 1], fitz
                        )
                    except (PDFExtractorError, subprocess.SubprocessError, OSError) as error:
                        _PDF_PAGE_ISSUES.append(
                            f"page {page_number}: layout-aware extraction failed ({error})"
                        )
                        continue

                if ("misa.gov.sa/" in source_url
                        and "Investor-Guide_12-05-compressed.pdf" in source_url
                        and page_number == 7):
                    text = _reorder_misa_real_estate_page(text)

                letters = len(re.findall(r"[A-Za-z\\u0600-\\u06FF]", text))
                if letters < 150 or len(text) < 250:
                    if shutil.which("tesseract"):
                        try:
                            png = alternate[page_number - 1].get_pixmap(
                                matrix=fitz.Matrix(3, 3), alpha=False
                            ).tobytes("png")
                            result = subprocess.run(
                                ["tesseract", "stdin", "stdout", "-l", "eng"],
                                input=png, capture_output=True, timeout=90, check=True,
                            )
                            ocr_text = result.stdout.decode(
                                "utf-8", errors="replace"
                            ).strip()
                            if len(ocr_text) > len(text):
                                text = ocr_text
                        except (subprocess.SubprocessError, OSError) as error:
                            _PDF_PAGE_ISSUES.append(
                                f"page {page_number}: OCR failed ({error})"
                            )
                    else:
                        _PDF_PAGE_ISSUES.append(
                            f"page {page_number}: OCR required but Tesseract unavailable"
                        )

                letters = len(re.findall(r"[A-Za-z\\u0600-\\u06FF]", text))
                if letters < 20 or len(text.strip()) < 50:
                    _PDF_PAGE_ISSUES.append(
                        f"page {page_number}: unreadable or too short after extraction/OCR"
                    )
                    continue

                try:
                    extracted_pages.append(
                        ExtractedPDFPage(page_number=page_number, text=text)
                    )
                except ValidationError as error:
                    _PDF_PAGE_ISSUES.append(
                        f"page {page_number}: schema validation failed ({error})"
                    )
            except Exception as error:
                _PDF_PAGE_ISSUES.append(
                    f"page {page_number}: extraction failed ({error})"
                )
                continue
    finally:
        alternate.close()

    if not extracted_pages:
        raise PDFExtractorError(
            "No usable pages extracted; page issues: "
            + "; ".join(_PDF_PAGE_ISSUES)
        )
    return extracted_pages


def extract_pdf_from_url(
    url: str
) -> list[ExtractedPDFPage]:
    """
    Download a PDF from a URL and extract its pages.
    """

    pdf_bytes = download_pdf(url)

    return extract_pdf_pages(
        pdf_bytes, source_url=url
    )

# ======================== API_EXTRACTOR ========================
from urllib.parse import urlparse
from typing import Any

import requests


class APIExtractorError(Exception):
    """Raised when API extraction or validation fails."""
    pass


def validate_api_url(url: str) -> None:
    """
    Validate an API endpoint before making a request.
    """

    if not isinstance(url, str):
        raise APIExtractorError(
            "API URL must be a string."
        )

    url = url.strip()

    if not url:
        raise APIExtractorError(
            "API URL cannot be empty."
        )

    parsed = urlparse(url)

    if parsed.scheme not in {"http", "https"}:
        raise APIExtractorError(
            f"Unsupported API URL scheme: {parsed.scheme}"
        )

    if not parsed.netloc:
        raise APIExtractorError(
            f"Invalid API URL: {url}"
        )


def flatten_json(
    value: Any,
    parent_key: str = ""
) -> list[str]:
    """
    Convert nested JSON into readable text lines
    while preserving keys and values.
    """

    lines = []

    if isinstance(value, dict):

        for key, item in value.items():

            current_key = (
                f"{parent_key}.{key}"
                if parent_key
                else str(key)
            )

            lines.extend(
                flatten_json(
                    item,
                    current_key
                )
            )

    elif isinstance(value, list):

        for index, item in enumerate(
            value,
            start=1
        ):

            current_key = (
                f"{parent_key}[{index}]"
                if parent_key
                else f"[{index}]"
            )

            lines.extend(
                flatten_json(
                    item,
                    current_key
                )
            )

    else:

        if value is None:
            return lines

        text_value = str(value).strip()

        if text_value:
            lines.append(
                f"{parent_key}: {text_value}"
            )

    return lines


def extract_api_json(
    url: str,
    params: dict | None = None,
    headers: dict | None = None
) -> str:
    """
    Retrieve JSON from an API endpoint and convert it
    into readable text suitable for Nawah's knowledge base.
    """

    validate_api_url(url)

    request_headers = {
        "Accept": "application/json",
        "User-Agent": (
            "Mozilla/5.0 "
            "(Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) "
            "Chrome/140 Safari/537.36"
        ),
        "Accept-Language": "ar,en;q=0.9"
    }

    if headers:
        request_headers.update(headers)

    try:
        response = _get_with_retries(
            url, params=params, headers=request_headers, timeout=30
        )

    except requests.exceptions.Timeout as error:
        raise APIExtractorError(
            f"API request timed out for: {url}"
        ) from error

    except requests.exceptions.ConnectionError as error:
        raise APIExtractorError(
            f"Connection failed for API: {url}"
        ) from error

    except requests.exceptions.RequestException as error:
        raise APIExtractorError(
            f"API request failed for: {url}. "
            f"Error: {error}"
        ) from error

    if response.status_code != 200:
        raise APIExtractorError(
            f"Unexpected API HTTP status "
            f"{response.status_code} for: {url}"
        )

    content_type = response.headers.get(
        "Content-Type",
        ""
    ).lower()

    if (
        "application/json" not in content_type
        and "+json" not in content_type
    ):
        raise APIExtractorError(
            f"Expected JSON but received "
            f"'{content_type}' from: {url}"
        )

    try:
        data = response.json()

    except ValueError as error:
        raise APIExtractorError(
            f"API response is not valid JSON: {url}"
        ) from error

    if data is None:
        raise APIExtractorError(
            f"API returned no data: {url}"
        )

    if isinstance(data, dict) and not data:
        raise APIExtractorError(
            f"API returned an empty JSON object: {url}"
        )

    if isinstance(data, list) and not data:
        raise APIExtractorError(
            f"API returned an empty JSON list: {url}"
        )

    flattened_lines = flatten_json(
        data
    )

    if not flattened_lines:
        raise APIExtractorError(
            f"No useful text could be extracted "
            f"from API response: {url}"
        )

    text = "\n".join(
        flattened_lines
    ).strip()

    if len(text) < 50:
        raise APIExtractorError(
            f"API response contains too little useful "
            f"content ({len(text)} characters): {url}"
        )

    return text

# ======================== CLEANER ========================
import re
import unicodedata


class CleanerError(Exception):
    """Raised when text cleaning fails."""
    pass


def clean_text(text: str) -> str:
    """
    Normalize extracted Arabic/English text while preserving
    meaningful regulatory wording.

    Length/quality thresholds are handled by the appropriate
    extractor or document builder, not by the cleaner.
    """

    if not isinstance(text, str):
        raise CleanerError(
            "Input text must be a string."
        )

    if not text.strip():
        raise CleanerError(
            "Input text is empty."
        )

    # Normalize Unicode representation without aggressively
    # changing Arabic letters or regulatory wording.
    text = unicodedata.normalize(
        "NFKC",
        text
    )

    # Remove invisible Unicode formatting characters.
    text = re.sub(
        r"[\u200B-\u200F\u202A-\u202E\u2060\uFEFF]",
        "",
        text
    )

    # Preserve paragraph boundaries for downstream regulatory chunking.
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\r?\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove unnecessary whitespace before punctuation.
    text = re.sub(
        r"\s+([,.;:!?،؛؟])",
        r"\1",
        text
    )

    # Add a space after punctuation when necessary.
    text = re.sub(
        r"([,;:!?،؛؟]|\.(?![0-9]))(?=\S)",
        r"\1 ",
        text
    )

    text = text.strip()

    if not text:
        raise CleanerError(
            "Cleaning produced empty text."
        )

    return text

# ======================== DOCUMENT_BUILDER ========================
from app.schemas import SourceConfig, DocumentMetadata, ExtractedPDFPage
from datetime import datetime, timezone

from langchain_core.documents import Document
from pydantic import ValidationError



class DocumentBuilderError(Exception):
    """Raised when document creation fails."""
    pass


def build_document(
    text: str,
    source: SourceConfig,
    title: str | None = None,
    page_number: int | None = None
) -> Document:
    """
    Convert cleaned source text into a LangChain Document
    with validated and traceable metadata.
    """

    if not isinstance(text, str):
        raise DocumentBuilderError(
            "Document text must be a string."
        )

    text = text.strip()

    if not text:
        raise DocumentBuilderError(
            "Document text cannot be empty."
        )

    if len(text) < 50:
        raise DocumentBuilderError(
            "Document text is too short to be useful."
        )

    if not isinstance(source, SourceConfig):
        raise DocumentBuilderError(
            "Source must be a validated SourceConfig object."
        )

    if (
        page_number is not None
        and not isinstance(page_number, int)
    ):
        raise DocumentBuilderError(
            "Page number must be an integer."
        )

    try:
        metadata = DocumentMetadata(
            authority=source.authority,
            category=source.category,
            source_url=source.url,
            source_type=source.source_type,
            language=source.language,
            business_activity=source.business_activity,
            region=source.region,
            ingested_at=datetime.now(timezone.utc),
            title=title,
            page_number=page_number
        )

    except ValidationError as error:
        raise DocumentBuilderError(
            f"Metadata validation failed: {error}"
        ) from error

    document_metadata = {
        "authority": metadata.authority,
        "category": metadata.category,
        "source_url": str(metadata.source_url),
        "source_type": metadata.source_type,
        "language": metadata.language,
        "business_activity": metadata.business_activity,
        "region": metadata.region,
        "ingested_at": metadata.ingested_at.isoformat()
    }

    if metadata.title is not None:
        document_metadata["title"] = metadata.title

    if metadata.page_number is not None:
        document_metadata["page_number"] = metadata.page_number

    return Document(
        page_content=text,
        metadata=document_metadata
    )

# ======================== QUALITY_VALIDATOR ========================
import re

from langchain_core.documents import Document


class ContentQualityError(Exception):
    """Raised when extracted knowledge content fails quality checks."""
    pass


def validate_document_quality(
    document: Document
) -> None:
    """
    Validate extracted document text before it is allowed
    into Nawah's knowledge base.
    """

    if not isinstance(document, Document):
        raise ContentQualityError(
            "Expected a LangChain Document."
        )

    text = document.page_content.strip()

    if not text:
        raise ContentQualityError(
            "Document contains no text."
        )

    if len(text) < 50:
        raise ContentQualityError(
            "Document contains too little useful text."
        )

    # Unicode replacement character usually indicates
    # damaged decoding.
    if "�" in text:
        raise ContentQualityError(
            "Document contains corrupted Unicode characters."
        )

    mojibake_markers = [
        "Ø",
        "Ù",
        "Ã",
        "Â"
    ]

    mojibake_count = sum(
        text.count(marker)
        for marker in mojibake_markers
    )

    if mojibake_count > 5:
        raise ContentQualityError(
            "Document appears to contain encoding corruption."
        )

    # Require at least some real Arabic/English letters.
    letters = re.findall(
        r"[A-Za-z\u0600-\u06FF]",
        text
    )

    if len(letters) < 20:
        raise ContentQualityError(
            "Document contains insufficient readable language content."
        )

    readable_ratio = (
        len(letters) / len(text)
    )

    if readable_ratio < 0.15:
        raise ContentQualityError(
            "Document contains too much non-language or corrupted content."
        )


def filter_valid_documents(
    documents: list[Document]
) -> tuple[list[Document], list[str]]:
    """
    Separate valid documents from documents that fail
    Nawah's content-quality validation.
    """

    valid_documents = []
    rejected_documents = []

    for document in documents:

        try:
            validate_document_quality(
                document
            )

            valid_documents.append(
                document
            )

        except ContentQualityError as error:

            source_url = document.metadata.get(
                "source_url",
                "unknown"
            )

            page_number = document.metadata.get(
                "page_number"
            )

            location = (
                f"{source_url}"
                + (
                    f" | page {page_number}"
                    if page_number
                    else ""
                )
            )

            rejected_documents.append(
                f"{location} | {error}"
            )

    return (
        valid_documents,
        rejected_documents
    )

# ======================== SOURCE_LOADER ========================
from app.schemas import SourceConfig, DocumentMetadata, ExtractedPDFPage
import json
from pathlib import Path

from pydantic import ValidationError



SOURCE_FILE = Path("data/sources.json")


def load_sources() -> list[SourceConfig]:

    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source file not found: {SOURCE_FILE}"
        )

    try:
        with SOURCE_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:
            raw_sources = json.load(file)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"sources.json contains invalid JSON: {error}"
        ) from error

    if not isinstance(raw_sources, list):
        raise ValueError(
            "sources.json must contain a list of sources."
        )

    if not raw_sources:
        raise ValueError(
            "sources.json is empty."
        )

    validated_sources = []

    for index, source in enumerate(
        raw_sources,
        start=1
    ):
        try:
            validated_source = SourceConfig(
                **source
            )

            validated_sources.append(
                validated_source
            )

        except ValidationError as error:
            raise ValueError(
                f"Invalid source at position {index}: "
                f"{error}"
            ) from error

    # Check for duplicate URLs
    urls = [
        str(source.url)
        for source in validated_sources
    ]

    if len(urls) != len(set(urls)):
        raise ValueError(
            "Duplicate source URLs found "
            "in sources.json."
        )

    return validated_sources

# ======================== PIPELINE ========================
from app.schemas import SourceConfig, DocumentMetadata, ExtractedPDFPage
from langchain_core.documents import Document










class IngestionPipelineError(Exception):
    """Raised when the ingestion pipeline cannot complete."""
    pass


def process_webpage_source(
    source: SourceConfig
) -> list[Document]:
    """
    Process one webpage source from extraction
    through final LangChain Document creation.
    """

    raw_text = scrape_webpage(
        str(source.url)
    )

    cleaned_text = clean_text(
        raw_text
    )

    document = build_document(
        text=cleaned_text,
        source=source
    )

    return [document]


def process_pdf_source(source: SourceConfig) -> list[Document]:
    """Retain extracted pages independently; record source-level gaps separately."""
    extracted_pages = extract_pdf_from_url(str(source.url))
        # Verified correction for Investor Guide PDF page 6 only.
    correction_path = Path("data/corrections/investor_guide_page_6.txt")

    investor_guide_url = (
        "https://misa.gov.sa/app/uploads/2026/01/"
        "Investor-Guide_12-05-compressed.pdf"
    )

    if str(source.url) == investor_guide_url:
        if not correction_path.is_file():
            raise PDFExtractorError(
                f"Missing verified correction: {correction_path}"
            )

        corrected_text = correction_path.read_text(
            encoding="utf-8"
        ).strip()

        if len(corrected_text) < 200:
            raise PDFExtractorError(
                "Investor Guide page 6 correction is empty or incomplete."
            )

        for page in extracted_pages:
            if page.page_number == 6:
                page.text = corrected_text
                break
        else:
            raise PDFExtractorError(
                "Investor Guide page 6 was not extracted."
            )
    documents = []
    short_pages = []

    for page in extracted_pages:
        try:
            cleaned_text = clean_text(page.text)
            if len(cleaned_text) < 50:
                short_pages.append(page.page_number)
                continue
            documents.append(build_document(
                text=cleaned_text, source=source, page_number=page.page_number
            ))
        except (CleanerError, DocumentBuilderError) as error:
            _PDF_PAGE_ISSUES.append(
                f"page {page.page_number}: document processing failed ({error})"
            )

    for page_number in short_pages:
        _PDF_PAGE_ISSUES.append(f"page {page_number}: cleaned content too short")

    if not documents:
        raise IngestionPipelineError(
            f"No usable PDF pages remained: {source.url}; "
            + "; ".join(_PDF_PAGE_ISSUES)
        )

    if _PDF_PAGE_ISSUES:
        for document in documents:
            # Source completeness is a warning, not proof this particular page is bad.
            document.metadata["pdf_extraction_incomplete"] = True
            document.metadata["pdf_page_issues"] = list(_PDF_PAGE_ISSUES)
    return documents


def process_api_source(
    source: SourceConfig
) -> list[Document]:
    """
    Process one JSON API source into a validated
    LangChain Document.
    """

    raw_text = extract_api_json(
        str(source.url)
    )

    cleaned_text = clean_text(
        raw_text
    )

    document = build_document(
        text=cleaned_text,
        source=source
    )

    return [document]


def process_source(
    source: SourceConfig
) -> list[Document]:
    """
    Route one validated source to the correct extractor
    according to its source_type.
    """

    if not isinstance(source, SourceConfig):
        raise IngestionPipelineError(
            "Source must be a validated SourceConfig object."
        )

    if not source.enabled:
        return []

    if source.source_type == "webpage":
        return process_webpage_source(
            source
        )

    if source.source_type == "pdf":
        return process_pdf_source(
            source
        )

    if source.source_type == "api":
        return process_api_source(
            source
        )

    raise IngestionPipelineError(
        f"Unsupported source type: {source.source_type}"
    )


def run_ingestion_pipeline() -> tuple[
    list[Document],
    list[str]
]:
    """
    Run Nawah's complete source-ingestion pipeline.

    Steps:
    1. Load and validate sources.json.
    2. Keep enabled sources.
    3. Route each source to the correct extractor.
    4. Clean extracted content.
    5. Build LangChain Documents with metadata.
    6. Isolate individual source failures.
    7. Validate final document content quality.
    8. Return valid documents and failure reports.
    """

    try:
        sources = load_sources()

    except Exception as error:
        raise IngestionPipelineError(
            f"Failed to load source registry: {error}"
        ) from error

    enabled_sources = [
        source
        for source in sources
        if source.enabled
    ]

    if not enabled_sources:
        raise IngestionPipelineError(
            "No enabled sources were found."
        )

    documents = []
    failures = []

    for source in enabled_sources:

        try:
            source_documents = process_source(
                source
            )

            documents.extend(
                source_documents
            )
            if source.source_type == "pdf":
                failures.extend(
                    f"{source.authority} | {source.category} | {source.url} | {issue}"
                    for issue in _PDF_PAGE_ISSUES
                )

        except (
            ScraperError,
            PDFExtractorError,
            APIExtractorError,
            CleanerError,
            DocumentBuilderError,
            IngestionPipelineError
        ) as error:

            failures.append(
                (
                    f"{source.authority} | "
                    f"{source.category} | "
                    f"{source.url} | "
                    f"{error}"
                )
            )

        except Exception as error:

            failures.append(
                (
                    f"{source.authority} | "
                    f"{source.category} | "
                    f"{source.url} | "
                    f"Unexpected error: {error}"
                )
            )

    # If extraction produced absolutely nothing,
    # stop before quality validation.
    if not documents:

        failure_details = "\n".join(
            failures
        )

        raise IngestionPipelineError(
            "The ingestion pipeline produced no documents."
            + (
                f"\nFailures:\n{failure_details}"
                if failure_details
                else ""
            )
        )

    # Final quality gate:
    # corrupted or unreadable documents are rejected
    # before they can reach chunking/embeddings.
    valid_documents, rejected_documents = (
        filter_valid_documents(
            documents
        )
    )

    # Add quality-rejected documents to the same
    # failure report.
    failures.extend(
        rejected_documents
    )

    # Do not allow an entirely invalid knowledge base.
    if not valid_documents:
        raise IngestionPipelineError(
            "All extracted documents failed quality validation."
        )

    return valid_documents, failures

# ======================== CONTENT_AUDITOR ========================
import re
from collections import Counter
from dataclasses import dataclass

from langchain_core.documents import Document


NAVIGATION_PHRASES = (
    "skip to main content",
    "filter display by",
    "cookie preferences",
    "sign in",
    "login",
    "back to top",
    "تخطي إلى المحتوى",
)

REGULATORY_TERMS = (
    # English
    "license",
    "licence",
    "permit",
    "requirement",
    "registration",
    "regulation",
    "fee",
    "penalty",
    "application",
    "eligibility",
    "compliance",
    "obligation",
    "approval",
    "authorization",
    "certificate",
    "renewal",
    "procedure",
    "condition",
    "inspection",
    "safety",
    "risk assessment",
    "commercial register",

    # Arabic
    "رخص",
    "ترخيص",
    "تصريح",
    "اشتراط",
    "متطلب",
    "تسجيل",
    "لائح",
    "رسوم",
    "غرام",
    "طلب",
    "التزام",
    "امتثال",
    "موافق",
    "تفويض",
    "شهاد",
    "تجديد",
    "إجراء",
    "إجراءات",
    "شروط",
    "شرط",
    "تفتيش",
    "سلامة",
    "مخاطر",
    "السجل التجاري",
    "يجب",
    "يلزم",
    "يتعين",
    "على المنشأة",
    "على المنظم",
)

FORM_TERMS = (
    "signature",
    "company name",
    "identity no",
    "authorized representative",
    "التوقيع",
    "اسم الشركة",
    "رقم الهوية",
    "المفوض",
)


@dataclass
class AuditResult:
    document: Document
    status: str
    reasons: list[str]


def audit_document(document: Document) -> AuditResult:

    text = document.page_content.strip()
    lowered = text.lower()

    reasons = []
    # A different PDF page failing extraction must not automatically fail this page.
    # pdf_page_issues remain in metadata and source_failures.txt for completeness reporting.
    # The page still has to pass every content check below and is never auto-approved.

    # Detect common placeholder pages even when they contain keyword matches.
    placeholders = ("no content available", "لايوجد محتوى متاح", "لا يوجد محتوى متاح", "loading. . .", "جارٍ التحميل")
    if any(text.lower().count(phrase) >= 2 for phrase in placeholders):
        reasons.append("Repeated placeholder or loading text")

    # A source category is not evidence of topical relevance. This is a
    # conservative review flag, not an automatic deletion.
    category = str(document.metadata.get("category", "")).lower()

    unrelated_service_indicators = {
        "employer_establishment_registration": (
            "early retirement",
            "صرف المعاش التقاعدي المبكر",
            "صرف المعاش",
            "الاشتراك الاختياري",
        ),
        "employer_contributions_and_fines": (
            "early retirement",
            "صرف المعاش التقاعدي المبكر",
            "الاشتراك الاختياري",
        ),
    }

    indicators = unrelated_service_indicators.get(category, ())

    if any(phrase in lowered for phrase in indicators):
        reasons.append(
            "Potentially unrelated services mixed into source"
        )
     # Flag service-directory pages dominated by filters and categories.
    directory_indicators = (
        "تصفية البحث",
        "فئات المستفيدين",
        "تصنيف الخدمات",
        "التصنيفات الفرعية",
        "ترتيب الخدمات",
        "filter results",
        "service categories",
        "sort services",
        "filter by", "show results", "all services", "service directory",
        "عرض النتائج", "جميع الخدمات", "البحث عن خدمة", "عدد الخدمات",
    )

    directory_matches = [
        phrase
        for phrase in directory_indicators
        if phrase in lowered
    ]

    if len(directory_matches) >= 3:
        reasons.append(
            "Possible service directory or filter page"
        )

    navigation_matches = [
        phrase
        for phrase in NAVIGATION_PHRASES
        if phrase in lowered
    ]

    if len(navigation_matches) >= 2:
        reasons.append(
            "Contains multiple website navigation phrases"
        )

    regulatory_matches = [
        term
        for term in REGULATORY_TERMS
        if term in lowered
    ]

    form_matches = [
        term
        for term in FORM_TERMS
        if term in lowered
    ]

    # Terms such as 'signature' and 'company name' also appear in real laws.
    # Only flag when several form cues occur in a short document.
    if len(form_matches) >= 3 and len(text) < 2500:
        reasons.append(
            "Possible application or authorization form"
        )

    if not regulatory_matches:
        reasons.append(
            "No recognized regulatory terminology"
        )

    words = re.findall(r"\w+", lowered)

    if len(words) >= 30:

        frequencies = Counter(words)

        most_common_count = frequencies.most_common(1)[0][1]

        if most_common_count / len(words) > 0.30:
            reasons.append(
                "Unusually repetitive content"
            )

    status = (
        "REVIEW"
        if reasons
        else "PASS"
    )

    return AuditResult(
        document=document,
        status=status,
        reasons=reasons
    )


def audit_documents(
    documents: list[Document]
) -> list[AuditResult]:

    return [
        audit_document(document)
        for document in documents
    ]
