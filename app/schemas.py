from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, HttpUrl, field_validator


class SourceConfig(BaseModel):
    authority: str
    category: str
    source_type: Literal["webpage", "pdf", "api"]
    url: HttpUrl
    language: Literal["ar", "en"]
    business_activity: str
    region: str = "national"
    enabled: bool = True

    @field_validator(
        "authority",
        "category",
        "business_activity",
        "region"
    )
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Source text fields cannot be empty."
            )

        return value


class DocumentMetadata(BaseModel):
    authority: str
    category: str
    source_url: HttpUrl
    source_type: Literal["webpage", "pdf", "api"]
    language: Literal["ar", "en"]
    business_activity: str
    region: str
    ingested_at: datetime

    title: Optional[str] = None
    page_number: Optional[int] = None

    @field_validator(
        "authority",
        "category",
        "business_activity",
        "region"
    )
    @classmethod
    def validate_metadata_text(cls, value: str) -> str:
        value = value.strip()

        if not value:
            raise ValueError(
                "Metadata text fields cannot be empty."
            )

        return value

    @field_validator("title")
    @classmethod
    def validate_optional_title(
        cls,
        value: Optional[str]
    ) -> Optional[str]:

        if value is None:
            return None

        value = value.strip()

        if not value:
            return None

        return value

    @field_validator("page_number")
    @classmethod
    def validate_page_number(
        cls,
        value: Optional[int]
    ) -> Optional[int]:

        if value is not None and value < 1:
            raise ValueError(
                "Page number must be greater than or equal to 1."
            )

        return value


class ExtractedPDFPage(BaseModel):
    page_number: int
    text: str

    @field_validator("page_number")
    @classmethod
    def validate_pdf_page_number(
        cls,
        value: int
    ) -> int:

        if value < 1:
            raise ValueError(
                "PDF page number must be greater than or equal to 1."
            )

        return value

    @field_validator("text")
    @classmethod
    def validate_pdf_text(
        cls,
        value: str
    ) -> str:

        value = value.strip()

        if not value:
            raise ValueError(
                "Extracted PDF page text cannot be empty."
            )

        return value