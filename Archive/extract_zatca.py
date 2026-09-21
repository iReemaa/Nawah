import json
from datetime import datetime, timezone
from pathlib import Path

from app.ingestion import (
    load_sources,
    process_source,
    validate_document_quality,
    audit_document,
)

# The 11 URLs we just added to sources.json.
TARGET_URLS = {
    "https://zatca.gov.sa/ar/eServices/Pages/eServices-009.aspx",
    "https://zatca.gov.sa/ar/eServices/Pages/eServices-225.aspx",
    "https://zatca.gov.sa/ar/eServices/Pages/eServices-234.aspx",
    "https://zatca.gov.sa/ar/eServices/Pages/eServices-048.aspx",
    "https://zatca.gov.sa/ar/eServices/Pages/eServices-001.aspx",
    "https://zatca.gov.sa/ar/eServices/UserManual/Pages/UserManual-001.aspx?service=eServices-001",
    "https://zatca.gov.sa/ar/E-Invoicing/Pages/default.aspx",
    "https://zatca.gov.sa/ar/E-Invoicing/Introduction/Pages/Roll-out-phases.aspx",
    "https://zatca.gov.sa/ar/E-Invoicing/Introduction/Guidelines/Pages/default.aspx",
    "https://zatca.gov.sa/ar/eServices/Pages/eServices-273.aspx",
    "https://zatca.gov.sa/ar/eServices/Pages/Product-Identification.aspx",
}

OUTPUT = Path("data/processed/zatca_extraction_review.json")


def main():
    sources = [
        source
        for source in load_sources()
        if str(source.url) in TARGET_URLS and source.enabled
    ]

    found_urls = {str(source.url) for source in sources}
    missing_urls = sorted(TARGET_URLS - found_urls)

    if missing_urls:
        raise RuntimeError(
            "Some ZATCA URLs are missing or disabled:\n"
            + "\n".join(missing_urls)
        )

    results = []

    for index, source in enumerate(sources, start=1):
        print(f"[{index}/{len(sources)}] {source.url}")

        try:
            documents = process_source(source)

            for document in documents:
                validate_document_quality(document)
                audit = audit_document(document)

                # ZATCA body fallback may include website navigation.
                # Every document requires manual content review.
                reasons = list(audit.reasons)
                reasons.append(
                    "Manual review required: verify service content "
                    "and remove unrelated website navigation."
                )

                results.append({
                    "source_url": str(source.url),
                    "status": "REVIEW",
                    "audit_screening_status": audit.status,
                    "reasons": reasons,
                    "metadata": document.metadata,
                    "extracted_text": document.page_content,
                })

            print(f"  Extracted {len(documents)} document(s)")

        except Exception as error:
            results.append({
                "source_url": str(source.url),
                "status": "FAILED",
                "error": str(error),
            })
            print(f"  FAILED: {error}")

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "total_sources": len(sources),
        "approved": 0,
        "results": results,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    # Never overwrite a previous review report.
    if OUTPUT.exists():
        raise FileExistsError(
            f"Review report already exists: {OUTPUT}. "
            "Preserve it before creating another report."
        )

    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\nExtraction finished.")
    print(f"Report: {OUTPUT}")
    print(f"Documents extracted: {sum(r['status'] == 'REVIEW' for r in results)}")
    print(f"Failures: {sum(r['status'] == 'FAILED' for r in results)}")
    print("Approved: 0 — manual review is still required.")


if __name__ == "__main__":
    main()