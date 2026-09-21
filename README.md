# Nawah

Saudi SME compliance knowledge ingestion project.

## Current implementation
- `app/ingestion.py`: source loading, webpage/PDF/API extraction, cleaning, quality screening, document building, ingestion pipeline, and content audit.
- `app/schemas.py`: Pydantic source and document schemas.
- `app/config.py`: environment and model configuration.
- `app/main.py`: ingestion runner and text/audit report exporter.
- `data/sources.json`: source registry.
- `tests/`: extractor validation tests.

The retrieval, embeddings, vector store, agent implementations and LangGraph workflow are **not implemented** in this uploaded version. Empty placeholder files have been removed, not represented as working features.

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m tests.test_api_extractor
python -m tests.test_pdf_extractor
python -m app.main
```

Set `OPENAI_API_KEY` in `.env` as required by `app/config.py`. Do not commit `.env`. Live ingestion depends on external websites and may produce different document counts and failures. Audit PASS indicates automated screening only, not verified regulatory accuracy.

## Quality review (updated)
After `python -m app.main`, inspect `data/processed/content_audit_report.txt`,
`data/processed/document_review_queue.jsonl`, and `data/processed/source_failures.txt`.
The JSONL queue retains text, source metadata, status, and reasons for every document;
no document is certified as legally current by automated screening. Do not feed REVIEW
items into retrieval until reviewed. Failed URLs require individual investigation;
HTTP 403 is not bypassed automatically. The HTML extractor prefers substantial
main/article sections and removes common page furniture, with a body fallback.
