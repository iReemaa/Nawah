"""Read-only, evidence-gated Nawah retrieval.

Place in app/unified_retrieval.py. Run:
    python -m app.unified_retrieval "your question"

Never treats structured records as approved. Does not modify ingestion, reviews,
knowledge.json, decisions, or the FAISS index. No LLM generation is performed.
"""
import json
import sys
from pathlib import Path

from app import knowledge_base as kb

KNOWLEDGE = Path('data/knowledge.json')


def retrieve(question: str, k: int = 4) -> dict:
    """Return approved evidence and safe, non-regulatory structured catalog metadata."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError('Question must be a nonempty string.')
    if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 10:
        raise ValueError('k must be an integer from 1 to 10.')

    # Read structured information without publishing unverified rules or dependencies.
    with KNOWLEDGE.open(encoding='utf-8') as stream:
        structured = json.load(stream)
    if not isinstance(structured, dict):
        raise ValueError('knowledge.json must be a JSON object.')
    expected = ('source_registry', 'balady_activities_with_rules',
                'sbc_activities', 'service_knowledge', 'activity_crosswalk_candidates')
    if any(key not in structured for key in expected):
        raise ValueError('knowledge.json is missing required sections.')
    if not isinstance(structured['source_registry'], list):
        raise ValueError('source_registry must be a list.')

    # Same approval/version checks as app.knowledge_base.search(), before loading pickle.
    if not kb.INDEX.is_dir() or not kb.MANIFEST.is_file():
        raise RuntimeError('No FAISS index/manifest. No evidence returned; do not build automatically.')
    approved = kb.approved_rows()
    approved_ids = sorted(doc_id for doc_id, _ in approved)
    manifest = json.loads(kb.MANIFEST.read_text(encoding='utf-8'))
    if not approved_ids or approved_ids != manifest.get('document_ids') or manifest.get('model') != kb.MODEL:
        raise RuntimeError('FAISS index is stale or approvals changed. No evidence returned.')

    # A registry entry is a source address, NOT proof that its rules are approved.
    registry = {}
    for source in structured['source_registry']:
        if isinstance(source, dict) and isinstance(source.get('url'), str):
            registry[source['url'].rstrip('/').casefold()] = source

    _, _, FAISS, Embeddings = kb.dependencies()
    embedding = Embeddings(model_name=kb.MODEL, model_kwargs={'device': 'cpu'})
    # Load only an index generated and controlled locally by this project.
    store = FAISS.load_local(str(kb.INDEX), embedding,
                             allow_dangerous_deserialization=True)
    permitted = set(approved_ids)
    evidence = []
    for doc in store.similarity_search(question.strip(), k=k):
        meta = doc.metadata
        doc_id = meta.get('document_id')
        if doc_id not in permitted:
            raise RuntimeError('Index returned a document without current approval.')
        url = meta.get('source_url')
        if not isinstance(url, str) or not url:
            raise RuntimeError('Approved evidence lacks a source URL.')
        source = registry.get(url.rstrip('/').casefold())
        evidence.append({
            'document_id': doc_id,
            'source_url': url,
            'page_number': meta.get('page_number'),
            'chunk_number': meta.get('chunk_number'),
            'text': doc.page_content,
            'registry_metadata': ({
                'authority': source.get('authority'),
                'category': source.get('category'),
                'business_activity': source.get('business_activity'),
            } if source else None),
        })
    return {
        'question': question.strip(),
        'evidence': evidence,
        'structured_status': 'UNVERIFIED_NOT_USED_FOR_REGULATORY_CLAIMS',
        'structured_rules_returned': 0,
        'note': ('Registry metadata is descriptive only; a matching source URL does not '
                 'approve any structured requirement or dependency. No LLM answer generated.'),
    }


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m app.unified_retrieval "question"')
    try:
        print(json.dumps(retrieve(' '.join(sys.argv[1:])), ensure_ascii=False, indent=2))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        raise SystemExit(f'Retrieval stopped safely: {exc}') from exc


if __name__ == '__main__':
    main()
