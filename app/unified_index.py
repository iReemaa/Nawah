"""Unified Nawah indexing: structured knowledge + quarantined scraped content.
Run: ./.venv/bin/python -m app.unified_index build
Search: ./.venv/bin/python -m app.unified_index search 'question'
Only structured results are returned to normal search; scraped chunks remain quarantined
until a separate evidence-backed automated validation step is implemented.
"""
import hashlib
import json
import sys
from pathlib import Path

DATA = Path('data')
OUT = DATA / 'processed'
MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode()).hexdigest()


def load_knowledge():
    path = DATA / 'knowledge.json'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}; put your manually checked knowledge.json in data/')
    obj = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(obj, dict) or not all(k in obj for k in ('balady_activities_with_rules', 'sbc_activities', 'service_knowledge')):
        raise ValueError('knowledge.json missing expected unified data sections')
    return obj


def structured_records(obj):
    # Preserve one activity per record so activity-specific rules cannot be confused.
    for entry in obj['balady_activities_with_rules']:
        activity = entry.get('activity') or {}
        rules = entry.get('rules') or []
        if not activity.get('activity_id') or not isinstance(rules, list):
            continue
        for rule in rules:
            if not isinstance(rule, dict) or not str(rule.get('text') or '').strip():
                continue
            payload = {'activity': activity, 'rule': rule}
            text = (f"Balady activity: {activity.get('activity_name', '')}\n"
                    f"Activity ID: {activity['activity_id']}\n"
                    f"ISIC: {activity.get('isic_id', '')}\n"
                    f"Rule: {rule['text']}\n"
                    f"Updated: {rule.get('updated_date') or 'unknown'}")
            yield text, {'record_id': digest(payload), 'data_type': 'balady_rule',
                         'activity_id': str(activity['activity_id']),
                         'rule_id': str(rule.get('rule_id') or ''),
                         'source_url': next((u for u in (rule.get('urls') or []) if isinstance(u, str) and u.startswith('https://')), ''),
                         'validation_status': 'structured_user_checked',
                         'source_section': 'balady_activities_with_rules'}
    for entry in obj['sbc_activities']:
        if not isinstance(entry, dict) or not entry.get('activityId'):
            continue
        text = json.dumps(entry, ensure_ascii=False, sort_keys=True)
        yield text, {'record_id': digest(entry), 'data_type': 'sbc_activity',
                     'activity_id': str(entry['activityId']), 'source_url': '',
                     'validation_status': 'structured_user_checked',
                     'source_section': 'sbc_activities'}
    for key, service in obj['service_knowledge'].items():
        if not isinstance(service, dict):
            continue
        # Exclude explicitly unverified and raw content from trusted structured index.
        safe = {k: v for k, v in service.items() if k not in ('unverified_rules', 'unresolved_for_nawah', 'raw_extracted_content')}
        if not safe:
            continue
        yield json.dumps(safe, ensure_ascii=False, sort_keys=True), {
            'record_id': digest(safe), 'data_type': 'service_knowledge',
            'service_id': key, 'source_url': '',
            'validation_status': 'structured_user_checked', 'source_section': 'service_knowledge'}
    # activity_crosswalk_candidates deliberately excluded: CANDIDATE_VERIFY.


def scraped_records():
    path = OUT / 'document_review_queue.jsonl'
    if not path.is_file():
        raise FileNotFoundError(f'Missing {path}; run your existing ingestion/rescreen first')
    with path.open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            content = str(row.get('content') or '').strip()
            meta = row.get('metadata') or {}
            if not content or not meta.get('source_url'):
                continue
            if '�' in content or len(content) < 30:
                continue  # genuinely damaged or empty; not a 150-character rule
            doc_id = row.get('document_id') or digest({'content': content, 'metadata': meta})
            yield content, {'record_id': str(doc_id), 'data_type': 'scraped',
                            'source_url': str(meta['source_url']),
                            'page_number': meta.get('page_number') or -1,
                            'authority': str(meta.get('authority') or ''),
                            'screening_status': str(row.get('status') or ''),
                            'validation_status': 'pending_automated_validation',
                            'source_section': 'document_review_queue'}


def dependencies():
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return Document, RecursiveCharacterTextSplitter, FAISS, HuggingFaceEmbeddings


def build():
    Document, Splitter, FAISS, Embeddings = dependencies()
    splitter = Splitter(chunk_size=900, chunk_overlap=120,
                        separators=['\n\n', '\n', '。', '. ', ' ', ''])
    obj = load_knowledge()
    embedding = Embeddings(model_name=MODEL, model_kwargs={'device': 'cpu'})
    OUT.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, records in [('structured', structured_records(obj)), ('quarantine', scraped_records())]:
        chunks = []
        for content, meta in records:
            for n, part in enumerate(splitter.split_text(content)):
                if part.strip():
                    chunks.append(Document(page_content=part,
                                           metadata={**meta, 'chunk_number': n,
                                                     'chunk_id': digest([meta['record_id'], n, part])}))
        if not chunks:
            raise RuntimeError(f'No {name} chunks; existing index left unchanged')
        target = OUT / f'unified_{name}_faiss'
        # Separate stores: quarantined content is never available to normal search.
        store = FAISS.from_documents(chunks, embedding)
        store.save_local(str(target))
        counts[name] = len(chunks)
        print(f'{name}: {len(chunks)} chunks -> {target}', flush=True)
    (OUT / 'unified_index_manifest.json').write_text(json.dumps({
        'model': MODEL, 'chunk_counts': counts,
        'structured_source_digest': digest(obj),
        'scraped_queue_digest': hashlib.sha256((OUT / 'document_review_queue.jsonl').read_bytes()).hexdigest(),
        'quarantine_not_for_answers': True,
        'excluded_sections': ['activity_crosswalk_candidates', 'unverified_rules', 'unresolved_for_nawah', 'raw_extracted_content']
    }, indent=2), encoding='utf-8')
    print('Built both indexes. Scraped chunks are QUARANTINED, not approved.', flush=True)


def search(question):
    _, _, FAISS, Embeddings = dependencies()
    manifest_path = OUT / 'unified_index_manifest.json'
    if not manifest_path.is_file():
        raise RuntimeError('Build the unified indexes first')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest['model'] != MODEL or manifest['structured_source_digest'] != digest(load_knowledge()):
        raise RuntimeError('Structured data/model changed: rebuild index')
    # Only load locally generated, trusted pickle-backed FAISS files.
    store = FAISS.load_local(str(OUT / 'unified_structured_faiss'),
                             Embeddings(model_name=MODEL, model_kwargs={'device': 'cpu'}),
                             allow_dangerous_deserialization=True)
    for i, doc in enumerate(store.similarity_search(question, k=4), 1):
        print(f'\nRESULT {i} | {doc.metadata}', flush=True)
        print(doc.page_content, flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'build':
        build()
    elif len(sys.argv) >= 3 and sys.argv[1] == 'search':
        search(' '.join(sys.argv[2:]))
    else:
        raise SystemExit('Usage: python -m app.unified_index build | search "question"')
