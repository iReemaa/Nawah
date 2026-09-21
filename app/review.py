"""Fail-closed, content-addressed human review. Run: python -m app.review_documents"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path('data/processed')
QUEUE = ROOT / 'document_review_queue.jsonl'
DECISIONS = ROOT / 'review_decisions.jsonl'


def identity(doc):
    meta = doc.get('metadata', {})
    value = json.dumps({'source_url': meta.get('source_url'),
                        'page_number': meta.get('page_number'),
                        'content': doc.get('content')}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(value.encode('utf-8')).hexdigest()


def read_jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    if not QUEUE.is_file():
        raise SystemExit(f'Missing review queue: {QUEUE}')
    docs = read_jsonl(QUEUE)
    decisions = {d['document_id']: d for d in read_jsonl(DECISIONS)
                 if 'document_id' in d and 'decision' in d}
    print(f'{len(docs)} documents. Decisions: {len(decisions)}. Enter a number, N for next undecided, Q to quit.')
    while True:
        selection = input('Document number / N / Q: ').strip().lower()
        if selection == 'q':
            return
        if selection == 'n':
            index = next((i for i, d in enumerate(docs) if identity(d) not in decisions), None)
            if index is None:
                print('No undecided documents.'); continue
        elif selection.isdigit() and 1 <= int(selection) <= len(docs):
            index = int(selection) - 1
        else:
            print('Invalid selection.'); continue
        doc = docs[index]
        meta = doc.get('metadata', {})
        document_id = identity(doc)
        print('\n' + '=' * 65)
        print(f'DOCUMENT {index + 1} | {document_id[:16]}')
        print('SOURCE URL:', meta.get('source_url', 'MISSING'))
        print('PDF PAGE (1-based):', meta.get('page_number', 'N/A'))
        print('AUTHORITY:', meta.get('authority', 'N/A'))
        print('CATEGORY:', meta.get('category', 'N/A'))
        print('SCREENING:', doc.get('status'), '| FLAGS:', doc.get('reasons'))
        print('PREVIOUS DECISION:', decisions.get(document_id, {}).get('decision', 'NONE'))
        print('\nEXTRACTED CONTENT:\n', doc.get('content', ''))
        print('\nCompare this exact page with its official URL. A=approve, R=reject, F=needs extraction repair, S=skip, Q=quit')
        action = input('Decision: ').strip().lower()
        if action == 'q':
            return
        if action == 's':
            continue
        if action not in {'a', 'r', 'f'}:
            print('No decision saved.'); continue
        note = input('Required review note: ').strip()
        if not note:
            print('Note required; nothing saved.'); continue
        if action == 'a' and (not meta.get('source_url') or not doc.get('content')):
            print('Cannot approve without source URL and content.'); continue
        record = {'document_id': document_id,
                  'decision': {'a': 'approved', 'r': 'rejected', 'f': 'needs_repair'}[action],
                  'reviewer_note': note,
                  'reviewed_at': datetime.now(timezone.utc).isoformat(),
                  'source_url': meta.get('source_url'), 'page_number': meta.get('page_number')}
        ROOT.mkdir(parents=True, exist_ok=True)
        with DECISIONS.open('a', encoding='utf-8') as output:
            output.write(json.dumps(record, ensure_ascii=False) + '\n')
        decisions[document_id] = record
        print('Saved:', record['decision'])

if __name__ == '__main__':
    main()
