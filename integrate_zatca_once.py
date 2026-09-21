"""One-time, fail-closed ZATCA queue integration. Run from Nawah project root.
Does not modify review decisions, knowledge.json, or FAISS.
"""
import hashlib
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('data/processed')
QUEUE = ROOT / 'document_review_queue.jsonl'
DECISIONS = ROOT / 'review_decisions.jsonl'
INPUT = ROOT / 'zatca_cleaned_review.json'


def doc_id(content, metadata):
    identity = json.dumps({'source_url': metadata.get('source_url'),
                           'page_number': metadata.get('page_number'),
                           'content': content}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(identity.encode('utf-8')).hexdigest()


def main():
    if not QUEUE.is_file() or not DECISIONS.is_file() or not INPUT.is_file():
        raise SystemExit('STOP: queue, decisions, and cleaned ZATCA report must all exist.')
    original = QUEUE.read_bytes()
    existing = [json.loads(line) for line in original.decode('utf-8').splitlines() if line.strip()]
    if not existing or len({r['document_id'] for r in existing}) != len(existing):
        raise SystemExit('STOP: existing queue is empty or has duplicate document IDs.')
    if any(r['document_id'] != doc_id(r['content'], r['metadata']) for r in existing):
        raise SystemExit('STOP: existing queue has a document ID/content mismatch.')
    report = json.loads(INPUT.read_text(encoding='utf-8'))
    results = report.get('results', [])
    if len(results) != 11 or report.get('approved') != 0:
        raise SystemExit('STOP: expected exactly 11 unapproved ZATCA results.')
    existing_ids = {r['document_id'] for r in existing}
    existing_urls = {r['metadata']['source_url'] for r in existing}
    new = []
    for result in results:
        url = result.get('source_url')
        meta = result.get('metadata')
        content = result.get('extracted_section')
        if (not isinstance(meta, dict) or not isinstance(content, str)
                or len(content.strip()) < 200 or not isinstance(url, str)
                or not url.startswith('https://zatca.gov.sa/')
                or meta.get('source_url') != url
                or result.get('approved') is not False
                or result.get('review_status') != 'PENDING_HUMAN_REVIEW'):
            raise SystemExit(f'STOP: invalid or unexpectedly approved source: {url}')
        identifier = doc_id(content, meta)
        if identifier in existing_ids or url in existing_urls:
            raise SystemExit(f'STOP: source already in queue; resolve duplicate manually: {url}')
        existing_ids.add(identifier)
        existing_urls.add(url)
        new.append({'document_id': identifier, 'status': 'REVIEW',
                    'approved_for_indexing': False,
                    'reasons': ['ZATCA cleaned text requires original-page human verification',
                                *result.get('warnings', [])],
                    'content': content, 'metadata': meta})
    if len(new) != 11:
        raise SystemExit('STOP: unexpected new document count.')
    backup = ROOT / 'review_backups' / ('zatca_integration_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(QUEUE, backup / QUEUE.name)
    shutil.copy2(DECISIONS, backup / DECISIONS.name)
    if QUEUE.read_bytes() != original:
        raise SystemExit('STOP: queue changed during preparation; backup created, nothing overwritten.')
    fd, temp_name = tempfile.mkstemp(prefix='.zatca_queue_', suffix='.tmp', dir=ROOT)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as out:
            for row in existing + new:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp_name, QUEUE)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    print(f'Queue: {len(existing)} -> {len(existing) + len(new)} documents')
    print('Added: 11 ZATCA documents; all REVIEW and unapproved')
    print(f'Backup: {backup}')
    print('Decisions, knowledge.json, and FAISS unchanged.')


if __name__ == '__main__':
    main()
