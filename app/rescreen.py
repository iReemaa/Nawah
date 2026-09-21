"""Re-screen existing ingestion queue without network access or changing decisions."""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from langchain_core.documents import Document
from app.ingestion import audit_document

QUEUE = Path('data/processed/document_review_queue.jsonl')
DECISIONS = Path('data/processed/review_decisions.jsonl')

def main():
    if not QUEUE.is_file():
        raise SystemExit(f'Missing queue: {QUEUE}; run from Nawah project root.')
    rows = [json.loads(line) for line in QUEUE.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not rows:
        raise SystemExit('Queue empty: no changes made.')
    backup = QUEUE.parent / 'review_backups' / ('rescreen_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    backup.mkdir(parents=True, exist_ok=False)
    shutil.copy2(QUEUE, backup / QUEUE.name)
    if DECISIONS.is_file():
        shutil.copy2(DECISIONS, backup / DECISIONS.name)
    counts = {'PASS': 0, 'REVIEW': 0}
    temp = QUEUE.with_suffix('.jsonl.rescreen.tmp')
    try:
        with temp.open('w', encoding='utf-8') as stream:
            for row in rows:
                result = audit_document(Document(page_content=row['content'], metadata=row['metadata']))
                row['status'] = result.status
                row['reasons'] = result.reasons
                # Preserve document_id, original content, metadata, and review decisions.
                counts[result.status] += 1
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        temp.replace(QUEUE)
    finally:
        temp.unlink(missing_ok=True)
    print(f'Re-screened {len(rows)} existing documents: PASS={counts["PASS"]}, REVIEW={counts["REVIEW"]}')
    print(f'Backup: {backup}')
    print('No sources downloaded, no decisions changed, no documents auto-approved.')

if __name__ == '__main__':
    main()
