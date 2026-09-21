"""Build/search a FAISS knowledge base from explicitly approved exact document versions.
Install: python -m pip install langchain-core langchain-community langchain-text-splitters sentence-transformers faiss-cpu
Build: python -m app.knowledge_base build
Search: python -m app.knowledge_base search "your question"
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path('data/processed')
QUEUE = ROOT / 'document_review_queue.jsonl'
DECISIONS = ROOT / 'review_decisions.jsonl'
INDEX = ROOT / 'faiss_index'
MANIFEST = ROOT / 'faiss_manifest.json'
MODEL = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'


def rows(path):
    if not path.is_file():
        raise FileNotFoundError(f'Required file missing: {path}')
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def document_id(row):
    meta = row.get('metadata', {})
    canonical = json.dumps({'source_url': meta.get('source_url'),
                            'page_number': meta.get('page_number'),
                            'content': row.get('content')}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def approved_rows():
    decisions = {}
    for item in rows(DECISIONS):
        if 'document_id' in item and 'decision' in item:
            decisions[item['document_id']] = item
    result = []
    seen = set()
    for row in rows(QUEUE):
        doc_id = document_id(row)
        decision = decisions.get(doc_id, {})
        if (doc_id not in seen and decision.get('decision') == 'approved'
                and decision.get('reviewer_note', '').strip()
                and row.get('content', '').strip()
                and row.get('metadata', {}).get('source_url')):
            result.append((doc_id, row))
            seen.add(doc_id)
    return result


def dependencies():
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return Document, RecursiveCharacterTextSplitter, FAISS, HuggingFaceEmbeddings


def build():
    approved = approved_rows()
    print(f'Explicitly approved current document versions: {len(approved)}')
    if not approved:
        if INDEX.exists():
            shutil.rmtree(INDEX)
        MANIFEST.unlink(missing_ok=True)
        print('Index NOT created; any old index invalidated. Review and approve documents first.')
        return
    Document, Splitter, FAISS, Embeddings = dependencies()
    splitter = Splitter(chunk_size=900, chunk_overlap=120,
                        separators=['\n\n', '\n', '。', '. ', ' ', ''])
    chunks = []
    for doc_id, row in approved:
        meta = dict(row['metadata'])
        meta['document_id'] = doc_id
        for number, chunk in enumerate(splitter.split_text(row['content'])):
            if chunk.strip():
                chunks.append(Document(page_content=chunk,
                                       metadata={**meta, 'chunk_number': number}))
    if not chunks:
        raise RuntimeError('No chunks created; index remains unchanged.')
    embedding = Embeddings(model_name=MODEL, model_kwargs={'device': 'cpu'})
    store = FAISS.from_documents(chunks, embedding)
    # Never load an untrusted FAISS pickle. This index is locally generated.
    store.save_local(str(INDEX))
    MANIFEST.write_text(json.dumps({'document_ids': sorted(doc_id for doc_id, _ in approved),
                                    'model': MODEL}, indent=2), encoding='utf-8')
    print(f'Indexed {len(chunks)} chunks from {len(approved)} approved documents: {INDEX}')


def search(question):
    if not INDEX.is_dir() or not MANIFEST.is_file():
        raise SystemExit('No valid index exists. Approve documents and run build first.')
    current = sorted(doc_id for doc_id, _ in approved_rows())
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    if not current or current != manifest.get('document_ids') or manifest.get('model') != MODEL:
        raise SystemExit('Approval decisions or source content changed. Rebuild the index before searching.')
    _, _, FAISS, Embeddings = dependencies()
    embedding = Embeddings(model_name=MODEL, model_kwargs={'device': 'cpu'})
    # FAISS docstore serialization uses pickle; ONLY load an index you built locally
    # and control. Never use downloaded or externally supplied index directories.
    store = FAISS.load_local(str(INDEX), embedding,
                             allow_dangerous_deserialization=True)
    for i, doc in enumerate(store.similarity_search(question, k=4), 1):
        print(f'\nRESULT {i} | {doc.metadata.get("source_url")} | page {doc.metadata.get("page_number")}')
        print(doc.page_content)


if __name__ == '__main__':
    if len(sys.argv) == 2 and sys.argv[1] == 'build':
        build()
    elif len(sys.argv) >= 3 and sys.argv[1] == 'search':
        search(' '.join(sys.argv[2:]))
    else:
        raise SystemExit('Usage: python -m app.knowledge_base build | search "question"')
