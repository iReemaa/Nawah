"""Read-only, bounded ZATCA link discovery. Does not modify Nawah data or index."""
import argparse
import csv
import json
import re
import time
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse, urldefrag, unquote

import requests
from bs4 import BeautifulSoup

SEED = 'https://zatca.gov.sa/en/RulesRegulations/Pages/rules.aspx'
HOSTS = {'zatca.gov.sa', 'www.zatca.gov.sa'}
TOPICS = ('rulesregulations', 'vat', 'zakat', 'income-tax', 'incometax', 'e-invoic',
          'einvoic', 'fatoora', 'customs', 'regulation', 'لائحة', 'الزكاة', 'الضريبة',
          'الفوترة', 'الجمارك')
EXTENSIONS = ('.pdf', '.doc', '.docx', '.xls', '.xlsx')
HEADERS = {'User-Agent': 'NawahSourceDiscovery/1.0 (research; bounded requests)',
           'Accept-Language': 'en,ar;q=0.9'}


def normalize(url):
    url, _ = urldefrag(url.strip())
    p = urlparse(url)
    if p.scheme not in ('https', 'http') or (p.hostname or '').lower() not in HOSTS:
        return None
    if p.username or p.password or p.port not in (None, 80, 443):
        return None
    return p._replace(scheme='https', netloc='zatca.gov.sa').geturl()


def relevant(url, title):
    return any(word in (unquote(url) + ' ' + title).lower() for word in TOPICS)


def discover(max_pages=60, depth=3, delay=1.0, output=Path('data/processed/zatca_discovery.csv')):
    if max_pages < 1 or depth < 0 or delay < 0.5:
        raise ValueError('Require max-pages >= 1, depth >= 0, delay >= 0.5')
    registry = Path('data/sources.json')
    registered = set()
    if registry.exists():
        data = json.loads(registry.read_text(encoding='utf-8'))
        if not isinstance(data, list):
            raise ValueError('sources.json must contain a JSON array')
        registered = {normalize(str(item.get('url', ''))) for item in data if isinstance(item, dict)}
    queue = deque([(SEED, 0, 'seed')])
    queued = {SEED}
    visited = set()
    found = {}
    errors = []
    session = requests.Session()
    session.headers.update(HEADERS)
    while queue and len(visited) < max_pages:
        url, level, parent = queue.popleft()
        if url in visited:
            continue
        visited.add(url)
        try:
            response = session.get(url, timeout=20, allow_redirects=False)
            if response.status_code in (301, 302, 303, 307, 308):
                target = normalize(urljoin(url, response.headers.get('Location', '')))
                if target and target not in queued and level <= depth:
                    queue.appendleft((target, level, parent))
                    queued.add(target)
                else:
                    errors.append({'url': url, 'issue': 'Redirect not followed: external/duplicate/missing target'})
                continue
            if response.status_code != 200:
                errors.append({'url': url, 'issue': f'HTTP {response.status_code}'})
                continue
            content_type = response.headers.get('Content-Type', '').lower()
            if 'text/html' not in content_type:
                errors.append({'url': url, 'issue': f'Not HTML: {content_type}'})
                continue
            soup = BeautifulSoup(response.content, 'html.parser')
            links = []
            for element in soup.select('a[href], iframe[src], embed[src], object[data]'):
                raw = element.get('href') or element.get('src') or element.get('data')
                if not raw:
                    continue
                link = normalize(urljoin(url, raw))
                if not link:
                    continue
                title = ' '.join(element.stripped_strings) or element.get('title', '') or element.get('aria-label', '')
                title = re.sub(r'\s+', ' ', title).strip()[:250]
                links.append((link, title, element.name))
            # Some viewer pages put PDF URLs in HTML attributes rather than anchors.
            for element in soup.select('[data-url], [data-src]'):
                raw = element.get('data-url') or element.get('data-src')
                link = normalize(urljoin(url, raw)) if raw else None
                if link:
                    links.append((link, element.get('title', '')[:250], 'embedded-attribute'))
            for link, title, kind in links:
                is_document = urlparse(link).path.lower().endswith(EXTENSIONS)
                if is_document:
                    found.setdefault(link, {'url': link, 'title': title, 'parent_page': url,
                                            'discovered_via': kind, 'type': 'pdf' if link.lower().split('?')[0].endswith('.pdf') else 'other_document',
                                            'already_registered': link in registered,
                                            'status': 'UNVERIFIED_DISCOVERY'})
                elif level < depth and relevant(link, title) and link not in queued:
                    queue.append((link, level + 1, url))
                    queued.add(link)
        except requests.RequestException as exc:
            errors.append({'url': url, 'issue': f'{type(exc).__name__}: {exc}'})
        finally:
            time.sleep(delay)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=['url', 'title', 'parent_page', 'discovered_via', 'type', 'already_registered', 'status'])
        writer.writeheader()
        writer.writerows(sorted(found.values(), key=lambda row: row['url']))
    report = {'pages_checked': len(visited), 'documents_discovered': len(found),
              'already_registered': sum(row['already_registered'] for row in found.values()),
              'errors': errors, 'pending_pages': len(queue), 'csv': str(output),
              'note': 'Discovery only; no document contents or regulatory claims verified. Dynamic JS viewers may be missed.'}
    report_path = output.with_suffix('.report.json')
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-pages', type=int, default=60)
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--delay', type=float, default=1.0)
    args = parser.parse_args()
    discover(args.max_pages, args.depth, args.delay)
