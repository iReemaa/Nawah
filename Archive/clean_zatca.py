"""Create a conservative, unapproved ZATCA content-review file from the extraction report.

Run from the Nawah project root: python -m app.clean_zatca
No network calls, no edits to knowledge.json or the FAISS index.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

INPUT = Path('data/processed/zatca_extraction_review.json')
OUTPUT = Path('data/processed/zatca_cleaned_review.json')
START = 'الرئيسية '
SERVICE_END = 'الموارد المفيدة'
FEEDBACK_END = 'التعليقات والاقتراحات'
SERVICE_RELATED = 'خدمات ذات صلة'


def extract_section(text: str, url: str) -> tuple[str, list[str]]:
    warnings = []
    start = text.find(START)
    if start < 0:
        return '', ['Missing start marker; manual extraction required.']
    start += len(START)
    # Service pages include a related-services section with descriptions of OTHER services.
    service_page = '/eServices/Pages/' in url
    markers = [FEEDBACK_END]
    if service_page:
        markers += [SERVICE_END, SERVICE_RELATED, 'تم تقييم هذه الخدمة']
    ends = [text.find(marker, start) for marker in markers]
    ends = [pos for pos in ends if pos >= 0]
    if not ends:
        return '', ['Missing end marker; manual extraction required.']
    end = min(ends)
    section = text[start:end].strip()
    if len(section) < 120:
        warnings.append('Extracted section unusually short; inspect original report.')
    if '<ul>' in section or '<li>' in section or '&#160;' in section:
        warnings.append('HTML fragments remain; preserve original wording and review manually.')
    if 'https: //' in section or '. pdf' in section:
        warnings.append('URLs may be corrupted by upstream cleaning; verify links on the original page.')
    if 'اضغط هنا' in section or 'الجدول أعلاه' in section or 'صورة الصفحة' in section:
        warnings.append('Referenced links, tables or images are not fully represented; check source.')
    if 'مدة تنفيذ الخدمة تكلفة الخدمة' in section:
        warnings.append('Duration/cost fields appear empty; do not infer values.')
    if not service_page:
        warnings.append('Informational page: linked PDFs, images and legal details are not included automatically.')
    warnings.append('Not verified for legal completeness, applicability or currency.')
    return section, warnings


def main() -> None:
    if not INPUT.is_file():
        raise FileNotFoundError(f'Missing input report: {INPUT}')
    if OUTPUT.exists():
        raise FileExistsError(f'Refusing to overwrite existing review file: {OUTPUT}')
    raw = INPUT.read_bytes()
    report = json.loads(raw.decode('utf-8'))
    results = report.get('results')
    if not isinstance(results, list) or len(results) != 11:
        raise ValueError('Expected the 11-source ZATCA extraction report; input differs.')
    output = []
    urls = set()
    for item in results:
        url = item.get('source_url')
        text = item.get('extracted_text')
        if not isinstance(url, str) or url in urls:
            raise ValueError('Missing or duplicate source URL.')
        urls.add(url)
        if item.get('status') != 'REVIEW' or not isinstance(text, str):
            output.append({'source_url': url, 'review_status': 'EXTRACTION_FAILED',
                           'warnings': ['Original report did not contain reviewable text.']})
            continue
        section, warnings = extract_section(text, url)
        # Preserve an exact substring: no fabricated fields, rewording, or URL repair.
        if section and section not in text:
            raise AssertionError('Extracted content is not an exact substring of the source text.')
        output.append({
            'source_url': url,
            'metadata': item.get('metadata', {}),
            'review_status': 'PENDING_HUMAN_REVIEW' if section else 'MANUAL_EXTRACTION_REQUIRED',
            'original_text_sha256': hashlib.sha256(text.encode('utf-8')).hexdigest(),
            'extracted_section': section,
            'warnings': warnings,
            'approved': False,
        })
    result = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'input_file': str(INPUT),
        'input_file_sha256': hashlib.sha256(raw).hexdigest(),
        'total_sources': len(output),
        'approved': 0,
        'note': 'Conservative text boundaries only. All entries require human review and original-page verification.',
        'results': output,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'Review file: {OUTPUT}')
    print(f'Sources: {len(output)} | Approved: 0')
    for row in output:
        print(f"{row['review_status']}: {row['source_url']} ({len(row.get('extracted_section', ''))} chars)")


if __name__ == '__main__':
    main()
