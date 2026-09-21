"""Validate simplified Nawah structured data; never approve or index automatically.
Run: python -m app.structured_loader
"""
import json
from collections import Counter
from pathlib import Path

DATA = Path('data')

def load(path, expected):
    with path.open(encoding='utf-8') as stream:
        value = json.load(stream)
    if not isinstance(value, expected):
        raise ValueError(f'{path}: expected {expected.__name__}')
    return value

def audit():
    knowledge = load(DATA / 'knowledge.json', dict)
    issues = load(DATA / 'quality_issues.json', list)
    sources = knowledge['source_registry']
    balady = knowledge['balady_activities_with_rules']
    sbc = knowledge['sbc_activities']
    services = knowledge['service_knowledge']
    crosswalk = knowledge['activity_crosswalk_candidates']
    errors = []
    if not all(isinstance(v, list) for v in (sources, balady, sbc, crosswalk)) or not isinstance(services, dict):
        raise ValueError('Unexpected knowledge.json section types')
    source_keys = set()
    for i, row in enumerate(sources):
        if not isinstance(row, dict) or not isinstance(row.get('url'), str) or not row['url'].startswith('https://'):
            errors.append(f'source_registry[{i}]: missing HTTPS URL')
            continue
        key = row['url'].rstrip('/').casefold()
        if key in source_keys:
            errors.append(f'source_registry[{i}]: duplicate URL')
        source_keys.add(key)
    balady_ids = set()
    rule_count = 0
    for i, row in enumerate(balady):
        if not isinstance(row, dict) or not isinstance(row.get('activity'), dict) or not isinstance(row.get('rules'), list):
            errors.append(f'balady[{i}]: invalid activity/rules structure')
            continue
        activity_id = str(row['activity'].get('activity_id', '')).strip()
        if not activity_id or activity_id in balady_ids:
            errors.append(f'balady[{i}]: missing/duplicate activity ID')
        balady_ids.add(activity_id)
        rule_count += len(row['rules'])
    sbc_ids = set()
    for i, row in enumerate(sbc):
        if not isinstance(row, dict):
            errors.append(f'sbc[{i}]: expected object')
            continue
        activity_id = str(row.get('activityId', '')).strip()
        if not activity_id or activity_id in sbc_ids:
            errors.append(f'sbc[{i}]: missing/duplicate activity ID')
        sbc_ids.add(activity_id)
    for i, row in enumerate(crosswalk):
        if not isinstance(row, dict) or str(row.get('balady_activity_id')) not in balady_ids or str(row.get('sbc_activity_id')) not in sbc_ids:
            errors.append(f'crosswalk[{i}]: unknown activity reference')
    if not all(key in services for key in ('ejar', 'misa', 'sbc')):
        errors.append('service_knowledge: missing ejar/misa/sbc section')
    if len(issues) == 0:
        errors.append('quality_issues.json is empty: unresolved verification status cannot be established')
    return {
        'status': 'STRUCTURE_VALID_WITH_UNVERIFIED_CONTENT' if not errors else 'STRUCTURE_ERRORS',
        'counts': {'sources': len(sources), 'balady_activities': len(balady), 'balady_rules': rule_count,
                   'sbc_activities': len(sbc), 'crosswalk_candidates_unverified': len(crosswalk),
                   'quality_issues_unresolved': len(issues)},
        'issue_severities': dict(Counter(str(x.get('severity', 'unspecified')) for x in issues if isinstance(x, dict))),
        'errors': errors[:100], 'error_count': len(errors), 'indexing_allowed': False,
        'note': 'Structural validation is not source verification. No structured records are approved or indexed.'
    }

if __name__ == '__main__':
    result = audit()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result['error_count']:
        raise SystemExit(1)
