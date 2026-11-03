"""Validate the review and optionally append while preserving the exact prefix."""
import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

root = Path(__file__).resolve().parents[2]
out = root / 'tests/glm5/result/phase2760_system_review_20261103'
memo = root / 'research/glm5/docs/AGI_GLM5_MEMO.md'
draft = root / 'tests/glm5_temp/phase2760_system_review_draft.md'
parser = argparse.ArgumentParser()
parser.add_argument('--append', action='store_true')
args = parser.parse_args()
sha = lambda b: hashlib.sha256(b).hexdigest()
manifest = json.loads((out/'source_manifest.json').read_text(encoding='utf-8'))
arithmetic = json.loads((out/'arithmetic_checks.json').read_text(encoding='utf-8'))
assert len(arithmetic) == 31 and all(row['pass'] for row in arithmetic)
changed = [row['path'] for row in manifest['sources']
           if sha((root/row['path']).read_bytes()) != row['sha256']]
assert not changed, f'Sources changed since review: {changed}'
old = memo.read_bytes()
old_text = old.decode('utf-8-sig')
assert max(map(int, re.findall(r'^## Phase (\d+)', old_text, re.M))) == 2759
text = draft.read_text(encoding='utf-8')
assert text.count('__REVIEW_TIME__') == 1
assert len(re.findall(r'^### \d+\.', text, re.M)) == 14
assert text.count('$$') % 2 == 0
stamp = datetime.now(ZoneInfo('America/Chicago'))
text = text.replace('__REVIEW_TIME__', stamp.strftime('%Y-%m-%d %H:%M'))
payload = text.encode('utf-8')
review = {'created_at': stamp.isoformat(), 'draft_characters': len(text),
          'sections': 14, 'source_hashes_unchanged': len(manifest['sources']),
          'arithmetic_checks_passed': len(arithmetic), 'math_delimiters_balanced': True,
          'before_bytes': len(old), 'before_sha256': sha(old),
          'append_sha256': sha(payload), 'start_line': old_text.count('\n')+3,
          'status': 'validated_only'}
if args.append:
    assert memo.read_bytes() == old, 'Concurrent memo modification; do not append.'
    with memo.open('ab') as f:
        f.write(payload)
        f.flush()
    after = memo.read_bytes()
    assert after[:len(old)] == old, 'Original prefix changed.'
    assert after[len(old):] == payload, 'Unexpected appended suffix.'
    review.update(status='appended', after_bytes=len(after), after_sha256=sha(after),
                  original_prefix_unchanged=True)
    (out/'append_receipt.json').write_text(json.dumps(review, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
else:
    (out/'delivery_review.json').write_text(json.dumps(review, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
print(json.dumps(review, ensure_ascii=False))
