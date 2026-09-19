"""Read-only descriptive counts from completed Q4 readout; no fitting."""
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'glm5'))
from rdc_question_common import *

result = read(OUT/'fit/qwen4/readout/nonconfirmation/result.json')
assert result['all_passed']
summary = []
for record in result['records']:
    ref = record['field']; assert sha(ROOT/ref['path']) == ref['sha256']
    with np.load(ROOT/ref['path'], allow_pickle=False) as z: values = z['statistics']
    counts = {}
    for name in ['native_choice', 'predicted_choice']:
        ids, frequency = np.unique(values[:, record['columns'].index(name)].astype(np.int64), return_counts=True)
        counts[name] = {str(i): int(n) for i, n in zip(ids, frequency)}
    summary.append({k: record[k] for k in ['variant', 'kind', 'split']} | counts)
_, _, rows, _ = material('qwen4')
teacher = {}
for split in ['validation', 'diagnostic']:
    ids, frequency = np.unique([r['tokens']['teacher_ids_including_EOS'][0] for r in rows if r['split'] == split], return_counts=True)
    teacher[split] = {str(i): int(n) for i, n in zip(ids, frequency)}
print(json.dumps({'source_readout_sha256': sha(OUT/'fit/qwen4/readout/nonconfirmation/result.json'),
                  'counts': summary, 'teacher_first_ids': teacher}, ensure_ascii=False, indent=2))
