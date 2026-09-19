"""P2820 hash registration probe."""
import hashlib
import json
import time
from pathlib import Path

ROOT = Path('D:/AI2050/Ai2050-OpenOne')
FILES = [
    ROOT / 'tests' / 'glm5' / 'phase2820_edit_generalization.py',
    ROOT / 'tests' / 'glm5' / 'result' /
    'rdc_query_construction_20260913' / 'phase2820' /
    'edit_generalization' / 'execution.json',
    ROOT / 'tests' / 'glm5' / 'result' /
    'rdc_query_construction_20260913' / 'phase2820' /
    'edit_generalization' / 'result.json',
    ROOT / 'tests' / 'glm5' / 'result' /
    'rdc_query_construction_20260913' / 'phase2820' /
    'edit_generalization' / 'edit.npz',
]
rep = {'timestamp_local': time.strftime('%Y-%m-%d %H:%M:%S'),
       'files': {}}
for p in FILES:
    if p.exists():
        b = p.read_bytes()
        rep['files'][str(p)] = {
            'sha256': hashlib.sha256(b).hexdigest(),
            'bytes': len(b)}
    else:
        rep['files'][str(p)] = 'MISSING'

out = ROOT / 'tests' / 'gpt5_temp' / 'probe_2820_hash.json'
out.write_text(json.dumps(rep, indent=1), encoding='utf-8')
print('WROTE', out)
