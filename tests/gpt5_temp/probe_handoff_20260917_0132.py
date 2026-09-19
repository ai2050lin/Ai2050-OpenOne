# -*- coding: utf-8 -*-
"""Handoff probe 2026-09-17: verify phase2807/2808/2809/2810 products,
dump result.json summaries, check dependency modules and CUDA."""
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(r'D:\AI2050\Ai2050-OpenOne')
GLM5 = ROOT / 'tests' / 'glm5'
BASE = GLM5 / 'result' / 'rdc_query_construction_20260913'
REPORT = ROOT / 'tests' / 'gpt5_temp' / 'probe_handoff_20260917_0132.txt'

lines = []


def w(s=''):
    lines.append(str(s))


w('=== probe time: %s ===' % time.strftime('%Y-%m-%d %H:%M:%S'))

# ---------- 1. product dirs ----------
for ph in ['phase2806', 'phase2807', 'phase2808', 'phase2809', 'phase2810']:
    d = BASE / ph
    w('')
    w('--- %s ---' % d)
    if not d.exists():
        w('MISSING')
        continue
    for dp, dn, fn in os.walk(d):
        for f in sorted(fn):
            p = Path(dp) / f
            w('%10d  %s' % (p.stat().st_size, p))


def smart(x, depth=0, maxlist=30):
    ind = '  ' * depth
    if isinstance(x, dict):
        return '\n'.join('%s%s: %s' % (ind, k, smart(v, depth + 1, maxlist))
                         for k, v in x.items())
    if isinstance(x, list):
        if len(x) > maxlist:
            return '[list len=%d] first=%s' % (
                len(x), json.dumps(x[:maxlist], ensure_ascii=False))
        return json.dumps(x, ensure_ascii=False)
    return json.dumps(x, ensure_ascii=False)


# ---------- 2. result/execution dumps ----------
for rel in ['phase2807/qwen4_heldout/execution.json',
            'phase2807/qwen4_heldout/result.json',
            'phase2808/crossmodel_hierarchy/result.json',
            'phase2809/qwen4_nesting_tautology/result.json',
            'phase2806/qwen4_hierarchy/execution.json']:
    p = BASE / rel
    w('')
    w('===== %s =====' % rel)
    if not p.exists():
        w('MISSING')
        continue
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        w(smart(data, maxlist=30))
    except Exception as e:
        w('ERROR %s' % e)

# ---------- 3. dependency modules ----------
checks = [
    ('rdc_construction_common.py',
     [r'^ROOT\s*=', r'^BASE\s*=', r'def snapshot', r'def ledger']),
    ('rdc_feature_common.py',
     [r'def stamp', r'def save', r'def npz']),
    ('phase2662_symmetric_mapping_contract.py', [r'def load_native']),
]
for mod, pats in checks:
    p = GLM5 / mod
    w('')
    w('===== %s =====' % mod)
    if not p.exists():
        w('MISSING')
        continue
    src = p.read_text(encoding='utf-8', errors='replace').splitlines()
    seen = set()
    for i, ln in enumerate(src):
        for pat in pats:
            if re.search(pat, ln) and i not in seen:
                seen.add(i)
                w('--- match @%d ---' % (i + 1))
                for j in range(i, min(i + 28, len(src))):
                    w('%5d %s' % (j + 1, src[j]))
                break

# ---------- 4. torch / cuda ----------
w('')
w('=== torch / cuda ===')
try:
    import torch
    w('torch %s cuda=%s ngpu=%s' % (torch.__version__,
                                    torch.cuda.is_available(),
                                    torch.cuda.device_count()))
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        w('gpu0 %s total=%.1fGB' % (props.name, props.total_memory / 2**30))
except Exception as e:
    w('torch error: %r' % e)

# ---------- 5. model dir ----------
w('')
w('=== model dir ===')
mp = ROOT / 'models' / 'hf' / 'qwen3-4b'
w('%s exists=%s' % (mp, mp.exists()))
if mp.exists():
    for f in sorted(mp.iterdir())[:25]:
        w('  %s' % f.name)

# ---------- 6. cc module values (safe import) ----------
w('')
w('=== cc.ROOT / cc.BASE ===')
try:
    sys.path.insert(0, str(GLM5))
    import rdc_construction_common as cc
    w('cc.ROOT=%s' % cc.ROOT)
    w('cc.BASE=%s' % cc.BASE)
except Exception as e:
    w('cc import error: %r' % e)

REPORT.parent.mkdir(parents=True, exist_ok=True)
REPORT.write_text('\n'.join(lines), encoding='utf-8')
print('PROBE DONE -> %s' % REPORT)
