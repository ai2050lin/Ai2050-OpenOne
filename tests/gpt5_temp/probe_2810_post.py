# -*- coding: utf-8 -*-
"""Phase 2810 post-run probe: verify products on real disk, dump verdict,
register SHA256 hashes."""
import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(r'D:\AI2050\Ai2050-OpenOne')
GLM5 = ROOT / 'tests' / 'glm5'
BASE = GLM5 / 'result' / 'rdc_query_construction_20260913'
PH = BASE / 'phase2810' / 'phrase_modulation'
REPORT = ROOT / 'tests' / 'gpt5_temp' / 'probe_2810_hashes.txt'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as s:
        for chunk in iter(lambda: s.read(8 * 1024 ** 2), b''):
            h.update(chunk)
    return h.hexdigest()


lines = []
w = lines.append
w('=== probe time: %s ===' % time.strftime('%Y-%m-%d %H:%M:%S'))
w('')
w('script %s' % sha(GLM5 / 'phase2810_phrase_modulation.py'))

if not PH.exists():
    w('PHASE2810 DIR MISSING')
else:
    for dp, dn, fn in os.walk(PH):
        for f in sorted(fn):
            p = Path(dp) / f
            w('%10d  sha=%s  %s' % (p.stat().st_size, sha(p), p))

    rj = PH / 'result.json'
    if rj.exists():
        data = json.loads(rj.read_text(encoding='utf-8'))
        w('')
        w('--- verdict ---')
        w(json.dumps(data.get('verdict', {}), ensure_ascii=False, indent=2))
        w('')
        w('--- prereg ---')
        w(json.dumps(data.get('prereg', {}), ensure_ascii=False, indent=2))
        w('')
        w('--- per_word summary (final layer) ---')
        for wr in data.get('per_word', []):
            row = {'word': wr['word'], 'cat': wr['category']}
            for cn, cd in wr['conditions'].items():
                row[cn + '_cos_f'] = cd['cos'][-1]
                row[cn + '_shift_f'] = cd['rel_shift'][-1]
                row[cn + '_cls'] = cd['class_share'][-1]
                row[cn + '_dom'] = cd['dom_share'][-1]
            w(json.dumps(row, ensure_ascii=False))

    ej = PH / 'execution.json'
    if ej.exists():
        e = json.loads(ej.read_text(encoding='utf-8'))
        w('')
        w('--- execution ---')
        w('timestamp: %s' % e.get('timestamp'))
        w('seed: %s' % e.get('seed'))
        w('source: %s' % json.dumps(e.get('source', {}), ensure_ascii=False))

REPORT.write_text('\n'.join(lines), encoding='utf-8')
print('PROBE DONE -> %s' % REPORT)
