# -*- coding: utf-8 -*-
"""3162 probe: index all evidence files for atlas audit."""
import os, re, json

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3162_probe_out.txt')
L = []
def w(s):
    L.append(str(s))

def flatten(d, prefix='', out=None, depth=0):
    if out is None:
        out = {}
    if depth > 4:
        return out
    if isinstance(d, dict):
        for k, v in d.items():
            flatten(v, prefix + '.' + str(k) if prefix else str(k), out, depth + 1)
    elif isinstance(d, list):
        if len(d) <= 8:
            for i, v in enumerate(d):
                flatten(v, prefix + '[%d]' % i, out, depth + 1)
        else:
            out[prefix] = 'list(len=%d)' % len(d)
    else:
        s = repr(d)
        if len(s) > 100:
            s = s[:100] + '...'
        out[prefix] = s
    return out

# ---- A. glm5 result files ----
w('== A. glm5 result files')
glm_files = []
gdir = os.path.join(ROOT, 'tests', 'glm5', 'result')
for dp, dns, fns in os.walk(gdir):
    for fn in fns:
        if fn.startswith('result') and fn.endswith('.json'):
            glm_files.append(os.path.join(dp, fn))
glm_files.sort()
for p in glm_files:
    w(os.path.relpath(p, ROOT))

w('')
w('== B. glm5 verdicts + flattened values')
for p in glm_files:
    try:
        r = json.load(open(p, encoding='utf-8'))
    except Exception as e:
        w('%s -> LOAD_ERR %s' % (os.path.relpath(p, ROOT), e))
        continue
    w('--- ' + os.path.relpath(p, ROOT))
    flat = flatten(r)
    for k in sorted(flat):
        w('  %s = %s' % (k, flat[k]))

# ---- C. deepseek result dirs ----
w('')
w('== C. deepseek result dirs (with result json)')
ddir = os.path.join(ROOT, 'tests', 'deepseek', 'result')
ds_files = []
for dp, dns, fns in os.walk(ddir):
    for fn in fns:
        if fn.startswith('result') and fn.endswith('.json'):
            ds_files.append(os.path.join(dp, fn))
ds_files.sort()
w('total deepseek result files: %d' % len(ds_files))

kw = ['e_read', 'eread', 'e_ar', 'ear', 'steer', 'gate', 'q0', 'phase38', 'phase39',
      'phase40', 'phase41', 'phase42', 'r08', 'r8', 'r9', 'r10', 'r11', 'read', 'wlr', 'wlr_']
cand = [p for p in ds_files if any(k in p.lower().replace('\\', '/').split('/')[-2] + '/' + os.path.basename(p).lower() for k in kw)]
w('keyword candidates: %d' % len(cand))
for p in cand[:60]:
    w('  ' + os.path.relpath(p, ROOT))

w('')
w('== D. deepseek candidate verdicts + flattened values (first 24 files)')
for p in cand[:24]:
    try:
        r = json.load(open(p, encoding='utf-8'))
    except Exception as e:
        w('%s -> LOAD_ERR %s' % (os.path.relpath(p, ROOT), e))
        continue
    w('--- ' + os.path.relpath(p, ROOT))
    flat = flatten(r)
    items = sorted(flat.items())
    for k, v in items[:60]:
        w('  %s = %s' % (k, v))
    if len(items) > 60:
        w('  ...(%d more keys)' % (len(items) - 60))

# ---- E. atlas infra ----
w('')
w('== E. atlas infra files')
infra = []
for base in ('research', 'tests'):
    b = os.path.join(ROOT, base)
    for dp, dns, fns in os.walk(b):
        dns[:] = [d for d in dns if d not in ('.git', 'node_modules', '__pycache__', '.venv')]
        for fn in fns:
            if fn == 'atlas_ledger.json' or fn.startswith('metric_dict') or fn.startswith('phase_queue'):
                infra.append(os.path.join(dp, fn))
for p in sorted(set(infra)):
    w(os.path.relpath(p, ROOT))

for p in sorted(set(infra)):
    w('')
    w('--- CONTENT ' + os.path.relpath(p, ROOT))
    try:
        r = json.load(open(p, encoding='utf-8'))
        if isinstance(r, dict) and 'entries' in r:
            ents = r['entries']
            w('n=%s chain=%s' % (r.get('n'), r.get('chain')))
            for e in ents[-6:]:
                w('  ' + json.dumps(e, ensure_ascii=False)[:400])
        elif isinstance(r, dict) and isinstance(r.get('metrics'), dict):
            for k, v in r['metrics'].items():
                w('  metric %s: %s' % (k, json.dumps(v, ensure_ascii=False)[:300]))
            w('  (other keys: %s)' % [k for k in r.keys() if k != 'metrics'])
        elif isinstance(r, dict):
            for k, v in list(r.items())[:40]:
                w('  %s: %s' % (k, json.dumps(v, ensure_ascii=False)[:260]))
        else:
            w('  ' + json.dumps(r, ensure_ascii=False)[:800])
    except Exception as e:
        w('  LOAD_ERR %s' % e)

open(OUTP, 'w', encoding='utf-8').write('\n'.join(L))
print('OK lines=%d' % len(L))
