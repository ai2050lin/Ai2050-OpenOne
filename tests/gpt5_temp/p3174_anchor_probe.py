# -*- coding: utf-8 -*-
# Probe: resolve every feature anchor to its source doc and navigate all asserts.
import os
import json
import io

ROOT = r'D:\AI2050\Ai2050-OpenOne'
S = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')

SRC_ROOTS = {
    'p3151': [os.path.join(S, 'phase3151')],
    'p3154': [os.path.join(S, 'phase3154')],
    'p3155': [os.path.join(S, 'phase3155')],
    'p3156': [os.path.join(S, 'phase3156')],
    'p3157': [os.path.join(S, 'phase3157')],
    'p3158': [os.path.join(S, 'phase3158')],
    'p3159': [os.path.join(S, 'phase3159')],
    'p3160': [os.path.join(S, 'phase3160')],
    'p3161': [os.path.join(S, 'phase3161')],
    'p3162': [os.path.join(S, 'phase3162')],
    'p3163': [os.path.join(S, 'phase3163')],
    'p3164a': [os.path.join(S, 'phase3164')],
    'p3164b': [os.path.join(S, 'phase3164')],
    'p3164c': [os.path.join(S, 'phase3164')],
    'p3165': [os.path.join(S, 'phase3165')],
    'p3166': [os.path.join(S, 'phase3166')],
    'p3169': [os.path.join(S, 'phase3169')],
    'p3171': [os.path.join(S, 'phase3171')],
    'p3172': [os.path.join(S, 'phase3172')],
    'q03': [os.path.join(ROOT, 'tests', 'deepseek', 'result')],
    'q05': [os.path.join(ROOT, 'tests', 'deepseek', 'result')],
    'q06': [os.path.join(ROOT, 'tests', 'deepseek', 'result')],
    'agate': [os.path.join(ROOT, 'research', 'deepseek', 'atlas')],
}

cand_cache = {}


def candidates(src):
    if src in cand_cache:
        return cand_cache[src]
    docs = []
    for root in SRC_ROOTS[src]:
        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith('.json'):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    if os.path.getsize(p) > 5 * 1024 * 1024:
                        continue
                    d = json.load(io.open(p, encoding='utf-8'))
                except Exception:
                    continue
                if isinstance(d, dict):
                    docs.append((os.path.relpath(p, ROOT), d))
    cand_cache[src] = docs
    return docs


def resolve(src, res8, seal8):
    for path, d in candidates(src):
        r = d.get('res_sha8')
        s = d.get('seal_sha8')
        if res8 and r != res8:
            continue
        if seal8 and s != seal8:
            continue
        if not res8 and not seal8:
            continue
        return path, d
    return None, None


def nav(d, key):
    parts = key.split('.')
    cur = d
    for pt in parts:
        if isinstance(cur, dict) and pt in cur:
            cur = cur[pt]
        elif isinstance(cur, dict) and pt.startswith('node:') and 'nodes' in cur \
                and pt.split(':', 1)[1] in cur['nodes']:
            cur = cur['nodes'][pt.split(':', 1)[1]]
        else:
            return False, None
    return True, cur


def val_eq(actual, expect):
    if isinstance(expect, bool) or isinstance(actual, bool):
        return actual == expect
    if isinstance(expect, float):
        return isinstance(actual, (int, float)) and abs(float(actual) - expect) <= 1e-9 * max(1.0, abs(expect))
    if isinstance(expect, int):
        return actual == expect
    return actual == expect


Rg = json.load(io.open(os.path.join(S, 'phase3173', 'g5a10_atlas_v13',
                                     'atlas_registry_v1_2.json'), encoding='utf-8'))
out = []
n_res = n_nav_fail = 0
for f in Rg['features']:
    for a in f['anchors']:
        src = a.get('src')
        asr = a.get('asserts', {})
        res8 = asr.get('res_sha8')
        seal8 = asr.get('seal_sha8')
        if res8 or seal8:
            path, doc = resolve(src, res8, seal8)
            if path is None:
                out.append('%s %s: UNRESOLVED res=%s seal=%s (cands=%d)'
                           % (f['id'], src, res8, seal8, len(candidates(src))))
                n_res += 1
                continue
            out.append('%s %s: %s' % (f['id'], src, path))
        else:
            # audit / gate-citation anchors: use canonical doc by src
            if src == 'p3162':
                path = os.path.join(S, 'phase3162', 'g5a1_atlas_foundation', 'result_audit.json')
                doc = json.load(io.open(os.path.join(ROOT, path), encoding='utf-8'))
            elif src == 'agate':
                path = 'research\\deepseek\\atlas\\a_gate_closure_v1.json'
                doc = json.load(io.open(os.path.join(ROOT, path), encoding='utf-8'))
            else:
                out.append('%s %s: NO-SHA anchor src unknown' % (f['id'], src))
                continue
            out.append('%s %s: %s (no-sha)' % (f['id'], src, path))
        fails = []
        for k, v in asr.items():
            if k in ('res_sha8', 'seal_sha8'):
                continue
            ok, actual = nav(doc, k)
            if not ok:
                fails.append(k + ' [missing]')
            elif not val_eq(actual, v):
                fails.append('%s [%r != %r]' % (k, actual, v))
        if fails:
            n_nav_fail += 1
            out.append('   NAV-FAIL: ' + ' | '.join(fails[:6]))
out.append('')
out.append('unresolved_or_navfail_anchors=%d' % (n_res + n_nav_fail))
open(os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3174_anchors.txt'), 'w',
     encoding='utf-8').write('\n'.join(out))
print('done')
