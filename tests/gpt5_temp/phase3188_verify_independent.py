# -*- coding: utf-8 -*-
"""Phase 3188 independent verification (15 checks, disk read-back).
Verify-only: fixes go to verify script, never to sealed artifacts."""
import hashlib
import io
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3179_g5b4_ftr23_registration as p79

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
P88 = os.path.join(SRC_DIR, 'phase3188', 'g5c1_gap4_dv_attribution')
OK = []
BAD = []


def chk(i, name, cond, detail=''):
    (OK if cond else BAD).append('#%d %s %s' % (i, name, detail))
    print('%s #%d %s %s' % ('PASS' if cond else 'FAIL', i, name, detail))


def s8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


fin = json.load(open(os.path.join(P88, 'final_result.json'), encoding='utf-8'))
res = json.load(open(os.path.join(P88, 'result.json'), encoding='utf-8'))
reg = json.load(open(os.path.join(P88, 'atlas_registry_v1_15.json'), encoding='utf-8'))
html = open(os.path.join(P88, 'atlas_v1_16.html'), encoding='utf-8').read()

base_arm = dict({k: v for k, v in fin.items() if k not in ('res_sha8', 'seal_sha8')},
                res_sha8=res['res_sha8'], seal_sha8=res['seal_sha8'])
raw = json.dumps(base_arm, ensure_ascii=False, indent=1, sort_keys=True)
f_res8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
mid = json.dumps(dict(base_arm, res_sha8=f_res8), ensure_ascii=False, indent=1, sort_keys=True)
f_seal8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
chk(1, 'final res/seal chain', f_res8 == fin['res_sha8'] and f_seal8 == fin['seal_sha8'],
    '%s/%s' % (f_res8, f_seal8))

# 2 arm result chain (res dict without res/seal)
base = {k: v for k, v in res.items() if k not in ('res_sha8', 'seal_sha8')}
r_raw = json.dumps(base, ensure_ascii=False, indent=1, sort_keys=True)
r8 = hashlib.sha256(r_raw.encode('utf-8')).hexdigest()[:8]
m_raw = json.dumps(dict(base, res_sha8=r8), ensure_ascii=False, indent=1, sort_keys=True)
s8v = hashlib.sha256(m_raw.encode('utf-8')).hexdigest()[:8]
chk(2, 'arm result res/seal chain', r8 == res['res_sha8'] and s8v == res['seal_sha8'],
    '%s/%s' % (r8, s8v))

# 3 SMOKE/FULL arm numbers identical (per_model fields except mode)
sm = json.load(open(os.path.join(P88, 'smoke_result.json'), encoding='utf-8'))
same = all(
    json.dumps(sm['per_model'][m], sort_keys=True) == json.dumps(res['per_model'][m], sort_keys=True)
    for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'))
chk(3, 'SMOKE/FULL per_model identical', same)

# 4 execution.json design sha
exe = json.load(open(os.path.join(P88, 'execution.json'), encoding='utf-8'))
core = exe['design']
dsha = hashlib.sha256(json.dumps(core, ensure_ascii=False, sort_keys=True,
                                 indent=1).encode('utf-8')).hexdigest()[:8]
chk(4, 'design sha8', dsha == exe['design_sha8'] == '977bb327', dsha)

# 5 sources sha re-assert
SRC = json.load(open(os.path.join(P88, 'result.json'), encoding='utf-8'))
missing = []
# rebuild source paths from main script map
sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import importlib.util
spec = importlib.util.spec_from_file_location('p3188m', os.path.join(
    ROOT, 'tests', 'glm5', 'phase3188_g5c1_gap4_dv_attribution.py'))
# do NOT exec module (would run main); read SRC/SHA via regex instead
txt = open(os.path.join(ROOT, 'tests', 'glm5', 'phase3188_g5c1_gap4_dv_attribution.py'),
           encoding='utf-8').read()
blk = txt.split('SHA_ANCHOR = {')[1].split('}')[0]
pairs = re.findall(r"'([^']+)': '([0-9a-f]{8})'", blk)
n_ok = 0
for k, v in pairs:
    m = re.search(r"'%s': os\.path\.join\(SRC_DIR, r'([^']+)'\)" % re.escape(k), txt)
    if not m:
        missing.append(k)
        continue
    p = os.path.join(SRC_DIR, m.group(1))
    if not os.path.exists(p) or s8(p) != v:
        missing.append(k)
    else:
        n_ok += 1
chk(5, 'source sha re-assert (%d files)' % n_ok, len(missing) == 0, str(missing))

# 6 registry prefix preservation vs v1.14
reg14 = json.load(open(os.path.join(
    SRC_DIR, r'phase3187\g5b12_alpha_calibration\atlas_registry_v1_14.json'),
    encoding='utf-8'))
chk(6, 'registry version/counts', reg['version'] == '1.15' and
    len(reg['features']) == 24 and len(reg['failures']) == 19 and
    len(reg['upgrade_log']) == 6)
f24n = [f for f in reg['features'] if f['id'] == 'FTR-24'][0]
f24o = [f for f in reg14['features'] if f['id'] == 'FTR-24'][0]
pres = all(json.dumps(f, ensure_ascii=False, sort_keys=True) ==
           json.dumps([x for x in reg['features'] if x['id'] == f['id']][0],
                      ensure_ascii=False, sort_keys=True)
           for f in reg14['features'] if f['id'] != 'FTR-24')
ce_append = (len(f24n['counter_evidence']) == len(f24o['counter_evidence']) + 1 and
             f24n['counter_evidence'][:-1] == f24o['counter_evidence'] and
             f24n['statement'] == f24o['statement'] and
             f24n['anchors'] == f24o['anchors'])
chk(7, 'features byte-preserved + FTR-24 ce append-only', pres and ce_append)
others = all(json.dumps(reg14.get(k), ensure_ascii=False, sort_keys=True) ==
             json.dumps(reg.get(k), ensure_ascii=False, sort_keys=True)
             for k in ('upgrade_log', 'evidence_level_taxonomy', 'principles', 'schema',
                       'models_mainline', 'models_pending', 'provenance', 'created',
                       'phase', 'name', 'design_sha8', 'failures'))
chk(8, 'registry other fields unchanged', others)

# 9 registry file sha matches final
chk(9, 'registry_v115 sha8', s8(os.path.join(P88, 'atlas_registry_v1_15.json')) ==
    fin['registry_v115_sha8'] == '0a55d2e7', fin['registry_v115_sha8'])

# 10 html field check re-run
nodes = json.load(open(os.path.join(
    SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
    encoding='utf-8'))['audit']['nodes']
gl = json.load(open(os.path.join(
    SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'), encoding='utf-8'))
mc = re.search(r'data-k="meta\.created">([^<]+)<', html)
CREATED = mc.group(1)
txt2 = open(os.path.join(ROOT, 'tests', 'glm5', 'phase3188_g5c1_gap4_dv_attribution.py'),
            encoding='utf-8').read()


def flatten(regx, nodes, gl):
    d = {}
    for f in regx['features']:
        p = f['id']
        d[p + '.statement'] = f['statement']
        d[p + '.family'] = f['family']
        d[p + '.evidence_level'] = f['evidence_level']
        d[p + '.model_scope'] = ', '.join(f['model_scope'])
        d[p + '.n_anchors'] = str(len(f['anchors']))
        d[p + '.counter_evidence'] = ' | '.join(f['counter_evidence']) if f['counter_evidence'] else '(none)'
        d[p + '.replication'] = f['replication']
        d[p + '.scope_limits'] = f['scope_limits'] if f['scope_limits'] else '(none)'
        for i, a in enumerate(f['anchors']):
            d['%s.anchors.%d.src' % (p, i)] = str(a['src'])
            d['%s.anchors.%d.phase' % (p, i)] = str(a['phase'])
            d['%s.anchors.%d.role' % (p, i)] = a['role']
            d['%s.anchors.%d.asserts' % (p, i)] = json.dumps(a['asserts'], ensure_ascii=False, sort_keys=True)
        for k, v in (f.get('values') or {}).items():
            d['%s.values.%s' % (p, k)] = p79.fmt(v) if hasattr(p79, 'fmt') else str(v)
    for n in nodes:
        d[n['id'] + '.title'] = n['title']
        d[n['id'] + '.elevel'] = n['evidence_level']
        d[n['id'] + '.status'] = n['status']
        d[n['id'] + '.checks'] = '%d/%d' % (n['n_pass'], n['n_checks'])
    for i, u in enumerate(regx['upgrade_log']):
        d['UPG.%d.node' % i] = u['node']
        d['UPG.%d.change' % i] = u['change']
        d['UPG.%d.reason' % i] = u['reason']
    for fl in regx['failures']:
        d[fl['id'] + '.kind'] = fl['kind']
        d[fl['id'] + '.phase'] = str(fl['phase'])
        d[fl['id'] + '.text'] = fl['text']
        d[fl['id'] + '.evidence'] = str(fl['evidence'])
    for g in gl['gaps']:
        p = g['id']
        d[p + '.title'] = g['title']
        d[p + '.status'] = g['status']
        if g['status'].startswith('closed'):
            d[p + '.closed_at'] = str(g['closed_at_phase'])
        d[p + '.statement'] = g['statement']
        d[p + '.evidence'] = ' || '.join(g['evidence'])
        d[p + '.anchors'] = ', '.join('%s=%s' % (k, v) for k, v in sorted(g['anchor_sha8'].items()))
        if p == 'GAP-4':
            pr = g['prereg']
            d[p + '.prereg.phase'] = str(pr['phase'])
            d[p + '.prereg.name'] = pr['name']
            d[p + '.prereg.hypothesis'] = pr['hypothesis']
            d[p + '.prereg.protocol'] = pr['protocol']
            d[p + '.prereg.gate'] = pr['gate']
            d[p + '.prereg.status'] = pr['status']
            d[p + '.mechanism_note'] = g['mechanism_note']
    for a in gl['appendix']:
        p = a['id']
        d[p + '.title'] = a['title']
        d[p + '.status'] = a['status']
        d[p + '.note'] = a['note']
        d[p + '.anchor'] = a['anchor']
        d[p + '.anchor_sha8'] = a['anchor_sha8']
    d['meta.registry_v115_sha8'] = fin['registry_v115_sha8']
    d['meta.registry_v114_sha8'] = '95fe55f5'
    d['meta.registry_v113_sha8'] = '730729f5'
    d['meta.registry_v112_sha8'] = '269a37c0'
    d['meta.registry_v111_sha8'] = '84aecc21'
    d['meta.gap_v15_sha8'] = '4fb3f37d'
    d['meta.p3157_4b'] = '95a25965'
    d['meta.p3157_14b'] = '9552086d'
    d['meta.p3157_glm4'] = '23dd74eb'
    d['meta.p3169_result'] = '5b51c2c1'
    d['meta.p3175_result'] = 'e0d2d459'
    d['meta.p3183_result'] = '0ab70579'
    d['meta.created'] = CREATED
    return d


expected = flatten(reg, nodes, gl)
v = p79.verify_html(html, expected)
chk(10, 'html fields', not v['missing'] and not v['extra'] and not v['mismatch']
    and not v['dups'], '%d/%d %s' % (v['n_found'], v['n_expected'],
                                     (len(v['missing']), len(v['extra']), len(v['mismatch']), len(v['dups']))))
chk(11, 'html sha8', s8(os.path.join(P88, 'atlas_v1_16.html')) ==
    fin['html_sha8'] == '384d0834', fin['html_sha8'])

# 12 ledger #340 + prefix
lg = json.load(open(os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json'),
                    encoding='utf-8'))
ms = lg['measurements']
chk(12, 'ledger n=340 last=3188', len(ms) == 340 and ms[-1]['phase'] == 3188 and
    ms[-1]['chain']['res_sha8'] == '8c1d94a3')

# 13 verdict coherence: DECOUPLED branch facts
pm = res['per_model']
n_pass = sum(1 for m in pm if pm[m]['gate_model'])
signs = [1 if pm[m]['med_diff'] > 0 else -1 for m in pm]
chk(13, 'verdict DECOUPLED facts', n_pass == 2 and len(set(signs)) == 2 and
    'decoupled' in res['verdict'] and fin['arm']['gate_coupled'] is False,
    'n_pass=%d signs=%s' % (n_pass, signs))

# 14 device gates all zero drift
d_ok = all(pm[m]['drift_3169'] == 0.0 for m in pm)
app = res['appendix_3175_out_k8']
a_ok = all(app[m]['drift_out_arm'] == 0.0 and app[m]['anchor_ok'] and app[m]['n_rows'] == 780
           for m in app)
chk(14, 'device drifts zero + appendix anchors', d_ok and a_ok)

# 15 MEMO section + verdict triple
memo = open(os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md'),
            encoding='utf-8').read()
chk(15, 'MEMO 3188 section + triple verdict', '## Phase 3188 — G5-C1' in memo and
    'verdict（重复一）' in memo and 'verdict（重复二）' in memo and
    'verdict（重复三）' in memo and '3189 预注册' in memo)

print('\n%d PASS / %d FAIL' % (len(OK), len(BAD)))
if BAD:
    sys.exit(1)
