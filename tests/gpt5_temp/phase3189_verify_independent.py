# -*- coding: utf-8 -*-
"""Phase 3189 independent verification (15 checks, disk read-back).
Verify-only: fixes go to verify script, never to sealed artifacts."""
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3179_g5b4_ftr23_registration as p79

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC_DIR = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
P89 = os.path.join(SRC_DIR, 'phase3189', 'g5c2_residual_structure')
P88 = os.path.join(SRC_DIR, 'phase3188', 'g5c1_gap4_dv_attribution')
OK = []
BAD = []


def chk(i, name, cond, detail=''):
    (OK if cond else BAD).append('#%d %s %s' % (i, name, detail))
    print('%s #%d %s %s' % ('PASS' if cond else 'FAIL', i, name, detail))


def s8(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


fin = json.load(open(os.path.join(P89, 'final_result.json'), encoding='utf-8'))
res = json.load(open(os.path.join(P89, 'result.json'), encoding='utf-8'))
sm = json.load(open(os.path.join(P89, 'smoke_result.json'), encoding='utf-8'))
reg = json.load(open(os.path.join(P89, 'atlas_registry_v1_16.json'), encoding='utf-8'))
html = open(os.path.join(P89, 'atlas_v1_17.html'), encoding='utf-8').read()
fin88 = json.load(open(os.path.join(P88, 'final_result.json'), encoding='utf-8'))

# 1 final chain (arm anchors replacement)
base_arm = dict({k: v for k, v in fin.items() if k not in ('res_sha8', 'seal_sha8')},
                res_sha8=res['res_sha8'], seal_sha8=res['seal_sha8'])
raw = json.dumps(base_arm, ensure_ascii=False, indent=1, sort_keys=True)
f_res8 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
mid = json.dumps(dict(base_arm, res_sha8=f_res8), ensure_ascii=False, indent=1, sort_keys=True)
f_seal8 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
chk(1, 'final res/seal chain', f_res8 == fin['res_sha8'] and f_seal8 == fin['seal_sha8'],
    '%s/%s' % (f_res8, f_seal8))

# 2 arm result chain
base = {k: v for k, v in res.items() if k not in ('res_sha8', 'seal_sha8')}
r8 = hashlib.sha256(json.dumps(base, ensure_ascii=False, indent=1,
                               sort_keys=True).encode('utf-8')).hexdigest()[:8]
s8v = hashlib.sha256(json.dumps(dict(base, res_sha8=r8), ensure_ascii=False, indent=1,
                                sort_keys=True).encode('utf-8')).hexdigest()[:8]
chk(2, 'arm result res/seal chain', r8 == res['res_sha8'] and s8v == res['seal_sha8'],
    '%s/%s' % (r8, s8v))

# 3 SMOKE/FULL identical numbers
same = all(json.dumps(sm['per_model'][m], sort_keys=True) ==
           json.dumps(res['per_model'][m], sort_keys=True)
           for m in ('qwen3-4b', 'qwen3-14b', 'glm4-9b'))
chk(3, 'SMOKE/FULL per_model identical', same)

# 4 design sha
exe = json.load(open(os.path.join(P89, 'execution.json'), encoding='utf-8'))
dsha = hashlib.sha256(json.dumps(exe['design'], ensure_ascii=False, sort_keys=True,
                                 indent=1).encode('utf-8')).hexdigest()[:8]
chk(4, 'design sha8', dsha == exe['design_sha8'] == '10339511', dsha)

# 5 sources sha re-assert
txt = open(os.path.join(ROOT, 'tests', 'glm5', 'phase3189_g5c2_residual_structure.py'),
           encoding='utf-8').read()
blk = txt.split('SHA_ANCHOR = {')[1].split('}')[0]
pairs = re.findall(r"'([^']+)': '([0-9a-f]{8})'", blk)
missing = []
for k, v in pairs:
    m = re.search(r"'%s': os\.path\.join\(SRC_DIR, r'([^']+)'\)" % re.escape(k), txt)
    if not m:
        missing.append(k)
        continue
    p = os.path.join(SRC_DIR, m.group(1))
    if not os.path.exists(p) or s8(p) != v:
        missing.append(k)
chk(5, 'source sha re-assert (%d files)' % len(pairs), len(missing) == 0, str(missing))

# 6 registry counts
reg15 = json.load(open(os.path.join(P88, 'atlas_registry_v1_15.json'), encoding='utf-8'))
chk(6, 'registry version/counts', reg['version'] == '1.16' and
    len(reg['features']) == 25 and len(reg['failures']) == 19 and
    len(reg['upgrade_log']) == 6)

# 7 features byte-preserved + FTR-25 appended
pres = all(json.dumps(f, ensure_ascii=False, sort_keys=True) ==
           json.dumps([x for x in reg['features'] if x['id'] == f['id']][0],
                      ensure_ascii=False, sort_keys=True)
           for f in reg15['features'])
chk(7, '24 legacy features byte-preserved + FTR-25 append', pres and
    len([f for f in reg['features'] if f['id'] == 'FTR-25']) == 1)

# 8 other registry fields unchanged
others = all(json.dumps(reg15.get(k), ensure_ascii=False, sort_keys=True) ==
             json.dumps(reg.get(k), ensure_ascii=False, sort_keys=True)
             for k in ('upgrade_log', 'evidence_level_taxonomy', 'principles', 'schema',
                       'models_mainline', 'models_pending', 'provenance', 'created',
                       'phase', 'name', 'design_sha8', 'failures'))
chk(8, 'registry other fields unchanged', others)

# 9 registry sha matches final
chk(9, 'registry_v116 sha8', s8(os.path.join(P89, 'atlas_registry_v1_16.json')) ==
    fin['registry_v116_sha8'] == '114aa114', fin['registry_v116_sha8'])

# 10 html field check re-run
nodes = json.load(open(os.path.join(
    SRC_DIR, r'phase3162\g5a1_atlas_foundation\atlas_registry.json'),
    encoding='utf-8'))['audit']['nodes']
gl = json.load(open(os.path.join(
    SRC_DIR, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json'), encoding='utf-8'))
mc = re.search(r'data-k="meta\.created">([^<]+)<', html)
import importlib.util
spec = importlib.util.spec_from_file_location(
    'p89', os.path.join(ROOT, 'tests', 'glm5', 'phase3189_g5c2_residual_structure.py'))
p89 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p89)
exp = p89.flatten_v117(reg, nodes, gl, {'reg16': fin['registry_v116_sha8']})
exp['meta.created'] = mc.group(1)
v = p79.verify_html(html, exp)
chk(10, 'html fields', not v['missing'] and not v['extra'] and not v['mismatch']
    and not v['dups'], '%d/%d %s' % (v['n_found'], v['n_expected'],
                                     (len(v['missing']), len(v['extra']),
                                      len(v['mismatch']), len(v['dups']))))
chk(11, 'html sha8', s8(os.path.join(P89, 'atlas_v1_17.html')) ==
    fin['html_sha8'] == '4c779d24', fin['html_sha8'])

# 12 ledger #341
lg = json.load(open(os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json'),
                    encoding='utf-8'))
ms = lg['measurements']
chk(12, 'ledger n=341 last=3189', len(ms) == 341 and ms[-1]['phase'] == 3189 and
    ms[-1]['chain']['res_sha8'] == '66a8a8c2')

# 13 verdict SHARED facts: gate 3/3; within/cross honest reading is 2/3
# (4b 0.457<0.544, 14b 0.424<0.566; glm4 0.499>0.445) - correction appended to MEMO
pm = res['per_model']
n_pass = sum(1 for m in pm if pm[m]['gate_model'])
wm = {m: float(sum(pm[m]['within_class']) / 4) for m in pm}
rel = {m: wm[m] < pm[m]['cross_med'] for m in pm}
chk(13, 'verdict SHARED facts', n_pass == 3 and
    'shared_oov_collapse_direction' in res['verdict'] and
    fin['arm']['gate_shared'] is True and rel['qwen3-4b'] and rel['qwen3-14b']
    and not rel['glm4-9b'],
    'within_mean vs cross_med: %s' % {m: (round(wm[m], 4), round(pm[m]['cross_med'], 4))
                                      for m in pm})

# 14 device: drifts zero + 3188 scalar anchors bitwise + FTR-25 anchor numbers
d_ok = all(pm[m]['drift_3169'] == 0.0 and pm[m]['drift_3188'] == 0.0 for m in pm)
a25 = [f for f in reg['features'] if f['id'] == 'FTR-25'][0]['anchors'][0]['asserts']
num_ok = all(abs(a25['top1.%s' % m] - pm[m]['top1']) < 1e-15 for m in pm)
chk(14, 'device drifts zero + FTR-25 anchor numbers', d_ok and num_ok)

# 15 MEMO section + triple verdict
memo = open(os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md'),
            encoding='utf-8').read()
chk(15, 'MEMO 3189 section + triple verdict', '## Phase 3189 — G5-C2' in memo and
    memo.count('verdict（重复') >= 3 and '3190 预注册' in memo)

print('\n%d PASS / %d FAIL' % (len(OK), len(BAD)))
if BAD:
    sys.exit(1)
