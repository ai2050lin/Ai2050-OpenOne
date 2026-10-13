# -*- coding: utf-8 -*-
"""Phase 3178 independent disk verification (separate process, independent impl).

Checks:
  A. sources sha8 (6 re-verified)
  B. independent null-device recomputation (vectorized grid masks, not the
     main script's list-append style) vs sealed result (float32 ULP tol 1e-6)
  C. g/r arithmetic identity vs 3177 sealed result + internal consistency
  D. gate logic re-derivation
  E. seal reconstruction (res8/seal8)
  F. registry v1.5 invariants + preservation vs v1.4 + counter numbers vs result
  G. html data-k flatten rebuild vs on-disk bytes (meta.created whitelisted)
  H. five writes read-back (ledger chain / MEMO / daily / MEMORY)
  I. SMOKE/FULL arm-number identity + registry byte identity
"""
import hashlib
import io
import json
import re

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
S = ROOT + r'\tests\glm5\result\rdc_query_construction_20260913'
PDIR = S + r'\phase3178\g5b3_ftr14_a2_attribution'
MS = ('qwen3-4b', 'qwen3-14b', 'glm4-9b')
LOG = []
ok = 0
fail = 0


def chk(name, cond, extra=''):
    global ok, fail
    if cond:
        ok += 1
        LOG.append('PASS %s %s' % (name, extra))
        print('PASS %s %s' % (name, extra), flush=True)
    else:
        fail += 1
        LOG.append('FAIL %s %s' % (name, extra))
        print('FAIL %s %s' % (name, extra), flush=True)


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def sha8_file(p):
    return sha8(open(p, 'rb').read())


R = json.load(io.open(PDIR + r'\result.json', encoding='utf-8'))
R_SMOKE = json.load(io.open(PDIR + r'\smoke_result.json', encoding='utf-8'))

# ---------- A. sources
SHA = {
    'p3157_4b': (S + r'\phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz', '95a25965'),
    'p3157_14b': (S + r'\phase3157\g2p2_transform_algebra_commutator\qwen3-14b\collect.npz', '9552086d'),
    'p3157_glm4': (S + r'\phase3157\g2p2_transform_algebra_commutator\glm4\collect.npz', '23dd74eb'),
    'p3177_result': (S + r'\phase3177\g5b2_ftr14_heldout\result.json', 'b2ad52ef'),
    'registry_v14': (S + r'\phase3177\g5b2_ftr14_heldout\atlas_registry_v1_4.json', 'fb11d633'),
    'gap_v15': (S + r'\phase3175\g5a12_residual_arms\gap_ledger_v1_5.json', '4fb3f37d'),
}
bad = [k for k, (p, h) in SHA.items() if sha8_file(p) != h]
chk('A sources sha8 (6 re-verified)', not bad, str(bad))

# ---------- B. independent null recomputation (vectorized grid masks)
TOL = 1e-6
npz_map = {'qwen3-4b': 'p3157_4b', 'qwen3-14b': 'p3157_14b', 'glm4-9b': 'p3157_glm4'}
for m in MS:
    z = np.load(SHA[npz_map[m]][0])
    H = z['H'].astype(np.float32)
    NL = H.shape[1] - 1
    KOUT = NL - 1
    ei = z['ei'].astype(int)
    rl = z['rel'].astype(int)
    pl = z['pol'].astype(int)
    cx = z['ctx'].astype(int)
    IDX = {(int(ei[i]), int(rl[i]), int(pl[i]), int(cx[i])): i for i in range(H.shape[0])}
    e_g, r_g, p_g = np.meshgrid(np.arange(16), np.arange(2), np.arange(2), indexing='ij')
    ctx1 = np.array([IDX[(int(e), int(r), int(p), 1)] for e, r, p in
                     zip(e_g.ravel(), r_g.ravel(), p_g.ravel())])
    ctx0 = np.array([IDX[(int(e), int(r), int(p), 0)] for e, r, p in
                     zip(e_g.ravel(), r_g.ravel(), p_g.ravel())])
    dC = H[ctx1, KOUT] - H[ctx0, KOUT]
    dCn = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-18)
    for rel in (0, 1):
        rname = 'isa' if rel == 0 else 'hasa'
        mask_rel = (r_g.ravel() == rel)
        rows_all = np.where(mask_rel)[0]
        p_folds = []
        for e in range(16):
            held = np.where(mask_rel & (e_g.ravel() == e))[0]
            train = np.setdiff1d(rows_all, held)
            assert len(train) == 30 and len(held) == 2
            v_null = dCn[train].mean(0)
            v_null = v_null / (float(np.linalg.norm(v_null)) + 1e-18)
            cos_in = float((dCn[train] @ v_null).mean())
            cos_out = float((dCn[held] @ v_null).mean())
            p_folds.append(cos_in - cos_out)
        sealed = R['arm']['per_model'][m]['null'][rname]
        d_pf = max(abs(a - b['p_fold']) for a, b in zip(p_folds, sealed['folds']))
        d_pm = abs(float(np.mean(p_folds)) - sealed['p_mean'])
        chk('B null %s %s (16 folds, ULP<=%.1e)' % (m, rname, TOL),
            d_pf < TOL and d_pm < TOL, 'p_mean %.6f dp_fold=%.2e' % (sealed['p_mean'], d_pf))

# ---------- C. g/r arithmetic identity
R17 = json.load(io.open(SHA['p3177_result'][0], encoding='utf-8'))
for m in MS:
    pm = R['arm']['per_model'][m]
    e17 = R17['arm']['per_model'][m]
    d_full = abs(pm['cos_full_a2'] - e17['cos_full_a2'])
    d_tc = abs(pm['tc_full'] - e17['tc_full'])
    d_ex = abs(pm['exchange_obs'] - e17['exchange_obs'])
    worst = max(d_full, d_tc, d_ex)
    for rel in ('isa', 'hasa'):
        f8 = pm['folds'][rel]
        f17 = e17['folds'][rel]
        g_recomputed = f8['cos_train_a2'] - f8['cos_held_a2']
        d_g_seal = abs(g_recomputed - f17['a2_gap_train'])
        d_gap_field = abs(g_recomputed - pm['resid'][rel]['gap'])
        r_field = pm['resid'][rel]['gap'] - pm['resid'][rel]['p']
        d_r = abs(r_field - pm['resid'][rel]['r'])
        worst = max(worst, d_g_seal, d_gap_field, d_r)
    r_mean = (pm['resid']['isa']['r'] + pm['resid']['hasa']['r']) / 2.0
    d_rm = abs(r_mean - pm['r_m'])
    chk('C g/r identity %s (3177 bitwise + internal)' % m,
        worst < 1e-9 and d_rm < 1e-12, 'worst=%.2e d_rm=%.2e' % (worst, d_rm))

# ---------- D. gate logic re-derivation
r_vals = [R['arm']['per_model'][m]['r_m'] for m in MS]
chk('D gate re-derivation (calib FAIL x3)',
    R['gates']['calibration_bar'] is False
    and R['arm']['calibration_pass'] is False
    and all(v >= 0.05 for v in r_vals)
    and R['gates']['device_3157'] is True and R['gates']['replication_3177'] is True,
    'r=%s' % '/'.join('%.4f' % v for v in r_vals))
chk('D verdict string', R['arm']['verdict'].startswith('relation_specific_residual_confirmed'))

# ---------- E. seal reconstruction
summary = {k: v for k, v in R.items() if k not in ('res_sha8', 'seal_sha8')}
raw = json.dumps(summary, ensure_ascii=False, indent=1, sort_keys=True)
res8 = sha8(raw.encode('utf-8'))
mid = json.dumps(dict(summary, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
seal8 = sha8(mid.encode('utf-8'))
chk('E seal reconstruction', res8 == R['res_sha8'] and seal8 == R['seal_sha8'],
    '%s/%s' % (res8, seal8))

# ---------- F. registry v1.5
reg14 = json.load(io.open(SHA['registry_v14'][0], encoding='utf-8'))
reg15 = json.load(io.open(PDIR + r'\atlas_registry_v1_5.json', encoding='utf-8'))
chk('F1 version/supersedes', reg15['version'] == '1.5'
    and 'atlas_registry_v1_4.json (fb11d633, Phase 3177)' == reg15['supersedes'])
chk('F2 counts', len(reg15['features']) == 22 and len(reg15['failures']) == 16
    and len(reg15['upgrade_log']) == 5)
f14_14 = [f for f in reg14['features'] if f['id'] == 'FTR-14'][0]
f14_15 = [f for f in reg15['features'] if f['id'] == 'FTR-14'][0]
chk('F3 FTR-14 stays E1 + counter 2->3',
    f14_15['evidence_level'] == 'E1_repeatable'
    and len(f14_14['counter_evidence']) == 2 and len(f14_15['counter_evidence']) == 3)
ce = f14_15['counter_evidence'][-1]
num_ok = all(('%.4f' % R['arm']['per_model'][m]['r_m']) in ce for m in MS)
chk('F4 counter quantified numbers match result', '3178' in ce and num_ok)
pres = all(json.dumps(f14, ensure_ascii=False, sort_keys=True) ==
           json.dumps([x for x in reg15['features'] if x['id'] == f14['id']][0],
                      ensure_ascii=False, sort_keys=True)
           for f14 in reg14['features'] if f14['id'] != 'FTR-14')
chk('F5 21 features byte-for-byte', pres)
chk('F6 failures + upgrades unchanged',
    json.dumps(reg14['failures'], ensure_ascii=False, sort_keys=True) ==
    json.dumps(reg15['failures'], ensure_ascii=False, sort_keys=True)
    and json.dumps(reg14['upgrade_log'], ensure_ascii=False, sort_keys=True) ==
    json.dumps(reg15['upgrade_log'], ensure_ascii=False, sort_keys=True))
chk('F7 registry file sha', sha8_file(PDIR + r'\atlas_registry_v1_5.json') == 'e1942d03')

# ---------- G. html flatten rebuild
gl15 = json.load(io.open(SHA['gap_v15'][0], encoding='utf-8'))
reg2 = json.load(io.open(S + r'\phase3162\g5a1_atlas_foundation\atlas_registry.json', encoding='utf-8'))
nodes = reg2['audit']['nodes']


def fmt(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return '%.6g' % v
    return str(v)


d = {}
for f in reg15['features']:
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
        d['%s.values.%s' % (p, k)] = fmt(v)
for n in nodes:
    d[n['id'] + '.title'] = n['title']
    d[n['id'] + '.elevel'] = n['evidence_level']
    d[n['id'] + '.status'] = n['status']
    d[n['id'] + '.checks'] = '%d/%d' % (n['n_pass'], n['n_checks'])
for i, u in enumerate(reg15['upgrade_log']):
    d['UPG.%d.node' % i] = u['node']
    d['UPG.%d.change' % i] = u['change']
    d['UPG.%d.reason' % i] = u['reason']
for fl in reg15['failures']:
    d[fl['id'] + '.kind'] = fl['kind']
    d[fl['id'] + '.phase'] = str(fl['phase'])
    d[fl['id'] + '.text'] = fl['text']
    d[fl['id'] + '.evidence'] = str(fl['evidence'])
for g in gl15['gaps']:
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
for a in gl15['appendix']:
    p = a['id']
    d[p + '.title'] = a['title']
    d[p + '.status'] = a['status']
    d[p + '.note'] = a['note']
    d[p + '.anchor'] = a['anchor']
    d[p + '.anchor_sha8'] = a['anchor_sha8']
d['meta.registry_v15_sha8'] = 'e1942d03'
d['meta.registry_v14_sha8'] = 'fb11d633'
d['meta.registry_v13_sha8'] = 'e64733f7'
d['meta.registry_v12_sha8'] = '0e5abcaf'
d['meta.gap_v15_sha8'] = '4fb3f37d'
d['meta.p3157_4b'] = '95a25965'
d['meta.p3157_14b'] = '9552086d'
d['meta.p3157_glm4'] = '23dd74eb'

import html as html_mod
htxt = open(PDIR + r'\atlas_v1_6.html', 'rb').read().decode('utf-8')
pairs = re.findall(r'data-k="([^"]+)"[^>]*>(.*?)</span>', htxt, re.S)
found = {}
dups = []
for k, v in pairs:
    if k in found:
        dups.append(k)
    found[k] = html_mod.unescape(v)
d['meta.created'] = found.get('meta.created')
missing = sorted(set(d) - set(found))
extra = sorted(set(found) - set(d))
mism = [k for k in sorted(set(d) & set(found)) if found[k] != d[k]]
chk('G1 html key set 669', len(d) == 669 and not missing and not extra, 'exp=%d found=%d' % (len(d), len(found)))
chk('G2 html values (meta.created whitelisted)', not mism and not dups,
    'mism=%d dups=%d' % (len(mism), len(dups)))
chk('G3 meta.created present', 'meta.created' in found)
chk('G4 html file sha', sha8_file(PDIR + r'\atlas_v1_6.html') == 'b95e67a2')

# ---------- H. five writes read-back
LEDGER = ROOT + r'\research\gpt5\atlas\atlas_ledger.json'
led = json.load(io.open(LEDGER, encoding='utf-8'))
led_tmp = json.loads(json.dumps(led, ensure_ascii=False))
led_tmp.pop('ledger_sha256_8', None)
chain = sha8(json.dumps(led_tmp, ensure_ascii=False, indent=1, sort_keys=False).encode('utf-8'))
chk('H1 ledger n=330 + chain', len(led['measurements']) == 330
    and led['measurements'][-1]['phase'] == 3178 and led['ledger_sha256_8'] == chain, chain)
MEMO = ROOT + r'\research\gpt5\docs\AGI_GPT5_MEMO.md'
mraw = open(MEMO, 'rb').read()
mtxt = mraw.decode('utf-8').lstrip('\ufeff')
chk('H2 MEMO 3178 section on disk', mraw.startswith(b'\xef\xbb\xbf')
    and mtxt.count('## Phase 3178: G5-B3 FTR-14 A2 归因分解') == 1
    and mtxt.find('## Phase 3178: G5-B3') > mtxt.find('### 接续：预注册 3178')
    and mtxt.endswith('水果类、K4。\r\n'))
DAILY = ROOT + r'\.workbuddy\memory\2026-10-10.md'
dtxt = io.open(DAILY, encoding='utf-8').read()
chk('H3 daily 3178 entry', '3178 G5-B3' in dtxt and '预注册 3179=G5-B4 FTR-23 立项' in dtxt)
WMEM = ROOT + r'\.workbuddy\memory\MEMORY.md'
wtxt = open(WMEM, 'rb').read().decode('utf-8')
chk('H4 MEMORY 3178 + typo fixed', '✅ 3178 FTR-14 A2 归因分解闭环' in wtxt
    and '字节恒乙' not in wtxt and '字节恒等' in wtxt
    and '下一步 3179=G5-B4 FTR-23 立项入册' in wtxt)

# ---------- I. SMOKE/FULL identity
arm_fields = ('cos_full_a2', 'tc_full', 'exchange_obs', 'r_m')
same = all(R['arm']['per_model'][m][k] == R_SMOKE['arm']['per_model'][m][k]
           for m in MS for k in arm_fields)
same = same and all(
    R['arm']['per_model'][m]['resid'][rel][kk] == R_SMOKE['arm']['per_model'][m]['resid'][rel][kk]
    for m in MS for rel in ('isa', 'hasa') for kk in ('gap', 'p', 'r'))
chk('I1 SMOKE/FULL arm numbers identical', same)
chk('I2 SMOKE verdict identical',
    R_SMOKE['arm']['verdict'] == R['arm']['verdict']
    and R_SMOKE['gates'] == R['gates'])
chk('I3 smoke registry byte identity',
    sha8_file(PDIR + r'\smoke_atlas_registry_v1_5.json') == 'e1942d03')

LOG.append('TOTAL PASS=%d FAIL=%d' % (ok, fail))
print('TOTAL PASS=%d FAIL=%d' % (ok, fail))
with io.open(ROOT + r'\tests\gpt5_temp\p3178_verify_out.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(LOG) + '\n')
print('VERIFY DONE')
