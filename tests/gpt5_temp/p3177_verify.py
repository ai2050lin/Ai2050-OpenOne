# -*- coding: utf-8 -*-
# Phase 3177 independent verify: seal byte-level reconstruction + full
# independent re-implementation of the relation-LOO arm (from sealed 3157 npz,
# independently written indexing/loops, not by importing the main script) +
# device anchors re-derivation vs 3157 sealed results + registry v1.4
# invariants vs v1.3 + html flatten independent rebuild + five-write readback.
import hashlib
import io
import json
import os
import re
import html as htmlmod

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
S = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
PDIR = os.path.join(S, 'phase3177', 'g5b2_ftr14_heldout')
SRC57 = os.path.join(S, r'phase3157\g2p2_transform_algebra_commutator')
REG13 = os.path.join(S, r'phase3176\g5b1_elev_upgrade\atlas_registry_v1_3.json')
GV15 = os.path.join(S, r'phase3175\g5a12_residual_arms\gap_ledger_v1_5.json')
NODES = os.path.join(S, r'phase3162\g5a1_atlas_foundation\atlas_registry.json')
HTML = os.path.join(PDIR, 'atlas_v1_5.html')
OUT = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3177_verify_out.txt')
LINES = []


def chk(name, ok, note=''):
    LINES.append(('%s %s %s' % ('PASS' if ok else 'FAIL', name, note)).rstrip())


def sha8(b):
    return hashlib.sha256(b).hexdigest()[:8]


def sha8_file(p):
    return sha8(open(p, 'rb').read())


R = json.load(io.open(os.path.join(PDIR, 'result.json'), encoding='utf-8'))

# ---------- 1. seal reconstruction ----------
core = {k: v for k, v in R.items() if k not in ('res_sha8', 'seal_sha8')}
r1 = json.dumps(core, ensure_ascii=False, indent=1, sort_keys=True)
res8 = sha8(r1.encode('utf-8'))
mid = json.dumps(dict(core, res_sha8=res8), ensure_ascii=False, indent=1, sort_keys=True)
seal8 = sha8(mid.encode('utf-8'))
chk('1.1 res_sha8 reconstruction', res8 == R['res_sha8'], res8)
chk('1.2 seal_sha8 reconstruction', seal8 == R['seal_sha8'], seal8)
chk('1.3 verdict', R['verdict'].startswith(
    'g5b2_ftr14_heldout|gate_fail|a1_6of6|a2_broken|exch_band_keep|'), R['verdict'][:70])

# ---------- 2. independent arm re-implementation ----------
# npz layout: H(128, NL+1, D) fp16; ei/rel/pol/ctx int arrays; row order of the
# 3157 collect loop is (ei, rel, pol, ctx) nested.  Here the index map is
# rebuilt by scanning the arrays (independent of the main script's IDX dict).
MS = ('qwen3-4b', 'qwen3-14b', 'glm4-9b')
NPZ = {'qwen3-4b': os.path.join(SRC57, 'qwen3-4b', 'collect.npz'),
       'qwen3-14b': os.path.join(SRC57, 'qwen3-14b', 'collect.npz'),
       'glm4-9b': os.path.join(SRC57, 'glm4', 'collect.npz')}
TC_ANCH = {'qwen3-4b': 0.5629908442497253, 'qwen3-14b': 0.5614360570907593,
           'glm4-9b': 0.5640953183174133}
EX_ANCH = {'qwen3-4b': (1.0320992213108435, 1.1928509676419494),
           'qwen3-14b': (1.1211844589001665, 1.294480635663979),
           'glm4-9b': (1.0723271372900645, 1.3311207542158712)}
arm_ok = True
for m in MS:
    z = np.load(NPZ[m])
    H = z['H'].astype(np.float32)
    ei, rl, pl, cx = (z['ei'].astype(int), z['rel'].astype(int),
                      z['pol'].astype(int), z['ctx'].astype(int))
    NH = H.shape[1]
    K = NH - 2
    # independent cell lookup: linear scan
    cell = {}
    for i in range(H.shape[0]):
        cell[(int(ei[i]), int(rl[i]), int(pl[i]), int(cx[i]))] = i
    assert len(cell) == 128
    # dC stack (independent build: explicit double loop over relations/entities)
    dC = np.zeros((64, H.shape[2]), np.float32)
    pos = 0
    for relation in range(2):
        for e in range(16):
            for p in range(2):
                dC[pos] = (H[cell[(e, relation, p, 1)], K] -
                           H[cell[(e, relation, p, 0)], K])
                pos += 1
    dCn = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-18)
    cosM = dCn @ dCn.T
    iu = np.triu_indices(64, 1)
    tc_full = float(cosM[iu].mean())
    # rows: dC filled with relation as OUTER loop -> pos = relation*32 + e*2 + p
    rows_rel = {0: [0 * 32 + e * 2 + p for e in range(16) for p in range(2)],
                1: [1 * 32 + e * 2 + p for e in range(16) for p in range(2)]}
    v_full = dCn.mean(0)
    v_full = v_full / (float(np.linalg.norm(v_full)) + 1e-18)
    cos_full = float((dCn @ v_full).mean())
    folds = {}
    for held in (0, 1):
        tr = rows_rel[1 - held]
        he = rows_rel[held]
        v_sh = dCn[tr].mean(0)
        v_sh = v_sh / (float(np.linalg.norm(v_sh)) + 1e-18)
        cos_held = float((dCn[he] @ v_sh).mean())
        cos_train = float((dCn[tr] @ v_sh).mean())
        Mh = dCn[he] @ dCn[he].T
        tc_held = float(Mh[np.triu_indices(32, 1)].mean())
        folds[held] = (tc_held, cos_held, cos_train,
                       abs(tc_held - tc_full), abs(cos_held - cos_full))
    # exch (independent vectorized build at C=0)
    isa_p = H[[cell[(e, 0, 0, 0)] for e in range(16)], K]
    has_p = H[[cell[(e, 1, 0, 0)] for e in range(16)], K]
    isa_m = H[[cell[(e, 0, 1, 0)] for e in range(16)], K]
    has_m = H[[cell[(e, 1, 1, 0)] for e in range(16)], K]
    dR_p, dR_m = isa_p - has_p, isa_m - has_m
    dN_i, dN_h = isa_m - isa_p, has_m - has_p
    exR = float(np.linalg.norm((dR_p - dR_m).ravel())) / \
        (0.5 * float(np.linalg.norm(dR_p.ravel()) + np.linalg.norm(dR_m.ravel())) + 1e-18)
    exN = float(np.linalg.norm((dN_i - dN_h).ravel())) / \
        (0.5 * float(np.linalg.norm(dN_i.ravel()) + np.linalg.norm(dN_h.ravel())) + 1e-18)
    exp = R['arm']['per_model'][m]
    # NOTE: tc quantities derive from a (64,D)x(D,64) float32 sgemm; across
    # differently-constructed (value-identical) arrays BLAS may pick a
    # different kernel path -> up to 1 ULP (~6e-8 at these magnitudes)
    # wobble.  Reduction-only quantities (exch) are bitwise stable (<1e-9).
    # The main run reproduced the 3157 anchors bitwise (drift 0.00e+00) via
    # the same list-append construction as 3157; this verify uses an
    # independent zeros+fill construction, hence 1e-6 tolerance on tc.
    ok = (abs(tc_full - TC_ANCH[m]) < 1e-6 and
          abs(tc_full - exp['tc_full']) < 1e-6 and
          abs(exR - EX_ANCH[m][0]) < 1e-9 and abs(exN - EX_ANCH[m][1]) < 1e-9 and
          abs(max(exR, exN) - exp['exchange_obs']) < 1e-9 and
          abs(cos_full - exp['cos_full_a2']) < 1e-6)
    for held, fname in ((0, 'isa'), (1, 'hasa')):
        th, ch, ctr, d1, d2 = folds[held]
        e = exp['folds'][fname]
        ok &= (abs(th - e['tc_held_a1']) < 1e-6 and abs(ch - e['cos_held_a2']) < 1e-6 and
               abs(ctr - e['cos_train_a2']) < 1e-6 and abs(d1 - e['a1_diff']) < 1e-6 and
               abs(d2 - e['a2_diff']) < 1e-6)
    arm_ok &= ok
    chk('2.x arm %s' % m, ok,
        'tc=%.6f exR=%.4f exN=%.4f cos_full=%.4f a2isa=%.4f' % (
            tc_full, exR, exN, cos_full, folds[0][4]))
chk('2.y independent arm re-derivation', arm_ok, '')
exm = float(np.mean([R['arm']['per_model'][m]['exchange_obs'] for m in MS]))
chk('2.z exch mean + gate', abs(exm - R['arm']['exch_mean']) < 1e-9 and
    abs(exm - 1.2728174525072664) < 1e-9 and 1.0 <= exm <= 1.3, '%.6f' % exm)

# ---------- 3. gate logic re-derivation ----------
pm = R['arm']['per_model']
a1_all = all(pm[m]['folds'][f]['a1_diff'] < 0.1 for m in MS for f in ('isa', 'hasa'))
a2_pat = (all(pm['glm4-9b']['folds'][f]['a2_diff'] >= 0.1 for f in ('isa', 'hasa')) and
          all(pm[m]['folds'][f]['a2_diff'] < 0.1 for m in MS[:2] for f in ('isa', 'hasa')))
chk('3.1 gate A1 6/6 re-derive', a1_all == R['arm']['gate_a1'] is True, '')
chk('3.2 gate A2 glm4-only failure', a2_pat and R['arm']['gate_a2'] is False, '')
chk('3.3 tc_held > tc_full all cells',
    all(pm[m]['folds'][f]['tc_held_a1'] > pm[m]['tc_full'] for m in MS for f in ('isa', 'hasa')), '')
chk('3.4 fold-to-fold diffs <= 0.022 (per-model 0.0021/0.0219/0.0007)',
    all(abs(pm[m]['folds']['isa']['cos_held_a2'] - pm[m]['folds']['hasa']['cos_held_a2']) <= 0.022
        for m in MS) and
    abs(abs(pm['qwen3-14b']['folds']['isa']['cos_held_a2'] -
            pm['qwen3-14b']['folds']['hasa']['cos_held_a2']) - 0.0219) < 5e-4, '')

# ---------- 4. registry v1.4 invariants vs v1.3 ----------
reg13 = json.load(io.open(REG13, encoding='utf-8'))
reg14 = json.load(io.open(os.path.join(PDIR, 'atlas_registry_v1_4.json'), encoding='utf-8'))
chk('4.1 version/supersedes', reg14['version'] == '1.4' and
    reg14['supersedes'].endswith('(e64733f7, Phase 3176)'), reg14['version'])
m13 = {f['id']: f for f in reg13['features']}
m14 = {f['id']: f for f in reg14['features']}
chk('4.2 22 features', len(reg14['features']) == 22, '')
unch = sum(1 for fid in m13 if fid != 'FTR-14' and
           json.dumps(m13[fid], ensure_ascii=False, sort_keys=True) ==
           json.dumps(m14[fid], ensure_ascii=False, sort_keys=True))
chk('4.3 21 features byte-for-byte', unch == 21, str(unch))
f14o, f14n = m13['FTR-14'], m14['FTR-14']
chk('4.4 FTR-14 counter append only', f14n['evidence_level'] == 'E1_repeatable' and
    f14n['counter_evidence'][:len(f14o['counter_evidence'])] == f14o['counter_evidence'] and
    len(f14n['counter_evidence']) == len(f14o['counter_evidence']) + 1 and
    json.dumps({k: v for k, v in f14n.items() if k != 'counter_evidence'},
               ensure_ascii=False, sort_keys=True) ==
    json.dumps({k: v for k, v in f14o.items() if k != 'counter_evidence'},
               ensure_ascii=False, sort_keys=True), '')
ce_new = f14n['counter_evidence'][-1]
chk('4.5 counter text names gate', '3177 arm FAILED gate' in ce_new and
    'A2' in ce_new and '0.1114' in ce_new, ce_new[:60])
chk('4.6 failures 16 + F16 last', len(reg14['failures']) == 16 and
    reg14['failures'][-1]['id'] == 'F16' and reg14['failures'][-1]['phase'] == 3177, '')
chk('4.7 failures first 15 byte-for-byte', all(
    json.dumps(reg13['failures'][i], ensure_ascii=False, sort_keys=True) ==
    json.dumps(reg14['failures'][i], ensure_ascii=False, sort_keys=True) for i in range(15)), '')
chk('4.8 upgrade_log 5 byte-for-byte', len(reg14['upgrade_log']) == 5 and all(
    json.dumps(reg13['upgrade_log'][i], ensure_ascii=False, sort_keys=True) ==
    json.dumps(reg14['upgrade_log'][i], ensure_ascii=False, sort_keys=True)
    for i in range(5)), '')
chk('4.9 lineage per 3176 precedent', reg14['created'] == reg13['created'] and
    reg14['phase'] == reg13['phase'] and reg14['provenance'] == reg13['provenance'], '')
chk('4.10 file sha anchors', sha8_file(os.path.join(PDIR, 'atlas_registry_v1_4.json')) ==
    R['registry_v14_sha8'] == 'fb11d633' and sha8_file(HTML) == R['html_sha8'] == '82c087de', '')
chk('4.11 smoke registry byte-identity', sha8_file(
    os.path.join(PDIR, 'smoke_atlas_registry_v1_4.json')) == R['registry_v14_sha8'], '')

# ---------- 5. html flatten independent rebuild ----------
gl15 = json.load(io.open(GV15, encoding='utf-8'))
chk('5.0 gap v1.5 anchored', sha8_file(GV15) == '4fb3f37d' and gl15['version'] == '1.5', '')


def fmt(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, float):
        return '%.6g' % v
    return str(v)


nodes = json.load(io.open(NODES, encoding='utf-8'))['audit']['nodes']
d = {}
for f in reg14['features']:
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
for i, u in enumerate(reg14['upgrade_log']):
    d['UPG.%d.node' % i] = u['node']
    d['UPG.%d.change' % i] = u['change']
    d['UPG.%d.reason' % i] = u['reason']
for fl in reg14['failures']:
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
d['meta.registry_v14_sha8'] = R['registry_v14_sha8']
d['meta.registry_v13_sha8'] = 'e64733f7'
d['meta.registry_v12_sha8'] = '0e5abcaf'
d['meta.gap_v15_sha8'] = '4fb3f37d'
d['meta.p3157_4b'] = '95a25965'
d['meta.p3157_14b'] = '9552086d'
d['meta.p3157_glm4'] = '23dd74eb'
d['meta.created'] = None  # runtime timestamp: whitelisted below
htxt = io.open(HTML, encoding='utf-8').read()
pairs = re.findall(r'data-k="([^"]+)"[^>]*>(.*?)</span>', htxt, re.S)
found = {}
dups = []
for k, v in pairs:
    if k in found:
        dups.append(k)
    found[k] = htmlmod.unescape(v)
missing = sorted(set(d) - set(found))
extra = sorted(set(found) - set(d))
mism = [k for k in sorted(set(d) & set(found)) if found[k] != d[k]]
mism = [k for k in mism if k != 'meta.created']
import datetime as _dt
ok_created = False
if 'meta.created' in found:
    try:
        _dt.datetime.strptime(found['meta.created'], '%Y-%m-%d %H:%M')
        ok_created = True
    except ValueError:
        ok_created = False
missing = [k for k in missing if k != 'meta.created']
extra = [k for k in extra if k != 'meta.created']
if not ok_created:
    mism.append('meta.created (whitelist fail)')
d.pop('meta.created', None)
chk('5.1 html fields', not missing and not extra and not mism and not dups,
    'exp=%d(+meta.created) found=%d miss=%d extra=%d mism=%d dup=%d' % (
        len(d) + 1, len(found), len(missing), len(extra), len(mism), len(dups)))
chk('5.2 no external', '<link' not in htxt and '<script' not in htxt and
    'http://' not in htxt and 'https://' not in htxt, '')
chk('5.3 FTR-14 counter rendered', 'FTR-14.counter_evidence' in found and
    '3177 arm FAILED gate' in found['FTR-14.counter_evidence'], '')
chk('5.4 GAP-4 mechanism_note rendered', 'GAP-4.mechanism_note' in found and
    found['GAP-4.mechanism_note'] == gl15['gaps'][3]['mechanism_note']
    if gl15['gaps'][3]['id'] == 'GAP-4' else False, '')
sm_html = io.open(os.path.join(PDIR, 'smoke_atlas_v1_5.html'), encoding='utf-8').read()
sm_n = len(re.findall(r'data-k="', sm_html))
chk('5.5 smoke html subset', sm_n < len(pairs) and 'FTR-14.counter_evidence' in sm_html,
    'smoke %d < full %d' % (sm_n, len(pairs)))

# ---------- 6. five-write readback ----------
LEDGER = os.path.join(ROOT, 'research', 'gpt5', 'atlas', 'atlas_ledger.json')
MEMO = os.path.join(ROOT, 'research', 'gpt5', 'docs', 'AGI_GPT5_MEMO.md')
DAILY = os.path.join(ROOT, '.workbuddy', 'memory', '2026-10-09.md')
WMEM = os.path.join(ROOT, '.workbuddy', 'memory', 'MEMORY.md')
led = json.load(io.open(LEDGER, encoding='utf-8'))
e = [m for m in led['measurements'] if m.get('phase') == 3177]
chk('6.1 ledger 3177 single entry', len(e) == 1 and len(led['measurements']) == 329,
    'n=%d' % len(led['measurements']))
chk('6.2 ledger verdict + prereg id', e and 'gate_fail' in e[0]['verdict'] and
    e[0]['prereg_id'] == 'G5-B2' and e[0]['evidence_level'] == 'E1_repeatable', '')
memo = open(MEMO, 'rb').read().decode('utf-8')
chk('6.3 MEMO 3177 + shas + prereg', '## Phase 3177' in memo and R['res_sha8'] in memo and
    R['seal_sha8'] in memo and 'fb11d633' in memo and '预注册 3178' in memo, '')
daily = io.open(DAILY, encoding='utf-8').read() if os.path.exists(DAILY) else ''
chk('6.4 daily 3177', '3177 FTR-14 关系留出臂' in daily and 'd451030f' in daily and
    '7b746eb2' in daily and 'ledger n→329' in daily, '')
wm = open(WMEM, 'rb').read().decode('utf-8')
chk('6.5 workspace MEMORY 3177', '3177 FTR-14 关系留出臂闭环' in wm and 'n=328→**329**' in wm, '')
sm = json.load(io.open(os.path.join(PDIR, 'smoke_result.json'), encoding='utf-8'))
chk('6.6 smoke vs full', sm['res_sha8'] != R['res_sha8'] and
    sm['registry_v14_sha8'] == R['registry_v14_sha8'], 'arm identical, render subset differs')

n_ok = sum(1 for c in LINES if c.startswith('PASS'))
n_fail = sum(1 for c in LINES if c.startswith('FAIL'))
LINES.append('TOTAL PASS=%d FAIL=%d' % (n_ok, n_fail))
with io.open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(LINES) + '\n')
print('\n'.join(LINES))
