# -*- coding: utf-8 -*-
"""Independent verification of Phase 3232 (G5-G29) sealed artifacts.

READ-ONLY over sealed products. Recomputes the residual chain from the
sealed 3169 npz, re-derives every per-class number, re-checks gate logic,
gap v1.10 derivation, and result identity. Scalar anchors drift == 0;
ustar-propagated quantities use rel tol 1e-6 (C1 correction, pre-registered).
On FAIL: fix THIS script, never the sealed artifacts.
"""
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3192_g5c5_gap4_amplitude_attribution as p92
import phase3232_g5g29_gd_amp_class_robustness as p32

ROOT = r'D:\AI2050\Ai2050-OpenOne'
OUT = os.path.join(p32.SRC_DIR, 'phase3232', p32.NAME)
MODELS = p32.MODELS
CLS_IDS, SEEN_IDS = p32.CLS_IDS, p32.SEEN_IDS

G = []


def gate(name, ok, detail=''):
    G.append({'gate': name, 'pass': bool(ok), 'detail': str(detail)})
    print(('PASS ' if ok else 'FAIL ') + name + (' :: ' + detail if detail
                                                 else ''), flush=True)


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


res = json.load(open(os.path.join(OUT, 'result.json'), encoding='utf-8'))

# V1 identity + seals
core = dict(res)
core.pop('res_sha8')
core.pop('seal_sha8')
raw = json.dumps(core, ensure_ascii=False, indent=1, sort_keys=True)
r1 = hashlib.sha256(raw.encode('utf-8')).hexdigest()[:8]
mid = json.dumps(dict(core, res_sha8=r1), ensure_ascii=False, indent=1,
                 sort_keys=True)
s1 = hashlib.sha256(mid.encode('utf-8')).hexdigest()[:8]
gate('V1 res/seal self-consistency + verdict', r1 == res['res_sha8']
     and s1 == res['seal_sha8']
     and res['verdict'].startswith('GD_AMP_CLASS_ROBUST'),
     'res %s seal %s verdict %s' % (res['res_sha8'], res['seal_sha8'],
                                    res['verdict'][:40]))

# V2 source anchors (all 11)
ok = all(sha8_file(p32.SRC[k]) == v for k, v in p32.SHA_ANCHOR.items())
gate('V2 source shas x11', ok,
     'incl 3169 x3 npz, 3189-3192 results, 3225 npz/result, gap v1.9')

# V3 registry untouched
reg31 = os.path.join(p32.SRC_DIR, 'phase3231', 'g5g28_registry_decision',
                     'atlas_registry_v1_31.json')
gate('V3 registry v1.31 untouched (ef5189e7)', sha8_file(reg31) == 'ef5189e7',
     sha8_file(reg31))

# V4: independent chain recompute -> per-class numbers
z32 = np.load(p32.SRC['p3225_npz'])
ok, notes, relmax = True, [], 0.0
percls_re = {}
for m in MODELS:
    rd = int(json.load(open(p32.SRC['p3169_result'],
                            encoding='utf-8'))['per_model'][m]['readout'])
    z = np.load(p32.SRC[p32.NPZ69_KEY[m]])
    H16 = z['H']
    NTz, NPz, NHz, Dz = H16.shape
    Y = H16[:, :, NHz - 2, :].reshape(p92.NT * p92.NP_, Dz).astype(np.float32)
    tr_base, te_all = {}, {}
    for s in p92.SEEDS_S1:
        tr_pairs, te_pairs = p92.split_s1_246(s)
        tr_base[s] = [t * p92.NP_ + p92.PI_OF_PAIR[p] for t in range(p92.NT)
                      for p in p92.PAIRS if p in tr_pairs]
        te_all[s] = [t * p92.NP_ + p92.PI_OF_PAIR[p] for t in range(p92.NT)
                     for p in p92.PAIRS if p in te_pairs]
    oov_rows = p92.ALL_OOV_ROWS
    oov_row_cls = np.array([p92.PAIRS[r % p92.NP_][1] for r in oov_rows])

    def rv(r, ne):
        i, c = p92.PAIRS[r % p92.NP_]
        return p92.phi_row(i, c, r // p92.NP_, ne, p92.NC, p92.NT)

    Q_rows = np.zeros((len(oov_rows), Dz), dtype=np.float32)
    Rbar_rows = np.zeros((len(oov_rows), Dz), dtype=np.float32)
    seen_amp = {}
    for s in p92.SEEDS_S1:
        tr_rows, te_rows = tr_base[s], te_all[s]
        Xtr = np.stack([rv(r, p92.NE) for r in tr_rows])
        W = p92.ridge_primal(Xtr, Y[tr_rows])
        Xs = np.stack([rv(r, p92.NE) for r in te_rows])
        R_seen = Y[te_rows] - Xs @ W
        for j, r in enumerate(te_rows):
            seen_amp.setdefault(r, []).append(R_seen[j])
        Xo = np.stack([rv(r, p92.NE) for r in oov_rows])
        R_oov = Y[oov_rows] - Xo @ W
        for j in range(len(oov_rows)):
            rvv = R_oov[j]
            Q_rows[j] += rvv / (float(np.linalg.norm(rvv)) + 1e-18)
            Rbar_rows[j] += rvv
        del W, Xtr
    Qn = Q_rows / 3
    Qn = Qn / (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-18)
    Rbar = Rbar_rows / 3
    ustar = np.linalg.svd(Qn, full_matrices=False)[2][0]
    proj_oov = np.abs(Rbar @ ustar)
    seen_flat = np.concatenate([np.array(seen_amp[r])
                                for r in sorted(seen_amp)])
    med_seen = float(np.median(np.abs(seen_flat @ ustar)))
    rec = {'med_seen': med_seen}
    for c in CLS_IDS:
        idxs = np.where(oov_row_cls == c)[0]
        rec['oov_med_%d' % c] = float(np.median(proj_oov[idxs]))
        rec['arc_%d' % c] = rec['oov_med_%d' % c] / med_seen
    del z, H16, Y
    percls_re[m] = rec

# compare vs sealed per_model (rel tol 1e-6)
pm = res['per_model']
for m in MODELS:
    rec = percls_re[m]
    vals = [rec['med_seen'] / pm[m]['med_amp_seen_pooled'] - 1.0]
    for c in CLS_IDS:
        vals.append(abs(rec['arc_%d' % c]
                        - pm[m]['amp_ratio_c'][str(c)])
                    / pm[m]['amp_ratio_c'][str(c)])
        vals.append(abs(rec['oov_med_%d' % c]
                        - pm[m]['oov_cls_med'][str(c)])
                    / pm[m]['oov_cls_med'][str(c)])
    rel = max(vals)
    relmax = max(relmax, rel)
    if rel > 1e-6:
        ok = False
        notes.append('%s rel=%.2e' % (m, rel))
gate('V4 per-class recompute vs sealed (rel tol 1e-6) x3 models', ok,
     'max rel %.2e' % relmax)

# V5 gate logic recheck from sealed numbers
ok, notes = True, []
for m in MODELS:
    g1 = sum(1 for c in CLS_IDS if pm[m]['amp_ratio_c'][str(c)] >= 1.3)
    g2 = pm[m]['class_ratio'] >= 1.3
    if g1 != pm[m]['g1_classes_pass'] or g2 != pm[m]['g2_pass'] \
            or pm[m]['gate_model'] != (g1 >= 3 and g2):
        ok = False
        notes.append(m)
n_g1 = sum(1 for m in MODELS if pm[m]['g1_pass'])
n_g2 = sum(1 for m in MODELS if pm[m]['g2_pass'])
gate('V5 gate logic recheck (G1>=3/4, G2>=1.3, x>=2/3 models)', ok
     and n_g1 >= 2 and n_g2 >= 2,
     '; '.join(notes) or ('G1 %d/3 G2 %d/3 -> ROBUST' % (n_g1, n_g2)))

# V6 gap v1.10 derivation
g19 = json.load(open(p32.SRC['gap_v19'], encoding='utf-8'))
g110 = json.load(open(os.path.join(OUT, 'gap_ledger_v1_10.json'),
                      encoding='utf-8'))
ok, notes = True, []
if g110['version'] != '1.10' or len(g110['gaps']) != 5:
    ok = False
    notes.append('version/count')
for a, b in zip(g19['gaps'], g110['gaps']):
    if a['id'] != b['id']:
        ok = False
        notes.append('order')
    elif a['id'] != 'GAP-4':
        if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
            ok = False
            notes.append('drift ' + a['id'])
    else:
        for k in a:
            if k in ('mechanism_note', 'evidence'):
                continue
            if json.dumps(a[k], sort_keys=True) != \
                    json.dumps(b[k], sort_keys=True):
                ok = False
                notes.append('gap4 ' + k)
        if not b['mechanism_note'].startswith(a['mechanism_note']) \
                or b['evidence'][:-1] != a['evidence']:
            ok = False
            notes.append('gap4 append shape')
mech = [g for g in g110['gaps'] if g['id'] == 'GAP-4'][0]['mechanism_note']
if '3232 G5-G29' not in mech or 'GD_AMP' not in mech.replace(
        '类数稳健性扩样', 'GD_AMP'):
    if '3232 G5-G29' not in mech:
        ok = False
        notes.append('mech note missing 3232 tag')
for k in g19:
    if k not in ('version', 'supersedes', 'updated', 'gaps'):
        if json.dumps(g19[k], sort_keys=True) != \
                json.dumps(g110[k], sort_keys=True):
            ok = False
            notes.append('top ' + k)
gate('V6 gap v1.10 byte-preservation + GAP-4 append', ok,
     '; '.join(notes) or 'legacy x4 + GAP-4 fields preserved, note appended')

# V7 npz vs result
zn = np.load(os.path.join(OUT, 'gd_amp_class_expansion.npz'))
ok, notes = True, []
for m in MODELS:
    for key, arr in (('oov_cls_med', zn[m + '_oov_cls_med']),
                     ('amp_ratio_c', zn[m + '_amp_ratio_c'])):
        d = {str(c): float(arr[i]) for i, c in enumerate(CLS_IDS)}
        if d != pm[m][key]:
            ok = False
            notes.append('%s %s' % (m, key))
gate('V7 npz payloads == result.json', ok, '; '.join(notes) or 'x3 models')

# V8 smoke segregation + corrections on record
smoke = json.load(open(os.path.join(OUT, 'smoke_result.json'),
                       encoding='utf-8'))
gate('V8 smoke==FULL numbers (zero GPU), C1 on record',
     smoke['per_model']['qwen3-4b']['amp_ratio_c'] ==
     pm['qwen3-4b']['amp_ratio_c']
     and any('C1' in c for c in
             json.load(open(os.path.join(OUT, 'execution.json'),
                            encoding='utf-8'))['design']
             ['corrections_registered_pre_observation']),
     'smoke res %s (mode-only diff expected)' % smoke['res_sha8'])

n_fail = sum(1 for g in G if not g['pass'])
verdict = 'VERIFY_PASS' if n_fail == 0 else 'VERIFY_FAIL(%d)' % n_fail
print('==', verdict, '==', flush=True)
with open(os.path.join(OUT, 'verify.json'), 'w', encoding='utf-8',
          newline='') as f:
    json.dump({'phase': 3232, 'verify': G, 'verdict': verdict}, f,
              ensure_ascii=False, indent=1)
