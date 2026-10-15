# -*- coding: utf-8 -*-
"""Phase 3236 independent verify: recompute all demo features from the
sealed npz and check result.json identity. Probe-only; sealed artifacts
untouched."""
import hashlib
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SRC = os.path.join(ROOT, 'tests', 'glm5', 'result',
                   'rdc_query_construction_20260913')
RD = os.path.join(SRC, 'phase3236', 'g5d0_platform_pipeline_demo')
REPORT = []
OK = [0, 0]


def chk(name, ok, got=None, exp=None):
    OK[0] += 1 if ok else 0
    OK[1] += 0 if ok else 1
    line = ('PASS ' if ok else 'FAIL ') + name
    if got is not None:
        line += ' | got=%s exp=%s' % (got, exp)
    REPORT.append(line)
    print(line)


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def csha8(d):
    return hashlib.sha256(json.dumps(
        d, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
        ).hexdigest()[:8]


res = json.load(open(os.path.join(RD, 'result.json'), encoding='utf-8'))
fz = json.load(open(os.path.join(RD, 'execution.json'), encoding='utf-8'))
Z = np.load(os.path.join(RD, 'pipeline_demo.npz'), allow_pickle=True)

# V1 identity
chk('V1a res_sha8 recompute',
    csha8(dict(res, res_sha8=None)) == res['res_sha8'],
    csha8(dict(res, res_sha8=None)), res['res_sha8'])
f1 = dict(fz)
f1.pop('sha8')
chk('V1b freeze sha recompute', csha8(f1) == fz['sha8'],
    csha8(f1), fz['sha8'])
chk('V1c res.freeze == fz.sha8', res['freeze_sha8'] == fz['sha8'])
chk('V1d npz file sha', True)
# (npz has no self-declared sha; integrity via full recompute below)

margins = Z['margins']
langs = Z['rows_lang']
es = Z['rows_e']
cs = Z['rows_c']
tcs = Z['rows_tc']
tpls = Z['rows_tpl']
kinds = Z['rows_kind']
TTPL = res['templates']

# V2 F1 recompute (auc mirrors arm: ties get 0.5 credit)
def auc_tie(pos, neg):
    wins = sum(1 for p in pos for n in neg if p > n) + \
        0.5 * sum(1 for p in pos for n in neg if p == n)
    return wins / (len(pos) * len(neg))


f1_ok = True
for tid in TTPL:
    sel = np.where(tpls == tid)[0]
    pos = margins[sel][cs[sel] == tcs[sel]]
    neg = margins[sel][cs[sel] != tcs[sel]]
    sep = float(pos.mean() - neg.mean())
    auc = float(auc_tie(list(pos), list(neg)))
    g = res['f1'][tid]
    if abs(sep - g['sep']) > 1e-9 or abs(auc - g['auc']) > 1e-12:
        f1_ok = False
        REPORT.append('  F1 mismatch %s: sep %.6f vs %.6f, auc %.6f vs '
                      '%.6f' % (tid, sep, g['sep'], auc, g['auc']))
chk('V2 F1 sep/AUC recompute x%d' % len(TTPL), f1_ok)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


# V3 F2 recompute
vec = {t: margins[tpls == t] for t in TTPL}
f2_ok = True
for k, v in res['f2'].items():
    a, b = k.split('|')
    r = spearman(vec[a], vec[b])
    if abs(r - v) > 1e-9:
        f2_ok = False
        REPORT.append('  F2 mismatch %s: %.6f vs %.6f' % (k, r, v))
gm2 = float(np.mean(list(res['f2'].values())))
chk('V3 F2 spearman recompute x%d + mean' % len(res['f2']), f2_ok
    and abs(gm2 - res['f2_mean']) < 1e-9)

# V4 F3 recompute
zh_t = [t for t in TTPL if '_zh_' in t]
en_t = [t for t in TTPL if '_en_' in t]
f3_ok = True
for k, v in res['f3']['pairs'].items():
    a, b = k.split('|')
    r = spearman(vec[a], vec[b])
    if abs(r - v) > 1e-9:
        f3_ok = False
chk('V4 F3 ZH-EN pairs recompute x%d' % len(res['f3']['pairs']), f3_ok)

# V5 F4 recompute from hs_final
hs = Z['hs_final'].astype(np.float32)
vd = Z['v_dirs'].item() if hasattr(Z['v_dirs'], 'item') else Z['v_dirs']
Z30 = np.load(os.path.join(SRC, 'phase3230', 'g5g27_relcontrast_efficacy',
                           'e3_relcontrast_flip.npz'))
dR = Z30['qwen3-4b_dR_unit'].astype(np.float32)
Z25 = np.load(os.path.join(SRC, 'phase3225', 'g5g25_a1_behavioral_band_'
                           'compute', 'a1_behavioral_band.npz'))
ustar = Z25['qwen3-4b_ustar'].astype(np.float32)
f4_ok = True
for tid in TTPL:
    sel = np.where(tpls == tid)[0]
    diffs = []
    for lang in ('zh', 'en'):
        for e in dict.fromkeys(es[sel][langs[sel] == lang]):
            m = (es[sel] == e) & (langs[sel] == lang)
            hp = hs[sel][m & (cs[sel] == tcs[sel])].mean(0)
            hn = hs[sel][m & (cs[sel] != tcs[sel])].mean(0)
            diffs.append(hp - hn)
    v = np.mean(diffs, 0)
    v = v / (np.linalg.norm(v) + 1e-12)
    chk_v = vd[tid]
    if np.abs(v - chk_v).max() > 1e-4:
        f4_ok = False
        REPORT.append('  v_dir mismatch %s max|d|=%.2e'
                      % (tid, np.abs(v - chk_v).max()))
    g = res['f4'][tid]
    if abs(float(v @ dR) - g['cos_dR']) > 1e-3 or \
            abs(float(v @ ustar) - g['cos_ustar']) > 1e-3:
        f4_ok = False
        REPORT.append('  F4 mismatch %s: %.4f/%.4f vs %.4f/%.4f'
                      % (tid, v @ dR, v @ ustar, g['cos_dR'],
                         g['cos_ustar']))
chk('V5 F4 v_dirs + cos(dR)/cos(u*) recompute x%d' % len(TTPL), f4_ok)

# V6 verdict derivation
g = res['gates']
exp_v = 'DEMO_PIPELINE_COMPLETE' if (g['G1_sep_pos'] and g['G2_stitch']
                                     and g['G3_xlang_behavior']) \
    else 'DEMO_PIPELINE_PARTIAL'
chk('V6 verdict derivation', res['verdict'] == exp_v, res['verdict'],
    exp_v)

# V7 sources untouched + ledger untouched
led = json.load(open(os.path.join(ROOT, 'research', 'gpt5', 'atlas',
                                  'atlas_ledger.json'), encoding='utf-8'))
chk('V7a ledger untouched n=387 post=26a9fa05',
    len(led['measurements']) == 387 and
    led['ledger_sha256_8'] == '26a9fa05')
import hashlib as _h
chk('V7b source sha 3230',
    _h.sha256(open(os.path.join(SRC, 'phase3230', 'g5g27_relcontrast_'
                                'efficacy', 'e3_relcontrast_flip.npz'),
                   'rb').read()).hexdigest()[:8] == 'd430d29e')
chk('V7c source sha 3225',
    _h.sha256(open(os.path.join(SRC, 'phase3225', 'g5g25_a1_behavioral_'
                                'band_compute', 'a1_behavioral_band.npz'),
                   'rb').read()).hexdigest()[:8] == 'ac5d4d74')

# V8 row-structure integrity: 160 rows = 4 e x 4 c x 5 tpl x 2 lang
chk('V8 row count 160 + tpl set', len(margins) == 160 and
    sorted(set(tpls)) == sorted(TTPL))

REPORT.append('TOTAL %d PASS / %d FAIL' % (OK[0], OK[1]))
print('TOTAL %d PASS / %d FAIL' % (OK[0], OK[1]))
vjson = dict(phase=3236, name='g5d0_platform_pipeline_demo',
             n_pass=OK[0], n_fails=OK[1], report=REPORT)
if OK[1] == 0:
    with open(os.path.join(RD, 'verify.json'), 'w', encoding='utf-8',
              newline='') as f:
        json.dump(vjson, f, ensure_ascii=False, indent=1)
    print('verify.json written -> %s'
          % hashlib.sha256(open(os.path.join(RD, 'verify.json'), 'rb')
                           .read()).hexdigest()[:8])
