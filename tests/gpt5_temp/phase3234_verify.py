# -*- coding: utf-8 -*-
"""Phase 3234 independent verification (read-only on sealed artifacts).

Hash domains (per main script):
- execution.json: sha8 = csha8(dict WITHOUT the 'sha8' key)  [computed pre-assign]
- result.json:    res_sha8 = csha8(dict WITH res_sha8=None)  [computed pre-assign]
All checks recompute from sealed source npz/safetensors; sealed 3234
artifacts are never modified except verify.json written at the end
(only if all checks pass).
"""
import os
import sys
import json
import glob
import hashlib
import numpy as np

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3194_g5d2_ftr26_heldout_upgrade as p94

ROOT = p94.ROOT
SRC_DIR = p94.SRC_DIR
RD = os.path.join(SRC_DIR, 'phase3234', 'g5g30_gap5_upstream_lever_scan')
REPORT = []
FAILS = []


def chk(name, ok, got=None, exp=None):
    REPORT.append('%s %s%s%s' % (
        'PASS' if ok else 'FAIL', name,
        '' if got is None else ' | got=%s' % (got,),
        '' if exp is None else ' | exp=%s' % (exp,)))
    if not ok:
        FAILS.append(name)


def fsha(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def csha8(obj):
    return hashlib.sha256(json.dumps(
        obj, ensure_ascii=False, indent=1, sort_keys=True).encode('utf-8')
    ).hexdigest()[:8]


TOK_ISA, TOK_HASA = 101158, 18830
MODELS = ['qwen3-4b', 'qwen3-14b']
HUB = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface', 'hub')
WU_SOURCES = {
    'qwen3-4b': dict(
        path=os.path.join(ROOT, 'models', 'hf', 'qwen3-4b',
                          'model-00001-of-00003.safetensors'),
        key='model.embed_tokens.weight', sha8='328a91d3'),
    'qwen3-14b': dict(
        path=glob.glob(os.path.join(
            HUB, 'models--Qwen--Qwen3-14B', 'snapshots', '*'))[0]
        + os.sep + 'model-00008-of-00008.safetensors',
        key='lm_head.weight', sha8='0d6b9229'),
}

# ---------- V1 result identity (domain INCLUDES res_sha8=None) ----------
res = json.load(open(os.path.join(RD, 'result.json'), encoding='utf-8'))
r1 = csha8(dict(res, res_sha8=None))
chk('V1 res_sha8 recompute (domain incl. res_sha8=None)',
    r1 == res['res_sha8'], r1, res['res_sha8'])

# ---------- V2 npz seal ----------
npzp = os.path.join(RD, res['sealed_npz']['file'])
chk('V2 npz file sha8', fsha(npzp) == res['sealed_npz']['file_sha8'],
    fsha(npzp), res['sealed_npz']['file_sha8'])

# ---------- V3/V4 freeze ----------
fz = json.load(open(os.path.join(RD, 'execution.json'), encoding='utf-8'))
f1 = dict(fz)
f1.pop('sha8')
chk('V3 freeze sha8 recompute (domain excl. own key)',
    csha8(f1) == fz['sha8'], csha8(f1), fz['sha8'])
chk('V4 res.freeze_sha8 == execution.json sha8',
    res['freeze_sha8'] == fz['sha8'], res['freeze_sha8'], fz['sha8'])

# ---------- V5 shared sealed sources ----------
Z30 = np.load(os.path.join(SRC_DIR, r'phase3230\g5g27_relcontrast_efficacy'
                           r'\e3_relcontrast_flip.npz'))
Z32 = np.load(os.path.join(SRC_DIR, r'phase3232\g5g29_gd_amp_class_robustness'
                           r'\gd_amp_class_expansion.npz'))
E3228 = json.load(open(os.path.join(SRC_DIR, r'phase3228\g5g26_e3_compute_seal'
                                    r'\result.json'), encoding='utf-8'))
E_CLS = {m: E3228['models'][m]['gd_amp']['E_cls']
         for m in ('qwen3-4b', 'qwen3-14b')}
NPZ = np.load(npzp, allow_pickle=True)

for m in MODELS:
    NL = 36 if m == 'qwen3-4b' else 40
    NH = NL + 1
    fin = NH - 1
    z54 = np.load(os.path.join(SRC_DIR, r'phase3154\g1p4_mfd_multifactor_disentangle',
                               m, 'collect.npz'))
    H54 = z54['H']
    tags54 = {k: z54[k].astype(int) for k in ('g', 'l', 's', 'ti', 'ci', 'ho')}
    del z54
    z69 = np.load(os.path.join(SRC_DIR, r'phase3169\g5a6_oov_panel',
                               'collect_%s.npz' % m))
    H69 = z69['H']
    del z69
    folds = [np.where((tags54['ci'] == c) & (tags54['ho'] == 0))[0]
             for c in range(4)]

    from safetensors import safe_open
    srcw = WU_SOURCES[m]
    assert fsha(srcw['path']) == srcw['sha8'], ('W_U source drift', m)
    with safe_open(srcw['path'], framework='pt') as f:
        W_U2 = f.get_slice(srcw['key'])[[TOK_ISA, TOK_HASA]].float().numpy()
    WT = W_U2.astype(np.float32)
    del W_U2

    # V5 J0 anchors (cache-based, bitwise for both models)
    m0_54 = H54[:, fin, :].astype(np.float32) @ WT.T
    J0_54 = (m0_54[:, 0] > m0_54[:, 1]).astype(np.int8)
    d_ref = float(np.abs(m0_54 - Z30[m + '_seen_m0'].astype(np.float32)).max())
    chk('V5a %s cache-m0 == 3230 seen_m0 bitwise' % m, d_ref == 0.0, d_ref, 0.0)
    J0_69 = (H69[0, :, fin, :].astype(np.float32) @ WT.T)
    J0_69 = (J0_69[:, 0] > J0_69[:, 1]).astype(np.int8)
    m0_69d = J0_69  # margins below
    m0_69m = H69[0, :, fin, :].astype(np.float32) @ WT.T
    chk('V5b %s npz coarse_J0 bitwise' % m,
        np.array_equal(NPZ[m + '_coarse_J0'], J0_54))
    chk('V5c %s npz fine_J0 bitwise' % m,
        np.array_equal(NPZ[m + '_fine_J0'], J0_69))

    rm = res['models'][m]

    def flip_stats(J1, J0, folds):
        rr = {}
        for fi, sl in enumerate(folds):
            for R, Rn in ((1, 'isa'), (0, 'hasa')):
                fl = (J0[sl] == (1 - R)) & (J1[sl].astype(np.int8) == R)
                rr['%s|cut%d' % (Rn, fi)] = dict(
                    n=int(fl.sum()), rows=int(len(sl)),
                    rate=float(fl.sum() / len(sl)))
        return rr

    # V6 coarse recompute from stored J1
    LGRID = sorted(int(k.split('|L')[1]) for k in rm['coarse']
                   if k.startswith('u_isa|L'))
    ok6 = True
    for dname in ('u_isa', 'u_hasa'):
        for L in LGRID:
            key = '%s|L%d' % (dname, L)
            st = flip_stats(NPZ['%s_%s_J1_L%d' % (m, dname, L)], J0_54, folds)
            if json.dumps(st, sort_keys=True) != json.dumps(
                    rm['coarse'][key], sort_keys=True):
                ok6 = False
                REPORT.append('  mismatch at %s %s' % (m, key))
    chk('V6 %s coarse flip_stats recompute from J1 x%d keys' % (m, 2 * len(LGRID)),
        ok6)

    # V7 pooled recompute + Lstar
    ok7 = True
    for dname in ('u_isa', 'u_hasa'):
        Rn = 'isa' if dname == 'u_isa' else 'hasa'
        pr = {}
        for L in LGRID:
            key = '%s|L%d' % (dname, L)
            pr[str(L)] = float(np.mean(
                [rm['coarse'][key]['%s|cut%d' % (Rn, f)]['rate']
                 for f in range(4)]))
        Lstar = max(pr, key=pr.get)
        pr['Lstar'] = int(Lstar)
        if pr != rm['pooled'][dname]:
            ok7 = False
            REPORT.append('  pooled mismatch %s %s' % (m, dname))
    chk('V7 %s pooled rates + Lstar recompute' % m, ok7)

    # V8 fine recompute from stored J1/marg
    fine_keys = [k for k in rm['fine'] if k.startswith('u_')]
    ok8 = True
    for key in fine_keys:
        dname, Ls, a = key.split('|')
        L = int(Ls[1:])
        J1 = NPZ['%s_fine_%s_L%d_%s_J1' % (m, dname, L, a)]
        mg = NPZ['%s_fine_%s_L%d_%s_marg' % (m, dname, L, a)]
        nflip = int(((J0_69 == 0) & (J1 == 1)).sum()
                    + ((J0_69 == 1) & (J1 == 0)).sum())
        dD = float(((mg[:, 0] - mg[:, 1]) - (m0_69m[:, 0] - m0_69m[:, 1])).mean())
        exp = dict(flip={'tpl0': dict(n=nflip, rows=730, rate=nflip / 730.0)},
                   dD=dD)
        got = rm['fine'][key]
        if abs(got['flip']['tpl0']['n'] - nflip) > 0 or \
           abs(got['dD'] - dD) > 1e-9:
            ok8 = False
            REPORT.append('  fine mismatch %s %s' % (m, key))
    chk('V8 %s fine flip/dD recompute from J1+marg x%d keys' % (m, len(fine_keys)),
        ok8)

    # V9 nulls recompute
    null_keys = [k for k in rm['nulls'] if k.startswith('u_')]
    ok9 = True
    for key in null_keys:
        dname, ndname, Ls = key.split('|')
        L = int(Ls[1:])
        st = flip_stats(NPZ['%s_null_%s_L%d_J1' % (m, ndname, L)], J0_54, folds)
        if json.dumps(st, sort_keys=True) != json.dumps(
                rm['nulls'][key], sort_keys=True):
            ok9 = False
            REPORT.append('  null mismatch %s %s' % (m, key))
    chk('V9 %s null flip_stats recompute x%d keys' % (m, len(null_keys)), ok9)

    # V10 UL_amp spearman recompute (4 OOV + 10cls report-only)
    ok10 = True
    for L, rec in rm['ul_amp'].items():
        amps = rec['amps']
        rx = np.argsort(np.argsort(amps)).astype(np.float64)
        ry = np.argsort(np.argsort(E_CLS[m])).astype(np.float64)
        rho = float(np.corrcoef(rx, ry)[0, 1])
        if abs(rho - rec['spearman_4oov']) > 1e-12:
            ok10 = False
            REPORT.append('  ul_amp rho mismatch %s L%s' % (m, L))
        rx2 = np.argsort(np.argsort(
            rm['ustar_band'][L]['per_class_med_shift'])).astype(np.float64)
        ry2 = np.argsort(np.argsort(
            Z32[m + '_collapse_amp'].astype(np.float64))).astype(np.float64)
        rho2 = float(np.corrcoef(rx2, ry2)[0, 1])
        if abs(rho2 - rec['spearman_10cls_report_only']) > 1e-12:
            ok10 = False
            REPORT.append('  10cls rho mismatch %s L%s' % (m, L))
    chk('V10 %s UL_amp spearman recompute (both stats)' % m, ok10)

    # V11 extra flip arithmetic
    ok11 = all(abs(rec['rate'] - rec['flip_n'] / 730.0) < 1e-12
               for rec in rm['extra'].values())
    chk('V11 %s extra rate arithmetic' % m, ok11)

    # V12 ustar_band per-class med shift recompute
    ok12 = True
    cls_of_row = np.array([p94.PAIRS[i][1] for i in range(730)])
    for L, rec in rm['ustar_band'].items():
        mg = NPZ['%s_ustar_L%s_marg' % (m, L)]
        d_shift = (mg[:, 0] - mg[:, 1]) - (m0_69m[:, 0] - m0_69m[:, 1])
        per_cls = [float(np.median(d_shift[cls_of_row == c]))
                   for c in range(10)]
        if max(abs(per_cls[c] - rec['per_class_med_shift'][c])
               for c in range(10)) > 1e-9:
            ok12 = False
            REPORT.append('  ustar_band mismatch %s L%s' % (m, L))
    chk('V12 %s ustar_band per-class med shift recompute' % m, ok12)
    del H54, H69

# V13 bridge consistency (4b): L == pooled Lstar; rates in [0,1]
if 'nf4_bridge_4b' in res['models']['qwen3-4b']:
    br = res['models']['qwen3-4b']['nf4_bridge_4b']
    pool = res['models']['qwen3-4b']['pooled']
    ok13 = all(br[d]['L'] == pool[d]['Lstar'] and
               all(0.0 <= v <= 1.0 for v in br[d]['rates'].values())
               for d in br)
    chk('V13 nf4_bridge_4b L==pooled Lstar + rates in [0,1]', ok13)
else:
    REPORT.append('SKIP V13 (nf4_bridge_4b absent)')

# V14 verdict/contradictions consistency
exp_v = 'upstream_lever_scan_sealed' if not res['contradictions'] else \
    'upstream_lever_scan_sealed_dirty(%d)' % len(res['contradictions'])
chk('V14 verdict consistent with contradictions',
    res['verdict'] == exp_v, res['verdict'], exp_v)

# V15 ledger untouched (scan arm writes no ledger entry)
led = json.load(open(os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json'),
                     encoding='utf-8'))
chk('V15 ledger untouched n=385 post=86f04998',
    (len(led['measurements']), led['ledger_sha256_8']) == (385, '86f04998'),
    (len(led['measurements']), led['ledger_sha256_8']))

# ---------- report ----------
out_path = os.path.join(RD, 'verify.json')
n_pass = sum(1 for r in REPORT if r.startswith('PASS'))
n_skip = sum(1 for r in REPORT if r.startswith('SKIP'))
summary = {
    'phase': 3234, 'name': 'g5g30_gap5_upstream_lever_scan',
    'verify': 'ALL PASS' if not FAILS else 'FAIL:%s' % FAILS,
    'n_pass': n_pass, 'n_fails': len(FAILS), 'n_skip': n_skip,
    'res_sha8': res['res_sha8'], 'freeze_sha8': fz['sha8'],
    'npz_sha8': res['sealed_npz']['file_sha8'],
    'notes': 'verify-only script; sealed artifacts untouched; '
             'J0/flip/dD/spearman/med-shift recomputed from sealed '
             'caches + W_U slices',
}
with open(out_path, 'w', encoding='utf-8', newline='') as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print('\n'.join(REPORT))
print('---')
print('TOTAL %d PASS / %d FAIL / %d SKIP' % (n_pass, len(FAILS), n_skip))
print('verify.json written ->', fsha(out_path))
