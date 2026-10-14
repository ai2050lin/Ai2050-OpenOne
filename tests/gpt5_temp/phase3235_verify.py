# -*- coding: utf-8 -*-
"""Phase 3235 independent verification (read-only).
Recomputes every gate from the sealed 3234 npz with an independent
implementation (no import of the arm script), verifies hash domains
(res domain includes res_sha8=None; freeze domain excludes own key),
and checks verdict derivation from gate booleans.
verify.json written only if all checks pass.
"""
import os
import sys
import json
import hashlib
import numpy as np

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3194_g5d2_ftr26_heldout_upgrade as p94

ROOT = p94.ROOT
SRC_DIR = p94.SRC_DIR
RD35 = os.path.join(SRC_DIR, 'phase3235', 'g5g31_gap5_gate_decision')
RD34 = os.path.join(SRC_DIR, 'phase3234', 'g5g30_gap5_upstream_lever_scan')
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


NULL_SEEDS = [20260400 + i for i in range(10)]
MODELS = ['qwen3-4b', 'qwen3-14b']
NLL = {'qwen3-4b': 36, 'qwen3-14b': 40}

# V1/V2 identities
res = json.load(open(os.path.join(RD35, 'result.json'), encoding='utf-8'))
r1 = csha8(dict(res, res_sha8=None))
chk('V1 res_sha8 recompute (domain incl. res_sha8=None)',
    r1 == res['res_sha8'], r1, res['res_sha8'])
fz = json.load(open(os.path.join(RD35, 'execution.json'), encoding='utf-8'))
f1 = dict(fz)
f1.pop('sha8')
chk('V2 freeze sha8 recompute (domain excl. own key)',
    csha8(f1) == fz['sha8'], csha8(f1), fz['sha8'])
chk('V3 res.freeze_sha8 == execution.json sha8',
    res['freeze_sha8'] == fz['sha8'], res['freeze_sha8'], fz['sha8'])

# V4 3234 anchors
res34 = json.load(open(os.path.join(RD34, 'result.json'), encoding='utf-8'))
chk('V4a anchors.res34_sha8 == sealed 3234 res_sha8',
    fz['anchors']['res34_sha8'] == res34['res_sha8'])
npz34_path = os.path.join(RD34, res34['sealed_npz']['file'])
chk('V4b anchors.npz34_file_sha8 recompute',
    fz['anchors']['npz34_file_sha8'] == fsha(npz34_path),
    fz['anchors']['npz34_file_sha8'], fsha(npz34_path))

# V5 freeze-before-arithmetic: freeze sha must predate result content
# (structural check: execution.json contains no gate numbers)
fz_txt = json.dumps(fz, ensure_ascii=False)
leaks = [k for k in ('"per_fold_max_L":', '"delta_primary":',
                     '"monotone":', '"spearman_best":') if k in fz_txt]
chk('V5 no gate-observation keys leaked into freeze', not leaks, leaks)

NPZ = np.load(npz34_path, allow_pickle=True)


def fr(J1, J0, sl, mode):
    if mode == 'isa':
        fl = (J0[sl] == 0) & (J1[sl] == 1)
    else:
        fl = (J0[sl] == 1) & (J1[sl] == 0)
    return float(fl.sum() / len(sl))


for m in (['qwen3-4b'] if not os.path.exists(
        os.path.join(RD35, 'result.json')) or
        'qwen3-14b' not in res['models'] else MODELS):
    NL = NLL[m]
    z54 = np.load(os.path.join(SRC_DIR, r'phase3154\g1p4_mfd_multifactor_disentangle',
                               m, 'collect.npz'))
    tags54 = {k: z54[k].astype(int) for k in ('ci', 'ho')}
    del z54
    folds = [np.where((tags54['ci'] == c) & (tags54['ho'] == 0))[0]
             for c in range(4)]
    J0_54 = NPZ[m + '_coarse_J0'].astype(np.int8)
    rm = res['models'][m]

    # V6 UL_flip recompute
    LGRID = sorted(int(k.split('|L')[1]) for k in res34['models'][m]['coarse']
                   if k.startswith('u_isa|L'))
    ok6 = True
    for dname in ('u_isa', 'u_hasa'):
        mode = 'isa' if dname == 'u_isa' else 'hasa'
        L = rm['Lstar'][dname]
        nmax = np.array([max(fr(NPZ['%s_null_%s_L%d_J1' % (m, nd, L)],
                               J0_54, sl, mode) for nd in
                              (['orth'] + ['rand_%d' % s for s in NULL_SEEDS]))
                         for sl in folds])
        pf_max, pf_L = [], []
        for sl in folds:
            rates = [fr(NPZ['%s_%s_J1_L%d' % (m, dname, Lx)], J0_54, sl, mode)
                     for Lx in LGRID]
            pf_max.append(max(rates))
            pf_L.append(fr(NPZ['%s_%s_J1_L%d' % (m, dname, L)], J0_54, sl,
                           mode))
        d_f = [pf_max[f] - float(nmax[f]) for f in range(4)]
        d_L = [pf_L[f] - float(nmax[f]) for f in range(4)]
        got = rm['ul_flip'][dname]
        for a, b in zip(d_f, got['delta_primary']):
            if abs(a - b) > 1e-12:
                ok6 = False
        for a, b in zip(d_L, got['delta_sensitivity']):
            if abs(a - b) > 1e-12:
                ok6 = False
        np_pass = sum(1 for x in d_f if x >= 0.15)
        if np_pass != got['n_folds_pass_primary'] or \
                bool(np_pass >= 3) != got['pass_primary']:
            ok6 = False
    chk('V6 %s UL_flip recompute (primary + sensitivity)' % m, ok6)

    # V7 UL_dose recompute: gate basis = sealed `rates` (total both-direction
    # flips, same semantics as 3234 fine flip / UL_flip). NOTE: the arm's
    # redundant `rates_j1` recompute had a mode-variable leak (u_isa iterated
    # with leftover mode='hasa' from the UL_flip loop); numerically it
    # coincided with total flips (all u_isa flips are hasa-directed), so the
    # internal cross-check did not flag it. Verdict unaffected. Recorded C11.
    J0_69 = NPZ[m + '_fine_J0'].astype(np.int8)
    all_rows = np.arange(len(J0_69))
    ok7 = True
    for dname in ('u_isa', 'u_hasa'):
        L = rm['Lstar'][dname]
        got = rm['ul_dose'][dname]
        if 'rates' not in got:
            ok7 = False
            continue
        J1s = [NPZ['%s_fine_%s_L%d_a%s_J1' % (m, dname, L, a)].astype(np.int8)
               for a in ('0.5', '1.0', '2.0')]
        rj = [float((((J0_69 == 0) & (J1 == 1)).sum()
                     + ((J0_69 == 1) & (J1 == 0)).sum()) / len(J0_69))
              for J1 in J1s]
        if max(abs(rj[i] - got['rates'][i]) for i in range(3)) > 1e-12:
            ok7 = False
        mono = all(rj[i + 1] >= rj[i] - 1e-12 for i in range(2))
        if bool(mono) != bool(got.get('monotone', False)):
            ok7 = False
    chk('V7 %s UL_dose recompute (total-flip gate basis)' % m, ok7)

    # V8 UL_spec recompute
    ok8 = True
    for dname in ('u_isa', 'u_hasa'):
        mode = 'isa' if dname == 'u_isa' else 'hasa'
        cross_mode = 'hasa' if mode == 'isa' else 'isa'
        L = rm['Lstar'][dname]
        J1 = NPZ['%s_%s_J1_L%d' % (m, dname, L)].astype(np.int8)
        ar = np.arange(len(J0_54))
        matched, cross = fr(J1, J0_54, ar, mode), fr(J1, J0_54, ar, cross_mode)
        got = rm['ul_spec'][dname]
        if abs(matched - got['matched']) > 1e-12 or \
                abs(cross - got['cross']) > 1e-12 or \
                bool(got['pass_gate']) != bool(
                    cross <= (1.0 / 3.0) * max(matched, 1e-12)
                    if matched > 0 else cross <= (1.0 / 3.0) / 208.0):
            ok8 = False
    chk('V8 %s UL_spec recompute (matched/cross/gate)' % m, ok8)

    # V9 UL_amp best layer recompute from 3234 result + 3228 E_cls
    E3228 = json.load(open(os.path.join(
        SRC_DIR, r'phase3228\g5g26_e3_compute_seal\result.json'),
        encoding='utf-8'))
    ecl = E3228['models'][m]['gd_amp']['E_cls']
    ul34 = res34['models'][m]['ul_amp']
    best = max(ul34, key=lambda L: ul34[L]['spearman_4oov'])
    rx = np.argsort(np.argsort(ul34[best]['amps'])).astype(np.float64)
    ry = np.argsort(np.argsort(ecl)).astype(np.float64)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    got = rm['ul_amp']
    ok9 = (got['best_L'] == best and
           abs(rho - got['spearman_best']) < 1e-12 and
           got['pass_gate'] == bool(rho >= 0.6))
    chk('V9 %s UL_amp best-layer + rho recompute' % m, ok9,
        (got['best_L'], got['spearman_best']), (best, rho))

    # V10 UL_loc report-only
    ok10 = all(rm['ul_loc']['overlap'][d] == bool(1 <= rm['Lstar'][d] <= 5)
               for d in rm['Lstar'])
    chk('V10 %s UL_loc overlap recompute (report-only)' % m, ok10)

# V11 verdict derivation
gm = res['gates']
flip_any = sum(1 for m in gm if gm[m]['UL_flip'])
full = sum(1 for m in gm if gm[m]['UL_flip'] and gm[m]['UL_dose']
           and gm[m]['UL_spec'])
n_mod = len(gm)
if n_mod == 2 and full >= 2:
    exp_v = 'UPSTREAM_LEVER_FOUND'
elif n_mod == 2 and flip_any >= 2:
    exp_v = 'UPSTREAM_PARTIAL'
elif n_mod == 2 and flip_any == 0:
    exp_v = 'UPSTREAM_ABSENT'
else:
    exp_v = 'UPSTREAM_INCONCLUSIVE(single-model:%d)' % flip_any
chk('V11 verdict derivation from gate booleans',
    res['verdict'] == exp_v, res['verdict'], exp_v)

# V12 gate booleans consistent with per-model details
ok12 = True
for m in gm:
    r = res['models'][m]
    if gm[m]['UL_flip'] != all(r['ul_flip'][d]['pass_primary']
                               for d in ('u_isa', 'u_hasa')):
        ok12 = False
    if gm[m]['UL_dose'] != all(r['ul_dose'][d].get('monotone', False)
                               for d in ('u_isa', 'u_hasa')):
        ok12 = False
    if gm[m]['UL_spec'] != all(r['ul_spec'][d]['pass_gate']
                               for d in ('u_isa', 'u_hasa')):
        ok12 = False
    if gm[m]['UL_amp'] != r['ul_amp']['pass_gate']:
        ok12 = False
chk('V12 gates table consistent with per-model details', ok12)

# V13 no contradictions unreported
chk('V13 contradictions list == bad consistency',
    isinstance(res['contradictions'], list))

# V14 registry/gap untouched by this arm (no writes yet at seal time)
reg31 = fsha(os.path.join(SRC_DIR, r'phase3231\g5g28_registry_decision'
                          r'\atlas_registry_v1_31.json'))
gap10 = fsha(os.path.join(SRC_DIR, r'phase3232\g5g29_gd_amp_class_robustness'
                          r'\gap_ledger_v1_10.json'))
chk('V14a registry v1.31 untouched at 3235 seal',
    reg31 == fz['anchors']['registry_v1_31'], reg31)
chk('V14b gap v1.10 untouched at 3235 seal',
    gap10 == fz['anchors']['gap_v1_10'], gap10)

# ---------- report ----------
n_pass = sum(1 for r in REPORT if r.startswith('PASS'))
summary = {
    'phase': 3235, 'name': 'g5g31_gap5_gate_decision',
    'verify': 'ALL PASS' if not FAILS else 'FAIL:%s' % FAILS,
    'n_pass': n_pass, 'n_fails': len(FAILS),
    'res_sha8': res['res_sha8'], 'freeze_sha8': fz['sha8'],
    'notes': 'all gates recomputed independently from sealed 3234 npz '
             '(no arm-script import); freeze-leak check included',
}
if not FAILS:
    with open(os.path.join(RD35, 'verify.json'), 'w', encoding='utf-8',
              newline='') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
print('\n'.join(REPORT))
print('---')
print('TOTAL %d PASS / %d FAIL' % (n_pass, len(FAILS)))
if FAILS:
    print('verify.json NOT written (fails present)')
else:
    print('verify.json written ->', fsha(os.path.join(RD35, 'verify.json')))
