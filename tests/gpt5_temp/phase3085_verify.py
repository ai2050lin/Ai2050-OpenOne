# -*- coding: utf-8 -*-
"""Phase 3085 independent verify: seal sha8,
npz bit-replay (repro anchor / spectrum /
T/U/F / E2 / G_DS / partial / verdict),
ledger, documents.
Writes VERIFY report; exits nonzero on fail."""
import hashlib
import io
import json
import os
from statistics import NormalDist

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
R = (ROOT + r'\tests\glm5\result'
     r'\rdc_query_construction_20260913'
     r'\phase3085'
     r'\omega_p82_l34_full_arbitration')
SCRIPT = (ROOT + r'\tests\glm5'
          r'\phase3085_omega_p82_l34_full_'
          r'arbitration.py')
NPZ84 = (ROOT + r'\tests\glm5\result'
         r'\rdc_query_construction_20260913'
         r'\phase3084'
         r'\omega_p81_3b_layer_scan'
         r'\omega_p81_3b_layer_scan.npz')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG = ROOT + r'\.workbuddy\memory\2026-09-22.md'
MEMW = ROOT + r'\.workbuddy\memory\MEMORY.md'
REPF = (ROOT + r'\tests\gpt5_temp'
        r'\p3085_verify_report.txt')

ok = []
fail = []


def chk(cond, tag):
    if cond:
        ok.append(tag)
    else:
        fail.append(tag)


def sha8(path):
    with io.open(path, 'rb') as f:
        return hashlib.sha256(
            f.read()).hexdigest()[:8]


def spearman(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.argsort(np.argsort(a)) \
        .astype(np.float64)
    rb = np.argsort(np.argsort(b)) \
        .astype(np.float64)
    ra -= ra.mean()
    rb -= rb.mean()
    den = np.sqrt((ra * ra).sum()
                  * (rb * rb).sum())
    if den == 0:
        return 0.0
    return float((ra * rb).sum() / den)


def cosv(a, b):
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b) / (na * nb)


def pearson(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt((a * a).sum()
                  * (b * b).sum())
    if den == 0:
        return 0.0
    return float((a * b).sum() / den)


def ranks(x):
    x = np.asarray(x, np.float64)
    return np.argsort(np.argsort(x)) \
        .astype(np.float64)


def partial_sp(x, y, z):
    rx = ranks(x)
    ry = ranks(y)
    rz = ranks(z)
    r_xy = pearson(rx, ry)
    r_xz = pearson(rx, rz)
    r_yz = pearson(ry, rz)
    den = np.sqrt(
        (1.0 - r_xz * r_xz)
        * (1.0 - r_yz * r_yz))
    if den == 0:
        return 0.0
    return float((r_xy - r_xz * r_yz) / den)


def perm_p(a, b, n_perm, seed):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    obs = abs(spearman(a, b))
    if n_perm <= 0:
        return 1.0
    ra = np.argsort(np.argsort(a)) \
        .astype(np.float64)
    ra -= ra.mean()
    B = np.tile(b, (n_perm, 1))
    B = rng.permuted(B, axis=1)
    rb = np.argsort(np.argsort(B, axis=1),
                    axis=1).astype(np.float64)
    rb -= rb.mean(axis=1, keepdims=True)
    num = (rb * ra[None, :]).sum(axis=1)
    den = np.sqrt(
        (rb * rb).sum(axis=1)
        * float((ra * ra).sum()))
    den[den == 0] = 1.0
    stats = np.abs(num / den)
    return float((stats >= obs - 1e-12)
                 .mean())


def partial_sp_perm_p(x, y, z, n_perm, seed):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    z = np.asarray(z, np.float64)
    obs = abs(partial_sp(x, y, z))
    if n_perm <= 0:
        return 1.0
    Y = np.tile(y, (n_perm, 1))
    Y = rng.permuted(Y, axis=1)
    rx = ranks(x)
    rz = ranks(z)
    r_xz = pearson(rx, rz)
    cnt = 0
    for i in range(n_perm):
        ry = ranks(Y[i])
        r_xy = pearson(rx, ry)
        r_yz = pearson(ry, rz)
        den = np.sqrt(
            (1.0 - r_xz * r_xz)
            * (1.0 - r_yz * r_yz))
        val = 0.0 if den == 0 else \
            (r_xy - r_xz * r_yz) / den
        if abs(val) >= obs - 1e-12:
            cnt += 1
    return float(cnt) / n_perm


def spectrum_struct(M):
    M = np.asarray(M, dtype=np.float64)
    Mc = M - M.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Mc, compute_uv=False)
    lam = s * s
    tot = float(lam.sum())
    pr = float(lam.sum() ** 2
               / (lam * lam).sum())
    p = lam / tot
    p = p[p > 0]
    H = float(-(p * np.log(p)).sum())
    keff = float(np.exp(H))
    top3 = float(lam[:3].sum() / tot)
    return pr, keff, top3


# ==== seal & result ====
seal = json.load(io.open(
    R + r'\seal.json', encoding='utf-8'))
res = json.load(io.open(
    R + r'\result.json', encoding='utf-8'))
exe = json.load(io.open(
    R + r'\execution.json', encoding='utf-8'))
chk(seal['npz_sha256_8'] == sha8(
    R + r'\omega_p82_l34_full_'
    r'arbitration.npz'), 'seal npz8')
chk(seal['result_sha256_8'] == sha8(
    R + r'\result.json'), 'seal result8')
chk(seal['exec_sha256_8'] == sha8(
    R + r'\execution.json'), 'seal exec8')
chk(seal['script_sha256_8'] == sha8(
    SCRIPT), 'seal script8')
chk(seal['verdict'] == res['verdict'],
    'seal verdict==result')
chk(res['phase'] == 3085
    and res['name']
    == 'omega_p82_l34_full_arbitration',
    'res phase/name')
chk(exe['created'] == res['created']
    == seal['created'], 'created align')
chk(exe.get('prereg') == res.get('prereg'),
    'prereg frozen (exec==result)')

verdict = res['verdict']
ALLOWED = {
    'third_setup_failed',
    'third_top8_degenerate',
    'third_trunk_migrates',
    'third_dispersed_migrates',
    'third_trunk_no_migrate',
    'third_dispersed_no_migrate',
    'third_mixed_locked',
    'third_mixed_partial',
    'third_mixed_absent'}
chk(verdict in ALLOWED, 'verdict allowed')

z = np.load(R + r'\omega_p82_l34_full_'
            r'arbitration.npz',
            allow_pickle=False)
chk(bool(z['SMOKE']) is False, 'not smoke')
chk(int(z['FORWARDS']) > 13000,
    'forwards >13k')
degen = verdict == 'third_top8_degenerate'
chk(bool(z['TOP8_ALL_OK'])
    == (not degen), 'TOP8_ALL_OK vs verdict')
chk(bool(z['SETUP_OK']), 'SETUP_OK')
chk(int(z['L_INJ']) == 34
    and int(z['L_POST']) == 35,
    'L_INJ/L_POST')
chk(bool(z['TIED']), 'TIED recorded True')

FKEYS = ('A', 'B', 'C')
# ==== anchors ====
for fk in FKEYS:
    for bk in ('B0', 'B1', 'B4', 'B6',
               'B7A', 'B8'):
        chk(bool(z[bk + '_OK_' + fk]),
            '%s_OK_%s' % (bk, fk))
        chk(float(z[bk + '_DIFF_' + fk])
            == 0.0,
            '%s_DIFF0_%s' % (bk, fk))
    chk(bool(z['B3_OK_' + fk]), 'B3_' + fk)
    chk(len(z['TOP8_' + fk])
        == min(8, int(z['N_NEG_' + fk])),
        'top8 len %s' % fk)
    if bool(z['TOP8_SEL_OK_' + fk]):
        chk(len(z['TOP8_' + fk]) == 8,
            'top8 full %s' % fk)
    else:
        chk(degen and int(z['N_NEG_'
                          + fk]) < 8,
            'top8 degen %s' % fk)

# ==== repro anchor (vs 3084 npz) ====
chk(bool(z['REPRO_OK']), 'REPRO_OK')
z84 = np.load(NPZ84, allow_pickle=False)
for fk in FKEYS:
    chk(bool(z['REPRO_NNEG_OK_' + fk]),
        'repro nneg ok %s' % fk)
    chk(bool(z['REPRO_TOP8_OK_' + fk]),
        'repro top8 ok %s' % fk)
    chk(float(z['REPRO_MEDC_DIFF_' + fk])
        <= 1e-9, 'repro medc tol %s' % fk)
    chk(float(z['REPRO_CS1H_DIFF_' + fk])
        <= 1e-9, 'repro cs1h tol %s' % fk)
    chk(float(z['REPRO_RALL_DIFF_' + fk])
        <= 1e-9, 'repro rall tol %s' % fk)
    # independent cross-check vs 3084
    chk(int(z['N_NEG_' + fk])
        == int(z84['N_NEG_L34_' + fk]),
        'repro xcheck nneg %s' % fk)
    chk([int(h) for h in z['TOP8_' + fk]]
        == [int(h)
            for h in z84['TOP8_L34_' + fk]],
        'repro xcheck top8 %s' % fk)
    chk(abs(float(z['MED_C_' + fk])
            - float(z84['MED_C_L34_' + fk]))
        <= 1e-9, 'repro xcheck medc %s' % fk)

# ==== spectrum replay ====
spec_cls = str(z['SPEC_CLASS'])
t3s = {}
for fk in FKEYS:
    pr, keff, t3 = spectrum_struct(
        z['CS_' + fk])
    chk(pr == float(z['E3_PR_CS_' + fk]),
        'spec PR bit %s' % fk)
    chk(keff == float(z['E3_KEFF_CS_'
                      + fk]),
        'spec keff bit %s' % fk)
    chk(t3 == float(z['E3_TOP3_CS_'
                     + fk]),
        'spec top3 bit %s' % fk)
    prh, keffh, t3h = spectrum_struct(
        z['CS1H_' + fk])
    chk(prh == float(z['E3_PR_CS1H_'
                      + fk]),
        'spec1h PR bit %s' % fk)
    chk(t3h == float(z['E3_TOP3_CS1H_'
                      + fk]),
        'spec1h top3 bit %s' % fk)
    t3s[fk] = float(z['E3_TOP3_CS_' + fk])
cls_replay = ('trunk' if min(
    t3s.values()) >= 0.9 else
    ('dispersed' if max(t3s.values())
     <= 0.5 else 'mixed'))
chk(cls_replay == spec_cls,
    'spec_class replay')
if not degen:
    chk('spec_class' in res['gates']
        and res['gates']['spec_class']
        == spec_cls, 'gates spec_class')

# ==== cross replay ====
if bool(z['SETUP_OK']) and bool(
        z['TOP8_ALL_OK']) and verdict not in (
        'third_setup_failed',
        'third_top8_degenerate'):
    SEED = 3085
    TT64 = {f: z['TT_' + f].astype(
        np.float64) for f in FKEYS}
    LG64 = {f: z['LG_' + f].astype(
        np.float64) for f in FKEYS}
    CIDX = np.array([k // 8 + 1
                     for k in range(24)])
    BIDX = np.array([k % 8
                     for k in range(24)])
    PREF_K = BIDX * 4 + CIDX
    BASE_KX = BIDX * 4
    cr = res['stats']['cross']
    T = {}
    U = {}
    F = {}
    for fa, fb in (('A', 'B'), ('A', 'C'),
                   ('B', 'C')):
        key = fa + fb
        T[key] = np.array([
            spearman(z['CS1H_' + fa][:, k],
                     z['CS1H_' + fb][:, k])
            for k in range(24)])
        U[key] = np.array([
            spearman(z['CS_' + fa][:, k],
                     z['CS_' + fb][:, k])
            for k in range(24)])
        F[('f1', key)] = np.array([
            spearman(TT64[fa][k],
                     TT64[fb][k])
            for k in range(24)])
        F[('f2', key)] = np.array([
            cosv(TT64[fa][k],
                 TT64[fb][k])
            for k in range(24)])
        F[('f3', key)] = np.array([
            spearman(LG64[fa][PREF_K[k]],
                     LG64[fb][PREF_K[k]])
            for k in range(24)])
        F[('f4', key)] = np.array([
            spearman(LG64[fa][BASE_KX[k]],
                     LG64[fb][BASE_KX[k]])
            for k in range(24)])
        n_fa = np.linalg.norm(TT64[fa],
                              axis=1)
        n_fb = np.linalg.norm(TT64[fb],
                              axis=1)
        F[('f5', key)] = np.minimum(
            n_fa, n_fb) / np.maximum(
            n_fa, n_fb)
        chk(np.allclose(
            T[key], z['T_' + key],
            rtol=0, atol=0),
            'T bit %s' % key)
        chk(np.allclose(
            U[key], z['U_' + key],
            rtol=0, atol=0),
            'U bit %s' % key)
        for fid, tag in (('f1', 'F1_STT'),
                         ('f2', 'F2_CTT'),
                         ('f3', 'F3_SLGP'),
                         ('f4', 'F4_SLGB'),
                         ('f5', 'F5_AMP')):
            chk(np.allclose(
                F[(fid, key)],
                z[tag + '_' + key],
                rtol=0, atol=0),
                '%s bit %s' % (tag, key))
        # mig / sp_as / tt_med replay
        med_c = {f: float(z['MED_C_' + f])
                 for f in FKEYS}
        r1 = {f: (np.median(
            z['CS1H_' + f], axis=1)
            - med_c[f]) for f in FKEYS}
        mig = float(spearman(r1[fa],
                             r1[fb]))
        chk(mig == float(z['MIG_' + key]),
            'MIG bit %s' % key)
        A_S = {
            f: med_c[f]
            - np.median(z['CS_' + f],
                        axis=1)
            for f in FKEYS}
        sp_as = float(spearman(A_S[fa],
                               A_S[fb]))
        chk(sp_as == float(z['SP_AS_'
                           + key]),
            'SP_AS bit %s' % key)
        tt_med = float(np.median(
            [spearman(TT64[fa][k],
                      TT64[fb][k])
             for k in range(24)]))
        chk(tt_med == float(z['TT_MED_'
                             + key]),
            'TT_MED bit %s' % key)
    # E2/G_DS replay
    E2 = {}
    for rn, RSP in (('T', T), ('U', U)):
        for key in ('AB', 'AC', 'BC'):
            sp_ = spearman(F[('f2', key)],
                           RSP[key])
            p_ = perm_p(F[('f2', key)],
                        RSP[key], 20000,
                        SEED)
            E2['%s_%s' % (rn, key)] = (
                sp_, p_)
            chk(sp_ == float(z['E3_F2_CTT_'
                             + rn + '_'
                             + key]),
                'E2 sp bit %s_%s'
                % (rn, key))
            chk(p_ == float(z['E3P_F2_CTT_'
                             + rn + '_'
                             + key]),
                'E2 p bit %s_%s'
                % (rn, key))
    cnt_pos = sum(
        1 for v in E2.values()
        if v[0] > 0 and v[1] < 0.05)
    min_sp = min(v[0] for v in
                 E2.values())
    g_ds = bool(cnt_pos >= 4
                and min_sp > 0)
    chk(int(z['GDS_COUNT']) == cnt_pos,
        'GDS count replay')
    chk(abs(float(z['GDS_MIN_SP'])
            - min_sp) < 1e-15,
        'GDS min_sp replay')
    zl = [NormalDist().inv_cdf(
        min(1.0 - 1e-12, 1.0 - v[1]))
        for v in E2.values()]
    stouffer = float(np.sum(zl)
                     / np.sqrt(len(zl)))
    chk(abs(float(z['STOUFFER_Z'])
            - stouffer) < 1e-9,
        'stouffer replay')
    p_f2_TAB = E2['T_AB'][1]
    if g_ds and p_f2_TAB < 0.05:
        mig_state = 'locked'
    elif g_ds:
        mig_state = 'partial'
    else:
        mig_state = 'absent'
    if spec_cls == 'mixed':
        vr = 'third_mixed_' + mig_state
    elif mig_state == 'absent':
        vr = ('third_trunk_no_migrate'
              if spec_cls == 'trunk' else
              'third_dispersed_no_'
              'migrate')
    else:
        vr = ('third_trunk_migrates'
              if spec_cls == 'trunk' else
              'third_dispersed_migrates')
    chk(vr == verdict, 'verdict replay')
    # partial spot (AB/T)
    p21 = partial_sp(F[('f2', 'AB')],
                     T['AB'], F[('f1', 'AB')])
    chk(p21 == float(z['PSP_F2G1_T_AB']),
        'partial sp bit T_AB')
    p21p = partial_sp_perm_p(
        F[('f2', 'AB')], T['AB'],
        F[('f1', 'AB')], 20000, SEED)
    chk(p21p == float(z['PSP_F2G1P_T_AB']),
        'partial p bit T_AB')
    # result cross mirrors npz
    chk(cr is not None
        and cr['spec_class'] == spec_cls,
        'result cross spec_class')
    chk(abs(cr['g_ds_min_sp']
            - float(z['GDS_MIN_SP']))
        < 1e-15, 'result g_ds_min_sp')

# ==== ledger ====
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
meas = [m for m in led['measurements']
        if isinstance(m, dict)
        and m.get('phase') == 3085]
chk(len(meas) == 1, 'ledger meas3085')
chk(len(led['measurements']) == 224,
    'ledger n=224')
l14 = [l for l in led['linkage']
       if isinstance(l, dict)
       and l.get('link_id')
       == 'L14_readout_spectrum_cross_model']
chk(len(l14) == 1, 'ledger L14 exists')
c3085 = [c for c in l14[0]['connects']
         if isinstance(c, dict)
         and c.get('phase') == 3085]
chk(len(c3085) == 1, 'L14 connects 3085')
if 'ledger_sha256_8' in led:
    saved = led.pop('ledger_sha256_8')
    blob = json.dumps(led, sort_keys=True,
                      ensure_ascii=False)
    chk(hashlib.sha256(blob.encode(
        'utf-8')).hexdigest()[:8] == saved,
        'ledger sha replay')
    led['ledger_sha256_8'] = saved

# ==== documents ====
memo = io.open(MEMO, encoding='utf-8').read()
chk('## Phase 3085:' in memo,
    'MEMO has Phase 3085')
if '## Phase 3085:' in memo:
    tail = memo[memo.rindex('## Phase 3085:'):]
    chk('L34' in tail
        and 'CS top3' in tail,
        'MEMO 3085 tail content')
else:
    chk(False, 'MEMO 3085 tail content')
aud = io.open(AUDIT,
              encoding='utf-8').read()
chk('## 四十七、3085' in aud,
    'audit 47 present')
wl = io.open(WLOG, encoding='utf-8').read()
chk('Phase 3085' in wl, 'wlog 3085')
mem = io.open(MEMW, encoding='utf-8').read()
chk('max=3085' in mem, 'MEMORY max=3085')

# ==== report ====
rep = ['VERIFY_OK %d/%d'
       % (len(ok), len(ok) + len(fail))]
if fail:
    rep.append('FAILED: ' + '; '.join(fail))
io.open(REPF, 'w', encoding='utf-8').write(
    '\n'.join(rep) + '\n')
print('VERIFY_DONE %d/%d'
      % (len(ok), len(ok) + len(fail)))
