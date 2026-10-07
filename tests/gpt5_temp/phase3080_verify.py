# -*- coding: utf-8 -*-
"""Phase 3080 independent verify: recompute anchors
and ALL statistics from frozen npz files (no
forwards), replay E2 permutation p values, partial
spearman perms, the verdict, and verify
Ledger/MEMO/audit/wlog/MEMORY on-disk state."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
R = RDIR + (r'\phase3080'
            r'\omega_p77_ab_anatomy')
P76 = RDIR + (r'\phase3076'
              r'\omega_p73_cross_prompt_family')
P79 = RDIR + (r'\phase3079'
              r'\omega_p76_migration_lock')
P78 = RDIR + (r'\phase3078'
              r'\omega_p75_routing_timing')
P74 = RDIR + (r'\phase3074'
              r'\omega_p71_capacity_law')
LEDGER = ROOT + (r'\research\gpt5\atlas'
                 r'\atlas_ledger.json')
MEMO = ROOT + (r'\research\gpt5\docs'
               r'\AGI_GPT5_MEMO.md')
AUDIT = ROOT + (r'\research\gpt5\docs'
                r'\hdmcc_knowledge_map_review_'
                r'20260921.md')
WLOG = ROOT + (r'\.workbuddy\memory'
               r'\2026-09-21.md')
MEMW = ROOT + (r'\.workbuddy\memory\MEMORY.md')
chk = []


def ck(name, cond):
    chk.append((name, bool(cond)))


res = json.load(io.open(
    R + r'\result.json', encoding='utf-8'))
seal = json.load(io.open(
    R + r'\seal.json', encoding='utf-8'))
z80 = np.load(R + (r'\omega_p77_ab_anatomy'
                   r'.npz'))
z76 = np.load(P76 + (r'\omega_p73_cross_prompt_'
                     r'family.npz'))
z79 = np.load(P79 + (r'\omega_p76_migration_'
                     r'lock.npz'))
z78 = np.load(P78 + (r'\omega_p75_routing_'
                     r'timing.npz'))
z74 = np.load(P74 + (r'\omega_p71_capacity_law'
                     r'.npz'))

FKEYS = ('A', 'B', 'C')
CPAIRS = (('A', 'B'), ('A', 'C'), ('B', 'C'))


def spearman(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
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


def spearman76(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ra = np.empty(len(a))
    rb = np.empty(len(b))
    ra[np.argsort(a, kind='stable')] = \
        np.arange(len(a), dtype=np.float64)
    rb[np.argsort(b, kind='stable')] = \
        np.arange(len(b), dtype=np.float64)
    if ra.std() == 0 or rb.std() == 0:
        return float('nan')
    return float(np.corrcoef(ra, rb)[0, 1])


def cosv(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b) / (na * nb)


def perm_p(a, b, n_perm=20000, seed=3080):
    rng = np.random.default_rng(seed)
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    obs = abs(spearman(a, b))
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
    den = np.sqrt((1.0 - r_xz * r_xz)
                  * (1.0 - r_yz * r_yz))
    if den == 0:
        return 0.0
    return float((r_xy - r_xz * r_yz) / den)


def partial_sp_perm_p(x, y, z,
                      n_perm=20000,
                      seed=3080):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    z = np.asarray(z, np.float64)
    obs = abs(partial_sp(x, y, z))
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


# ---- frozen data + responses/predictors ----
A_S = {f: z76['A_S_' + f].astype(np.float64)
       for f in FKEYS}
CS = {f: z76['CS_' + f].astype(np.float64)
      for f in FKEYS}
CS1H = {f: z76['CS1H_' + f].astype(np.float64)
        for f in FKEYS}
R1 = {f: z76['R1_ALL32_' + f]
      .astype(np.float64) for f in FKEYS}
TT64 = {f: z76['TT_' + f].astype(np.float64)
        for f in FKEYS}
LG64 = {f: z76['LG_' + f].astype(np.float64)
        for f in FKEYS}
CIDX = np.array([k // 8 + 1
                 for k in range(24)])
BIDX = np.array([k % 8
                 for k in range(24)])
PREF_K = BIDX * 4 + CIDX
BASE_K = BIDX * 4

T = {}
U = {}
F = {fid: {} for fid in
     ('f1_sTT', 'f2_cTT', 'f3_sLGp',
      'f4_sLGb', 'f5_amp')}
FUP = {'f1_sTT': 'F1_STT', 'f2_cTT': 'F2_CTT',
       'f3_sLGp': 'F3_SLGP',
       'f4_sLGb': 'F4_SLGB', 'f5_amp': 'F5_AMP'}
for fa, fb in CPAIRS:
    key = fa + fb
    T[key] = np.array([
        spearman(CS1H[fa][:, k],
                 CS1H[fb][:, k])
        for k in range(24)])
    U[key] = np.array([
        spearman(CS[fa][:, k], CS[fb][:, k])
        for k in range(24)])
    F['f1_sTT'][key] = np.array([
        spearman(TT64[fa][k], TT64[fb][k])
        for k in range(24)])
    F['f2_cTT'][key] = np.array([
        cosv(TT64[fa][k], TT64[fb][k])
        for k in range(24)])
    F['f3_sLGp'][key] = np.array([
        spearman(LG64[fa][PREF_K[k]],
                 LG64[fb][PREF_K[k]])
        for k in range(24)])
    F['f4_sLGb'][key] = np.array([
        spearman(LG64[fa][BASE_K[k]],
                 LG64[fb][BASE_K[k]])
        for k in range(24)])
    n_fa = np.linalg.norm(TT64[fa], axis=1)
    n_fb = np.linalg.norm(TT64[fb], axis=1)
    F['f5_amp'][key] = np.minimum(
        n_fa, n_fb) / np.maximum(n_fa, n_fb)

# ---- a1: cross-npz 25-array replay ----
d1 = 0.0
for key in ('AB', 'AC', 'BC'):
    d1 = max(d1, float(np.max(np.abs(
        T[key] - z79['T_' + key]))))
    d1 = max(d1, float(np.max(np.abs(
        U[key] - z79['U_' + key]))))
    for fid in F:
        d1 = max(d1, float(np.max(np.abs(
            F[fid][key]
            - z79['%s_%s' % (FUP[fid],
                             key)]))))
ck('a1_cross_npz_25arrays', d1 == 0.0)

# ---- a2: SP_R1 spearman76 replay ----
sp_re = {}
d2 = 0.0
for fa, fb in CPAIRS:
    sp_re[fa + fb] = spearman76(R1[fa],
                                R1[fb])
    d2 = max(d2, abs(sp_re[fa + fb]
                     - float(z76['SP_R1_'
                                 + fa + fb])))
ck('a2_SP_R1_replay', d2 == 0.0)

# ---- a3: R1_A vs 3071 r34 ----
r34_71 = np.array(json.load(io.open(
    RDIR + (r'\phase3071'
            r'\omega_p68_attn_head_decomp'
            r'\result.json'),
    encoding='utf-8'))['stats']['head']['r34'],
    dtype=np.float64)
ck('a3_R1A_vs_3071',
   float(np.max(np.abs(R1['A'] - r34_71)))
   == 0.0)

# ---- a4: DM_L34 cross vs 3078 result ----
res78 = json.load(io.open(
    P78 + r'\result.json', encoding='utf-8'))
re8 = [spearman(np.abs(z78['DM_L34_' + fa]),
                np.abs(z78['DM_L34_' + fb]))
       for fa, fb in CPAIRS]
ck('a4_DML34_cross_replay',
   float(np.max(np.abs(
       np.array(re8)
       - np.array(res78['stats']
                  ['sp_cross'][7])))) == 0.0)

# ---- a5: A_S_A vs 3074 ----
ck('a5_A_S_vs_3074',
   float(np.max(np.abs(
       A_S['A'] - z74['A_S']
       .astype(np.float64)))) == 0.0)

# ---- E2 f2 six tests: sp bit + perm p ----
E2 = {}
for rn, RESP in (('T', T), ('U', U)):
    for key in ('AB', 'AC', 'BC'):
        tk = '%s_%s' % (rn, key)
        s_ = spearman(F['f2_cTT'][key],
                      RESP[key])
        ck('E2sp_' + tk,
           s_ == float(z80['F2SP_' + tk]))
        p_ = perm_p(F['f2_cTT'][key],
                    RESP[key])
        ck('E2p_' + tk,
           abs(p_ - float(z80['F2P_' + tk]))
           < 1e-12)
        E2[tk] = {'sp': s_, 'p': p_}
cnt_pos = sum(1 for v in E2.values()
              if v['sp'] > 0
              and v['p'] < 0.05)
min_sp2 = min(v['sp'] for v in E2.values())
ck('G2_replay',
   bool(cnt_pos >= 4 and min_sp2 > 0))

# ---- E3.2 f1-f2 coupling ----
for key in ('AB', 'AC', 'BC'):
    ck('F1F2_' + key,
       spearman(F['f1_sTT'][key],
                F['f2_cTT'][key])
       == float(z80['SP_F1F2_' + key]))

# ---- E3.3 partial spearman + perm p ----
for key in ('AB', 'AC'):
    for rn, RESP in (('T', T), ('U', U)):
        tag = '%s_%s' % (rn, key)
        p21 = partial_sp(F['f2_cTT'][key],
                         RESP[key],
                         F['f1_sTT'][key])
        ck('PSP_f2g1_' + tag,
           abs(p21
               - float(z80['PSP_F2G1_'
                           + tag.upper()]))
           < 1e-15)
        ck('PSPp_f2g1_' + tag,
           abs(partial_sp_perm_p(
               F['f2_cTT'][key], RESP[key],
               F['f1_sTT'][key])
               - float(z80['PSP_F2G1P_'
                           + tag.upper()]))
           < 1e-12)
        p52 = partial_sp(F['f5_amp'][key],
                         RESP[key],
                         F['f2_cTT'][key])
        ck('PSP_f5g2_' + tag,
           abs(p52
               - float(z80['PSP_F5G2_'
                           + tag.upper()]))
           < 1e-15)

# ---- verdict replay ----
best = None
for tk, v in E2.items():
    cand = (abs(v['sp']), v['sp'], v['p'],
            tk)
    if best is None or tk == 'T_AB':
        best = cand
f2_tab_p = E2['T_AB']['p']
g2 = bool(cnt_pos >= 4 and min_sp2 > 0)
if g2 and f2_tab_p < 0.05:
    vexp = 'ab_cos_locked'
elif g2:
    vexp = 'ab_multi_source'
else:
    vexp = 'ab_unexplained'
ck('verdict', res['verdict'] == vexp
   == 'ab_cos_locked')
ck('verdict_npz', str(z80['VERDICT'])
   == 'ab_cos_locked')
ck('forwards0', res['forwards'] == 0
   and int(z80['N_PERM']) == 20000)
g = res['gates']
ck('gates', g['G2'] is True
   and g['count_sig_pos'] == 5
   and abs(g['min_sp']
           - 0.3591304347826087) < 1e-15
   and abs(g['stouffer_z']
           - 8.091847859848315) < 1e-12
   and abs(g['f2_TAB_sp']
           - 0.5452173913043479) < 1e-15
   and g['f2_TAB_p'] < 0.05)
ck('family_orientation',
   abs(res['stats']['family']['sp_dmmig']
       - 0.5) < 1e-12
   and abs(res['stats']['family']
           ['sp_asmig'] + 1.0) < 1e-12)

# ---- seal ----
for k, p in (('npz_sha256_8',
              R + (r'\omega_p77_ab_anatomy'
                   r'.npz')),
             ('result_sha256_8',
              R + r'\result.json'),
             ('exec_sha256_8',
              R + r'\execution.json'),
             ('script_sha256_8',
              ROOT + (r'\tests\glm5'
                      r'\phase3080_omega_p77_'
                      r'ab_anatomy.py'))):
    h = hashlib.sha256(io.open(
        p, 'rb').read()).hexdigest()[:8]
    ck(k, h == seal[k])

# ---- ledger / memo / audit / wlog / memory ----
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
m3080 = [m for m in led['measurements']
         if isinstance(m, dict)
         and m.get('phase') == 3080]
ck('ledger_n', len(led['measurements']) == 219)
ck('ledger_meas', len(m3080) == 1
   and m3080[0]['verdict']
   == 'ab_cos_locked')
l14 = [l for l in led['linkage']
       if isinstance(l, dict)
       and l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
ck('ledger_l14', len(l14['connects']) == 187
   and l14['connects'][-1]['phase'] == 3080)
led2 = json.loads(json.dumps(led))
led2.pop('ledger_sha256_8')
blob = json.dumps(led2, sort_keys=True,
                  ensure_ascii=False)
ck('ledger_sha', hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
   == led['ledger_sha256_8'])
memo = io.open(MEMO, encoding='utf-8').read()
ck('memo_phase3080', '## Phase 3080:' in memo)
ck('memo_verdict', 'ab_cos_locked' in memo)
ck('audit_42', '## 四十二、3080'
   in io.open(AUDIT, encoding='utf-8').read())
ck('wlog', 'Phase 3080' in io.open(
    WLOG, encoding='utf-8').read())
ck('memory_max3080', 'max=3080' in io.open(
    MEMW, encoding='utf-8').read())

# ---- report ----
bad = [n for n, okc in chk if not okc]
io.open(R + r'\verify_log.txt', 'w',
        encoding='utf-8').write(
    '%d checks, %d failed\n'
    % (len(chk), len(bad))
    + ('\n'.join('FAIL ' + n for n in bad)
       if bad else 'ALL OK') + '\n')
print('VERIFY_%s (%d/%d)'
      % ('OK' if not bad else 'FAIL',
         len(chk) - len(bad), len(chk)))
