# -*- coding: utf-8 -*-
"""Phase 3079 independent verify: recompute anchors
and ALL statistics from frozen npz files (no
forwards), replay main permutation p values and the
verdict, verify Ledger/MEMO/audit/wlog/MEMORY
on-disk state."""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
RDIR = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
R = RDIR + (r'\phase3079'
            r'\omega_p76_migration_lock')
P76 = RDIR + (r'\phase3076'
              r'\omega_p73_cross_prompt_family')
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
z9 = np.load(R + (r'\omega_p76_migration_lock'
                  r'.npz'))
z76 = np.load(P76 + (r'\omega_p73_cross_prompt_'
                     r'family.npz'))
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


def perm_p(a, b, n_perm=20000, seed=3079):
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


# ---- E1 anchor replay (no forwards) ----
ck('a1_A_S_vs_3074',
   float(np.max(np.abs(
       z76['A_S_A'].astype(np.float64)
       - z74['A_S'].astype(np.float64)))) == 0.0)
r34_71 = np.array(json.load(io.open(
    RDIR + (r'\phase3071'
            r'\omega_p68_attn_head_decomp'
            r'\result.json'),
    encoding='utf-8'))['stats']['head']['r34'],
    dtype=np.float64)
ck('a2_R1A_vs_3071',
   float(np.max(np.abs(
       z76['R1_ALL32_A'] - r34_71))) == 0.0)
for fa, fb in CPAIRS:
    key = fa + fb
    ck('a3_SP_R1_' + key,
       spearman76(
           z76['R1_ALL32_' + fa].astype(np.float64),
           z76['R1_ALL32_' + fb]
           .astype(np.float64))
       == float(z76['SP_R1_' + key]))
ck('a4_MASKS_ident',
   float(np.max(np.abs(
       z76['MASKS_A'].astype(np.float64)
       - z76['MASKS_B'].astype(np.float64))))
   == 0.0
   and float(np.max(np.abs(
       z76['MASKS_A'].astype(np.float64)
       - z76['MASKS_C'].astype(np.float64))))
   == 0.0)
for fk in FKEYS:
    R1f = (z76['R1_ALL32_' + fk]
           .astype(np.float64))
    medc = float(z76['MED_C_34_' + fk])
    ck('a5_R1_replay_' + fk,
       float(np.max(np.abs(
           np.median(
               z76['CS1H_' + fk]
               .astype(np.float64), axis=1)
           - medc - R1f))) == 0.0)
    ck('a6_MED_replay_' + fk,
       float(np.max(np.abs(
           np.median(
               z76['DAH34_' + fk]
               .astype(np.float64), axis=0)
           - z76['DAH34_MED_' + fk]
           .astype(np.float64)))) == 0.0)
    ck('a7_top8_replay_' + fk,
       np.array_equal(
           np.argsort(R1f)[:8].astype(np.int64),
           z76['TOP8_' + fk].astype(np.int64)))

# ---- frozen data ----
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

# ---- E2 family level replay ----
sp_re = {}
for fa, fb in CPAIRS:
    sp_re[fa + fb] = spearman76(R1[fa], R1[fb])
    key = fa + fb
    ck('E2_SP_AS_' + key,
       spearman(A_S[fa], A_S[fb])
       == float(z9['SP_AS_' + key]))
    ck('E2_mig_' + key,
       sp_re[key] == float(z9['SP_R1_REPLAY_'
                               + key]))
sp_fam = spearman(
    [float(z9['SP_AS_' + k])
     for k in ('AB', 'AC', 'BC')],
    [float(z9['SP_R1_REPLAY_' + k])
     for k in ('AB', 'AC', 'BC')])
ck('E2_SP_FAM_AS_MIG',
   sp_fam == float(z9['SP_FAM_AS_MIG']))

# ---- E3 responses + predictors bit replay ----
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
    ck('T_bit_' + key,
       float(np.max(np.abs(
           T[key] - z9['T_' + key]))) == 0.0)
    ck('U_bit_' + key,
       float(np.max(np.abs(
           U[key] - z9['U_' + key]))) == 0.0)
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
    for fid in F:
        ck('%s_bit_%s' % (FUP[fid], key),
           float(np.max(np.abs(
               F[fid][key]
               - z9['%s_%s' % (FUP[fid],
                               key)]))) == 0.0)

# ---- E3 spearman bit replay (30 tests) ----
RESP = {'T': T, 'U': U}
for fid in F:
    for rn in ('T', 'U'):
        for key in ('AB', 'AC', 'BC'):
            tk = '%s~%s_%s' % (fid, rn, key)
            zkey = ('E3_%s_%s_%s'
                    % (FUP[fid], rn, key))
            ck('E3sp_' + tk,
               spearman(F[fid][key],
                        RESP[rn][key])
               == float(z9[zkey]))

# ---- main permutation p replay (f1, 6) ----
for rn in ('T', 'U'):
    for key in ('AB', 'AC', 'BC'):
        tk = 'f1_sTT~%s_%s' % (rn, key)
        p_re = perm_p(F['f1_sTT'][key],
                      RESP[rn][key])
        p_ref = float(
            res['stats']['e3'][tk]['p'])
        ck('perm_' + tk,
           abs(p_re - p_ref) < 1e-12)

# ---- E4 replay ----
for key in ('AB', 'AC', 'BC'):
    ck('E4_SP_UT_' + key,
       spearman(U[key], T[key])
       == float(z9['SP_UT_' + key]))

# ---- verdict replay ----
best = None
for key in ('AB', 'AC', 'BC'):
    for rn in ('T', 'U'):
        e = res['stats']['e3'][
            'f1_sTT~%s_%s' % (rn, key)]
        cand = (abs(e['sp']), e['sp'],
                e['p'], rn, key)
        if best is None \
                or cand[0] > best[0]:
            best = cand
g1 = bool(best[0] >= 0.5 and best[2] < 0.05)
ck('best_is_TAC',
   best[3] == 'T' and best[4] == 'AC'
   and abs(best[1] - 0.5060869565217392)
   < 1e-15)
ck('G1_replay', g1 is True)
ck('verdict', res['verdict']
   == 'migration_tt_locked')
ck('verdict_npz', str(z9['VERDICT'])
   == 'migration_tt_locked')
ck('forwards0', res['forwards'] == 0
   and int(z9['N_PERM']) == 20000)
ck('max_abs_all',
   abs(res['gates']['max_abs_sp_all']
       - 0.7713043478260869) < 1e-15)

# ---- seal ----
for k, p in (('npz_sha256_8',
              R + (r'\omega_p76_migration_lock'
                   r'.npz')),
             ('result_sha256_8',
              R + r'\result.json'),
             ('exec_sha256_8',
              R + r'\execution.json'),
             ('script_sha256_8',
              ROOT + (r'\tests\glm5'
                      r'\phase3079_omega_p76_'
                      r'migration_lock.py'))):
    h = hashlib.sha256(io.open(
        p, 'rb').read()).hexdigest()[:8]
    ck(k, h == seal[k])

# ---- ledger / memo / audit / wlog / memory ----
led = json.load(io.open(LEDGER,
                        encoding='utf-8'))
m3079 = [m for m in led['measurements']
         if isinstance(m, dict)
         and m.get('phase') == 3079]
ck('ledger_n', len(led['measurements']) == 218)
ck('ledger_meas', len(m3079) == 1
   and m3079[0]['verdict']
   == 'migration_tt_locked')
l14 = [l for l in led['linkage']
       if isinstance(l, dict)
       and l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
ck('ledger_l14', len(l14['connects']) == 186
   and l14['connects'][-1]['phase'] == 3079)
led2 = json.loads(json.dumps(led))
led2.pop('ledger_sha256_8')
blob = json.dumps(led2, sort_keys=True,
                  ensure_ascii=False)
ck('ledger_sha', hashlib.sha256(
    blob.encode('utf-8')).hexdigest()[:8]
   == led['ledger_sha256_8'])
memo = io.open(MEMO, encoding='utf-8').read()
ck('memo_phase3079', '## Phase 3079:' in memo)
ck('memo_verdict', 'migration_tt_locked'
   in memo)
ck('audit_41', '## 四十一、3079'
   in io.open(AUDIT, encoding='utf-8').read())
ck('wlog', 'Phase 3079' in io.open(
    WLOG, encoding='utf-8').read())
ck('memory_max3079', 'max=3079' in io.open(
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
