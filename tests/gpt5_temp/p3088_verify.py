# -*- coding: utf-8 -*-
"""Phase 3088 independent verify: seal hash recompute,
npz scalar asserts, spearman + permutation bit-level
recompute from stored unit keys, ledger/memo/audit/
wlog/memory checks, artifact placement (incl. smoke
dir + reverse check).  Output -> p3088_verify_out.txt"""
import hashlib
import io
import json
import os

import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
BASE = (ROOT + r'\tests\glm5\result'
        r'\rdc_query_construction_20260913')
R = BASE + r'\phase3088\omega_p86_continuum_n12'
F86 = (ROOT + r'\tests\glm5'
       r'\phase3086_omega_p83_continuum_test.py')
F88 = (ROOT + r'\tests\glm5'
       r'\phase3088_omega_p86_continuum_n12.py')
LEDGER = (ROOT + r'\research\gpt5\atlas'
          r'\atlas_ledger.json')
MEMO = (ROOT + r'\research\gpt5\docs'
        r'\AGI_GPT5_MEMO.md')
AUDIT = (ROOT + r'\research\gpt5\docs'
         r'\hdmcc_knowledge_map_review_'
         r'20260921.md')
WLOG = (ROOT + r'\.workbuddy\memory'
        r'\2026-09-22.md')
MEMW = (ROOT + r'\.workbuddy\memory\MEMORY.md')
OUTF = (ROOT + r'\tests\gpt5_temp'
        r'\p3088_verify_out.txt')

o = []
n_pass = 0
n_fail = 0


def chk(label, cond):
    global n_pass, n_fail
    if cond:
        n_pass += 1
        o.append('PASS %s' % label)
    else:
        n_fail += 1
        o.append('FAIL %s' % label)


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


# ---- 1. seal hashes ----
seal = json.load(io.open(
    R + r'\seal.json', encoding='utf-8'))
chk('seal npz8', seal['npz_sha256_8']
    == sha8(R + r'\omega_p86_continuum_n12.npz'))
chk('seal result8', seal['result_sha256_8']
    == sha8(R + r'\result.json'))
chk('seal exec8', seal['exec_sha256_8']
    == sha8(R + r'\execution.json'))
chk('seal script8', seal['script_sha256_8']
    == sha8(F88))
chk('seal verdict',
    seal['verdict']
    == 'cross_architecture_confirmed')
chk('seal setup_ok', seal['setup_ok'] is True)

# ---- 2. npz scalars ----
z = np.load(R + r'\omega_p86_continuum_n12.npz',
            allow_pickle=False)
chk('npz VERDICT', str(z['VERDICT'])
    == 'cross_architecture_confirmed')
chk('npz SMOKE False', bool(z['SMOKE']) is False)
chk('npz FORWARDS 0', int(z['FORWARDS']) == 0)
chk('npz SEED 3088', int(z['SEED']) == 3088)
chk('npz N_PERM 20000', int(z['N_PERM']) == 20000)
chk('npz N_UNITS 12', int(z['N_UNITS']) == 12)
chk('npz SETUP_OK', bool(z['SETUP_OK']))
chk('npz ANCH_A1', bool(z['ANCH_A1']))
chk('npz ANCH_A2', bool(z['ANCH_A2']))
chk('npz ANCH_A3', bool(z['ANCH_A3']))
chk('npz SHA87', str(z['SHA87']) == '32d3bf08')

# ---- 3. bit-level recompute from unit keys ----
# NOTE: must replicate the main script's
# sorted(units) enumeration order -- the
# 3077-series manual spearman uses ordinal
# ranks (no tie averaging), so with tied
# s_lo values rho depends on the unit
# order (n=12: 0.8951 in tag order vs
# 0.9021 in sorted order).
tags = ['M3B_AB', 'M3B_AC', 'M3B_BC',
        'M4B_AB', 'M4B_AC', 'M4B_BC',
        'MDS7B_AB', 'MDS7B_AC', 'MDS7B_BC',
        'MGLM4_AB', 'MGLM4_AC', 'MGLM4_BC']
s_lo = np.array([float(z['S_LO_' + t])
                 for t in tags])
s_mean = np.array([float(z['S_MEAN_' + t])
                   for t in tags])
T_med = np.array([float(z['T_MED_' + t])
                  for t in tags])
U_med = np.array([float(z['U_MED_' + t])
                  for t in tags])
MIG = np.array([float(z['MIG_' + t])
                for t in tags])
chk('units 12 keys', len(tags) == 12
    and all(('S_LO_' + t) in z.files
            for t in tags))
rt = spearman(s_lo, T_med)
ru = spearman(s_lo, U_med)
rm = spearman(s_lo, MIG)
chk('rho_T bit', abs(rt - float(z['RHO_T']))
    < 1e-12)
chk('rho_U bit', abs(ru - float(z['RHO_U']))
    < 1e-12)
chk('rho_MIG bit', abs(rm - float(z['RHO_MIG']))
    < 1e-12)
pt = perm_p(s_lo, T_med, 20000, 3088)
pu = perm_p(s_lo, U_med, 20000, 3088)
pm = perm_p(s_lo, MIG, 20000, 3088)
chk('p_T bit', abs(pt - float(z['P_T'])) < 1e-12)
chk('p_U bit', abs(pu - float(z['P_U'])) < 1e-12)
chk('p_MIG bit', abs(pm - float(z['P_MIG']))
    < 1e-12)
chk('p_T significant', pt < 0.05 and rt > 0)
smt = spearman(s_mean, T_med)
chk('sp_mean_T bit',
    abs(smt - float(z['SP_MEAN_T'])) < 1e-12)

# ---- 4. qwen-only subset vs 3086 ----
idx_q = [i for i, t in enumerate(tags)
         if not t.startswith('MGLM4')]
r_q = spearman(s_lo[idx_q], T_med[idx_q])
chk('qwen-only n=9 = +0.8667',
    abs(r_q - 0.8666666666666667) < 1e-12)

# ---- 5. units vs result.json ----
res = json.load(io.open(
    R + r'\result.json', encoding='utf-8'))
ru_ok = True
for i, t in enumerate(tags):
    k = t.replace('M4B', '4B') \
         .replace('MDS7B', 'DS7B') \
         .replace('M3B', '3B') \
         .replace('MGLM4', 'GLM4')
    u = res['stats']['units'][k]
    ru_ok = ru_ok and \
        abs(u['s_lo'] - s_lo[i]) < 1e-12 and \
        abs(u['T_med'] - T_med[i]) < 1e-12 and \
        abs(u['U_med'] - U_med[i]) < 1e-12 and \
        abs(u['MIG'] - MIG[i]) < 1e-12
chk('result.json units match npz', ru_ok)
chk('result verdict', res['verdict']
    == 'cross_architecture_confirmed')
chk('result n_units',
    res['stats']['n_units'] == 12)
chk('result forwards 0',
    res['forwards'] == 0)

# ---- 6. execution.json ----
exe = json.load(io.open(
    R + r'\execution.json', encoding='utf-8'))
chk('exec phase/name',
    exe['phase'] == 3088
    and exe['name'] == 'omega_p86_continuum_n12')
chk('exec created matches seal',
    exe['created'] == seal['created'])
chk('exec smoke False', exe['smoke'] is False)
chk('exec prereg verdict tree',
    'cross_architecture_confirmed'
    in exe['prereg']['verdict']
    and 'qwen_lineage_internal'
    in exe['prereg']['verdict'])

# ---- 7. stats block identity 3086 vs 3088 ----
s86 = io.open(F86, encoding='utf-8').read()
s88 = io.open(F88, encoding='utf-8').read()
b86 = s86[s86.index('def spearman(a, b):')
          :s86.index('def sha8(path):')]
b88 = s88[s88.index('def spearman(a, b):')
          :s88.index('def sha8(path):')]
chk('stats block bit-identical to 3086',
    b86 == b88)

# ---- 8. ledger ----
led = json.load(io.open(
    LEDGER, encoding='utf-8'))
chk('ledger n=227',
    len(led['measurements']) == 227)
l14 = [l for l in led['linkage']
       if l.get('link_id')
       == 'L14_readout_spectrum_cross_model'][0]
chk('ledger L14=195',
    len(l14['connects']) == 195)
m88 = [m for m in led['measurements']
       if isinstance(m, dict)
       and m.get('phase') == 3088]
chk('ledger meas3088 present',
    len(m88) == 1
    and m88[0]['verdict']
    == 'cross_architecture_confirmed')
chk('ledger meas3088 claim values',
    '0.9021' in m88[0]['claim']
    and '32d3bf08' in m88[0]['claim']
    and '+0.8667' in m88[0]['claim'])
saved = dict(led)
ref = saved.pop('ledger_sha256_8')
blob = json.dumps(saved, sort_keys=True,
                  ensure_ascii=False)
chk('ledger sha recompute',
    hashlib.sha256(
        blob.encode('utf-8')).hexdigest()[:8]
    == ref)

# ---- 9. memo ----
memo = io.open(MEMO, encoding='utf-8').read()
i87 = memo.rfind('## Phase 3087:')
i88 = memo.rfind('## Phase 3088:')
chk('memo 3088 after 3087',
    i87 > 0 and i88 > i87)
chk('memo 3088 title timestamp',
    ('[%s]' % exe['created'])
    in memo[i88:i88 + 300])
chk('memo 3088 key values',
    '+0.9021' in memo[i88:]
    and 'cross_architecture_confirmed'
    in memo[i88:]
    and '0.6992' in memo[i88:])

# ---- 10. audit / wlog / memory ----
aud = io.open(AUDIT, encoding='utf-8').read()
chk('audit 50 section',
    '## 五十、3088' in aud
    and 'cross_architecture_confirmed' in aud)
wl = io.open(WLOG, encoding='utf-8').read()
chk('wlog 3088 line',
    'Phase 3088' in wl
    and 'cross_architecture_confirmed' in wl)
mem = io.open(MEMW, encoding='utf-8').read()
chk('memory max=3088', 'max=3088' in mem)

# ---- 11. artifacts placement ----
need = ['omega_p86_continuum_n12.npz',
        'run_log.txt', 'execution.json',
        'result.json', 'seal.json',
        'closeout_log.txt']
chk('authority artifacts present',
    all(os.path.exists(
        os.path.join(R, f)) for f in need))
smoke_npz = os.path.join(
    R, 'smoke', 'omega_p86_continuum_n12.npz')
chk('smoke npz in correct smoke dir',
    os.path.exists(smoke_npz))
stale = [f for f in os.listdir(
    os.path.join(BASE, 'phase3087',
                 'omega_p85_glm4_l37_full_'
                 'arbitration'))
    if f.startswith('omega_p86')]
chk('no misplaced 3088 artifacts in 3087',
    len(stale) == 0)

o.append('VERIFY %d/%d PASS'
         % (n_pass, n_pass + n_fail))
io.open(OUTF, 'w', encoding='utf-8').write(
    '\n'.join(o) + '\n')
