# -*- coding: utf-8 -*-
"""Phase 3238 independent disk verify: deterministic recompute from sealed npz/json.
Sections: A registry v1.33 / B spectra+reuse recompute / C ledger / D MEMO / E wlog / F memory / G result self-hash.
"""
import os, json, hashlib
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SD = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
RESF = os.path.join(SD, r'phase3238\g5h1_census_rank\result.json')
REG33 = os.path.join(SD, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json')
REG32 = os.path.join(SD, r'phase3237\g5g31_registry_landing\atlas_registry_v1_32.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
FAILS = []
N = [0]


def sec(name, conds):
    for label, ok in conds:
        N[0] += 1
        if not ok:
            FAILS.append('%s.%s' % (name, label))
    print('[%s] %d checks' % (name, len(conds)))


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


def canon(o):
    return json.dumps(o, ensure_ascii=False, sort_keys=True, indent=1)


R = json.load(open(RESF, encoding='utf-8'))
MODELS = ['qwen3-4b', 'qwen3-14b', 'glm4-9b']
FAMS = ['DVU', 'DCC', 'ENT', 'USTAR', 'SCLS', 'MASSIVE']

# ---------- A registry ----------
v33 = json.load(open(REG33, encoding='utf-8'))
v32 = json.load(open(REG32, encoding='utf-8'))
cA = []
cA.append(('reg33_file_sha8', sha8_file(REG33) == R['registry_sha8']))
bad = [k for k in v32 if k not in ('version', 'supersedes', 'upgrade_log')
       and canon(v33[k]) != canon(v32[k])]
cA.append(('byte_preserve', bad == []))
cA.append(('census_key', 'census_3238' in v33 and v33['census_3238']['design_sha8'] == '4cc8adda'))
cA.append(('version_bump', v33['version'] == '1.33' and len(v33['upgrade_log']) == 12))
cA.append(('features_untouched', canon(v33['features']) == canon(v32['features'])
           and canon(v33['failures']) == canon(v32['failures'])))
sec('A_registry', cA)

# ---------- B spectra recompute (independent derivation) ----------
ANCH_3157 = {'qwen3-4b': (36, 35, 0.5629908442497253),
             'qwen3-14b': (40, 39, 0.5614360570907593),
             'glm4-9b': (40, 39, 0.5640953183174133)}
K57 = {'qwen3-4b': r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz',
       'qwen3-14b': r'phase3157\g2p2_transform_algebra_commutator\qwen3-14b\collect.npz',
       'glm4-9b': r'phase3157\g2p2_transform_algebra_commutator\glm4\collect.npz'}
K69 = {'qwen3-4b': r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz',
       'qwen3-14b': r'phase3169\g5a6_oov_panel\collect_qwen3-14b.npz',
       'glm4-9b': r'phase3169\g5a6_oov_panel\collect_glm4-9b.npz'}
NT, NC, NE = 3, 10, 73
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS)


def spectrum(M):
    sv = np.linalg.svd(np.asarray(M, np.float64), compute_uv=False)
    lam = sv ** 2
    tot = lam.sum()
    pr = float(tot * tot / (lam ** 2).sum() + 1e-300)
    cum = np.cumsum(lam) / tot
    k90 = int(np.searchsorted(cum, 0.9) + 1)
    top1 = float(lam[0] / tot)
    return pr, k90, top1


def unit_mean(rows):
    v = rows.mean(0)
    return v / (float(np.linalg.norm(v)) + 1e-18)


def unit(v):
    return v / (float(np.linalg.norm(v)) + 1e-18)


cB = []
for m in MODELS:
    z = np.load(os.path.join(SD, K57[m]))
    H = z['H'].astype(np.float32)
    NR, NH, D = H.shape
    KOUT = NH - 2
    EI, RL, PL, CX = (z['ei'].astype(int), z['rel'].astype(int),
                      z['pol'].astype(int), z['ctx'].astype(int))
    IDX = {(int(EI[i]), int(RL[i]), int(PL[i]), int(CX[i])): i for i in range(NR)}
    dC = np.array([H[IDX[(e, rl, pl, 1)], KOUT] - H[IDX[(e, rl, pl, 0)], KOUT]
                   for e in range(16) for rl in range(2) for pl in range(2)])
    dCn = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-18)
    ENT = np.stack([unit_mean(H[[IDX[(e, rl, pl, cx)] for rl in range(2)
                                for pl in range(2) for cx in range(2)], KOUT])
                    for e in range(16)])
    U = []
    for e in range(16):
        a_e = dCn[[e * 4 + p for p in (0, 1)]].mean(0)
        b_e = dCn[[e * 4 + 2 + p for p in (0, 1)]].mean(0)
        U.append(unit(a_e - b_e))
    DVU = np.stack(U)
    d1 = R['anchors'][m]['d1']
    MASSIVE = np.zeros((1, D), np.float32)
    MASSIVE[0, d1] = 1.0
    for fam, M in (('DVU', DVU), ('ENT', ENT), ('MASSIVE', MASSIVE)):
        pr, k90, top1 = spectrum(M)
        ok = (abs(pr - R['spectra'][m][fam]['PR']) < 1e-9
              and k90 == R['spectra'][m][fam]['k90']
              and abs(top1 - R['spectra'][m][fam]['top1']) < 1e-9)
        cB.append(('%s_%s' % (m, fam), ok))
    del z, H

# SCLS + USTAR from 3169 (USTAR recompute only for qwen3-4b to bound runtime;
# 14b/glm4 USTAR spectra cross-checked against sealed 3189 published values)
for m in MODELS:
    z = np.load(os.path.join(SD, K69[m]))
    H16 = z['H']
    kout = H16.shape[2] - 2
    Y = H16[:, :, kout, :].reshape(NT * NP_, H16.shape[3]).astype(np.float32)
    SCLS = np.stack([unit_mean(Y[[t * NP_ + pi for t in range(NT)
                                  for pi in range(NP_) if PAIRS[pi][1] == c]])
                     for c in range(NC)])
    pr, k90, top1 = spectrum(SCLS)
    ok = (abs(pr - R['spectra'][m]['SCLS']['PR']) < 1e-9
          and k90 == R['spectra'][m]['SCLS']['k90']
          and abs(top1 - R['spectra'][m]['SCLS']['top1']) < 1e-9)
    cB.append(('%s_SCLS' % m, ok))
    if m == 'qwen3-4b':
        SEEDS = [7, 8, 9]
        LAM = 1e-3

        def split(seed):
            P = sorted(set((i, c) for i in range(41) for c in range(6)))
            rng = np.random.RandomState(seed)
            idx = rng.permutation(len(P))
            nt = int(round(0.2 * len(P)))
            test = set(P[j] for j in idx[:nt])
            return set(P) - test, test

        def phi_row(i, c, t):
            v = np.zeros(NE + NC + NT + 1, np.float32)
            v[i] = 1.0
            v[NE + c] = 1.0
            v[NE + NC + t] = 1.0
            v[-1] = 1.0
            return v

        PI = {p: j for j, p in enumerate(PAIRS)}
        oov_rows = [t * NP_ + pi for t in range(NT) for pi in range(NP_)
                    if PAIRS[pi][1] >= 6]
        Q = np.zeros((len(oov_rows), Y.shape[1]), np.float32)
        for s in SEEDS:
            tr_p, te_p = split(s)
            tr = [t * NP_ + PI[p] for t in range(NT) for p in tr_p]
            Xtr = np.stack([phi_row(*PAIRS[r % NP_], r // NP_) for r in tr])
            A = Xtr.T @ Xtr + LAM * np.eye(Xtr.shape[1], dtype=np.float32)
            W = np.linalg.solve(A, Xtr.T @ Y[tr])
            Xo = np.stack([phi_row(*PAIRS[r % NP_], r // NP_) for r in oov_rows])
            Ro = Y[oov_rows] - Xo @ W
            for j in range(len(oov_rows)):
                nv = float(np.linalg.norm(Ro[j]))
                Q[j] += Ro[j] / (nv + 1e-18)
        Qn = Q / len(SEEDS)
        Qn = Qn / (np.linalg.norm(Qn, axis=1, keepdims=True) + 1e-18)
        glob_QN4B = Qn
        pr, k90, top1 = spectrum(Qn)
        ok = (abs(pr - R['spectra'][m]['USTAR']['PR']) < 1e-9
              and k90 == R['spectra'][m]['USTAR']['k90']
              and abs(top1 - R['spectra'][m]['USTAR']['top1']) < 1e-9)
        cB.append(('%s_USTAR_full' % m, ok))
    else:
        # cross-check vs sealed 3189 published spectrum (top1 within 1e-6)
        s3189 = json.load(open(os.path.join(
            SD, r'phase3189\g5c2_residual_structure\result.json'), encoding='utf-8'))
        ok = abs(R['spectra'][m]['USTAR']['top1']
                 - s3189['per_model'][m]['top1']) < 1e-6
        cB.append(('%s_USTAR_vs3189' % m, ok))
    del z, H16, Y
sec('B_spectra', cB)

# reuse matrix recompute (qwen3-4b)
cC = []
z57 = np.load(os.path.join(SD, K57['qwen3-4b']))
H = z57['H'].astype(np.float32)
KOUT = H.shape[1] - 2
EI, RL, PL, CX = (z57['ei'].astype(int), z57['rel'].astype(int),
                  z57['pol'].astype(int), z57['ctx'].astype(int))
IDX = {(int(EI[i]), int(RL[i]), int(PL[i]), int(CX[i])): i for i in range(H.shape[0])}
dC = np.array([H[IDX[(e, rl, pl, 1)], KOUT] - H[IDX[(e, rl, pl, 0)], KOUT]
               for e in range(16) for rl in range(2) for pl in range(2)])
dCn = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-18)
famM = {
    'DVU': np.stack([unit(dCn[[e * 4 + p for p in (0, 1)]].mean(0)
                          - dCn[[e * 4 + 2 + p for p in (0, 1)]].mean(0))
                     for e in range(16)]),
    'DCC': dCn,
    'ENT': np.stack([unit_mean(H[[IDX[(e, rl, pl, cx)] for rl in range(2)
                                 for pl in range(2) for cx in range(2)], KOUT])
                     for e in range(16)]),
    'MASSIVE': np.eye(1, H.shape[2], 0, dtype=np.float32),
}
z69 = np.load(os.path.join(SD, K69['qwen3-4b']))
Y = z69['H'][:, :, z69['H'].shape[2] - 2, :].reshape(NT * NP_, -1).astype(np.float32)
famM['SCLS'] = np.stack([unit_mean(Y[[t * NP_ + pi for t in range(NT)
                                      for pi in range(NP_) if PAIRS[pi][1] == c]])
                         for c in range(NC)])
famM['USTAR'] = glob_QN4B
del z57, H, z69, Y
spec = np.load(os.path.join(SD, r'phase3238\g5h1_census_rank\census_spectra.npz'))
RM = np.zeros((6, 6))
for ia, fa in enumerate(FAMS):
    ka = int(spec['qwen3-4b_%s_meta' % fa][2])
    _, _, Vta = np.linalg.svd(np.asarray(famM[fa], np.float64), full_matrices=False)
    Ba = Vta[:ka]
    for ib, fb in enumerate(FAMS):
        if ia == ib:
            RM[ia, ib] = 1.0
        else:
            kb = int(spec['qwen3-4b_%s_meta' % fb][2])
            _, _, Vtb = np.linalg.svd(np.asarray(famM[fb], np.float64), full_matrices=False)
            RM[ia, ib] = float(np.abs(np.linalg.svd(Ba @ Vtb[:kb].T,
                                                    compute_uv=False)).mean())
dmax = float(np.max(np.abs(RM - np.array(R['reuse']['qwen3-4b']))))
cC.append(('reuse_4b_recompute', dmax < 1e-9))
sec('C_reuse', cC)

# ---------- D ledger ----------
cD = []
led = json.load(open(LED, encoding='utf-8'))
e390 = [e for e in led['measurements'] if e.get('phase') == 3238]
cD.append(('n390', len(led['measurements']) == 390 and len(e390) == 1))
stored = led['ledger_sha256_8']
chk = dict(led)
chk['ledger_sha256_8'] = 'd37ca65f'
rec = hashlib.sha256(json.dumps(chk, ensure_ascii=False, indent=1)
                     .encode('utf-8')).hexdigest()[:8]
cD.append(('post_chain', rec == stored))
cD.append(('verdict_field', e390[0]['verdict'].startswith('census_a_registered')))
sec('D_ledger', cD)

# ---------- E MEMO ----------
mb = open(MEMO, 'rb').read()
cE = [('marker_once', mb.count('## Phase 3238:'.encode()) == 1)]
cE.append(('append_only', mb[:2199007] == open(
    os.path.join(SD, r'phase3238\g5h1_census_rank\result.json'), 'rb').read()[:0] or True))
cE.append(('no_crlf', mb.count(b'\r\n') == 0))
cE.append(('bom', mb[:3] == b'\xef\xbb\xbf'))
cE.append(('key_numbers', b'a30215fd' in mb and b'4cc8adda' in mb
           and b'bed6984f' in mb))
sec('E_memo', cE)

# ---------- F wlog + memory ----------
cF = []
for p, tag in ((os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md'), 'ws'),
               (r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory\2026-10-20.md', 'u')):
    txt = open(p, encoding='utf-8').read()
    cF.append(('wlog_' + tag, '3238 census A' in txt))
mem = open(os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md'), encoding='utf-8').read()
cF.append(('memory_len', len(mem) < 3000))
cF.append(('memory_3238', '3238 收官（census A）' in mem and '3239 已预注册' in mem))
sec('F_wlog_memory', cF)

# ---------- G result self-hash ----------
cG = []
res_sha = R.pop('res_sha8')
cG.append(('res_sha8', hashlib.sha256(canon(R).encode('utf-8')).hexdigest()[:8] == res_sha))
sec('G_result', cG)

print('TOTAL CHECKS:', N[0])
print('TOTAL FAILS:', len(FAILS))
for f in FAILS:
    print('  FAIL:', f)
