# -*- coding: utf-8 -*-
"""Phase 3240 independent disk verify (read-only re-derivation from sealed artifacts)."""
import os, json, hashlib
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SD = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
PD = os.path.join(SD, r'phase3240\g5h3_cls_extrapolation')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
CHECKS = []


def ck(name, ok, detail=''):
    CHECKS.append((name, bool(ok), detail))


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(open(os.path.join(PD, 'result.json'), encoding='utf-8'))
E = json.load(open(os.path.join(PD, 'execution.json'), encoding='utf-8'))

# 1 result integrity
ck('1.1 verdict', R['verdict'] == 'cls_extrap_registered', R['verdict'])
ck('1.2 res_sha8 self (deterministic fields only: excludes created/runtime_s, '
   'mirroring freeze discipline)',
   R['res_sha8'] == hashlib.sha256(json.dumps(
       {k: v for k, v in R.items()
        if k not in ('res_sha8', 'created', 'runtime_s')}, ensure_ascii=False,
       sort_keys=True, indent=1).encode()).hexdigest()[:8], R['res_sha8'])
ck('1.3 design sha match', E['design_sha8'] == R['design_sha8'] == '69c1a6f9',
   R['design_sha8'])
ck('1.4 execution frozen-before', E['frozen_before'] == 'any observation')
ck('1.5 prereg_correction x3 in freeze',
   E['design'].get('prereg_correction', '').count('(1)') == 1
   and E['design'].get('prereg_correction', '').count('(3)') == 1)
ck('1.6 gates all True', all(R['gates'][k] is True for k in
   ('G1_replication', 'G2_efficacy', 'G3_determinism', 'G4_registry_untouched')))
ck('1.7 determinism 0.0', R['gates']['determinism_diff'] == 0.0)
ck('1.8 d39 vs sealed 3239 = 0.0', R['gates']['d39_vs_sealed_3239'] is True
   and R['aggregate']['F0']['d_vs_sealed_3239_max'] == 0.0)
ck('1.9 no CRLF result.json', open(os.path.join(PD, 'result.json'), 'rb')
   .read().count(b'\r\n') == 0)

# 2 run log + stdout capture
rl = open(os.path.join(PD, 'run_log.txt'), encoding='utf-8').read()
ck('2.1 verdict line (run_log VERDICT + stdout RUN_COMPLETE; RUN_COMPLETE is '
   'logged after run_log write by design)', 'VERDICT: cls_extrap_registered' in rl
   and 'RUN_COMPLETE cls_extrap_registered'
   in open(os.path.join(ROOT, r'tests\gpt5_temp\p3240_formal_out.txt'), encoding='utf-8').read())
ck('2.2 run_log anchors', 'd83=0.000e+00' in rl and 'd39=0.000e+00' in rl)

# 3 sealed npz structure
z = np.load(os.path.join(PD, 'extrap.npz'))
ck('3.1 oov ids 876', z['oov_row_ids'].shape == (876,), str(z['oov_row_ids'].shape))
ck('3.2 agg_fid_matrix 3x6', z['agg_fid_matrix'].shape == (3, 6),
   str(z['agg_fid_matrix'].shape))
ok_fid = all(z['seed%d_fid_%s' % (s, rn)].shape == (6,) and np.isfinite(
    z['seed%d_fid_%s' % (s, rn)]).all()
    for s in (7, 8, 9) for rn in ('F0', 'F1', 'F2'))
ck('3.3 per-seed fid arrays finite', ok_fid)
ck('3.4 agg_r2/agg_kl finite', np.isfinite(z['agg_r2']).all()
   and np.isfinite(z['agg_kl']).all())

# 4 aggregation recompute from per_seed in result
ps = R['per_seed']
FAMS = ['DVU', 'DCC', 'ENT', 'USTAR', 'SCLS', 'MASSIVE']
ok4 = True
for rn in ('F0', 'F1', 'F2'):
    for fi, f in enumerate(FAMS):
        m_re = float(np.mean([ps[str(s)][rn]['fam_fid'][f] for s in (7, 8, 9)]))
        if abs(m_re - z['agg_fid_matrix'][['F0', 'F1', 'F2'].index(rn), fi]) > 1e-9:
            ok4 = False
ck('4.1 agg fam_fid recompute (3 recipes x 6 fams)', ok4)
r2_re = float(np.mean([ps[str(s)]['F2']['r2_oov'] for s in (7, 8, 9)]))
ck('4.2 agg F2 R2 recompute', abs(r2_re - R['aggregate']['F2']['r2_oov_mean']) < 1e-9)

# 5 independent re-derivation: F2 seed7 SCLS/ENT fam_fid from raw collect npz
NE, NC, NT = 73, 10, 3
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS)
PI = {p: j for j, p in enumerate(PAIRS)}
LAM = 1e-3


def phi_row(i, c, t):
    v = np.zeros(NE + NC + NT + 1, np.float32)
    v[i] = 1.; v[NE + c] = 1.; v[NE + NC + t] = 1.; v[-1] = 1.
    return v


z69 = np.load(os.path.join(SD, r'phase3169\g5a6_oov_panel\collect_qwen3-4b.npz'))
H16 = z69['H']
kout = H16.shape[2] - 2
Y = H16[:, :, kout, :].reshape(NT * NP_, H16.shape[3]).astype(np.float32)
SEEN = set((i, c) for i in range(41) for c in range(6))
pairs246 = sorted(SEEN)
rng = np.random.RandomState(7)
idx = rng.permutation(len(pairs246))
test = set(pairs246[j] for j in idx[:int(round(0.2 * 246))])
tr = sorted(SEEN - test)
tr_rows = [t * NP_ + PI[p] for t in range(NT) for p in PAIRS if p in tr]
Xtr = np.stack([phi_row(*PAIRS[r % NP_], r // NP_) for r in tr_rows])
A = Xtr.T @ Xtr + LAM * np.eye(Xtr.shape[1], dtype=np.float32)
W = np.linalg.solve(A, Xtr.T @ Y[tr_rows])
B0 = Y[tr_rows].mean(0)
oov_rows = [t * NP_ + pi for t in range(NT) for pi in range(NP_) if PAIRS[pi][1] >= 6]
Xo = np.stack([phi_row(*PAIRS[r % NP_], r // NP_) for r in oov_rows])
Yhat0 = Xo @ W
cls_mean = {}
for r in tr_rows:
    cls_mean.setdefault(PAIRS[r % NP_][1], []).append(Y[r])
cls_mean = {c: np.stack(v).mean(0) for c, v in cls_mean.items()}
S_cls = np.stack([cls_mean[c] - B0 for c in range(6)])
Qs, _ = np.linalg.qr(S_cls.T.astype(np.float32))
Qs = Qs[:, :6].astype(np.float32)
Yhat2 = Yhat0 + (B0[None, :] - Yhat0) @ Qs @ Qs.T
Rb0 = Y[oov_rows] - np.stack([B0] * len(oov_rows))
zc = np.load(os.path.join(SD, r'phase3238\g5h1_census_rank\census_spectra.npz'))
# SCLS basis (rank-1)
SCLS = []
for c in range(NC):
    rows_c = [t * NP_ + PI[(i, c)] for t in range(NT) for i in range(NE)]
    v = Y[rows_c].mean(0)
    SCLS.append(v / (np.linalg.norm(v) + 1e-18))
SCLS = np.stack(SCLS)
_, _, Vt2 = np.linalg.svd(np.asarray(SCLS, np.float64), full_matrices=False)
B_scls = Vt2[:1].astype(np.float32)
em = float((np.linalg.norm((Y[oov_rows] - Yhat2) @ B_scls.T, axis=1) ** 2).mean())
eb = float((np.linalg.norm(Rb0 @ B_scls.T, axis=1) ** 2).mean())
fid2_scls = 1.0 - em / max(eb, 1e-300)
ck('5.1 F2 SCLS fam_fid recompute seed7',
   abs(fid2_scls - ps['7']['F2']['fam_fid']['SCLS']) < 1e-4,
   '%.6f vs %.6f' % (fid2_scls, ps['7']['F2']['fam_fid']['SCLS']))
# ENT basis (16x d entity means from 3157)
z57 = np.load(os.path.join(SD, r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz'))
H57 = z57['H'].astype(np.float32)
EI, RL, PL, CX = (z57[k].astype(int) for k in ('ei', 'rel', 'pol', 'ctx'))
IDX = {(int(EI[i]), int(RL[i]), int(PL[i]), int(CX[i])): i for i in range(128)}
ENTm = np.stack([
    H57[[IDX[(e, rl, pl, cx)] for rl in range(2) for pl in range(2)
         for cx in range(2)], 35].mean(0)
    / (np.linalg.norm(H57[[IDX[(e, rl, pl, cx)] for rl in range(2)
                           for pl in range(2) for cx in range(2)], 35].mean(0)) + 1e-18)
    for e in range(16)])
k90_ent = int(round(zc['qwen3-4b_ENT_meta'][2]))
_, _, Vt3 = np.linalg.svd(np.asarray(ENTm, np.float64), full_matrices=False)
B_ent = Vt3[:k90_ent].astype(np.float32)
em3 = float((np.linalg.norm((Y[oov_rows] - Yhat2) @ B_ent.T, axis=1) ** 2).mean())
eb3 = float((np.linalg.norm(Rb0 @ B_ent.T, axis=1) ** 2).mean())
fid2_ent = 1.0 - em3 / max(eb3, 1e-300)
ck('5.2 F2 ENT fam_fid recompute seed7',
   abs(fid2_ent - ps['7']['F2']['fam_fid']['ENT']) < 1e-4,
   '%.6f vs %.6f' % (fid2_ent, ps['7']['F2']['fam_fid']['ENT']))
# F0 vs sealed 3239 headline (bit-level)
R39 = json.load(open(os.path.join(SD, r'phase3239\g5h2_fidelity_minloop\result.json'),
                     encoding='utf-8'))
ck('5.3 F0 agg == sealed 3239 headline',
   R['aggregate']['F0']['r2_oov_mean'] == R39['aggregate']['r2_oov_model_mean']
   and R['aggregate']['F0']['kl_mean'] == R39['aggregate']['kl_model_mean_mean'])
ck('5.4 F1 collapse registered (|dSCLS|,|dENT| < 0.05)',
   abs(R['delta_vs_F0']['F1']['d_fam_fid']['SCLS']) < 0.05
   and abs(R['delta_vs_F0']['F1']['d_fam_fid']['ENT']) < 0.05,
   '%.4f/%.4f' % (R['delta_vs_F0']['F1']['d_fam_fid']['SCLS'],
                  R['delta_vs_F0']['F1']['d_fam_fid']['ENT']))
ck('5.5 all six families improve under F2',
   all(R['delta_vs_F0']['F2']['d_fam_fid'][f] > 0 for f in FAMS))

# 6 ledger
led = json.load(open(LED, encoding='utf-8'))
ck('6.1 ledger n=392', len(led['measurements']) == 392, str(len(led['measurements'])))
e392 = [e for e in led['measurements'] if e.get('phase') == 3240]
ck('6.2 3240 entry present', len(e392) == 1)
led['ledger_sha256_8'] = '016f79aa'
rec = hashlib.sha256(json.dumps(led, ensure_ascii=False, indent=1).encode()).hexdigest()[:8]
ck('6.3 ledger chain recompute', rec == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'] == '0cf46b3a', rec)
ck('6.4 ledger no CRLF', open(LED, 'rb').read().count(b'\r\n') == 0)

# 7 MEMO
mb = open(MEMO, 'rb').read()
ck('7.1 MEMO 3240 once', mb.count(b'## Phase 3240:') == 1)
ck('7.2 MEMO headline numbers', ('%.4f' % R['aggregate']['F2']['kl_mean']).encode() in mb
   and ('%+.3f' % R['delta_vs_F0']['F2']['d_fam_fid']['SCLS']).encode() in mb)

# 8 wlogs + memory
for tag, p in [('8.1 wlog ws', os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md')),
               ('8.2 wlog user', os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory', '2026-10-20.md'))]:
    ck(tag, '3240 类方向外推完成' in open(p, encoding='utf-8').read())
mem = open(MEM, encoding='utf-8').read()
ck('8.3 MEMORY len<3000', len(mem) < 3000, str(len(mem)))
ck('8.4 MEMORY 3240+3241', '3240 收官（cls extrapolation）' in mem and '3241 已预注册' in mem)

# 9 registry untouched
ck('9.1 registry v1.33 byte sha8', sha8_file(os.path.join(
    SD, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json')) == 'a30215fd')

n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
for name, ok, detail in CHECKS:
    print('%s %s %s' % ('PASS' if ok else 'FAIL', name, detail))
print('TOTAL: %d checks, %d FAILS' % (len(CHECKS), n_fail))
