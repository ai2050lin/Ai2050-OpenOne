# -*- coding: utf-8 -*-
"""Phase 3243 independent disk verify (read-only re-derivation from sealed artifacts)."""
import os, json, hashlib
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SD = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
PD = os.path.join(SD, r'phase3243\g5h6_map_oov_extrap')
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
ck('1.1 verdict partial', R['verdict'] == 'mapfix_partial', R['verdict'])
ck('1.2 res_sha8 self (deterministic fields only)',
   R['res_sha8'] == hashlib.sha256(json.dumps(
       {k: v for k, v in R.items()
        if k not in ('res_sha8', 'created', 'runtime_s')}, ensure_ascii=False,
       sort_keys=True, indent=1).encode()).hexdigest()[:8], R['res_sha8'])
ck('1.3 design sha match', E['design_sha8'] == R['design_sha8'] == '5ccf2bdc',
   R['design_sha8'])
ck('1.4 execution frozen-before', E['frozen_before'] == 'any observation')
ck('1.5 map variants preregistered in freeze',
   all(k in E['design'].get('map_variants', {})
       for k in ('base', 'M1_centered', 'M2_clscond', 'M3_anchor_min_norm')))
ck('1.6 gates', R['gates']['G1_replication'] is True
   and R['gates']['G2_efficacy'] is False
   and R['gates']['G3_determinism'] is True
   and R['gates']['G4_registry_untouched'] is True
   and R['gates']['d40_vs_sealed_3240'] is True
   and R['gates']['d42_vs_sealed_3242'] is True)
ck('1.7 determinism 0.0', R['gates']['determinism_diff'] == 0.0)
ck('1.8 G0 bit-level vs sealed 3240', R['aggregate']['G0']['d_vs_sealed_3240_max'] == 0.0)
ck('1.9 T3_rep bit-level vs sealed 3242',
   R['aggregate']['T3_rep']['d_vs_sealed_3242_max'] == 0.0)
ck('1.10 no CRLF result.json', open(os.path.join(PD, 'result.json'), 'rb')
   .read().count(b'\r\n') == 0)

# 2 run log + stdout capture
rl = open(os.path.join(PD, 'run_log.txt'), encoding='utf-8').read()
ck('2.1 verdict line', 'VERDICT: mapfix_partial' in rl
   and 'RUN_COMPLETE mapfix_partial'
   in open(os.path.join(ROOT, r'tests\gpt5_temp\p3243_formal_out.txt'), encoding='utf-8').read())
ck('2.2 run_log anchors', 'd83=0.000e+00' in rl and 'd40=' in rl and 'd42=' in rl)

# 3 sealed npz structure
z = np.load(os.path.join(PD, 'mapfix.npz'))
RECI = ['G0', 'T3_rep', 'M1_dual', 'M2_dual', 'M3_dual',
        'M1r_dual', 'M2r_dual', 'M3r_dual']
FAMS = ['DVU', 'DCC', 'ENT', 'USTAR', 'SCLS', 'MASSIVE']
ck('3.1 oov ids 876', z['oov_row_ids'].shape == (876,), str(z['oov_row_ids'].shape))
ck('3.2 agg_fid_matrix 8x6', z['agg_fid_matrix'].shape == (8, 6),
   str(z['agg_fid_matrix'].shape))
ck('3.3 per-seed fid arrays finite',
   all(np.isfinite(z['seed%d_fid_%s' % (s, rn)]).all()
       for s in (7, 8, 9) for rn in RECI))
ck('3.4 transfer cos npz present (per variant)',
   all(('xfer_cos_real_M3_%s' % d.replace('-', '_')) in z.files
       for d in ('qwen3-14b', 'glm4-9b')))

# 4 aggregation recompute
ps = R['per_seed']
ok4 = True
for rn in RECI:
    for fi, f in enumerate(FAMS):
        m_re = float(np.mean([ps[str(s)][rn]['fam_fid'][f] for s in (7, 8, 9)]))
        if abs(m_re - z['agg_fid_matrix'][RECI.index(rn), fi]) > 1e-9:
            ok4 = False
ck('4.1 agg fam_fid recompute (8 recipes x 6 fams)', ok4)

# 5 key findings re-derivation
D = R['delta_vs_G0']
ck('5.1 T3_rep reproduces 3242 T3_dual headline deltas',
   abs(D['T3_rep']['d_fam_fid']['SCLS']
       - json.load(open(os.path.join(SD, r'phase3242\g5h5_xmodel_transfer\result.json'),
                        encoding='utf-8'))['delta_vs_G0']['T3_dual']['d_fam_fid']['SCLS']) < 1e-9)
ck('5.2 ALL map variants fail to beat base (core negative finding)',
   all(D[rn]['d_fam_fid']['SCLS'] < D['T3_rep']['d_fam_fid']['SCLS']
       for rn in ('M1_dual', 'M2_dual', 'M3_dual')))
ck('5.3 centering HURTS (M1 < base on SCLS and ENT)',
   D['M1_dual']['d_fam_fid']['SCLS'] < D['T3_rep']['d_fam_fid']['SCLS']
   and D['M1_dual']['d_fam_fid']['ENT'] < D['T3_rep']['d_fam_fid']['ENT'],
   'M1 %+.3f/%+.3f vs base %+.3f/%+.3f' % (
       D['M1_dual']['d_fam_fid']['SCLS'], D['M1_dual']['d_fam_fid']['ENT'],
       D['T3_rep']['d_fam_fid']['SCLS'], D['T3_rep']['d_fam_fid']['ENT']))
ck('5.4 M2 collapse: seen-class mapping residuals at noise scale',
   all(R['aggregate']['transfer_diag'][d]['M2']['res_norm_mean'] < 0.01
       for d in ('qwen3-14b', 'glm4-9b')),
   'res %.1e/%.1e' % (R['aggregate']['transfer_diag']['qwen3-14b']['M2']['res_norm_mean'],
                      R['aggregate']['transfer_diag']['glm4-9b']['M2']['res_norm_mean']))
ck('5.5 M3 geometry without behavior: cos up, KL not above permuted control',
   float(np.mean(R['aggregate']['transfer_diag']['qwen3-14b']['M3']['cos_transfer_real']))
   > float(np.mean(R['aggregate']['transfer_diag']['qwen3-14b']['base']['cos_transfer_real']))
   and D['M3r_dual']['d_kl_mean'] <= D['M3_dual']['d_kl_mean'] + 1e-12,
   'cos14b %.3f vs base %.3f; KL M3 %+.4f M3r %+.4f' % (
       float(np.mean(R['aggregate']['transfer_diag']['qwen3-14b']['M3']['cos_transfer_real'])),
       float(np.mean(R['aggregate']['transfer_diag']['qwen3-14b']['base']['cos_transfer_real'])),
       D['M3_dual']['d_kl_mean'], D['M3r_dual']['d_kl_mean']))
ck('5.6 centering does not fix OOV extrapolation',
   all(R['aggregate']['transfer_diag'][d]['M1']['r2_map_oov'] < 0.2
       for d in ('qwen3-14b', 'glm4-9b')))
ck('5.7 G0 == sealed 3240 F2 headline',
   R['aggregate']['G0']['r2_oov_mean']
   == json.load(open(os.path.join(SD, r'phase3240\g5h3_cls_extrapolation\result.json'),
                     encoding='utf-8'))['aggregate']['F2']['r2_oov_mean'])

# independent re-derivation of T3_rep seed7 SCLS fidelity from raw collect npz
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
d_cls = np.stack([cls_mean[c] - B0 for c in range(6)])
d_hats = []
for don in ('qwen3-14b', 'glm4-9b'):
    zd = np.load(os.path.join(SD, r'phase3169\g5a6_oov_panel\collect_%s.npz' % don))
    Hd = zd['H']
    kd = Hd.shape[2] - 2
    Yd = Hd[:, :, kd, :].reshape(NT * NP_, Hd.shape[3]).astype(np.float32)
    Hd_tr = Yd[tr_rows]
    n = len(Hd_tr)
    K = (Hd_tr @ Hd_tr.T).astype(np.float64)
    K[np.diag_indices(n)] += LAM
    K += 1.0
    alpha = np.linalg.solve(K, Y[tr_rows].astype(np.float64)).astype(np.float32)
    don_oov_c = {c: [r for r in oov_rows if PAIRS[r % NP_][1] == c] for c in range(6, 10)}
    m_oov = np.stack([Yd[don_oov_c[c]].mean(0) for c in range(6, 10)])
    d_hats.append((m_oov @ Hd_tr.T @ alpha + alpha.sum(0, keepdims=True)).astype(np.float32)
                  - B0[None, :])
SJE2 = np.concatenate([d_cls] + d_hats, axis=0)
Qx, _ = np.linalg.qr(SJE2.T.astype(np.float32))
Qx = Qx[:, :SJE2.shape[0]].astype(np.float32)
Yt3 = (Yhat0 + (B0[None, :] - Yhat0) @ Qx @ Qx.T).astype(np.float32)
Rb0 = Y[oov_rows] - np.stack([B0] * len(oov_rows))
SCLS = []
for c in range(NC):
    rows_c = [t * NP_ + PI[(i2, c)] for t in range(NT) for i2 in range(NE)]
    v = Y[rows_c].mean(0)
    SCLS.append(v / (np.linalg.norm(v) + 1e-18))
SCLS = np.stack(SCLS)
_, _, Vt2 = np.linalg.svd(np.asarray(SCLS, np.float64), full_matrices=False)
B_scls = Vt2[:1].astype(np.float32)
em_ = float((np.linalg.norm((Y[oov_rows] - Yt3) @ B_scls.T, axis=1) ** 2).mean())
eb_ = float((np.linalg.norm(Rb0 @ B_scls.T, axis=1) ** 2).mean())
fid_t3 = 1.0 - em_ / max(eb_, 1e-300)
ck('5.8 T3_rep SCLS fam_fid recompute seed7 (joint dual-donor span)',
   abs(fid_t3 - ps['7']['T3_rep']['fam_fid']['SCLS']) < 1e-4,
   '%.6f vs %.6f' % (fid_t3, ps['7']['T3_rep']['fam_fid']['SCLS']))

# 6 ledger
led = json.load(open(LED, encoding='utf-8'))
ck('6.1 ledger n=395', len(led['measurements']) == 395, str(len(led['measurements'])))
e395 = [e for e in led['measurements'] if e.get('phase') == 3243]
ck('6.2 3243 entry present', len(e395) == 1)
led['ledger_sha256_8'] = 'face54fa'
rec = hashlib.sha256(json.dumps(led, ensure_ascii=False, indent=1).encode()).hexdigest()[:8]
ck('6.3 ledger chain recompute', rec == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'] == '4b5b12a8', rec)
ck('6.4 ledger no CRLF', open(LED, 'rb').read().count(b'\r\n') == 0)

# 7 MEMO
mb = open(MEMO, 'rb').read()
ck('7.1 MEMO 3243 once', mb.count(b'## Phase 3243:') == 1)
ck('7.2 MEMO headline numbers',
   ('%+.3f' % R['delta_vs_G0']['M1_dual']['d_fam_fid']['SCLS']).encode() in mb
   and b'mapfix_partial' in mb)

# 8 wlogs + memory
for tag, p in [('8.1 wlog ws', os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md')),
               ('8.2 wlog user', os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory', '2026-10-20.md'))]:
    ck(tag, '3243 迁移映射 OOV 外推改进完成' in open(p, encoding='utf-8').read())
mem = open(MEM, encoding='utf-8').read()
ck('8.3 MEMORY len<3000', len(mem) < 3000, str(len(mem)))
ck('8.4 MEMORY 3243+3244', '3243 收官（map OOV extrap' in mem and '3244 已预注册' in mem)

# 9 registry untouched
ck('9.1 registry v1.33 byte sha8', sha8_file(os.path.join(
    SD, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json')) == 'a30215fd')

n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
for name, ok, detail in CHECKS:
    print('%s %s %s' % ('PASS' if ok else 'FAIL', name, detail))
print('TOTAL: %d checks, %d FAILS' % (len(CHECKS), n_fail))
