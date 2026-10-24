# -*- coding: utf-8 -*-
"""Phase 3242 independent disk verify (read-only re-derivation from sealed artifacts)."""
import os, json, hashlib
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SD = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
PD = os.path.join(SD, r'phase3242\g5h5_xmodel_transfer')
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
ck('1.1 verdict partial', R['verdict'] == 'xmodel_transfer_partial', R['verdict'])
ck('1.2 res_sha8 self (deterministic fields only)',
   R['res_sha8'] == hashlib.sha256(json.dumps(
       {k: v for k, v in R.items()
        if k not in ('res_sha8', 'created', 'runtime_s')}, ensure_ascii=False,
       sort_keys=True, indent=1).encode()).hexdigest()[:8], R['res_sha8'])
ck('1.3 design sha match', E['design_sha8'] == R['design_sha8'] == '02af8bd7',
   R['design_sha8'])
ck('1.4 execution frozen-before', E['frozen_before'] == 'any observation')
ck('1.5 prereg_correction in freeze (row-level control redesign)',
   'ROW-LEVEL' in E['design'].get('prereg_correction', '') or
   'row-level' in E['design'].get('prereg_correction', '')
   or '行级' in E['design'].get('prereg_correction', '')
   or 'ROW-LEVEL permutation' in json.dumps(E['design'], ensure_ascii=False))
ck('1.6 gates', R['gates']['G1_replication'] is True
   and R['gates']['G2_efficacy'] is False
   and R['gates']['G3_determinism'] is True
   and R['gates']['G4_registry_untouched'] is True
   and R['gates']['d40_vs_sealed_3240'] is True)
ck('1.7 determinism 0.0', R['gates']['determinism_diff'] == 0.0)
ck('1.8 G0 bit-level vs sealed 3240', R['aggregate']['G0']['d_vs_sealed_3240_max'] == 0.0)
ck('1.9 no CRLF result.json', open(os.path.join(PD, 'result.json'), 'rb')
   .read().count(b'\r\n') == 0)

# 2 run log + stdout capture
rl = open(os.path.join(PD, 'run_log.txt'), encoding='utf-8').read()
ck('2.1 verdict line', 'VERDICT: xmodel_transfer_partial' in rl
   and 'RUN_COMPLETE xmodel_transfer_partial'
   in open(os.path.join(ROOT, r'tests\gpt5_temp\p3242_formal_out.txt'), encoding='utf-8').read())
ck('2.2 run_log anchors', 'd83=0.000e+00' in rl and 'd40=0.000e+00' in rl)

# 3 sealed npz structure
z = np.load(os.path.join(PD, 'xmodel.npz'))
RECI = ['G0', 'T1_14b', 'T2_glm', 'T3_dual', 'T1r_14b', 'T2r_glm']
FAMS = ['DVU', 'DCC', 'ENT', 'USTAR', 'SCLS', 'MASSIVE']
ck('3.1 oov ids 876', z['oov_row_ids'].shape == (876,), str(z['oov_row_ids'].shape))
ck('3.2 agg_fid_matrix 6x6', z['agg_fid_matrix'].shape == (6, 6),
   str(z['agg_fid_matrix'].shape))
ck('3.3 per-seed fid arrays finite',
   all(np.isfinite(z['seed%d_fid_%s' % (s, rn)]).all()
       for s in (7, 8, 9) for rn in RECI))
ck('3.4 transfer cos npz present',
   all('xfer_cos_real_%s' % d.replace('-', '_') in z.files
       for d in ('qwen3-14b', 'glm4-9b')))

# 4 aggregation recompute
ps = R['per_seed']
ok4 = True
for rn in RECI:
    for fi, f in enumerate(FAMS):
        m_re = float(np.mean([ps[str(s)][rn]['fam_fid'][f] for s in (7, 8, 9)]))
        if abs(m_re - z['agg_fid_matrix'][RECI.index(rn), fi]) > 1e-9:
            ok4 = False
ck('4.1 agg fam_fid recompute (6 recipes x 6 fams)', ok4)

# 5 key findings re-derivation
D = R['delta_vs_G0']
ck('5.1 T3_dual positive SCLS/ENT, seeds unanimous',
   all(ps[str(s)]['T3_dual']['fam_fid']['SCLS'] > ps[str(s)]['G0']['fam_fid']['SCLS']
       and ps[str(s)]['T3_dual']['fam_fid']['ENT'] > ps[str(s)]['G0']['fam_fid']['ENT']
       for s in (7, 8, 9)))
ck('5.2 T3_dual below EFF_DELTA 0.5 (G2 magnitude clause)',
   0.4 < D['T3_dual']['d_fam_fid']['SCLS'] < 0.5,
   '%.4f' % D['T3_dual']['d_fam_fid']['SCLS'])
ck('5.3 T3_dual zero regression DVU/DCC',
   min(D['T3_dual']['d_fam_fid']['DVU'], D['T3_dual']['d_fam_fid']['DCC']) > -0.01,
   '%.4f' % min(D['T3_dual']['d_fam_fid']['DVU'], D['T3_dual']['d_fam_fid']['DCC']))
ck('5.4 KL separates real vs permuted, seeds unanimous',
   all(ps[str(s)]['T1_14b']['kl_mean'] < ps[str(s)]['T1r_14b']['kl_mean']
       and ps[str(s)]['T2_glm']['kl_mean'] < ps[str(s)]['T2r_glm']['kl_mean']
       for s in (7, 8, 9)))
ck('5.5 transfer cos real > permuted per donor',
   all(float(np.mean(R['aggregate']['transfer_diag'][d]['cos_transfer_real']))
       > float(np.mean(R['aggregate']['transfer_diag'][d]['cos_transfer_perm']))
       for d in ('qwen3-14b', 'glm4-9b')),
   '%.3f/%.3f' % (
       float(np.mean(R['aggregate']['transfer_diag']['qwen3-14b']['cos_transfer_real'])),
       float(np.mean(R['aggregate']['transfer_diag']['glm4-9b']['cos_transfer_real']))))
ck('5.6 G0 == sealed 3240 F2 headline',
   R['aggregate']['G0']['r2_oov_mean']
   == json.load(open(os.path.join(SD, r'phase3240\g5h3_cls_extrapolation\result.json'),
                     encoding='utf-8'))['aggregate']['F2']['r2_oov_mean'])

# independent re-derivation of T1_14b seed7 SCLS fidelity from raw collect npz
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
te_pairs = test
te_rows = [t * NP_ + PI[p] for t in range(NT) for p in PAIRS if p in te_pairs]
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
# donor 14b kout activations
z14 = np.load(os.path.join(SD, r'phase3169\g5a6_oov_panel\collect_qwen3-14b.npz'))
H14 = z14['H']
kout14 = H14.shape[2] - 2
Yd = H14[:, :, kout14, :].reshape(NT * NP_, H14.shape[3]).astype(np.float32)
Hd_tr = Yd[tr_rows]
n = len(Hd_tr)
K = (Hd_tr @ Hd_tr.T).astype(np.float64)
K[np.diag_indices(n)] += LAM
K += 1.0
alpha = np.linalg.solve(K, Y[tr_rows].astype(np.float64)).astype(np.float32)


def apply_map(V):
    return (V @ Hd_tr.T @ alpha + alpha.sum(0, keepdims=True)).astype(np.float32)


don_oov_c = {c: [r for r in oov_rows if PAIRS[r % NP_][1] == c] for c in range(6, 10)}
m_oov = np.stack([Yd[don_oov_c[c]].mean(0) for c in range(6, 10)])
d_hat = apply_map(m_oov) - B0[None, :]
SJE2 = np.concatenate([d_cls, d_hat], axis=0)
Qx, _ = np.linalg.qr(SJE2.T.astype(np.float32))
Qx = Qx[:, :SJE2.shape[0]].astype(np.float32)
Yt1 = (Yhat0 + (B0[None, :] - Yhat0) @ Qx @ Qx.T).astype(np.float32)
Rb0 = Y[oov_rows] - np.stack([B0] * len(oov_rows))
SCLS = []
for c in range(NC):
    rows_c = [t * NP_ + PI[(i2, c)] for t in range(NT) for i2 in range(NE)]
    v = Y[rows_c].mean(0)
    SCLS.append(v / (np.linalg.norm(v) + 1e-18))
SCLS = np.stack(SCLS)
_, _, Vt2 = np.linalg.svd(np.asarray(SCLS, np.float64), full_matrices=False)
B_scls = Vt2[:1].astype(np.float32)
em_ = float((np.linalg.norm((Y[oov_rows] - Yt1) @ B_scls.T, axis=1) ** 2).mean())
eb_ = float((np.linalg.norm(Rb0 @ B_scls.T, axis=1) ** 2).mean())
fid_t1 = 1.0 - em_ / max(eb_, 1e-300)
ck('5.7 T1_14b SCLS fam_fid recompute seed7',
   abs(fid_t1 - ps['7']['T1_14b']['fam_fid']['SCLS']) < 1e-4,
   '%.6f vs %.6f' % (fid_t1, ps['7']['T1_14b']['fam_fid']['SCLS']))

# 6 ledger
led = json.load(open(LED, encoding='utf-8'))
ck('6.1 ledger n=394', len(led['measurements']) == 394, str(len(led['measurements'])))
e394 = [e for e in led['measurements'] if e.get('phase') == 3242]
ck('6.2 3242 entry present', len(e394) == 1)
led['ledger_sha256_8'] = '794ea771'
rec = hashlib.sha256(json.dumps(led, ensure_ascii=False, indent=1).encode()).hexdigest()[:8]
ck('6.3 ledger chain recompute', rec == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'] == 'face54fa', rec)
ck('6.4 ledger no CRLF', open(LED, 'rb').read().count(b'\r\n') == 0)

# 7 MEMO
mb = open(MEMO, 'rb').read()
ck('7.1 MEMO 3242 once', mb.count(b'## Phase 3242:') == 1)
ck('7.2 MEMO headline numbers',
   ('%+.3f' % R['delta_vs_G0']['T3_dual']['d_fam_fid']['SCLS']).encode() in mb
   and b'xmodel_transfer_partial' in mb)

# 8 wlogs + memory
for tag, p in [('8.1 wlog ws', os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md')),
               ('8.2 wlog user', os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory', '2026-10-20.md'))]:
    ck(tag, '3242 跨模型类方向迁移完成' in open(p, encoding='utf-8').read())
mem = open(MEM, encoding='utf-8').read()
ck('8.3 MEMORY len<3000', len(mem) < 3000, str(len(mem)))
ck('8.4 MEMORY 3242+3243', '3242 收官（xmodel transfer' in mem and '3243 已预注册' in mem)

# 9 registry untouched
ck('9.1 registry v1.33 byte sha8', sha8_file(os.path.join(
    SD, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json')) == 'a30215fd')

n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
for name, ok, detail in CHECKS:
    print('%s %s %s' % ('PASS' if ok else 'FAIL', name, detail))
print('TOTAL: %d checks, %d FAILS' % (len(CHECKS), n_fail))
