# -*- coding: utf-8 -*-
"""Phase 3239 independent disk verify (read-only re-derivation from sealed artifacts)."""
import os, json, hashlib
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SD = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
PD = os.path.join(SD, r'phase3239\g5h2_fidelity_minloop')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
MEM = os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md')
SRC_DIR = SD
CHECKS = []


def ck(name, ok, detail=''):
    CHECKS.append((name, bool(ok), detail))


def sha8_file(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


R = json.load(open(os.path.join(PD, 'result.json'), encoding='utf-8'))
E = json.load(open(os.path.join(PD, 'execution.json'), encoding='utf-8'))

# 1 result integrity
ck('1.1 verdict', R['verdict'] == 'fidelity_loop_registered', R['verdict'])
ck('1.2 res_sha8 self', R['res_sha8'] == hashlib.sha256(json.dumps(
    {k: v for k, v in R.items() if k != 'res_sha8'}, ensure_ascii=False,
    sort_keys=True, indent=1).encode()).hexdigest()[:8], R['res_sha8'])
ck('1.3 design sha match', E['design_sha8'] == R['design_sha8'] == '3776313c',
   R['design_sha8'])
ck('1.4 execution frozen-before', E['frozen_before'] == 'any observation')
ck('1.5 prereg_correction in freeze', 'prereg_correction' in E['design'])
ck('1.6 gates all True', all(R['gates'][k] is True for k in
   ('G1_replication', 'G2_metrics', 'G3_determinism', 'G4_registry_untouched')))
ck('1.7 determinism 0.0', R['gates']['determinism_diff'] == 0.0)
ck('1.8 no CRLF result.json', open(os.path.join(PD, 'result.json'), 'rb')
   .read().count(b'\r\n') == 0)

# 2 run log
rl = open(os.path.join(PD, 'run_log.txt'), encoding='utf-8').read()
ck('2.1 verdict line (run_log VERDICT + stdout RUN_COMPLETE; RUN_COMPLETE is '
   'logged after run_log write by design)', 'VERDICT: fidelity_loop_registered' in rl
   and 'RUN_COMPLETE fidelity_loop_registered'
   in open(os.path.join(ROOT, r'tests\gpt5_temp\p3239_formal_out.txt'), encoding='utf-8').read())
ck('2.2 run_log anchors', 'd83=0.000e+00' in rl and 'drift69=0.000e+00' in rl)

# 3 sealed npz structure
z = np.load(os.path.join(PD, 'fidelity.npz'))
oov = z['oov_row_ids']
ck('3.1 oov ids 876', oov.shape == (876,), str(oov.shape))
ok_kl = True
det = []
for s in (7, 8, 9):
    for k3 in ('model', 'B0', 'B1'):
        a = z['seed%d_kl_%s' % (s, k3)]
        if a.shape != (876,) or not np.isfinite(a).all() or a.min() < -1e-3:
            ok_kl = False
            det.append('seed%d_%s' % (s, k3))
ck('3.2 kl arrays shape/finite/nonneg', ok_kl, ','.join(det))
ck('3.3 fam_fid shape 3x6', z['fam_fid'].shape == (3, 6), str(z['fam_fid'].shape))

# 4 aggregation recompute from per_seed in result
ps = R['per_seed']
m_re = float(np.mean([ps[str(s)]['kl_model_mean'] for s in (7, 8, 9)]))
ck('4.1 agg KL model recompute', abs(m_re - R['aggregate']['kl_model_mean_mean']) < 1e-9,
   '%.9f' % m_re)
r_re = float(np.mean([ps[str(s)]['r2_oov_model'] for s in (7, 8, 9)]))
ck('4.2 agg R2 recompute', abs(r_re - R['aggregate']['r2_oov_model_mean']) < 1e-9)

# 5 independent re-derivation: ridge seed7 + fam_fid (DVU/SCLS) + KL subset
NE, NC, NT = 73, 10, 3
PAIRS = [(i, c) for i in range(NE) for c in range(NC)]
NP_ = len(PAIRS)
PI = {p: j for j, p in enumerate(PAIRS)}
LAM = 1e-3


def phi_row(i, c, t):
    v = np.zeros(NE + NC + NT + 1, np.float32)
    v[i] = 1.; v[NE + c] = 1.; v[NE + NC + t] = 1.; v[-1] = 1.
    return v


zc = np.load(os.path.join(SD, r'phase3238\g5h1_census_rank\census_spectra.npz'))
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
Yhat = Xo @ W
Rm = Y[oov_rows] - Yhat
Rb0 = Y[oov_rows] - np.stack([B0] * len(oov_rows))
fam_map = {'DVU': None, 'SCLS': None}
# rebuild DVU basis via 3157 verbatim (seed-free)
z57 = np.load(os.path.join(SD, r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz'))
H57 = z57['H'].astype(np.float32)
EI, RL, PL, CX = (z57[k].astype(int) for k in ('ei', 'rel', 'pol', 'ctx'))
IDX = {(int(EI[i]), int(RL[i]), int(PL[i]), int(CX[i])): i for i in range(128)}
dC = np.array([H57[IDX[(e, rl, pl, 1)], 35] - H57[IDX[(e, rl, pl, 0)], 35]
               for e in range(16) for rl in range(2) for pl in range(2)])
dCn = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-18)
rows_isa = [e * 4 + rl * 2 + pl for e in range(16) for rl in range(2) for pl in range(2) if rl == 0]
rows_hasa = [e * 4 + rl * 2 + pl for e in range(16) for rl in range(2) for pl in range(2) if rl == 1]
um = lambda rows: (dCn[rows].mean(0) / (np.linalg.norm(dCn[rows].mean(0)) + 1e-18))
U = np.stack([um([e * 4 + p_, p_ + 1]) if False else
              ((dCn[[e * 4 + 0 * 2 + p for p in (0, 1)]].mean(0) -
                dCn[[e * 4 + 1 * 2 + p for p in (0, 1)]].mean(0)) /
               (np.linalg.norm(dCn[[e * 4 + 0 * 2 + p for p in (0, 1)]].mean(0) -
                               dCn[[e * 4 + 1 * 2 + p for p in (0, 1)]].mean(0)) + 1e-18))
              for e in range(16)])
k90_dvu = int(round(zc['qwen3-4b_DVU_meta'][2]))
_, _, Vt = np.linalg.svd(np.asarray(U, np.float64), full_matrices=False)
B_dvu = Vt[:k90_dvu].astype(np.float32)
em = float((np.linalg.norm(Rm @ B_dvu.T, axis=1) ** 2).mean())
eb = float((np.linalg.norm(Rb0 @ B_dvu.T, axis=1) ** 2).mean())
fid_dvu = 1.0 - em / max(eb, 1e-300)
ck('5.1 DVU fam_fid recompute seed7', abs(fid_dvu - ps['7']['fam_fid_oov']['DVU']) < 1e-4,
   '%.6f vs %.6f' % (fid_dvu, ps['7']['fam_fid_oov']['DVU']))
# SCLS basis
SCLS = []
for c in range(NC):
    rows_c = [t * NP_ + PI[(i, c)] for t in range(NT) for i in range(NE)]
    v = Y[rows_c].mean(0)
    SCLS.append(v / (np.linalg.norm(v) + 1e-18))
SCLS = np.stack(SCLS)
_, _, Vt2 = np.linalg.svd(np.asarray(SCLS, np.float64), full_matrices=False)
B_scls = Vt2[:1].astype(np.float32)
em2 = float((np.linalg.norm(Rm @ B_scls.T, axis=1) ** 2).mean())
eb2 = float((np.linalg.norm(Rb0 @ B_scls.T, axis=1) ** 2).mean())
fid_scls = 1.0 - em2 / max(eb2, 1e-300)
ck('5.2 SCLS fam_fid recompute seed7', abs(fid_scls - ps['7']['fam_fid_oov']['SCLS']) < 1e-4,
   '%.6f vs %.6f' % (fid_scls, ps['7']['fam_fid_oov']['SCLS']))
# KL subset recompute (first 128 rows, seed 7)
from safetensors import safe_open
cfg = json.load(open(os.path.join(ROOT, r'models\hf\qwen3-4b\config.json'), encoding='utf-8'))
with safe_open(os.path.join(ROOT, r'models\hf\qwen3-4b\model-00001-of-00003.safetensors'),
               framework='pt', device='cpu') as f:
    Emb = f.get_tensor('model.embed_tokens.weight').float().numpy()
with safe_open(os.path.join(ROOT, r'models\hf\qwen3-4b\model-00003-of-00003.safetensors'),
               framework='pt', device='cpu') as f:
    gamma = f.get_tensor('model.norm.weight').float().numpy()
eps = float(cfg['rms_norm_eps'])
nb = 128
ya = Y[np.array(oov_rows[:nb])]


def logp(H):
    rms = np.sqrt((H * H).mean(1) + eps).astype(np.float32)
    nh = H / rms[:, None] * gamma[None, :]
    lg = nh @ Emb.T
    m = lg.max(1, keepdims=True)
    return lg - (m + np.log(np.exp(lg - m).sum(1, keepdims=True)))


la = logp(ya)
lm = logp(Yhat[:nb])
kl_sub = (np.exp(la.astype(np.float64)) * (la.astype(np.float64) - lm.astype(np.float64))).sum(1)
ck('5.3 KL subset recompute seed7', float(np.max(np.abs(kl_sub - z['seed7_kl_model'][:nb]))) < 1e-4,
   'max diff %.2e' % float(np.max(np.abs(kl_sub - z['seed7_kl_model'][:nb]))))

# 6 ledger
led = json.load(open(LED, encoding='utf-8'))
ck('6.1 ledger n=391', len(led['measurements']) == 391, str(len(led['measurements'])))
e391 = [e for e in led['measurements'] if e.get('phase') == 3239]
ck('6.2 3239 entry present', len(e391) == 1)
led['ledger_sha256_8'] = '2c2e0d4f'
rec = hashlib.sha256(json.dumps(led, ensure_ascii=False, indent=1).encode()).hexdigest()[:8]
ck('6.3 ledger chain recompute', rec == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'] == '016f79aa', rec)
ck('6.4 ledger no CRLF', open(LED, 'rb').read().count(b'\r\n') == 0)

# 7 MEMO
mb = open(MEMO, 'rb').read()
ck('7.1 MEMO 3239 once', mb.count(b'## Phase 3239:') == 1)
ck('7.2 MEMO headline numbers', ('%.4f' % R['aggregate']['kl_model_mean_mean']).encode() in mb
   and b'38.7' in mb)

# 8 wlogs + memory
for tag, p in [('8.1 wlog ws', os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md')),
               ('8.2 wlog user', os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory', '2026-10-20.md'))]:
    ck(tag, '3239 fidelity min-loop' in open(p, encoding='utf-8').read())
mem = open(MEM, encoding='utf-8').read()
ck('8.3 MEMORY len<3000', len(mem) < 3000, str(len(mem)))
ck('8.4 MEMORY 3239+3240', '3239 收官（fidelity min-loop）' in mem and '3240 已预注册' in mem)

# 9 registry untouched
ck('9.1 registry v1.33 byte sha8', sha8_file(os.path.join(
    SD, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json')) == 'a30215fd')

n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
for name, ok, detail in CHECKS:
    print('%s %s %s' % ('PASS' if ok else 'FAIL', name, detail))
print('TOTAL: %d checks, %d FAILS' % (len(CHECKS), n_fail))
