# -*- coding: utf-8 -*-
"""Phase 3241 independent disk verify (read-only re-derivation from sealed artifacts)."""
import os, json, hashlib
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SD = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
PD = os.path.join(SD, r'phase3241\g5h4_lexical_bridge')
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
ck('1.1 verdict partial', R['verdict'] == 'lexical_bridge_partial', R['verdict'])
ck('1.2 res_sha8 self (deterministic fields only)',
   R['res_sha8'] == hashlib.sha256(json.dumps(
       {k: v for k, v in R.items()
        if k not in ('res_sha8', 'created', 'runtime_s')}, ensure_ascii=False,
       sort_keys=True, indent=1).encode()).hexdigest()[:8], R['res_sha8'])
ck('1.3 design sha match', E['design_sha8'] == R['design_sha8'] == 'd986f7ae',
   R['design_sha8'])
ck('1.4 execution frozen-before', E['frozen_before'] == 'any observation')
ck('1.5 prereg_correction x3 in freeze',
   E['design'].get('prereg_correction', '').count('(1)') == 1
   and E['design'].get('prereg_correction', '').count('(3)') == 1
   and 'leakage_declaration' in E['design'])
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
ck('2.1 verdict line', 'VERDICT: lexical_bridge_partial' in rl
   and 'RUN_COMPLETE lexical_bridge_partial'
   in open(os.path.join(ROOT, r'tests\gpt5_temp\p3241_formal_out.txt'), encoding='utf-8').read())
ck('2.2 run_log anchors', 'd83=0.000e+00' in rl and 'd40=0.000e+00' in rl)

# 3 sealed npz structure
z = np.load(os.path.join(PD, 'bridge.npz'))
RECI = ['G0', 'G1c', 'G1w', 'G1e', 'G1r', 'G1rw']
FAMS = ['DVU', 'DCC', 'ENT', 'USTAR', 'SCLS', 'MASSIVE']
ck('3.1 oov ids 876', z['oov_row_ids'].shape == (876,), str(z['oov_row_ids'].shape))
ck('3.2 agg_fid_matrix 6x6', z['agg_fid_matrix'].shape == (6, 6),
   str(z['agg_fid_matrix'].shape))
ck('3.3 per-seed fid arrays finite',
   all(np.isfinite(z['seed%d_fid_%s' % (s, rn)]).all()
       for s in (7, 8, 9) for rn in RECI))

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
ck('5.1 class bridge null (|dSCLS| G1c,G1w < 0.05)',
   abs(R['delta_vs_G0']['G1c']['d_fam_fid']['SCLS']) < 0.05
   and abs(R['delta_vs_G0']['G1w']['d_fam_fid']['SCLS']) < 0.05,
   '%.4f/%.4f' % (R['delta_vs_G0']['G1c']['d_fam_fid']['SCLS'],
                  R['delta_vs_G0']['G1w']['d_fam_fid']['SCLS']))
ck('5.2 dual bridge positive SCLS/ENT, seeds unanimous',
   all(ps[str(s)]['G1e']['fam_fid']['SCLS'] > ps[str(s)]['G0']['fam_fid']['SCLS']
       and ps[str(s)]['G1e']['fam_fid']['ENT'] > ps[str(s)]['G0']['fam_fid']['ENT']
       for s in (7, 8, 9)))
ck('5.3 G2 fail clause: DVU/DCC regression beyond -0.05',
   min(R['delta_vs_G0']['G1e']['d_fam_fid']['DVU'],
       R['delta_vs_G0']['G1e']['d_fam_fid']['DCC']) < -0.05,
   '%.4f' % min(R['delta_vs_G0']['G1e']['d_fam_fid']['DVU'],
                R['delta_vs_G0']['G1e']['d_fam_fid']['DCC']))
ck('5.4 real identity beats permuted, seeds unanimous',
   all(min(ps[str(s)]['G1e']['fam_fid']['SCLS'], ps[str(s)]['G1e']['fam_fid']['ENT'])
       > min(ps[str(s)]['G1r']['fam_fid']['SCLS'], ps[str(s)]['G1r']['fam_fid']['ENT'])
       for s in (7, 8, 9)))
ck('5.5 G0 == sealed 3240 F2 headline',
   R['aggregate']['G0']['r2_oov_mean']
   == json.load(open(os.path.join(SD, r'phase3240\g5h3_cls_extrapolation\result.json'),
                     encoding='utf-8'))['aggregate']['F2']['r2_oov_mean'])

# independent re-derivation of G1e seed7 SCLS fidelity from raw collect npz
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
ent_mean = {}
for r in tr_rows:
    cls_mean.setdefault(PAIRS[r % NP_][1], []).append(Y[r])
    ent_mean.setdefault(PAIRS[r % NP_][0], []).append(Y[r])
cls_mean = {c: np.stack(v).mean(0) for c, v in cls_mean.items()}
ent_mean = {i: np.stack(v).mean(0) for i, v in ent_mean.items()}
d_cls = np.stack([cls_mean[c] - B0 for c in range(6)])
e_ent = np.stack([ent_mean[i] - B0 for i in range(41)])
d_mean, e_mean = d_cls.mean(0), e_ent.mean(0)
Qs, _ = np.linalg.qr(d_cls.T.astype(np.float32))
Qs = Qs[:, :6].astype(np.float32)
SJE = np.concatenate([d_cls, e_ent], axis=0)
Qje, _ = np.linalg.qr(SJE.T.astype(np.float32))
Qje = Qje[:, :SJE.shape[0]].astype(np.float32)
# lexical bridge weights (softmax beta=1) from embed_tokens
from safetensors import safe_open
with safe_open(os.path.join(ROOT, r'models\hf\qwen3-4b\model-00001-of-00003.safetensors'),
               framework='pt', device='cpu') as f:
    Emb = f.get_tensor('model.embed_tokens.weight').float().numpy()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, r'models\hf\qwen3-4b'))
CLASSES_SEEN = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
CLASSES_OOV = ['乐器', '天气', '运动', '电器']
CLS_WORDS = CLASSES_SEEN + CLASSES_OOV


def wrow2(w):
    ids = tok.encode(w, add_special_tokens=False)
    return np.asarray(Emb[np.array(ids, int)], np.float32).mean(0)


def sfn(cosv):
    e = np.exp(cosv - cosv.max())
    return e / e.sum()


cls_w = np.stack([wrow2(w) for w in CLS_WORDS]).astype(np.float32)
unn = cls_w / (np.linalg.norm(cls_w, axis=1, keepdims=True) + 1e-18)
cos_cm = unn @ unn.T
Ycorr = np.zeros_like(Yhat0)
for j, r in enumerate(oov_rows):
    i, c = PAIRS[r % NP_]
    A_ = B0 + (sfn(cos_cm[c][:6])[:, None] * (d_cls - d_mean[None, :])).sum(0)
    Ycorr[j] = (A_ - Yhat0[j]) @ Qs @ Qs.T
Yg1c = Yhat0 + Ycorr
Rb0 = Y[oov_rows] - np.stack([B0] * len(oov_rows))
SCLS = []
for c in range(NC):
    rows_c = [t * NP_ + PI[(i2, c)] for t in range(NT) for i2 in range(NE)]
    v = Y[rows_c].mean(0)
    SCLS.append(v / (np.linalg.norm(v) + 1e-18))
SCLS = np.stack(SCLS)
_, _, Vt2 = np.linalg.svd(np.asarray(SCLS, np.float64), full_matrices=False)
B_scls = Vt2[:1].astype(np.float32)
em_ = float((np.linalg.norm((Y[oov_rows] - Yg1c) @ B_scls.T, axis=1) ** 2).mean())
eb_ = float((np.linalg.norm(Rb0 @ B_scls.T, axis=1) ** 2).mean())
fid_g1c = 1.0 - em_ / max(eb_, 1e-300)
ck('5.6 G1c SCLS fam_fid recompute seed7',
   abs(fid_g1c - ps['7']['G1c']['fam_fid']['SCLS']) < 1e-4,
   '%.6f vs %.6f' % (fid_g1c, ps['7']['G1c']['fam_fid']['SCLS']))

# 6 ledger
led = json.load(open(LED, encoding='utf-8'))
ck('6.1 ledger n=393', len(led['measurements']) == 393, str(len(led['measurements'])))
e393 = [e for e in led['measurements'] if e.get('phase') == 3241]
ck('6.2 3241 entry present', len(e393) == 1)
led['ledger_sha256_8'] = '0cf46b3a'
rec = hashlib.sha256(json.dumps(led, ensure_ascii=False, indent=1).encode()).hexdigest()[:8]
ck('6.3 ledger chain recompute', rec == json.load(open(LED, encoding='utf-8'))['ledger_sha256_8'] == '794ea771', rec)
ck('6.4 ledger no CRLF', open(LED, 'rb').read().count(b'\r\n') == 0)

# 7 MEMO
mb = open(MEMO, 'rb').read()
ck('7.1 MEMO 3241 once', mb.count(b'## Phase 3241:') == 1)
ck('7.2 MEMO headline numbers',
   ('%+.3f' % R['delta_vs_G0']['G1e']['d_fam_fid']['SCLS']).encode() in mb
   and b'lexical_bridge_partial' in mb)

# 8 wlogs + memory
for tag, p in [('8.1 wlog ws', os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md')),
               ('8.2 wlog user', os.path.join(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal\memory', '2026-10-20.md'))]:
    ck(tag, '3241 词汇桥完成' in open(p, encoding='utf-8').read())
mem = open(MEM, encoding='utf-8').read()
ck('8.3 MEMORY len<3000', len(mem) < 3000, str(len(mem)))
ck('8.4 MEMORY 3241+3242', '3241 收官（lexical bridge' in mem and '3242 已预注册' in mem)

# 9 registry untouched
ck('9.1 registry v1.33 byte sha8', sha8_file(os.path.join(
    SD, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json')) == 'a30215fd')

n_fail = sum(1 for _, ok, _ in CHECKS if not ok)
for name, ok, detail in CHECKS:
    print('%s %s %s' % ('PASS' if ok else 'FAIL', name, detail))
print('TOTAL: %d checks, %d FAILS' % (len(CHECKS), n_fail))
