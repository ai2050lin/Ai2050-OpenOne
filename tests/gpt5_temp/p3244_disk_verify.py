# -*- coding: utf-8 -*-
"""Phase 3244 independent disk verify (read-only).
Re-derives G0 / P1_pull / R1_clean seed-7 fidelities from raw collect npz
and checks all sealed artifacts + ledger chain + docs."""
import os, sys, json, hashlib
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
SD = os.path.join(ROOT, r'tests\glm5\result\rdc_query_construction_20260913')
RD = os.path.join(SD, r'phase3244\g5h7_bridge_attribution')
RES = os.path.join(RD, 'result.json')
LED = os.path.join(ROOT, r'research\gpt5\atlas\atlas_ledger.json')
MEMO = os.path.join(ROOT, r'research\gpt5\docs\AGI_GPT5_MEMO.md')
REG = os.path.join(SD, r'phase3238\g5h1_census_rank\atlas_registry_v1_33.json')
RESULTS = []


def ck(tag, ok, extra=''):
    RESULTS.append('%s %s %s' % ('PASS' if ok else 'FAIL', tag, extra))
    return ok


R = json.load(open(RES, encoding='utf-8'))
AGG = R['aggregate']
D = R['delta_vs_G0']

# ---------- 1. sealed result ----------
ck('1.1 verdict/sha', R['verdict'] == 'bridge_attribution_registered'
   and R['design_sha8'] == '2392a57d' and R['res_sha8'] == '482eeaa4')
G = R['gates']
ck('1.2 gates', G['G1_replication'] and G['G2_attribution']
   and G['clause_a_span_extension'] and G['clause_b_rescue']
   and G['G3_determinism'] and G['G4_registry_untouched']
   and G['determinism_diff'] == 0.0 and G['d40_vs_sealed_3240'] is True)
A = R['anchors']
ck('1.3 anchors', A['d83'] == 0.0 and A['d_tc'] == 0.0 and A['drift69'] == 0.0
   and A['d88'] == 0.0 and A['d89'] < 1e-6 and A['kout'] == 35)
E = json.load(open(os.path.join(RD, 'execution.json'), encoding='utf-8'))
pc = E['design'].get('prereg_correction', '')
ck('1.4 prereg_correction 2-stage', pc.count('(1)') == 1 and pc.count('(2)') == 1
   and 'SEMANTIC' in pc and 'artifact' in pc.lower())
ck('1.5 G0 d40 max == 0.0', AGG['G0']['d_vs_sealed_3240_max'] == 0.0)
ck('1.6 d41 artifact in band', 0 < AGG['G1e_rep']['d_vs_sealed_3241_max'] < 0.05)
ck('1.7 v1 artifact archived',
   os.path.exists(os.path.join(RD, 'result_formal_v1_anchor_artifact.json'))
   and os.path.exists(os.path.join(RD, 'execution_v1.json')))

# ---------- 2. ledger chain ----------
PREV_POST = '4b5b12a8'
led = json.load(open(LED, encoding='utf-8'))
e396 = [e for e in led['measurements'] if e.get('phase') == 3244]
ck('2.1 ledger n=396 + has 3244', len(led['measurements']) == 396 and len(e396) == 1)
led['ledger_sha256_8'] = PREV_POST
rec = hashlib.sha256(json.dumps(led, ensure_ascii=False, indent=1)
                     .encode('utf-8')).hexdigest()[:8]
cur = json.load(open(LED, encoding='utf-8'))['ledger_sha256_8']
ck('2.2 ledger chain recompute', rec == cur == '530d0ca2', 'post=%s' % cur)

# ---------- 3. docs ----------
mb = open(MEMO, 'rb').read()
ck('3.1 MEMO 3244 section + artifact note',
   mb.count(b'## Phase 3244:') == 1 and '伪影'.encode('utf-8') in mb
   and '+0.989'.encode('utf-8') in mb)
ws = open(os.path.join(ROOT, r'.workbuddy\memory\2026-10-20.md'), encoding='utf-8').read()
us = open(r'C:\Users\Admin\.workbuddy\user-8ae6cef1-1e4d-4c0f-982f-7edf906d7def-personal'
          r'\memory\2026-10-20.md', encoding='utf-8').read()
ck('3.2 dual wlog 3244', '3244 实体桥归因完成' in ws and '3244 实体桥归因完成' in us)
mem = open(os.path.join(ROOT, r'.workbuddy\memory\MEMORY.md'), encoding='utf-8').read()
ck('3.3 MEMORY 3244 block', '3244 收官（bridge attribution' in mem and len(mem) < 3000,
   'len=%d' % len(mem))

# ---------- 4. registry untouched ----------
def sha8f(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:8]


ck('4.1 registry v1.33 byte sha', sha8f(REG) == 'a30215fd')

# ---------- 5. independent re-derivation (seed 7, raw collect npz) ----------
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
Amat = Xtr.T @ Xtr + LAM * np.eye(Xtr.shape[1], dtype=np.float32)
W = np.linalg.solve(Amat, Xtr.T @ Y[tr_rows])
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

# family bases (3157 constructions + census k90)
z57 = np.load(os.path.join(SD, r'phase3157\g2p2_transform_algebra_commutator\qwen3-4b\collect.npz'))
H57 = z57['H'].astype(np.float32)
EI = z57['ei'].astype(int); RL = z57['rel'].astype(int)
PL = z57['pol'].astype(int); CX = z57['ctx'].astype(int)
IDX = {(int(EI[i]), int(RL[i]), int(PL[i]), int(CX[i])): i for i in range(128)}
dC = np.array([H57[IDX[(ei, rl, pl, 1)], 35] - H57[IDX[(ei, rl, pl, 0)], 35]
               for ei in range(16) for rl in range(2) for pl in range(2)])
dCn = dC / (np.linalg.norm(dC, axis=1, keepdims=True) + 1e-18)
U = np.stack([(dCn[ei*4:ei*4+2].mean(0) - dCn[ei*4+2:ei*4+4].mean(0)) for ei in range(16)])
U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-18)
zc = np.load(os.path.join(SD, r'phase3238\g5h1_census_rank\census_spectra.npz'))


def k90_of(fam):
    return int(round(zc['qwen3-4b_%s_meta' % fam][2]))


def basis(M, k):
    _, _, Vt = np.linalg.svd(np.asarray(M, np.float64), full_matrices=False)
    return Vt[:k]


Bdv = basis(U, k90_of('DVU')).astype(np.float32)
Bcc = basis(dCn, k90_of('DCC')).astype(np.float32)
Bd_rel = np.concatenate([Bdv, Bcc], axis=0)
Qcl, _ = np.linalg.qr(Bd_rel.T.astype(np.float32))
Bd = Qcl[:, :Bd_rel.shape[0]].T.astype(np.float32)

# lexical weights
from safetensors import safe_open
with safe_open(os.path.join(ROOT, r'models\hf\qwen3-4b\model-00001-of-00003.safetensors'),
               framework='pt', device='cpu') as f:
    Emb = f.get_tensor('model.embed_tokens.weight').float().numpy()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, r'models\hf\qwen3-4b'))
CLASSES_SEEN = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
CLASSES_OOV = ['乐器', '天气', '运动', '电器']
ENT_SEEN = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
ENT_OOV_PANEL = {
    '乐器': ['钢琴', '小提琴', '吉他', '鼓', '笛子', '二胡', '琵琶', '口琴'],
    '天气': ['雨', '雪', '雷', '雾', '冰雹', '台风', '露水', '霜'],
    '运动': ['足球', '篮球', '乒乓球', '游泳', '跑步', '体操', '拳击', '围棋'],
    '电器': ['电视', '冰箱', '洗衣机', '空调', '微波炉', '电饭煲', '吸尘器', '风扇'],
}
ENTS = [e for cl in (CLASSES_SEEN + CLASSES_OOV) for e in (ENT_SEEN.get(cl, []) + ENT_OOV_PANEL.get(cl, []))]
assert len(ENTS) == 73


def wrow(w):
    ids = tok.encode(w, add_special_tokens=False)
    return np.asarray(Emb[np.array(ids, int)], np.float32).mean(0)


def sfn(cosv):
    e = np.exp(cosv - cosv.max())
    return e / e.sum()


W_cls = np.stack([wrow(w) for w in (CLASSES_SEEN + CLASSES_OOV)])
W_ent = np.stack([wrow(w) for w in ENTS])
ucl = W_cls / (np.linalg.norm(W_cls, axis=1, keepdims=True) + 1e-18)
cos_cm = (ucl @ ucl.T)[:, :6]
uent = W_ent / (np.linalg.norm(W_ent, axis=1, keepdims=True) + 1e-18)
cos_em = (uent @ uent.T)[:, :41]

Yact = Y[oov_rows]
Yb0 = np.stack([B0] * len(oov_rows))
mse_b0 = float(((Yact - Yb0) ** 2).sum(1).mean())

corr_G0 = (B0[None, :] - Yhat0) @ Qs @ Qs.T
corr_full = np.zeros_like(Yhat0)
for j, r in enumerate(oov_rows):
    i, c = PAIRS[r % NP_]
    A_ = B0 + (sfn(cos_cm[c])[:, None] * (d_cls - d_mean[None, :])).sum(0) \
        + (sfn(cos_em[i])[:, None] * (e_ent - e_mean[None, :])).sum(0)
    corr_full[j] = (A_ - Yhat0[j]) @ Qje @ Qje.T
extra = corr_full - corr_G0
extra_clean = extra - (extra @ Bd.T) @ Bd
P_G0 = (Yhat0 + corr_G0).astype(np.float32)
P_pull = (Yhat0 + (B0[None, :] - Yhat0) @ Qje @ Qje.T).astype(np.float32)
P_R1 = (Yhat0 + corr_G0 + extra_clean).astype(np.float32)


def scls_fid(P, which_base):
    Rr = Yact - P
    em_ = float((np.linalg.norm(Rr @ which_base.T, axis=1) ** 2).mean())
    eb_ = float((np.linalg.norm((Yact - Yb0) @ which_base.T, axis=1) ** 2).mean())
    return 1.0 - em_ / max(eb_, 1e-300)


# G0 replication vs sealed 3240 F2 — SCLS base = k90 SVD basis of the
# unit-normalized class-mean matrix (verbatim phase construction)
SCLS_mat = np.stack([np.asarray(np.stack(
    [Y[t * NP_ + PI[(i, c)]] for t in range(NT) for i in range(NE)]).mean(0))
    for c in range(NC)])
SCLS_mat = SCLS_mat / (np.linalg.norm(SCLS_mat, axis=1, keepdims=True) + 1e-18)
B_scls = basis(SCLS_mat, k90_of('SCLS')).astype(np.float32)


# direct comparisons vs sealed 3244 per-seed
ps = R['per_seed']['7']
d_g0 = abs(scls_fid(P_G0, B_scls) - ps['G0']['fam_fid']['SCLS'])
d_pull = abs(scls_fid(P_pull, Bdv) - ps['P1_pull']['fam_fid']['DVU'])
d_r1 = abs(scls_fid(P_R1, B_scls) - ps['R1_clean']['fam_fid']['SCLS'])
ck('5.2 G0 SCLS re-derive (tol 1e-6)', d_g0 < 1e-6, 'diff=%.2e' % d_g0)
ck('5.3 P1_pull DVU re-derive (tol 1e-6)', d_pull < 1e-6, 'diff=%.2e' % d_pull)
ck('5.4 R1_clean SCLS re-derive (tol 1e-6)', d_r1 < 1e-6, 'diff=%.2e' % d_r1)
dvdv = scls_fid(P_R1, Bdv) - scls_fid(P_G0, Bdv)
ck('5.5 R1_clean DVU delta f32-zero (tol 1e-6)', abs(dvdv) < 1e-6, 'delta=%.2e' % dvdv)
dvdv_pull = scls_fid(P_pull, Bdv) - scls_fid(P_G0, Bdv)
ck('5.6 clause_a re-derive: P1_pull DVU damage < -0.05', dvdv_pull < -0.05,
   'delta=%.4f' % dvdv_pull)

# ---------- report ----------
n_fail = sum(1 for r in RESULTS if r.startswith('FAIL'))
out = chr(10).join(RESULTS) + chr(10) + 'VERIFY: %d checks, %d FAILS%s' % (
    len(RESULTS), n_fail, '' if n_fail == 0 else '  <<< ATTENTION')
sys.stdout.write(out + chr(10))
with open(os.path.join(ROOT, r'tests\gpt5_temp\p3244_disk_verify_out.txt'), 'w',
          encoding='utf-8') as f:
    f.write(out + chr(10))
