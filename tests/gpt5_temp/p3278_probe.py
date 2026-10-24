# -*- coding: utf-8 -*-
"""APPX-1 probe (zero GPU): reproduce 14b R_logic x K_entity eff_ge05=2 from
sealed collects and localize the two shared directions (K_ent rows x R_logic
rows, layer profile, vocab projection). Feeds phase 3278 protocol."""
import hashlib
import io
import json
import os
import sys

import numpy as np

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

ROOT = r'D:\AI2050\Ai2050-OpenOne'
B = os.path.join(ROOT, 'tests', 'glm5', 'result', 'rdc_query_construction_20260913')
OUTP = os.path.join(ROOT, 'tests', 'gpt5_temp', 'p3278_probe_out.txt')

# panel (3151/3152 verbatim, from p3166)
CLASSES = ['水果', '动物', '交通工具', '家具', '金属', '颜色']
ENT = {
    '水果': ['苹果', '香蕉', '梨', '西瓜', '葡萄', '草莓', '芒果', '柠檬'],
    '动物': ['狗', '猫', '老虎', '大象', '兔子', '猴子', '马', '牛'],
    '交通工具': ['汽车', '火车', '飞机', '摩托车', '卡车', '地铁'],
    '家具': ['桌子', '椅子', '床', '沙发', '地毯', '窗帘'],
    '金属': ['铁', '铜', '铝', '金', '银', '锌', '铅'],
    '颜色': ['红', '蓝', '绿', '黄', '黑', '白'],
}
ENTS = [e for cl in CLASSES for e in ENT[cl]]
CLS_OF = [CLASSES.index(cl) for cl in CLASSES for e in ENT[cl]]
NE, NC, NT = len(ENTS), 6, 3
PAIRS_FULL = [(i, c) for i in range(NE) for c in range(NC)]
NP = len(PAIRS_FULL)

SRC_3152 = B + r'\phase3152\g1p2_tri_model_k1\qwen3-14b\collect.npz'
SRC_3157 = B + r'\phase3157\g2p2_transform_algebra_commutator\qwen3-14b\collect.npz'
SRC_3158 = B + r'\phase3158\g4p1_output_equivalence_class\qwen3-14b\collect.npz'
MDIR = os.path.join(ROOT, 'models', 'hf', 'Qwen3-14B')
MDIR_NF4 = os.path.join(ROOT, 'models', 'hf', 'Qwen3-14B-bnb-nf4')

L = []


def log(s):
    L.append(str(s))


def unit(x):
    return x / float(np.linalg.norm(x))


def krows(A):
    Qa, _ = np.linalg.qr(np.asarray(A, dtype=np.float64).T)
    return Qa.T  # row-form orthonormal basis of rowspace


def kross(A, Bd):
    Qa, _ = np.linalg.qr(A.T)
    Qb, _ = np.linalg.qr(Bd.T)
    s = np.linalg.svd(Qa.T @ Qb, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return s, float(np.degrees(np.arccos(s[0])))


# ---- 1. K_readout (3158 top64) + K_entity (3157) ----
z58 = np.load(SRC_3158)
K_read = z58['top64'].astype(np.float64).T          # (64, D)
z57 = np.load(SRC_3157)
H57 = z57['H'].astype(np.float64)
NL57 = H57.shape[1] - 1
X57 = H57[:, NL57, :]
Xc = X57 - X57.mean(0, keepdims=True)
_, sv_ent, Vh57 = np.linalg.svd(Xc, full_matrices=False)
K_ent = Vh57[:8]

# ---- 2. R_logic (3166 build_R verbatim, k_final=NL) ----
cfg = json.load(io.open(os.path.join(MDIR, 'config.json'), encoding='utf-8'))
NL = int(cfg['num_hidden_layers'])
zm = np.load(SRC_3152)
H16 = zm['H']
MARG = zm['marg']
NTn, NPn, NHn, Dn = H16.shape
assert (NTn, NPn, NHn) == (NT, NP, NL + 1), (H16.shape, NL)
ent_rows = {i: [] for i in range(NE)}
for i in range(NE):
    for t in range(NT):
        for pi, (ei, ci) in enumerate(PAIRS_FULL):
            if ei == i:
                ent_rows[i].append(t * NP + pi)
y = np.zeros(NT * NP, bool)
for t in range(NT):
    for pi, (ei, ci) in enumerate(PAIRS_FULL):
        if ci == CLS_OF[ei]:
            y[t * NP + pi] = True
Hk = H16[:, :, NL, :].reshape(NT * NP, Dn).astype(np.float64)
ds = []
for i in range(NE):
    tr = [r for r in ent_rows[i] if y[r]]
    fa = [r for r in ent_rows[i] if not y[r]]
    ds.append(Hk[tr].mean(0) - Hk[fa].mean(0))
Dm = np.stack(ds)
scales = np.linalg.norm(Dm, axis=1)
DmU = np.stack([unit(v) for v in Dm])
Dc = DmU - DmU.mean(0, keepdims=True)
_, sv_r, Vh_r = np.linalg.svd(Dc, full_matrices=False)
R_logic = Vh_r[:8]

# ---- 3. reproduce 3166 14b census ----
pairs14 = {}
for a, bn in (('R_logic', 'K_entity'), ('R_logic', 'K_readout'),
              ('K_readout', 'K_entity')):
    s, t1 = kross({'R_logic': R_logic, 'K_entity': K_ent,
                   'K_readout': K_read}[a],
                  {'R_logic': R_logic, 'K_entity': K_ent,
                   'K_readout': K_read}[bn])
    pairs14['%s__%s' % (a, bn)] = (round(t1, 3), int((s ** 2 >= 0.5).sum()),
                                   s.round(6).tolist())
log('repro census: ' + json.dumps(
    {k: v[:2] for k, v in pairs14.items()}, ensure_ascii=False))
# anchors from 3166 result: R_logic__K_entity 26.641/2, R_logic__K_readout
# 71.796/0, K_readout__K_entity 70.235/0
OK1 = abs(pairs14['R_logic__K_entity'][0] - 26.641) < 0.05 and pairs14['R_logic__K_entity'][1] == 2
OK2 = abs(pairs14['R_logic__K_readout'][0] - 71.796) < 0.05
OK3 = abs(pairs14['K_readout__K_entity'][0] - 70.235) < 0.05
log('anchor check: RK=%s RR=%s KR=%s' % (OK1, OK2, OK3))

# ---- 4. localize the 2 shared dirs: SVD of cross-Gram ----
QR, _ = np.linalg.qr(R_logic.T)     # (D,8)
QK, _ = np.linalg.qr(K_ent.T)       # (D,8)
M = QR.T @ QK                        # (8,8)
U, s, Wt = np.linalg.svd(M)
log('cross-gram singulars: ' + str(s.round(6).tolist()))
# shared dim j: left vec a_j (R_logic coords), right vec w_j (K_ent coords)
for j in range(3):
    a = U[:, j]                      # (8,) in R_logic row basis
    w = Wt[j]                        # (8,) in K_ent row basis
    deg_j = float(np.degrees(np.arccos(min(1.0, s[j]))))
    log('shared dim %d: cos=%.4f (%.2f deg) | R_logic coeff=%s | K_ent coeff=%s'
        % (j, s[j], deg_j, a.round(4).tolist(), w.round(4).tolist()))
    # physical rows: top |coeff| rows
    ra = int(np.argmax(np.abs(a)))
    rk = int(np.argmax(np.abs(w)))
    log('   top R_logic row #%d (sv_r=%.3f), top K_ent row #%d (sv_ent=%.3f)'
        % (ra, sv_r[ra], rk, sv_ent[rk]))

# ---- 5. layer profile of shared dirs on R_logic side (p3152 H all slots) ----
# R_logic orthonormal rows QR (8,D). For each layer slot k: build R_logic_k
# contrastive matrix via same ent_rows/y, project its rowspace onto a_j dir.
prof = {}
for k in range(NHn):
    Hkk = H16[:, :, k, :].reshape(NT * NP, Dn).astype(np.float64)
    dsk = []
    for i in range(NE):
        tr = [r for r in ent_rows[i] if y[r]]
        fa = [r for r in ent_rows[i] if not y[r]]
        dsk.append(unit(Hkk[tr].mean(0) - Hkk[fa].mean(0)))
    Dmk = np.stack(dsk)
    Dck = Dmk - Dmk.mean(0, keepdims=True)
    Qk, _ = np.linalg.qr(Dck.T)
    # alignment of layer-k rowspace with each shared dir a_j (R_logic coords)
    Mj = U.T @ (QR.T @ Qk)           # (8, k)
    prof[k] = [float(np.linalg.norm(Mj[j])) for j in range(3)]
top_layers = sorted(range(NHn), key=lambda k: -prof[k][0])[:6]
log('layer profile (shared dim0 norm top layers): ' +
    str([(k, round(prof[k][0], 3)) for k in sorted(top_layers)]))
log('layer profile dim1 top: ' +
    str([(k, round(prof[k][1], 3)) for k in sorted(range(NHn), key=lambda k: -prof[k][1])[:4]]))

# ---- 6. vocab projection of shared dims (physical directions in D space) ----
# physical vector v_j = QR @ a_j (unit in D space); project onto W_U rows
from safetensors import safe_open
want = 'lm_head.weight'
W = None
for sh in sorted(os.listdir(MDIR_NF4)):
    if sh.endswith('.safetensors'):
        with safe_open(os.path.join(MDIR_NF4, sh), framework='pt') as f:
            if want in set(f.keys()):
                W = f.get_tensor(want).float().numpy()
                break
assert W is not None
W = W.astype(np.float64)
# W_U rowspace basis
QW, _ = np.linalg.qr(W.T)            # (D, min(D,V))
Wn = float(np.linalg.norm(W))
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MDIR, local_files_only=True,
                                    trust_remote_code=True, use_fast=True)
for j in range(3):
    v = QR @ U[:, j]
    v = unit(v)
    cos_w = W @ v                     # (V,)
    top = np.argsort(-np.abs(cos_w))[:8]
    toks = []
    for tid in top:
        t = tok.decode([int(tid)])
        toks.append('%s:%.3f' % (t, cos_w[tid]))
    # component inside W rowspace vs total
    vw = QW @ (QW.T @ v)
    log('shared dim %d vocab top8: %s | rowspace frac=%.3f'
        % (j, toks, float(np.linalg.norm(vw))))

# ---- 7. K_ent side: is the shared direction also entity-subspace-generic? ----
# compare K_ent rows vs 4b/glm4 K_ent? (cross-model descriptive only)
log('K_ent singulars: ' + str(sv_ent.round(4).tolist()))
log('R_logic singulars: ' + str(sv_r.round(4).tolist()))

with io.open(OUTP, 'w', encoding='utf-8') as f:
    f.write('\n'.join(L) + '\n')
print('probe done ->', OUTP)
print('\n'.join(L))
