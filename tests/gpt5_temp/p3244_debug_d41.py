# -*- coding: utf-8 -*-
"""Debug: diff my corr_full (phase3244) vs verbatim 3241 G1e loop, seed 7."""
import os, sys, json
import numpy as np

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3244_g5h7_bridge_attribution as P44
import phase3241_g5h4_lexical_bridge as P41

# point both modules' SRC at same sealed files (they already are)
P41.SMOKE = True   # avoid rewriting anything formal; only used for freeze guard
# Build shared machinery via P41's own functions (verbatim 3241 path)
sealed_3169 = json.load(open(P41.SRC['p3169_result'], encoding='utf-8'))
sealed_3183 = json.load(open(P41.SRC['p3183_result'], encoding='utf-8'))
sealed_3188 = json.load(open(P41.SRC['p3188_result'], encoding='utf-8'))
sealed_3189 = json.load(open(P41.SRC['p3189_result'], encoding='utf-8'))

m = 'qwen3-4b'
b57 = P41.build_3157(m)
dCn = b57['DCC']

def dC_rows(rl_f):
    return [ei * 4 + rl * 2 + pl for ei in range(16) for rl in range(2)
            for pl in range(2) if rl == rl_f]

dv_vec = P41.unit_mean(dCn[dC_rows(0)]) - P41.unit_mean(dCn[dC_rows(1)])
dv_vec = dv_vec / (float(np.linalg.norm(dv_vec)) + 1e-18)
P41.sealed_3188_dvu[m] = dv_vec
b69 = P41.build_3169(m, sealed_3169, sealed_3188, sealed_3189)
P41.load_lens()
Y = b69['Y']
s = 7
tr_rows = b69['tr_base'][s]
Xtr = np.stack([P41.phi_row(P41.PAIRS[r % P41.NP_][0], P41.PAIRS[r % P41.NP_][1], r // P41.NP_)
                for r in tr_rows])
W = P41.ridge_primal(Xtr, Y[tr_rows])
B0 = Y[tr_rows].mean(0)
cls_mean = {}
for r in tr_rows:
    cls_mean.setdefault(P41.PAIRS[r % P41.NP_][1], []).append(Y[r])
cls_mean = {c: np.stack(v).mean(0) for c, v in cls_mean.items()}
ent_mean = {}
for r in tr_rows:
    ent_mean.setdefault(P41.PAIRS[r % P41.NP_][0], []).append(Y[r])
ent_mean = {i: np.stack(v).mean(0) for i, v in ent_mean.items()}
d_cls = np.stack([cls_mean[c] - B0 for c in range(P41.NCS)])
e_ent = np.stack([ent_mean[i] - B0 for i in range(P41.NTR_ENT)])
Qs, _ = np.linalg.qr(d_cls.T.astype(np.float32))
Qs = Qs[:, :P41.NCS].astype(np.float32)

kl_rows_idx = b69['oov_rows']
Xo = np.stack([P41.phi_row(P41.PAIRS[r % P41.NP_][0], P41.PAIRS[r % P41.NP_][1], r // P41.NP_)
               for r in kl_rows_idx])
Yhat0 = Xo @ W

# ---- verbatim 3241 bridge weights ----
cls_words = P41.CLASSES_SEEN + P41.CLASSES_OOV
W_cls = np.stack([P41.token_row(w) for w in cls_words])
W_ent = np.stack([P41.token_row(w) for w in P41.ENTS])
cos_cm = np.array([[float(P41.unit(W_cls[a]) @ P41.unit(W_cls[b])) for b in range(P41.NCS)]
                   for a in range(P41.NC)])
cos_em = np.array([[float(P41.unit(W_ent[a]) @ P41.unit(W_ent[b])) for b in range(P41.NTR_ENT)]
                   for a in range(P41.NE)])
d_mean = d_cls.mean(0)
e_mean = e_ent.mean(0)
SJE = np.concatenate([d_cls, e_ent], axis=0)
Qje, _ = np.linalg.qr(SJE.T.astype(np.float32))
Qje = Qje[:, :SJE.shape[0]].astype(np.float32)

# 3241 verbatim loop for G1e
corr_41 = np.zeros_like(Yhat0)
for j, r in enumerate(kl_rows_idx):
    i, c = P41.PAIRS[r % P41.NP_]
    cc = c
    ii = i
    A = B0.copy()
    w_c = P41.softmax_weights(cos_cm[cc])
    A = A + (w_c[:, None] * (d_cls - d_mean[None, :])).sum(0)
    w_e = P41.softmax_weights(cos_em[ii])
    A = A + (w_e[:, None] * (e_ent - e_mean[None, :])).sum(0)
    corr_41[j] = (A - Yhat0[j]) @ Qje @ Qje.T

# my 3244 corr_full
corr_44 = np.zeros_like(Yhat0)
for j, r in enumerate(kl_rows_idx):
    i, c = P41.PAIRS[r % P41.NP_]
    w_c = P41.softmax_weights(cos_cm[c])
    dA_c = (w_c[:, None] * (d_cls - d_mean[None, :])).sum(0)
    w_e = P41.softmax_weights(cos_em[i])
    dA_e = (w_e[:, None] * (e_ent - e_mean[None, :])).sum(0)
    corr_44[j] = (B0 + dA_c + dA_e - Yhat0[j]) @ Qje @ Qje.T

g = np.max(np.abs(corr_41 - corr_44))
print('corr diff max: %.3e' % g)
if g > 0:
    jj = int(np.argmax(np.abs(corr_41 - corr_44).max(1)))
    i, c = P41.PAIRS[kl_rows_idx[jj] % P41.NP_]
    print('worst row', jj, 'pair', i, c)
    w_c = P41.softmax_weights(cos_cm[c])
    w_e = P41.softmax_weights(cos_em[i])
    dA_c = (w_c[:, None] * (d_cls - d_mean[None, :])).sum(0)
    dA_e = (w_e[:, None] * (e_ent - e_mean[None, :])).sum(0)
    A41 = B0.copy()
    A41 = A41 + dA_c
    A41 = A41 + dA_e
    A44 = B0 + dA_c + dA_e
    print('A diff max: %.3e' % np.max(np.abs(A41 - A44)))
    print('anchor proj diff: %.3e' % np.max(np.abs((A41 - Yhat0[jj]) @ Qje @ Qje.T
                                                   - (A44 - Yhat0[jj]) @ Qje @ Qje.T)))
    print('corr_41 row norm %.6f  corr_44 row norm %.6f'
          % (np.linalg.norm(corr_41[jj]), np.linalg.norm(corr_44[jj])))
    # decompose the corr diff onto families: constant vector?
    dv = corr_41 - corr_44
    print('corr diff mean-vector norm %.3e ; per-row norm mean %.3e std %.3e'
          % (np.linalg.norm(dv.mean(0)), np.linalg.norm(dv, axis=1).mean(),
             np.linalg.norm(dv, axis=1).std()))

# ---- decisive: compute fam_fid for G1e here, compare to sealed 3241 AND 3244 ----
import hashlib
SRC_DIR = P41.SRC_DIR
FAMS = P41.FAMS

# rebuild fams/bases exactly as both phases do
v32_path = os.path.join(SRC_DIR, r'phase3237\g5g31_registry_landing\atlas_registry_v1_32.json')
ftr12 = [f for f in json.load(open(v32_path, encoding='utf-8'))['features']
         if f['id'] == 'FTR-12'][0]
d1 = int(ftr12['values']['d1_4b'])
massive = np.zeros((1, b57['D']), np.float32)
massive[0, d1] = 1.0
fams = {'DVU': b57['DVU'], 'DCC': b57['DCC'], 'ENT': b57['ENT'],
        'USTAR': b69['USTAR'], 'SCLS': b69['SCLS'], 'MASSIVE': massive}

def spectrum(M):
    sv = np.linalg.svd(np.asarray(M, np.float64), compute_uv=False)
    lam = sv ** 2
    tot = lam.sum()
    pr = float(tot * tot / (lam ** 2).sum() + 1e-300)
    cum = np.cumsum(lam) / tot
    k90 = int(np.searchsorted(cum, 0.9) + 1)
    top1 = float(lam[0] / tot)
    return dict(PR=pr, k90=k90, top1=top1, n=int(len(sv)))

def subspace_basis(M, k):
    _, _, Vt = np.linalg.svd(np.asarray(M, np.float64), full_matrices=False)
    return Vt[:k]

bases = {fam: subspace_basis(fams[fam], spectrum(fams[fam])['k90']) for fam in FAMS}

Yact = Y[kl_rows_idx]
Yb0 = np.stack([B0] * len(kl_rows_idx))
mse_b0 = float(((Yact - Yb0) ** 2).sum(1).mean())

for tag, corr in (('verbatim41', corr_41), ('formula44', corr_44)):
    P = (Yhat0 + corr).astype(np.float32)
    mse_r = float(((Yact - P) ** 2).sum(1).mean())
    r2 = 1.0 - mse_r / mse_b0
    Rr = Yact - P
    ff = {}
    for fam in FAMS:
        Bf = bases[fam].astype(np.float32)
        em_ = float((np.linalg.norm(Rr @ Bf.T, axis=1) ** 2).mean())
        eb_ = float((np.linalg.norm((Yact - Yb0) @ Bf.T, axis=1) ** 2).mean())
        ff[fam] = 1.0 - em_ / max(eb_, 1e-300)
    print(tag, 'r2=%.9f' % r2, {f: round(v, 6) for f, v in ff.items()})

s41 = json.load(open(os.path.join(SRC_DIR, r'phase3241\g5h4_lexical_bridge\result.json'), encoding='utf-8'))
s44 = json.load(open(os.path.join(SRC_DIR, r'phase3244\g5h7_bridge_attribution\result.json'), encoding='utf-8'))
print('sealed 3241 G1e seed7 r2=%.9f' % s41['per_seed']['7']['G1e']['r2_oov'])
print('sealed 3244 G1e_rep seed7 r2=%.9f' % s44['per_seed']['7']['G1e_rep']['r2_oov'])
