# -*- coding: utf-8 -*-
"""Debug part 2: hypothesis testing for sealed-3241 G1e historical path (file mode)."""
import os, sys, json
import numpy as np

sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
import phase3241_g5h4_lexical_bridge as P41

sealed_r2 = 0.080435832
sealed_3169 = json.load(open(P41.SRC['p3169_result'], encoding='utf-8'))
sealed_3188 = json.load(open(P41.SRC['p3188_result'], encoding='utf-8'))
sealed_3189 = json.load(open(P41.SRC['p3189_result'], encoding='utf-8'))
m = 'qwen3-4b'
b57 = P41.build_3157(m)
dCn = b57['DCC']

def dC_rows(rl_f):
    return [ei * 4 + rl * 2 + pl for ei in range(16) for rl in range(2)
            for pl in range(2) if rl == rl_f]

dv = P41.unit_mean(dCn[dC_rows(0)]) - P41.unit_mean(dCn[dC_rows(1)])
P41.sealed_3188_dvu[m] = dv / (float(np.linalg.norm(dv)) + 1e-18)
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
ent_mean = {}
for r in tr_rows:
    cls_mean.setdefault(P41.PAIRS[r % P41.NP_][1], []).append(Y[r])
    ent_mean.setdefault(P41.PAIRS[r % P41.NP_][0], []).append(Y[r])
cls_mean = {c: np.stack(v).mean(0) for c, v in cls_mean.items()}
ent_mean = {i: np.stack(v).mean(0) for i, v in ent_mean.items()}
d_cls = np.stack([cls_mean[c] - B0 for c in range(P41.NCS)])
e_ent = np.stack([ent_mean[i] - B0 for i in range(P41.NTR_ENT)])
Qs, _ = np.linalg.qr(d_cls.T.astype(np.float32))
Qs = Qs[:, :P41.NCS].astype(np.float32)
kl_rows_idx = b69['oov_rows']
Xo = np.stack([P41.phi_row(P41.PAIRS[r % P41.NP_][0], P41.PAIRS[r % P41.NP_][1], r // P41.NP_)
               for r in kl_rows_idx])
Yhat0 = Xo @ W
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
Yact = Y[kl_rows_idx]
Yb0 = np.stack([B0] * len(kl_rows_idx))
mse_b0 = float(((Yact - Yb0) ** 2).sum(1).mean())

rng = np.random.RandomState(42)
perm_cls = rng.permutation(P41.NC)
perm_ent = rng.permutation(P41.NE)


def run(hyp, return_corr=False):
    corr = np.zeros_like(Yhat0)
    for j, r in enumerate(kl_rows_idx):
        i, c = P41.PAIRS[r % P41.NP_]
        if hyp == 'H5':
            cc, ii = int(perm_cls[c]), int(perm_ent[i])
        else:
            cc, ii = c, i
        A = B0.copy()
        if hyp in ('H6', 'H7'):
            w_c = P41.linear_weights(cos_cm[cc])
        else:
            w_c = P41.softmax_weights(cos_cm[cc])
        dc = d_cls if hyp in ('H2', 'H3') else (d_cls - d_mean[None, :])
        A = A + (w_c[:, None] * dc).sum(0)
        if hyp in ('H6', 'H8'):
            w_e = P41.linear_weights(cos_em[ii])
        else:
            w_e = P41.softmax_weights(cos_em[ii])
        de = e_ent if hyp in ('H1', 'H3') else (e_ent - e_mean[None, :])
        A = A + (w_e[:, None] * de).sum(0)
        corr[j] = (A - Yhat0[j]) @ Qje @ Qje.T
    if return_corr:
        return corr
    P = (Yhat0 + corr).astype(np.float32)
    r2 = 1.0 - float(((Yact - P) ** 2).sum(1).mean()) / mse_b0
    cn = float(np.linalg.norm(corr))
    return r2, cn


out = []
c0 = run('H0_verbatim', return_corr=True)
# component norms
pull = (B0[None, :] - Yhat0) @ Qje @ Qje.T
i0, cc0 = P41.PAIRS[kl_rows_idx[0] % P41.NP_]
w_c0 = P41.softmax_weights(cos_cm[cc0])
off_c0 = (w_c0[:, None] * (d_cls - d_mean[None, :])).sum(0)
w_e0 = P41.softmax_weights(cos_em[i0])
off_e0 = (w_e0[:, None] * (e_ent - e_mean[None, :])).sum(0)
out.append('||d_mean||=%.6f ||e_mean||=%.6f' % (np.linalg.norm(d_mean), np.linalg.norm(e_mean)))
out.append('d_cls row norms=%s' % np.round(np.linalg.norm(d_cls, axis=1), 4).tolist())
out.append('e_ent row norms[:8]=%s' % np.round(np.linalg.norm(e_ent, axis=1)[:8], 4).tolist())
out.append('row0: ||off_c||=%.6e ||off_e||=%.6e' % (np.linalg.norm(off_c0), np.linalg.norm(off_e0)))
out.append('row0: ||corr_H0||=%.6f ||pull_only||=%.6f max|corr-pull|=%.6e'
           % (np.linalg.norm(c0[0]), np.linalg.norm(pull[0]),
              float(np.max(np.abs(c0[0] - pull[0])))))
out.append('global: max|corr_H0 - pull_only| = %.6e'
           % float(np.max(np.abs(c0 - pull))))

for hyp in ('H0_verbatim', 'H6_linear_both', 'H7_linear_cls', 'H8_linear_ent'):
    r2, cn = run(hyp)
    out.append('%s r2=%.9f corr_norm=%.4f %s'
               % (hyp, r2, cn, 'MATCH' if abs(r2 - sealed_r2) < 1e-6 else ''))
out.append('sealed_3241 r2=%.9f' % sealed_r2)
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\p3244_dbg2_out.txt', 'w',
          encoding='utf-8') as f:
    f.write(chr(10).join(out) + chr(10))
print(chr(10).join(out))
