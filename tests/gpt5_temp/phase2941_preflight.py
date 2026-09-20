# -*- coding: utf-8 -*-
"""Phase 2941 preflight (discipline 10): criterion reachability
check BEFORE freezing PREREG. Uses only existing artifacts
(2939/2927/2940/2937 npz). No new forward passes.

Outputs report to phase2941_preflight_report.txt.
"""
import json
import os

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
P2939 = os.path.join(BASE, 'phase2939', 'rotation_target',
                     'rotation_target.npz')
P2927 = os.path.join(BASE, 'phase2927', 'probe_relativity',
                     'probe_relativity.npz')
P2940 = os.path.join(BASE, 'phase2940', 'v3_decode',
                     'v3_decode.npz')
P2937 = os.path.join(BASE, 'phase2937', 'scale_collapse',
                     'scale_collapse.npz')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2941_preflight_report.txt')

lines = []


def log(m):
    lines.append(m)
    print(m, flush=True)


z39 = np.load(P2939, allow_pickle=True)
log('2939 keys: %s' % sorted(z39.files))
z27 = np.load(P2927, allow_pickle=True)
log('2927 keys: %s' % sorted(z27.files))
z40 = np.load(P2940, allow_pickle=True)
log('2940 keys: %s' % sorted(z40.files))
z37 = np.load(P2937, allow_pickle=True)
log('2937 keys: %s' % sorted(z37.files))

Vt8 = z39['Vt8'].astype(np.float64)          # (8, 2560)
conds39 = [str(s) for s in z39['cond_names']]
coords = z39['coords'].astype(np.float64)     # (6, 57, 8)
proj35 = z39['proj_dir35'].astype(np.float64)  # (6, 57)
nfin = z39['fin_norm'].astype(np.float64)      # (6, 57)
if39 = conds39.index('func')
i0 = conds39.index('null0')
words = [str(w) for w in z39['words']]

dirs_word = z27['dirs_word'].astype(np.float64)  # (36, 2560)
v3 = Vt8[2]
u35 = dirs_word[35]
cos_v3_dir35 = float(v3 @ u35)
log('cos(v3, dirs_word[35]) = %.6f' % cos_v3_dir35)
# full 8-basis vs dir35
cos_all = Vt8 @ u35
log('cos(v_k, dir35) k=1..8: %s'
    % np.round(cos_all, 4).tolist())

w_li = z40['w_li'].astype(np.float64)  # (36,)
order = np.argsort(-w_li)
log('w_li top6 layers: %s'
    % [(int(li), round(float(w_li[li]), 3)) for li in order[:6]])
L_inj = int(order[0])
log('L_inj = argmax w_li = L%d (w=%.3f)' % (L_inj, w_li[L_inj]))

# ---- synthetic prediction: does v3-push alone reproduce the
# ---- null direction rewrite (cos collapse)?
# Approximation: x = x_perp + sum_k c_k v_k, where x_perp is the
# component outside U8, assumed preserved func->null (2938).
# cos(x_null, x_func) ~ (|xp|^2 + cf.cn) / (|xf| |xn|)
cf = coords[if39]        # (57, 8)
cn = coords[i0]          # (57, 8)
nf = nfin[if39]
nn = nfin[i0]
xpf2 = np.maximum(nf ** 2 - (cf ** 2).sum(1), 0.0)
c_dot_actual = (cf * cn).sum(1)
cos_actual = (xpf2 + c_dot_actual) / np.maximum(nf * nn, 1e-30)
log('approx cos(x_null0, x_func): median %.4f min %.4f max %.4f'
    % (float(np.median(cos_actual)), float(cos_actual.min()),
       float(cos_actual.max())))

# synthetic: c' = cf but c3' = cf3 + delta (delta = mean actual
# per-word d3 shift along v3; also try class-mean 19.283)
d3 = z40['d3'].astype(np.float64)  # (57,)
log('d3 (2940 per-word null-minus-func c3): median %+.3f '
    'mean %+.3f min %+.3f max %+.3f'
    % (float(np.median(d3)), float(d3.mean()),
       float(d3.min()), float(d3.max())))

for tag, delta in [('d3_perword', None), ('const+19.283', 19.283),
                   ('const+8', 8.0), ('const+16', 16.0),
                   ('const+32', 32.0)]:
    if delta is None:
        c3n = cf[:, 2] + d3
        eff = d3
    else:
        c3n = cf[:, 2] + delta
        eff = np.full(57, delta)
    cn_syn = cf.copy()
    cn_syn[:, 2] = c3n
    # norm of synthetic vector: perp preserved + new U8 coords
    nsyn = np.sqrt(xpf2 + (cn_syn ** 2).sum(1))
    c_dot_syn = (cf * cn_syn).sum(1)
    cos_syn = (xpf2 + c_dot_syn) / np.maximum(nf * nsyn, 1e-30)
    drop_syn = 1.0 - cos_syn
    drop_act = 1.0 - cos_actual
    ratio = float(np.median(drop_syn) / max(float(np.median(
        drop_act)), 1e-30))
    log('synthetic v3-push [%s]: cos med %.4f (null actual med '
        '%.4f) | drop med %.4f vs actual %.4f | ratio %.3f'
        % (tag, float(np.median(cos_syn)),
           float(np.median(cos_actual)),
           float(np.median(drop_syn)),
           float(np.median(drop_act)), ratio))

# also: v3-push with ALL delta_e inflow (v1..v8 actual shifts)?
# reference only
log('')
log('---- what does 2937 npz contain for measured rewrite ----')
for k in z37.files:
    a = np.asarray(z37[k])
    log('2937[%s] shape %s dtype %s'
        % (k, a.shape, a.dtype))
    if a.ndim == 1 and a.size <= 12:
        log('   values: %s' % np.round(a.astype(float), 4))
log('')
log('---- proj35 sep under func/null0 (2939) ----')
lab = np.asarray(z27['labels_lang']).astype(int)
sep_f = float(proj35[if39][lab == 0].mean()
              - proj35[if39][lab == 1].mean())
sep_n = float(proj35[i0][lab == 0].mean()
              - proj35[i0][lab == 1].mean())
log('sep proj35: func %.3f null0 %.3f (delta %+.3f)'
    % (sep_f, sep_n, sep_n - sep_f))
log('proj35 func: med %.3f range [%.3f, %.3f]'
    % (float(np.median(proj35[if39])),
       float(proj35[if39].min()), float(proj35[if39].max())))
log('c3 func coords: med %.3f range [%.3f, %.3f]'
    % (float(np.median(cf[:, 2])), float(cf[:, 2].min()),
       float(cf[:, 2].max())))
log('fin_norm func: med %.1f range [%.1f, %.1f]'
    % (float(np.median(nf)), float(nf.min()), float(nf.max())))

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('WROTE', OUT, flush=True)
