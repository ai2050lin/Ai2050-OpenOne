# -*- coding: utf-8 -*-
"""Phase 2941 preflight round 2: geometric decomposition of the
dir35 readout shift under null0. Which fraction is predicted by
the v3 coordinate shift (x cos(v3,u35)) vs other U8 bases vs
residual? Uses only existing npz artifacts.

Also reads 2937 result.json for the rewrite-caliber reference.
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
R2937 = os.path.join(BASE, 'phase2937', 'scale_collapse',
                     'result.json')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\phase2941_preflight2_report.txt')

lines = []


def log(m):
    lines.append(m)
    print(m, flush=True)


z39 = np.load(P2939, allow_pickle=True)
z27 = np.load(P2927, allow_pickle=True)
z40 = np.load(P2940, allow_pickle=True)

Vt8 = z39['Vt8'].astype(np.float64)
conds39 = [str(s) for s in z39['cond_names']]
coords = z39['coords'].astype(np.float64)
proj35 = z39['proj_dir35'].astype(np.float64)
nfin = z39['fin_norm'].astype(np.float64)
if39 = conds39.index('func')
i0 = conds39.index('null0')
lab = np.asarray(z27['labels_lang']).astype(int)
dirs_word = z27['dirs_word'].astype(np.float64)
u35 = dirs_word[35]
v3 = Vt8[2]
cosk = Vt8 @ u35  # (8,)
log('cos(v_k, u35): %s' % np.round(cosk, 4).tolist())

# ---- readout shift decomposition ----
dp35 = proj35[i0] - proj35[if39]          # actual per-word shift
dcks = coords[i0] - coords[if39]           # (57, 8) coord shifts
pred_u8 = dcks @ cosk                      # U8-predicted shift
pred_v3 = dcks[:, 2] * cosk[2]             # v3-only term
resid = dp35 - pred_u8                     # outside-U8 part
log('actual dp35: med %+.3f mean %+.3f std %.3f'
    % (float(np.median(dp35)), float(dp35.mean()),
       float(dp35.std())))
log('U8-pred  : med %+.3f mean %+.3f std %.3f'
    % (float(np.median(pred_u8)), float(pred_u8.mean()),
       float(pred_u8.std())))
log('v3-term  : med %+.3f mean %+.3f std %.3f'
    % (float(np.median(pred_v3)), float(pred_v3.mean()),
       float(pred_v3.std())))
log('resid    : med %+.3f mean %+.3f std %.3f'
    % (float(np.median(resid)), float(resid.mean()),
       float(resid.std())))
var_dp = float(((dp35 - dp35.mean()) ** 2).sum())
var_u8 = float(((pred_u8 - pred_u8.mean()) ** 2).sum())
var_v3 = float(((pred_v3 - pred_v3.mean()) ** 2).sum())
var_res = float(((resid - resid.mean()) ** 2).sum())
log('variance shares of dp35: U8 %.3f | v3-only %.3f | resid %.3f'
    % (var_u8 / max(var_dp, 1e-30), var_v3 / max(var_dp, 1e-30),
       var_res / max(var_dp, 1e-30)))
# per-basis contribution to mean shift
mean_contrib = dcks.mean(0) * cosk
log('mean-shift contribution per basis k: %s'
    % np.round(mean_contrib, 3).tolist())
log('sum of U8 mean contributions %+.3f vs actual mean %+.3f'
    % (float(mean_contrib.sum()), float(dp35.mean())))
# per-word correlation
def _rho(a, b):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    den = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den > 1e-30 else 0.0
log('Spearman(dp35, U8pred) %.4f | Spearman(dp35, v3term) %.4f'
    % (_rho(dp35, pred_u8), _rho(dp35, pred_v3)))

# ---- separation shift decomposition ----
m0 = lab == 0
m1 = lab == 1
sep_f = float(proj35[if39][m0].mean() - proj35[if39][m1].mean())
sep_n = float(proj35[i0][m0].mean() - proj35[i0][m1].mean())
d_sep = sep_n - sep_f
sep_pred_u8 = float((dcks[m0].mean(0) - dcks[m1].mean(0)) @ cosk)
sep_pred_v3 = float((dcks[m0].mean(0) - dcks[m1].mean(0))[2]
                    * cosk[2])
log('sep: func %.3f null0 %.3f delta %+.3f | U8pred %+.3f '
    '(v3 part %+.3f)' % (sep_f, sep_n, d_sep, sep_pred_u8,
                         sep_pred_v3))

# ---- scales for injection design ----
c3f = coords[if39][:, 2]
log('c3 func: med %.2f std %.2f range [%.2f, %.2f]'
    % (float(np.median(c3f)), float(c3f.std()),
       float(c3f.min()), float(c3f.max())))
log('d3 (2940) vs dcks[:,2] max diff: %.2e'
    % float(np.abs(z40['d3'].astype(np.float64)
                   - dcks[:, 2]).max()))
log('fin_norm func: med %.1f' % float(np.median(nfin[if39])))
log('proj35 func: med %.2f std %.2f'
    % (float(np.median(proj35[if39])),
       float(proj35[if39].std())))
# per-word proj35 std vs shift: SNR of a -0.227*19.5 = -4.4 mean
# v3 push against per-word spread
log('proj35 func per-word std %.2f -> v3 mean push %.2f is %.1f%%'
    % (float(proj35[if39].std()),
       float(pred_v3.mean()),
       100.0 * abs(float(pred_v3.mean()))
       / float(proj35[if39].std())))

# ---- 2937 reference calibers ----
try:
    with open(R2937, encoding='utf-8') as f:
        r37 = json.load(f)
    log('2937 verdict: %s' % r37.get('final_verdict'))
    for key in ('P1', 'P2', 'P3'):
        if key in r37 and r37[key]:
            log('2937 %s: %s' % (key, json.dumps(
                r37[key], ensure_ascii=False)[:400]))
except Exception as e:
    log('2937 result.json read fail: %r' % e)

with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('WROTE', OUT, flush=True)
