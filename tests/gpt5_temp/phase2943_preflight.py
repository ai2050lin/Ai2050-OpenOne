# -*- coding: utf-8 -*-
"""Phase 2943 preflight: gamma negative-offset anatomy reachability.

Zero-forward, all data on disk (2937/2939/2942 npz). Checks:
  R0 proj consistency 2937 vs 2939 vs 2942 (anchor feasibility)
  R1 OLS beta/gamma rebuild vs 2937 result.json
  R2 gamma identity: mean(dproj_c) = gamma_c + (beta_c-1)*mean(f)
  R3 U8 attribution: P_c = mean(dproj) rebuilt from U8 coords
     vs gamma_c; per-null consistency
  R4 class-asymmetry 3-term decomposition sep contributions:
     constant gamma vs slope (beta-1)*f vs residual
  R5 partial correlation d_inj(s=2) vs d_null0 controlling f
"""
import json
import numpy as np

B = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
     r'\rdc_query_construction_20260913')
OUT = r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp' \
      r'\phase2943_preflight_report.txt'

z37 = np.load(B + r'\phase2937\scale_collapse'
                 r'\scale_collapse.npz', allow_pickle=True)
z39 = np.load(B + r'\phase2939\rotation_target'
                 r'\rotation_target.npz', allow_pickle=True)
z42 = np.load(B + r'\phase2942\u8_joint_injection'
                 r'\u8_joint_injection.npz', allow_pickle=True)
r37 = json.load(open(B + r'\phase2937\scale_collapse'
                        r'\result.json', encoding='utf-8'))
r42 = json.load(open(B + r'\phase2942\u8_joint_injection'
                        r'\result.json', encoding='utf-8'))

lines = []
def log(m):
    lines.append(m)

conds37 = [str(s) for s in z37['cond_names']]
proj37 = z37['proj'].astype(np.float64)      # (6,57)
conds39 = [str(s) for s in z39['cond_names']]
coords = z39['coords'].astype(np.float64)    # (6,57,8)
proj39 = z39['proj_dir35'].astype(np.float64)
Vt8 = z42['Vt8'].astype(np.float64)
u35 = z42['dirs_word'].astype(np.float64)[35]
lab = z42['labels_lang'].astype(int)
scales = z42['scales'].astype(float)
proj_inj = z42['proj_inj'].astype(np.float64)
proj_f0 = z42['proj_func0'].astype(np.float64)
proj_n0 = z42['proj_null0'].astype(np.float64)
c8_f0 = z42['c8_func0'].astype(np.float64)
c8_n0 = z42['c8_null0'].astype(np.float64)
dcks = z42['dcks'].astype(np.float64)

# R0 consistency
i_f = conds37.index('func')
i_n0 = conds37.index('null0')
d_3739 = float(np.abs(proj37 - proj39).max())
d_3742f = float(np.abs(proj37[i_f] - proj_f0).max())
d_3742n = float(np.abs(proj37[i_n0] - proj_n0).max())
log('R0 proj37 vs proj39 max %.3e | vs 2942 func %.3e '
    'null0 %.3e' % (d_3739, d_3742f, d_3742n))
log('R0 conds37=%s conds39=%s' % (conds37, conds39))

# R1 OLS rebuild
log('R1 OLS rebuild vs 2937 result:')
fits = {}
for cn in conds37:
    if cn == 'func':
        continue
    x = proj37[i_f]
    y = proj37[conds37.index(cn)]
    xm, ym = x.mean(), y.mean()
    b = float(((x - xm) * (y - ym)).sum()
              / ((x - xm) ** 2).sum())
    g = float(ym - b * xm)
    fits[cn] = (b, g)
    ref = r37['P2']['fits'][cn]
    log('  %s beta %.4f (ref %.4f) gamma %.4f (ref %.4f)'
        % (cn, b, ref['beta'], g, ref['gamma']))

# R2 identity
log('R2 identity mean(dproj) ?= gamma + (beta-1)*mean(f):')
mf = float(proj37[i_f].mean())
for cn, (b, g) in fits.items():
    md = float(proj37[conds37.index(cn)].mean() - mf)
    pred = g + (b - 1.0) * mf
    log('  %s mean_dproj %.3f pred %.3f gap %.2e'
        % (cn, md, pred, abs(md - pred)))

# R3 U8 attribution: P_c = mean over words of U8-rebuilt
# dproj = (dcoords @ Vt8.T) @ u35
log('R3 U8 attribution (2939 coords):')
for ci, cn in enumerate(conds39):
    if cn == 'func':
        continue
    dc = coords[ci] - coords[conds39.index('func')]
    P = float((dc @ Vt8 @ u35).mean())
    # also raw full-space mean dproj from 2937
    md = float(proj37[conds37.index(cn)].mean() - mf)
    b, g = fits[cn]
    gamma_pred = P - (b - 1.0) * mf
    log('  %s P_u8 %.3f mean_dproj %.3f resid %.3f | '
        'gamma %.3f gamma_pred_from_P %.3f ratio %.3f'
        % (cn, P, md, md - P, g, gamma_pred,
           g / gamma_pred if abs(gamma_pred) > 1e-9 else 0.0))
# direct check: does U8 rebuild match full dproj per word?
dc0 = coords[conds39.index('null0')] - coords[conds39.index('func')]
rebuild_w = dc0 @ Vt8 @ u35
actual_w = proj37[i_n0] - proj37[i_f]
r_word = float(np.corrcoef(rebuild_w, actual_w)[0, 1])
log('R3 null0 per-word U8-rebuild vs actual dproj pearson %.4f '
    'mean(rebuild) %.3f mean(actual) %.3f'
    % (r_word, rebuild_w.mean(), actual_w.mean()))

# R4 class asymmetry 3-term decomposition
log('R4 sep contributions of dproj terms (null0):')
x = proj37[i_f]
b, g = fits['null0']
d = proj37[i_n0] - x
slope_term = (b - 1.0) * x
const_term = np.full_like(x, g)
resid = d - slope_term - const_term
def sep(v):
    return float(v[lab == 0].mean() - v[lab == 1].mean())
log('  sep(d) %.2f | slope %.2f const %.2f resid %.2f'
    % (sep(d), sep(slope_term), sep(const_term), sep(resid)))
log('  mean f lang0 %.2f lang1 %.2f | beta-1 %.4f gamma %.3f'
    % (x[lab == 0].mean(), x[lab == 1].mean(), b - 1.0, g))

# R5 partial correlation: d_inj(s=2) vs d_null0 controlling f
i_s2 = int(np.argmin(np.abs(scales - 2.0)))
d_inj = proj_inj[i_s2] - proj_f0
d_null = proj_n0 - proj_f0
def spear(a, b_):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b_)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])
r_raw = spear(d_inj, d_null)
# partial via residualized Spearman ranks
def partial_spear(a, b_, c):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b_)).astype(float)
    rc = np.argsort(np.argsort(c)).astype(float)
    def res(y, z):
        zm = z.mean()
        beta = ((y - y.mean()) * (z - zm)).sum() \
            / ((z - zm) ** 2).sum()
        return y - y.mean() - beta * (z - zm)
    return float(np.corrcoef(res(ra, rc), res(rb, rc))[0, 1])
r_par = partial_spear(d_inj, d_null, x)
log('R5 d_inj(s=%.1f) vs d_null0: raw spearman %.4f | '
    'partial controlling f %.4f' % (scales[i_s2], r_raw, r_par))
log('R5 mean d_inj %.3f mean d_null %.3f intercept gap %.3f '
    '(null0 gamma %.3f)'
    % (d_inj.mean(), d_null.mean(),
       d_inj.mean() - d_null.mean(), fits['null0'][1]))
# does adding gamma constant to injection close shape gap?
d_inj_g = d_inj + fits['null0'][1]
log('R5 after +gamma shift: mean %.3f spearman vs d_null %.4f'
    % (d_inj_g.mean(), spear(d_inj_g, d_null)))

open(OUT, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
print('OK preflight written')
