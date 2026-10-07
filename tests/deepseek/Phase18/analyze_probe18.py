# -*- coding: utf-8 -*-
"""
Phase 18 探针离线派生：从 _probe_feasibility_A0.json 的 rows 计算 seal 所需的全部分量。

只做只读后处理（无 GPU）。输出 _probe_analysis_A0.txt / .json。
口径与主脚本 n2h1a11_behavioral_component_budget.py 逐字一致（区间求和质心、同域比）。
"""
import os
import io
import json
import numpy as np

ROOT = r'D:\AI2050\Ai2050-OpenOne'
P18T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase18')
P17T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase17')
P16T = os.path.join(ROOT, 'tests', 'deepseek_temp', 'Phase16')

P = json.load(io.open(os.path.join(P18T, '_probe_feasibility_A0.json'), encoding='utf-8'))
R17 = json.load(io.open(os.path.join(P17T, 'result_phase17.json'), encoding='utf-8'))
R16 = json.load(io.open(os.path.join(P16T, 'result_phase16.json'), encoding='utf-8'))
ARM = P['arm']
REACH = [int(x) for x in R16['E7_reach'][ARM]['reach']]
NB = [int(x) for x in R17['arms'][ARM]['E5_com_V']['neighbourhood']]
W = {k: {int(l): float(v) for l, v in enumerate(R17['arms'][ARM]['E5_com_V'][k])}
     for k in ('w_all', 'w_mlp', 'w_attn')}
J16 = {int(s): float(v) for s, v in zip(R16['E4_summary'][ARM]['sites'], R16['E4_summary'][ARM]['J'])}
FULL_SWAP = float(R16['arms'][ARM]['E2_full_swap']['FULL_SWAP'])

LOG = []


def w(s=''):
    LOG.append(str(s))
    print(s)


def com_of_mass(mass_by_site, sites):
    s = np.asarray(sites, float)
    vals = np.asarray([float(sum(mass_by_site.get(int(l), 0.0)
                                for l in range(int(sites[j]), int(sites[j + 1]))))
                       for j in range(len(sites) - 1)], float)
    mid = (s[:-1] + s[1:]) / 2.0
    den = float(vals.sum())
    if den <= 1e-12:
        return None
    return float((vals * mid).sum() / den)


def stat_com_layer(jumps, sites):
    j = np.asarray(jumps, float)
    if len(j) == 0 or len(sites) != len(j) + 1:
        return None
    mid = (np.asarray(sites, float)[:-1] + np.asarray(sites, float)[1:]) / 2.0
    a = np.abs(j)
    den = float(a.sum())
    if den <= 1e-12:
        return None
    return float((a * mid).sum() / den)


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    if len(a) < 3 or float(np.std(a)) <= 1e-9 or float(np.std(b)) <= 1e-9:
        return None
    ra = np.argsort(np.argsort(a)).astype(float); rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra * rb).sum() / (np.linalg.norm(ra) * np.linalg.norm(rb)))


def perm_null_com(mass_by_site, sites, seed, n_bp=2000):
    vals = np.asarray([float(sum(mass_by_site.get(int(l), 0.0)
                                 for l in range(int(sites[j]), int(sites[j + 1]))))
                       for j in range(len(sites) - 1)], float)
    s = np.asarray(sites, float); mid = (s[:-1] + s[1:]) / 2.0
    rng = np.random.default_rng(seed)
    out = np.array([float((vals[rng.permutation(len(vals))] * mid).sum() / vals.sum())
                    for _ in range(n_bp)])
    obs = float((vals * mid).sum() / vals.sum())
    return obs, float(np.percentile(out, 5)), float(np.percentile(out, 95))


def perm_null_share(bm, ba, sites, nb, seed, n_bp=2000):
    mi = {int(s): i for i, s in enumerate(sites)}
    idx = [mi[l] for l in nb if l in mi]
    m = np.array([abs(bm.get(int(l), 0.0)) for l in sites], float)
    a = np.array([abs(ba.get(int(l), 0.0)) for l in sites], float)
    obs = float(m[idx].sum() / (m[idx].sum() + a[idx].sum()))
    rng = np.random.default_rng(seed)
    out = np.array([float(m[rng.permutation(len(m))][idx].sum() /
                          (m[rng.permutation(len(m))][idx].sum() + a[idx].sum()))
                    for _ in range(n_bp)])
    return obs, float(np.percentile(out, 5)), float(np.percentile(out, 95))


SITES = sorted(int(s) for s in P['rows'].keys())
KEY = {'INC_ALL': 'INC_ALL@a1.00', 'INC_MLP': 'INC_MLP@a1.00', 'INC_ATTN': 'INC_ATTN@a1.00',
       'INC_TOP1': 'INC_TOP1@a1.00', 'CUM_ALL': 'CUM_ALL@a1.00', 'INC_ALL5': 'INC_ALL@a0.50'}
B = {}
for c, k in KEY.items():
    B[c] = {s: float(P['rows'][str(s)][k]['dDonor']) for s in SITES if k in P['rows'][str(s)]}

w('=== Phase18 探针离线派生 (arm=%s) ===' % ARM)
w('probe sites n=%d  (%s .. %s)' % (len(SITES), SITES[0], SITES[-1]))
w('REACH n=%d ; nb(P17)=%s ; median(REACH)=%.1f' % (len(REACH), NB, float(np.median(REACH))))
w('')
w('%-10s %8s %8s %8s %8s %8s' % ('site', 'ALL', 'MLP', 'ATTN', 'TOP1', 'CUM'))
for s in SITES:
    tg = '*' if s in REACH else ' '
    w('  %sL%-3d %8.3f %8.3f %8.3f %8.3f %8.3f'
      % (tg, s, B['INC_ALL'][s], B['INC_MLP'][s], B['INC_ATTN'][s], B['INC_TOP1'][s], B['CUM_ALL'][s]))
w('')

# --- 质心
com_B = {c: com_of_mass({s: abs(v) for s, v in B[c].items()}, REACH) for c in B}
com_V_recomputed = com_of_mass(W['w_all'], REACH)
com_V_p17 = float(R17['arms'][ARM]['E5_com_V']['com_V'])
w('com_B(all)  = %.4f  [REACH 域, 全域质量支撑]' % com_B['INC_ALL'])
for c in ('INC_MLP', 'INC_ATTN', 'INC_TOP1', 'CUM_ALL'):
    w('com_B(%-7s) = %.4f' % (c, com_B[c]))
w('com_V (P17 记录)      = %.4f' % com_V_p17)
w('com_V (同域重算 w_all) = %.4f  (drift %.3e)' % (com_V_recomputed, abs(com_V_recomputed - com_V_p17)))
w('gap = com_V - com_B(all) = %.4f' % (com_V_p17 - com_B['INC_ALL']))
w('')

# --- 邻域份额
def sh(cn, cd, sites):
    num = sum(abs(B[cn][s]) for s in sites); den = sum(abs(B[cd][s]) for s in sites)
    return num / den if den > 1e-12 else None


sm_beh_nb = sh('INC_MLP', 'INC_ALL', NB)
sa_beh_nb = sh('INC_ATTN', 'INC_ALL', NB)
st_beh_nb = sh('INC_TOP1', 'INC_ALL', NB)
sm_beh_re = sh('INC_MLP', 'INC_ALL', REACH)
sm_vec_nb = (sum(W['w_mlp'][s] for s in NB) / sum(W['w_all'][s] for s in NB))
sa_vec_nb = (sum(W['w_attn'][s] for s in NB) / sum(W['w_all'][s] for s in NB))
w('--- 邻域 nb=%s ---' % NB)
w('  share_mlp_beh(nb) = %.4f   | share_mlp_vec(nb) = %.4f   (P17 记录 %.4f)'
  % (sm_beh_nb, sm_vec_nb, float(R17['arms'][ARM]['E5_com_V']['share_mlp_nb'])))
w('  share_attn_beh(nb)= %.4f   | share_attn_vec(nb)= %.4f' % (sa_beh_nb, sa_vec_nb))
w('  share_top1_beh(nb)= %.4f' % st_beh_nb)
w('  share_mlp_beh(REACH全) = %.4f' % sm_beh_re)
w('')

# --- 行为剖面的 com_layer
cl_all = stat_com_layer(np.diff(np.array([B['INC_ALL'][s] for s in REACH])), REACH)
cl_mlp = stat_com_layer(np.diff(np.array([B['INC_MLP'][s] for s in REACH])), REACH)
cl_attn = stat_com_layer(np.diff(np.array([B['INC_ATTN'][s] for s in REACH])), REACH)
cl_absall = stat_com_layer(np.diff(np.array([abs(B['INC_ALL'][s]) for s in REACH])), REACH)
w('com_layer(b_all)  = %.4f' % cl_all)
w('com_layer(|b_all|)= %.4f' % cl_absall)
w('com_layer(b_mlp)  = %.4f ; com_layer(b_attn) = %.4f' % (cl_mlp, cl_attn))
w('  (P16 锚: com_layer(x)=%.4f com_layer(J)=%.4f)'
  % (R16['E5_concentration'][ARM]['new_stat']['x']['obs_com'],
     R16['E5_concentration'][ARM]['new_stat']['j']['obs_com']))
w('')

# --- 同对象耦合
sp_wall_ball = spearman([W['w_all'][s] for s in REACH], [abs(B['INC_ALL'][s]) for s in REACH])
sp_wmlp_bmlp = spearman([W['w_mlp'][s] for s in REACH], [abs(B['INC_MLP'][s]) for s in REACH])
sp_wattn_battn = spearman([W['w_attn'][s] for s in REACH], [abs(B['INC_ATTN'][s]) for s in REACH])
xl = [s for s in REACH if s in J16]
sp_wall_J = spearman([W['w_all'][s] for s in xl], [J16[s] for s in xl])
sp_ball_J = spearman([abs(B['INC_ALL'][s]) for s in xl], [J16[s] for s in xl])
w('--- 耦合 ---')
w('  spearman(w_all, |b_all|) = %+.4f   <== 同对象（本 Phase 口）' % sp_wall_ball)
w('  spearman(w_mlp, |b_mlp|) = %+.4f' % sp_wmlp_bmlp)
w('  spearman(w_attn,|b_attn|) = %+.4f' % sp_wattn_battn)
w('  [对照] spearman(w_all, J_P16) = %+.4f   (P17 P6 口径)' % sp_wall_J)
w('  [对照] spearman(|b_all|, J_P16) = %+.4f' % sp_ball_J)
w('')

# --- 线性残差
rlin = {s: abs(B['INC_ALL'][s] - (B['INC_MLP'][s] + B['INC_ATTN'][s])) / max(abs(B['INC_ALL'][s]), 1e-12)
        for s in SITES}
w('--- 线性残差 r_lin ---')
for s in SITES:
    if s in (6, 7, 8, 9, 10, 12, 14, 18, 22, 24, 26, 28, 30, 32, 34):
        tg = '*' if s in REACH else ' '
        w('  %sL%-3d r_lin=%.4f  (b_all=%+.3f b_mlp+b_attn=%+.3f)'
          % (tg, s, rlin[s], B['INC_ALL'][s], B['INC_MLP'][s] + B['INC_ATTN'][s]))
w('  r_lin@L6=%.4f ; mean(nb)=%.4f ; mean(REACH)=%.4f ; mean(全域)=%.4f'
  % (rlin[6], float(np.mean([rlin[s] for s in NB])), float(np.mean([rlin[s] for s in REACH])),
     float(np.mean([rlin[s] for s in SITES]))))
w('')

# --- 桥接
cum6 = B['CUM_ALL'][6]
w('--- 桥接 ---')
w('  CUM_ALL@L6 = %+.4f ; P16 FULL_SWAP = %+.4f ; rel = %.4f'
  % (cum6, FULL_SWAP, abs(cum6 - FULL_SWAP) / abs(FULL_SWAP)))
w('')

# --- 零假设
o1, p5_1, p95_1 = perm_null_com({s: abs(v) for s, v in B['INC_ALL'].items()}, REACH, 20261174)
o2, p5_2, p95_2 = perm_null_com({s: abs(v) for s, v in B['INC_MLP'].items()}, REACH, 20261186)
o3, p5_3, p95_3 = perm_null_share(B['INC_MLP'], B['INC_ATTN'], REACH, NB, 20261198)
w('--- 置换零假设 (BP=2000) ---')
w('  com_B(all): obs=%.3f  p5=%.3f p95=%.3f -> %s'
  % (o1, p5_1, p95_1, 'low' if o1 <= p5_1 else 'high' if o1 >= p95_1 else 'none'))
w('  com_B(mlp): obs=%.3f  p5=%.3f p95=%.3f -> %s'
  % (o2, p5_2, p95_2, 'low' if o2 <= p5_2 else 'high' if o2 >= p95_2 else 'none'))
w('  share_mlp_beh(nb): obs=%.4f  p5=%.4f p95=%.4f -> %s'
  % (o3, p5_3, p95_3, 'low' if o3 <= p5_3 else 'high' if o3 >= p95_3 else 'none'))
w('')

OUT = dict(arm=ARM, sites=SITES, reach=REACH, nb=NB,
           b={c: {str(s): v for s, v in B[c].items()} for c in B},
           com_B=com_B, com_V_p17=com_V_p17, com_V_recomputed=com_V_recomputed,
           gap=com_V_p17 - com_B['INC_ALL'],
           share_mlp_beh_nb=sm_beh_nb, share_attn_beh_nb=sa_beh_nb, share_top1_beh_nb=st_beh_nb,
           share_mlp_beh_reach=sm_beh_re, share_mlp_vec_nb=sm_vec_nb, share_attn_vec_nb=sa_vec_nb,
           com_layer_b_all=cl_all, com_layer_absball=cl_absall, com_layer_b_mlp=cl_mlp,
           com_layer_b_attn=cl_attn,
           spearman=dict(wall_ball=sp_wall_ball, wmlp_bmlp=sp_wmlp_bmlp, wattn_battn=sp_wattn_battn,
                         wall_J=sp_wall_J, ball_J=sp_ball_J),
           rlin={str(s): rlin[s] for s in SITES},
           rlin_L6=rlin[6], rlin_nb_mean=float(np.mean([rlin[s] for s in NB])),
           rlin_reach_mean=float(np.mean([rlin[s] for s in REACH])),
           cum_bridge=cum6, full_swap=FULL_SWAP,
           bridge_rel=abs(cum6 - FULL_SWAP) / abs(FULL_SWAP),
           null=dict(comB_all=[o1, p5_1, p95_1], comB_mlp=[o2, p5_2, p95_2],
                     share=[o3, p5_3, p95_3]),
           pert_rel_inc={s: P['rows'][str(s)]['INC_ALL@a1.00']['pert_rel_mean'] for s in SITES},
           pert_rel_cum={s: P['rows'][str(s)]['CUM_ALL@a1.00']['pert_rel_mean'] for s in SITES})
io.open(os.path.join(P18T, '_probe_analysis_A0.json'), 'w', encoding='utf-8').write(
    json.dumps(OUT, ensure_ascii=False, indent=1))
io.open(os.path.join(P18T, '_probe_analysis_A0.txt'), 'w', encoding='utf-8').write(
    '\r\n'.join(LOG) + '\r\n')
print('WROTE _probe_analysis_A0.{json,txt}')
