# -*- coding: utf-8 -*-
"""Phase 2951 preflight: reachability check (discipline 10).

Question: is the 2950 rebalance profile Delta_h = sc_I1[h] - sc_I0[h]
sorted by the linear W_ov head gain g_h (2948), and is the residual
rebalance (after removing the direct linear term) gain-decoupled?
All inputs already displayed in 2948/2950 results -> final verdicts
will be labeled quasi-post-hoc (discipline 9).
"""
import numpy as np
from scipy.stats import spearmanr

B = r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result\rdc_query_construction_20260913'
z48 = np.load(B + r'\phase2948\wov_head_gain\wov_head_gain.npz',
              allow_pickle=True)
z50 = np.load(B + r'\phase2950\rebalance_anatomy\rebalance_anatomy.npz',
              allow_pickle=True)

TOP5 = {'L17': [0, 7, 24, 22, 19], 'L16': [13, 16, 1, 17, 6]}
RNG = np.random.default_rng(2904)
N_PERM = 20000


def perm_p95(x, y, rng):
    """Null p95 of |spearman| under label permutation of y."""
    n = len(x)
    stats = np.empty(N_PERM)
    for i in range(N_PERM):
        yy = rng.permutation(y)
        stats[i] = abs(spearmanr(x, yy).statistic)
    return float(np.quantile(stats, 0.95)), stats


lines = []
for li in ('L17', 'L16'):
    sc0 = z50['sc_B1_%s' % li]   # ablation-only baseline (no injection)
    sc1 = z50['sc_I1_%s' % li]   # ablation + injection
    g = z48['g_%s' % li]
    abl = np.array(TOP5[li])
    keep = np.array([h for h in range(32) if h not in abl])
    # sanity: sc_I0 vs sc_I1 identical on keep heads (2950 npz semantics)
    d01 = float(np.abs(z50['sc_I1_%s' % li][keep]
                       - z50['sc_I0_%s' % li][keep]).max())
    delta = sc1[keep] - sc0[keep]
    gk = g[keep]

    rho_d, _ = spearmanr(gk, delta)
    p95_d, _ = perm_p95(gk, delta, RNG)

    # OLS removal of direct linear term
    beta = float(np.dot(gk, delta) / np.dot(gk, gk))
    resid = delta - beta * gk
    rho_r, _ = spearmanr(gk, resid)
    p95_r, _ = perm_p95(gk, resid, RNG)

    # movers: top5 |delta| heads -> g percentile among keep
    order = np.argsort(-np.abs(delta))
    movers = keep[order[:5]]
    g_ranks = spearmanr(gk, np.arange(len(gk))[np.argsort(np.argsort(gk))])
    # percentile of each mover's |g| among keep
    pct = [round(float((np.abs(gk) < abs(g[h])).mean()) * 100, 1)
           for h in movers]

    lines.append('== %s ==  (max|I1-I0| on keep = %.4f)' % (li, d01))
    lines.append('beta(direct)=%.4f  rho(g,delta)=%.4f  null_p95=%.4f'
                 % (beta, rho_d, p95_d))
    lines.append('rho(g,resid)=%.4f  null_p95=%.4f  |resid| med=%.3f'
                 % (rho_r, p95_r, float(np.median(np.abs(resid)))))
    lines.append('variance share of direct term: %.3f'
                 % float(np.dot(beta * gk, beta * gk) / np.dot(delta, delta)))
    lines.append('top5 movers %s  |g| percentiles %s'
                 % (movers.tolist(), pct))
    lines.append('delta top5 values %s'
                 % [round(float(delta[order[:5][i]]), 2) for i in range(5)])
    lines.append('')

txt = chr(10).join(lines)
open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp\phase2951_preflight_report.txt',
     'w', encoding='utf-8').write(txt + chr(10))
print('written')
