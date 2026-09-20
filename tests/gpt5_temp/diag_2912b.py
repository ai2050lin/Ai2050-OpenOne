# -*- coding: utf-8 -*-
"""diag_2912b.py -- pin down the calibration p_median mechanism.

Uses the EXACT phase2912 functions (sign_mat, gap_seq, zigzag_k,
rho1, extended perm loop).  For many iid calibration-style
matrices, record k_obs, the full null k distribution, the tie rate
tau = P(k_perm == k_obs), and the one-sided p = P(perm >= obs).

Theory: for iid zigzag counts on 10 points, the inner-point
reversal probability is 2/3 (classic up-down run theory: the middle
of three iid values is a local extremum with prob 1/3+1/3), so
E[k] ~ 8*2/3 = 5.33 (less after tie loss).  For a one-sided
p = P(K' >= K) with K, K' iid DISCRETE, E[p] = 1/2 + tau/2 where
tau = sum p_k^2.  The phase2912 calibration band [0.40, 0.60] on
the p median ignored this discreteness correction.

Outputs diag_2912b.txt.
"""
import numpy as np

N_MAT = 1000
N_PERM = 500
n, w, n0 = 57, 10, 22
rng = np.random.default_rng([2912, 0])
lab = np.array([0] * n0 + [1] * (n - n0))


def zigzag_k(g):
    d = np.diff(g)
    return int(np.sum(d[:-1] * d[1:] < 0))


ps, kobs, ties, kperm_all = [], [], [], []
null_hist = np.zeros(9, dtype=int)
for _ in range(N_MAT):
    B = rng.normal(size=(n, w))
    Sm = np.sign(B)
    Sm[Sm == 0] = 1.0
    m0 = lab == 0
    m1 = lab == 1
    pf0 = (Sm[m0] > 0).mean(axis=0)
    pf1 = (Sm[m1] > 0).mean(axis=0)
    g0 = np.abs(pf0 - pf1)
    ko = zigzag_k(g0)
    kobs.append(ko)
    ck = 0
    tie = 0
    for _ in range(N_PERM):
        idx = np.argsort(rng.random((n, w)), axis=0)
        Sp = np.take_along_axis(Sm, idx, axis=0)
        q0 = (Sp[:n0] > 0).mean(axis=0)
        q1 = (Sp[n0:] > 0).mean(axis=0)
        gp = np.abs(q0 - q1)
        kp = zigzag_k(gp)
        null_hist[kp] += 1
        kperm_all.append(kp)
        if kp >= ko:
            ck += 1
        if kp == ko:
            tie += 1
    ps.append((ck + 1) / float(N_PERM + 1))
    ties.append(tie / float(N_PERM))

ps = np.asarray(ps)
kobs = np.asarray(kobs)
ties = np.asarray(ties)
kperm_all = np.asarray(kperm_all)
pk = null_hist / float(null_hist.sum())
tau_pool = float((pk ** 2).sum())
tau_emp = float(ties.mean())
mu_p_theory = 0.5 + 0.5 * tau_pool

lines = [
    'diag_2912b: exact phase2912 construction, %d iid matrices '
    'x %d perms' % (N_MAT, N_PERM),
    'k_obs  mean %.3f  median %.1f  sd %.3f'
    % (kobs.mean(), np.median(kobs), kobs.std()),
    'k_perm mean %.3f  sd %.3f'
    % (kperm_all.mean(), kperm_all.std()),
    'k_perm hist %s' % null_hist.tolist(),
    'binomial(8, 2/3) mean 5.333  (2/3 = up-down reversal law)',
    'tie rate tau_emp  %.4f' % tau_emp,
    'tie rate tau_pool(sum pk^2) %.4f' % tau_pool,
    'E[p] theory = 0.5 + tau_pool/2 = %.4f' % mu_p_theory,
    'p      mean %.4f  median %.4f'
    % (ps.mean(), float(np.median(ps))),
    'p quantiles 10/25/50/75/90: %s'
    % np.round(np.percentile(ps, [10, 25, 50, 75, 90]), 4).tolist(),
    'frac p in [0.05,0.95]: %.4f'
    % float(np.mean((ps >= 0.05) & (ps <= 0.95))),
    'median SE (N_MAT=%d, ~uniform) %.4f'
    % (N_MAT, 1.0 / (2.0 * np.sqrt(N_MAT))),
]
# binomial(8, 2/3) reference probs
from math import comb
bp = [comb(8, k) * (2.0 / 3) ** k * (1.0 / 3) ** (8 - k)
      for k in range(9)]
lines.append('binom(8,2/3) probs %s'
             % np.round(bp, 4).tolist())
lines.append('binom tau = sum pk^2 = %.4f  -> E[p] %.4f'
             % (sum(p * p for p in bp), 0.5 + 0.5 * sum(p * p for p in bp)))

out = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\diag_2912b.txt')
with open(out, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK diag_2912b')
