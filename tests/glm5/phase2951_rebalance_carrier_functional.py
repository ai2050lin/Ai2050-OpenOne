# -*- coding: utf-8 -*-
"""Phase 2951: rebalance-carrier W_ov functionality (zero forward).

Question: is the 2950 rebalance profile Delta_h = sc_I1[h] - sc_B1[h]
(injection-induced change of each non-ablated head's snapshot sep
contribution under the head-group-ablated condition) sorted by the
linear W_ov head gain g_h (2948)?  And is the residual after removing
the direct linear term gain-decoupled?

Semantics note (preflight correction, frozen here): 2950 npz stores
sc_I0 == sc_I1 on keep heads; the 2950 movers were computed against
the ablation-only baseline sc_B1.  Delta is therefore I1 - B1.

All input quantities (g from 2948, sc arrays from 2950) were already
displayed side-by-side in prior results => all verdicts are labeled
quasi-post-hoc mechanism integration (discipline 9); they do not
override 2949/2950 verdicts.
"""
import json
import os
import time

import numpy as np
from scipy.stats import spearmanr

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2951', 'rebalance_carrier_functional')
SCRIPT_DIR = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5'
              r'\phase2951_rebalance_carrier_functional.py')
SRC_2948_NPZ = os.path.join(BASE, 'phase2948', 'wov_head_gain',
                            'wov_head_gain.npz')
SRC_2948_RES = os.path.join(BASE, 'phase2948', 'wov_head_gain',
                            'result.json')
SRC_2950_NPZ = os.path.join(BASE, 'phase2950', 'rebalance_anatomy',
                            'rebalance_anatomy.npz')
SRC_2950_RES = os.path.join(BASE, 'phase2950', 'rebalance_anatomy',
                            'result.json')
SRC_2939_NPZ = os.path.join(BASE, 'phase2939', 'rotation_target',
                            'rotation_target.npz')

TOP5 = {'L17': [0, 7, 24, 22, 19], 'L16': [13, 16, 1, 17, 6]}
S_INJ = {'L17': 1.0, 'L16': 2.0}
N_PERM = 20000
SEED = 2904

PREREG = {
    'phase': 2951,
    'title': 'rebalance-carrier W_ov functionality',
    'created': None,           # filled at freeze time
    'zero_forward': True,
    'data_sources': {
        'g_vectors': 'phase2948/wov_head_gain npz (g_L17, g_L16)',
        'sc_arrays': 'phase2950/rebalance_anatomy npz '
                     '(sc_I1_*, sc_B1_*)',
        'chain_check': 'phase2939 rotation_target npz Vt8',
    },
    'delta_definition': 'Delta_h = sc_I1[h] - sc_B1[h] over keep '
                        'heads (non-ablated); 2950 movers were '
                        'I1-B1 based (npz semantics verified in '
                        'preflight: max|I1-I0| on keep = 0)',
    'main_tests': {
        'T1 gain-sorted delta': 'spearman(g, Delta) >= null p95 '
                                '(|rho| permutation null, N=20000, '
                                'seed 2904) for BOTH L17 and L16',
        'T2 residual decoupled': 'after OLS removal of direct term '
                                 '(R = Delta - beta*g): '
                                 '|spearman(g, R)| < null p95 for '
                                 'BOTH L17 and L16',
    },
    'verdict': 'anchor fail => anchor_fail_all_void; '
               'T1 pass AND T2 pass => '
               'rebalance_carriers_gain_functional; '
               'T1 pass only => rebalance_partially_gain_sorted; '
               'else => rebalance_not_gain_sorted',
    'quasi_post_hoc': 'discipline 9: g (2948) and Delta components '
                      '(2950) were displayed in prior results; '
                      'verdicts are mechanism integration only',
    'correction_note': 'run1 anchor fixes (pre-observation of any '
                       'main statistic): (1) a2 threshold 1e-6 per '
                       'institutionalized cross-phase bf16 noise '
                       'level 2.17e-08 (2940 convention); dirs_word '
                       '2948-vs-2950 observed 2.17e-08, same level '
                       'as a1 dirs rebuild anchor; (2) a5 self-check '
                       'redesigned: spearmanr(xs, xs)=1 and '
                       'spearmanr(xs, -xs)=-1 (value reversal, not '
                       'position reversal - probe confirmed scipy '
                       '1.18 semantics correct via manual rankdata)',
    'preflight_semantics_note': 'preflight round-1 used Delta = I1 '
                       '- I0 and found all-zero (npz semantics: keep '
                       'heads identical); corrected to I1 - B1 '
                       'before any formal observation of the new '
                       'statistic; run2 a7 fix: raw-vs-raw p95 '
                       'comparison (run1 compared unrounded p95 '
                       'against round(.,4) - pure rounding diff '
                       '4.19e-05)',
}
os.makedirs(OUT, exist_ok=True)
stamp = time.strftime('%Y-%m-%dT%H:%M:%S')
PREREG['created'] = stamp
with open(os.path.join(OUT, 'execution.json'), 'w',
          encoding='utf-8') as f:
    json.dump(PREREG, f, ensure_ascii=False, indent=2)

lines = []


def log(msg, buf):
    buf.append(msg)


def main():
    t0 = time.time()
    z48 = np.load(SRC_2948_NPZ, allow_pickle=True)
    z50 = np.load(SRC_2950_NPZ, allow_pickle=True)
    z39 = np.load(SRC_2939_NPZ, allow_pickle=True)
    r50 = json.load(open(SRC_2950_RES, encoding='utf-8'))

    # ---------------- anchors ----------------
    a1 = float(np.abs(z48['Vt8'] - z39['Vt8']).max())
    a1_ok = bool(a1 == 0.0)
    log('a1 Vt8 vs 2939 bit-diff %.2e ok=%s' % (a1, a1_ok), lines)

    a2 = float(np.abs(z48['dirs_word'] - z50['dirs_word']).max())
    a2_ok = bool(a2 < 1e-6)   # cross-phase bf16 noise convention
    log('a2 dirs_word 2948 vs 2950 max diff %.2e ok=%s'
        % (a2, a2_ok), lines)

    a3_diff = 0.0
    for li in ('L17', 'L16'):
        r_d1 = r50['D1_head_sep_c'][li]
        s0j = np.array(r_d1['sc_I1'])
        s1j = np.array(r_d1['sc_B1'])
        a3_diff = max(a3_diff,
                      float(np.abs(z50['sc_I1_%s' % li] - s0j).max()),
                      float(np.abs(z50['sc_B1_%s' % li] - s1j).max()))
    a3_ok = bool(a3_diff <= 5.01e-3)   # json rounded to 2 decimals
    log('a3 sc arrays vs 2950 result.json max diff %.2e ok=%s'
        % (a3_diff, a3_ok), lines)

    a4_diff = 0.0
    for li in ('L17', 'L16'):
        abl = TOP5[li]
        a4_diff = max(a4_diff, float(
            np.abs(z50['sc_I1_%s' % li][abl]).max()))
    a4_ok = bool(a4_diff == 0.0)
    log('a4 ablated heads sc_I1 bit-zero %.2e ok=%s'
        % (a4_diff, a4_ok), lines)

    rng_a = np.random.default_rng(11)
    xs = rng_a.standard_normal(32)
    a5_diff = max(abs(spearmanr(xs, xs.copy()).statistic - 1.0),
                  abs(spearmanr(xs, -xs.copy()).statistic + 1.0))
    a5_ok = bool(a5_diff < 1e-12)
    log('a5 spearman self-check (identity + value-reversal) '
        '%.2e ok=%s' % (a5_diff, a5_ok), lines)

    # ---------------- main compute ----------------
    anchors_ok = all([a1_ok, a2_ok, a3_ok, a4_ok, a5_ok])
    verdict = None
    t1 = t2 = d1 = None
    save = {}

    if anchors_ok:
        t1 = {}
        t2 = {}
        d1 = {}
        p95_d_raw = {}
        for li in ('L17', 'L16'):
            g = z48['g_%s' % li]
            abl = np.array(TOP5[li])
            keep = np.array([h for h in range(32) if h not in abl])
            delta = (z50['sc_I1_%s' % li][keep]
                     - z50['sc_B1_%s' % li][keep])
            gk = g[keep]

            rho_d = float(spearmanr(gk, delta).statistic)
            rng = np.random.default_rng(SEED)
            null_d = np.array([
                abs(spearmanr(
                    gk, rng.permutation(delta)).statistic)
                for _ in range(N_PERM)])
            p95_d = float(np.quantile(null_d, 0.95))
            p95_d_raw[li] = p95_d
            t1_pass_li = bool(rho_d >= p95_d)
            t1[li] = {'rho': round(rho_d, 4),
                      'null_p95': round(p95_d, 4),
                      'pass': t1_pass_li}
            save['delta_%s' % li] = delta
            save['g_%s' % li] = gk
            save['keep_%s' % li] = keep
            save['null_p95_delta_%s' % li] = np.array([p95_d])

            beta = float(np.dot(gk, delta) / np.dot(gk, gk))
            resid = delta - beta * gk
            a6 = float(np.abs(delta - (beta * gk + resid)).max())
            if a6 > 1e-12:
                log('a6 OLS reconstruction diff %.2e FAIL' % a6,
                    lines)
            rho_r = float(spearmanr(gk, resid).statistic)
            rng2 = np.random.default_rng(SEED)
            null_r = np.array([
                abs(spearmanr(
                    gk, rng2.permutation(resid)).statistic)
                for _ in range(N_PERM)])
            p95_r = float(np.quantile(null_r, 0.95))
            t2_pass_li = bool(abs(rho_r) < p95_r)
            t2[li] = {'rho_resid': round(rho_r, 4),
                      'null_p95': round(p95_r, 4),
                      'pass': t2_pass_li}
            save['resid_%s' % li] = resid
            save['null_p95_resid_%s' % li] = np.array([p95_r])

            share = float(np.dot(beta * gk, beta * gk)
                          / np.dot(delta, delta))
            amp = beta / S_INJ[li]
            order = np.argsort(-np.abs(delta))
            movers = keep[order[:5]]
            pct = [round(float((np.abs(gk) < abs(g[h])).mean())
                         * 100, 1) for h in movers]
            d1[li] = {
                'beta': round(beta, 4),
                'amp_factor': round(amp, 3),
                'direct_var_share': round(share, 3),
                'resid_med_abs': round(
                    float(np.median(np.abs(resid))), 3),
                'top5_movers': movers.tolist(),
                'mover_|g|_pct': pct,
                'delta_top5': [round(float(delta[i]), 2)
                               for i in order[:5]],
            }
            log('%s: rho(g,delta)=%.4f (p95 %.4f, pass=%s) | '
                'beta=%.4f amp=%.2fx share=%.3f | '
                'rho(g,resid)=%.4f (p95 %.4f, pass=%s)'
                % (li, rho_d, p95_d, t1_pass_li, beta, amp,
                   share, rho_r, p95_r, t2_pass_li), lines)

        # a7: perm null determinism (same seed -> same p95)
        g = z48['g_L17']
        keep = save['keep_L17']
        delta = save['delta_L17']
        gk = g[keep]
        rng3 = np.random.default_rng(SEED)
        null_r3 = np.array([
            abs(spearmanr(gk, rng3.permutation(delta)).statistic)
            for _ in range(N_PERM)])
        a7 = abs(float(np.quantile(null_r3, 0.95))
                 - p95_d_raw['L17'])
        a7_ok = bool(a7 == 0.0)
        log('a7 perm-null determinism %.2e ok=%s'
            % (a7, a7_ok), lines)

        t1_all = all(v['pass'] for v in t1.values())
        t2_all = all(v['pass'] for v in t2.values())
        if t1_all and t2_all:
            verdict = 'rebalance_carriers_gain_functional'
        elif t1_all:
            verdict = 'rebalance_partially_gain_sorted'
        else:
            verdict = 'rebalance_not_gain_sorted'
    else:
        a7 = float('nan')
        a7_ok = False
        verdict = 'anchor_fail_all_void'

    log('==== VERDICT: %s ====' % verdict, lines)
    runtime = round(time.time() - t0, 1)

    result = {
        'phase': 2951,
        'verdict': verdict,
        'quasi_post_hoc': True,
        'prereg': PREREG,
        'anchors': {
            'a1_vt8_bit': a1, 'a2_dirs_bit': a2,
            'a3_sc_vs_2950': round(a3_diff, 6),
            'a4_abl_zero': a4_diff,
            'a5_spearman_self': float('%.3e' % a5_diff),
            'a7_perm_determinism': a7,
            'all_ok': bool(anchors_ok and a7_ok),
        },
        'T1_gain_sorted': t1,
        'T2_residual_decoupled': t2,
        'D1_anatomy': d1,
        'runtime_s': runtime,
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    np.savez(os.path.join(OUT, 'rebalance_carrier_functional.npz'),
             **save)
    with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
              r'\phase2951_run_report.txt', 'w',
              encoding='utf-8') as f:
        f.write(chr(10).join(lines) + chr(10))
    print('OK phase2951 verdict=%s runtime=%.1fs'
          % (verdict, runtime), flush=True)


if __name__ == '__main__':
    main()
