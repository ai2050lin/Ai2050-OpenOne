# -*- coding: utf-8 -*-
"""ledger_2912_update.py -- register M2912 + refine L14 + add N13."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    L = json.load(f)

# --- guard: idempotency ---
m_ids = [m.get('meas_id') for m in L['measurements']]
n_ids = [n.get('neg_id') for n in L['negatives']]
assert 'M2912_sign_balance_zigzag' not in m_ids, 'M2912 already in'
assert 'N13_gap_zigzag_absent_2_3_reversal_law' not in n_ids
assert m_ids[-1] == 'M2911_alternation_structure', m_ids[-1]

# --- M2912 measurement ---
m2912 = {
    "meas_id": "M2912_sign_balance_zigzag",
    "type": "gap_zigzag_formal_test",
    "verdict": (
        "gap_zigzag_absent - anchors 4/4 (per-layer d=1 scores vs "
        "2910 score_true maxabs 4.5-5.0e-7; Delta_B vs 2905 maxabs "
        "4.4-4.9e-5; full margin/acc vs stored within 2e-5) and "
        "calibration v2 pass (frac in band 0.8950; p_mean 0.5994 "
        "vs analytic tie-corrected expectation 0.6100, band "
        "[0.5039,0.7160]). HISTORY: run1 was judged "
        "audit_calib_fail_all_void under the v1 band (p_median "
        "0.687 > 0.60); diag_2912a/b then proved the null "
        "construction unbiased (iid obs vs null zigzag mean 5.18 "
        "vs 5.21; null k matches binomial(8,2/3), E[k]=5.33 by "
        "the 2/3 up-down reversal law - NOT binomial(8,0.5)) and "
        "the v1 failure to be a band-caliber defect: for a "
        "one-sided discrete p = P(perm >= obs) with ties included, "
        "E[p] = 0.5 + tau/2 with tau = sum pk^2 ~ 0.22, so E[p] ~ "
        "0.61 and the p median centers near 0.70 (diag_2912b, "
        "1000 iid matrices: p_mean 0.618 vs theory 0.610). v2 "
        "replaces the median band by the analytic p_mean band; p "
        "definition and all other frozen elements unchanged; run1 "
        "products deleted per rerun discipline. MAIN TEST (20000 "
        "perms/group, per-layer independent class-label "
        "permutation, sizes fixed 22/35): S = groups significant "
        "on both p_zig<=0.05 and p_rho<=0.05 is EMPTY => "
        "gap_zigzag_absent. Per group: qwen_attn zigzag k=7/8 "
        "p=0.172, rho1=-0.694 p=0.177; glm4_mlp k=7 p=0.539, "
        "rho1=-0.449 p=0.567; glm4_attn k=6 p=0.794, rho1=-0.108 "
        "p=0.928; qwen_mlp k=5 p=0.708, rho1=-0.528 p=0.440. KEY "
        "READING: the descriptive zigzag 7/8 (diag_2911c) is NOT "
        "rare under the correct null - iid sequences carry "
        "E[k]=2/3*8=5.33 reversals (middle-of-three is a local "
        "extremum with prob 2/3), so k=7/8 has p~0.17; all four "
        "rho1 are negative (-0.11..-0.69), qwen_attn most "
        "negative, none significant. The 2911 alternation "
        "evidence therefore lives in the margin SIGN sequence "
        "(2911 P1: 8/9 flips vs null flip rate 0.5, p=0.0195), "
        "not in the gap AMPLITUDE zigzag (null reversal rate 2/3 "
        "makes k=7/8 unremarkable). P3 descriptive: top-5 "
        "flip-word share 0.25-0.41; fair-coin tail of the "
        "max-flip word significant only for qwen_mlp (8 flips, "
        "p=0.0195); qwen_attn max 7 flips p=0.090"),
    "source": {
        "path": "phase2912/sign_balance_zigzag/result.json",
        "sha256_8": "4611db7f",
        "phase": 2912
    }
}
L['measurements'].append(m2912)

# --- L14 refinement ---
l14 = None
for c in L['linkage']:
    if c.get('link_id') == 'L14_readout_spectrum_cross_model':
        l14 = c
        break
assert l14 is not None
assert 'M2912_sign_balance_zigzag' not in l14['connects']
l14['connects'].append('M2912_sign_balance_zigzag')
l14['notes'] += (
    " | 2912 gap-zigzag formalization: the alternation is NOT "
    "carried by gap amplitude - per-layer sign-balance gap zigzag "
    "absent under the per-layer permutation null (qwen_attn k=7/8 "
    "p=0.172, rho1 -0.694 p=0.177; S empty); the correct null for "
    "zigzag counts is the 2/3 up-down reversal law (E[k]=5.33/8), "
    "under which the descriptive 7/8 (diag_2911c) is "
    "unremarkable; alternation evidence remains "
    "margin-sign-sequence-only (2911 P1); calibration-protocol "
    "lesson: a one-sided discrete permutation p with ties "
    "included has E[p]=0.5+tau/2~0.61 (tau=sum pk^2~0.22), the v1 "
    "median band spuriously failed (run1 all_void), the v2 "
    "analytic p_mean band replaces it")
l14['phase_updated'] = 2912

# --- N13 negative ---
n13 = {
    "neg_id": "N13_gap_zigzag_absent_2_3_reversal_law",
    "kind": "descriptive_pattern_null_under_formal_test",
    "claim": (
        "the class-wise sign-balance gap sequence "
        "|pos_frac0_j - pos_frac1_j| carries a real per-layer "
        "zigzag (alternation) beyond chance for qwen_attn, "
        "generalizing the 2911 margin-sign alternation to the "
        "amplitude domain"),
    "evidence": (
        "2912 preregistered per-layer independent permutation "
        "test, 20000 perms: qwen_attn zigzag k=7/8 p=0.172, "
        "lag-1 rho1=-0.694 p=0.177 (all four groups p>=0.17; S "
        "empty => gap_zigzag_absent); the discrepancy vs the "
        "descriptive 7/8 impression is the null law: for iid "
        "sequences E[k] = 2/3*(n-2) = 5.33 of 8 (up-down "
        "reversal law; middle-of-three is a local extremum with "
        "prob 1/3+1/3), NOT n/2 - diag_2912b verifies the null k "
        "distribution matches binomial(8,2/3) (mean 5.21 vs "
        "5.33 after tie loss); the margin-sign alternation "
        "(2911 P1 p=0.0195) is unaffected"),
    "status": "confirmed_negative",
    "phase": 2912,
    "sources": [{
        "path": "phase2912/sign_balance_zigzag/result.json",
        "sha256_8": "4611db7f",
        "phase": 2912
    }],
    "notes": (
        "methodological constants institutionalized: (1) "
        "zigzag/local-extremum counts null at the 2/3 reversal "
        "law; (2) a one-sided discrete permutation p with ties "
        "included has E[p] = 0.5 + sum pk^2 / 2 - calibration "
        "bands on p must be tie-corrected (2912 v1->v2 lesson)")
}
L['negatives'].append(n13)

with open(P, 'w', encoding='utf-8') as f:
    json.dump(L, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
out = ('ledger updated: measurements %d, negatives %d, '
       'L14 connects %d, file sha256_8 %s'
       % (len(L['measurements']), len(L['negatives']),
          len(l14['connects']), h))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\ledger_2912_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(out + '\n')
print('OK ledger 2912')
