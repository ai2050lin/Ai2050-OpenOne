# -*- coding: utf-8 -*-
"""_p2911_ledger_update.py -- register M2911, L14 refine."""
import hashlib
import json
import time

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\ledger_update_2911.txt')

led = json.load(open(P, encoding='utf-8'))

M2911 = {
    "meas_id": "M2911_alternation_structure",
    "type": "alternation_formal_test",
    "verdict": (
        "alternation_not_confirmed_margin_only - anchors 4/4 (incl. "
        "per-layer d=1 scores vs 2910 score_true, maxabs 5.0e-7) "
        "and permutation-null calibration pass (frac in band 0.88, "
        "p_median 0.505). P1 CONFIRMS the margin-level "
        "alternation as real and qwen_attn-specific: adjacent-"
        "sign flips of per-layer d=1 margin scores 8/9, exact "
        "binomial p=0.0195, while glm4_mlp 6/11, glm4_attn 6/11, "
        "qwen_mlp 6/9 all p>=0.25. P3 REFUTES extension to the "
        "column-correlation structure: osc = mean_corr(adj) - "
        "mean_corr(skip1) is POSITIVE for qwen_attn (+0.050, "
        "perm p=0.728) and no group is significant (S empty) - "
        "adjacent B columns are if anything MORE correlated than "
        "two-apart, i.e. the raw responses vary smoothly across "
        "layers. P2 (delta signs) also null for qwen_attn (5/9, "
        "p=0.50); descriptive side-finding: glm4_attn delta signs "
        "are 0/11 flips (all-same-sign block, opposite-tail "
        "p~0.0005). Post-hoc sign decomposition (diag_2911c, "
        "descriptive): the alternation carrier is the CLASS-WISE "
        "SIGN BALANCE - per-layer gap |pos_frac(class0) - "
        "pos_frac(class1)| zigzags 7/8 for qwen_attn and tracks "
        "margin_j (large gap -> positive margin, small gap -> "
        "negative), driven mainly by class-0 positive-rate "
        "fluctuation (pf0 range 0.32-0.86 vs pf1 0.37-0.60); the "
        "same gap->margin law holds across groups (glm4_attn "
        "L08/L09 gap 0.37/0.41 -> margin 0.273/0.276). Reading: "
        "the alternation lives in the per-layer SIGN-SEPARATION "
        "QUALITY, not in signal direction or amplitude"),
    "source": {
        "path": "phase2911/alternation_structure/result.json",
        "sha256_8": "61a08678",
        "phase": 2911,
    },
}

L14_ADD = (
    " | 2911 alternation formal tests: margin-level alternation "
    "real and qwen_attn-specific (8/9 flips p=0.0195; others "
    "p>=0.25) but absent from column correlations (osc positive "
    "+0.050, S empty) and delta signs (5/9) - carrier located "
    "(descriptive diag_2911c) in the class-wise sign balance: "
    "per-layer |pos_frac0 - pos_frac1| zigzags 7/8 and tracks "
    "margin_j; sign-separation quality oscillates while signal "
    "direction/amplitude stay smooth")

meas = led['measurements']
assert all(m['meas_id'] != M2911['meas_id'] for m in meas)
meas.append(M2911)

link = None
for e in led['linkage']:
    if e.get('link_id') == 'L14_readout_spectrum_cross_model':
        link = e
        break
assert link is not None
assert '2911 alternation' not in link['notes']
link['notes'] = link['notes'] + L14_ADD
if 'M2911_alternation_structure' not in link['connects']:
    link['connects'].append('M2911_alternation_structure')
link['phase_updated'] = 2911

with open(P, 'w', encoding='utf-8') as f:
    json.dump(led, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
lines = [
    'ledger updated %s' % time.strftime('%Y-%m-%d %H:%M:%S'),
    'measurements n=%d' % len(meas),
    'errata_ledger n=%d' % len(led['errata_ledger']),
    'negatives n=%d' % len(led['negatives']),
    'linkage n=%d' % len(led['linkage']),
    'growth_curve n=%d' % len(led.get('growth_curve', [])),
    'new ledger sha256_8 = %s' % h,
    'L14 connects n=%d last=%s' % (len(link['connects']),
                                   link['connects'][-1]),
    'L14 phase_updated=%s' % link['phase_updated'],
]
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK', OUT)
