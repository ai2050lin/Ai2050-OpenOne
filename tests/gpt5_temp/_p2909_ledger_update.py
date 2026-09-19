# -*- coding: utf-8 -*-
"""_p2909_ledger_update.py -- register M2909, N12, L14 refine."""
import hashlib
import json
import time

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\ledger_update_2909.txt')

led = json.load(open(P, encoding='utf-8'))

M2909 = {
    "meas_id": "M2909_p_axis_stability",
    "type": "spectrum_axis_stability_grid",
    "verdict": (
        "p_axis_order_unstable - promotion of the null-percentile "
        "p (M2908) to a second primary spectrum axis REFUTED by "
        "its own preregistered stability grid (28 configs = 4 "
        "groups x [4 margin variants + 3 layer subsets], 5 seeds "
        "x 10000 draws each, 7 matched coverage audits 38-40/40 "
        "all pass; anchors 4/4). The four-channel ordering "
        "qwen_mlp 0.473 > glm4_mlp 0.380 > glm4_attn 0.168 > "
        "qwen_attn 0.026 is reproduced ONLY inside the cosine-"
        "mean margin family (V1_diagin 0.484>0.391>0.184>0.027 "
        "preserves it); column z-score V2 reorders to qwen_mlp "
        "0.535 > qwen_attn 0.097 > glm4_mlp 0.061 > glm4_attn "
        "0.052, LOO-acc V3 reorders completely (glm4_mlp 0.874 > "
        "qwen_mlp 0.855 > qwen_attn 0.735 > glm4_attn 0.378), "
        "and every layer subset reorders as well. glm4_attn "
        "interior robustness (p_med in [0.05,0.95]) is the only "
        "invariant component (all 7 configs, min 0.052 at V2). "
        "New structure finding: qwen_attn lower-tail status "
        "(p<0.05) is a FULL-LAYER AGGREGATION effect - S_front "
        "0.140, S_back 0.130, S_key (argmax|delta| layer alone) "
        "0.607 are each null-like, only the full-layer cosine "
        "mean aggregates the small per-layer effects into "
        "p=0.026. The 2908 ordering and its qwen_attn edge "
        "reading remain valid strictly under the declared "
        "2896-family definition (see N12)"),
    "source": {
        "path": "phase2909/p_axis_stability/result.json",
        "sha256_8": "2fb42fa1",
        "phase": 2909,
    },
}

N12 = {
    "neg_id": "N12_p_axis_not_score_family_invariant",
    "kind": "spectrum_axis_promotion_refuted",
    "claim": (
        "the null-percentile p = P(score_synth <= score_true) "
        "(M2908) is a score-family- and layer-subset-invariant "
        "second spectrum axis, with qwen_attn robustly in the "
        "lower tail across scoring definitions and layer subsets"),
    "evidence": (
        "2909 preregistered grid (28 configs, 5 seeds x 10000 "
        "draws, 7 coverage audits pass): ordering reproduced only "
        "under V1_diagin (cosine family); V2_colz reorders to "
        "qwen_mlp>qwen_attn>glm4_mlp>glm4_attn (qwen_attn "
        "0.026->0.097), V3_acc reorders completely (qwen_attn "
        "0.735); layer subsets scored by V0: qwen_attn S_front "
        "0.140 / S_back 0.130 / S_key 0.607 - the lower tail "
        "vanishes in every proper subset; glm4_attn stays "
        "interior in all 7 configs (only invariant component)"),
    "status": "refuted",
    "notes": (
        "p values are definition-relative statistics, not "
        "invariants; the surviving structural finding is that "
        "qwen_attn's lower tail is a full-layer aggregation "
        "effect (per-layer effects individually null-like, "
        "aggregate p=0.026 under the declared definition); the "
        "margin-amplitude hierarchy (L14 primary axis) is "
        "unaffected"),
    "phase": 2909,
    "sources": [
        {"path": "phase2909/p_axis_stability/result.json",
         "sha256_8": "2fb42fa1",
         "phase": 2909}
    ],
}

L14_ADD = (
    " | 2909 p-axis promotion REFUTED (N12): null-percentile "
    "ordering invariant only within the cosine-mean margin "
    "family; V2_colz / V3_acc / layer subsets reorder it "
    "(qwen_attn 0.026 -> 0.097 / 0.735 / 0.13-0.61); qwen_attn "
    "lower tail is a full-layer aggregation effect - second "
    "axis withdrawn, margin-amplitude hierarchy remains the "
    "primary spectrum axis under the declared 2896-family "
    "definition")

meas = led['measurements']
assert all(m['meas_id'] != M2909['meas_id'] for m in meas)
meas.append(M2909)

neg = led['negatives']
assert all(n.get('neg_id') != N12['neg_id'] for n in neg)
neg.append(N12)

link = None
for e in led['linkage']:
    if e.get('link_id') == 'L14_readout_spectrum_cross_model':
        link = e
        break
assert link is not None
assert '2909 p-axis' not in link['notes']
link['notes'] = link['notes'] + L14_ADD
if 'M2909_p_axis_stability' not in link['connects']:
    link['connects'].append('M2909_p_axis_stability')
link['phase_updated'] = 2909

with open(P, 'w', encoding='utf-8') as f:
    json.dump(led, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
lines = [
    'ledger updated %s' % time.strftime('%Y-%m-%d %H:%M:%S'),
    'measurements n=%d' % len(meas),
    'errata_ledger n=%d' % len(led['errata_ledger']),
    'negatives n=%d' % len(neg),
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
