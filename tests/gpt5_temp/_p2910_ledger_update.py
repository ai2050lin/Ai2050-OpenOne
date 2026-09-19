# -*- coding: utf-8 -*-
"""_p2910_ledger_update.py -- register M2910, L14 refine."""
import hashlib
import json
import time

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\ledger_update_2910.txt')

led = json.load(open(P, encoding='utf-8'))

M2910 = {
    "meas_id": "M2910_cross_layer_coherence",
    "type": "per_layer_aggregation_coherence",
    "verdict": (
        "qwen_attn_partial_coherence (below_frac 0.50, between "
        "the frozen 0.90 coherent and 0.50 cancellation bounds) - "
        "anchors 4/4, coverage audits d=1 39/40 + d=5 40/40 pass. "
        "Per-layer d=1 percentiles reveal an ALTERNATING pattern: "
        "L00 0.327(-) L01 0.778(+) L02 0.104(-) L03 0.697(+) L04 "
        "0.253(-) L05 0.849(+) L06 0.556(~) L07 0.205(-) L08 "
        "0.609(+) L09 0.143(-) - adjacent window layers alternate "
        "below/above their own nulls (L00-L05 strictly). The "
        "cumulative curve is decisive against a pure averaging "
        "artifact: cum_p k=1..4 oscillates 0.34-0.54, then k=5 "
        "0.139, k=6 0.085, k=7 0.022, k=8 0.010, stabilising "
        "0.029/0.025 at k=9/10 - monotone descent into the deep "
        "lower tail from the second half onward, and the k=10 "
        "value 0.025 reproduces the independently-seeded 2908 "
        "full-layer p=0.026. Half-block values also cross-check "
        "(S_front 0.140 at 2909 vs cum k=5 0.139). Reading: the "
        "lower-tail status is carried jointly by an alternating "
        "layer structure and a dominant second-half aggregation, "
        "not by single layers (S_key 0.607) nor by uniform "
        "coherence. Descriptive: glm4_attn below_frac 0.42, "
        "cumulative descends early (k=3 0.143) but stabilises "
        "0.12-0.24 (interior); glm4_mlp 0.50 with late-layer "
        "descent reversed by the final two layers (0.563/0.377); "
        "qwen_mlp 0.60 dominated by L01 (score 1.0195, p 0.954) "
        "- only qwen_attn reaches the deep tail"),
    "source": {
        "path": "phase2910/cross_layer_coherence/result.json",
        "sha256_8": "48001e65",
        "phase": 2910,
    },
}

L14_ADD = (
    " | 2910 coherence decomposition: qwen_attn lower tail = "
    "alternating per-layer structure (L00-L05 strictly below/"
    "above own nulls) + dominant second-half cumulative descent "
    "(k=5 0.139 -> k=8 0.010 -> k=10 0.025, reproducing 2908's "
    "0.026 under independent seeds); partial_coherence verdict "
    "(below_frac 0.50) - aggregation is structured, not a "
    "uniform coherence nor an averaging artifact")

meas = led['measurements']
assert all(m['meas_id'] != M2910['meas_id'] for m in meas)
meas.append(M2910)

link = None
for e in led['linkage']:
    if e.get('link_id') == 'L14_readout_spectrum_cross_model':
        link = e
        break
assert link is not None
assert '2910 coherence' not in link['notes']
link['notes'] = link['notes'] + L14_ADD
if 'M2910_cross_layer_coherence' not in link['connects']:
    link['connects'].append('M2910_cross_layer_coherence')
link['phase_updated'] = 2910

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
