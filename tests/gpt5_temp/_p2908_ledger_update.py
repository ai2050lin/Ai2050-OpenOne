# -*- coding: utf-8 -*-
"""_p2908_ledger_update.py -- register M2908, L14 refine."""
import hashlib
import json
import time

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\ledger_update_2908.txt')

led = json.load(open(P, encoding='utf-8'))

M2908 = {
    "meas_id": "M2908_qwen_attn_boundary_precision",
    "type": "high_power_percentile_adjudication",
    "verdict": (
        "qwen_attn_at_m1_edge - the 2907 both-M1 reading is "
        "confirmed with 800x draw mass (N_SYNTH=20000 x 5 "
        "independent seeds, RNG [2908,10+k]) and a percentile "
        "test p = P(margin_synth <= margin_full) with matched-"
        "protocol coverage audit 40/40. qwen_attn p_median=0.0276, "
        "seed range [0.0262,0.0295]: 5/5 seeds inside the nominal "
        "95% interval (p>0.025 for every seed - 'below M1' is "
        "refuted at high precision), but p_median falls in the "
        "preregistered 3*SE conservative band (0.0217,0.0283), "
        "0.0007 below the confirmed-inside threshold, so the "
        "honest reading is AT the M1 lower edge. Other groups "
        "deep inside (descriptive): qwen mlp p_med=0.474, glm4 "
        "mlp 0.375, glm4 attn 0.168 - under the E9-corrected RMS "
        "sigma the glm4 attn channel is firmly inside, closing "
        "the 2906 channel-split question. New spectrum dimension: "
        "percentile-within-own-null (p) orders qwen mlp 0.474 > "
        "glm4 mlp 0.375 > glm4 attn 0.168 > qwen attn 0.028, "
        "different from the margin-amplitude order (qwen mlp "
        "0.180 > glm4 attn 0.121 > glm4 mlp 0.020 > qwen attn "
        "-0.019) - relative-to-own-scatter position and absolute "
        "amplitude are independent axes"),
    "source": {
        "path": "phase2908/qwen_attn_boundary_precision/"
                "result.json",
        "sha256_8": "dff824f8",
        "phase": 2908,
    },
}

L14_ADD = (
    " | 2908 high-precision adjudication (20000 draws x 5 seeds + "
    "matched coverage audit 40/40): qwen_attn p_median=0.0276, "
    "5/5 seeds inside nominal 95% - 'below M1' refuted at high "
    "precision, honest reading AT the M1 lower edge; glm4 attn "
    "firmly inside (p=0.168) under E9-corrected RMS sigma, "
    "closing the 2906 channel-split question; new spectrum axis: "
    "percentile-within-own-null qwen mlp 0.474 > glm4 mlp 0.375 "
    "> glm4 attn 0.168 > qwen attn 0.028, independent of the "
    "margin-amplitude order")

meas = led['measurements']
assert all(m['meas_id'] != M2908['meas_id'] for m in meas)
meas.append(M2908)

link = None
for e in led['linkage']:
    if e.get('link_id') == 'L14_readout_spectrum_cross_model':
        link = e
        break
assert link is not None
assert '2908 high-precision' not in link['notes']
link['notes'] = link['notes'] + L14_ADD
if 'M2908_qwen_attn_boundary_precision' not in link['connects']:
    link['connects'].append('M2908_qwen_attn_boundary_precision')
link['phase_updated'] = 2908

with open(P, 'w', encoding='utf-8') as f:
    json.dump(led, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
lines = [
    'ledger updated %s' % time.strftime('%Y-%m-%d %H:%M:%S'),
    'measurements n=%d' % len(meas),
    'errata_ledger n=%d' % len(led['errata_ledger']),
    'linkage n=%d' % len(led['linkage']),
    'negatives n=%d' % len(led['negatives']),
    'growth_curve n=%d' % len(led.get('growth_curve', [])),
    'new ledger sha256_8 = %s' % h,
    'L14 connects n=%d last=%s' % (len(link['connects']),
                                   link['connects'][-1]),
    'L14 phase_updated=%s' % link['phase_updated'],
]
with open(OUT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines) + '\n')
print('OK', OUT)
