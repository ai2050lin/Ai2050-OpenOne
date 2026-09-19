# -*- coding: utf-8 -*-
"""_p2907_ledger_update.py -- register M2907, E9 errata, L14 refine."""
import hashlib
import json
import time

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
OUT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
       r'\ledger_update_2907.txt')

led = json.load(open(P, encoding='utf-8'))

M2907 = {
    "meas_id": "M2907_shape_correction_law",
    "type": "shape_correction_ladder_test",
    "verdict": (
        "frozen mapping label shape_correction_mixed is a MAPPING "
        "GAP (both-M1 case not enumerated); actual result "
        "BOTH-ATTN-GROUPS-LEVEL-M1. anchors 4/4 (incl. Delta_B vs "
        "2905 round4 maxabs 4.9e-5); coverage audits M2a 39/40 + "
        "M2b 39/40 both pass. Ladder M1 (mu_c + sigma_c*I, "
        "sigma_c = RMS sqrt(tr(Sigma_c)/d)) -> M2a (diag sqrt) -> "
        "M2b (chol): ALL FOUR groups fall inside M1 already "
        "(glm4 attn 0.1213 in [0.0915,0.2262]; qwen attn -0.0195 "
        "in [-0.0198,0.0631] borderline slack 0.0004; glm4 mlp "
        "0.0198 in [-0.0035,0.0692]; qwen mlp 0.1799 in "
        "[0.0970,0.2911]) - no shape correction needed at the "
        "preregistered RMS definition. Post-hoc 2x2x2 attribution "
        "grid (diag_2907b; recheck anchors reproduce registered "
        "2906/2907 intervals to 9e-7): glm4 attn M1 verdict is "
        "100% sigma-definition driven (mean-std: below in 4/4 "
        "combos; rms: inside in 4/4, independent of seed and "
        "stream structure); qwen attn is MC-noise borderline "
        "(rms: inside 2/4, below 2/4). 2906 both-attn-below was "
        "partly a prereg-text-vs-implementation sigma drift (E9); "
        "'attn shape actively suppresses margin' is retracted as "
        "an amplitude fact, surviving only as: qwen attn sits AT "
        "the M1 lower edge within MC resolution"),
    "source": {
        "path": "phase2907/shape_correction_law/result.json",
        "sha256_8": "390f27c9",
        "phase": 2907,
    },
}

E9 = {
    "corrects": "M2906_amplitude_law_neuron_attribution",
    "note": (
        "the 2906 prereg TEXT froze sigma_hat_c = "
        "sqrt(tr(Sigma_c)/d) (RMS) but the implementation used "
        "B[m].std(0).mean() (mean-std, a downward-biased proxy; "
        "ratio rms/mean-std measured 1.06-1.34 across groups, "
        "Jensen inequality). The 2906 coverage audit used the same "
        "mean-std estimator, so the drift was self-consistent and "
        "passed. Recomputing M1 intervals under the preregistered "
        "RMS definition (diag_2907b 2x2x2 grid; both recheck "
        "anchors reproduce registered intervals to 9e-7): glm4 "
        "attn flips to INSIDE in 4/4 combos; qwen attn is "
        "borderline (inside 2/4, MC-noise level). The 2906 frozen "
        "verdict amplitude_law_not_established remains on record "
        "per discipline, but its attn-below-M1 reading is not "
        "robust to the sigma definition; M2907 implements the "
        "preregistered definition and finds both attn groups at "
        "level M1. Lesson: prereg text-vs-implementation drift "
        "must be caught by an independent audit read of the "
        "implementation itself, not by an audit whose estimator "
        "shares the drifted definition."),
    "phase": 2907,
}

L14_ADD = (
    " | 2907 shape ladder: both attn groups reconstruct under M1 "
    "isotropy with the preregistered RMS sigma (glm4 attn 0.1213 "
    "in [0.0915,0.2262]; qwen attn -0.0195 in [-0.0198,0.0631], "
    "borderline slack 0.0004) - shape correction NOT needed; 2906 "
    "attn-below was a sigma-definition artifact (E9), 'attn shape "
    "suppresses margin' retracted as amplitude fact; spectrum "
    "stands as amplitude hierarchy with qwen attn at the M1 "
    "lower edge")

meas = led['measurements']
assert all(m['meas_id'] != M2907['meas_id'] for m in meas)
meas.append(M2907)
led['errata_ledger'].append(E9)

link = None
for e in led['linkage']:
    if e.get('link_id') == 'L14_readout_spectrum_cross_model':
        link = e
        break
assert link is not None
assert '2907 shape ladder' not in link['notes']
link['notes'] = link['notes'] + L14_ADD
if 'M2907_shape_correction_law' not in link['connects']:
    link['connects'].append('M2907_shape_correction_law')
link['phase_updated'] = 2907

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
