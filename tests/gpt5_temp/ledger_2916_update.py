# -*- coding: utf-8 -*-
"""ledger_2916_update.py -- register M2916 + refine L14."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    L = json.load(f)

m_ids = [m.get('meas_id') for m in L['measurements']]
assert 'M2916_early_carrier_selection' not in m_ids, 'already in'
assert m_ids[-1] == 'M2915_carrier_robustness_domain', m_ids[-1]

m2916 = {
    "meas_id": "M2916_early_carrier_selection",
    "type": "early_carrier_selection",
    "verdict": (
        "early_carriers_selection_confirmed - artifact-domain "
        "(zero forward, 0.6 s) selection-corrected greedy test "
        "(2913 P3 caliber verbatim) on the 2915 npz B_heads: V2 "
        "(early window [20,26)) top carrier h27 ALONE p3=0.004975 "
        "(obs 0.78360, k=1); V4 ([16,26)) top carrier h7 ALONE "
        "p3=0.004975 (obs 1.17970, k=1); both <= 0.05 => "
        "confirmed - the early-segment carriers are statistically "
        "real and need only ONE head each. V1 test EXACTLY "
        "replicates 2913 P3 (p3 0.014925, absdev 0.0, k=2 {7,8} "
        "obs 0.28036); anchors a1 Spearman 1.0 / rel 0.0 (2915 npz "
        "margins bit-identical to 2913 after fp32 storage), a2 "
        "m78 0.28036. P2 leave-one-layer-out decomposition: EVERY "
        "carrier margin is dominated by a SINGLE (head, layer) "
        "event - V4 h7 1.17970 collapses to 0.121 without L19 "
        "(delta -1.059, sign peak L19 1.3463); V2 h27 0.78360 -> "
        "L24 (delta -0.773); V2 h31 0.44323 -> L22 (-0.381); V4 "
        "h27 0.78650 -> L24 (-0.661); late-segment V1 h7 0.17710 "
        "-> L34 (-0.157, sign peak L34 0.187) and h8 0.13426 -> "
        "L34 (-0.071); V4 h8 0.21920 -> L23 (-0.143). P3: top "
        "sign-margin events V1 = h17@L28 (0.568), h10@L34 (0.493), "
        "h8@L34; V4 = h7@L19 (1.346), h24@L23 (1.016), h13@L22. "
        "KEY READING: the 'carrier head' is really a (head, layer) "
        "EVENT - h7's cross-segment dual role is two distinct "
        "events (h7@L19 early, h7@L34 late); head identity varies "
        "across windows because different windows contain "
        "different events; the 2913 P3 {7,8} significance is the "
        "sum of two events h7@L34 + h8@L34 at the SAME layer 34, "
        "suggesting a layer-34 mechanism recruiting both heads "
        "rather than a head-pair mechanism spread over layers."),
    "source": {
        "path": "phase2916/early_carrier_selection/result.json",
        "sha256_8": "fc0a8b7a",
        "phase": 2916
    }
}
L['measurements'].append(m2916)

l14 = None
for c in L['linkage']:
    if c.get('link_id') == 'L14_readout_spectrum_cross_model':
        l14 = c
        break
assert l14 is not None
assert 'M2916_early_carrier_selection' not in l14['connects']
l14['connects'].append('M2916_early_carrier_selection')
l14['notes'] += (
    " | M2916: early-segment carriers selection-confirmed (V2 h27 "
    "p=0.005 k=1, V4 h7 p=0.005 k=1); leave-one-layer-out shows "
    "EVERY carrier margin is a single (head,layer) event - h7@L19 "
    "(early, -1.059 of 1.18), h27@L24, h31@L22, and late "
    "h7@L34+h8@L34 (same layer) - 'carrier head' = (head,layer) "
    "event, h7's dual role is two distinct events, and the 2913 "
    "{7,8} significance is a layer-34 pair event")
l14['phase_updated'] = 2916

with open(P, 'w', encoding='utf-8') as f:
    json.dump(L, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
out = ('ledger updated: measurements %d, L14 connects %d, '
       'file sha256_8 %s'
       % (len(L['measurements']), len(l14['connects']), h))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\ledger_2916_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(out + '\n')
print('OK ledger 2916')
