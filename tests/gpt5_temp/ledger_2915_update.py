# -*- coding: utf-8 -*-
"""ledger_2915_update.py -- register M2915 + refine L14."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    L = json.load(f)

m_ids = [m.get('meas_id') for m in L['measurements']]
assert 'M2915_carrier_robustness_domain' not in m_ids, 'already in'
assert m_ids[-1] == 'M2914_head_identity_replication', m_ids[-1]

m2915 = {
    "meas_id": "M2915_carrier_robustness_domain",
    "type": "carrier_robustness_domain",
    "verdict": (
        "head_identity_broadly_stable - per-head jacobian re-run "
        "across 4 windows + 1 stale-direction control (V1 [26,36) "
        "REF / V2 [20,26) / V3 [30,36) / V4 [16,26), layer-matched "
        "2886 class-diff dirs; V5 = V1 window with the FIXED 2887 "
        "global lang_dir). Anchors 3/3: a1 per-variant block "
        "identity max 1.42e-15; a2 V1 margin -0.02139 (1 word vs "
        "2903); a3 V1 margins vs 2913 npz Spearman 1.000000 / rel "
        "2.73e-08 (third consecutive exact reproduction: 2914 "
        "relB 3.16e-08 same magnitude). Verdict axis "
        "n_top2({7,8} both in top2, V1-V4) = 2/4 => "
        "broadly_stable (n_top5 also 2/4). STRUCTURE: V3 (second "
        "half of V1) keeps {7,8} as top2 with ranks SWAPPED (h8 "
        "0.16431 #1, h7 0.15076 #2; top5 4/5 overlap with V1) "
        "but family gate False (6-layer null p95 0.20731 wider "
        "than max 0.16431); V2/V4 (layers <=26) switch carriers "
        "(V2 top2 {27,31}, h7 rank14 / h8 rank8; V4 top2 {7,27} "
        "with h7 rank1 at margin 1.17970, h8 rank10) - and the "
        "CHANNEL ITSELF is positive in the early segment (V2 "
        "+0.0367 acc 0.789; V4 +0.1600 acc 0.842) vs negative "
        "late segment (V1 -0.0214 acc 0.632): the {7,8} pair "
        "carries the LATE-SEGMENT CANCELLATION STRUCTURE, not a "
        "global language carrier; head-margin ordering is nearly "
        "orthogonal across segments (Spearman V1-V2 0.026, V1-V4 "
        "0.076) but consistent within the early segment (V2-V4 "
        "0.866). V5 stale fixed direction: carriers vanish (top2 "
        "{4,20}, h7 rank14, gate False, channel margin 0.0027) - "
        "head-level separation requires the layer-matched "
        "direction family (2903 readout spectrum reproduced at "
        "head level). h7 is the only cross-segment head (V1 #1, "
        "V3 #2, V4 #1)."),
    "source": {
        "path": "phase2915/carrier_robustness_domain/result.json",
        "sha256_8": "07314fa6",
        "phase": 2915
    }
}
L['measurements'].append(m2915)

l14 = None
for c in L['linkage']:
    if c.get('link_id') == 'L14_readout_spectrum_cross_model':
        l14 = c
        break
assert l14 is not None
assert 'M2915_carrier_robustness_domain' not in l14['connects']
l14['connects'].append('M2915_carrier_robustness_domain')
l14['notes'] += (
    " | M2915 robustness domain: {7,8} identity is SEGMENT-BOUND, "
    "not protocol-global - stable as top2 across late windows "
    "[26,36) and [30,36) (ranks swap in V3), switched in early "
    "windows <=26 (V2 {27,31}; V4 {7,27} with h7 #1 margin 1.18); "
    "channel margin is segment-signed (early +0.16 acc 0.84 vs "
    "late -0.02 acc 0.63), so the pair carries the late-segment "
    "cancellation structure rather than a global language "
    "carrier; stale fixed lang_dir kills head-level separation "
    "(V5 gate False, margin ~0) - head carriers require "
    "layer-matched directions")
l14['phase_updated'] = 2915

with open(P, 'w', encoding='utf-8') as f:
    json.dump(L, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
out = ('ledger updated: measurements %d, L14 connects %d, '
       'file sha256_8 %s'
       % (len(L['measurements']), len(l14['connects']), h))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\ledger_2915_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(out + '\n')
print('OK ledger 2915')
