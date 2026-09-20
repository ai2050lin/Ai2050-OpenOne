# -*- coding: utf-8 -*-
"""ledger_2917_update.py -- register M2917 + refine L14 (with run1 BH erratum history)."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    L = json.load(f)

m_ids = [m.get('meas_id') for m in L['measurements']]
assert 'M2917_event_atlas' not in m_ids, 'already in'
assert m_ids[-1] == 'M2916_early_carrier_selection', m_ids[-1]

m2917 = {
    "meas_id": "M2917_event_atlas",
    "type": "event_atlas",
    "verdict": (
        "event_atlas_not_replicated (with material structure found) - "
        "full-layer (head,layer) event atlas: 32 heads x 36 layers = 1152 "
        "sign-margin cells (2910 caliber), dirs = 2886 class-diff unit "
        "directions per layer, one forward capture, null = 200 mask "
        "perms SEED=2896. EXECUTION HISTORY: run1 adjudicated under "
        "per-event BH-FDR q=0.05 and returned 0/1152 - audited as a "
        "design error with STRUCTURALLY ZERO POWER (smallest attainable "
        "per-event permutation p = 1/201 = 0.004975 while the BH first "
        "threshold is q/m = 0.05/1152 = 4.34e-5; BH could only fire if "
        ">=115 events sat at the granularity floor simultaneously); "
        "verdict label kept as frozen, criterion re-frozen as v2 maxT "
        "(Westfall-Young single-step: p_maxT = (1 + #{perms: "
        "max_{h,l} sign_perm >= sign(event)})/201, BH demoted to "
        "descriptive), artifacts wiped and rerun. v2 RESULT: 24/1152 "
        "events family-wise significant. KNOWN 2/6 survive: (7,19) "
        "margin 1.34634 p_maxT=0.005 - the TOP event of the entire "
        "atlas, matching the 2916 LOO sign peak L19 1.3463 - and "
        "(27,24) 0.81057 p=0.010; NOT surviving: (31,22) 0.456 "
        "p=0.189, (8,23) 0.171 p=1.0, (7,34) 0.187 p=1.0, (8,34) "
        "0.297 p=0.896 - the late-segment window events are "
        "WINDOW-FAMILY-RELATIVE (significant against a ~32-head "
        "window family, not against the full 1152-cell family). "
        "NOVEL 22 events concentrated at EARLY layers (16/24 "
        "significant events at layer<=16): h26@L6 1.23584, h8@L2 "
        "1.12525, h25@L3 1.12278, h22@L12 1.11380, h24@L23 1.01633, "
        "h21@L6 0.91948, h13@L22 0.82254, h5@L6 0.81462, h6@L19 "
        "0.81057, h4@L1 0.80581, h21@L16 0.73379, h4@L22 0.72903, "
        "h20@L8 0.72604, h1@L6 0.64759, h4@L19 0.64495, h14@L9 "
        "0.63896, h1@L4 0.56835, h15@L13 0.56817, h17@L28 0.56817, "
        "h26@L5 0.56659, h28@L1 0.56659, h24@L9 0.56571. READING: "
        "the atlas is NOT closed - the two 2916 single-layer driver "
        "events (7,19) and (27,24) are real at the strongest "
        "correction level, the four late-window events are "
        "family-size artifacts of window-restricted testing, and a "
        "population of early-layer (L1-L16) lang-diff events enters "
        "at full-family level that window scans (2915: [16,36) only) "
        "never tested. Anchors a1 rel 3.16e-08, a2 m78 0.28036 "
        "absdiff 0.0."),
    "source": {
        "path": "phase2917/event_atlas/result.json",
        "sha256_8": "1f988c40",
        "phase": 2917
    }
}
L['measurements'].append(m2917)

l14 = None
for c in L['linkage']:
    if c.get('link_id') == 'L14_readout_spectrum_cross_model':
        l14 = c
        break
assert l14 is not None
assert 'M2917_event_atlas' not in l14['connects']
l14['connects'].append('M2917_event_atlas')
l14['notes'] += (
    " | M2917: full-layer 1152-cell event atlas, v2 maxT after v1 "
    "BH power-zero erratum (min perm p 1/201 > BH first threshold "
    "4.34e-5): 24/1152 family-wise significant; known 2/6 - (7,19) "
    "1.346 p=0.005 top-of-atlas, (27,24) 0.811 p=0.010; late-window "
    "events (31,22)/(8,23)/(7,34)/(8,34) are window-family-relative "
    "only; 22 novel events concentrated at early layers L1-L16 "
    "(h26@L6 1.236, h8@L2 1.125, h25@L3 1.123, h22@L12 1.114) - "
    "window-restricted scans never tested early layers")
l14['phase_updated'] = 2917

with open(P, 'w', encoding='utf-8') as f:
    json.dump(L, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
out = ('ledger updated: measurements %d, L14 connects %d, '
       'file sha256_8 %s'
       % (len(L['measurements']), len(l14['connects']), h))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\ledger_2917_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(out + '\n')
print('OK ledger 2917')
