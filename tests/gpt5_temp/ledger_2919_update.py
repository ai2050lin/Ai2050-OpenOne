# -*- coding: utf-8 -*-
"""ledger_2919_update.py -- register M2919 + refine L14."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    L = json.load(f)

m_ids = [m.get('meas_id') for m in L['measurements']]
assert 'M2919_multiaxis_families' not in m_ids, 'already in'
assert m_ids[-1] == 'M2918_event_anatomy', m_ids[-1]

m2919 = {
    "meas_id": "M2919_multiaxis_families",
    "type": "multiaxis_direction_families",
    "verdict": (
        "multiaxis_families_ready - forward (17.9 s, qwen3-4b, "
        "200 frozen sentences) built per-layer direction "
        "families for speed/size/moisture under the 2886 "
        "last-token caliber, rows [0,36) input-to-block "
        "convention, plus collinearity audit BEFORE the "
        "multi-axis atlas (roadmap 2919). DESIGN: same-subject "
        "pairs (20 per axis, 'The {s} is fast./slow.' huge/"
        "tiny, wet/dry; pole convention dirs = unit(mean HIGH - "
        "mean LOW)) - subject contribution cancels exactly in "
        "the class-diff of means. ANCHORS: a1 fresh lang "
        "sentences vs 2886 npz S_last rel 0.0 (bit-exact "
        "deterministic forward), a2 dirs_lang rows [0,36) vs "
        "the 2917-consumed derivation rel 0.0; BONUS "
        "cross-phase check: recomputed lang LOO probe curve "
        "matches the 2886 stored curve bit-wise. P1 quality: "
        "LOO nearest-centroid best acc 1.0 for ALL axes (gate "
        "trivially passed early; informative part is the DEPTH "
        "PROFILE) - lang readable ~1.0 at ALL layers (matches "
        "2886 registered negative: hourglass_validated=False, "
        "no probe dip), while speed DECAYS deep (1.0 -> 0.65 "
        "at L31-35), size mild decay (-> 0.875), moist stays "
        "~0.95-1.0: ATTRIBUTE SEMANTICS GET INTEGRATED AWAY "
        "DEEP, LANGUAGE IDENTITY PERSISTS - axes have "
        "DIFFERENT DEPTH PROFILES. P2 collinearity: global max "
        "|cos| 0.4472 (speed-moist @L32), attr-attr 0.4472, "
        "lang-attr 0.1274 => no residualization needed for "
        "2920. P3 diff norms grow superlinearly with depth "
        "(lang 320.7 vs 78.7-94.0 attr at L35: language "
        "dominates the last-token class-diff). DEGENERATE ROW "
        "REGISTERED: layer-0 rows are exactly zero (last token "
        "= final period '.', embedding identical across "
        "sentences -> dirs[0] = 0 vector; explains sign_M[:,0]"
        "=0 in 2917/2918 curves); rows 1..35 substantive. "
        "CAVEAT: attribute dirs built from sentence contexts; "
        "2920 injects into single-word contexts - cross-"
        "protocol transfer is an explicit 2920 check (lang "
        "precedent: 2886 sentence dirs worked in 2913-2918 "
        "single-word injections)."),
    "source": {
        "path": "phase2919/multiaxis_direction_families/result.json",
        "sha256_8": "a5381b06",
        "phase": 2919
    }
}
L['measurements'].append(m2919)

l14 = None
for c in L['linkage']:
    if c.get('link_id') == 'L14_readout_spectrum_cross_model':
        l14 = c
        break
assert l14 is not None
assert 'M2919_multiaxis_families' not in l14['connects']
l14['connects'].append('M2919_multiaxis_families')
l14['notes'] += (
    " | M2919: speed/size/moisture direction families built "
    "under the 2886 caliber (same-subject pairs, bit-exact "
    "anchors vs 2886); collinearity low (global max |cos| "
    "0.447) so the multi-axis atlas needs no residualization; "
    "axes have DIFFERENT DEPTH PROFILES - lang readable ~1.0 "
    "at all layers (2886 hourglass negative confirmed), speed "
    "decays deep 1.0->0.65, size ->0.875, moist ~1.0 - "
    "attribute semantics integrated away deep while language "
    "identity persists")
l14['phase_updated'] = 2919

with open(P, 'w', encoding='utf-8') as f:
    json.dump(L, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
out = ('ledger updated: measurements %d, L14 connects %d, '
       'file sha256_8 %s'
       % (len(L['measurements']), len(l14['connects']), h))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\ledger_2919_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(out + '\n')
print('OK ledger 2919')
