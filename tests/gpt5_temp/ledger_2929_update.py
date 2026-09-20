# -*- coding: utf-8 -*-
"""Phase 2929 ledger update: append M2929 to measurements and
L14_readout_spectrum_cross_model.connects (35 -> 36).
Verify counts after write; print report file for disk recheck.
"""
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2929_ledger_update_report.txt')

M2929 = {
    "meas_id": "M2929_response_structure_atlas",
    "type": "response_structure_atlas",
    "verdict": (
        "skeleton_event_aligned - zero-forward (0.9 s) on the "
        "2927 npz, no model load. ANCHORS 3/3: a1 E17-event rho "
        "recompute vs 2928 rho_rows max diff 0.0 (exact "
        "reproduction); a2 sets 24/32/7 rebuilt; a3 sign_M86 vs "
        "2917 npz diff 0.0. P1 GRID-WIDE PROBE-INVARIANT "
        "SKELETON: rho_grid[h,l]=Spearman(B86[h,:,l],"
        "B_word[h,:,l]) over the 57 words; layer-wise "
        "head-permutation null (rng base 2904, 2000 perms/layer, "
        "vectorized average ranks) gives n_skeleton=501/1152 "
        "(43.5%) vs 5% expectation 57.6, exact binomial "
        "(lgamma domain) p=0.0 (underflow), rho median 0.2039 "
        "p90 0.7297 max 0.9659 => skeleton_present. REGISTERED "
        "DEGENERATE-TIE ARTIFACT: L0 all 32 heads have rho=0.0 "
        "and null p95=0.0 (degenerate rank vectors), the "
        "rho_obs >= p95 boundary 0>=0 admits all 32; true "
        "skeleton 469 (40.7%) is still 8.1x expectation - "
        "verdict unaffected. Skeleton peaks L8-L11 "
        "(22/23/24/19 of 32). P2 EVENT COUPLING: event cells 49 "
        "(E17 union Ewd) rho median 0.4862 vs background 1103 "
        "median 0.1945, U=37096.0, 10000-label-permutation "
        "p=0.0001 (1/10001 bound) => coupled. Group rho "
        "medians: survivor 0.8251 / lost 0.3774 / new 0.4743 "
        "(2928 P3 values reproduced exactly under a1). P3: no "
        "hard gap (skeleton min 0.0000 <= background max "
        "0.4862, soft overlap); 33/49 event cells inside the "
        "skeleton; per-group membership survivor 7/7 (1.000), "
        "lost 11/17 (0.647), new 15/25 (0.600). MECHANISM "
        "CLOSED: the probe-invariant response structure IS the "
        "survivor-core mechanism of 2928 - survivors are "
        "exactly the cells whose 57-word response pattern is "
        "probe-independent; maxT event selection samples this "
        "skeleton at 67% vs 42% background; 2928 overlap-null "
        "conclusion untouched (overlap chance-level stands; "
        "rho is the informative axis)."),
    "source": {
        "path": "phase2929/response_structure_atlas/"
                "response_structure_atlas.npz",
        "sha256_8": "57ed5651",
        "phase": 2929,
    },
}

d = json.load(open(P, encoding='utf-8'))
meas = d['measurements']
link = None
for it in d['linkage']:
    if it['link_id'] == 'L14_readout_spectrum_cross_model':
        link = it
assert len(meas) == 67, 'unexpected meas len %d' % len(meas)
assert meas[-1]['meas_id'] == 'M2928_survivor_core_anatomy'
assert link is not None and len(link['connects']) == 35
assert link['connects'][-1] == 'M2928_survivor_core_anatomy'

meas.append(M2929)
link['connects'].append('M2929_response_structure_atlas')
with open(P, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=1, ensure_ascii=False)

# disk recheck
d2 = json.load(open(P, encoding='utf-8'))
l14 = [it for it in d2['linkage']
       if it['link_id'] == 'L14_readout_spectrum_cross_model'][0]
out = ['measurements len: %d' % len(d2['measurements']),
       'last meas_id: %s' % d2['measurements'][-1]['meas_id'],
       'L14 connects len: %d' % len(l14['connects']),
       'L14 last3: %s' % l14['connects'][-3:]]
with open(REPORT, 'w', encoding='utf-8') as f:
    f.write('\n'.join(out) + '\n')
print('ledger update OK')
