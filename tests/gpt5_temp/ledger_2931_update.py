# -*- coding: utf-8 -*-
"""Phase 2931 ledger update: append M2931 to measurements and
L14_readout_spectrum_cross_model.connects (37 -> 38).
"""
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2931_ledger_update_report.txt')

M2931 = {
    "meas_id": "M2931_skeleton_overlap_null",
    "type": "skeleton_overlap_null",
    "verdict": (
        "skeleton_overlap_above_chance - zero forward (2 s) "
        "on the 2927/2929/2930 npz. ANCHORS 4/4: a1 rho_grid "
        "recompute diff 2.22e-16; a2 rho_mirror recompute "
        "diff 0.0; a3 sets 501/414/inter 327 rebuilt; a4 "
        "2929 null_p95 replay (rng 2904) diff 0.0. "
        "DEGENERATE GATE (discipline 12): exactly 32 cells "
        "all in L0 (rank-std <= 1e-10; exact-zero rho cells "
        "= 32 = degenerate cells, no other layer affected); "
        "corrected skeletons 469/382, intersection 295, "
        "obs_jacc 0.5306. P1 MAIN: independent double "
        "within-layer head permutation null (rng 2906, "
        "R=1000) gives jaccard median 0.2711 p95 0.2953 max "
        "0.3173 - observed 0.5306 exceeds ALL 1000 null "
        "replicates (P(null>=obs)=0.0000); excess vs "
        "independence +139.5 cells (ratio 1.897). P1b "
        "sanity: same-permutation pairing reproduces obs "
        "bit-exactly on every replicate. P2 cell-level "
        "Spearman(rho29, rhomir)=0.6709; worst layer L32 "
        "-0.4223 (strong-nonlinearity reversal, lin_r "
        "1.66), best L11 0.9806. SURVIVOR-CORE 7/7 IN BOTH "
        "CORRECTED SKELETONS. SEAL STRUCTURE: skeleton "
        "convention-invariance is LAYER-STRUCTURED - "
        "quasi-linear layers (lin_r<0.9, L4-L17) reproduce "
        "the skeleton almost cell-by-cell (L10 24/24/24, "
        "L9 23/23/22, L8 22/23/21), strong-nonlinearity "
        "layers (lin_r>1.4, L24-L35) collapse on the "
        "mirror side (L32 15/2/0, L31 15/3/1) while the "
        "original caliber keeps its cells - lin_r is the "
        "predictor of convention robustness. CONTRAST WITH "
        "2928: maxT significant-set overlap was EXACTLY "
        "chance (null median 7 = obs 7); rho-skeleton "
        "overlap is 1.9x independence and beyond all null "
        "replicates - the overlap-no-information law is "
        "selector-specific, not universal: rho (structure) "
        "earns overlap claims, maxT (selection) does not."),
    "source": {
        "path": "phase2931/skeleton_overlap_null/"
                "skeleton_overlap_null.npz",
        "sha256_8": "5307afe1",
        "phase": 2931,
    },
}

d = json.load(open(P, encoding='utf-8'))
meas = d['measurements']
link = None
for it in d['linkage']:
    if it['link_id'] == 'L14_readout_spectrum_cross_model':
        link = it
assert len(meas) == 69, 'unexpected meas len %d' % len(meas)
assert meas[-1]['meas_id'] == 'M2930_direction_flip_control'
assert link is not None and len(link['connects']) == 37
assert link['connects'][-1] == 'M2930_direction_flip_control'

meas.append(M2931)
link['connects'].append('M2931_skeleton_overlap_null')
with open(P, 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=1, ensure_ascii=False)

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
