# -*- coding: utf-8 -*-
"""ledger_2926_update.py -- register M2926 + refine L14."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    L = json.load(f)

m_ids = [m.get('meas_id') for m in L['measurements']]
assert 'M2926_event_selection_polar_fix' not in m_ids, 'already in'
assert m_ids[-1] == 'M2925_event_selection_anatomy', m_ids[-1]

m2926 = {
    "meas_id": "M2926_event_selection_polar_fix",
    "type": "event_selection_polar_fix",
    "verdict": (
        "events_polar_separation_selected (criterion-corrected "
        "confirmatory re-test of M2925) - zero-forward |phi| "
        "retest (0.4 s), 2925 protocol verbatim with phi->|phi| "
        "and neighbor_phi->max|phi| (both signed-form defects "
        "corrected). ANCHORS 3/3: a1 ~6e-10; a2 d_pole13 "
        "bit-exact 0.0; a3 -0.060746. P1 FLIP CONFIRMED: n_BH "
        "3/7 (phi_abs, d_pole_abs, contrast_raw all sign p "
        "1.22e-4, BH q 2.85e-4) and median pct(phi_abs) = 1.0 "
        "with 13/13 cells at pct 1.0 (non-event-head pool) => "
        "the frozen 2925 mapping flips margin_only -> "
        "polar_separation_selected exactly as predicted by the "
        "M2925 post-hoc diagnostic. P2 |phi|-margin INDEPENDENCE "
        "TEST: NOT independent - rho13(Spearman over 13 sig "
        "cells) = 0.978, p_perm = 0.0 (5000 perms rng(2901), "
        "null max 0.863), full-grid per-axis Spearman 0.81-0.90 "
        "=> |phi| and sign-Gram margin are near-monotone "
        "aliases of one underlying quantity (symbol-consistency "
        "strength), NOT a new selection dimension; the selection "
        "mechanism is ONE axis: polar contrast/symbol-"
        "consistency strength extremum. P3 layer-internal rank: "
        "12/13 rank-1; the single rank-2 cell is (18,7) at L7, "
        "suppressed by same-layer double event (21,7) (|phi| "
        "0.811 vs 0.606; margin 1.238 vs 0.654) - L7 is a "
        "double-selection layer (2922 linked pair rho 0.676); "
        "P1 pool (non-event heads) still 13/13 = 1.0, so the "
        "M2925 diagnostic reproduces exactly under its own "
        "pool definition. CAVEAT: correction originates from "
        "the M2925 post-hoc diagnostic (quasi-post-hoc, "
        "annotated in prereg correction_note) - verdict weight "
        "is confirmatory-of-prediction, not blind. NEXT 2927 "
        "candidates: A probe relativity (lang events under "
        "word-level dirs_word probe, one forward ~1 min), B h4 "
        "L1<->L19 reuse principal angles (zero-forward), C "
        "contrast ceiling profile (layer-wise |phi| distribution "
        "shape: gap vs continuum below sig cells), D L7 "
        "double-selection anatomy (why one layer hosts two "
        "events of the same axis - head/layer division of "
        "labor)."),
    "source": {
        "path": "phase2926/event_selection_polar_fix/result.json",
        "sha256_8": "999151fa",
        "phase": 2926
    }
}
L['measurements'].append(m2926)

l14 = None
for c in L['linkage']:
    if c.get('link_id') == 'L14_readout_spectrum_cross_model':
        l14 = c
        break
assert l14 is not None
assert 'M2926_event_selection_polar_fix' not in l14['connects']
l14['connects'].append('M2926_event_selection_polar_fix')
l14['notes'] += (
    " | M2926: |phi| corrected re-test flips the frozen verdict "
    "to events_polar_separation_selected (3/7 BH features: "
    "phi_abs pct 13/13=1.0, d_pole_abs, contrast_raw); P2 shows "
    "|phi|~margin coupling rho13 0.978 (full-grid 0.81-0.90) - "
    "selection is ONE axis, symbol-consistency strength extremum; "
    "P3 12/13 rank-1, the exception (18,7) is suppressed by "
    "same-layer double event (21,7) at L7")
l14['phase_updated'] = 2926

with open(P, 'w', encoding='utf-8') as f:
    json.dump(L, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
out = ('ledger updated: measurements %d, L14 connects %d, '
       'file sha256_8 %s'
       % (len(L['measurements']), len(l14['connects']), h))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\ledger_2926_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(out + '\n')
print('OK ledger 2926')
