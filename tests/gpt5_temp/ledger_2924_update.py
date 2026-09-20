# -*- coding: utf-8 -*-
"""ledger_2924_update.py -- register M2924 + refine L14."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    L = json.load(f)

m_ids = [m.get('meas_id') for m in L['measurements']]
assert 'M2924_depth_polarity_gradient' not in m_ids, 'already in'
assert m_ids[-1] == 'M2923_polarity_sign_anatomy', m_ids[-1]

m2924 = {
    "meas_id": "M2924_depth_polarity_gradient",
    "type": "depth_polarity_gradient",
    "verdict": (
        "depth_gradient_absent - zero-forward depth-polarity "
        "gradient test (1.2 s): is 'shallow LOW-driven / deep "
        "HIGH-driven' (2923, sig events) a FULL-GRID continuous "
        "gradient? Statistic: per-axis layer profile c_ax[l] = "
        "median_h D[h,l], D = m_hi - m_lo over ALL 32 heads; "
        "rho_axis = Spearman(l, c_ax); null = 2000 size-"
        "preserving pole permutations (rng 2899). ANCHORS 3/3: "
        "a1 ~6e-10; a2 d_pole13 recompute BIT-EXACT (max diff "
        "0.0); a3 2923 t5_perm p50 6.0 max 11. MAIN TEST "
        "NEGATIVE 0/3 axes: speed rho +0.203 p 0.175 (direction "
        "right, ns), size rho -0.061 p 0.607 (REVERSED: shallow "
        "+1.1e-05 -> deep -2.4e-05), moist rho -0.054 p 0.640 "
        "(both ends LOW-side); per-layer sig cells scattered "
        "(speed 4, size 2, moist 0), no coherent gradient; "
        "lang en/L control rho -0.079 p 0.632 (no gradient, "
        "clean control). YET event-level structure HOLDS: "
        "Spearman(d_pole, event layer) over the 13 sig events "
        "= +0.654, p_perm 0.025 (quasi-post-hoc, rng 2900) - "
        "and rho(|d_pole|, l) = -0.495 (shallow events have "
        "STRONGER contrast). INTERPRETATION: the 2923 layer "
        "structure is an EVENT-SELECTION property (which cells "
        "become maxT-sig events), NOT a grid-wide polarity "
        "code: sig event cells are special bipolar contrast "
        "detectors whose sign is depth-organized, while the "
        "background grid's contrast direction is noise-level / "
        "not depth-organized. This BOUNDS 2923: layer_structured "
        "holds at the event level, does not extend to a "
        "depth-graded readout code. Also: contrast strength is "
        "depth-flat per axis (|D| medians ~1e-4 both ends) - "
        "no amplification with depth. NEXT 2925 candidates: A "
        "event-selection anatomy (what distinguishes the 13 sig "
        "cells from same-layer non-sig cells beyond margin: "
        "zero-forward discriminant analysis on grid features), "
        "B probe relativity (lang events under word-level "
        "probe, one forward), C h4 L1<->L19 reuse principal "
        "angles (zero-forward), D within-axis per-layer event "
        "replication (are sig cells' neighbors also "
        "contrast-structured? spatial extent of event cells)."),
    "source": {
        "path": "phase2924/depth_polarity_gradient/result.json",
        "sha256_8": "27f034a7",
        "phase": 2924
    }
}
L['measurements'].append(m2924)

l14 = None
for c in L['linkage']:
    if c.get('link_id') == 'L14_readout_spectrum_cross_model':
        l14 = c
        break
assert l14 is not None
assert 'M2924_depth_polarity_gradient' not in l14['connects']
l14['connects'].append('M2924_depth_polarity_gradient')
l14['notes'] += (
    " | M2924: depth-polarity gradient ABSENT at grid level "
    "(0/3 axes, median-head contrast profiles within null; size "
    "reversed) but event-level structure holds (rho(d_pole, "
    "layer) +0.654 p 0.025, |d_pole| shallower-stronger -0.495) "
    "-> 2923 layer structure is an EVENT-SELECTION property, "
    "not a grid-wide depth-graded polarity code")
l14['phase_updated'] = 2924

with open(P, 'w', encoding='utf-8') as f:
    json.dump(L, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
out = ('ledger updated: measurements %d, L14 connects %d, '
       'file sha256_8 %s'
       % (len(L['measurements']), len(l14['connects']), h))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\ledger_2924_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(out + '\n')
print('OK ledger 2924')
