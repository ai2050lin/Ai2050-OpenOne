# -*- coding: utf-8 -*-
"""ledger_2925_update.py -- register M2925 + refine L14."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    L = json.load(f)

m_ids = [m.get('meas_id') for m in L['measurements']]
assert 'M2925_event_selection_anatomy' not in m_ids, 'already in'
assert m_ids[-1] == 'M2924_depth_polarity_gradient', m_ids[-1]

m2925 = {
    "meas_id": "M2925_event_selection_anatomy",
    "type": "event_selection_anatomy",
    "verdict": (
        "events_margin_only (frozen verdict) with a registered "
        "phi implementation defect and a decisive post-hoc |phi| "
        "diagnostic - zero-forward feature discrimination (1.0 "
        "s): 13 sig cells vs same-layer non-sig heads on 7 "
        "non-trivial features + margin reference. ANCHORS 3/3: "
        "a1 ~6e-10; a2 d_pole13 bit-exact 0.0; a3 2924 rho_axis "
        "[size] -0.060746. FROZEN RESULT: only 2/7 features BH-"
        "significant - d_pole_abs median pct 1.0 (13/13 > 0.5, "
        "sign p 1.22e-4, BH q 4.27e-4) and contrast_raw median "
        "0.906 (13/13, same p/q); mean_abs_r 0.75 ns; pr_word "
        "0.50, top3_mass 0.32, neighbor_phi 0.50 ns - response "
        "shape (width/magnitude) and spatial extent do NOT "
        "select events; n_BH 2 < 3 => margin_only per the "
        "frozen mapping. REGISTERED DEFECT: the preregistered "
        "phi test used SIGNED phi with a one-sided percentile - "
        "LOW-driven cells have the most EXTREME NEGATIVE phi "
        "and mathematically cannot reach 'median pct >= 0.9' "
        "(detail shows the bimodal signature: 8 cells pct ~1.0, "
        "5 cells pct ~0.03, both extremes). POST-HOC |phi| "
        "DIAGNOSTIC (descriptive, not the frozen verdict): "
        "|phi| percentile = 1.0 for 13/13 events (sign p "
        "1.22e-4) - EVERY sig cell is the strongest pole-sign "
        "association in its entire layer; together with "
        "d_pole_abs pct 1.0 (13/13) and contrast_raw 13/13, "
        "events are selected by POLAR CONTRAST STRENGTH, full "
        "stop - the margin criterion is essentially the same "
        "quantity, and response shape plays no role. NEXT 2926 "
        "candidates: A |phi| preregistered retest (corrected "
        "feature definition, expected flip to "
        "events_polar_separation_selected; clean prereg->"
        "defect->retest discipline loop), B probe relativity "
        "(lang events under word-level probe, one forward), C "
        "h4 L1<->L19 reuse principal angles (zero-forward), D "
        "contrast ceiling (are non-sig cells' |phi| uniformly "
        "weak or is there a gap - layer-wise |phi| distributions)."),
    "source": {
        "path": "phase2925/event_selection_anatomy/result.json",
        "sha256_8": "7846d213",
        "phase": 2925
    }
}
L['measurements'].append(m2925)

l14 = None
for c in L['linkage']:
    if c.get('link_id') == 'L14_readout_spectrum_cross_model':
        l14 = c
        break
assert l14 is not None
assert 'M2925_event_selection_anatomy' not in l14['connects']
l14['connects'].append('M2925_event_selection_anatomy')
l14['notes'] += (
    " | M2925: event selection anatomy - frozen verdict "
    "events_margin_only (2/7 features BH) but registered phi "
    "sign defect; post-hoc |phi|: 13/13 sig cells are the "
    "strongest pole-sign association of their layer (pct all "
    "1.0, p 1.22e-4), with d_pole_abs pct 13/13 = 1.0 - events "
    "are selected by polar contrast strength; response shape "
    "(PR/top3/magnitude) and spatial extent play no role")
l14['phase_updated'] = 2925

with open(P, 'w', encoding='utf-8') as f:
    json.dump(L, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
out = ('ledger updated: measurements %d, L14 connects %d, '
       'file sha256_8 %s'
       % (len(L['measurements']), len(l14['connects']), h))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\ledger_2925_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(out + '\n')
print('OK ledger 2925')
