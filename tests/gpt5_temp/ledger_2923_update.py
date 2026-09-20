# -*- coding: utf-8 -*-
"""ledger_2923_update.py -- register M2923 + refine L14."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    L = json.load(f)

m_ids = [m.get('meas_id') for m in L['measurements']]
assert 'M2923_polarity_sign_anatomy' not in m_ids, 'already in'
assert m_ids[-1] == 'M2922_attr_event_anatomy', m_ids[-1]

m2923 = {
    "meas_id": "M2923_polarity_sign_anatomy",
    "type": "polarity_sign_anatomy",
    "verdict": (
        "polarity_sign_layer_structured - zero-forward polarity "
        "sign anatomy (1.0 s) of the 13 attribute events: is the "
        "LOW/HIGH driving sign (d_pole < 0 vs > 0, 2922) "
        "predictable from top-word polarity composition, layer "
        "position, or event-specific? ANCHORS 3/3: a1 sign_M "
        "recompute ~6e-10; a2 d_pole recompute matches 2922 P1 "
        "13/13 within 1e-3; a3 2922 npz edge (21,7)-(18,7) rho "
        "0.6764 exact + sign_M_ref equality. P1 MAIN TEST "
        "NEGATIVE (preregistered criterion p_perm <= 0.05 AND "
        "T_obs >= 10 at k=5): top-|r| word polarity composition "
        "f5 agrees with the d_pole sign in only 7/13 events "
        "(p_perm 0.4818, null p50 6 max 11; f10 5/13 p 0.6012) "
        "- and several events are sign-REVERSED (e.g. (22,2) "
        "d=-1.20 with f5=+1.0 all-HIGH top5; (8,15) d=-1.60 "
        "with f5=+0.6; (14,12) d=-0.87 with f5=+0.6): the "
        "strongest-responding words' polarity composition does "
        "NOT determine the driving sign. P2 LAYER STRUCTURE "
        "(quasi-post-hoc, descriptive weight): Fisher exact "
        "{low_shallow 7, low_deep 1, high_shallow 0, high_deep "
        "5} p 0.0047, point-biserial r 0.623 - LOW-driven "
        "events concentrate shallow (7/8), HIGH-driven ALL "
        "deep (5/5, peaks 16-23). P3: 13/13 events are "
        "CONTRAST type (m_hi and m_lo have OPPOSITE signs, "
        "zero magnitude-type) - every attribute event cell is "
        "a bipolar contrast detector between pole means, not a "
        "one-sided detector. P4: h11 carries opposite-sign "
        "events ((11,23) +0.84 / (11,27) -0.79, same peak "
        "layer 23 - within-head sign flips exist), h18 "
        "cross-category (+1.36 speed / -1.44 size). P5: the 4 "
        "significant size linkage edges are 4/4 sign-matched "
        "(binom p 0.0625, quasi-post-hoc) - linked events "
        "share the driving sign. INTERPRETATION: attribute "
        "event cells implement signed pole-contrast readouts "
        "whose direction is organized by LAYER DEPTH (shallow "
        "= LOW-driven, deep = HIGH-driven), not by which words "
        "respond strongest - consistent with a depth-"
        "progressing polarity code; the 2919 dirs convention "
        "(HIGH - LOW) is anchor-side, the cell-side readout "
        "sign is layer-dependent. NEXT 2924 candidates: A "
        "depth-graded polarity code formalization (d_pole vs "
        "peak layer regression + per-layer sign census, "
        "zero-forward), B probe relativity (lang events under "
        "word-level probe, one forward), C h4 L1<->L19 reuse "
        "principal angles (zero-forward), D contrast-detector "
        "asymmetry (|m_hi| vs |m_lo| ratio structure)."),
    "source": {
        "path": "phase2923/polarity_sign_anatomy/result.json",
        "sha256_8": "87f9b769",
        "phase": 2923
    }
}
L['measurements'].append(m2923)

l14 = None
for c in L['linkage']:
    if c.get('link_id') == 'L14_readout_spectrum_cross_model':
        l14 = c
        break
assert l14 is not None
assert 'M2923_polarity_sign_anatomy' not in l14['connects']
l14['connects'].append('M2923_polarity_sign_anatomy')
l14['notes'] += (
    " | M2923: polarity sign anatomy - driving sign NOT "
    "predicted by top-word composition (7/13, p 0.48) but "
    "LAYER-STRUCTURED (LOW-driven shallow 7/8, HIGH-driven "
    "deep 5/5, Fisher p 0.0047); 13/13 events are bipolar "
    "contrast detectors (m_hi/m_lo opposite signs); the 4 "
    "size linkage edges are sign-matched 4/4")
l14['phase_updated'] = 2923

with open(P, 'w', encoding='utf-8') as f:
    json.dump(L, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
out = ('ledger updated: measurements %d, L14 connects %d, '
       'file sha256_8 %s'
       % (len(L['measurements']), len(l14['connects']), h))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\ledger_2923_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(out + '\n')
print('OK ledger 2923')
