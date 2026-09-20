# -*- coding: utf-8 -*-
"""Phase 2930 ledger update: append M2930 to measurements and
L14_readout_spectrum_cross_model.connects (36 -> 37).
"""
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
REPORT = (r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\phase2930_ledger_update_report.txt')

M2930 = {
    "meas_id": "M2930_direction_flip_control",
    "type": "direction_flip_control",
    "verdict": (
        "direction_flip_margin_shifts - one forward (57 s), "
        "2917/2927 protocol verbatim with THREE pass-2 "
        "directions (+dirs86 anchor, +dirs_word repro, "
        "-dirs_word mirror). ANCHORS 8/8: a1 rel 3.16e-08 "
        "(8th consecutive forward anchoring); a2 m78 "
        "0.280360; a3 4.71e-08; a4 24/24 set equality; a5 "
        "dirs_word rebuild diff 2.17e-08; a6 Bwd repro rel "
        "2.10e-08; a7 Ewd==2927 + pdiff 2.76e-08; a8 "
        "dirs_neg==-dirs_word and G_neg==-Gwd bit-exact. "
        "P1 MAIN: E_mirror n=36 vs Ewd27 n=32, overlap 20/48 "
        "jaccard 0.417, sign_M diff max 1.29 - the maxT event "
        "selection is DIRECTION-CONVENTION RELATIVE; top1 "
        "(7,19) survives; 12 lost (incl survivors (8,2) "
        "p 0.045->0.204 and (20,8) 0.025->0.090 boundary) + "
        "16 new (incl L+ class (25,3) and (27,24) ENTERING). "
        "P2 QUADRATIC-FORM BREAK QUANTIFIED: r-level "
        "||r(-d)+r(d)||/||r(d)|| median 1.1747 (even-order "
        "term same magnitude as linear term at eps=1.0; "
        "L4-L13 quasi-linear 0.4-0.7 vs L30-L35 1.6-1.76), "
        "B-level mir_err median 1.0192; corr(lin_r_layer, "
        "mir_err_layer)=0.9827 closes the mechanism chain "
        "even-order nonlinearity -> quadratic-form break -> "
        "margin rewrite. P3 skeleton partially "
        "convention-invariant: skel_mirror 414 vs 501, "
        "intersection 327, jaccard 0.556; L0 degenerate-tie "
        "artifact replicated (all 32, rho max 0.0). P4 "
        "SURVIVOR-CORE RHO IS CONVENTION-INVARIANT: all 7 "
        "survivors keep high positive rho under the mirror "
        "probe (|delta| median 0.0154 max 0.0703; 6/7 > "
        "0.74, (1,6) 0.699 edge) while 2/7 lose maxT "
        "significance - rho (response structure) is the "
        "stable object, maxT the fragile selector; L+ deep-"
        "negative rho (25,3) -0.64->-0.63 and (26,5) "
        "-0.47->-0.50 preserved across convention (real "
        "anti-phase structure, not artifact). 2928 L+ 0/4 "
        "wipe-out is PARTIALLY convention-relative at the "
        "maxT level (2/4 flip in). METHODOLOGICAL: all "
        "injection-response interpretations in the 2917-2930 "
        "protocol family carry an O(1) even-order nonlinear "
        "mixture at eps=1.0; B is 'quadratic form + H[d,d] "
        "correction', and direction-sign convention is a "
        "free parameter that maxT-level claims must control "
        "for."),
    "source": {
        "path": "phase2930/direction_flip_control/"
                "direction_flip_control.npz",
        "sha256_8": "cb655825",
        "phase": 2930,
    },
}

d = json.load(open(P, encoding='utf-8'))
meas = d['measurements']
link = None
for it in d['linkage']:
    if it['link_id'] == 'L14_readout_spectrum_cross_model':
        link = it
assert len(meas) == 68, 'unexpected meas len %d' % len(meas)
assert meas[-1]['meas_id'] == 'M2929_response_structure_atlas'
assert link is not None and len(link['connects']) == 36
assert link['connects'][-1] == 'M2929_response_structure_atlas'

meas.append(M2930)
link['connects'].append('M2930_direction_flip_control')
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
