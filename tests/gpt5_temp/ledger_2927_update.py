# -*- coding: utf-8 -*-
"""ledger_2927_update.py -- register M2927 + refine L14."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    L = json.load(f)

m_ids = [m.get('meas_id') for m in L['measurements']]
assert 'M2927_probe_relativity' not in m_ids, 'already in'
assert m_ids[-1] == 'M2926_event_selection_polar_fix', m_ids[-1]

m2927 = {
    "meas_id": "M2927_probe_relativity",
    "type": "probe_relativity",
    "verdict": (
        "events_probe_partially_invariant - one forward (111 s), "
        "2917 protocol verbatim with DUAL readout directions per "
        "perturbation: dirs86 (2886 sentence caliber, anchor) "
        "and dirs_word (word-level probe built from this run's "
        "func-condition pos-1 attn-input captures, en/L group "
        "diff per layer; sign convention irrelevant - sign-Gram "
        "margin is sign-flip invariant). ANCHORS 4/4: a1 rel "
        "3.16e-08 vs 2913 (seventh consecutive forward anchor "
        "in the 2917 lineage); a2 m78 0.280360 exact; a3 "
        "sign_M diff 4.71e-08 vs 2917 npz; a4 maxT event set "
        "24/24 set-equal - the 2886 caliber of THIS run is a "
        "certified 2917 replica. P1 MAIN: word probe cos(dirs_"
        "word, dirs86) median 0.165 (replicates 2920); E_word "
        "n=32, n_overlap 7/24 (jaccard 0.143), top1 (7,19) "
        "survives => partially_invariant per frozen mapping "
        "(6 <= overlap < 12). SURVIVOR CORE (7): (1,6) (5,6) "
        "(7,19) (8,2) (14,9) (20,8) (21,6); (7,19) margin "
        "1.346->1.018 (-24%) but layer-internal rank #1 under "
        "BOTH probes - the only rank-stable event. LOST 17 "
        "(ALL 5 deep events l>=20: (4,22) (13,22) (17,28) "
        "(24,23) (27,24)); NEW 25 with 21/25 at l<=10 - the "
        "word probe shifts the atlas toward EARLY layers and "
        "erases deep events. P2 full-grid Spearman(sign_M_word,"
        " sign_M86) = 0.176 and E17-internal rank corr 0.184 - "
        "probe change re-orders the grid, not just the "
        "significance set. INTERPRETATION: the event atlas is "
        "a probe x circuitry INTERACTION product, not a pure "
        "circuit property: sentence dirs (aggregated semantics) "
        "and word dirs (lexical identity) read out different "
        "aspects; a 7-event probe-invariant hard core exists "
        "and (7,19) is its anchor. The 2925/2926 selection "
        "mechanism (polar-contrast strength extremum) holds "
        "PER PROBE - 'which cell is strongest' is probe-"
        "dependent. NEXT 2928 candidates: A survivor-core "
        "anatomy (what makes the 7 events probe-invariant - "
        "word-response correlation, polarity, density, "
        "zero-forward on this npz), B deep-vs-early event "
        "profile under a third probe family (e.g. shuffled-"
        "label dirs or per-layer PCA-1, one forward), C h4 "
        "L1<->L19 reuse principal angles (zero-forward, "
        "roadmap legacy), D dirs_word-based attribute atlas "
        "(2921 protocol with word probe, one forward)."),
    "source": {
        "path": "phase2927/probe_relativity/result.json",
        "sha256_8": "4302c248",
        "phase": 2927
    }
}
L['measurements'].append(m2927)

l14 = None
for c in L['linkage']:
    if c.get('link_id') == 'L14_readout_spectrum_cross_model':
        l14 = c
        break
assert l14 is not None
assert 'M2927_probe_relativity' not in l14['connects']
l14['connects'].append('M2927_probe_relativity')
l14['notes'] += (
    " | M2927: probe relativity - word-level probe shifts the "
    "event atlas (overlap 7/24, jaccard 0.143; all 5 deep events "
    "lost, 21/25 new events early-layer; full-grid rank corr "
    "0.176) with a 7-event probe-invariant hard core anchored by "
    "(7,19) (layer-rank #1 under both probes) - atlas is a "
    "probe x circuitry interaction product; anchors 4/4 with "
    "this run's 2886 caliber exactly replicating 2917")
l14['phase_updated'] = 2927

with open(P, 'w', encoding='utf-8') as f:
    json.dump(L, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
out = ('ledger updated: measurements %d, L14 connects %d, '
       'file sha256_8 %s'
       % (len(L['measurements']), len(l14['connects']), h))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\ledger_2927_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(out + '\n')
print('OK ledger 2927')
