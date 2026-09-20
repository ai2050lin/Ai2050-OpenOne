# -*- coding: utf-8 -*-
"""ledger_2918_update.py -- register M2918 + refine L14."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    L = json.load(f)

m_ids = [m.get('meas_id') for m in L['measurements']]
assert 'M2918_event_anatomy' not in m_ids, 'already in'
assert m_ids[-1] == 'M2917_event_atlas', m_ids[-1]

m2918 = {
    "meas_id": "M2918_event_anatomy",
    "type": "event_anatomy",
    "verdict": (
        "early_events_same_mechanism - zero-forward (2.3 s) "
        "anatomy of the 24 maxT-significant (head,layer) events "
        "on the 2917 npz. Anchors: a1 sign_M recompute bit-level "
        "(median diff 8.56e-10, max 4.71e-08, 0 cells > 0.02), "
        "a2 (7,19) argmax margin 1.346335 p=0.004975, registry "
        "24 ok. FOCUS = top-5 novel (26,6),(8,2),(25,3),(22,12),"
        "(24,23): ALL 5 mechanism-linked to the late survivors "
        "at the frozen 10-slot maxT family (p_link 0.0005 / "
        "0.0180 / 0.0035 / 0.0085 / 0.0085); |rho| to (7,19)/"
        "(27,24) = 0.42-0.72 (mean 0.542); rho within early set "
        "mean 0.612. KEY FINDING - POLARITY STRUCTURE: all 24 "
        "events share ONE en-vs-nonen word-response pattern in "
        "two polarity classes - en+ {(8,2),(22,12),(7,19)}, L+ "
        "{(26,6),(25,3),(24,23),(27,24)}; same-polarity pairs "
        "correlate +0.45..+0.72, opposite-polarity -0.42..-0.60; "
        "polarity ALTERNATES with depth (L2 en+ -> L3 L+ -> L6 "
        "L+ -> L12 en+ -> L19 en+ -> L23 L+ -> L24 L+). GATING "
        "DENSITY (P1/P3): word level DENSE - strong events "
        "PR_word 37.6-39.7/57 at 79-97th percentile of "
        "same-layer null (NOT few-word gates; top-3 words carry "
        "only 14-31% of |r| mass), weak events sparser ((24,23) "
        "PR 20.9, 6.5th pct); layer level SHARP - temporal PR "
        "6.3-10.3 of 36, adjacent contrast 0.76-1.15; early "
        "event responses 5-10x smaller in magnitude than (7,19) "
        "yet sign-Gram detects them (alignment not magnitude). "
        "HEAD REUSE (P4): within-head cross-layer patterns "
        "mostly DECORRELATED or ANTI-correlated (7/8 below "
        "null95: h26 L5 vs L6 -0.620 adjacent-layer polarity "
        "flip, h4 L19 vs L22 -0.479, h21 L6 vs L16 -0.345, h7 "
        "L19 vs L34 -0.377); only h4 L1<->L19 +0.447 above "
        "null95 - genuine head-channel reuse across depth. "
        "CAVEAT (registered): the 24x24 linkage graph at the "
        "harsher 276-pair maxT family has ZERO edges - linkage "
        "clears the 10-slot family but not the 276-family; "
        "cross-cell baseline correlation is high (per-distance "
        "single-pair null95 0.24-0.27), so event linkage is a "
        "MODERATE effect detectable in pre-frozen small "
        "families. READING: the language axis is re-expressed "
        "along depth as a chain of (head,layer) events sharing "
        "one bimodal word pattern with alternating polarity - "
        "a polarity-alternating carrier chain, neither "
        "independent mechanisms nor a single static carrier."),
    "source": {
        "path": "phase2918/event_anatomy/result.json",
        "sha256_8": "d57378eb",
        "phase": 2918
    }
}
L['measurements'].append(m2918)

l14 = None
for c in L['linkage']:
    if c.get('link_id') == 'L14_readout_spectrum_cross_model':
        l14 = c
        break
assert l14 is not None
assert 'M2918_event_anatomy' not in l14['connects']
l14['connects'].append('M2918_event_anatomy')
l14['notes'] += (
    " | M2918: event anatomy - all 5 top novel early events "
    "mechanism-linked to (7,19)/(27,24) at 10-slot maxT (p "
    "0.0005-0.018); ONE en-vs-L word pattern in two polarity "
    "classes alternating with depth (en+ {8,2; 22,12; 7,19}, L+ "
    "{26,6; 25,3; 24,23; 27,24}); word-level DENSE (PR 38-40/57 "
    "for strong events, top-3 mass only 14-31%), layer-level "
    "SHARP (temporal PR 6-10/36), within-head cross-layer "
    "patterns decorrelate or flip (only h4 L1<->L19 reuses "
    "above null95) - language axis = polarity-alternating "
    "event chain, not a static carrier")
l14['phase_updated'] = 2918

with open(P, 'w', encoding='utf-8') as f:
    json.dump(L, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
out = ('ledger updated: measurements %d, L14 connects %d, '
       'file sha256_8 %s'
       % (len(L['measurements']), len(l14['connects']), h))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\ledger_2918_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(out + '\n')
print('OK ledger 2918')
