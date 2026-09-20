# -*- coding: utf-8 -*-
"""ledger_2922_update.py -- register M2922 + refine L14."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    L = json.load(f)

m_ids = [m.get('meas_id') for m in L['measurements']]
assert 'M2922_attr_event_anatomy' not in m_ids, 'already in'
assert m_ids[-1] == 'M2921_attr_vocab_expansion', m_ids[-1]

m2922 = {
    "meas_id": "M2922_attr_event_anatomy",
    "type": "attr_event_anatomy",
    "verdict": (
        "attr_events_linked_polar - zero-forward artifact-domain "
        "anatomy (1.1 s) of the 13 attribute events from 2921 "
        "(speed 2 / size 9 / moist 2) using the 2918 protocol "
        "verbatim: word-response vectors B_ax[h,:,l], within-axis "
        "Spearman linkage vs layer-distance-matched non-sig nulls "
        "(maxT over the merged 38-pair family), pole-alignment "
        "permutation test, sparsity, temporal curves; 2917 npz "
        "for the lang comparison. ANCHORS 3/3: a1 sign_M "
        "recomputed from npz B per axis - median absdiff ~6e-10, "
        "max 0, zero cells > 0.02; a2 registry sig sizes 2/9/2 + "
        "lang 24 + per-axis top1 margins match 2921 P1 within "
        "1e-4; a3 (7,19) argmax 1.346335. POLE ALIGNMENT: 13/13 "
        "events p_pole = 0.0005 (floor) - every attribute event "
        "splits words by polarity; d_pole |0.67-1.84| median "
        "1.086, and 8/13 are LOW-driven (d_pole < 0: lo words "
        "respond higher, incl the size top event (21,7) d=-1.84 "
        "and both moist events) vs 5 HIGH-driven - events are "
        "polarity-structured but polarity SIGN is event-specific. "
        "LINKAGE: n_linked_all 4/38, ALL in size: (21,7)-(18,7) "
        "rho 0.676 p 0.0295, (21,7)-(25,15) 0.571 p 0.0345, "
        "(11,27)-(18,7) 0.571 p 0.0345, (24,19)-(12,18) 0.572 "
        "p 0.0345; size components: {(21,7),(25,15),(18,7),"
        "(11,27)} 4-event component + {(24,19),(12,18)} pair + 3 "
        "singletons; speed and moist event pairs do NOT link "
        "(both rho far below null95) - size is the only "
        "internally coherent attr axis. LAYER SHIFT: attr "
        "median peak layer 15 vs lang 6 - attribute events "
        "concentrate deeper. SPARSITY: attr PR_word median 24.7 "
        "vs lang 30.6, top3 mass 0.207 vs 0.196, pr_pct 0.484 "
        "vs 0.582 - same broad-response regime, slightly more "
        "concentrated. SAME-HEAD CROSS-CATEGORY: 8 heads carry "
        "events from >1 category (h21 lang(21,6)/(21,16) + "
        "size(21,7); h18 speed(18,16) + size(18,7); h26 lang "
        "x2 + size(26,21); h8/h14/h22/h24/h25) - heads are "
        "shared across categories while CELLS are private "
        "(2921 P2 shared=0): category privacy is layer-level "
        "within shared heads. NEXT 2923 candidates: A "
        "polarity-sign structure (what determines HIGH vs LOW "
        "driving; hi/lo response asymmetry vs 2919 dirs "
        "convention), B probe relativity (lang events under "
        "word-level probe), C h4 L1<->L19 reuse principal "
        "angles (zero-forward), D attr linkage power expansion "
        "(linkage nulls for n=43-48 vocab at 2000 draws; "
        "confirm the 4 size edges)."),
    "source": {
        "path": "phase2922/attr_event_anatomy/result.json",
        "sha256_8": "629f94c7",
        "phase": 2922
    }
}
L['measurements'].append(m2922)

l14 = None
for c in L['linkage']:
    if c.get('link_id') == 'L14_readout_spectrum_cross_model':
        l14 = c
        break
assert l14 is not None
assert 'M2922_attr_event_anatomy' not in l14['connects']
l14['connects'].append('M2922_attr_event_anatomy')
l14['notes'] += (
    " | M2922: attr event anatomy (zero-forward, 2918 protocol) - "
    "13/13 attribute events are polarity-aligned (p=0.0005 floor, "
    "d_pole |0.67-1.84|, 8/13 LOW-driven); size is the only "
    "internally linked attr axis (4/36 edges, one 4-event "
    "component); attr events sit deeper (median peak L15 vs lang "
    "L6) with similar broad-response sparsity; 8 heads carry "
    "cross-category events (head shared, cell private)")
l14['phase_updated'] = 2922

with open(P, 'w', encoding='utf-8') as f:
    json.dump(L, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
out = ('ledger updated: measurements %d, L14 connects %d, '
       'file sha256_8 %s'
       % (len(L['measurements']), len(l14['connects']), h))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\ledger_2922_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(out + '\n')
print('OK ledger 2922')
