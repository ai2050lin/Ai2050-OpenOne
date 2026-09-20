# -*- coding: utf-8 -*-
"""ledger_2928_update.py -- register M2928 + refine L14."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    L = json.load(f)

m_ids = [m.get('meas_id') for m in L['measurements']]
assert 'M2928_survivor_core_anatomy' not in m_ids, 'already in'
assert m_ids[-1] == 'M2927_probe_relativity', m_ids[-1]

m2928 = {
    "meas_id": "M2928_survivor_core_anatomy",
    "type": "survivor_core_anatomy",
    "verdict": (
        "survivor_core_not_established (frozen mapping: P1 fail) "
        "with the REAL findings in P2/P3 - zero-forward (212 s) "
        "on the 2927 npz. ANCHORS 4/4: a1 sign_M diff 0.0; a2 "
        "sets 24/32/7 rebuilt; a3 margins (1.34634, 1.01827); "
        "a4 cos median 0.1651. P1 OVERLAP NULL REFUTES THE "
        "OVERLAP EVIDENCE: maxT on fixed G stacks with 100 "
        "independent permutation sets gives overlap null "
        "median 7.0 (mean 7.22, max 9) - the observed 7/24 is "
        "EXACTLY AT CHANCE (p=0.85); set sizes stay 24-33 and "
        "null significant sets cluster on FIXED hot layers (L6 "
        "47/10-sets, L9 31, L8 25, L19 21) shared by both "
        "calibers because the Gram structure is probe-shared - "
        "SIGNIFICANT-SET OVERLAP IS UNINFORMATIVE about "
        "circuit invariance; the 2927 '7-event invariant hard "
        "core' reading is CORRECTED (wrong selection statistic, "
        "not wrong data). P2 DUAL-STRENGTH PASS: survivor "
        "min_pct median 0.9688 vs lost 0.8125 (U=100, "
        "permutation p 0.0049) - survivors are layer-internal "
        "top cells under BOTH calibers. P3 RESPONSE-STRUCTURE "
        "INVARIANCE PASS (the mechanistic answer): survivor "
        "rho(Spearman B86[h,:,l], B_word[h,:,l] over 57 words) "
        "median 0.825 vs lost 0.377 (U=104, p 0.0025) - "
        "survivor word-response PATTERNS are probe-invariant, "
        "lost events' response vectors are probe-rewritten "
        "(L+ class events worst: (25,3) -0.64, (26,5) -0.47). "
        "(27,24) FORENSICS: dual-caliber layer-internal #1 "
        "(pct 1.000/1.000) yet lost - margin_wd 0.560 with "
        "p_maxT 0.0647 misses by 0.0147 - dual strength is "
        "necessary NOT sufficient under family-wise maxT. "
        "2918 CLASS ASYMMETRY (descriptive): en+ 2/3 survive, "
        "L+ 0/4 - the word probe (lab0-minus-lab1 direction) "
        "systematically erases L+ events. NEXT 2929 "
        "candidates: A grid-wide response-structure map "
        "(rho(B86, B_word) over all 1152 cells - rho is the "
        "correct probe-invariance statistic; threshold scan "
        "defines the true invariant set, zero-forward), B "
        "directional-bias control (rebuild dirs_word with "
        "flipped group convention or per-pole balanced pairs, "
        "one forward, tests the L+ erasure), C h4 L1<->L19 "
        "reuse principal angles (zero-forward, roadmap "
        "legacy), D min_pct/rho-based re-selection of the "
        "atlas and cross-model replication (glm4)."),
    "source": {
        "path": "phase2928/survivor_core_anatomy/result.json",
        "sha256_8": "47b23cc2",
        "phase": 2928
    }
}
L['measurements'].append(m2928)

l14 = None
for c in L['linkage']:
    if c.get('link_id') == 'L14_readout_spectrum_cross_model':
        l14 = c
        break
assert l14 is not None
assert 'M2928_survivor_core_anatomy' not in l14['connects']
l14['connects'].append('M2928_survivor_core_anatomy')
l14['notes'] += (
    " | M2928: survivor core anatomy - overlap null median 7.0 "
    "= observed: maxT significant-set overlap is UNINFORMATIVE "
    "(fixed hot layers shared via Gram structure); the correct "
    "probe-invariance statistics are dual-caliber layer strength "
    "(P2 p 0.0049) and word-response structure rho (P3 p 0.0025, "
    "survivor 0.825 vs lost 0.377); 2927 hard-core reading "
    "corrected; L+ class events systematically erased by the "
    "word probe (0/4)")
l14['phase_updated'] = 2928

with open(P, 'w', encoding='utf-8') as f:
    json.dump(L, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
out = ('ledger updated: measurements %d, L14 connects %d, '
       'file sha256_8 %s'
       % (len(L['measurements']), len(l14['connects']), h))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\ledger_2928_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(out + '\n')
print('OK ledger 2928')
