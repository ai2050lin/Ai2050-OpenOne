# -*- coding: utf-8 -*-
"""ledger_2921_update.py -- register M2921 + refine L14."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    L = json.load(f)

m_ids = [m.get('meas_id') for m in L['measurements']]
assert 'M2921_attr_vocab_expansion' not in m_ids, 'already in'
assert m_ids[-1] == 'M2920_multiaxis_word_atlas', m_ids[-1]

m2921 = {
    "meas_id": "M2921_attr_vocab_expansion",
    "type": "attr_vocab_expansion",
    "verdict": (
        "attribute_events_found - forward (79.8 s, qwen3-4b, 183 "
        "single-token words = 57 lang verbatim 2887 + speed 43 "
        "(21 hi / 22 lo) + size 48 (25 hi / 23 lo) + moist 35 "
        "(19 hi / 16 lo), tokenizer-screened in 2 probe rounds) "
        "retested the 2920 negative result after expanding "
        "attribute pools to lang-scale power, same frozen "
        "sign-Gram maxT statistic (2917 verbatim) and same 4 "
        "jacobian families (concept axis closed in 2920, removed). "
        "ANCHORS 4/4 bit-exact class: a0 dirs rel 2.86e-08; a1 "
        "B_lang vs 2917 npz rel 3.53e-09; a2 m78 0.280360 exact; "
        "a3 sign_M diff 4.71e-08 + sig-set equality 24/24 - the "
        "lang atlas reproduced EXACTLY (6th consecutive forward "
        "anchor). RESULT: lang 24 (top (7,19) 1.34634 identical), "
        "speed 2 ((18,16) 1.01608, (14,12) 0.76479), size 9 (top "
        "(21,7) 1.23834, (24,19) 0.98181, (12,18) 0.86648, ...), "
        "moist 2 ((8,15) 0.99258, (10,9) 0.9917) under per-axis "
        "maxT (q=0.05). GRANULARITY PRECHECK (preregistered, "
        "discipline 8): top-40 distinct margin values lang 30 / "
        "speed 26 / size 27 / moist 23, all >= 20 => all nulls "
        "powered => the 2920 zeros were a SMALL-n GRANULARITY "
        "ARTIFACT, not absence; the 2920->2921 flip (0 -> 13 "
        "events) directly validates discipline 8. Cross-phase "
        "check: the 2920-legacy size cell (18,7) margin 1.53125 "
        "(small pool) -> 0.65363 (expanded) yet p_maxT 0.024876 "
        "sig - survived pool expansion. P2 co-occurrence: "
        "overlap matrix DIAGONAL ONLY, shared pairs = 0 - attr "
        "events share ZERO channels with the 24 lang events "
        "(per-category private paths). P3 transfer recheck with "
        "expanded pools 4/4 FAILED (median |cos| L1-35: lang "
        "0.1664 / speed 0.2025 / size 0.2193 / moist 0.0943) - "
        "functional transfer != geometric identity reconfirmed. "
        "NEXT 2922 candidates: A attr-event anatomy via the 2918 "
        "protocol (word-position decomposition / polarity / "
        "density / curves, zero-forward), B probe relativity "
        "(lang events under a word-level probe), C h4 L1<->L19 "
        "reuse principal angles (zero-forward)."),
    "source": {
        "path": "phase2921/attr_vocab_expansion/result.json",
        "sha256_8": "e23cd5e9",
        "phase": 2921
    }
}
L['measurements'].append(m2921)

l14 = None
for c in L['linkage']:
    if c.get('link_id') == 'L14_readout_spectrum_cross_model':
        l14 = c
        break
assert l14 is not None
assert 'M2921_attr_vocab_expansion' not in l14['connects']
l14['connects'].append('M2921_attr_vocab_expansion')
l14['notes'] += (
    " | M2921: attr vocab expansion to 183 words FLIPS the 2920 "
    "verdict - attribute_events_found: speed 2 / size 9 / moist 2 "
    "(head,layer) events under the same frozen statistic, "
    "granularity precheck passed (top-40 distinct 23-30 >= 20) so "
    "the 2920 zeros were a small-n granularity artifact; "
    "co-occurrence still empty across axes (0 shared cells with "
    "the 24 lang events -> per-category private paths); word-level "
    "direction transfer still 4/4 failed (median |cos| 0.09-0.22)")
l14['phase_updated'] = 2921

with open(P, 'w', encoding='utf-8') as f:
    json.dump(L, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
out = ('ledger updated: measurements %d, L14 connects %d, '
       'file sha256_8 %s'
       % (len(L['measurements']), len(l14['connects']), h))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\ledger_2921_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(out + '\n')
print('OK ledger 2921')
