# -*- coding: utf-8 -*-
"""ledger_2920_update.py -- register M2920 + refine L14."""
import hashlib
import json

P = (r'D:\AI2050\Ai2050-OpenOne\research\gpt5\atlas'
     r'\atlas_ledger.json')
with open(P, encoding='utf-8') as f:
    L = json.load(f)

m_ids = [m.get('meas_id') for m in L['measurements']]
assert 'M2920_multiaxis_word_atlas' not in m_ids, 'already in'
assert m_ids[-1] == 'M2919_multiaxis_families', m_ids[-1]

m2920 = {
    "meas_id": "M2920_multiaxis_word_atlas",
    "type": "multiaxis_word_atlas",
    "verdict": (
        "nonlang_events_absent - forward (55.1 s, qwen3-4b, 110 "
        "single-token words = 57 lang verbatim 2887 + 53 "
        "tokenizer-screened attribute adjectives) extended the "
        "2917 event-atlas protocol to 5 axis partitions and 4 "
        "jacobian families (lang/concept share the 2886 lang-dir "
        "probe; speed/size/moist inject 2919 dirs_all[1..3]). "
        "ANCHORS 4/4 bit-exact class: a0 dirs2919[lang] vs 2886 "
        "derived rel 2.86e-08; a1 B_lang vs 2917 npz rel 3.53e-09; "
        "a2 m78 0.280360 exact; a3 sign_M diff 4.71e-08 + sig-set "
        "equality 24/24 - the lang atlas reproduced EXACTLY. "
        "RESULT: lang 24 events (top (7,19) 1.34634 identical), "
        "concept 0, speed 0, size 0, moist 0 under per-axis maxT "
        "(q=0.05); cross-axis co-occurrence EMPTY (only lang "
        "nonempty). Clean dissociation at full power (n=57, same "
        "probe, same forward): en/L partition -> 24 events, "
        "concept partition -> 0 (top margin 0.487 BELOW its null "
        "median 0.496, 124/200 perms >= top) - the (head,layer) "
        "event response to the lang probe encodes LANGUAGE "
        "IDENTITY, not concept content. Attribute-axis nulls are "
        "SMALL-n POWER-LIMITED, not evidence of absence: margins "
        "quantize (size top-40 = 9 distinct values) and permuted "
        "nulls routinely reach observed tops (speed null p50 "
        "0.971 == obs top; moist 100/200); size closest "
        "((18,7) 1.5312, p_maxT 0.0796, 15/200). POST-HOC "
        "power diagnostic registered descriptive. P3 transfer "
        "ALL FAILED (median |cos| L1-35: lang 0.166, speed 0.237, "
        "size 0.222, moist 0.084; peaks 0.28-0.37) yet 2913-2918 "
        "proved sentence dirs causally effective at word "
        "positions => FUNCTIONAL TRANSFER != GEOMETRIC IDENTITY: "
        "the injection direction is a probe, not the word-level "
        "encoding direction; 2919 sentence dirs must not be "
        "assumed to be word-level attribute encoding directions. "
        "METHODOLOGY: sign-Gram maxT requires n >~ 40 at word "
        "level (small-n granularity failure parallels the 2917 "
        "p-granularity lesson). NEXT: 2921 candidates - attr "
        "vocab expansion to n~40-60 (main), probe-relativity "
        "(lang events under a word-level probe), lang-internal "
        "reuse principal angles (roadmap 2921)."),
    "source": {
        "path": "phase2920/multiaxis_word_atlas/result.json",
        "sha256_8": "bf96339b",
        "phase": 2920
    }
}
L['measurements'].append(m2920)

l14 = None
for c in L['linkage']:
    if c.get('link_id') == 'L14_readout_spectrum_cross_model':
        l14 = c
        break
assert l14 is not None
assert 'M2920_multiaxis_word_atlas' not in l14['connects']
l14['connects'].append('M2920_multiaxis_word_atlas')
l14['notes'] += (
    " | M2920: multi-axis word atlas NEGATIVE for non-lang axes - "
    "concept partition 0 events at full power (same 57 words, "
    "same lang-dir probe as the 24-event lang atlas: events "
    "encode language identity, not concept content); "
    "speed/size/moist 0 under maxT but small-n granularity "
    "power-limited (size p 0.0796 closest); word-vs-sentence "
    "direction transfer |cos| ~0.1-0.3 everywhere => functional "
    "transfer != geometric identity; word-level event tests need "
    "n >~ 40")
l14['phase_updated'] = 2920

with open(P, 'w', encoding='utf-8') as f:
    json.dump(L, f, indent=1, ensure_ascii=False)

h = hashlib.sha256(open(P, 'rb').read()).hexdigest()[:8]
out = ('ledger updated: measurements %d, L14 connects %d, '
       'file sha256_8 %s'
       % (len(L['measurements']), len(l14['connects']), h))
with open(r'D:\AI2050\Ai2050-OpenOne\tests\gpt5_temp'
          r'\ledger_2920_report.txt', 'w',
          encoding='utf-8') as f:
    f.write(out + '\n')
print('OK ledger 2920')
