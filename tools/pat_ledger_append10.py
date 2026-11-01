# -*- coding: utf-8 -*-
"""追加 Q4-R 一条测量（首个规则型正面结果）。幂等。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

ENTRIES = [
    {
        'phase': 'Q4-R', 'name': 'pat_q4r', 'line': 'G',
        'date': '2026-11-01',
        'verdict': 'query_to_slot_routing_is_fully_predictable_by_a_rule_that_names_no_answer_word',
        'res_sha8': '0f295b75',
        'note': (
            'First prediction-type Phase under PSF-2 §9.3, and the first non-NULL verdict in twelve '
            'rounds. The question is no longer "where is the role stored" (localization) but "which rule '
            'predicts the answer". HARD CONSTRAINT: the rule may not name the answer word, otherwise '
            'accuracy is trivially 1.0 and the measurement reflects my Chinese, not the model. All three '
            'rules therefore output only a SLOT; the word in that slot comes from the sentence. '
            'R1 = argmax over the three entity slots of last-position attention mass (full s=q.k, RoPE '
            'applied, computed in-house). R2 = same but all three entity keys replaced by their '
            'context-free static codes kbar(w). R3 = the last entity slot of the sentence, ignoring the '
            'query entirely - the real alternative hypothesis. '
            'FIVE ANTI-CHEAT RULES: (1) no answer word in any rule; (2) layer choice by 5-fold split over '
            'triples, l* selected on train folds only; (3) kbar comes from an INDEPENDENT fixture (324 '
            'forward passes) that shares no prefix with the scored cells; (4) attention computed in-house '
            'rather than from output_attentions (bf16 storage, 4e-3, measured in T1); (5) full per-layer '
            'profile reported, not a single point. '
            'DESIGN ERROR CAUGHT IN SMOKE: only 1 of 12 person names is a single token ("大卫"); the rest '
            'are two (张/三). The original "slot = one position" assumption was false and was replaced by '
            'per-span handling, with static codes stored per position within the span. Without the SMOKE '
            'catch the full run would have been silently wrong. '
            'RESULT, 144 cells (24 triples x 2 voices x 3 queries), behaviour gate 140/144. '
            'R3 position rule 0.3333 (exactly one third - it is correct only for the th query). '
            'R1 held-out 1.0000, 144/144, and all five folds independently select l*=28. '
            'R2 static-code 1.0000, no degradation. Permutation consistency 48/48 = 1.000 (within a '
            'sentence the three queries get three distinct slots matching the true query-to-slot map). '
            'McNemar R1 vs R3: b=96, c=0, p=2.5e-29. D23 power: n=144, 95% half-width ~0.000. '
            'THREE LEAKAGE DIAGNOSTICS, all passed: (a) rule scored against the MODEL ACTUAL OUTPUT gives '
            '0.9722, exactly the model own behaviour gate 0.9722 - on the 4 cells the model gets wrong the '
            'rule still points at the correct slot, so the rule is not echoing the output; (b) margin at '
            'l*: median 0.816, min 0.251, so the decisions are decisive rather than threshold-luck; '
            '(c) R3 at 0.333 rules out a positional bias. '
            'PER-LAYER PROFILE (R1, all 36 layers): L0-L20 in 0.312-0.354, statistically indistinguishable from '
            'R3 0.333 at n=144, i.e. the low-layer argmax degenerates to position selection and the query does '
            'nothing yet; L21 0.590, L22 0.653, L23 0.972 - the routing decision is made in these three '
            'layers; L23-L34 in 0.875-1.000 with noise, l*=28 is the perfect point; L35 COLLAPSES to 0.562 '
            'as the last layer redistributes attention away from the answer, so the rule holds on L23-L34 '
            'and must not be extrapolated to L35. '
            'THE THREE FACTS: (F4-1) query-selects-slot is fully predictable by a rule naming no answer '
            'word; (F4-2) routing resolves in L21-L23 (0.590 / 0.653 / 0.972), NOT at the readout layer, '
            'and L35 collapses to 0.562 so the rule holds only on L23-L34 - consistent with Q2-C1 '
            'whose readout point is L33, so L19-L24 picks the slot and L33 reads its content; (F4-3) the '
            'match uses context-free static entity codes. '
            'CAVEAT ON F4-3, recorded in the same breath: T1 already measured same-token cross-context key '
            'cos = 0.998, so kbar ~= k is near-identity and R2 ~= R1 is largely entailed by T1 rather than '
            'independent evidence. It rules out context-SPECIFIC rewriting affecting slot selection, not '
            'context playing no role in computation. '
            'WHAT THIS IS NOT (D22): a mechanism. The three-piece mechanism kit has (i) a predictive rule '
            'yes, (ii) a directional intervention no, (iii) transform-family stability no (only t1, one of '
            'the eight frozen transforms). Under D10 this is an E1 regularity / description, not a '
            'mechanism. Q4-P is pre-registered to supply (ii): substitute the last-position query '
            'activation at L20 for a different query and check that the predicted slot AND the output both '
            'flip. The anchor sits on the CONTENT of a position (D20), not on any head.'
        ),
    },
]


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    have = {(m.get('phase'), m.get('name')) for m in doc['measurements']}
    bak = LEDGER.with_suffix('.json.bak_q4r')
    if not bak.exists():
        shutil.copy2(LEDGER, bak)
        print('备份 ->', bak.name)

    added = 0
    for e in ENTRIES:
        if (e['phase'], e['name']) in have:
            print('跳过（已存在）%s' % e['name'])
            continue
        prev = doc['ledger_sha256_8']
        e = dict(e)
        e['prev_post'] = prev
        doc['measurements'].append(e)
        head = hashlib.sha256(
            json.dumps(doc, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
        doc['ledger_sha256_8'] = head
        added += 1
        print('#%d %s  prev=%s  post=%s' % (len(doc['measurements']), e['name'], prev, head))

    LEDGER.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                      encoding='utf-8', newline='')
    print('n=%d  head=%s  新增=%d' % (len(doc['measurements']), doc['ledger_sha256_8'], added))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
