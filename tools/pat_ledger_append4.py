# -*- coding: utf-8 -*-
"""追加 Q2-H 一条测量 + 一条 negative（N2 判据"两极并存"推理不成立）。幂等。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

ENTRIES = [
    {
        'phase': 'Q2-H', 'name': 'pat_q2h_core', 'line': 'G',
        'date': '2026-11-01', 'verdict': 'no_core_heads_no_required_layer_total_amount_carriage',
        'res_sha8': '4ccb5dd6',
        'note': (
            'Nested head-count sweep inside L0-L3 (128 heads), layer-balanced, 12 independent draws '
            'per arm, 144 cells, base 1.0000. Arms k_per in {1,2,4,8,12,16} = {4,8,16,32,48,64} heads '
            '= {3.1,6.2,12.5,25.0,37.5,50.0}%. '
            'k<=4 (<=16 heads, 12.5%): 12/12 draws at exactly 1.0000, range 0.000 - every draw, '
            'all 144 cells correct. k=8: range 0.479 (one draw 0.5208, eight at 1.0000). '
            'k=12: range 0.674, one floor hit. k=16: range 0.667, two floor hits. '
            'Built-in replication of Q2-F A64 passed: k=16 range 0.6667 vs Q2-F 0.6597. '
            'DECISIVE EVIDENCE AGAINST A CORE: the 12 noise-free draws at k_per<=4 collectively touch '
            '119 of 128 heads (93.0%) - 93% of the band can be ablated with zero behavioural damage. '
            'PER-LAYER ALREADY EXCLUDED: Q2-D2 ablated each layer alone (32 heads, 100% within '
            'layer) and found <=0.007 for 35/36 layers, so no layer contains a required head either. '
            'NO PATHOGENIC COMBINATION: the two floor-reaching k=16 draws share exactly 32 heads, '
            'which equals the random expectation 4*(16*16/32)=32 - zero enrichment, so floor-reaching '
            'draws do not share a causative head set. '
            'Consequence: damage is a cumulative amount, not a structural property. What decides '
            'collapse is how much total capacity is removed, not which heads were removed. '
            'This closes the "find the critical neuron" route on this task via four independent '
            'routes (required head / required layer / pathogenic combination / local core). '
            'Re-reading of Q2-F: its 0.66 range is not "two classes of heads" but the same damage '
            'budget being crossed or not crossed under different draws, i.e. the budget boundary '
            'sits at the 64-head scale while draw-to-draw fluctuation is set by the distance to that '
            'boundary. '
            'Boundary: k<4 single-head necessity is not directly tested here (each layer 1 head, '
            '12 draws covers only 37.5% of a layer), but Q2-D2 independently excludes it because a '
            'required head would make its own layer fatal when fully ablated. Scope limited to '
            'L0-L3, to the role-unbinding readout, one model.'
        ),
    },
]

NEGATIVE = {
    'claim': (
        'the Q2-H preregistered criterion N2 is sound: "no arm reaches 12/12 floor AND some arm '
        'shows both perfect and floor draws" establishes a threshold-type (no fixed core) carriage'
    ),
    'evidence': (
        'The conclusion (no fixed core) is correct but the inference is invalid. The condition '
        'n_perfect>0 and n_floor>0 is merely "both extremes of the distribution are sampled", which '
        'every high-variance distribution satisfies; it cannot distinguish "a pathogenic '
        'configuration exists" from "pure total-amount driven damage". The actual discriminator is '
        'whether floor-reaching draws SHARE heads. Q2-H measures it: the two floor-reaching k=16 '
        'draws intersect in 32 heads, exactly the random expectation 4*(16*16/32)=32, i.e. zero '
        'enrichment. Verdict therefore re-adjudicated to N3-prime (total-amount carriage). Second '
        'criterion error in the same run: the original N1 ("range shrinks as the subset shrinks '
        'implies a stable core") had the direction reversed - R(k) necessarily goes to 0 as k grows '
        'regardless of whether a core exists, and with c core heads R(k) is LARGER at small k, so '
        'the criterion as written would have pointed the opposite way. Both were caught before the '
        'verdict was recorded: N1 by reasoning before execution, N2 by the intersection measurement '
        'after execution. result.json keeps verdict=N2 unchanged (results are immutable); the '
        're-adjudication lives in PAT_Q2H_FINDINGS_20261101.md and the MEMO. Standing rule D6: a '
        'preregistered verdict must be accompanied by the measurement that separates its '
        'alternatives; if the criterion cannot be computed from the data the run produces, it is '
        'not a criterion.'
    ),
}


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    have = {(m.get('phase'), m.get('name')) for m in doc['measurements']}
    bak = LEDGER.with_suffix('.json.bak_q2h')
    if not bak.exists():
        shutil.copy2(LEDGER, bak)
        print('备份 ->', bak.name)

    if not any(n.get('claim') == NEGATIVE['claim'] for n in doc['negatives']):
        doc['negatives'].append(NEGATIVE)
        print('negatives +=1 (共 %d)' % len(doc['negatives']))

    added = 0
    for e in ENTRIES:
        if (e['phase'], e['name']) in have:
            print('跳过（已存在）%s' % e['name'])
            continue
        prev = doc['ledger_sha256_8']
        e = dict(e)
        e['prev_post'] = prev
        doc['measurements'].append(e)
        doc['ledger_sha256_8'] = prev
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
