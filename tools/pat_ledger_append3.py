# -*- coding: utf-8 -*-
"""追加 Q2-E / Q2-F / Q2-G 三条测量 + 一条 negative（Q2-E 的 E1 判定撤回）。幂等。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

ENTRIES = [
    {
        'phase': 'Q2-E', 'name': 'pat_q2e_additivity', 'line': 'G',
        'date': '2026-11-01', 'verdict': 'uninformative_floor_collision_verdict_withdrawn',
        'res_sha8': '46845c52',
        'note': (
            'Two-band additivity test, 9 arms, 144 cells, base 1.0000. A=L0-L3, B=L16-L19. '
            'A128 0.3264 / B128 0.3403 / AB256 0.3333; gap_full = -0.007. Half density: '
            'A64 0.5255 (3 draws: 0.2917 / 0.3472 / 0.9375), B64 0.9074, AB128 0.4560, '
            'gap_half = +0.069 CI [-0.014, +0.174]. Verdict E1 "single saturating pathway" is '
            'WITHDRAWN in Q2-G: all three full-density arms sit at the 3-way chance floor 0.3333, '
            'so gap_full = 0 carries no information. gap_half used a mean over a bimodal draw '
            'distribution (see Q2-F) and is also unusable. What survives is the failure-set '
            'structure: |fail_A|=97, |fail_B|=95, |fail_AB|=96, union 137, Jaccard 0.401, and only '
            '2 cells failing under AB that fail under neither single band.'
        ),
    },
    {
        'phase': 'Q2-F', 'name': 'pat_q2f_hetero', 'line': 'G',
        'date': '2026-11-01', 'verdict': 'within_band_heterogeneous_threshold_not_additive',
        'res_sha8': '4996caf9',
        'note': (
            '12 independent 64-head draws per band (16 per layer, layer-balanced), 144 cells, '
            'base 1.0000. A64 (L0-L3) accuracy range 0.3403 to 1.0000, mean 0.7836, median 0.8854, '
            'sd 0.2500, range 0.6597 - same density, same layers, from no damage to floor. '
            'B64 (L16-L19) mean 0.8235, sd 0.1322, range 0.3194. Per-draw structure: mild group '
            '(dA<0.15 and dB<0.15, n=5) has dA+dB of 0.05/0.23/0.05/0.13/0.13 but dAB of '
            '0.00/0.38/0.54/0.35/0.37 - strongly super-additive; severe group (dA>=0.30, n=3) has '
            'dA 0.65/0.56/0.66 and dAB 0.64/0.62/0.68 - saturated at floor. Model fit over 12 draws: '
            'dAB~max(dA,dB) MAE 0.155, dAB~min(1,dA+dB) MAE 0.192, both fail in the mild group. '
            'Conclusion: the two bands are a SERIAL SHARED pathway with a damage budget - mild '
            'damage to one segment is absorbed, mild damage to both crosses the budget, severe '
            'damage to either already exhausts it. Not parallel redundancy, not additivity. '
            'Also: carriage is not localizable by layer position inside the band; which heads are '
            'drawn matters more than which half of the band.'
        ),
    },
    {
        'phase': 'Q2-G', 'name': 'pat_q2g_coverage_law', 'line': 'G',
        'date': '2026-11-01', 'verdict': 'chance_floor_1_3_and_concentration_amplifies_3_4x',
        'res_sha8': '4d1f7eb3',
        'note': (
            'Coverage law with layer identity eliminated: all 36 layers touched, k heads ablated '
            'per layer, k in {2,4,6,8,12,16,24,32}, 3 draws, 144 cells, base 1.0000. '
            'drop = 0.039 / 0.197 / 0.373 / 0.431 / 0.639 / 0.657 / 0.662 / 0.667. '
            'THE ACCURACY FLOOR IS 0.3333, the 3-way chance level: k>=12 arms read 0.3611 / 0.3426 '
            '/ 0.3380 / 0.3333, i.e. the behaviour is all-or-none, destroyed to guessing rather than '
            'degraded. Max possible drop is therefore 0.667, not 1.0. '
            'Concentration effect: 128 heads in 4 layers (11.1%, Q2-D2) drop 0.674 = floor, versus '
            '144 heads spread over 36 layers (12.5%, k=4 here) drop 0.197 - same head count, same '
            'coverage, 3.4x difference. Damage is coverage x concentration, not total head count. '
            'This single law explains the whole history: 1 layer 32 heads (2.8%) -> 0.000 '
            '(compensable), 4 layers 128 heads (11.1%) -> floor, 36 layers 12.5% -> 0.197, '
            '36 layers 37.5% -> floor. It also re-reads Q2-D2: L0-L3 0.674 and L16-L19 0.660 are '
            'BOTH at floor, so the 0.014 difference between the two peaks is noise and must not be '
            'read as L0-L3 mattering more.'
        ),
    },
]

NEGATIVE = {
    'claim': (
        'the two carrier bands L0-L3 and L16-L19 share one saturating pathway, established by '
        'gap_full = drop(AB256) - max(drop(A128), drop(B128)) being approximately zero at full band '
        'density (Q2-E preregistered verdict E1)'
    ),
    'evidence': (
        'Q2-G (res 4d1f7eb3) shows the accuracy floor is 0.3333, the 3-way chance level. In Q2-E '
        '(res 46845c52) A128 = 0.3264, B128 = 0.3403 and AB256 = 0.3333 all sit ON that floor, so '
        'gap_full = -0.007 is a floor collision and cannot distinguish saturation from additivity: '
        'an additive model would also have predicted a floor value, since 0.674 + 0.660 exceeds the '
        '0.667 maximum drop. The half-density gap (+0.069) is equally unusable because the A64 draw '
        'distribution is bimodal with range 0.66 (Q2-F), making the mean meaningless. The Q2-E E1 '
        'verdict is withdrawn as uninformative, not as wrong-in-direction: the surviving evidence '
        'for a shared pathway is the failure-set overlap (Jaccard 0.401, only 2 AB-only new '
        'failures) and the Q2-F per-draw super-additivity in the mild regime. Standing rule D4: '
        'report the distance to floor/ceiling before comparing any two drops. Standing rule D5: '
        'check the draw distribution before averaging over random subsets; a bimodal or range>=0.30 '
        'distribution forbids the mean. This is the third instance of the same error class in this '
        'line (top-k ties degrading to layer order in Q1; n=24 SMOKE CI reported as evidence in '
        'Q2-A2; mean over a bimodal ablation draw here).'
    ),
}


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    have = {(m.get('phase'), m.get('name')) for m in doc['measurements']}
    bak = LEDGER.with_suffix('.json.bak_q2efg')
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
