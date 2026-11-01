# -*- coding: utf-8 -*-
"""追加 Q6-A 一条测量 + 一条 negative（层级不是方向 / 关系方向按类型分裂）。幂等。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'
SHA = '71dfa425'

NOTE = (
    'Does a RELATION exist as a shared DIRECTION in the residual stream, reusable across different '
    'sources and different targets ("reuse-of-difference")? This is the owner hypothesis: everything is '
    'relational, so localisation must fail; and a limited parameter budget must reuse attribute patterns. '
    'PREREGISTERED and frozen (PAT_Q6A_PREREG_v1.md, five pre-run amendments documented, thresholds never '
    'changed). 6 relation families, 54 instances, 78 words, 3 carrier sentences each, residual stream at '
    'the word span averaged over the span, all 37 layers. Delta(a->b) = v_b - v_a, unnormalised. '
    'THREE CONTROLS, each of which kills a different false positive: '
    '(1) A1b uses only pairs whose TARGETS DIFFER, killing "Delta merely points at one target word"; '
    '(2) A1p is a WITHIN-FAMILY PERMUTATION of source-target pairing that PRESERVES THE TARGET MULTISET, '
    'so target-word similarity, delta norms and n are all absorbed by the null - this is the only test '
    'that can kill the "F4 has only 小/大 and those two words are themselves cos 0.82" artefact; '
    '(3) a family-by-family delta cosine matrix, to test whether "noun-to-modifier region crossing" '
    'explains everything. Verdicts use SHAPE (fraction of the 37 layers passing), never a single-layer '
    'extremum, so there is no 36-way comparison. '
    'A1 RESULT, verdict MIXED, and the split is BY RELATION TYPE, not by noise. '
    'F3 COLOUR: A1b passes 30/37 layers, margin +0.359 at L24, permutation p = 0.000. '
    'F4 SIZE: A1b passes 33/37, margin +0.466, p = 0.030. '
    'F1 TAXONOMIC L1: A1b passes only 2/37, but at the peak layer A1b = 0.256 with margin 0.256 and '
    'p = 0.000, within rising 0.09 at L0 to 0.34 at L17-L18 then decaying to 0.21 by L35 - a NARROW BAND '
    'L13-L24, not a whole-stack property. '
    'F6 TOOL-PURPOSE: 4/37 layers, margin +0.236, borderline. '
    'F5 ANTONYM: 0/37 layers, peak within only 0.134, margin +0.125; permutation p is below 0.05 but the '
    'effect size is three times smaller than colour, so it is statistically detectable and practically '
    'negligible. '
    'F2 TAXONOMIC L2 is UNINTERPRETABLE and must not be cited: all 8 instances share the target 食物, so '
    'the permutation null is mathematically identical to the observation and p lands on 0 or 1 purely by '
    'floating-point tie-breaking. Its within of 0.511 is target locking, not a relation. '
    'F6-2 RELATION DIRECTIONS DO NOT GENERALISE ACROSS RELATION TYPES. Family-by-family delta cosine: '
    'F1-F2 +0.208 and F4-F6 +0.201, but F3-F4 is only +0.081 - colour and size are BOTH noun-to-modifier '
    'yet barely aligned, while both taxonomic families align. So the carrier is the semantic relation, '
    'not the part-of-speech transition. Colour and size are two different arrows. F4-F5 -0.057 and '
    'F5-F6 -0.019 are negative. '
    'A2 HIERARCHY IS REFUTED, AND IN THE OPPOSITE DIRECTION. Same-chain cos(Delta(a->b), Delta(b->c)) is '
    'negative at EVERY layer, median -0.246, while the cross-chain control is -0.005, i.e. exactly zero. '
    'Pre-registered alternative (radial encoding: abstract words sit closer to the global mean direction, '
    'so the second hop retracts while the first extends) was tested and REFUTED: strictly monotone chains '
    'peak at 0.60 on one layer only, 0/37 layers pass. So "one abstraction arrow used twice" is false, and '
    'the cheap radial rescue is also false. What the negative value IS remains unknown. '
    'A3 ANALOGY ARITHMETIC IS UNDERPOWERED, NOT REFUTED. Word-holdout 5-fold, per-fold layer choice: '
    'accuracy 0.375 versus nearest-neighbour baseline 0.188, majority baseline 0.370, chance 0.013. '
    'CI [0.219, 0.562] with n = 32; the criterion was 0.52, and the CI upper bound 0.562 EXCEEDS it. '
    'Per D23 this may only be recorded as PENDING_power; about 170 held-out instances are needed. '
    'DISSOCIATION WORTH KEEPING: direction sharing and vector arithmetic go opposite ways here - F4 is '
    'strongest on A1b (33/37) and weakest on A3 (0/8), F5 is the reverse (0/37 and 6/8). So "the same '
    'direction is reused" and "the direction can be added" are different claims. '
    'WHAT THIS DOES NOT SETTLE: no behavioural link, so nothing here proves the model USES these '
    'directions; no intervention; no cross-model transfer, so not E3; only Qwen3-4B, Chinese, these 78 '
    'words. E1 profile only. F2 within must never be quoted. '
    'OWNER HYPOTHESIS SCORECARD: "relations are directions" partly true (F6-1); "reuse mechanism must '
    'exist" true but the axis is relation direction, not attribute-value reuse, whereas Q3-T3 already '
    'showed entity-indexed reuse; "hierarchy shares one arrow" refuted with the sign reversed; "everything '
    'is relative so localisation fails" now PARTIALLY DISCRIMINATED - pure redundancy predicts '
    'within ~ across ~ 0, which holds for antonym (0.134 vs 0.008) but clearly fails for colour and size '
    '(0.40-0.54 versus 0.036-0.071).'
)

NEG = {
    'name': 'pat_q6a_neg_single_abstraction_arrow',
    'phase': 'Q6-A',
    'date': '2026-11-01',
    'claim_refuted': (
        'Taxonomic hierarchy (apple -> fruit -> food -> object) is encoded as ONE reusable abstraction '
        'direction, applied repeatedly: the second hop reuses the same difference vector as the first.'),
    'why_refuted': (
        'Same-chain cos(Delta(a->b), Delta(b->c)) is NEGATIVE at every one of the 37 layers, median -0.246, '
        'while the cross-chain control is -0.005. If one arrow were reused the value would be strongly '
        'positive. The pre-registered rescue hypothesis - radial encoding, where more abstract words sit '
        'closer to the global mean so the second hop retracts - was tested and also fails: strictly '
        'monotone chains peak at 0.60 on a single layer, 0/37 layers pass the threshold. '
        'A second, independent control rules out part-of-speech or semantic-region crossing as the cause: '
        'colour (noun->noun) and size (noun->adjective) are both noun-to-modifier yet their family-level '
        'delta cosine is only +0.081, while the two taxonomic families align at +0.208.'),
    'scope': (
        'Qwen3-4B, Chinese, residual stream at the word span, 10 chains of the form specific -> category -> '
        'super-category, 3 carrier sentences. Refutes "one reusable abstraction arrow" as a description of '
        'the residual-stream geometry; does not refute that depth or abstraction is encoded somehow.'),
    'what_it_does_not_refute': (
        'That hierarchy may be encoded outside the residual stream geometry measured here (e.g. in the '
        'attention pattern or in the MLP weights), or that a direction exists but is used with a '
        'position-dependent or layer-dependent transform. The systematic NEGATIVE value is unexplained and '
        'is carried forward as an open anomaly, not as evidence for any alternative.'),
}


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    have = {(m.get('phase'), m.get('name')) for m in doc['measurements']}
    bak = LEDGER.with_suffix('.json.bak_q6a')
    if not bak.exists():
        shutil.copy2(LEDGER, bak)
        print('备份 ->', bak.name)

    entry = {
        'phase': 'Q6-A', 'name': 'pat_q6a', 'line': 'G',
        'date': '2026-11-01',
        'verdict': 'relation_is_a_shared_direction_for_attribute_relations_only_hierarchy_is_not_a_direction',
        'res_sha8': SHA,
        'note': NOTE,
    }
    added = 0
    if (entry['phase'], entry['name']) not in have:
        prev = doc['ledger_sha256_8']
        entry['prev_post'] = prev
        doc['measurements'].append(entry)
        head = hashlib.sha256(
            json.dumps(doc, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
        doc['ledger_sha256_8'] = head
        added += 1
        print('#%d %s  prev=%s  post=%s' % (len(doc['measurements']), entry['name'], prev, head))
    else:
        print('跳过（已存在）pat_q6a')

    negs = doc.setdefault('negatives', [])
    if not any(n.get('name') == NEG['name'] for n in negs):
        negs.append({
            'id': len(negs) + 1, 'name': NEG['name'], 'phase': NEG['phase'],
            'date': NEG['date'], 'claim_refuted': NEG['claim_refuted'],
            'why_refuted': NEG['why_refuted'], 'scope': NEG['scope'],
            'what_it_does_not_refute': NEG['what_it_does_not_refute'],
            'res_sha8': SHA,
        })
        head = hashlib.sha256(
            json.dumps(doc, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
        doc['ledger_sha256_8'] = head
        added += 1
        print('negative #%d 新增  head=%s' % (len(negs), head))
    else:
        print('跳过（negative 已存在）')

    LEDGER.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                      encoding='utf-8', newline='')
    print('n=%d  head=%s  negatives=%d  新增=%d'
          % (len(doc['measurements']), doc['ledger_sha256_8'], len(negs), added))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
