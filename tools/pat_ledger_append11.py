# -*- coding: utf-8 -*-
"""追加 Q5-A 一条测量 + 一条 negative（多维加权融合否证）。幂等。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

NOTE = (
    'Signature bit S (selection): is multi-pattern combination a WEIGHTED SUM or a WINNER-TAKE-ALL. '
    'Judged by SHAPE, not significance (D24 top priority), so it cannot be rescued by slicing. '
    'PREREGISTERED and frozen before running (PAT_Q5A_PREREG_v1.md). '
    'FIVE CRITERIA: A1 median three-slot entropy <=0.35; A2 median max single-position mass >=0.60; '
    'A3 signal-bearing head unanimity >=0.80; A4 three-slot coverage >=0.40 (framework precondition); '
    'A5 dual-query entropy increase >=+0.30 versus a LENGTH-MATCHED NON-QUERY control. '
    'RESULT, 144 cells, behaviour gate 140/144, 15 seconds: A1 FALSE (h3 0.880), A2 FALSE (maxpos 0.462), '
    'A3 TRUE (unanimity 0.875), A4 FALSE (coverage 0.100), A5 FALSE (delta h3 -0.025). '
    'PREREGISTERED VERDICT: FRAMEWORK_INVALID, because A4 is a precondition. '
    'WHAT THE FRAMEWORK GOT WRONG (my error, not the data): I assumed the competition happens BETWEEN '
    'ENTITY SLOTS. The actual L28 decomposition is pos0 (<|im_start|>, the sink) 0.462, three entity '
    'slots 0.100, remaining function words and punctuation ~0.44, with pos1 at 0.000. So the battlefield '
    'is REGISTER VERSUS CONTENT, not content versus content. '
    'F5-1 the three-slot normalised share is [0.741, 0.182, 0.076], margin 0.75, matching the 0.816 that '
    'Q4-R reported on L28 alone - Q4-R 1.000 was achieved inside a 10 percent mass difference. '
    'F5-2 NO CRITICAL HEAD, evidence accumulates gradually. Random k heads aggregated by slot mass on '
    'L28: k=1 0.638, 2 0.743, 4 0.909, 8 0.957, 16 0.995, 32 1.000, monotonically with no saturation '
    'point. Best single head 1.00, median 0.67. This is the QUANTITATIVE form of distributivity that '
    'nine earlier ablation rounds could only state qualitatively as no-necessary-head. '
    'F5-3 cross-validation of the implementation: k=32 per layer gives L21 0.590, L22 0.653, L23 0.972, '
    'L28 1.000, IDENTICAL to the independently measured Q4-R profile, so the in-house RoPE and GQA '
    'broadcast are correct. '
    'F5-4 no semantic-dimension division of labour among heads: signal-bearing heads agree 0.875, and '
    'more decisively the k curve REACHES 1.000 - if heads formed groups pointing at different slots the '
    'aggregate would saturate at an intermediate value, it cannot reach unity. '
    'F5-5 register versus content IS a real functional split: per-head sink mass on L28 ranges '
    '[0.087, 0.938], median 0.644, within-cell head sd 0.261, and correlates negatively with per-head '
    'accuracy at Pearson r = -0.371. So there are two kinds of head, but it is a BINARY functional '
    'split, not the four semantic dimensions (language, style, logic, grammar) that the hypothesis '
    'posited. '
    'F5-6 dual query does not split: true dual-query sentence versus length-matched non-query control '
    'gives delta h3 = -0.025 (IQR [-0.058, -0.003], n=192), against a +0.30 threshold for fusion, and '
    'the sign is even negative. The length-matched control is essential because a longer sentence moves '
    'RoPE phases. '
    'SIGNATURE BIT S RESOLVED: argmax, but at the AGGREGATE level and only after about 16 heads '
    'accumulate; the competition is between register and content, not between semantic dimensions. '
    'CONSEQUENCE: if S is argmax then C is almost necessarily serial composition, which turns Q5-C from '
    'a two-way shape choice into a strong prediction. '
    'CANNOT CLAIM: single-head entropy shape was not measured, so single heads may still be mixed; only '
    'Qwen3-4B and one sentence family; cross-model transfer not done, so not E3; and the 0.100 coverage '
    'depends on the definition of sink, hence is not transferable across models or templates.'
)

NEG = {
    'name': 'pat_q5a_neg_multidim_weighted_fusion',
    'phase': 'Q5-A',
    'date': '2026-11-01',
    'claim_refuted': (
        'Next-word generation combines several semantic dimensions (language, style, logical context, '
        'grammatical rules) by simultaneous weighted fusion inside one attention decision.'),
    'why_refuted': (
        'Three independent shape readings contradict simultaneous weighted fusion. '
        '(1) A true dual query does not split the attention: delta entropy -0.025 against a length-matched '
        'control, while fusion predicts a large positive increase. '
        '(2) The aggregation curve over heads rises monotonically to exactly 1.000, whereas a mixture of '
        'dimensions would saturate below unity or turn over as dimensions pull against each other. '
        '(3) Heads show no dimension-specific routing (unanimity 0.875), so the posited dimensions have '
        'no physical carrier in attention. '
        'The only functional split that does exist is binary: register (sink) heads versus content heads, '
        'per-head sink mass spanning 0.087 to 0.938 with r = -0.371 against accuracy.'),
    'scope': (
        'Qwen3-4B, relation-extraction templates, 24 triples x 2 voices x 3 queries, layers L21-L23 and '
        'L28. Does not refute weighted fusion in MLP or in the residual stream; it refutes it IN ATTENTION '
        'ROUTING for this model and task family.'),
    'what_it_does_not_refute': (
        'That the RESIDUAL STREAM may carry superposed semantic dimensions. Attention answers only which '
        'position to read; the content read may still be a superposition. This Phase says nothing about it.'),
}


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    have = {(m.get('phase'), m.get('name')) for m in doc['measurements']}
    bak = LEDGER.with_suffix('.json.bak_q5a')
    if not bak.exists():
        shutil.copy2(LEDGER, bak)
        print('备份 ->', bak.name)

    entry = {
        'phase': 'Q5-A', 'name': 'pat_q5a', 'line': 'G',
        'date': '2026-11-01',
        'verdict': 'attention_routing_is_single_winner_at_the_aggregate_level_not_weighted_fusion',
        'res_sha8': 'd780449a',
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
        print('跳过（已存在）pat_q5a')

    negs = doc.setdefault('negatives', [])
    if not any(n.get('name') == NEG['name'] for n in negs):
        negs.append({
            'id': len(negs) + 1, 'name': NEG['name'], 'phase': NEG['phase'],
            'date': NEG['date'], 'claim_refuted': NEG['claim_refuted'],
            'why_refuted': NEG['why_refuted'], 'scope': NEG['scope'],
            'what_it_does_not_refute': NEG['what_it_does_not_refute'],
            'res_sha8': 'd780449a',
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
