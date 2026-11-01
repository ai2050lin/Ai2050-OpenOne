# -*- coding: utf-8 -*-
"""追加 Q6-B2 / Q6-B 两条测量 + 两条 negative。幂等（按 (phase,name) 去重）。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

NOTE_B2 = (
    'Is A2\'s negative value (-0.246, same-chain cos of two consecutive hierarchy hops) a semantic '
    'finding or a geometric inevitability? The question had to be asked because Delta1 = v_b - v_a and '
    'Delta2 = v_c - v_b SHARE THE MIDDLE WORD b WITH OPPOSITE SIGNS: for isotropic vectors '
    'E[Delta1.Delta2] = -sigma^2 and E[|Delta1||Delta2|] = 2 sigma^2, hence E[cos] = -0.5. The '
    'cross-chain control of Q6-A is -0.005 precisely because cross-chain hops share no word, so it '
    'cannot adjudicate the same-chain value. PREREGISTERED (PAT_Q6B2_PREREG_v1.md) with a framework '
    'gate and two nulls that use the SAME estimator and differ only in which word replaces c: '
    'null_rand draws d uniformly from the 79-word vocabulary, null_level draws d from the six '
    'superordinates (food/creature/object/vegetable/fruit/animal) so that explanation "superordinates '
    'form their own region" is directly tested. 10 chains, 37 layers, 200 draws per (chain, layer). '
    'GATE G0 PASSED: null_rand all-layer median -0.3994, inside the pre-set window [-0.65,-0.35], '
    'empirically confirming the isotropic -0.5 prediction. '
    'RESULT (all-layer medians): obs -0.2461 (reproduces Q6-A\'s -0.246 exactly), null_rand -0.3994, '
    'null_level -0.3075. The value decomposes exactly: obs = null_rand + (null_level - null_rand) + '
    '(obs - null_level) = -0.3994 + 0.0919 + 0.0624. Geometry supplies 0.399, superordinate class '
    'supplies 0.092, and the SPECIFIC CHAIN supplies only 0.062. '
    'REFERENCE LINE measured in the same run with the same estimator: the already-established shared '
    'directions are F3 colour +0.323 median (peak +0.395) and F4 size +0.376 (peak +0.494). '
    'obs never exceeds -0.1716 at any layer, so hierarchy never comes within 0.49 of a shared direction. '
    'VERDICT: main NOT_A_SHARED_DIRECTION, attribution SUPERORDINATE_EXPLAINED (|obs-null_level|<0.05 in '
    '38% of layers vs |obs-null_rand|<0.05 in 5%). t>1 fraction is 0.00 for both obs and null, i.e. no '
    'chain places c beyond b on the a->b axis; residual ratio rho median 0.969, so the second hop is '
    'essentially perpendicular to the first, not weakly collinear. '
    'CONSEQUENCE, and this is the point of the phase: the ORIGINAL REASON for rejecting hierarchy-as-a-'
    'direction is void - it mistook a geometric artefact for a semantic discovery - while the '
    'CONCLUSION is strengthened, because it now rests on absolute level (max -0.172 versus +0.32..+0.49) '
    'rather than on a sign. -0.246 must never again be quoted on its own as evidence.'
)

NOTE_B = (
    'Can attribute knowledge be compressed the way "a million objects x colour/size/weight from a finite '
    'parameter budget" requires, and are attribute patterns decoupled from the concrete entities? '
    'PREREGISTERED (PAT_Q6B_PREREG_v1.md) with seven pre-run amendments, all of them recording a way the '
    'criterion would have validated itself automatically. '
    'M[o,a] = log softmax(logits at prefix end)[token(a)], 28 objects x 12 attribute values (8 colours, '
    '2 sizes, 2 temperatures), 2 templates each. '
    'MEASUREMENT GATE (all pass): colour label 8/8 with margins up to 20.3 nats, size 8/8, cross-template '
    'Spearman 0.655, within-type contrast 6.56 nats. The matrix is a hard signal, not noise. '
    'Q1 RANK: cell-wise 5-fold completion with main effects plus rank-R residual. held-out R2 is 0.197 at '
    'R=0 and DECREASES monotonically with rank: R=1 0.118, R=2 -0.181, R=3 -0.406, R=4 -1.059, R=11 '
    '-0.119; the shuffled null is worse at every R (0.016 at R=0, -0.631 at R=3), so the pipeline is '
    'sound and the real matrix does carry structure - it is simply NOT low-rank. R* is undefined. '
    'VERDICT NO_LOW_RANK_STRUCTURE. '
    'WHY, quantified: two-way decomposition gives row 0.106, column 0.315, interaction 0.579. The '
    'interaction is the attribute knowledge, it is 58% of the variance, and it is not predictable from '
    'any other cell at any rank. The reason is sparsity: (max-median)/(max-min) has median 0.50 and each '
    'object has only 1.18 values within 1 nat of its peak per attribute type (median 1.0). A sparse '
    'matrix is near full rank by construction, so low-rank completion must fail - this is the wrong model '
    'class, not a failed measurement. '
    'COMPRESSION: 336 cells reduce to about 28 x 3 x 1.18 = 99 active indices, and each index is an entry '
    'in a closed 12-value codebook. The stored form is therefore a pointer (entity -> attribute type -> '
    'codebook index), not an attribute vector. This converges with Q3-T3 (reuse indexed by ANSWER ENTITY '
    '7.88x versus by role 1.24x): the unit of reuse is the entity, not the relation or the attribute. '
    'Q2 DECOUPLING returned NOT_SEPARABLE but MUST NOT BE CITED as evidence in either direction: '
    'dec is 0.95-1.00 for attribute values and 0.97-1.00 for the adjective null, diff -0.045..-0.0005, '
    'i.e. the statistic is SATURATED. A 3-dimensional subspace spanned in 2560 dimensions occupies about '
    '0.12% of the volume, so every vector is trivially outside it. This is the third instance of the '
    'same failure (T1 R_id = 1.000, T3 share_id = 1.00): whenever the denominator contains a naturally '
    'tiny quantity the criterion pins to the ceiling and has no discriminating power.'
)

NEG = [
    {
        'name': 'pat_q6b2_neg_shared_middle_word_geometry',
        'phase': 'Q6-B2', 'date': '2026-11-01',
        'claim_refuted': (
            'A negative cosine between two difference vectors that share one word (Delta1 = b - a, '
            'Delta2 = c - b) can be interpreted semantically, e.g. as evidence that a relation is not '
            'a reusable direction.'),
        'why_refuted': (
            'Sharing the middle word with opposite signs puts the isotropic expectation at -0.5, which the '
            'measured null reproduces (-0.399 over 200 draws per chain-layer). The observed -0.246 decomposes '
            'into -0.399 geometry + 0.092 superordinate class + 0.062 chain-specific, so the semantic '
            'contribution is an order of magnitude smaller than what a real shared direction would need '
            '(+0.72 to reach the +0.32..+0.49 of the established colour and size directions). Any conclusion '
            'drawn from the raw sign is therefore unfounded.'),
        'scope': (
            'Qwen3-4B, Chinese, residual stream, 10 taxonomic chains, 37 layers. Applies to every experiment '
            'that compares two difference vectors sharing a term, which includes most analogy and '
            'compositionality probes.'),
        'what_it_does_not_refute': (
            'That hierarchy is encoded somewhere. It only removes the negative cosine as evidence; the '
            'conclusion that hierarchy is not a shared direction survives via absolute level, not via sign.'),
    },
    {
        'name': 'pat_q6b_neg_attribute_low_rank_reuse',
        'phase': 'Q6-B', 'date': '2026-11-01',
        'claim_refuted': (
            'A finite parameter budget stores "many objects x many attributes" by low-rank factorisation, '
            'i.e. attribute knowledge is a low-dimensional factor model over entities and attribute values, '
            'and therefore can be recovered by matrix completion.'),
        'why_refuted': (
            'Held-out cell R2 is maximal at rank 0 (0.197) and falls monotonically with rank (-0.406 at '
            'rank 3, -1.059 at rank 4), while the shuffled null is worse everywhere, so the structure is '
            'real but not low-rank. The interaction, which is the attribute knowledge itself, is 58% of the '
            'variance and is not predictable from other cells at any rank. Sparsity explains it: each object '
            'has only 1.18 values within 1 nat of its peak per attribute type, and a sparse matrix is near '
            'full rank. The correct description is a pointer into a closed 12-value codebook, not a factor '
            'model.'),
        'scope': (
            'Qwen3-4B, Chinese, 28 objects x 12 attribute values (colour/size/temperature), behavioural '
            'log-probability matrix. Refutes low-rank as the compression mechanism for this table; does not '
            'exclude low-rank structure elsewhere (e.g. across a much larger object set, or in the residual '
            'geometry rather than in the behavioural table).'),
        'what_it_does_not_refute': (
            'That compression exists. It does: 336 cells reduce to about 99 active codebook indices. What is '
            'refuted is the specific algebraic form "low rank", not the existence of reuse.'),
    },
]


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    have_m = {(m.get('phase'), m.get('name')) for m in doc['measurements']}
    have_n = {n.get('name') for n in doc.get('negatives', [])}
    bak = LEDGER.with_suffix('.json.bak_q6b')
    if not bak.exists():
        shutil.copy2(LEDGER, bak)
        print('备份 ->', bak.name)

    entries = [
        {'phase': 'Q6-B2', 'name': 'pat_q6b2', 'line': 'G', 'date': '2026-11-01',
         'verdict': 'hierarchy_negative_cosine_is_geometric_plus_superordinate_region_not_a_direction',
         'res_sha8': '4ee36119', 'note': NOTE_B2},
        {'phase': 'Q6-B', 'name': 'pat_q6b', 'line': 'G', 'date': '2026-11-01',
         'verdict': 'attribute_knowledge_is_sparse_pointer_not_low_rank',
         'res_sha8': '11531e1e', 'note': NOTE_B},
    ]
    added = 0
    for entry in entries:
        if (entry['phase'], entry['name']) in have_m:
            continue
        entry['prev_post'] = doc['ledger_sha256_8']
        doc['measurements'].append(entry)
        head = hashlib.sha256(
            json.dumps(doc, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
        doc['ledger_sha256_8'] = head
        added += 1
        print('+ measurement %-8s -> head %s (n=%d)' % (entry['name'], head, len(doc['measurements'])))

    for n in NEG:
        if n['name'] in have_n:
            continue
        doc.setdefault('negatives', []).append(n)
        head = hashlib.sha256(
            json.dumps(doc, ensure_ascii=False, indent=1).encode('utf-8')).hexdigest()[:8]
        doc['ledger_sha256_8'] = head
        added += 1
        print('+ negative   %-40s -> head %s (neg=%d)'
              % (n['name'], head, len(doc['negatives'])))

    if added:
        LEDGER.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                          encoding='utf-8', newline='')
    print('n=%d head=%s neg=%d added=%d'
          % (len(doc['measurements']), doc['ledger_sha256_8'], len(doc['negatives']), added))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
