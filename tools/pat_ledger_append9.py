# -*- coding: utf-8 -*-
"""追加 Q3-T3 一条测量 + 一条 negative（输出端无角色列族）。幂等。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

ENTRIES = [
    {
        'phase': 'Q3-T3', 'name': 'pat_q3t_t3', 'line': 'G',
        'date': '2026-11-01',
        'verdict': 'output_side_has_no_role_column_family_reuse_is_indexed_by_answer_entity_not_role',
        'res_sha8': 'b9fd3828',
        'note': (
            'T3 of the preregistered fingerprint family: the only place a "multi-mode fusion" could '
            'still live after T1/T2 showed the k side is inert. 144 cells (24 triples x 2 voices x 3 '
            'queries). Delta = log p(candidate) + log V over V=39 (12 persons + 30 nouns, deduped by '
            'first token); candidate mass coverage median 1.000, so no probability mass escapes the '
            'vector. Sparse dictionary K=32, <=3 nonzeros per cell, OMP, 5 random half-splits BY TRIPLE '
            '(not by cell) so no entity leaks across folds. '
            'THREE DESIGN DECISIONS RECORDED IN THE SCRIPT DOCSTRING BEFORE RUNNING: (1) triples rebuilt '
            'over 24 DISTINCT nouns - reusing T2 (12 persons x 2 nouns) would leave the th role with only '
            '2 distinct answers across 24 cells, making reuse(th) degenerate to near-ceiling and the '
            'criterion automatically true but informationless; (2) report HELD-OUT R2 as primary, since '
            'K=32 vs V=39 is near-overcomplete and in-sample R2 (0.989) is guaranteed high; (3) add a '
            'role-mean-removed variant so "different questions have different answers" cannot masquerade '
            'as a role column family. '
            'RESULT, behaviour gate 140/144 (ag 48/48, re 44/48, th 48/48). '
            'Held-out R2 median 0.819 (folds 0.845 / 0.819 / 0.888 / 0.753 / 0.807), mean 0.770; '
            'role-mean-removed held-out R2 0.851; in-sample 0.989. No fold reaches 0.90. '
            'Shared-subspace PCA held-out curve r=1,2,3,4,5,6,8 -> 0.346 0.394 0.447 0.489 0.546 0.578 '
            '0.653: a shared 8-dim subspace explains only 65.3% of an unseen triple, i.e. there is no '
            'shared low-dimensional mode that generalises across instances. '
            'Reuse ratios vs simulated random expectation (E[Jaccard]=0.0604, B=1000), bootstrap 95% CI, '
            'self-pairs excluded: raw ag 1.79 [1.52,2.10], re 1.61 [1.35,2.00], th 1.30 [1.04,1.60], '
            'cross-role 1.18; role-mean-removed 1.56 / 2.33 / 1.99, cross 1.35. Prereg prediction '
            'reuse >= 3 FAILED. VERDICT NULL_decomposition: the R2 premise fails, so reuse is diagnostic '
            'only. '
            'THE STRUCTURE THAT IS THERE: same-answer-entity 7.88x (n=168) vs different-answer 1.24x '
            '(n=3216), a 6.4x ratio, stable at 8.38 vs 1.62 after removing role means. Reuse is indexed '
            'by the answer entity, not by the role. Combined with the fact that role-internal Delta cosine '
            'is 0.9676 versus cross-role 0.9556 - a difference of only 0.012 - the three queries barely '
            'change the shape of the output distribution at all. '
            'Note the trivial-confound guard turned out UNNECESSARY: if answer vocabulary had dominated, '
            'cross-role cosine would have fallen well below within-role; it fell by 0.012. '
            'Mirrors 3281 from the output side: structure lives in q.k matching strength (write intensity '
            'D_norm predicting dynamic share, rho=0.97), not in a reusable set of output atoms. '
            'TWO OWN BUGS EXPOSED AND FIXED THIS ROUND, kept on record: (BUG-1) I asserted that global '
            'rank-3 PCA is an upper bound on any 3-atom sparse code; OMP measured 0.541 against that '
            '"bound" of 0.477, which is impossible, and the assertion is simply false - different cells '
            'may select different atoms, so the two constraint sets are not nested. The bound claim was '
            'withdrawn; the PCA curve is reported only as "how much a SHARED subspace explains". '
            '(BUG-2) the bootstrap resampled with replacement and kept (i,i) self-pairs whose Jaccard is '
            'identically 1.0, inflating the whole distribution - caught because the CI lower bound '
            '(1.813) exceeded the point estimate (1.788). Fixed by skipping same-index pairs, then the '
            'full run was repeated; CI now brackets the estimate. '
            'FAMILY SETTLEMENT: T1/T2/T3 all NULL, gate-1 budget of 3 rounds exhausted, family CLOSED. '
            'Closure rests on the budget cap, NOT on the inv criterion: inv requires sweeping all eight '
            'T-v1 transforms, which was not done, so no claim of permanent exclusion is made. No fourth '
            'Q3-T variant may be opened (that would be M2, an unclosed control space).'
        ),
    },
]

NEGATIVE = {
    'id': 'pat_q3t_neg_role_mode_reuse',
    'date': '2026-11-01',
    'claim': (
        '"多模式融合"指下一词分布由少数几个可复用的角色模式相加而成（角色各有专属的写入列族）。'
    ),
    'status': 'refuted',
    'evidence': ['Q3-T3 sha8 b9fd3828'],
    'detail': (
        'Held-out R2 of a K=32 / <=3-nonzero sparse dictionary over 144 cells is 0.819, below the '
        'preregistered 0.90; no single fold reaches 0.90 (best 0.888). A shared rank-8 PCA subspace '
        'explains only 65.3% of an unseen triple, so no shared low-dimensional mode transfers across '
        'instances. Role reuse is 1.30-1.79x random (median 1.61) against a preregistered threshold of '
        '3.0. Scope of the refutation is the OPERATIONALISED claim - a compact reusable role basis at '
        'the output distribution. What is NOT refuted, and is in fact supported, is entity-level '
        'structure: cells sharing the same answer word reuse dictionary columns 7.88x over chance '
        'versus 1.24x for different answers (6.4x ratio). The correct replacement statement is '
        '"the output distribution is indexed by answer entity, not organised into role modes", '
        'not "there is no reuse".'
    ),
}


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    have = {(m.get('phase'), m.get('name')) for m in doc['measurements']}
    bak = LEDGER.with_suffix('.json.bak_q3t3')
    if not bak.exists():
        shutil.copy2(LEDGER, bak)
        print('备份 ->', bak.name)

    if not any(n.get('id') == NEGATIVE['id'] for n in doc['negatives']):
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
