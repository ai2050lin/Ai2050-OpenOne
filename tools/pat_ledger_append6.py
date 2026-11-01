# -*- coding: utf-8 -*-
"""追加 Q2-C2 / C2b / C2c 三条测量 + 一条 negative（方向对照的长度混杂）。幂等。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

ENTRIES = [
    {
        'phase': 'Q2-C2', 'name': 'pat_q2c2_direction', 'line': 'G',
        'date': '2026-11-01', 'verdict': 'L33_readout_is_direction_selective_random_direction_null',
        'res_sha8': '72a749d5',
        'note': (
            'First direction-resolved probe at the Q2-C1 readout point. Intervention edits only the '
            'last-position vector of layers[33] output on the corrupt sentence; clean/corrupt are the '
            'same-form entity swap, T=39/39 length matched. n=48 (24 triples x active/passive), '
            'behaviour gate 48/48 + 48/48. Four arms at the same displacement magnitude '
            '(alpha x ||Delta||): A along own Delta, B along a random unit direction, S along the '
            'leave-one-out consensus direction, R along the residual orthogonal to that consensus. '
            'RESULT: A = 0.000 / 0.292 / 0.854 / 1.000 at alpha 0.25/0.50/0.75/1.00 (monotone, '
            'criterion F2 linear-additive); B = 0.000 at EVERY alpha, paired A-B = +0.292 '
            '[0.167,0.417] at 0.50 and +0.854 [0.750,0.938] at 0.75 (criterion G1 direction-specific). '
            'A random direction of the same magnitude never flips the answer, so the readout is '
            'directionally selective and this is not a generic disruption effect. '
            'OFFLINE DIRECTION GEOMETRY: cos(Delta_i, Delta_j) over 552 pairs has mean 0.245, median '
            '0.167, range [-0.436, 0.993] - the per-sample difference directions are NOT collinear; '
            'there is no consensus role axis at this layer. Leave-one-out cos(Delta_i, m_hat_-i) mean '
            '0.482, residual share ||resid||/||Delta|| mean 0.836. '
            'WITHDRAWN IN THIS SAME ROUND (see negatives): the S/R split reported here (S=0.479, '
            'R=0.604 at alpha 1.0) is NOT usable because the three arms have different displacement '
            'lengths (S moves alpha*||Delta||/cos, i.e. 2.07x at cos=0.482). The S curve is '
            'non-monotone (0.639 at 0.75 -> 0.611 at 1.00), which is overshoot, not extra efficacy. '
            'The clean same-length comparison is Q2-C2c. '
            'DESIGN DEFECT: this round SMOKE used triples(4), and those 4 triples all share the same '
            'person pair (P0,P3) with only the object varying, so SMOKE cos was 0.936 versus 0.245 in '
            'the full run - the SMOKE sample is not the same population as the full run.'
        ),
    },
    {
        'phase': 'Q2-C2b', 'name': 'pat_q2c2b_projection', 'line': 'G',
        'date': '2026-11-01', 'verdict': 'projection_alignment_insufficient_length_must_be_controlled',
        'res_sha8': '81ace239',
        'note': (
            'Attempt to remove the magnitude confound by aligning all arms to the same target '
            'PROJECTION ratio p along Delta (p in 0,0.25,0.5,0.75,1.0), with scaling factors '
            '1 / cos / sin. RESULT: A = 0.000 / 0.000 / 0.292 / 0.854 / 1.000; S = 0.000 / 0.000 / '
            '0.422 / 0.639 / 0.611; R = 0.000 / 0.000 / 0.000 / 0.479 / 0.958; random B = 0.000 with '
            'measured p_eff 0.015 and no alpha dependence. Baseline p=0 check passed (all arms 0.000, '
            'no hook side effect). THE DESIGN FAILED ITS OWN PURPOSE: aligning the projection does NOT '
            'align the displacement length - S moves p*||Delta||/cos, which at cos=0.482 is 2.07x, so '
            'S overshoots and its curve turns down at p=1.0 (0.639 -> 0.611). Only A and R have '
            'identical length (both exactly p*||Delta||) and identical Delta projection. '
            'THE ONE CLEAN COMPARISON from this round: at p=0.75, A=0.854 versus R=0.479 - same '
            'length, same projection, directions 33 degrees apart, 0.375 difference. This is the first '
            'evidence that direction per se, not projection, governs the readout. It is superseded by '
            'Q2-C2c (where R reaches 0.958 at p=1.0), which shows R needs a LARGER step, not that 33 '
            'degrees is fatal - so the correct statement is about the size of the step, not the angle '
            'alone. K2/K3 verdicts in this round are not usable.'
        ),
    },
    {
        'phase': 'Q2-C2c', 'name': 'pat_q2c2c_samelength', 'line': 'G',
        'date': '2026-11-01', 'verdict': 'readout_is_angle_gated_cone_no_shared_direction',
        'res_sha8': '88df297d',
        'note': (
            'Decisive same-length four-direction comparison at L33 last position. Displacement length '
            'held exactly equal across arms (L = p*||Delta||, p in 0.5/0.75/1.0), only direction '
            'varies: D along own Delta (0 deg), R along the residual (33.3 deg), M along the '
            'leave-one-out consensus (61.2 deg), Z random (about 90 deg). n=48, behaviour gate 48/48 + '
            '48/48, baseline p=0 all arms 0.000. '
            'RESULT AT p=1.0: D = 1.000, R = 0.958, M = 0.479, Z = 0.000. '
            'RESULT AT p=0.75: D = 0.854, R = 0.479, M = 0.458, Z = 0.000. '
            'RESULT AT p=0.50: D = 0.292, R = 0.000, M = 0.021, Z = 0.000. '
            'THE M ARM IS A MIXTURE, NOT A MECHANISM. Stratifying by cos(Delta, m_hat): '
            'cos in [0,0.3) n=12 -> M = 0.000; [0.3,0.6) n=13 -> M = 0.077; [0.6,0.8) n=23 -> '
            'M = 0.957. corr(cos_M, M flip) = +0.789, and 23/48 = 0.479 is exactly the pooled mean - '
            'it is the fraction of samples inside the gate, not a half-strength direction. '
            'VERDICT: the readout is an ANGLE-GATED CONE with the boundary at cos about 0.6 '
            '(about 53 degrees). Inside, nearly full effect (0.957); outside, zero (0.000). '
            'This REJECTS a one-dimensional linear scalar readout (M4: max between-arm difference '
            '1.000, far above the 0.15 threshold that linear projection would require). '
            'COMBINED WITH Q2-C2 (cos(Delta_i,Delta_j) mean 0.245): there is NO shared role direction. '
            'The cone is oriented per sample. Therefore the framing "is the role a direction or a '
            'permutation" is mis-posed: Q2-C1 shows a POINTER decides which position is read (agent '
            'position flips 1.00 through L32, recipient/theme/question-first all 0.00 at all 36 '
            'layers), and this round shows the CONTENT at that position is encoded along a '
            'sample-specific cone. Pointer and direction are serial stages, not competing hypotheses. '
            'LIMITS: one model, one task family (dative, active/passive), n=48; the 53 degree boundary '
            'is read off a bin edge (0.6), not measured - only 4 angles were sampled; the readout '
            'criterion uses the entity first token; Delta mixes entity identity with role assignment '
            'and was not decomposed; L33 comes from single-point patch.'
        ),
    },
]

NEGATIVE = {
    'claim': (
        'in Q2-C2 the shared-direction arm S and the residual arm R each carry about half of the '
        'readout effect (S=0.479, R=0.604 at alpha 1.0), and in Q2-C2b the verdict K2 is a valid '
        'test that projection does not explain the effect'
    ),
    'evidence': (
        'Both are invalidated by the same confound: the arms were never length-matched. In Q2-C2 all '
        'arms used displacement alpha*||Delta|| but in DIFFERENT directions, so their length along '
        'Delta differed by construction (S moves alpha*||Delta||/cos, R moves alpha*||Delta||/sin * '
        '|resid| = alpha*||Delta||, A moves alpha*||Delta||); with cos=0.482 the S arm travels 2.07x '
        'further. In Q2-C2b the arms were aligned on PROJECTION p along Delta, which does not align '
        'length either - S again travels p*||Delta||/cos. Diagnostic: the S curve is non-monotone '
        '(0.639 at p=0.75, 0.611 at p=1.0), which is overshoot past the clean point, not extra '
        'efficacy. Correct design is Q2-C2c: hold |displacement| exactly equal and vary only '
        'direction - that run gives D=1.000, R=0.958, M=0.479, Z=0.000 at p=1.0, and stratification '
        'shows M is a mixture of in-gate and out-of-gate samples rather than a half-strength channel. '
        'SECOND DEFECT in the same round: Q2-C2c initially reported M3 "angle-priced continuously" '
        'from the POOLED group means (D>=R>=M>=Z held), but stratification by cos showed the real '
        'shape is a step at cos about 0.6. This is the fifth recurrence of the same error family '
        '(Q1 top-k degenerating to a layer sort; Q2-A2 n=24 CI taken as evidence; Q2-E averaging a '
        'bimodal sample; Q2-F pooling a heterogeneous sample; now judging continuity from pooled '
        'means). D5 restated: any pooled mean must be reported with its stratification over the '
        'variable that could make it a mixture, BEFORE the mean is interpreted. '
        'D7 NEW: direction comparisons must control BOTH length and projection; if an arm scaling '
        'factor depends on a per-sample quantity (1/cos, 1/sin), its distribution and maximum must be '
        'reported, and any arm exceeding a set cap must be dropped rather than run. '
        'D8 NEW: a SMOKE sample must cover every structural dimension of the full sample. triples(4) '
        'returns 4 triples that all share one person pair, so SMOKE cos was 0.936 against 0.245 in '
        'the full run - conclusions from that SMOKE do not extrapolate. Print the value set of each '
        'sampling dimension before entering SMOKE.'
    ),
}


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    have = {(m.get('phase'), m.get('name')) for m in doc['measurements']}
    bak = LEDGER.with_suffix('.json.bak_q2c2')
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
