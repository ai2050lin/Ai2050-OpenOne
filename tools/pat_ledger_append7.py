# -*- coding: utf-8 -*-
"""追加 Q2-L / Q2-L2 两条测量 + 一条 negative（末位注入测的是线性读出）。幂等。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

ENTRIES = [
    {
        'phase': 'Q2-L', 'name': 'pat_q2l_decompose', 'line': 'G',
        'date': '2026-11-01',
        'verdict': 'delta_effect_is_carried_by_its_lexical_component_target_need_not_be_in_context',
        'res_sha8': '9a3703f8',
        'note': (
            'Decomposition of Delta = h(clean) - h(corrupt) at the Q2-C1 readout point (L33 last '
            'position), separating lexical identity from role assignment. ALL deltas are measured on '
            'and injected into the SAME base sentence B=(Y,X,T) answering Y, so no cross-context '
            'transplant confound (the first design measured the entity delta on S1 but injected into '
            'S2 and was discarded in SMOKE). Cells: P1=(X,Y,T) answering X (role+identity change, '
            'target X is IN context), P2=(Y\',X,T) answering Y\' (identity only, target NOT in '
            'context), P3=(Y,X,T\') answering Y (object changed, answer unchanged, specificity '
            'control). n=44 (24 triples x active/passive, 4 skipped for length), covering 12 person '
            'pairs via a purpose-built build_triples() because triples() only covers 4 (D8). '
            'RESULT at p=1.0 with displacement p*||d1||: d1=0.977, d2=1.000, lex1=1.000, '
            'lex2=0.909, orth1=0.159, orth2=0.000, rand=0.091, d3 natural-length stay=1.000. '
            'Sub-lexical sweep at fixed length: lambda 0/0.25/0.50/0.75/1.00 -> 1.000/1.000/0.795/'
            '0.364/0.159. Angles (median): cos(d1,d2)=0.545, cos(d1,lex1)=0.469, cos(d2,lex2)=0.555. '
            'Norm ratios: ||d2||/||d1||=1.160, ||d3||/||d1||=0.386, ||D_voice||/||d1||=0.564. '
            'TWO FINDINGS: (1) removing the lexical component of d1 collapses the effect to random '
            'level (0.159 vs rand 0.091) - the effect is carried by the lexical projection; '
            '(2) the target entity does NOT need to appear in the context (d2 targets an entity never '
            'mentioned and flips 1.000), so the readout position can be steered to any vocabulary '
            'item. Gate E0 PASS (d1 0.977 matched, 1.000 natural), specificity S0 PASS (d3 natural '
            'length leaves the answer unchanged 1.000; the matched-length over-drive of d3 answers '
            'the object instead, 28/44). '
            'BUT this round\'s own O1 verdict was then judged TRIVIAL by Q2-L2: the effect is '
            'layer-invariant and vocabulary-generic, i.e. it is the linear LM head, not a mechanism.'
        ),
    },
    {
        'phase': 'Q2-L2', 'name': 'pat_q2l2_controls', 'line': 'G',
        'date': '2026-11-01',
        'verdict': 'readout_injection_is_trivial_linear_head_layer_invariant_position_bound',
        'res_sha8': '7dae64c9',
        'note': (
            'Three controls that decide whether the Q2-L lexical flip is a readout-site effect or a '
            'trivial consequence of tied embeddings plus the final norm. Same vector '
            'unit(E[X]-E[Y]) throughout, displacement p*||d1||, n=44, behaviour gate 40/44 both '
            'directions, re-injection of the original value 44/44. '
            '(1) LAYER CONTROL, inject at the last position at layers 20/25/29/31/33/35: '
            '1.000 at EVERY layer, both p=0.75 and p=1.00, with no decay whatsoever. A vector that '
            'carried role information would be washed out by the 15 layers downstream of L20; it is '
            'not, so it is a direct bias on the output, not information. '
            '(2) POSITION CONTROL, same vector injected at the question-first token q0 at L33: '
            '0.091, identical to the random-direction baseline. The effect therefore lives only at '
            'the position the LM head reads - which is a consequence of causal attention, not '
            'evidence of mechanism. '
            '(3) VOCABULARY CONTROL, inject unit(E[object]-E[Y]) which points at a non-answer token: '
            'the model answers the object at 0.818 (p=0.75) and 0.886 (p=1.00). Whichever word\'s '
            'embedding difference you inject, the model says that word. '
            'VERDICT T1 trivial. CONSEQUENCE: the angle-gated cone reported in Q2-C2/C2b/C2c is the '
            'geometry of the linear readout, not a role mechanism; that whole verdict set is '
            'withdrawn. Q2-C2c measured R as the residual against the consensus direction m_hat, '
            'which is only 0.48-aligned with the actually functional lexical axis, so R very likely '
            'retained its lexical component - measuring anisotropy about the wrong axis turns an '
            'axis-choice error into a cone shape. What survives is Q2-C1 (real-activation patch: '
            'agent-position edits flip 1.00 through L32 and 0.00 from L33, last-position patch the '
            'reverse), which locates L33 as the readout convergence point without using embedding '
            'differences. '
            'D9 NEW: any injection experiment at a readout position must report three controls - '
            'vary layer, vary position, vary target token. If any of the three still works, the '
            'intervention cannot be used to infer mechanism.'
        ),
    },
]

NEGATIVE = {
    'claim': (
        'the L33 last-position readout has an angle-gated cone structure with a boundary at '
        'cos about 0.6 (about 53 degrees), the cone is oriented per sample, and there is no shared '
        'role direction at that layer (Q2-C2 / C2b / C2c)'
    ),
    'evidence': (
        'All three conclusions measure the linear LM head, not a mechanism. Q2-L2 injected a single '
        'pure embedding difference unit(E[X]-E[Y]) at the last position and flipped the answer at '
        '1.000 when injecting at ANY of L20, L25, L29, L31, L33, L35 - zero decay across 15 layers, '
        'which no information-carrying vector could survive. Injecting the embedding difference of a '
        'non-answer object made the model answer that object (0.886 at p=1.0). Injecting the same '
        'vector at the question-first token instead of the read position gave 0.091, the random '
        'baseline, i.e. the effect is confined to the position the LM head reads, which causal '
        'attention guarantees. Therefore: the 53 degree boundary is the geometry of token embeddings '
        'relative to the displacement, not a gate; the per-sample "cone orientation" is the lexical '
        'component of each sample\'s own Delta; and the low cross-sample cos (0.245) is driven by the '
        'roughly 76 percent of Delta energy that is sentence-specific and unrelated to the answer '
        '(Q2-L: removing the lexical component leaves a direction as inert as random, 0.159 vs '
        '0.091). Root cause of the original misreading: Q2-C2c took the residual R against the '
        'leave-one-out consensus direction m_hat, but m_hat is only 0.48-aligned with the actually '
        'functional lexical axis, so R almost certainly kept its lexical component and kept working; '
        'removing the lexical component itself (orth1) kills the effect. Measuring anisotropy about '
        'the wrong axis converts an axis-choice error into a cone shape. '
        'SAME ROUND, SELF-RETRACTION: Q2-L reported O1 "fully lexical, the cone is the unembedding '
        'geometry" - correct but trivial, and it is demoted to a description of the apparatus rather '
        'than a finding. '
        'D9 NEW: any injection experiment at a readout position must report three controls - vary '
        'the layer, vary the position, vary the target token. If any one of them still works, the '
        'intervention says nothing about mechanism. This joins D1-D8 in the same family: first ask '
        'under what conditions the number would fail to hold, then ask what it shows.'
    ),
}


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    have = {(m.get('phase'), m.get('name')) for m in doc['measurements']}
    bak = LEDGER.with_suffix('.json.bak_q2l')
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
