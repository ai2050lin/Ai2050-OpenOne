# -*- coding: utf-8 -*-
"""追加 Q2-C1 一条测量 + 一条 negative（判据 min 跨已知恒零量）。幂等。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

ENTRIES = [
    {
        'phase': 'Q2-C1', 'name': 'pat_q2c1_patch', 'line': 'G',
        'date': '2026-11-01', 'verdict': 'L33_readout_handoff_ablation_vs_patch_disagree',
        'res_sha8': '93daf69b',
        'note': (
            'First activation-patching probe in this line, after six ablation rounds. Design: '
            'same-form entity swap (clean 张三送给了赵六一个苹果 / corrupt 赵六送给了张三一个苹果), '
            '24 triples x 2 forms (active A / passive C) x 36 layers x 6 positions, 144 cells, '
            'behaviour gate 48/48 clean answers agent and 48/48 corrupt answers recipient. '
            'Clean and corrupt are exactly length-matched (T=39/39) with identical entity token '
            'indices, verified by tools/pat_q2c1_probe.py. Metric = answer flip rate '
            '(fraction of 48 runs where the patch makes corrupt answer the clean agent), bounded '
            'in [0,1]; the restore ratio was rejected after SMOKE showed it reaching 1.44 '
            '(patch injects out-of-vocabulary-context forms like 张六, exceeding the clean range). '
            'RESULT: patch of the agent FIRST token flips 1.00 from L0 through L29, decays at '
            'L30/L31/L32 (0.79/0.75/0.58 form A, 0.88/0.79/0.54 form C), and is exactly 0.00 from '
            'L33 on. Patch of the LAST position is 0.00 until L25, rises from L26 (0.04/0.29) to '
            'L29-L32 (0.75/0.75 in A, 0.75/0.79/0.96 in C), and is exactly 1.00 from L33 to L35. '
            'Patch of a THIRD entity (neither agent nor recipient) into the agent position flips '
            '1.00 through L29, decays EARLIER than the original agent (0.71 at L30, 0.46 at L31, '
            '0.21 at L32), and is 0.00 from L33. All three flip at the same layer L33. '
            'INTERPRETATION: the answer is written into the last-position residual stream at L33; '
            'after that the input channel is closed and the output is determined by that position. '
            'L29-L32 is an overlap band where both input and output positions can partially restore. '
            'THE CENTRAL QUANTITATIVE CONTRAST: ablating all 133 heads in L28-L35 (Q2-D2) drops '
            'behaviour by 0.000, while patching one position of the L33 residual stream flips it by '
            '1.00. Same model, same task, same layer band. Ablation measures whether missing parts '
            'can be compensated (capacity, distributed per Q2-H); patching measures whether content '
            'determines output (readout, single point). This reconciles Q2-H with the Q1-Q2 line: '
            'carriage can be distributed and replaceable while readout is single and irreplaceable. '
            'FORM DIFFERENCE: handoff layer is L33 in both forms, but the last position starts '
            'carrying the answer 2-4 layers earlier in the passive form (0.21 at L24 vs 0.00; 0.50 '
            'at L28 vs 0.25), consistent with the agent being closer to the end of the sentence. '
            'Passivisation changes transport distance, not the readout point. '
            'LIMITS: (1) the readout criterion uses the entity first token, and agent is 2 tokens - '
            'patching the second token flips 0.00 at almost every layer, so what is measured is '
            'first-token identity, not full role semantics; (2) L33 is localised by single-point '
            'patch only, no set patch was run, so the point may move under joint patching. '
            'POST-HOC, NOT A VERDICT: token position 17 is agent in form A (flip 1.00) and theme in '
            'form C (flip 0.00 at all 36 layers), i.e. the effect follows the role, not the position. '
            'Confounded because the question asks for the agent, so theme should not be read anyway; '
            'a design where both roles are queried at the same position is required. Recorded as a '
            'lead for Q2-C2, not as evidence.'
        ),
    },
]

NEGATIVE = {
    'claim': (
        'the Q2-C1 shape classifier discriminates the two sentence forms: form A is unclassified '
        'and form C is a clean S1 double-peak (S1 = agent-position flip high early, low mid, last '
        'position high late)'
    ),
    'evidence': (
        'The classifier line a_m = min(band[agent0][1], band[agent1][1]) takes the minimum across '
        'the two agent tokens, but agent1 (the second token of a 2-token name) has flip rate 0.00 '
        'at almost every layer by construction, because the readout uses only the first token. The '
        'min is therefore pinned at 0 and the "mid-band <= 0.40" clause is satisfied unconditionally. '
        'The two forms actually have near-identical profiles: agent-first-token decay L30/L31/L32 = '
        '0.79/0.75/0.58 (A) vs 0.88/0.79/0.54 (C), all within 0.09; both flip at exactly L33. The '
        'different verdicts come only from the late-band last-position mean falling at 0.53 (A) vs '
        '0.65 (C) on opposite sides of the 0.60 threshold - boundary noise, not a real difference. '
        'Correct procedure: judge on agent0 alone and report agent1 separately as a known-zero '
        'control. This is the third preregistration defect in this line and they share one form: '
        'Q2-H criterion direction reversed versus the alternative; Q2-H criterion necessary but not '
        'sufficient; Q2-C1 criterion fed a variable with a known identically-zero component. '
        'D6 addendum: every variable entering a threshold comparison must first be reported as its '
        'own full-layer profile; any "combine several quantities then compare to a threshold" step '
        'can be dominated by one identically-zero term.'
    ),
}


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    have = {(m.get('phase'), m.get('name')) for m in doc['measurements']}
    bak = LEDGER.with_suffix('.json.bak_q2c1')
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
