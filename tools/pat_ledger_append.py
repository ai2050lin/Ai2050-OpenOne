# -*- coding: utf-8 -*-
"""把 Q1 + Q2-A/A2/A3 四个测量相追加进 atlas ledger（幂等）。

链配方（scripts/atlas_cli.py 头部，p3280_disk_verify v8 实证）：
  head = sha256(json.dumps(doc, ensure_ascii=False, indent=1))[:8]
  其中 doc 为全量账本、且 ledger_sha256_8 字段置为【上一链头】；
  每条 measurement 的 prev_post == 加入它之前的链头。

因此逐条：写 prev_post -> append -> ledger_sha256_8 置为上一链头 -> 重算 -> 更新。
幂等：同名 (phase, name) 已存在则整条跳过，不重算链。
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

ENTRIES = [
    {
        'phase': 'Q1', 'name': 'pat_q1_unbind_role_address', 'line': 'G',
        'date': '2026-11-01', 'verdict': 'role_address_confirmed',
        'res_sha8': '4fd1bb45',
        'note': (
            'Role unbinding in qwen3-4b: forced-choice accuracy 0.9722 (70/72 cells, 24 triples '
            'x active/passive x 3 queried roles); behavioral gate requires chat template + '
            'system "output one word" + enable_thinking=False, bare-text prompt gives 0/12 and '
            'would have produced a false negative. Attention: 263/1152 heads put argmax mass on '
            'the queried role (permutation null 0.344), layers L17-L35, median layer 28 (readout '
            'era), consistent with 2831 (query happens at the read end). Single head L28H0: '
            'agent 0.661/0.704, recipient 0.843/0.913, theme 0.941/0.875 across active/passive. '
            'Caveat recorded at write time: top-k extraction degenerates to layer-ascending '
            'because hit rates tie at 1.0. CAUSAL CLAIM RETRACTED by Q2-A; see negatives.'
        ),
    },
    {
        'phase': 'Q2-A', 'name': 'pat_q2a_ablation_density_confound', 'line': 'G',
        'date': '2026-11-01', 'verdict': 'density_confound_resolved',
        'res_sha8': '84ff992d',
        'note': (
            '2x2 factorial (passed-filter x low-layer) at k=30/60/100, 144 cells, paired bootstrap '
            '1000x. k=60: D glob_rand 0.990, E high_ok (L31-L35) 1.000, C mid_ok 0.986, '
            'B low_rand (layer-matched) 0.810, A low_ok (L17-L24) 0.708. B-A +0.101 '
            '[+0.013,+0.186]; E-A +0.292 [+0.222,+0.368]; D-C +0.004 [-0.006,+0.014]. '
            'The Q1 gap +0.389 is ablation density: ok_all[:60] covers 23% of L17-L24 while 60 '
            'global random heads cover 5.2% (4.5x). Density-matched gap drops to 0.101. '
            'E beats global random while being the most attention-aligned segment.'
        ),
    },
    {
        'phase': 'Q2-A2', 'name': 'pat_q2a2_band_scan', 'line': 'G',
        'date': '2026-11-01', 'verdict': 'band_effect_explained_by_size',
        'res_sha8': 'e5c2a798',
        'note': (
            'Whole-band ablation, 144 cells. L21-L24 71 ok heads 0.778 vs same-band 71 random '
            'heads 0.806, diff -0.028 [-0.036,+0.094] NOT significant. L29-L32 (69 ok) and '
            'L33-L35 (45 ok) ablate to exactly 1.000. Null result P1-prime: single-band drop is '
            'explained by ablation size, not by role-following. SMOKE at n=24 cells had shown '
            '0.667 vs 0.958 (+0.292); it vanished at n=144 - dry runs verify plumbing only.'
        ),
    },
    {
        'phase': 'Q2-A3', 'name': 'pat_q2a3_layerwise_ablation', 'line': 'G',
        'date': '2026-11-01', 'verdict': 'L0_only_layer_effect_partial_vs_full_divergence',
        'res_sha8': '9759a687',
        'note': (
            'Constant-density probe: ablate all 32 heads of exactly one layer (2.78% of 1152, '
            '100% within layer), 36 layers, 144 cells. Only L0 drops (0.153 to 0.847); 35/36 '
            'layers <=0.007, 31 layers exactly 0.000. Yet 71 heads spread over 4 layers at 55% '
            'within-layer ablation drop 0.222. Conclusion: head-ablation damage is non-monotone '
            'in coverage - full ablation lets the residual stream pass cleanly, partial ablation '
            'breaks the layer direction structure. Damage tracks (number of partially weakened '
            'layers x weakening ratio), not whether heads were attention-aligned. Cumulative: '
            '<=L3 0.326, <=L35 0.333 - single layer compensable, three-plus layers not.'
        ),
    },
]

NEGATIVE = {
    'claim': (
        'the 263 attention heads whose argmax mass follows the queried role constitute a causal '
        'substrate for role unbinding in qwen3-4b (preregistered reading of the Q1 dose curve: '
        'ablating 60 of them drops accuracy 1.0->0.597 while 60 random heads are unharmed)'
    ),
    'evidence': (
        'Q2-A (res 84ff992d) and Q2-A2 (res e5c2a798) and Q2-A3 (res 9759a687), 144 cells each. '
        '(1) Density match: 60 layer-matched random heads 0.810 vs 60 ok heads 0.708, gap +0.101 '
        '[-0.013 widened to +0.186] versus the original +0.389 - three quarters of the effect was '
        'coverage, since ok_all[:60] occupies 23% of L17-L24 while global random occupies 5.2%. '
        '(2) Whole-band: 71 L21-L24 ok heads 0.778 vs 71 same-band random heads 0.806, diff '
        '-0.028, CI [-0.036,+0.094] - the band effect belongs to ablation size, not to filtering. '
        '(3) Constant density: ablating one full layer of 32 heads costs <=0.007 in 35 of 36 '
        'layers, and the most attention-aligned segment L29-L35 (100 heads, argmax mass 0.87-0.94) '
        'costs exactly 0.000, which is LESS than 100 global random heads (0.924). '
        'Hence alignment and causal load are negatively related in this task, and the Q1 causal '
        'claim is withdrawn. Standing lesson: any top-k head set used in an ablation must report '
        'its layer histogram; and full vs partial ablation are different interventions whose '
        'orderings can invert.'
    ),
}


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    ms = doc['measurements']
    have = {(m.get('phase'), m.get('name')) for m in ms}

    bak = LEDGER.with_suffix('.json.bak_q2a')
    if not bak.exists():
        shutil.copy2(LEDGER, bak)
        print('备份 ->', bak.name)

    if not any(n.get('claim') == NEGATIVE['claim'] for n in doc['negatives']):
        doc['negatives'].append(NEGATIVE)
        print('negatives +=1 (共 %d)' % len(doc['negatives']))

    added = 0
    for e in ENTRIES:
        if (e['phase'], e['name']) in have:
            print('跳过（已存在）%s %s' % (e['phase'], e['name']))
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

    if added or doc is not None:
        LEDGER.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                          encoding='utf-8', newline='')
    print('n=%d  head=%s  新增=%d' % (len(doc['measurements']), doc['ledger_sha256_8'], added))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
