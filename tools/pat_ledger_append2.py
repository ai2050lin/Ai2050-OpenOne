# -*- coding: utf-8 -*-
"""追加 Q2-D / Q2-D2 两条测量 + 一条 negative（倒 U 证伪）。幂等。"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'research' / 'gpt5' / 'atlas' / 'atlas_ledger.json'

ENTRIES = [
    {
        'phase': 'Q2-D', 'name': 'pat_q2d_partial_curve', 'line': 'G',
        'date': '2026-11-01', 'verdict': 'inverted_U_falsified_monotone',
        'res_sha8': '288a1504',
        'note': (
            'Preregistered inverted-U test of partial vs full ablation, 144 cells, 5 draws. '
            'Band L21-L24 (4 layers), within-layer fraction p: drop 0.000 / 0.022 / 0.117 / '
            '0.239 / 0.451 for p = 0 / 0.25 / 0.5 / 0.75 / 1.0. Strictly monotone, no falloff at '
            'p=1; inverted-U prediction FALSIFIED. Cold band L29-L32: <=0.0014 at every p. '
            'Consequence: the Q2-A3-derived claim "partial ablation hurts more than full" is '
            'withdrawn - it came from comparing 71 heads over 4 layers against 32 heads in 1 '
            'layer and reading the difference as coverage rather than extent. Correct quantity is '
            '(layers touched x within-layer ratio), with layer count dominating: same 32 heads, '
            '4 layers 0.022 vs 1 layer <=0.007.'
        ),
    },
    {
        'phase': 'Q2-D2', 'name': 'pat_q2d2_bandfull_ablation', 'line': 'G',
        'date': '2026-11-01', 'verdict': 'broad_band_carrier_antiphase_to_alignment',
        'res_sha8': 'e7999282',
        'note': (
            'Whole-band full ablation, 9 bands x 128 heads (11.1% of 1152), 144 cells, 3 draws, '
            'sd 0.000 throughout. Drop by band L0-L3 0.674, L4-L7 0.125, L8-L11 0.306, L12-L15 '
            '0.486, L16-L19 0.660, L20-L23 0.319, L24-L27 0.076, L28-L31 0.000, L32-L35 0.000. '
            'Two findings. (1) Carriage is a BROAD double-peaked band spanning L0-L23, not one '
            'layer and not L21-L24 alone; it reaches exactly zero at L28. The earlier claim that '
            'only L0 carries anything was an artifact of single-layer ablation - one layer of 32 '
            'heads (2.8%) is compensable by the other 35, so only whole-band ablation (11.1%) '
            'reveals layer position. (2) CARRIAGE IS ANTIPHASE TO ALIGNMENT: 260 of the 263 '
            'role-aligned heads sit in L20-L35, whereas L28-L35 (133 aligned heads) contributes '
            'exactly 0.000. Peak separation between the two profiles is about 20 layers. This is '
            'the sharpest form of "alignment is not causation" measured so far: not weak '
            'correlation (rho=0.106) but phase inversion on the layer axis. Hard qualifier on '
            '3280/3281 relay family: layers 23-30 are relay heads for write share but carry zero '
            'causation for role unbinding; quoting "layers 23-30 matter" without a task qualifier '
            'is now an error.'
        ),
    },
]

NEGATIVE = {
    'claim': (
        'partial attention-head ablation damages the task more than full ablation of the same '
        'layers, so the drop should follow an inverted U in within-layer ablation fraction p '
        '(preregistered Q2-D prediction: drop maximal near p=0.5 and falling back at p=1, because '
        'full ablation lets the residual stream bypass cleanly while partial ablation corrupts '
        'layer direction structure without giving a bypass)'
    ),
    'evidence': (
        'Q2-D (res 288a1504), 144 cells, 5 draws, band L21-L24: drop is strictly monotone in p - '
        '0.000, 0.022, 0.117, 0.239, 0.451 for p = 0, 0.25, 0.5, 0.75, 1.0 - with the maximum at '
        'p=1.0 and no falloff; cold band L29-L32 stays at or below 0.0014 for every p. The '
        'inverted U does not exist. The observation that motivated it (Q2-A3: 71 heads over 4 '
        'layers drop 0.222 while one full layer of 32 heads drops <=0.007) is a difference of '
        'EXTENT, not of coverage: layers touched, not fraction within a layer, is the dominant '
        'variable. This also corrects the closing sentence of negatives #14, which suggested that '
        'full and partial ablation orderings can invert; they do not invert here, they differ in '
        'scale by extent only. Standing rule D3 restated: convert to (layers touched, '
        'within-layer ratio) before comparing any two ablation results.'
    ),
}


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    have = {(m.get('phase'), m.get('name')) for m in doc['measurements']}
    bak = LEDGER.with_suffix('.json.bak_q2d')
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
