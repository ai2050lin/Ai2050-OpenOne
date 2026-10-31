# -*- coding: utf-8 -*-
"""P0.4 建池脚本（G5-DP01）：从 questions.v2.jsonl 按 tools.bench_categories 冻结映射
构建三大类功效池 pools.v1.jsonl + pools_manifest.v1.json。

验收：每类 n≥1,000；映射、科目清单、种子说明入册（manifest 带 provenance）。
C-Eval val 三类保留为外部锚，不进池。
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import bench_categories as bc            # noqa: E402
from tools import provenance                        # noqa: E402

SRC = ROOT / 'data' / 'lang_bench' / 'questions.v2.jsonl'
POOLS = ROOT / 'data' / 'lang_bench' / 'pools.v1.jsonl'
MANIFEST = ROOT / 'data' / 'lang_bench' / 'pools_manifest.v1.json'


def main() -> int:
    rows = []
    cat_counter = {c: Counter() for c in bc.CATEGORIES}          # category -> subject -> n
    seen = set()
    with open(SRC, encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            if not q['source'].startswith('CMMLU'):
                continue
            cats = bc.categories_of(q['subject'])
            if not cats:
                continue
            if q['id'] in seen:
                continue
            seen.add(q['id'])
            rows.append({'id': q['id'], 'source': q['source'], 'subject': q['subject'],
                         'subject_cn': q['subject_cn'], 'categories': cats})
            for c in cats:
                cat_counter[c][q['subject']] += 1

    POOLS.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows),
                     encoding='utf-8', newline='')

    counts = {c: sum(cat_counter[c].values()) for c in bc.CATEGORIES}
    core_counts = {}
    for c, subs in bc.CORE_ONLY.items():
        core_counts[c] = sum(cat_counter[c][s] for s in subs)

    manifest = {
        'schema': 'pools_manifest.v1',
        'map_version': bc.VERSION,
        'frozen_at': bc.FROZEN_AT,
        'seed_note': bc.SEED_NOTE,
        'mapping': {s: v for s, v in sorted(bc.MAP.items())},
        'per_category': {c: {'n': counts[c],
                             'n_core_only': core_counts[c],
                             'subjects': dict(cat_counter[c])}
                         for c in bc.CATEGORIES},
        'n_rows': len(rows),
        'provenance': provenance.provenance(seed=None, tool='scripts/build_pools_v1.py'),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=1) + '\n',
                        encoding='utf-8', newline='')

    # P0.1 验收联动：manifest 副本带 provenance 入 results/
    p = provenance.write_result('pools_manifest_v1', manifest, seed=None,
                                tool='scripts/build_pools_v1.py')

    ok = all(counts[c] >= 1000 for c in bc.CATEGORIES)
    print('每类题数:', counts)
    print('仅核心科目（灵敏度）:', core_counts)
    print('池行数:', len(rows), '| manifest 副本:', p.name)
    print('验收（每类≥1000）:', 'PASS' if ok else 'FAIL')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
