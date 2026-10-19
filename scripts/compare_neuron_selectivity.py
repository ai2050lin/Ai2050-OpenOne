# -*- coding: utf-8 -*-
"""比较多个模型的 neurons.v1 结果（同一模式族、同一因子对照）：按相对深度三段汇总。

每个模型、每个读出位置、每段（块号 / 块数 ∈ [0, 1/3)、[1/3, 2/3)、[2/3, 1]）报告：
  - 入选单元占该段全部单元的比例；
  - 按 |d| 排序的前 20 名、按残差贡献排序的前 20 名中主导符号区的分布；
  - 入选单元在 down_proj 残差均值差中的可加份额，以及全部单元按主导符号区的份额。
只比较各模型内部的统计结构，不对齐不同模型的单元或坐标。

用法：python scripts/compare_neuron_selectivity.py --inputs a.json b.json c.json --out research/neurons/compare.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REG = ('++', '--', '+-', '-+')
SEGS = (('浅', 0.0, 1 / 3), ('中', 1 / 3, 2 / 3), ('深', 2 / 3, 1.01))


def summarize(res):
    blocks = sorted({t['block'] for t in res['tags']})
    nb = max(blocks) + 1
    out = {}
    for r in sorted({t['readout'] for t in res['tags']}):
        rows = {}
        for name, lo, hi in SEGS:
            tags = [t for t in res['tags'] if t['readout'] == r and lo <= t['block'] / nb < hi]
            if not tags:
                continue
            n_units = sum(t['n_units'] for t in tags)
            row = {'blocks': [tags[0]['block'], tags[-1]['block']],
                   'selected_frac': sum(t['n_selected'] for t in tags) / n_units,
                   'peak_block': max(tags, key=lambda t: t['n_selected'])['block'],
                   'top_d_regime': {k: sum(u['dominant_regime'] == k for t in tags for u in t['units']) for k in REG}}
            if res.get('contribution'):
                row['top_contrib_regime'] = {k: sum(t['top_contrib_dominant_regime'][k] for t in tags) for k in REG}
                row['share_selected_mean'] = sum(t['share_selected'] for t in tags) / len(tags)
                row['share_by_regime_all_mean'] = {k: sum(t['share_by_regime_all'][k] for t in tags) / len(tags) for k in REG}
                row['share_by_regime_selected_mean'] = {k: sum(t['share_by_regime_selected'][k] for t in tags) / len(tags) for k in REG}
            rows[name] = row
        out[r] = rows
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--inputs', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)
    comp = {'schema': 'neurons_compare.v1', 'models': {}}
    for p in args.inputs:
        res = json.loads(Path(p).read_text(encoding='utf-8'))
        key = res.get('model_key', Path(p).stem)
        comp['models'][key] = {'source': p, 'tm_id': res.get('tm_id'), 'version': res.get('version'),
                               'factor': res['factor'], 'hi': res['hi'], 'lo': res['lo'], 'd_ff': res.get('d_ff'),
                               'segments': summarize(res)}
    Path(args.out).write_text(json.dumps(comp, ensure_ascii=False, indent=1), encoding='utf-8')

    def pct(d):
        tot = max(sum(d.values()), 1)
        return '/'.join(f"{100 * d[k] / tot:.0f}" for k in REG)

    for key, m in comp['models'].items():
        print(f'== {key}（d_ff={m["d_ff"]}；符号区顺序 ++/--/+-/-+，单位 %）')
        for r, rows in m['segments'].items():
            for name, row in rows.items():
                line = (f"  {r:12s} {name} B{row['blocks'][0]}–{row['blocks'][1]} 入选 {100 * row['selected_frac']:.1f}% "
                        f"峰值 B{row['peak_block']} | 前20(|d|) {pct(row['top_d_regime'])}")
                if 'top_contrib_regime' in row:
                    line += (f" | 前20(贡献) {pct(row['top_contrib_regime'])} | 入选份额 {row['share_selected_mean']:+.2f}"
                             f" | 全部单元份额 " + '/'.join(f"{row['share_by_regime_all_mean'][k]:+.2f}" for k in REG))
                print(line)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
