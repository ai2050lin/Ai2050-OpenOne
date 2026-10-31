# -*- coding: utf-8 -*-
"""P0 地基相冒烟（G5-DP01）：provenance / atom_registry / power_analysis / bench_categories 全过 = P0 机制可用。

零 GPU、零网络。注册表写测试用临时文件，不碰真册。
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools import atom_registry as ar
from tools import bench_categories as bc
from tools import power_analysis as pa
from tools.provenance import check_provenance, provenance, write_result


def main() -> None:
    # 1) provenance：五字段齐全 + write_result 自检
    doc = {'schema': 'x', 'payload': {'k': 1}}
    prov = provenance(seed=42, tool='_p0_smoke')
    doc['provenance'] = prov
    errs = check_provenance(doc)
    assert not errs, errs
    assert prov['git_sha'] != 'unknown' and prov['model_revision'].startswith('sha8:')
    p = write_result('_p0_smoke', {'ok': True}, seed=42, tool='_p0_smoke')
    r = json.loads(p.read_text(encoding='utf-8'))
    assert not check_provenance(r) and len(r['sha8']) == 8
    print('1 provenance ok:', prov['git_sha'][:8], '| dirty =', prov['git_dirty'])

    # 2) atom_registry：校验拦截 + 临时册追加/查重/查询（不碰真册）
    bad = {'atom_id': 'a1', 'kind': 'wrong_kind'}
    assert ar.validate_atom(bad), 'bad atom 应被拦截'
    good = {
        'atom_id': 'smoke-a1', 'kind': 'unit_group', 'layer': 34,
        'units': [[34, i] for i in (100, 101, 102)],
        'method_version': 'census_typing.v1',
        'provenance': provenance(seed=42, tool='_p0_smoke'),
        'activation_stats': {'median': 0.1}, 'template_selectivity': {},
        'causal_tests': [], 'evidence': [{'kind': 'observation', 'ref': 'test'}],
        'counterexamples': [], 'grade': 'candidate',
    }
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / 'reg.json'
        r1 = ar.append(good, path=tmp)
        assert r1['ok'] and r1['n_atoms'] == 1, r1
        r2 = ar.append(good, path=tmp)
        assert not r2['ok'] and '已存在' in r2['errors'][0]
        hits = ar.query(kind='unit_group', layer=34, path=tmp)
        assert len(hits) == 1 and hits[0]['atom_id'] == 'smoke-a1'
        assert ar.query(kind='head_dir', path=tmp) == []
    real = ar.load()
    assert real['schema'] == 'atom_registry.v1' and real['n_atoms'] == 0
    print('2 atom_registry ok: 校验/追加/查重/查询/真册空')

    # 3) power_analysis：规划值与反向检验
    n = pa.n_required(0.03, p_d=0.10)
    assert 800 <= n <= 950, n                     # ≈872，公式回归锚
    chk = pa.check_phase(0.03, 1000)
    assert chk['pass'] is True and chk['achieved_power'] > 0.8
    assert pa.check_phase(0.02, 400)['pass'] is False   # 小效应小 n 必须拦下
    print('3 power ok: n(3pt)=%d, power(1000)=%.3f' % (n, chk['achieved_power']))

    # 4) bench_categories：映射冻结 + 池文件一致 + 验收线
    assert bc.categories_of('ancient_chinese') == ['语法规则']
    assert bc.categories_of('jurisprudence') == ['功能推理']
    assert bc.categories_of('genetics') == ['知识网络']
    assert bc.categories_of('machine_learning') == []    # 未映射科目诚实为空
    man = json.loads((ROOT / 'data/lang_bench/pools_manifest.v1.json').read_text(encoding='utf-8'))
    rows = [json.loads(l) for l in
            open(ROOT / 'data/lang_bench/pools.v1.jsonl', encoding='utf-8')]
    assert len(rows) == man['n_rows']
    assert all(man['per_category'][c]['n'] >= 1000 for c in bc.CATEGORIES), man['per_category']
    for r in rows:
        assert r['categories'] == bc.categories_of(r['subject'])
    print('4 bench ok:', {c: man['per_category'][c]['n'] for c in bc.CATEGORIES})

    print('P0 SMOKE: ALL PASS')


if __name__ == '__main__':
    main()
