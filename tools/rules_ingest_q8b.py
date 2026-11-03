# -*- coding: utf-8 -*-
"""把 Q8-B 的三条新规则 upsert 进 data/rules/ledger.json（幂等：按 id 覆盖）。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'data' / 'rules' / 'ledger.json'
TS = datetime.now().isoformat(timespec='seconds')

E_Q = dict(phase='Q8-B', script='tools/pat_q8b.py',
           data='results/2026-11-01/111407_pat_q8b.json', model='Qwen3-4B')
E_G = dict(phase='Q8-B', script='tools/pat_q8b.py',
           data='results/2026-11-01/111608_pat_q8b.json', model='GLM-4-9B')
E_V = dict(phase='Q8-B', script='tools/pat_q8b_verify.py',
           data='results/2026-11-01/111407_pat_q8b.json', model='Qwen3-4B')

RULES = [
    dict(
        id='R-MET-34', domain='measurement', topic='少数高频差组构成材料时，单坐标结论必须全池复核',
        claim='只要材料分成几个频率量级不同的组，任意能分出组的坐标都会给出高相关，'
              '于是"单坐标 / 单参数承载结构"的结论是被材料造出来的。',
        status='confirmed', confidence='high',
        criterion='同一"最强单坐标相关系数"在两个池上算：①由少数几个频率量级不同的组拼成的材料集 '
                  '②全部有频率定义的 token。若①远高于②，则①的数值不得作为"单参数"证据。'
                  '同时必须报全空间方向的 R² 作上限。',
        measure=dict(q8a_materials_n=228, q8a_single_coord_rho=0.802,
                     q8b_pool_n_qwen=24421, q8b_pool_n_glm=27215,
                     q8b_single_coord_pearson_qwen=-0.520, q8b_single_coord_pearson_glm=-0.590,
                     q8b_single_coord_r2=0.195, q8b_full_space_r2_qwen=0.826,
                     q8b_full_space_r2_glm=0.721, gap_x=4.2),
        evidence=[dict(E_Q, note='单坐标 0.520/0.548 vs 全空间 0.826；坐标曲线到 1024 维未饱和'),
                  dict(E_G, note='单坐标 0.590/0.598 vs 全空间 0.721；同结论'),
                  dict(E_V, note='20/20 PASS，含 coord_pearson/spearman 逐位复算')],
        retest=dict(status='block', count=1,
                    note='这是抽样设计问题，不随模型变化；换材料必须重算两个池并同时报'),
        why='Q8-A 的「一根旋钮」就是这条错误的实例：0.802 来自四组材料跨四个数量级的频率差。'
            '项目此前也踩过一次同类（Q6-B 的 share 饱和）。凡"某某只由一个参数承载"的提法，'
            '先问材料是不是分组拼的。',
        tags=['measurement', 'sampling', 'confound'], keywords=['单坐标', '单参数', '材料', '选择偏差', '全池复核'],
    ),
    dict(
        id='R-MTH-24', domain='method', topic='带折的缓存键必须含折划分指纹',
        claim='按 (命名空间, 折序号) 缓存 Gram 矩阵是安全的，因为同一份数据算出来的 Gram 一样。',
        status='refuted', confidence='high',
        criterion='在同一进程里用两套不同的折划分调用同一缓存函数，检查 G_all − G_fold 是否仍等于 '
                  'G_train。或直接比较两套 key 下的结果：若同 X 同 y 同折却给出不同的 R²，即缓存被污染。',
        measure=dict(symptom_r2_cached=0.7757, symptom_r2_fresh=0.8261,
                     culprit='零分布先用 _f0 调 cv_fast(ns="main")，把 (f,main,j) 写成 _f0[j] 的 Gram',
                     fix='key 加折指纹 (len(folds), sum(folds[0]), size(folds[0]))'),
        evidence=[dict(E_Q, note='同 X 同 y 同 folds，仅换命名空间即 0.776→0.826，一步定位'),
                  dict(E_V, note='修复后 20/20 PASS，B1 与 B3 的全维 R² 互相对上（0.8263 vs 0.8261）')],
        retest=dict(status='block', count=1,
                    note='实现层面的陷阱，与模型无关；任何带缓存的重采样流程都适用'),
        why='缓存是为省时间，但它把"这份 Gram 属于哪套划分"这一信息丢掉了，出错时表现为'
            'R² 莫名其妙偏低，而且只在"先跑了别的划分"时才出现 —— 最难查的一类 bug。',
        tags=['method', 'caching', 'resampling'], keywords=['缓存', '折', 'Gram', '污染', 'R²偏低'],
    ),
    dict(
        id='R-MTH-25', domain='method', topic='每个 held-out R² 都要有一条独立代码路径复算并报差值',
        claim='主脚本自己算的 held-out R² 可信，只要打印出来就够了。',
        status='refuted', confidence='high',
        criterion='用另一套实现（不同求解器 / 不同掩码写法 / 不同折划分重复 5 次）复算同一个 R²，'
                  '报出"主脚本值、独立值、5 次划分的极差"。若差值超过极差，必须先当作 bug 排查，'
                  '不得解释为"口径不同"。',
        measure=dict(primary_r2_masked_bug=0.8650, primary_r2_after_fix=0.8263,
                     independent_single_split_mean=0.8343, independent_range=0.018,
                     gap_after_fix=0.008),
        evidence=[dict(E_Q, note='独立实现给 0.842，主脚本 0.776（缓存污染）+ 0.865（掩码反）；两次都靠这条抓到'),
                  dict(E_G, note='修复后主脚本 0.7214 与独立 0.7225 差 0.0011')],
        retest=dict(status='block', count=1,
                    note='流程纪律，与模型无关'),
        why='本轮两个致命 bug 都不是靠"看代码"发现的，是靠"另一个实现给出不同数字"发现的。'
            '差值超过划分极差 = 有系统性错误，这个判据比任何代码 review 都硬。',
        tags=['method', 'verification', 'r2'], keywords=['复核', 'held-out', '独立实现', '差值', '划分极差'],
    ),
]


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    by_id = {r['id']: i for i, r in enumerate(doc['rules'])}
    added, updated = 0, 0
    for r in RULES:
        r = dict(r)
        r['updated'] = TS
        if r['id'] in by_id:
            r.setdefault('created', doc['rules'][by_id[r['id']]].get('created', TS))
            doc['rules'][by_id[r['id']]] = r
            updated += 1
        else:
            r['created'] = TS
            doc['rules'].append(r)
            added += 1
    doc['version'] = doc.get('version', 1) + 1
    doc['updated'] = TS
    LEDGER.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8', newline='')
    st = {}
    for r in doc['rules']:
        st[r['status']] = st.get(r['status'], 0) + 1
    bl = {}
    for r in doc['rules']:
        k = r.get('retest', {}).get('status', 'allow')
        bl[k] = bl.get(k, 0) + 1
    print('added %d updated %d total %d' % (added, updated, len(doc['rules'])))
    print('status', st)
    print('block-policy', bl)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
