# -*- coding: utf-8 -*-
"""把 Q8-A 的五条新规则 upsert 进 data/rules/ledger.json（幂等：按 id 覆盖）。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'data' / 'rules' / 'ledger.json'
TS = datetime.now().isoformat(timespec='seconds')

E_QWEN = dict(phase='Q8-A', script='tools/pat_q8a.py',
              data='results/2026-11-01/091106_pat_q8a.json', model='Qwen3-4B')
E_GLM = dict(phase='Q8-A', script='tools/pat_q8a.py',
             data='results/2026-11-01/091311_pat_q8a.json', model='GLM-4-9B')
E_VER = dict(phase='Q8-A', script='tools/pat_q8a_verify.py',
             data='results/2026-11-01/091106_pat_q8a_verify.json', model='Qwen3-4B')

RULES = [
    dict(
        id='R-ATT-27', domain='attention-free static', topic='词表按单位类型分层，生僻字多数无向量',
        claim='词表是一张同质的表，生僻字和其它字一样有自己的一行向量。',
        status='refuted', confidence='high',
        criterion='统计词表行的单位类型（整词 / 单字 / 字节残片 / 其它）与 CJK 基本区覆盖率；'
                  '对字节回退字，其"词向量"只能是 2–3 个字节残片的算术平均，均值池化抹掉字节顺序。',
        measure=dict(qwen_rows=151936, qwen_word=16374, qwen_char=8806, qwen_fragment=1457,
                     qwen_other=125299, qwen_cjk_missing=12404, qwen_cjk_total=20902,
                     qwen_rare_fallback='26/74', glm_cjk_missing=16686, glm_rare_fallback='73/74',
                     within_fragment=0.0298, within_word=0.0062, ratio=4.8),
        evidence=[dict(E_QWEN, note='CJK 基本区 59.3% 缺行；残片 within 是整词 4.8×（超零分布 p95 0.0064，p<0.005）'),
                  dict(E_GLM, note='79.8% 缺行；残片 +0.0100 同样超 p95 0.0088'),
                  dict(E_VER, note='15/15 PASS，含覆盖率与 within 复算')],
        retest=dict(status='block', count=1,
                    note='这是分词器与词表的构造事实，换模型只会换数字不会换结论；换模型必须重报覆盖率'),
        why='「生僻字没有词嵌入」不是修辞：模型拿到的是拼写零件，语义必须在注意力里现拼。'
            '任何关于生僻字「语义几何」的结论都必须先报覆盖率，否则测的是字节表。',
        tags=['embedding', 'tokenizer', 'coverage'], keywords=['生僻字', '字节', '回退', '词表', '分层', '残片'],
    ),
    dict(
        id='R-ATT-28', domain='attention-free static', topic='嵌入表有一根显式频率旋钮',
        claim='频率只能在训练后通过行为间接体现，嵌入表里没有一个可直接读出的频率量。',
        status='refuted', confidence='high',
        criterion='对每个嵌入坐标算它与 zipf 的秩相关，取 |ρ| 最大者；再做 top-k 留出分类曲线，'
                  '对比"原始坐标"与"对 [1,zipf,字节长,log范数] 回归后的残差坐标"。',
        measure=dict(qwen_max_abs_rho=0.802, qwen_coord=27, glm_max_abs_rho=0.774, glm_coord=411,
                     qwen_top5=[0.804, 0.758, 0.764, 0.698, 0.696],
                     raw_k1=0.780, raw_k32=0.751, resid_k32=0.411, resid_full=0.357,
                     glm_raw_k32=0.896, glm_resid_k32=0.531),
        evidence=[dict(E_QWEN, note='单坐标 |ρ|=0.802；4 分类 k=1 达 0.780（机会 0.25）几乎全是频率'),
                  dict(E_GLM, note='0.774，同结构')],
        retest=dict(status='block', count=1,
                    note='两模型同向同量级；若换非 tied 或更小模型需重测（tied 会让范数与输出端耦合）'),
        why='「单参数承载结构」在标量统计量上成立、在语义类别上不成立。看到单坐标高分必须先查它与频率的相关，'
            '否则会把频率轴当成语义轴。',
        tags=['embedding', 'frequency'], keywords=['频率', '单坐标', 'zipf', '旋钮', '标量'],
    ),
    dict(
        id='R-ATT-29', domain='attention-free static', topic='简繁与中英在静态层近乎复本',
        claim='每个汉字在嵌入表里只占一行，参数预算全部花在语义区分上。',
        status='refuted', confidence='medium',
        criterion='对简繁对，算 cos 与"繁体形式在简体形式近邻里的名次"；对中英对，算英文对译 token '
                  '在全部 151936 行里的名次（排除自身），零基线用随机英文词的名次。',
        measure=dict(qwen_st_cos_median=0.797, qwen_st_rank1_frac=0.94,
                     glm_st_cos_median=0.440, glm_st_rank1_frac=0.34,
                     qwen_trans_median_rank=1, qwen_trans_top5=1.00,
                     glm_trans_median_rank=11.5, glm_trans_top5=0.17,
                     random_baseline_rank=67472),
        evidence=[dict(E_QWEN, note='红→紅 0.835、红→` red` 0.799；苹果→` Apple` 0.718；40/40 对译进前 5'),
                  dict(E_GLM, note='近邻是汉语搭配（红的/的红/红了），对译中位名次 11.5')],
        retest=dict(status='allow', count=1,
                    note='两模型差异巨大（0.797 vs 0.440），说明这不是通用规律而是模型指纹；'
                         '需第三架构（DS-7B）与词表规模对照才能定论，材料为手选高频具体词有选择偏差'),
        why='参数不是全花在语义上：Qwen 把多语言折叠到同一个点，这直接解释了 Q6「静态层找不到关系方向」——'
            '方向只能从上下文里长出来。',
        tags=['embedding', 'multilingual'], keywords=['简繁', '繁体', '对译', '英文', '双语', '复本'],
    ),
    dict(
        id='R-MET-32', domain='metric', topic='语义聚集必须报共享字分解',
        claim='两个语义类之间的 within-cosine 差可以直接解释为语义结构。',
        status='refuted', confidence='high',
        criterion='把组内词对按"是否共享汉字"二分，分别报 within；共享字对占比与去掉后的下降比例都要列。',
        measure=dict(color_within=0.1988, color_shared_frac=0.11, color_shared_pairs_cos=0.6029,
                     color_within_no_shared=0.1504, color_drop_frac=0.244,
                     fruit_within=0.0975, fruit_shared_frac=0.07, fruit_shared_pairs_cos=0.4186,
                     fruit_within_no_shared=0.0738, fruit_drop_frac=0.243,
                     freq_matched_random=0.0153),
        evidence=[dict(E_QWEN, note='两族下降比例巧合相同（24.4% / 24.3%），颜色仍是水果 2.0×')],
        retest=dict(status='block', count=1, note='流程条款，与模型无关'),
        why='共享字的词对 cos 高达 0.60（整体 0.20），不分解就会把词形当语义。本轮预注册只对 FRUIT 做了分解，'
            'COLOR 是跑后补的——结论没变属于运气，流程必须固定。',
        tags=['metric', 'control'], keywords=['共享字', '词形', '汉字', '控制', '分解'],
    ),
    dict(
        id='R-MTH-23', domain='method', topic='语义类与频率在真实词表上不可分',
        claim='只要类别可分，就说明嵌入里有该类别的语义骨架。',
        status='refuted', confidence='high',
        criterion='任何"少数坐标/子空间承载类别"的结论，必须先做①逐坐标对 [1, zipf, 字节长, log范数] 回归取残差；'
                  '②频率分层零假设（只在 zipf 十分位内打乱标签）；③报残差曲线是否随 k 单调。',
        measure=dict(qwen_sig_coords_raw=364, qwen_sig_coords_stratified=3,
                     glm_sig_coords_raw=236, glm_sig_coords_stratified=3,
                     resid_peak_k=64, resid_peak_acc=0.415, resid_full_acc=0.357),
        evidence=[dict(E_QWEN, note='频率分层把显著坐标从 364 砍到 3；残差曲线 k=64 见顶后转降'),
                  dict(E_GLM, note='236→3，残差 k=32 0.531')],
        retest=dict(status='block', count=1, note='口径条款；只要材料组的频率分布不同就适用'),
        why='四组材料频率跨 4 个数量级时，一个坐标分开四组等于一个坐标分开频率。'
            '判定必须分两级（D27）：绝对水平看残差，相对差值才谈"还剩多少语义"。',
        tags=['method', 'confound'], keywords=['频率', '混淆', '残差', '分层', '零假设', '坐标'],
    ),
]


def main() -> int:
    doc = json.loads(LEDGER.read_text(encoding='utf-8'))
    rules = doc['rules']
    have = {r['id']: i for i, r in enumerate(rules)}
    added = upd = 0
    for r in RULES:
        r = dict(r)
        r.setdefault('supersedes', [])
        if r['id'] in have:
            i = have[r['id']]
            r['created_at'] = rules[i].get('created_at', TS)
            r['updated_at'] = TS
            rules[i] = r
            upd += 1
        else:
            r['created_at'] = TS
            r['updated_at'] = TS
            rules.append(r)
            added += 1
    doc['updated_at'] = TS
    LEDGER.write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                      encoding='utf-8', newline='')
    st = {}
    for r in rules:
        st[r['status']] = st.get(r['status'], 0) + 1
    rc = {}
    for r in rules:
        k = (r.get('retest') or {}).get('status', 'none')
        rc[k] = rc.get(k, 0) + 1
    print('新增 %d / 更新 %d  共 %d 条' % (added, upd, len(rules)))
    print('status', st)
    print('retest', rc)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
