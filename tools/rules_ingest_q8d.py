# -*- coding: utf-8 -*-
"""把 Q8-D 的四条新规则 upsert 进 data/rules/ledger.json（幂等：按 id 覆盖）。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / 'data' / 'rules' / 'ledger.json'
TS = datetime.now().isoformat(timespec='seconds')

E_Q = dict(phase='Q8-D', script='tools/pat_q8d.py',
           data='results/2026-11-01/162208_pat_q8d2.json', model='Qwen3-4B')
E_G = dict(phase='Q8-D', script='tools/pat_q8d.py',
           data='results/2026-11-01/162016_pat_q8d2.json', model='GLM-4-9B(8bit)')
E_V = dict(phase='Q8-D', script='tools/pat_q8d_verify.py',
           data='results/2026-11-01/162208_pat_q8d2.json', model='Qwen3-4B')

RULES = [
    dict(
        id='R-MTH-26', domain='method', topic='offload/量化加载下手算 head 前必须有设备与数值断言',
        claim='get_output_embeddings().weight 拿来直接用是安全的，transformers 会把权重放对位置。',
        status='refuted', confidence='high',
        criterion='用手算 logits @ W.T 与模型原生 logits 逐位比较（同一输入）。断言三点：'
                  '① W.device 不是 meta；② max|Δ| 小于该模型 bf16 噪声底；③ ‖手算‖ 与 ‖原生‖ 同量级。'
                  '任一失败立即中止，不得用猜的权重继续跑。',
        measure=dict(symptom='GLM offload 时 lm_head 停在 meta device，值是未初始化垃圾',
                     ref_norm=764, manual_norm=115761, max_abs_diff=1.8e4,
                     outcome='calibrate_head 断言正确中止，零假结论读出',
                     fix='从 model.safetensors.index.json 定位 lm_head 并 safetensors 直读'),
        evidence=[dict(E_G, note='断言触发→中止→修复→重跑，D2 数字全部产自真 head'),
                  dict(E_Q, note='同断言在 Qwen 常规加载下静默通过（真权重），证明断言不是只拦 GLM')],
        retest=dict(status='block', count=1,
                    note='实现陷阱，与本机 offload 配置绑定；任何 device_map=auto 的手算路径都适用'),
        why='meta 权重的值是地址垃圾，不会报错、只给错数。若没有"必须复现原生 logits"的断言，'
            '整轮 D2 会在假 head 上跑完并输出一套看似合理的假 C3/C4 —— 这正是校准断言存在的意义。',
        tags=['method', 'loading', 'assertion'], keywords=['meta', 'offload', 'lm_head', '断言', '手算'],
    ),
    dict(
        id='R-MET-35', domain='measurement', topic='方向特异性对照必须含同流形随机方向',
        claim='干预方向的效应超过各向同性随机方向（RAND）即可声明方向特异性。',
        status='refuted', confidence='high',
        criterion='对照集必须包含 EMB：随机两个真实 token 嵌入之差（在嵌入流形上、语义随机）。'
                  '只有当 FREQ 的 |ΔE| 与 KL 同时超过 EMB 分布（≥p95），且超过 RAND，才判 '
                  'DIRECTION_SPECIFIC；只超过 RAND 不超过 EMB 判 ON_MANIFOLD_ONLY。',
        measure=dict(qwen_freq_absdE=0.00114, qwen_rand_p95=0.0038, qwen_emb_p95=0.0027,
                     qwen_verdict='GENERIC_PERTURBATION',
                     glm_freq_absdE=0.00426, glm_rand_p95=0.0020, glm_emb_p95=0.0023,
                     glm_verdict='DIRECTION_SPECIFIC',
                     wide_grid_impression='KL 20x RAND（无 EMB 对照时的过度印象）'),
        evidence=[dict(E_Q, note='D1 的"KL 20×"加 EMB 对照后被击落'),
                  dict(E_G, note='GLM 过了 EMB 对照 ⇒ 对照有效且可过，不是形同虚设'),
                  dict(E_V, note='两模型判决词经独立脚本复算')],
        retest=dict(status='block', count=1,
                    note='对照设计纪律，与模型无关'),
        why='各向同性 RAND 天生"离开嵌入流形"，任何在流形上的扰动都显得比它特别。'
            'EMB 把"在流形上"这个平庸解释单独扣掉，剩下的才是方向本身的功能性。',
        tags=['measurement', 'control', 'direction-specificity'],
        keywords=['对照', '同流形', 'RAND', 'EMB', '方向特异性'],
    ),
    dict(
        id='R-MET-36', domain='measurement', topic='干预实验的材料闸：末位须为有定义内容词且句法不完整',
        claim='用自然的完整句子测"下一 token 的频率"是合适的材料。',
        status='refuted', confidence='high',
        criterion='闸门只看基线分布、不看干预结果：①p_M（有 zipf 定义 token 上的概率质量）≥0.80；'
                  '②基线熵 H≥1.00（单峰 0.999 的分布没有位移空间）；③prompt 末位是有频率定义的内容词。'
                  '候选模板逐条打分打印，通过集在跑前冻结。',
        measure=dict(syntactic_complete_pM=0.033, punct_mass=0.849,
                     incomplete_pM=0.971, gate='p_M>=0.80 AND H>=1.00',
                     kept_qwen=32, kept_glm=25, n_candidates=36),
        evidence=[dict(E_Q, note='诊断轮逐句打印 top-6 预测，"，"占 0.849 的直接证据'),
                  dict(E_G, note='同一闸门在 GLM 上通过 25 个模板，基线 p_M 0.977')],
        retest=dict(status='block', count=1,
                    note='材料口径纪律；闸门规则可复用，具体模板随模型/语言重筛'),
        why='句法完整的边界处语言模型输出由标点主导，任何 E[zipf] 类主指标在该处按构造失效——'
            '这不是模型性质，是读数位置选错了。不设闸会直接产出"干预无效"的假阴性。',
        tags=['measurement', 'materials', 'readout'], keywords=['材料', '句法', 'p_M', '标点', '读出位置'],
    ),
    dict(
        id='R-MTH-27', domain='method', topic='预注册的门限/网格必须写进代码默认值并打日志',
        claim='预注册文档写了阈值，跑的时候照着配就行。',
        status='refuted', confidence='high',
        criterion='阈值、网格、剂量等预注册量必须：①作为环境变量默认值写死在代码里；'
                  '②运行日志打印实际生效值；③主结果 JSON 里记录。发现文档与注册文档比对时，'
                  '以日志和 JSON 为准，不以"我记得配过"为准。',
        measure=dict(incident='D2 首跑用了 D1 的门限 0.5/0.75，注册值是 0.30/0.60',
                     detection='读 D2 结果明细时发现门限字段对不上',
                     fix='GATE_H/GATE_AGREE 改为按 PHASE 取注册默认值，删除错误产物重跑'),
        evidence=[dict(E_Q, note='首次 D2 产物已删除，重跑结果 162208_pat_q8d2.json 才是注册口径')],
        retest=dict(status='block', count=1,
                    note='流程纪律，与模型无关'),
        why='预注册的意义在"判决规则先于数据"，而规则躺在文档里、代码里另有一套时，'
            '等于没有注册——错误版本跑出的数字一旦进了结论再被引用，纠错成本远高于重跑。',
        tags=['method', 'preregistration', 'process'], keywords=['预注册', '门限', '默认值', '重跑'],
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
