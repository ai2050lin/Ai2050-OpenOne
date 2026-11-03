# -*- coding: utf-8 -*-
"""Q9-A 规则账本追加：R-MET-37/38/39 + R-MTH-28（全部 confirmed）。幂等，同步更新 LEDGER.md。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / 'data' / 'rules' / 'ledger.json'
MD = ROOT / 'data' / 'rules' / 'LEDGER.md'
NOW = datetime.now().isoformat(timespec='seconds')

COMMON_EV = lambda model, note: {
    'phase': 'Q9-A', 'script': 'tests/gpt5/pat_q9a.py',
    'data': ('results/2026-11-01/191456_pat_q9a_qwen.json' if 'Qwen' in model
             else 'results/2026-11-01/192208_pat_q9a_glm.json'),
    'model': model, 'note': note}

RULES = [
    {
        'id': 'R-MET-37', 'domain': 'measurement',
        'topic': '组件归因必须同时报原始写入份额与单位范数效率',
        'claim': '只报 |proj(attn)|:|proj(mlp)| 的原始份额，会把"某组件输出范数天然更大"误读成"它承载更多语义"（K-A 混淆的组件版）；反之只报归一化效率会丢掉因果写入份额。',
        'status': 'confirmed', 'confidence': 'high',
        'criterion': '组件归因（沿某语义轴的 attn_out/mlp_out 投影）必须并列报告：①原始份额 raw share（写入残差流的责任比，因果口径）②单位范数效率 |proj|/‖comp‖（排除范数混淆的口径）。两个口径的判决分开命名；不一致时按 D27 分级表述，不得只引其一。',
        'measure': {
            'q9a_mid_band_raw_qwen': '0.39/0.25 (zh/en)', 'q9a_mid_band_raw_glm': '0.17/0.11',
            'q9a_mid_band_normed_qwen': '0.74/0.50', 'q9a_mid_band_normed_glm': '0.43/0.33',
            'q9a_last_band_raw_glm': '0.93/0.80', 'q9a_last_band_normed_glm': '1.38/1.37',
            'q9a_last_band_raw_qwen': '0.32/0.26', 'q9a_last_band_normed_qwen': '1.07/0.61'},
        'evidence': [
            COMMON_EV('Qwen3-4B', '中段 MLP 主写在两口径下都成立；末段 GLM 注意力反超也是两口径一致'),
            COMMON_EV('GLM-4-9B', '同；归一化后外围层带比值回到 ~1 量级，只有中段保持 <0.5')],
        'retest': {'status': 'block', 'count': 1,
                   'note': '口径纪律，不随模型变化；新组件归因实验直接沿用双口径'},
        'why': 'Q3280 的 K-A/K-A2 已经在"静态中继头"上踩过一次绝对范数混淆；Q9-A 在组件级再次遇到：GLM 末段 raw 0.93 看似注意力强主导，归一化 1.38 才知道是每单位范数效率高而非总量碾压；反过来中段 MLP 的 raw 0.11 若只看归一化会被低估其写入责任。',
        'tags': ['measurement', 'component', 'confound'],
        'keywords': ['组件归因', '范数', '份额', 'attn', 'mlp', '双口径'],
        'created': NOW, 'updated': NOW,
    },
    {
        'id': 'R-MTH-28', 'domain': 'method',
        'topic': '残差流采集必须先用恒等式断言定位 hidden_states 每项的身份',
        'claim': 'transformers 的 output_hidden_states 元组结构随版本/模型变化，想当然地把末项当末层残差输出会静默污染采集。',
        'status': 'confirmed', 'confidence': 'high',
        'criterion': '任何 hook 采集管线，运行时必须断言 resid_in[l+1] = resid_in[l] + attn_out[l] + mlp_out[l]（逐样本逐层，相对容差按存储精度写明）。断言失败先诊断每项范数与 final-norm 关系，不得放宽阈值硬闯。本机实测：Qwen3-4B/GLM-4-9B 的 hidden_states 元组长 L+1，末项是 final_norm 之后的输出（cos(hss[L], norm(resid_out[L-1]))=0.9999967），不是末层残差输出；fp16 存储下恒等式地板 rel≈5e-3。',
        'measure': {
            'q9a_identity_rel_max_qwen': 0.005509, 'q9a_identity_rel_max_glm': 0.005718,
            'hss_last_is_post_final_norm_cos': 0.9999967,
            'diag_script': 'tests/gpt5_temp/q9a_diag.py'},
        'evidence': [
            COMMON_EV('Qwen3-4B', '首跑恒等式 FAIL（rel=7.995）定位到 hss[36]=post-norm；修正后 0.0055'),
            COMMON_EV('GLM-4-9B', '修正后同口径通过 0.0057（8-bit）')],
        'retest': {'status': 'block', 'count': 1,
                   'note': '版本相关的坑：transformers 升级后必须重验，而不是默认仍成立'},
        'why': '第一次 SMOKE 恒等式 rel=7.995 直接拦截了这个静默错误：若不断言，所有层的"resid_out"数据在末层会混入 post-norm 值，P3 组件归因在末段全部失真——恰好是本轮发现 GLM 末段注意力反超的位置，错误会直接制造一条假规律。',
        'tags': ['method', 'collection', 'assertion'],
        'keywords': ['hidden_states', '残差流', '恒等式', 'final_norm', 'hook'],
        'created': NOW, 'updated': NOW,
    },
    {
        'id': 'R-MET-38', 'domain': 'measurement',
        'topic': '迁移/读出层位禁止取 argmax(acc) 的首个饱和层，必须报逐层曲线',
        'claim': '探针准确率先 saturate 的层不是机制层位；在饱和层上做的迁移/归因结论会系统性偏早。',
        'status': 'confirmed', 'confidence': 'high',
        'criterion': '任何"在第 l 层做 X"的结论，l 必须有独立于准确率饱和的依据（预注册层位、或全剖面曲线上的结构性拐点），并附逐层数值。禁止 argmax(acc_resid) 作为层位选择器（np.argmax 返回首个最大值=首个饱和层）。',
        'measure': {
            'q9a_argmax_layer_qwen': 1, 'q9a_argmax_layer_glm': 2,
            'q9a_transfer_abstract_by': 'l~8（两模型 NC 曲线）',
            'q9a_glm_late_decay': 'en2zh 1.0→0.5 (l=28..40)',
            'q9a_lr_at_argmax': 'Qwen PARTIAL / GLM ABSTRACT（均错）'},
        'evidence': [
            COMMON_EV('Qwen3-4B', 'argmax l=1 处 zh2en=0.500，l=8 起 1.000'),
            COMMON_EV('GLM-4-9B', 'argmax l=2 处判 ABSTRACT，漏掉 l>=28 的 en2zh 塌陷'),
            {'phase': 'Q9-A', 'script': 'tests/gpt5/pat_q9a_verify.py',
             'data': 'tests/gpt5/result/pat_q9a_transfer_curve.json', 'model': 'both',
             'note': '逐层迁移曲线是本轮真正可信的 V3 载体'}],
        'retest': {'status': 'block', 'count': 1,
                   'note': '与 R-MET-30/D25 同族：剖面必须全层打印并保留'},
        'why': '主脚本 V3 用 argmax 选层得到 Qwen PARTIAL / GLM ABSTRACT，两个都错：迁移是层依赖量（0.5→1.0→GLM 回落 0.5）。若只看主脚本判决，会得出"Qwen 语域语言绑定、GLM 抽象"的假跨模型对比——方向恰好反了一半。',
        'tags': ['measurement', 'layer-selection', 'profile'],
        'keywords': ['层位选择', 'argmax', '饱和层', '迁移曲线', '剖面'],
        'created': NOW, 'updated': NOW,
    },
    {
        'id': 'R-MET-39', 'domain': 'measurement',
        'topic': '跨语言/跨域迁移结论必须双分类器口径并报分歧',
        'claim': '判别式线性探针（LR）与生成式最近质心（NC）在同一层同一材料上可以给出相反的迁移结论；单口径的迁移判决不可引用。',
        'status': 'confirmed', 'confidence': 'high',
        'criterion': '迁移实验（train 域 → test 域）至少用两个口径：判别式（LogisticRegression 类）与生成式（最近质心/余弦到类均值）。两个口径的数值并列报告；分歧 >0.05 时判决记口径依赖，不得取单一数值进结论。',
        'measure': {
            'q9a_zh2en_l36_lr': 0.500, 'q9a_zh2en_l36_nc': 1.000,
            'q9a_gap_direction': 'LR 在 OOD 语言上把判别面拧向语言方向',
            'q9a_within_agreement': '语言内两口径一致（均 1.0）'},
        'evidence': [
            COMMON_EV('Qwen3-4B', 'l=36（NC 全 1.0）vs 主脚本 l=1（LR zh2en 0.50）'),
            COMMON_EV('GLM-4-9B', '同形分歧；曲线文件双口径可复算')],
        'retest': {'status': 'block', 'count': 1,
                   'note': '口径纪律；新增迁移类实验直接内置双口径'},
        'why': 'Q9-A 主脚本（LR）与独立复核（NC，R-MTH-25 要求的独立路径）在同一数据上给出相反的 V3 判决（PARTIAL vs ABSTRACT）。这不是谁的 bug：LR 的决策面在 OOD 域上被语言分量主导，NC 的余弦到质心对它稳健。分歧本身就是发现（语言分量在中段仍在线性可分面里占权重），只报一个口径会把它扔掉。',
        'tags': ['measurement', 'transfer', 'classifier'],
        'keywords': ['迁移', '分类器口径', '最近质心', '逻辑回归', 'OOD'],
        'created': NOW, 'updated': NOW,
    },
]


def main() -> int:
    doc = json.loads(J.read_text(encoding='utf-8'))
    have = {r['id'] for r in doc['rules']}
    added = [r for r in RULES if r['id'] not in have]
    doc['rules'].extend(added)
    doc['updated'] = NOW
    doc['version'] = doc.get('version', 0) + (1 if added else 0)
    J.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8', newline='')

    # 重建 MD（已证实节插条目 + 改计数）
    md = MD.read_text(encoding='utf-8')
    for r in added:
        ev = '\n'.join(f"- 证据：{e['phase']} ｜ `{e['script']}` ｜ `{e['data']}`"
                       for e in r['evidence'])
        key = '，'.join(f'{k}={v}' for k, v in r['measure'].items())
        block = (f"\n### {r['id']} {r['topic']}\n\n"
                 f"- 领域：{r['domain']} ｜ 置信：{r['confidence']} ｜ 已测 1 次 ｜ 不要重测\n"
                 f"- 判据：{r['criterion']}\n"
                 f"- 关键数字：{key}\n"
                 f"- 为什么：{r['why']}\n"
                 f"- 复查：{r['retest']['note']}\n"
                 f"{ev}\n")
        anchor = '\n## 已否证（85）'
        assert anchor in md, 'MD anchor missing'
        md = md.replace(anchor, block + anchor, 1)
    n_conf = sum(1 for x in doc['rules'] if x['status'] == 'confirmed')
    md = md.replace('## 已证实（24）', f'## 已证实（{n_conf}）', 1)
    MD.write_text(md, encoding='utf-8', newline='')

    chk = json.loads(J.read_text(encoding='utf-8'))
    import collections
    c = collections.Counter(x['status'] for x in chk['rules'])
    print('rules now:', len(chk['rules']), dict(c), '| added:', [r['id'] for r in added])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
