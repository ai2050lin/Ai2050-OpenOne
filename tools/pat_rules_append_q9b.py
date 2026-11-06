# -*- coding: utf-8 -*-
"""Q9-B 规则账本追加：R-MET-41 / R-MET-42 / R-MTH-33（全部 confirmed）。幂等，同步 LEDGER.md。"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
J = ROOT / 'data' / 'rules' / 'ledger.json'
MD = ROOT / 'data' / 'rules' / 'LEDGER.md'
NOW = datetime.now().isoformat(timespec='seconds')

EV = lambda note: {'phase': 'Q9-B', 'script': 'tests/gpt5/pat_q9b.py',
                   'data': 'tests/gpt5/result/pat_q9b_qwen_patch.json',
                   'model': 'Qwen3-4B', 'note': note}

RULES = [
    {
        'id': 'R-MET-41', 'domain': 'measurement',
        'topic': '顶格探针下禁止用 argmax 翻转率作干预判据',
        'claim': '当读出探针在基线已 1.000 时，"翻转率"按构造恒为 0（没有可翻转的样本），会把强干预误报成零效应；反之在低基线格上翻转率又被天花板放大。',
        'status': 'confirmed', 'confidence': 'high',
        'criterion': '预注册的干预判据不得只用 argmax 翻转率。必须并列报告：①探针决策边际的连续位移 Δmargin（沿目标类的 logit 差）②下一 token 分布的 KL（与基线，逐样本）③top-1 变化率。三者与同族随机方向对照比。翻转率仅在基线 acc ≤ 0.95 的格上作主判。',
        'measure': {'q9b_probe_base': '1.000 (disc & conf, l=35)',
                    'q9b_flip_at_alpha1': '0.022 (MLP +s) / 0.006 (MLP -s)',
                    'q9b_kl_at_alpha1': '0.095 vs null p95 0.003',
                    'q9b_dmargin_alpha4': '+20.11 / -19.93',
                    'q9b_flip_at_alpha2': '0.333（acc 1.000->0.667）'},
        'evidence': [EV('SMOKE 首版只用翻转率，8 个条件全部报 0.000，与"干预完全无效"不可区分；加入 Δmargin/KL 后同一次前向立即显出 30 倍于零分布的效应')],
        'retest': {'status': 'block', 'count': 1, 'note': '判据纪律；任何顶格读出（语域/语言/风格类探针常见）直接适用'},
        'why': 'Q9-A 已记录"顶格 1.0 不比效应大小"，但只写成了散文告警，没有落成判据；Q9-B 第一版就踩了这个坑，说明告警必须升级为禁令。',
        'tags': ['measurement', 'intervention', 'ceiling'],
        'keywords': ['翻转率', '顶格', '天花板', 'KL', 'margin', '干预判据'],
        'created': NOW, 'updated': NOW,
    },
    {
        'id': 'R-MET-42', 'domain': 'measurement',
        'topic': '组件归因须并列 patch 与 ablation，patch 等强不能推出分担相等',
        'claim': '残差流是加性信道：把同一方向加到 attention 输出或 MLP 输出，下游读到的几乎是同一件事。因此 patch 效应的组件差异只反映"注入点离读出多近/该点增益多大"，不反映该组件在日常前向中分担多少写入。用 patch 大小判读写分离必然误判。',
        'status': 'confirmed', 'confidence': 'high',
        'criterion': '组件级因果声明必须两条腿：①patch（充分性：加方向能否推动读出）②ablation（必要性：删投影是否损伤读出），并给随机方向零分布。只做 patch 时结论只能写成"该注入点可推动读出"，不得写成"该组件负责写入"。patch 与 ablation 结论不一致时，以 ablation 为必要性判据、patch 为可及性判据，分开命名。',
        'measure': {'q9b_patch_mlp_alpha1_dmargin': '+5.69', 'q9b_patch_attn_alpha1_dmargin': '+6.20',
                    'q9b_patch_mlp_kl': '0.095', 'q9b_patch_attn_kl': '0.129',
                    'q9b_ablate_mlp_dmargin': '-0.92', 'q9b_ablate_attn_dmargin': '-0.61',
                    'q9b_ablate_both_acc': '0.989 (base 1.000)'},
        'evidence': [EV('W3 预注册判据要求 MLP 翻转率 > 注意力翻转率，实测注意力更强（0.061 vs 0.022），按判据记 ATT_ALSO_WRITES；W2b 联合消融后 acc 仅降 0.011，证明方向对读出是可读非必需')],
        'retest': {'status': 'block', 'count': 1, 'note': '组件归因纪律；后续 SAE 挂载点结论必须同时给 patch 与 ablation'},
        'why': 'Q9-A 的"中段 MLP 主写"是写入份额（raw share），Q9-B 的 patch 实验把它误读成"MLP 是唯一可写入点"。根因是没意识到加性残差流让 patch 与组件身份解耦。',
        'tags': ['measurement', 'intervention', 'component', 'residual'],
        'keywords': ['patch', 'ablation', '读写分离', '残差加性', '组件归因'],
        'created': NOW, 'updated': NOW,
    },
    {
        'id': 'R-MTH-33', 'domain': 'method',
        'topic': '多选题强制选择必须在 prompt 末位读候选字母 logit',
        'claim': '把字母 token 追加到 prompt 之后再取末位 logit，读的是"字母之后的下一个 token"，不是字母本身；该错误会让任何多选题评测塌到随机水平，且看起来像"模型不会做题"。',
        'status': 'confirmed', 'confidence': 'high',
        'criterion': '多选题评分：在 prompt 结束位置（如"答案："之后）取 next-token logits，比较候选字母 token（含带空格变体，取较大者）的 logit。追加候选后再读末位属实现错误。上线前用小样本（n≈30）对已知可答题做一次 sanity check：准确率应显著高于 1/选项数，否则先查评分位置。',
        'measure': {'q9b_wrong_position_acc': '0.327 (144/440) / 0.333 (20/60)',
                    'q9b_correct_position_acc': '0.693 (305/440)',
                    'q9b_gap': '+0.366 绝对准确率'},
        'evidence': [EV('同一批 440 题，仅改评分位置：0.327 -> 0.693；单题诊断显示模型确实在"答案："后直接输出 C/D 等字母')],
        'retest': {'status': 'block', 'count': 1, 'note': '评测实现纪律，所有 MC 基准（C-Eval/MMLU/M3Exam/本地池）通用'},
        'why': 'W6 能力保持门第一版在 440 题上得 0.327，接近随机，若直接采信会得出"模型能力不足"的错误结论，并让能力门失去灵敏度（编辑与基线都贴地板，McNemar 无检验力）。',
        'tags': ['method', 'evaluation', 'multiple_choice'],
        'keywords': ['多选题', 'logit 位置', '强制选择', '能力门', 'sanity check'],
        'created': NOW, 'updated': NOW,
    },
]


def main() -> int:
    doc = json.loads(J.read_text(encoding='utf-8'))
    have = {r['id'] for r in doc['rules']}
    added = []
    for r in RULES:
        if r['id'] in have:
            print('rule %s already present' % r['id'])
            continue
        doc['rules'].append(r)
        added.append(r)
    J.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding='utf-8', newline='')

    if added:
        md = MD.read_text(encoding='utf-8')
        for r in added:
            block = ('\n### %s %s\n\n- **状态**：%s（%s）\n- **命题**：%s\n- **判据**：%s\n- **实测**：%s\n'
                     '- **为什么**：%s\n' % (r['id'], r['topic'], r['status'], r['confidence'],
                                             r['claim'], r['criterion'],
                                             '；'.join('%s=%s' % (k, v) for k, v in r['measure'].items()),
                                             r['why']))
            md = md.rstrip('\n') + '\n' + block
        MD.write_text(md, encoding='utf-8', newline='')

    doc2 = json.loads(J.read_text(encoding='utf-8'))
    print('rules=%d confirmed=%d added=%s' % (
        len(doc2['rules']), sum(1 for r in doc2['rules'] if r['status'] == 'confirmed'),
        [r['id'] for r in added]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
