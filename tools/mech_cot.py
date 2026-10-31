# -*- coding: utf-8 -*-
"""H4：换因变量 —— 从 next-token 分布换到生成任务，测 top-1% 的行为必要性。

H3b 已经把「为什么 KL 测不出必要性」讲清楚：top-97 的扰动只有 0.07% 的能量落在高概率
token 上，几乎不动答案字母。只要因变量还在 next-token 分布上，测到的都是长尾。
所以唯一能问「必不必要」的地方是**行为**：让模型真的生成一段推理和答案。

三条件同一批题：
  base    不干预
  P       删 top-97（|Δ| 最大的 1%）
  full    删整层 9728（上界对照）
  rand97  删随机 97（噪声下界对照，补 H3 里「弱对照」的缺陷）

预注册判据：
  G1 基线可用   base 的格式遵循率 ≥ 0.70（否则任务不成立，本实验作废）
  G2 行为必要   acc(P) ≤ acc(base) − 0.10（绝对降幅 10 个点）
  G3 非全损     acc(P) ≥ acc(full)（整层是上界，1% 不应比 100% 更糟）
  判读：G1+G2 → 必要性在行为层成立，KL 的失败是「因变量选错」，不是「单元不重要」；
        G1 过 G2 不过 → 这些单元对生成行为也不必要，单元级因果路线到此为止；
        G1 不过 → 任务不成形，改用强制格式（few-shot 一条示例）重跑，本轮不入账。

用法：SMOKE=1 .venv/Scripts/python.exe -X utf8 tools/mech_cot.py   # 3 题探格式
      .venv/Scripts/python.exe -X utf8 tools/mech_cot.py            # 40 题正式
"""
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from server import lang_atlas_service as latlas
from server import mech_pair_service as mpair

B = int(os.environ.get('LAYER', 24))
SEED = 20261101
SMOKE = os.environ.get('SMOKE') == '1'
NQ = int(os.environ.get('NQ', 3 if SMOKE else 40))
CONDS = [c for c in os.environ.get('CONDS', 'base,P,rand97,full').split(',') if c]
TAG = os.environ.get('TAG', '')
OUT = Path('tempdata/mech_cot_L%d%s_smoke.json' % (B, TAG) if SMOKE else 'tempdata/mech_cot_L%d%s.json' % (B, TAG))
MAXNEW = int(os.environ.get('MAXNEW', 96))
SUFFIX = '\n请先简短分析理由，最后单独一行写「答案：X」（X 为选项字母）。\n分析：'
PAT = re.compile(r'答案\s*[：:]\s*([A-E])')
PAT2 = re.compile(r'([A-E])\s*(?:项)?\s*(?:是正确的|正确|符合题意|为正确答案)')


def parse(txt):
    """优先显式答案行；没有就退到「X 正确」这类陈述。两者都没有算格式失败，不猜。"""
    m = PAT.findall(txt)
    if m:
        return m[-1]
    m = PAT2.findall(txt)
    return m[-1] if m else None


def build(q):
    prompt, lines, _ = latlas._build_prompt(q)
    return prompt + SUFFIX


def main():
    t_all = time.time()
    model, tok, device, meta, model_name = mpair._wait_ready()

    rows_all = [r for r in latlas._load_questions() if r.get('source')]
    cm = [r for r in rows_all if r['source'] == 'CMMLU test']
    by_sub = {}
    for r in cm:
        by_sub.setdefault(r['subject'], []).append(r)
    rng = np.random.default_rng(SEED)
    spec = [('high_school_biology', 19), ('high_school_geography', 19),
            ('chinese_history', 10), ('world_history', 10),
            ('chinese_civil_service_exam', 47), ('logical', 22),
            ('modern_chinese', 23), ('ancient_chinese', 19)]
    pool = []
    for s, k in spec:
        p = by_sub[s]
        pool += [p[i] for i in rng.choice(len(p), size=k, replace=False)]
    rows = pool[:NQ]

    a = np.load('tempdata/unit_attr_L%d.npz' % B)['attr_sum'].astype(np.float64)
    d_ff = a.size
    allm = {
        'base': None,
        'P': np.isin(np.arange(d_ff), np.argsort(-np.abs(a))[:d_ff // 100]),
        'rand97': np.isin(np.arange(d_ff), np.random.default_rng(SEED + 3).choice(d_ff, size=d_ff // 100, replace=False)),
        'full': np.ones(d_ff, dtype=bool),
    }
    masks = {k: allm[k] for k in CONDS}
    print('ready %.0fs  %d 题' % (time.time() - t_all, len(rows)), flush=True)

    def gen(prompt, mask):
        enc = tok(prompt, return_tensors='pt').to(device)
        hs = []
        if mask is not None:
            msk = torch.as_tensor(mask, device=device)

            def abl(_m, args):
                v = args[0].clone()
                v[..., msk] = 0.0
                return (v,) + tuple(args[1:])
            hs.append(model.model.layers[B].mlp.down_proj.register_forward_pre_hook(abl))
        try:
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False,
                                   pad_token_id=tok.eos_token_id)
            return tok.decode(o[0][enc['input_ids'].shape[1]:], skip_special_tokens=True)
        finally:
            for h in hs:
                h.remove()

    recs = []
    t0 = time.time()
    for qi, q in enumerate(rows):
        prompt = build(q)
        r = {'qi': qi, 'subject': q['subject'], 'gold': q['answer'], 'out': {}}
        for nm, m in masks.items():
            txt = gen(prompt, m)
            r['out'][nm] = {'pred': parse(txt), 'text': txt.strip()}
        recs.append(r)
        print('  %d/%d %.0fs  gold %s  %s'
              % (qi + 1, len(rows), time.time() - t0, q['answer'],
                 '  '.join('%s=%s' % (nm, r['out'][nm]['pred']) for nm in masks)), flush=True)

    def acc(nm):
        return round(float(np.mean([r['out'][nm]['pred'] == r['gold'] for r in recs])), 3)

    def fmt(nm):
        return round(float(np.mean([r['out'][nm]['pred'] is not None for r in recs])), 3)

    summ = {'n': len(recs), 'acc': {nm: acc(nm) for nm in masks},
            'format_rate': {nm: fmt(nm) for nm in masks},
            'changed': {nm: int(sum(r['out'][nm]['pred'] != r['out']['base']['pred'] for r in recs))
                        for nm in masks}}
    G1 = summ['format_rate']['base'] >= 0.70
    G2 = summ['acc']['P'] <= summ['acc']['base'] - 0.10
    G3 = (summ['acc']['P'] >= summ['acc']['full']) if 'full' in masks else None
    out = {'schema': 'mech_cot.v1', 'model_name': model_name, 'block': B,
           'suffix': SUFFIX, 'max_new_tokens': MAXNEW, 'summary': summ,
           'verdict': {'G1_baseline_valid': bool(G1), 'G2_behaviour_necessary': bool(G2),
                       'G3_not_worse_than_full': (None if G3 is None else bool(G3))},
           'conditions': CONDS,
           'records': recs, 'secs': round(time.time() - t_all, 1)}
    out['reading'] = ('必要性在行为层成立，KL 的失败是因变量选错。' if (G1 and G2) else
                      ('格式不成形，本轮不入账，需加 few-shot 重跑。' if not G1 else
                       '生成行为上也不必要：单元级因果路线止步于描述。'))
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    print('verdict', out['verdict'], '|', out['reading'])
    print('written %s %.0fs' % (OUT, time.time() - t_all), flush=True)


if __name__ == '__main__':
    main()
