# -*- coding: utf-8 -*-
"""H2：为什么删 1% 比删整层更狠 —— 「平衡假说」的直接检验。

现象（C5' + H1 双重复现）：
  C-Eval   删 top-97  KL 0.17804  翻 5/169     删整层 KL 0.17081  翻 ? 
  CMMLU    删 top-97  KL 0.04764  翻 8/169     删整层 KL 0.04765  翻 5/169
1% 的单元与 100% 的单元打出相同的分布扰动，而且翻转更多。两种解释：
  H_bal  平衡破坏：top-97 是被 99% 的单元对冲着的净正杠杆，只删一侧就打破平衡；
         整层一起删，两侧同时消失，净扰动反而回到中心。
  H_mass 只是质量：top-97 本来就承载了几乎全部效应，整层多出来的部分互相抵消且不参与，
         两种干预在效果上等价，「更狠」只是噪声。

判别方法：直接构造「对冲对」。
  候选 N 取自 rest（99%）里与 P 净符号相反（P 净 Δ = +9.80 > 0，故 N 取 Δ<0）的单元。
  两个版本，用来区分「符号对冲」与「方向对冲」：
    Nabs  取 |Δ| 最大的 97 个反号单元（只按符号，不看写方向）
    Nopp  在反号单元里取写方向（down_proj 第 i 列）与 P 最相似者，逐 p 配对后去重补足
  若 H_bal 成立：删 P∪N 的翻转率 > 删 P 与删 N 各自翻转率之和（超加性）；
  且 Nopp 的超加性应强于 Nabs（真正的对冲需要同一读法方向）。

预注册判据：
  B1 单侧不对称  KL(P) ≥ 3×KL(N)，且 flip(P) > flip(N)（P 是净正杠杆，N 是配重）
  B2 超加性      flip(P∪Nopp) > flip(P) + flip(Nopp)，KL(P∪Nopp) > 0.85×[KL(P)+KL(Nopp)]
  B3 方向必要    flip(P∪Nopp) ≥ flip(P∪Nabs)（写方向对齐的对冲更有效）
  判读：B1+B2 过 → 平衡机制成立，top-97 是「可推点」不是「承重墙」；
        B2 不过 → H_mass，两种干预等价，C5' 的差异只是噪声。

顺带产出：按基线 margin 四分位的翻转率表（必要性是否只存在于低 margin 题）。

用法：.venv/Scripts/python.exe -X utf8 tools/mech_balance.py
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from server import lang_atlas_service as latlas
from server import mech_pair_service as mpair

B = 24
SEED = 20261101
OUT = Path('tempdata/mech_balance.json')


def prep(tok, letter_ids, q, max_len=640):
    prompt, lines, _ = latlas._build_prompt(q)
    ids = tok(prompt, return_tensors='pt', add_special_tokens=True)['input_ids'][0][:max_len]
    spans, tot = latlas._spans(tok, lines)
    if abs(int(ids.shape[0]) - tot) > 4:
        return None
    ans = next((c for c in letter_ids.get(q['answer'], [])), None)
    others = [next(iter(letter_ids.get(x) or []), None) for x in 'ABCDE'
              if x != q['answer'] and letter_ids.get(x)]
    others = [o for o in others if o is not None]
    if ans is None or not others:
        return None
    return ids, ans, others


def main():
    t_all = time.time()
    model, tok, device, meta, model_name = mpair._wait_ready()
    letter_ids = latlas._letter_token_ids(tok)

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
    rows = []
    for s, k in spec:
        pool = by_sub[s]
        rows += [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]

    a = np.load('tempdata/unit_attr_L24.npz')['attr_sum'].astype(np.float64)
    d_ff = a.size
    k1 = d_ff // 100
    P = np.argsort(-np.abs(a))[:k1]
    net = float(a[P].sum())
    rest = np.setdiff1d(np.arange(d_ff), P)
    opp_pool = rest[a[rest] * np.sign(net) < 0]

    # Nabs：只按 |Δ| 取反号最大
    Nabs = opp_pool[np.argsort(-np.abs(a[opp_pool]))][:k1]

    # Nopp：写方向与 P 最相似的反号单元，逐 p 配一个
    W = model.model.layers[B].mlp.down_proj.weight.detach().float()      # [d_model, d_ff]
    Wn = W / (W.norm(dim=0, keepdim=True) + 1e-9)
    cosP = (Wn[:, P].T @ Wn[:, opp_pool]).cpu().numpy()                  # [97, |opp|]
    chosen = []
    for i in range(len(P)):
        for j in np.argsort(-cosP[i]):
            if int(opp_pool[j]) not in chosen:
                chosen.append(int(opp_pool[j]))
                break
    Nopp = np.array(chosen, dtype=np.int64)

    rand194 = np.random.default_rng(SEED + 2).choice(d_ff, size=len(P) + len(Nopp), replace=False)
    rand97 = np.random.default_rng(SEED + 3).choice(d_ff, size=k1, replace=False)
    sets = {'P': P, 'Nabs': Nabs, 'Nopp': Nopp,
            'P_Nopp': np.union1d(P, Nopp), 'P_Nabs': np.union1d(P, Nabs),
            'rand97': rand97, 'rand194': rand194}
    print('ready %.0fs  P净Δ %+.2f  |P|=%d |Nabs|=%d |Nopp|=%d  余弦(P,Nopp)中位 %.3f'
          % (time.time() - t_all, net, len(P), len(Nabs), len(Nopp),
             float(np.median([cosP[i].max() for i in range(len(P))]))), flush=True)

    names = list(sets) + ['full']
    per_q = []
    t0 = time.time()
    for qi, q in enumerate(rows):
        p = prep(tok, letter_ids, q)
        if p is None:
            continue
        ids, ans, others = p

        def run(mask=None):
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
                    o = model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
                return o.logits[0, -1].float()
            finally:
                for hh in hs:
                    hh.remove()

        l0 = run(None)
        lp0 = torch.log_softmax(l0, dim=-1)
        m0 = float(l0[ans] - l0[others].max())
        best0 = int(l0.argmax())
        rec = {'qi': qi, 'margin0': round(m0, 4), 'correct0': bool(best0 == ans),
               'kl': {}, 'dm': {}, 'flip': {}}
        for nm, idx in sets.items():
            m = np.zeros(d_ff, dtype=bool)
            m[idx] = True
            l1 = run(m)
            lp1 = torch.log_softmax(l1, dim=-1)
            rec['kl'][nm] = round(float((lp0.exp() * (lp0 - lp1)).sum()), 5)
            rec['dm'][nm] = round(float(l1[ans] - l1[others].max()) - m0, 4)
            rec['flip'][nm] = bool(int(l1.argmax()) != best0)
            del m, l1, lp1
        l1 = run(np.ones(d_ff, dtype=bool))
        lp1 = torch.log_softmax(l1, dim=-1)
        rec['kl']['full'] = round(float((lp0.exp() * (lp0 - lp1)).sum()), 5)
        rec['dm']['full'] = round(float(l1[ans] - l1[others].max()) - m0, 4)
        rec['flip']['full'] = bool(int(l1.argmax()) != best0)
        per_q.append(rec)
        del l0, lp0, l1, lp1
        if (qi + 1) % 40 == 0:
            print('  %d/%d %.0fs' % (qi + 1, len(rows), time.time() - t0), flush=True)

    n = len(per_q)
    agg = {'n': n, 'kl': {}, 'flip': {}, 'dm': {}}
    for nm in names:
        agg['kl'][nm] = round(float(np.mean([r['kl'][nm] for r in per_q])), 5)
        agg['flip'][nm] = int(sum(r['flip'][nm] for r in per_q))
        agg['dm'][nm] = round(float(np.mean([r['dm'][nm] for r in per_q])), 4)

    # 按基线 margin 四分位的翻转率（只在基线答对的题上算，答错的题「翻转」无意义）
    qs = np.quantile([r['margin0'] for r in per_q], [0.25, 0.5, 0.75])
    strat = {}
    for bi, lab in enumerate(['Q1 最低', 'Q2', 'Q3', 'Q4 最高']):
        sel = [r for r in per_q
               if (bi == 0 and r['margin0'] <= qs[0]) or
                  (0 < bi < 3 and qs[bi - 1] < r['margin0'] <= qs[bi]) or
                  (bi == 3 and r['margin0'] > qs[2])]
        strat[lab] = {'n': len(sel),
                      **{nm: int(sum(r['flip'][nm] for r in sel)) for nm in ('P', 'P_Nopp', 'rand97', 'full')}}

    B1 = agg['kl']['P'] >= 3 * max(agg['kl']['Nabs'], 1e-9) and agg['flip']['P'] > agg['flip']['Nabs']
    B2 = (agg['flip']['P_Nopp'] > agg['flip']['P'] + agg['flip']['Nopp'] and
          agg['kl']['P_Nopp'] > 0.85 * (agg['kl']['P'] + agg['kl']['Nopp']))
    B3 = agg['flip']['P_Nopp'] >= agg['flip']['P_Nabs']

    out = {'schema': 'mech_balance.v1', 'model_name': model_name, 'block': B,
           'P_net_delta': round(net, 3), 'sets_size': {k: int(len(v)) for k, v in sets.items()},
           'agg': agg, 'by_margin_quartile': strat,
           'verdict': {'B1_single_side': bool(B1), 'B2_superadditive': bool(B2), 'B3_dir_matters': bool(B3)},
           'per_question': per_q, 'secs': round(time.time() - t_all, 1)}
    out['reading'] = ('平衡破坏成立：P 是净正杠杆、N 是配重，单删一侧比两侧同删更狠。'
                      if (B1 and B2) else
                      '未见过阈值超加性：P 与 N 近似独立叠加，C5\' 的差异更可能是质量效应/噪声。')
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('kl', agg['kl'])
    print('flip', agg['flip'])
    print('verdict', out['verdict'])
    print('written %s %.0fs' % (OUT, time.time() - t_all), flush=True)


if __name__ == '__main__':
    main()
