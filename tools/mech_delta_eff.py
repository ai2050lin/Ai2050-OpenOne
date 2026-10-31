# -*- coding: utf-8 -*-
"""H3b：扰动效率 —— top-1% 的扰动为什么「值钱」，随机 1% 的为什么不值钱。

H3 观测（CMMLU 169 题，L24，未中心化的 Δlogits）：
    ‖ΔP‖/‖Δfull‖ = 1.73    ‖Δcompl‖/‖Δfull‖ = 1.44    ‖Δrand97‖/‖Δfull‖ = 8.16
    KL:  P 0.04764          compl 0.02397              rand97 0.00089
随机删 97 个单元把 logits 推得最远（8 倍），却几乎不动分布（KL 0.0009）。
说明 Δlogits 的范数被「常数偏移」污染（softmax 对整体平移不敏感），而 KL 是
概率加权的范数：扰动落在高概率 token 上才值钱。

于是「删 1% ≈ 删 100%」的机制假设变成：
   top-1% 的单元把扰动精确投在模型正在分配概率的那些 token 上（定向杠杆），
   其余 99% 的扰动散在低概率区，删了等于没删 —— 所以整层 KL 与 top-1% 相同。

预注册判据（先中心化 Δ，去掉不影响 softmax 的常数分量）：
   E1 定向性  energy_share(P) ≥ 3 × energy_share(compl)
              energy_share = Σ_{t∈top20(p0)} δ_t² / Σ_t δ_t²
   E2 效率    [KL/‖δ_c‖²] 的 P : compl 比值 ≥ 3
   E3 选项聚焦 δ 在 5 个选项字母上的能量占比，P ≥ 2 × compl
  判读：E1+E2 成立 → 定向杠杆（可命名：这 97 个单元是「把概率推给某些 token」的执行端）；
        E1 不成立 → 扰动无方向偏好，KL 差异来自非线性放大，结论降级为「幅度」。

用法：.venv/Scripts/python.exe -X utf8 tools/mech_delta_eff.py
"""
import json
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
OUT = Path('tempdata/mech_delta_eff.json')
TOPK = 20


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
    P = np.argsort(-np.abs(a))[:d_ff // 100]
    compl = np.setdiff1d(np.arange(d_ff), P)
    rand97 = np.random.default_rng(SEED + 3).choice(d_ff, size=d_ff // 100, replace=False)
    setsl = {'P': P, 'compl': compl, 'rand97': rand97}
    letter_idx = sorted({t for L in 'ABCDE' for t in (letter_ids.get(L) or [])})
    print('ready %.0fs  |P|=%d  字母 token 数 %d' % (time.time() - t_all, len(P), len(letter_idx)), flush=True)

    def mask_of(idx):
        m = np.zeros(d_ff, dtype=bool)
        m[idx] = True
        return m
    masks = {nm: mask_of(v) for nm, v in setsl.items()}

    per = {nm: {'energy': [], 'kl': [], 'norm_c': [], 'opt': []} for nm in setsl}
    per['full'] = {'energy': [], 'kl': [], 'norm_c': [], 'opt': []}
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
        p0 = lp0.exp()
        top = torch.topk(p0, TOPK).indices
        D = {'full': run(np.ones(d_ff, dtype=bool))}
        for nm, m in masks.items():
            D[nm] = run(m)
        for nm, l1 in D.items():
            d = l1 - l0
            dc = d - d.mean()                       # 去掉不影响 softmax 的常数分量
            tot = float((dc * dc).sum())
            e_top = float((dc[top] ** 2).sum()) / (tot + 1e-12)
            lp1 = torch.log_softmax(l1, dim=-1)
            per[nm]['energy'].append(e_top)
            per[nm]['kl'].append(float((p0 * (lp0 - lp1)).sum()))
            per[nm]['norm_c'].append(float(dc.norm()))
            per[nm]['opt'].append(float((dc[letter_idx] ** 2).sum()) / (tot + 1e-12))
        del l0, lp0, p0, D
        if (qi + 1) % 40 == 0:
            print('  %d/%d %.0fs' % (qi + 1, len(rows), time.time() - t0), flush=True)

    def med(v):
        return round(float(np.median(v)), 4)

    summ = {}
    for nm, r in per.items():
        klm = float(np.mean(r['kl']))
        nc = float(np.median(r['norm_c']))
        eff = klm / (nc ** 2 + 1e-12) * 1e5
        summ[nm] = {'energy_share_top20': med(r['energy']), 'energy_share_letters': med(r['opt']),
                    'kl': round(klm, 5), 'norm_c_median': round(nc, 2),
                    'efficiency_x1e5': round(eff, 3)}

    E1 = summ['P']['energy_share_top20'] >= 3 * max(summ['compl']['energy_share_top20'], 1e-9)
    E2 = summ['P']['efficiency_x1e5'] >= 3 * max(summ['compl']['efficiency_x1e5'], 1e-9)
    E3 = summ['P']['energy_share_letters'] >= 2 * max(summ['compl']['energy_share_letters'], 1e-9)

    out = {'schema': 'mech_delta_eff.v1', 'model_name': model_name, 'block': B,
           'topk': TOPK, 'summary': summ,
           'verdict': {'E1_directed': bool(E1), 'E2_efficient': bool(E2), 'E3_option_focus': bool(E3)},
           'secs': round(time.time() - t_all, 1)}
    out['reading'] = ('定向杠杆：top-1% 的扰动落在模型真正在分配概率的 token 上，'
                      '其余 99% 散在低概率区，所以删了等于没删；这解释 C3\'/C5\'/H2 全部三个反直觉结果。'
                      if (E1 and E2) else
                      '扰动无方向偏好：KL 差异来自非线性放大，单元级结论只能写成「幅度画像」。')
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    print('verdict', out['verdict'], '|', out['reading'])
    print('written %s %.0fs' % (OUT, time.time() - t_all), flush=True)


if __name__ == '__main__':
    main()
