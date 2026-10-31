# -*- coding: utf-8 -*-
"""H3：扰动方向分析 —— 为什么「删 1%」的 KL 恰好等于「删 100%」。

H2 否证了平衡假说，但留下一个更硬的事实（CMMLU 169 题，L24）：
    KL(删P)     = 0.04764
    KL(删整层)  = 0.04765      ← 1% 与 100% 打平
    KL(删补集)  = 0.02397      ← 99% 只贡献一半
    KL(P∪N)     = 0.04543 < KL(P)   ← 加进更多单元，扰动反而变小
这是次加性（sub-additive），不是超加性。KL 是标量，看不出「为什么」。
把 Δlogits 当向量看，只有三种可能：
  (i)  主方向：ΔP 与 Δ整层 同向且同长（cos≈1, ‖ΔP‖≈‖Δfull‖），其余 99% 的扰动与之正交
       —— top-1% 就是整层效应的主方向，剩下的是零散的、彼此正交的小扰动。
  (ii) 抵消：Δ补集 与 ΔP 反号，整层 = 两者相消 —— 那 99% 里有系统性的对手。
  (iii) 饱和：Δ 的模随删除规模增长但方向乱转，KL 对方向不敏感才打平。

预注册判据：
  D1 主方向  median cos(ΔP, Δfull) ≥ 0.80
  D2 非抵消  median cos(ΔP, Δcompl) ≥ 0（若显著为负 = 存在系统性对手）
  D3 长度匹配  median ‖ΔP‖/‖Δfull‖ ≥ 0.80，且 ‖Δcompl‖/‖Δfull‖ 明显小于 1
  判读：D1+D3 → (i) 主方向，top-1% 是整层效应的承载子空间；
        D2 为负 → (ii) 抵消，需要回头找对手集；
        都不成立 → (iii) 饱和，KL 不是可加量，C 阶段的 KL 结论要降级为「扰动幅度」。

用法：.venv/Scripts/python.exe -X utf8 tools/mech_delta_cos.py
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
OUT = Path('tempdata/mech_delta_cos.json')


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
    sets = {'P': P, 'compl': compl, 'rand97': rand97}
    print('ready %.0fs  |P|=%d |compl|=%d' % (time.time() - t_all, len(P), len(compl)), flush=True)

    def mask_of(idx):
        m = np.zeros(d_ff, dtype=bool)
        m[idx] = True
        return m
    masks = {nm: mask_of(v) for nm, v in sets.items()}

    cos_ = {('P', 'full'): [], ('P', 'compl'): [], ('compl', 'full'): [], ('P', 'rand97'): []}
    norms = {'P': [], 'compl': [], 'full': [], 'rand97': []}
    kl = {'P': [], 'compl': [], 'full': [], 'rand97': []}
    ans_share = {'P': [], 'compl': [], 'full': []}

    t0 = time.time()
    for qi, q in enumerate(rows):
        p = prep(tok, letter_ids, q)
        if p is None:
            continue
        ids, ans, others = p
        tgt = np.array([ans] + list(others))

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
        D = {'full': run(np.ones(d_ff, dtype=bool))}
        for nm, m in masks.items():
            D[nm] = run(m)
        dl = {nm: (v - l0) for nm, v in D.items()}

        def cosv(u, v):
            return float(torch.dot(u, v) / (u.norm() * v.norm() + 1e-12))
        for key, (x, y) in {('P', 'full'): ('P', 'full'), ('P', 'compl'): ('P', 'compl'),
                            ('compl', 'full'): ('compl', 'full'), ('P', 'rand97'): ('P', 'rand97')}.items():
            cos_[key].append(cosv(dl[x], dl[y]))
        for nm in norms:
            norms[nm].append(float(dl[nm].norm()))
            lp1 = torch.log_softmax(D[nm], dim=-1)
            kl[nm].append(float((lp0.exp() * (lp0 - lp1)).sum()))
        for nm in ans_share:
            v = dl[nm]
            ans_share[nm].append(float(v[ans] - v[list(others)].mean()) / (float(v.norm()) + 1e-12))
        del l0, lp0, D, dl
        if (qi + 1) % 40 == 0:
            print('  %d/%d %.0fs' % (qi + 1, len(rows), time.time() - t0), flush=True)

    def med(v):
        return round(float(np.median(v)), 3)

    summ = {
        'n': len(rows),
        'cos': {('%s~%s' % k): med(v) for k, v in cos_.items()},
        'cos_mean': {('%s~%s' % k): round(float(np.mean(v)), 3) for k, v in cos_.items()},
        'norm_ratio_vs_full': {nm: med([n2 / (n1 + 1e-12) for n1, n2 in zip(norms[nm], norms['full'])])
                               for nm in ('P', 'compl', 'rand97')},
        'kl': {nm: round(float(np.mean(v)), 5) for nm, v in kl.items()},
        'net_shift_on_options': {nm: med(v) for nm, v in ans_share.items()},
    }
    D1 = summ['cos']['P~full'] >= 0.80
    D2 = summ['cos']['P~compl'] >= 0
    D3 = summ['norm_ratio_vs_full']['P'] >= 0.80
    out = {'schema': 'mech_delta_cos.v1', 'model_name': model_name, 'block': B,
           'summary': summ,
           'verdict': {'D1_main_direction': bool(D1), 'D2_not_cancelling': bool(D2), 'D3_norm_match': bool(D3)},
           'secs': round(time.time() - t_all, 1)}
    if D1 and D3:
        out['reading'] = '主方向：top-1% 的扰动与整层同向同长，其余 99% 只是正交的零散扰动。'
    elif not D2:
        out['reading'] = '抵消：补集与 top-1% 反号，整层是两者相消后的残差，需找对手集。'
    else:
        out['reading'] = '饱和：扰动模随规模增长但方向不定，KL 对方向不敏感，KL 结论降级为扰动幅度。'
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(summ, ensure_ascii=False, indent=1))
    print('verdict', out['verdict'], '|', out['reading'])
    print('written %s %.0fs' % (OUT, time.time() - t_all), flush=True)


if __name__ == '__main__':
    main()
