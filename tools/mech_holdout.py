# -*- coding: utf-8 -*-
"""H1：L24 杠杆集的 held-out 复现（C-Eval val → CMMLU test）。

为什么必须做：C3'/C4' 已经在 C-Eval 上证明「top-1% 单元扛的 KL 略大于整层」，
但这始终可能是「这批题特有」。要区分两种解释：
  (a) 杠杆集是模型属性 —— 换数据集，同一批单元仍高杠杆；
  (b) 杠杆集是数据集属性 —— 换数据集，效应换个地方落。
这是「单元级归因」能不能当结论用的唯一判决实验。

预注册判据（跑之前写死，跑完不改）：
  源集 S_ce  = C-Eval val 169 题上 |Δ_i| 的 top-97（1%，已存 tempdata/unit_attr_L24.npz）
  目标集两套（都在 CMMLU test 里抽 169 题，seed 20261101）：
    match  科目级配对：high_school_biology 19 / high_school_geography 19 /
           chinese_history 10 / world_history 10 / chinese_civil_service_exam 47 /
           logical 22 / modern_chinese 23 / ancient_chinese 19
    wide   全 69 个科目均匀随机（不加结构，最严）
  判据：
    K1 集稳定    |S_ce ∩ S_cm| ≥ 20          （随机期望 0.97，即 ≥20× 机会）
    K2 方向一致  spearman(|Δ_ce|, |Δ_cm|) ≥ 0.30  （CE 自身两半为 0.47–0.73）
    K3 杠杆可迁移 CMMLU 题上 KL(S_ce) ≥ 3×KL(rand97) 且 ≥ 0.5×KL(S_cm)
  解读：
    三条全过   → 杠杆集是模型属性
    K3 过 K1 不过 → 同一功能多重实现（单元不固定，但效应可迁移）
    K3 不过    → 杠杆集依附数据集，单元级结论不能跨集引用

顺带回答的第二个问题：C5' 在 C-Eval 上发现「整层删掉 KL 很大但答案几乎不翻」，
  在 CMMLU 上是否复现（记 acc / margin / 翻转数）。若不翻，说明 KL 测的是
  「分布扰动」而非「行为必要性」，②阶段必须换因变量。

用法：.venv/Scripts/python.exe -X utf8 tools/mech_holdout.py
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

B = int(os.environ.get('LAYER', 24))            # 源集所在层，源归因向量必须是这一层的
SEED = 20261101
OUT = Path('tempdata/holdout_L%d.json' % B)
NPZ = Path('tempdata/unit_attr_L%d.npz' % B)
SMOKE = os.environ.get('SMOKE') == '1'          # SMOKE=1 → 每组只 4 题，只为验证管线与钩子生效

MATCH_SPEC = [('high_school_biology', 19), ('high_school_geography', 19),
              ('chinese_history', 10), ('world_history', 10),
              ('chinese_civil_service_exam', 47), ('logical', 22),
              ('modern_chinese', 23), ('ancient_chinese', 19)]


def prep(tok, letter_ids, q, max_len=640):
    """返回 (ids, ans_tok, others) 或 None。分段边界不一致的题直接跳过，不猜。"""
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


def attr_all(model, device, rows, tok, letter_ids, b):
    """一阶归因：Δ_i = −Σ_t x_{t,i}·∂L/∂x_{t,i}，一次 fwd+bwd 拿全部 d_ff。"""
    d_ff = int(model.model.layers[b].mlp.down_proj.weight.shape[1])
    acc = np.zeros(d_ff)
    n = 0
    t0 = time.time()
    for qi, q in enumerate(rows):
        p = prep(tok, letter_ids, q)
        if p is None:
            continue
        ids, ans, others = p
        cap = {}

        def grab(_m, args):
            a = args[0]
            a.retain_grad()
            cap['x'] = a
        h = model.model.layers[b].mlp.down_proj.register_forward_pre_hook(grab)
        try:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                o = model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
                lg = o.logits[0, -1].float()
                L = lg[ans] - torch.logsumexp(lg[others], dim=0)
                L.backward()
        finally:
            h.remove()
        x = cap.get('x')
        g = None if x is None else x.grad
        if g is None:
            print('    ! 第 %d 题钩子未拿到梯度，跳过' % qi, flush=True)
            del cap
            continue
        acc += -(x.detach().float() * g.float()).sum(dim=(0, 1)).cpu().numpy()
        n += 1
        del o, lg, x, g, cap
        if (qi + 1) % 40 == 0:
            print('    归因 %d/%d %.0fs' % (qi + 1, len(rows), time.time() - t0), flush=True)
    return acc, n


def ablate_stats(model, device, rows, tok, letter_ids, b, sets):
    """对每个干预集算：平均 KL、平均 Δmargin、答案翻转数。"""
    d_ff = int(model.model.layers[b].mlp.down_proj.weight.shape[1])
    names = list(sets) + ['full']
    kl = {nm: 0.0 for nm in names}
    dmarg = {nm: 0.0 for nm in names}
    flip = {nm: 0 for nm in names}
    n = 0
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
                    a = args[0].clone()
                    a[..., msk] = 0.0
                    return (a,) + tuple(args[1:])
                hs.append(model.model.layers[b].mlp.down_proj.register_forward_pre_hook(abl))
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

        def scor(nm, mask):
            l1 = run(mask)
            lp1 = torch.log_softmax(l1, dim=-1)
            kl[nm] += float((lp0.exp() * (lp0 - lp1)).sum())
            dd = float(l1[ans] - l1[others].max()) - m0
            dmarg[nm] += dd
            if int(l1.argmax()) != best0:
                flip[nm] += 1

        for nm, idx in sets.items():
            m = np.zeros(d_ff, dtype=bool)
            m[idx] = True
            scor(nm, m)
            del m
        scor('full', np.ones(d_ff, dtype=bool))
        n += 1
        del l0, lp0
        if (qi + 1) % 40 == 0:
            print('    消融 %d/%d %.0fs' % (qi + 1, len(rows), time.time() - t0), flush=True)
    return ({nm: round(v / max(n, 1), 5) for nm, v in kl.items()},
            {nm: round(v / max(n, 1), 4) for nm, v in dmarg.items()},
            flip, n)


def ov(a, b, k):
    return int(np.intersect1d(np.argsort(-np.abs(a))[:k], np.argsort(-np.abs(b))[:k]).size)


def sp(a, b):
    ra = np.argsort(np.argsort(np.abs(a)))
    rb = np.argsort(np.argsort(np.abs(b)))
    return round(float(np.corrcoef(ra, rb)[0, 1]), 3)


def main():
    t_all = time.time()
    model, tok, device, meta, model_name = mpair._wait_ready()
    letter_ids = latlas._letter_token_ids(tok)

    rows = [r for r in latlas._load_questions() if r.get('source')]
    ce = [r for r in rows if r['source'] == 'C-Eval val']
    cm_all = [r for r in rows if r['source'] == 'CMMLU test']
    by_sub = {}
    for r in cm_all:
        by_sub.setdefault(r['subject'], []).append(r)
    rng = np.random.default_rng(SEED)

    # ---- 两套 held-out 抽样（抽样只依赖题库，和模型无关，先做）
    match = []
    for subj, k in MATCH_SPEC:
        pool = by_sub.get(subj) or []
        if len(pool) < k:
            raise SystemExit('科目 %s 只有 %d 题，不足 %d' % (subj, len(pool), k))
        idx = rng.choice(len(pool), size=k, replace=False)
        match += [pool[i] for i in idx]
    wide = [cm_all[i] for i in rng.choice(len(cm_all), size=169, replace=False)]
    if SMOKE:
        match, wide = match[:4], wide[:4]
        globals()['OUT'] = Path('tempdata/holdout_L%d_smoke.json' % B)
    print('ready %.0fs  CE %d / CM-match %d / CM-wide %d' % (time.time() - t_all, len(ce), len(match), len(wide)), flush=True)

    ce_attr = np.load(NPZ)['attr_sum'].astype(np.float64)
    d_ff = ce_attr.size
    s_ce = np.argsort(-np.abs(ce_attr))[:d_ff // 100]
    out = {'schema': 'mech_holdout.v1', 'model_name': model_name, 'block': B,
           'seed': SEED, 'n_ce': len(ce), 'targets': {}}

    res = {}
    for tag, sub in (('match', match), ('wide', wide)):
        print('-- %s' % tag, flush=True)
        attr, n = attr_all(model, device, sub, tok, letter_ids, B)
        s_cm = np.argsort(-np.abs(attr))[:d_ff // 100]
        res[tag] = {'n': n, 'attr': attr,
                    'top97_overlap': int(np.intersect1d(s_ce, s_cm).size),
                    'top500_overlap': ov(ce_attr, attr, 500),
                    'spearman_abs': sp(ce_attr, attr),
                    'share_top1pct_abs': round(float(np.sort(np.abs(attr))[::-1][:d_ff // 100].sum()
                                                      / np.abs(attr).sum()), 3),
                    'corr_signed': round(float(np.corrcoef(ce_attr, attr)[0, 1]), 3)}
        r = res[tag]
        print('   %s n=%d top97重叠 %d  两半无关期望 0.97  top500 %d  spearman %.3f'
              % (tag, n, r['top97_overlap'], r['top500_overlap'], r['spearman_abs']), flush=True)

    # ---- K3：只在 match 组做真实消融（最容易迁移的一组，先给它机会）
    attr = res['match']['attr']
    s_cm = np.argsort(-np.abs(attr))[:d_ff // 100]
    rand = np.random.default_rng(SEED + 1).choice(d_ff, size=d_ff // 100, replace=False)
    compl = np.ones(d_ff, dtype=bool)
    compl[s_ce] = False
    sets = {'ce97': s_ce, 'cm97': s_cm, 'rand97': rand, 'compl_ce97': np.flatnonzero(compl), 'cm97_minus_ce97': np.setdiff1d(s_cm, s_ce)}
    print('-- K3 消融（match 169 题，6 次前向/题）', flush=True)
    kl, dmarg, flip, n2 = ablate_stats(model, device, match, tok, letter_ids, B, sets)
    print('   kl', kl, '\n   dmargin', dmarg, '\n   flip', flip, flush=True)

    k1_pass = res['match']['top97_overlap'] >= 20 and res['wide']['top97_overlap'] >= 20
    k2_pass = res['match']['spearman_abs'] >= 0.30
    k3_pass = kl['ce97'] >= 3 * max(kl['rand97'], 1e-9) and kl['ce97'] >= 0.5 * max(kl['cm97'], 1e-9)

    out.update({
        'source_top97_units': [int(u) for u in s_ce],
        'targets': {t: {k: v for k, v in r.items() if k != 'attr'} for t, r in res.items()},
        'ablate_match': {'n': n2, 'kl': kl, 'dmargin': dmarg, 'flip': flip,
                         'sets_size': {k: int(len(v)) for k, v in sets.items()}},
        'verdict': {'K1_set_stable': bool(k1_pass), 'K2_direction': bool(k2_pass),
                    'K3_leverage_transfers': bool(k3_pass)},
        'secs': round(time.time() - t_all, 1),
    })
    out['concentration'] = round(kl['ce97'] / max(kl['full'], 1e-9), 3)   # 1% 承担整层效应的比例
    if k3_pass and k1_pass and k2_pass:
        out['reading'] = '杠杆集是模型属性：集稳定、方向一致、效应可迁移。'
    elif k3_pass:
        out['reading'] = ('效应可迁移但集合/排名不稳（多重实现）：同一功能由不同单元实现，'
                          '单元编号不能当稳定 ID 跨集引用。')
    else:
        out['reading'] = '杠杆集依附数据集：C-Eval 上的单元级结论不能跨集引用。'

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written %s %s | %s | %.0fs'
          % (OUT, out['verdict'], out['reading'], time.time() - t_all), flush=True)


if __name__ == '__main__':
    main()
