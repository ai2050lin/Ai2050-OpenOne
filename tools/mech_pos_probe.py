# -*- coding: utf-8 -*-
"""C3：把「关键组」放到提示的位置轴上，回答两个问题。

① 每层那个最致因的 MLP 组，在提示的哪一段最活跃？（指令 / 题干 / 选项 / 答案位）
   自然语料的普查只能告诉我们它「在标点和 URL 上亮」，但在选择题提示里它到底
   在解析选项标记、还是在读题干知识，必须用提示本身去测。
② 每层的关键组是不是同一套？（跨层稳定性）

做法（每题两次前向）：
  pass1：捕获 L{b} 的 down_proj 输入（即被 zero 消融的那个张量）在**每个位置**的
         各单元激活 → 按 A2 的簇标签聚合 → [T, K] 的活动剖面；
  pass2：只消融该层 KL 最大的 2 个簇 + 1 个同规模随机组，记每题 KL。

不需要重跑 24 个簇的全消融（C1/C2 已有）。
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
from server import mech_service as msvc

ME = Path('data/mech')
BLOCKS = [int(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['14', '24', '30', '34'])]
SEGS = ['instr', 'q', 'opt', 'ans']


def seg_of(spans, T):
    out = np.full(T, -1, dtype=np.int8)
    for i, name in enumerate(SEGS):
        if name == 'opt':
            for ch in 'ABCDE':
                lo, hi = spans.get('opt' + ch, (0, 0))
                if hi > lo:
                    out[lo:hi] = i
        else:
            lo, hi = spans.get(name, (0, 0))
            if hi > lo:
                out[lo:hi] = i
    return out


def main():
    t_all = time.time()
    model, tok, device, meta, model_name = mpair._wait_ready()
    L = int(meta['n_layers'])
    letter_ids = latlas._letter_token_ids(tok)
    ldirs = msvc._letter_dirs(model, tok)

    cj = json.loads(mpair.CLUST_JSON.read_text(encoding='utf-8'))
    cz = np.load(mpair.CLUST_NPZ)
    rows = [r for r in latlas._load_questions() if (r.get('source') or '') == 'C-Eval val']
    print('model ready %.0fs, %d questions' % (time.time() - t_all, len(rows)), flush=True)

    out = {'schema': 'mech_pos_probe.v1', 'model_name': model_name,
           'segs': SEGS, 'n_questions': 0, 'layers': {}}

    for b in BLOCKS:
        rec = next((r for r in cj['layers'] if r['block'] == b), None)
        if rec is None:
            print('skip L%d：无簇' % b, flush=True)
            continue
        lab = cz['lab_%d' % b].astype(np.int64)
        K = int(rec['k'])
        d_ff = lab.size
        cols = [np.flatnonzero(lab == c) for c in range(K)]

        ab = json.loads((ME / ('ablate_L%d.json' % b)).read_text(encoding='utf-8'))
        zc = sorted(ab['methods']['zero']['clusters'], key=lambda c: -c['kl'])
        pick_top = [c['cluster'] for c in zc[:2]]
        pick_top_size = {c['cluster']: c['size'] for c in zc[:2]}
        rng = np.random.default_rng(20261031)
        sham_mask = np.zeros(d_ff, dtype=bool)
        sham_mask[rng.choice(d_ff, size=int(np.mean([len(x) for x in cols])), replace=False)] = True

        prof_sum = np.zeros((K, len(SEGS)))
        prof_n = np.zeros(len(SEGS))
        act_ans = np.zeros((len(rows), K))
        kl_q = {c: np.zeros(len(rows)) for c in pick_top}
        kl_sham = np.zeros(len(rows))
        act_ans_top = np.zeros((len(rows), len(pick_top)))
        marg_delta = {c: np.zeros(len(rows)) for c in pick_top}
        marg_base = np.zeros(len(rows))
        ok = np.zeros(len(rows), dtype=bool)

        t0 = time.time()
        for qi, q in enumerate(rows):
            prompt, lines, _ = latlas._build_prompt(q)
            ids = tok(prompt, return_tensors='pt', add_special_tokens=True)['input_ids'][0][:640]
            T = int(ids.shape[0])
            spans, tot = latlas._spans(tok, lines)
            if abs(T - tot) > 4:
                continue
            seg = seg_of(spans, T)
            ans_tok = next((c for c in letter_ids.get(q['answer'], [])), None)
            others = [next(iter(letter_ids.get(x) or []), None) for x in 'ABCDE'
                      if x != q['answer'] and letter_ids.get(x)]
            others = [o for o in others if o is not None]
            if ans_tok is None or not others:
                continue

            cap = {}

            def grab(_m, args):
                cap['x'] = args[0].detach()[0].float()          # [T, d_ff]
            hs = [model.model.layers[b].mlp.down_proj.register_forward_pre_hook(grab)]

            def fwd():
                with torch.no_grad():
                    o = model(input_ids=ids.unsqueeze(0).to(device), use_cache=False)
                lg = o.logits[0, -1].float()
                m = float(lg[ans_tok] - lg[list(others)].max())
                return torch.log_softmax(lg, dim=-1), m

            try:
                lp0, m0 = fwd()
            finally:
                for h in hs:
                    h.remove()
            x = cap['x'].abs().cpu().numpy()                    # [T, d_ff]
            # 簇活动：簇内单元 |x| 的均值 → [T, K]
            for c in range(K):
                v = x[:, cols[c]].mean(axis=1)
                act_ans[qi, c] = float(v[-1])
                for si in range(len(SEGS)):
                    mm = seg == si
                    if mm.any():
                        prof_sum[c, si] += float(v[mm].mean())
            for si in range(len(SEGS)):
                if (seg == si).any():
                    prof_n[si] += 1
            for j, c in enumerate(pick_top):
                act_ans_top[qi, j] = act_ans[qi, c]
            marg_base[qi] = m0
            ok[qi] = True

            # pass2：逐个消融
            for c in pick_top:
                mask = lab == c
                msk = torch.as_tensor(mask, device=device)

                def abl(_m, args, msk=msk):
                    a = args[0].clone()
                    a[..., msk] = 0.0
                    return (a,) + tuple(args[1:])
                h2 = model.model.layers[b].mlp.down_proj.register_forward_pre_hook(abl)
                try:
                    lp1, m1 = fwd()
                finally:
                    h2.remove()
                kl_q[c][qi] = float((lp0.exp() * (lp0 - lp1)).sum())
                marg_delta[c][qi] = m1 - m0

            msk = torch.as_tensor(sham_mask, device=device)

            def abl2(_m, args, msk=msk):
                a = args[0].clone()
                a[..., msk] = 0.0
                return (a,) + tuple(args[1:])
            h3 = model.model.layers[b].mlp.down_proj.register_forward_pre_hook(abl2)
            try:
                lp1, _ = fwd()
            finally:
                h3.remove()
            kl_sham[qi] = float((lp0.exp() * (lp0 - lp1)).sum())

            if (qi + 1) % 20 == 0:
                print('  L%d %d/%d %.0fs' % (b, qi + 1, len(rows), time.time() - t0), flush=True)

        n = int(ok.sum())
        prof = prof_sum / np.maximum(prof_n, 1)[None, :]
        share = prof / np.maximum(prof.sum(axis=1, keepdims=True), 1e-12)
        lay = {'block': b, 'k': K, 'n_questions': n, 'pick_top': pick_top,
               'pick_size': pick_top_size, 'sham_size': int(sham_mask.sum()),
               'seg_share': {str(c): [round(float(x), 3) for x in share[c]] for c in range(K)},
               'seg_mean_abs': {str(c): [round(float(x), 3) for x in prof[c]] for c in range(K)},
               'ans_abs_mean': {str(c): round(float(act_ans[ok, c].mean()), 3) for c in range(K)},
               'kl': {str(c): {'mean': round(float(kl_q[c][ok].mean()), 5),
                               'median': round(float(np.median(kl_q[c][ok])), 5),
                               'p90': round(float(np.percentile(kl_q[c][ok], 90)), 5),
                               'max': round(float(kl_q[c][ok].max()), 5),
                               'n_gt_sham': int((kl_q[c][ok] > kl_sham[ok]).sum()),
                               'frac_of_questions_carrying_half':
                                   round(float(np.sort(kl_q[c][ok])[::-1][:max(1, n // 2)].sum()
                                               / max(kl_q[c][ok].sum(), 1e-12)), 3)}
                     for c in pick_top},
               'kl_sham': {'mean': round(float(kl_sham[ok].mean()), 5),
                           'p90': round(float(np.percentile(kl_sham[ok], 90)), 5)},
               'marg_base': round(float(marg_base[ok].mean()), 3),
               'marg_delta': {str(c): round(float(marg_delta[c][ok].mean()), 3) for c in pick_top},
               'corr_act_kl': {str(c): round(float(np.corrcoef(act_ans[ok, c], kl_q[c][ok])[0, 1]), 3)
                               for c in pick_top},
               'corr_act_ans_margin': {str(c): round(float(np.corrcoef(act_ans[ok, c], marg_delta[c][ok])[0, 1]), 3)
                                       for c in pick_top},
               'secs': round(time.time() - t0, 1)}
        out['layers'][str(b)] = lay
        print('L%-2d done %.0fs  kl_mean %s  sham %.5f' % (
            b, time.time() - t0,
            {c: round(float(kl_q[c][ok].mean()), 5) for c in pick_top}, kl_sham[ok].mean()), flush=True)

    out['n_questions'] = int(max([v['n_questions'] for v in out['layers'].values()] or [0]))
    p = Path('tempdata/mech_pos_probe.json')
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written', p, p.stat().st_size, '%.0fs' % (time.time() - t_all), flush=True)


if __name__ == '__main__':
    main()
