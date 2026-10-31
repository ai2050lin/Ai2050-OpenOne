# -*- coding: utf-8 -*-
"""C4：单元级归因——绕开不稳定的簇，直接在 9728 个单元上排效应量。

为什么必须做：C19（L24 那个「6.7% 单元扛 37% 效应」的组）在
  · 换 KMeans 初值后 Jaccard 只有 0.41–0.46
  · 窗口自助重采样 80% 后只有 0.19–0.33
也就是说「哪个组最致因」不可复现。簇这个统计单元作废，必须回到单元。

方法：一阶归因（attribution patching 的单元版）
  zero 掉 down_proj 的第 i 个输入单元，对答案 logit 的一阶效应
      Δ_i ≈ −Σ_t x_{t,i} · ∂L/∂x_{t,i}
  一次前向 + 一次反向即得全部 9728 个 Δ_i。

验证（缺一不可）：
  ① 线性近似的有效性：按 |Δ| 取 top-p% 真实消融，KL 必须显著高于同规模随机组；
  ② 集中度：top-X% 单元承担多少整层 KL；
  ③ 可复现性：把题分成两半，两半各自排出的 top 单元 overlap（这是「稳定」的正确判据，
     与簇的稳定性不同——单元是固定对象，不依赖任何切分）。

用法：.venv/Scripts/python.exe -X utf8 tools/mech_unit_attr.py 14,24,30,34
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
from server import live_analysis_service as las
from server import mech_pair_service as mpair
from server import mech_service as msvc

ME = Path('data/mech')
OUT = Path(os.environ.get('UA_OUT', 'tempdata/mech_unit_attr.json'))
BLOCKS = [int(x) for x in (sys.argv[1].split(',') if len(sys.argv) > 1 else ['24'])]
NQ = int(sys.argv[2]) if len(sys.argv) > 2 else 0          # 0 = 全量 169 题


def main():
    t_all = time.time()
    model, tok, device, meta, model_name = mpair._wait_ready()
    letter_ids = latlas._letter_token_ids(tok)
    ldirs = msvc._letter_dirs(model, tok)
    rows = [r for r in latlas._load_questions() if (r.get('source') or '') == 'C-Eval val']
    if NQ and NQ < len(rows):
        rows = [rows[i] for i in np.random.default_rng(20261031).choice(len(rows), size=NQ, replace=False)]
    print('ready %.0fs  %d 题' % (time.time() - t_all, len(rows)), flush=True)

    out = {'schema': 'mech_unit_attr.v1', 'model_name': model_name, 'layers': {}}

    for b in BLOCKS:
        d_ff = int(model.model.layers[b].mlp.down_proj.weight.shape[1])
        attr_sum = np.zeros(d_ff)
        attr_abs = np.zeros(d_ff)
        half_a = np.zeros(d_ff)
        half_b = np.zeros(d_ff)
        kl_curve = {}
        n_ok = 0
        t0 = time.time()

        for qi, q in enumerate(rows):
            prompt, lines, _ = latlas._build_prompt(q)
            ids = tok(prompt, return_tensors='pt', add_special_tokens=True)['input_ids'][0][:640]
            T = int(ids.shape[0])
            spans, tot = latlas._spans(tok, lines)
            if abs(T - tot) > 4:
                continue
            ans_tok = next((c for c in letter_ids.get(q['answer'], [])), None)
            others = [next(iter(letter_ids.get(x) or []), None) for x in 'ABCDE'
                      if x != q['answer'] and letter_ids.get(x)]
            others = [o for o in others if o is not None]
            if ans_tok is None or not others:
                continue

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
                    target = lg[ans_tok] - torch.logsumexp(lg[list(others)], dim=0)
                    target.backward()
            finally:
                h.remove()
            x = cap['x']
            g = x.grad
            if g is None:
                continue
            d = -(x.detach().float() * g.float()).sum(dim=(0, 1)).cpu().numpy()   # [d_ff]
            attr_sum += d
            attr_abs += np.abs(d)
            if qi % 2 == 0:
                half_a += d
            else:
                half_b += d
            n_ok += 1
            del o, lg, x, g, cap
            if (qi + 1) % 40 == 0:
                print('  L%d 归因 %d/%d %.0fs' % (b, qi + 1, len(rows), time.time() - t0), flush=True)

        # ---- 真实消融验证
        rng = np.random.default_rng(20261031)
        sets = {}
        for pct in (1, 5):
            k = max(1, int(d_ff * pct / 100))
            sets['top%d' % pct] = np.argsort(-np.abs(attr_sum))[:k]
            sets['rand%d' % pct] = rng.choice(d_ff, size=k, replace=False)
        # 补集：除了 top-1% 之外的全部。用来判定「top-1% 的 KL 大于整层」是超加性，
        # 还是仅仅因为整层消融里剩下的单元在往回拉。
        top1 = sets['top1']
        compl = np.ones(d_ff, dtype=bool)
        compl[top1] = False
        sets['compl1'] = np.flatnonzero(compl)
        kl = {nm: 0.0 for nm in sets}
        kl['full'] = 0.0
        n2 = 0

        for qi, q in enumerate(rows):
            prompt, lines, _ = latlas._build_prompt(q)
            ids = tok(prompt, return_tensors='pt', add_special_tokens=True)['input_ids'][0][:640]
            T = int(ids.shape[0])
            spans, tot = latlas._spans(tok, lines)
            if abs(T - tot) > 4:
                continue
            ans_tok = next((c for c in letter_ids.get(q['answer'], [])), None)
            others = [next(iter(letter_ids.get(x) or []), None) for x in 'ABCDE'
                      if x != q['answer'] and letter_ids.get(x)]
            if ans_tok is None or not others:
                continue

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
                    return torch.log_softmax(o.logits[0, -1].float(), dim=-1)
                finally:
                    for hh in hs:
                        hh.remove()

            lp0 = run(None)
            for nm, idx in sets.items():
                m = np.zeros(d_ff, dtype=bool)
                m[idx] = True
                lp1 = run(m)
                kl[nm] += float((lp0.exp() * (lp0 - lp1)).sum())
            lp1 = run(np.ones(d_ff, dtype=bool))
            kl['full'] += float((lp0.exp() * (lp0 - lp1)).sum())
            n2 += 1
            if (qi + 1) % 40 == 0:
                print('  L%d 消融 %d/%d %.0fs' % (b, qi + 1, len(rows), time.time() - t0), flush=True)

        rec = {'block': b, 'n_questions': n2, 'd_ff': d_ff,
               'kl': {nm: round(v / max(n2, 1), 5) for nm, v in kl.items()},
               'attr_abs_sum': round(float(np.abs(attr_sum).sum()), 3),
               'n_pos': int((attr_sum > 0).sum()), 'n_neg': int((attr_sum < 0).sum())}
        rec['concentration'] = {
            'top1pct_share_of_abs': round(float(np.sort(np.abs(attr_sum))[::-1][:max(1, d_ff // 100)].sum()
                                                / np.abs(attr_sum).sum()), 3),
            'top5pct_share_of_abs': round(float(np.sort(np.abs(attr_sum))[::-1][:max(1, d_ff // 20)].sum()
                                                / np.abs(attr_sum).sum()), 3),
        }
        # 两半排序的一致性（正确判据：单元是固定对象）
        ra = np.argsort(-np.abs(half_a))[:500]
        rb = np.argsort(-np.abs(half_b))[:500]
        rec['split_half_top500_overlap'] = int(np.intersect1d(ra, rb).size)
        rec['split_half_spearman'] = round(float(np.corrcoef(
            np.argsort(np.argsort(np.abs(half_a))), np.argsort(np.argsort(np.abs(half_b))))[0, 1]), 3)
        rec['top20_units'] = [[int(u), round(float(attr_sum[u]), 4)] for u in np.argsort(-np.abs(attr_sum))[:20]]
        rec['secs'] = round(time.time() - t0, 1)
        # 全量归因向量落盘，后面要复用（补集、跨层、和 readout 词对齐）
        np.savez_compressed(Path('tempdata') / ('unit_attr_L%d.npz' % b),
                            attr_sum=attr_sum.astype(np.float32),
                            attr_abs=attr_abs.astype(np.float32),
                            half_a=half_a.astype(np.float32), half_b=half_b.astype(np.float32))
        out['layers'][str(b)] = rec
        print('L%-2d kl %s  full %.5f | top1%%占|Δ| %s  两半 top500 重叠 %d  spearman %.3f (%.0fs)'
              % (b, {k: round(v / max(n2, 1), 5) for k, v in kl.items() if k != 'full'},
                 kl['full'] / max(n2, 1), rec['concentration']['top1pct_share_of_abs'],
                 rec['split_half_top500_overlap'], rec['split_half_spearman'], time.time() - t0), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev = {}
    if OUT.exists():                      # 分层跑，别互相覆盖
        try:
            prev = json.loads(OUT.read_text(encoding='utf-8')).get('layers', {})
        except Exception:
            prev = {}
    prev.update(out['layers'])
    out['layers'] = prev
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print('written', OUT, '%.0fs' % (time.time() - t_all), flush=True)


if __name__ == '__main__':
    main()
