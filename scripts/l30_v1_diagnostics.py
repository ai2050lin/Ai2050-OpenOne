# -*- coding: utf-8 -*-
"""L30-v1 验证未通过后的事后核查（不改变预注册判定）：
  1) 空操作：钩子 clone 后原样写回，损失与基线应完全相同（排除实现路径差异）；
  2) 汇聚位：只在位置 ≥16 消融 / 只在前 16 个位置消融，与全部位置消融比较（普查均值不含前 16 位）；
  3) 噪声底：给随机单元加 ±0.001 的极小扰动（真实效应≈0），测 bf16 前向下损失变化的分布。
输出 research/neurons/L30-v1-diagnostics.json。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_layer_attribution as R  # noqa: E402


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    man = json.loads((R.RUN / 'manifest.json').read_text(encoding='utf-8'))
    cdir = R.NATIVE / '_census' / 'corpus' / man['corpus_fp']
    cman = json.loads((cdir / 'manifest.json').read_text(encoding='utf-8'))
    sep, seq, skip = int(cman['separator_id']), int(man['seq_len']), int(man['skip_head'])
    streams = [np.load(cdir / s['file'])['tokens'] for s in cman['sources']]
    val_w = R.pick_windows([len(t) // seq for t in streams], 47, 23, 40)
    z = np.load(R.RUN / 'L30.npz')
    mu = torch.tensor(z['s1'] / float(z['n']), dtype=torch.float32, device='cuda')
    att = json.loads((ROOT / 'research' / 'neurons' / 'L30-v1-attribution.json').read_text(encoding='utf-8'))
    model = AutoModelForCausalLM.from_pretrained(str(R.MODEL), dtype=torch.bfloat16, device_map='cuda').eval()
    L = model.model.layers[30]
    mode = {'m': None}

    def hook(mod, args):
        m = mode['m']
        if m is None:
            return None
        y = args[0].clone()
        kind = m[0]
        if kind == 'clone_set_same':
            y[..., m[1]] = args[0][..., m[1]]
        elif kind == 'mean_all':
            y[..., m[1]] = mu[m[1]].to(y.dtype)
        elif kind == 'mean_from16':
            y[:, skip:, m[1]] = mu[m[1]].to(y.dtype)
        elif kind == 'mean_sink_only':
            y[:, :skip, m[1]] = mu[m[1]].to(y.dtype)
        elif kind == 'tiny':
            y[..., m[1]] = y[..., m[1]] + m[2]
        return (y,)
    L.mlp.down_proj.register_forward_pre_hook(hook)
    batches = [torch.tensor(np.stack([streams[s][w * seq:(w + 1) * seq] for s, w in val_w[i:i + 8]]).astype(np.int64), device='cuda')
               for i in range(0, len(val_w), 8)]

    @torch.inference_mode()
    def loss(m):
        mode['m'] = m
        s = k = 0
        for ids in batches:
            pos = torch.arange(seq, device='cuda')[None]
            nxt = torch.roll(ids, -1, 1)
            mask = ((pos >= skip) & (pos <= seq - 2) & (ids != sep) & (nxt != sep))[:, :-1]
            lp = torch.log_softmax(model(input_ids=ids).logits[:, :-1].float(), -1)
            tl = -lp.gather(-1, ids[:, 1:, None])[..., 0]
            s += float((tl * mask).sum())
            k += int(mask.sum())
        return s / k
    base = loss(None)
    noop = [{'unit': u, 'delta': loss(('clone_set_same', u)) - base} for u in (182, 5000, 7777)]
    sink = []
    for r in att['validation']['rows'][50:62] + att['validation']['rows'][:6]:
        u = r['unit']
        sink.append({'unit': u, 'set': r['set'], 'est_total': r['est_total'], 'all_positions': loss(('mean_all', u)) - base,
                     'from16': loss(('mean_from16', u)) - base, 'sink_only': loss(('mean_sink_only', u)) - base})
        print(f'[sink] {len(sink)}/18 · {time.time() - t0:.0f}s', flush=True)
    rng = np.random.default_rng(1)
    noise = []
    for u in rng.choice(9728, 20, replace=False):
        for eps in (1e-3, -1e-3):
            noise.append({'unit': int(u), 'eps': eps, 'delta': loss(('tiny', int(u), eps)) - base})
        print(f'[noise] {len(noise)}/40 · {time.time() - t0:.0f}s', flush=True)
    nv = np.array([x['delta'] for x in noise])
    rows = att['validation']['rows']
    top_true = np.array([r['true_delta'] for r in rows[:50]])
    rand_true = np.array([r['true_delta'] for r in rows[50:]])
    out = {'schema': 'l30_v1_diagnostics.v1', 'prereg': 'L30-v1', 'note': '事后核查，不改变预注册判定（未通过）。',
           'baseline_loss': base, 'noop_clone_set_same': noop, 'sink_positions': sink,
           'noise_floor': {'n': len(nv), 'mean': float(nv.mean()), 'sd': float(nv.std()), 'frac_negative': float((nv < 0).mean()),
                           'rows': noise},
           'compare': {'top50_true_mean': float(top_true.mean()), 'top50_true_abs_median': float(np.median(np.abs(top_true))),
                       'random50_true_mean': float(rand_true.mean()), 'random50_true_abs_median': float(np.median(np.abs(rand_true)))},
           'seconds': round(time.time() - t0, 1)}
    (ROOT / 'research' / 'neurons' / 'L30-v1-diagnostics.json').write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if k not in ('sink_positions',)}, ensure_ascii=False, indent=1)[:2500])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
