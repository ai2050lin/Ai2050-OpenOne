# -*- coding: utf-8 -*-
"""transcoder.v1 训练：用 transcoder_collect.py 采集的第 L 块 MLP 输入 x，在 GPU 上用该块 MLP 权重重算 y = MLP(x)，训练 TopK transcoder。

数据按块搬上 GPU（每块 chunk_tokens 个 token，块内打乱，块序每轮重排），不需要把全部 x 放进内存。
产物：<collect_dir>/tc-m<m>-k<k>-s<seed>/transcoder.npz + train.json（超参、曲线、数据 sha256、用时）。

用法：python scripts/transcoder_train.py --collect D:/ai2050_native/_transcoder/<key>/<fp>-L30 --m 16384 --k 64 --steps 6000
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_mlp(model_path: str, layer: int, device: str):
    """只读第 layer 块 MLP 的三个权重（safetensors 按名取），返回 y = MLP(x) 与 A(x) 两个函数。"""
    import torch
    from safetensors import safe_open
    idx = json.loads((Path(model_path) / 'model.safetensors.index.json').read_text(encoding='utf-8'))['weight_map']
    W = {}
    for n in ('gate_proj', 'up_proj', 'down_proj'):
        key = f'model.layers.{layer}.mlp.{n}.weight'
        with safe_open(str(Path(model_path) / idx[key]), framework='pt', device='cpu') as f:
            W[n] = f.get_tensor(key).to(device)                      # bf16，与模型前向相同
    def acts(x):
        xb = x.to(torch.bfloat16)
        return torch.nn.functional.silu(xb @ W['gate_proj'].T) * (xb @ W['up_proj'].T)
    def mlp(x):
        return acts(x) @ W['down_proj'].T
    return mlp, acts, W


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--collect', required=True)
    ap.add_argument('--m', type=int, default=16384)
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--steps', type=int, default=6000)
    ap.add_argument('--batch', type=int, default=4096)
    ap.add_argument('--lr', type=float, default=4e-4)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--chunk-tokens', type=int, default=500_000)
    args = ap.parse_args()
    import torch
    from shared.dist_protocol import transcoder as T

    cdir = Path(args.collect)
    man = json.loads((cdir / 'manifest.json').read_text(encoding='utf-8'))
    X = np.load(cdir / 'x_train.npy', mmap_mode='r')
    N, d = X.shape
    mlp, _, _ = load_mlp(man['model'], man['layer'], 'cuda')
    rng = np.random.default_rng(args.seed)
    starts = list(range(0, N, args.chunk_tokens))
    state = {'order': [], 'buf': None, 'perm': None, 'pos': 0}

    def next_chunk():
        if not state['order']:
            state['order'] = list(rng.permutation(len(starts)))
        s = starts[state['order'].pop()]
        state['buf'] = torch.as_tensor(np.asarray(X[s:s + args.chunk_tokens]), device='cuda')
        state['perm'] = torch.as_tensor(rng.permutation(len(state['buf'])), device='cuda')
        state['pos'] = 0

    def sample(n):
        if state['buf'] is None or state['pos'] + n > len(state['buf']):
            next_chunk()
        i = state['perm'][state['pos']:state['pos'] + n]
        state['pos'] += n
        x = state['buf'][i].float()
        with torch.no_grad():
            y = mlp(x).float()
        return x, y

    t0 = time.time()
    tc, curve = T.train(sample, d_in=d, d_out=d, m=args.m, k=args.k, steps=args.steps, batch=args.batch, lr=args.lr,
                        device='cuda', seed=args.seed, log=lambda r: print('@@progress ' + json.dumps(r), flush=True))
    out = cdir / f'tc-m{args.m}-k{args.k}-s{args.seed}'
    out.mkdir(exist_ok=True)
    np.savez(out / 'transcoder.npz', **tc.state())
    info = {'schema': T.SCHEMA, 'criteria': T.CRITERIA, 'collect': man['files_sha256'], 'model_snapshot': man['model_snapshot'],
            'layer': man['layer'], 'm': args.m, 'k': args.k, 'steps': args.steps, 'batch': args.batch, 'lr': args.lr,
            'seed': args.seed, 'tokens_seen': args.steps * args.batch, 'train_tokens': int(N),
            'epochs': round(args.steps * args.batch / N, 2), 'curve': curve, 'seconds': round(time.time() - t0, 1),
            'torch': torch.__version__}
    (out / 'train.json').write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'out': str(out), 'final': curve[-1], 'seconds': info['seconds']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
