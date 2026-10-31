# -*- coding: utf-8 -*-
"""transcoder.v1 训练材料采集：在普查语料上跑 Qwen3（截到第 L 块），保存第 L 块 MLP 的实际输入 x（post_attention_layernorm 输出）。

MLP 输出 y = MLP(x) 在训练时用该块 MLP 权重在 GPU 上重算（逐元素相同的计算），因此只存 x，磁盘减半。
窗口与普查一致：seq_len 512、每窗口前 skip_head 个位置不计入；训练窗口取各来源 token 流的开头，留出窗口取各来源末尾，
两者不重叠。产物写在节点本地（不进仓库）：
    <out>/x_train.f16 / x_eval.f16（np.memmap，float16，[N, d]）、tok_train.npy / tok_eval.npy（对应 token id）、
    eval_windows.npy（留出窗口的完整 token，供损失恢复评估）、manifest.json（模型快照、语料指纹、窗口来源、sha256）
显存不够整模型时用 accelerate 的 CPU 卸载（每次前向把各层权重依次搬上 GPU）。

用法：python scripts/transcoder_collect.py --model models/hf/qwen3-4b --layer 30 --train-tokens 2000000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _Stop(Exception):
    pass


def windows_from(stream: np.ndarray, n_win: int, seq: int, from_end: bool) -> np.ndarray:
    total = len(stream) // seq
    if n_win > total:
        raise SystemExit(f'来源只有 {total} 个窗口，不够 {n_win}')
    idx = range(total - n_win, total) if from_end else range(n_win)
    return np.stack([stream[i * seq:(i + 1) * seq] for i in idx]).astype(np.int64)


def load_truncated(model_path: str, n_layers: int, gpu_gib: float):
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_path)
    if n_layers:
        cfg.num_hidden_layers = n_layers
    free = torch.cuda.mem_get_info()[0] / 2**30
    model = AutoModelForCausalLM.from_pretrained(model_path, config=cfg, dtype=torch.bfloat16, device_map='cpu').eval()
    need = sum(p.numel() * 2 for p in model.parameters()) / 2**30
    if need * 1.15 < min(free, gpu_gib):
        model.to('cuda')
        mode = 'gpu'
    else:
        from accelerate import cpu_offload
        cpu_offload(model, execution_device=torch.device('cuda'))
        mode = 'cpu_offload'
    return model, cfg, {'placement': mode, 'weights_gib': round(need, 2), 'gpu_free_gib': round(free, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    ap.add_argument('--corpus', default='D:/ai2050_native/_census/corpus/2ca5f0a756d36c8a')
    ap.add_argument('--layer', type=int, default=30)
    ap.add_argument('--train-tokens', type=int, default=2_000_000)
    ap.add_argument('--eval-tokens', type=int, default=131_072)
    ap.add_argument('--seq', type=int, default=512)
    ap.add_argument('--skip-head', type=int, default=16)
    ap.add_argument('--batch', type=int, default=32, help='每次前向的窗口数')
    ap.add_argument('--gpu-gib', type=float, default=14.0, help='整模型放 GPU 的显存上限，超出则 CPU 卸载')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    import torch
    from shared.dist_protocol.census_runner import model_snapshot

    corpus = Path(args.corpus)
    man = json.loads((corpus / 'manifest.json').read_text(encoding='utf-8'))
    snap = model_snapshot(args.model)
    key = snap['sha256'][:16] + '-bf16'
    out = Path(args.out or os.path.join(os.environ.get('AI2050_NATIVE_DIR', 'D:/ai2050_native'), '_transcoder', key,
                                        f"{man['fingerprint']}-L{args.layer}"))
    out.mkdir(parents=True, exist_ok=True)
    per = args.seq - args.skip_head
    srcs = [s['source'] for s in man['sources']]
    n_tr = -(-args.train_tokens // (per * len(srcs)))
    n_ev = -(-args.eval_tokens // (per * len(srcs)))
    wins = {'train': [], 'eval': []}
    for s in srcs:
        stream = np.load(corpus / f'tokens_{s}.npz')['tokens']
        wins['train'].append(windows_from(stream, n_tr, args.seq, False))
        wins['eval'].append(windows_from(stream, n_ev, args.seq, True))
    wins = {k: np.concatenate(v) for k, v in wins.items()}

    model, cfg, place = load_truncated(args.model, args.layer + 1, args.gpu_gib)
    d = cfg.hidden_size
    grab = {}
    def pre(mod, inp):
        grab['x'] = inp[0][:, args.skip_head:].detach()
        raise _Stop
    hk = model.model.layers[args.layer].mlp.register_forward_pre_hook(pre)
    t0 = time.time()
    files = {}
    for split, W in wins.items():
        N = len(W) * per
        xm = np.lib.format.open_memmap(out / f'x_{split}.npy', mode='w+', dtype=np.float16, shape=(N, d))
        toks = W[:, args.skip_head:].reshape(-1).astype(np.int32)
        for b in range(0, len(W), args.batch):
            ids = torch.as_tensor(W[b:b + args.batch], device='cuda')
            try:
                with torch.no_grad():
                    model(input_ids=ids)
            except _Stop:
                pass
            x = grab.pop('x').float().reshape(-1, d).cpu().numpy().astype(np.float16)
            xm[b * per:b * per + len(x)] = x
            done = min(b + args.batch, len(W))
            if done % (args.batch * 8) == 0 or done == len(W):
                el = time.time() - t0
                print(f'@@progress {json.dumps({"split": split, "windows": done, "of": len(W), "seconds": round(el, 1)})}', flush=True)
        xm.flush()
        del xm
        np.save(out / f'tok_{split}.npy', toks)
        files[f'x_{split}.npy'] = None
        files[f'tok_{split}.npy'] = None
    np.save(out / 'eval_windows.npy', wins['eval'])
    hk.remove()
    for f in list(files) + ['eval_windows.npy']:
        h = hashlib.sha256()
        with open(out / f, 'rb') as fh:
            for chunk in iter(lambda: fh.read(1 << 24), b''):
                h.update(chunk)
        files[f] = h.hexdigest()
    manifest = {'schema': 'transcoder_collect.v1', 'model': args.model, 'model_snapshot': snap['sha256'], 'model_key': key,
                'corpus_fp': man['fingerprint'], 'sources': srcs, 'layer': args.layer,
                'site': 'mlp input (post_attention_layernorm output)', 'seq': args.seq, 'skip_head': args.skip_head,
                'windows_per_source': {'train': n_tr, 'eval': n_ev}, 'window_rule': 'train = first windows of each stream; eval = last windows',
                'n_tokens': {'train': int(len(wins['train']) * per), 'eval': int(len(wins['eval']) * per)},
                'dtype_store': 'float16', 'compute_dtype': 'bfloat16', 'placement': place, 'torch': torch.__version__,
                'seconds': round(time.time() - t0, 1), 'files_sha256': files}
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'out': str(out), 'n_tokens': manifest['n_tokens'], 'placement': place, 'seconds': manifest['seconds']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
