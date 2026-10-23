# -*- coding: utf-8 -*-
"""MLP 神经元自然语料普查（mlp_census.v1）：对模型全部层的全部 MLP 中间单元做可合并的激活统计。

语料来自 scripts/prepare_census_corpus.py 的 token 流（每个来源从头取 max_tokens/来源数 个 token），切成 seq_len 的
不重叠窗口独立前向（无 padding、无 KV 缓存）。每个窗口的前 skip_head 个位置与文档分隔符不计入统计（窗口开头的
注意力汇聚位会产生与语义无关的异常大激活）。在 gate_proj / up_proj 的输出上计算 A = silu(G) ⊙ V，交给
shared/dist_protocol/census.py 的累加器。另计算每个单元的直接输出读出 W_U (γ ⊙ W_down[:, j]) 的前 / 后 10 个 token
（γ 为最终 RMSNorm 增益；只是直接通路近似）。产物写在节点本地，研究记录只登记摘要。

用法：python scripts/run_mlp_census.py --model-path models/hf/qwen3-4b --corpus-fp <指纹> --max-tokens 1000000
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.dist_protocol import census as C  # noqa: E402
from shared.dist_protocol.runner_v3 import model_snapshot  # noqa: E402

NATIVE = Path(os.environ.get('AI2050_NATIVE_DIR', 'D:/ai2050_native'))


def main(argv=None) -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--model-path', required=True)
    ap.add_argument('--corpus-fp', required=True)
    ap.add_argument('--max-tokens', type=int, required=True, help='全部来源合计的 token 上限（各来源平分）')
    ap.add_argument('--seq-len', type=int, default=512)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--skip-head', type=int, default=16)
    ap.add_argument('--k', type=int, default=32)
    ap.add_argument('--summary-out', default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    cdir = NATIVE / '_census' / 'corpus' / args.corpus_fp
    cman = json.loads((cdir / 'manifest.json').read_text(encoding='utf-8'))
    sep = int(cman['separator_id'])
    per_src = args.max_tokens // len(cman['sources'])
    streams = []
    for sid, s in enumerate(cman['sources']):
        with np.load(cdir / s['file']) as z:
            toks = z['tokens'][:per_src]
        n_win = len(toks) // args.seq_len
        streams.append((sid, s['source'], toks[:n_win * args.seq_len].reshape(n_win, args.seq_len)))
    windows = [(sid, w) for sid, _, arr in streams for w in range(len(arr))]
    snap = model_snapshot(args.model_path)
    model_key = f"{snap['sha256'][:16]}-bf16"
    out_dir = NATIVE / '_census' / model_key / f"{args.corpus_fp}-{args.max_tokens}"
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    layers = list(model.get_decoder().layers)
    d_ff = int(layers[0].mlp.gate_proj.out_features)
    accs = [C.LayerAcc(d_ff, k=args.k, device='cuda') for _ in layers]
    ctx = {'g': None, 'mask': None, 'gidx': None}

    def gate_hook(mod, a, out):
        ctx['g'] = out

    def make_up_hook(b):
        def up_hook(mod, a, out):
            m = ctx['mask']
            G = ctx['g'].float().reshape(-1, d_ff)[m]
            V = out.float().reshape(-1, d_ff)[m]
            accs[b].update(torch.nn.functional.silu(G) * V, G, V, ctx['gidx'])
            ctx['g'] = None
        return up_hook
    for b, L in enumerate(layers):
        L.mlp.gate_proj.register_forward_hook(gate_hook)
        L.mlp.up_proj.register_forward_hook(make_up_hook(b))

    n_counted, last = 0, time.time()
    pos = torch.arange(args.seq_len, device='cuda')
    with torch.inference_mode():
        for i in range(0, len(windows), args.batch):
            chunk = windows[i:i + args.batch]
            ids = torch.tensor(np.stack([streams[sid][2][w] for sid, w in chunk]), dtype=torch.long, device='cuda')
            off = torch.tensor([w * args.seq_len for _, w in chunk], device='cuda')[:, None] + pos[None, :]
            src = torch.tensor([sid for sid, _ in chunk], device='cuda')[:, None].expand_as(off)
            mask = (pos[None, :] >= args.skip_head) & (ids != sep)
            ctx['mask'] = mask.reshape(-1)
            ctx['gidx'] = ((src.long() << C.SRC_SHIFT) | off.long()).reshape(-1)[ctx['mask']]
            model.model(input_ids=ids)                       # 只需各层 MLP 的中间激活，不计算 lm_head
            n_counted += int(mask.sum())
            if time.time() - last > 60 or i + args.batch >= len(windows):
                el = time.time() - t0
                print(f'[census] {i + len(chunk)}/{len(windows)} 窗口 · 计入 {n_counted} token · {n_counted / max(el, 1e-6):.0f} tok/s · '
                      f'显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB · {el:.0f}s', flush=True)
                last = time.time()
    t_fwd = time.time() - t0

    # 直接输出读出：W_U (γ ⊙ W_down[:, j]) 的前 / 后 10 个 token
    W_U = model.get_output_embeddings().weight                    # (vocab, d)
    gamma = model.model.norm.weight                               # (d,)
    files, summary_layers = [], []
    with torch.inference_mode():
        for b, L in enumerate(layers):
            Wd = L.mlp.down_proj.weight                           # (d, d_ff)
            top_ids, bot_ids = [], []
            for s in range(0, d_ff, 1024):
                R = (W_U @ (gamma[:, None] * Wd[:, s:s + 1024])).float()    # (vocab, chunk)
                top_ids.append(torch.topk(R, 10, dim=0).indices.T.cpu().numpy())
                bot_ids.append(torch.topk(-R, 10, dim=0).indices.T.cpu().numpy())
            d = accs[b].to_numpy()
            d['readout_top'] = np.concatenate(top_ids).astype(np.int32)
            d['readout_bot'] = np.concatenate(bot_ids).astype(np.int32)
            buf = io.BytesIO()
            np.savez(buf, **d)
            data = buf.getvalue()
            (out_dir / f'L{b}.npz').write_bytes(data)
            files.append({'path': f'L{b}.npz', 'block': b, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            summary_layers.append({'block': b, **C.summarize_layer(d)})
    import transformers
    manifest = {'schema': 'mlp_census.v1', 'model_snapshot': snap, 'model_key': model_key, 'model_path': str(Path(args.model_path).as_posix()),
                'corpus_fp': args.corpus_fp, 'corpus_manifest_sha256': hashlib.sha256((cdir / 'manifest.json').read_bytes()).hexdigest(),
                'sources': [{'source_id': sid, 'source': name, 'n_windows': len(arr)} for sid, name, arr in streams],
                'max_tokens': args.max_tokens, 'seq_len': args.seq_len, 'batch': args.batch, 'skip_head': args.skip_head,
                'k': args.k, 'd_ff': d_ff, 'n_layers': len(layers), 'n_tokens_counted': n_counted,
                'hist': {'n_side': C.N_SIDE, 'lo': C.LO, 'hi': C.HI, 'regimes': list(C.REGIMES)},
                'gidx': f'source_id << {C.SRC_SHIFT} | 来源 token 流内偏移', 'files': files,
                'seconds_forward': round(t_fwd, 1), 'seconds_total': round(time.time() - t0, 1),
                'tok_per_s': round(n_counted / max(t_fwd, 1e-6), 1),
                'gpu_peak_gib': round(torch.cuda.max_memory_allocated() / 2**30, 2),
                'torch': torch.__version__, 'transformers': transformers.__version__,
                'attn_implementation': getattr(model.config, '_attn_implementation', None), 'dtype': 'bf16',
                'note': '窗口独立前向，无 padding、无 KV 缓存；每个窗口前 skip_head 个位置与分隔符不计入；A = silu(G)⊙V'}
    mbytes = json.dumps(manifest, ensure_ascii=False, indent=1).encode('utf-8')
    (out_dir / 'manifest.json').write_bytes(mbytes)
    summary = {'schema': 'mlp_census_summary.v1', 'dir': str(out_dir), 'manifest_sha256': hashlib.sha256(mbytes).hexdigest(),
               **{k: manifest[k] for k in ('model_key', 'corpus_fp', 'max_tokens', 'n_tokens_counted', 'tok_per_s', 'gpu_peak_gib',
                                            'seconds_total', 'skip_head', 'seq_len')},
               'layers': summary_layers}
    if args.summary_out:
        Path(args.summary_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({k: summary[k] for k in ('dir', 'n_tokens_counted', 'tok_per_s', 'gpu_peak_gib', 'seconds_total')}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
