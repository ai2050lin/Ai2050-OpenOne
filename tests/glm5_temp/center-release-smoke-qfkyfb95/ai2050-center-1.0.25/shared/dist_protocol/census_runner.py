# -*- coding: utf-8 -*-
"""神经元普查分片 runner（census_job.v1，节点端）：中心随任务下发，参与者不需要也不应修改。

输入：job.json（作业规范）、tokens_S<s>.npz（该切片的窗口）、本机模型目录。输出：shard.npz（该层组每层的可合并普查数组，
键名 L<b>__<字段>）与 summary.json。只依赖同目录的 census.py、numpy、torch 与 transformers。

真实模式先核对模型快照（权重与 tokenizer 文件 sha256 的合成哈希）与作业一致，不一致直接退出；前向在层组最后一层之后停止。
synthetic 模式用确定性的伪激活代替模型，只用于协议联调，结果标为 synthetic，正式中心不接受。

用法：python census_runner.py --job job.json --shard S0-G1 --tokens tokens_S0.npz --model <模型目录> --out outputs
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import census as C  # noqa: E402  同目录（仓库内为 shared/dist_protocol/census.py，任务包内为同名副本）

SNAPSHOT_PATTERNS = ('*.safetensors', '*.bin', '*.pt', '*.gguf', 'config.json', 'generation_config.json',
                     'tokenizer*', 'vocab*', 'merges.txt', 'special_tokens_map.json', '*.model', '*.tiktoken')
CACHE = Path(os.environ.get('AI2050_HOME', str(Path.home() / '.ai2050'))) / 'snapshot_cache.json'


def _file_sha(p: Path) -> str:
    """大文件哈希按 (路径, 大小, 修改时间) 缓存，避免每个分片重算数 GB 权重。"""
    st = p.stat()
    key = f'{p.resolve()}|{st.st_size}|{int(st.st_mtime)}'
    try:
        cache = json.loads(CACHE.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        cache = {}
    if key in cache:
        return cache[key]
    h = hashlib.sha256()
    with p.open('rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b''):
            h.update(chunk)
    cache[key] = h.hexdigest()
    try:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_text(json.dumps(cache), encoding='utf-8')
    except OSError:
        pass
    return cache[key]


def model_snapshot(model_path) -> dict:
    """与 runner_v3.model_snapshot 同一口径（测试核对两者一致）。"""
    root = Path(model_path)
    files = {}
    if root.is_dir():
        seen = set()
        for pat in SNAPSHOT_PATTERNS:
            for p in sorted(root.glob(pat)):
                if p.is_file() and p.name not in seen:
                    seen.add(p.name)
                    files[p.name] = {'sha256': _file_sha(p), 'size': p.stat().st_size}
    total = hashlib.sha256(json.dumps(files, sort_keys=True).encode('utf-8')).hexdigest()
    return {'sha256': total, 'files': files, 'n_files': len(files)}


def _progress(done: int, total: int, n: int, t0: float) -> None:
    el = time.time() - t0
    print(f'@@progress {json.dumps({"done": done, "total": total, "tokens": n, "tok_per_s": round(n / max(el, 1e-6), 1), "seconds": round(el, 1)})}',
          flush=True)


def run(job: dict, sid: str, tokens_path: Path, model_path: str, out: Path, *, device: str = 'cuda',
        batch: int = 0, mode: str = 'real') -> dict:
    import torch
    sh = next(s for s in job['shards'] if s['shard'] == sid)
    layers_want = list(sh['layers'])
    F, k, sep, skip = int(job['d_ff']), int(job['k']), int(job['separator_id']), int(job['skip_head'])
    with np.load(tokens_path) as z:
        W, src, woff = z['windows'], z['src'], z['win_off']
    seq = W.shape[1]
    batch = batch or int(job.get('batch', 8))
    t0 = time.time()
    accs = {b: C.LayerAcc(F, k=k, device=device) for b in layers_want}
    pos = torch.arange(seq, device=device)
    env = {'python': platform.python_version(), 'torch': torch.__version__, 'device': device}
    snap = None
    if mode == 'real':
        snap = model_snapshot(model_path)
        if snap['sha256'] != job['model']['snapshot_sha256']:
            want = job['model'].get('files') or {}
            diff = sorted(n for n in set(want) | set(snap['files'])
                          if (want.get(n) or {}).get('sha256') != (snap['files'].get(n) or {}).get('sha256'))
            raise SystemExit(f'模型快照不一致：本机 {snap["sha256"][:16]} ≠ 作业 {job["model"]["snapshot_sha256"][:16]}；'
                             f'不同的文件：{diff[:8]}')
        import transformers
        from transformers import AutoModelForCausalLM
        env['transformers'] = transformers.__version__
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16, device_map=device)
        model.eval()
        env['attn_implementation'] = getattr(model.config, '_attn_implementation', None)
        if device.startswith('cuda'):
            env['gpu'] = torch.cuda.get_device_name(0)
        dec = list(model.get_decoder().layers)
        if len(dec) != job['n_layers'] or int(dec[0].mlp.gate_proj.out_features) != F:
            raise SystemExit('模型结构与作业不符')
        ctx = {'g': None, 'mask': None, 'gidx': None}

        class _Stop(Exception):
            pass

        def gate_hook(mod, a, o):
            ctx['g'] = o

        def make_up_hook(b):
            def up_hook(mod, a, o):
                m = ctx['mask']
                G = ctx['g'].float().reshape(-1, F)[m]
                V = o.float().reshape(-1, F)[m]
                accs[b].update(torch.nn.functional.silu(G) * V, G, V, ctx['gidx'])
                ctx['g'] = None
            return up_hook

        def stop_hook(mod, a, o):
            raise _Stop()
        for b in layers_want:
            dec[b].mlp.gate_proj.register_forward_hook(gate_hook)
            dec[b].mlp.up_proj.register_forward_hook(make_up_hook(b))
        dec[max(layers_want)].mlp.register_forward_hook(stop_hook)   # 层组最后一层的 MLP 之后不再前向

        def step(ids, mask, gidx):
            ctx['mask'], ctx['gidx'] = mask.reshape(-1), gidx
            try:
                model.model(input_ids=ids)
            except _Stop:
                pass
    elif mode == 'synthetic':
        env['note'] = 'synthetic：确定性伪激活，不经过模型'
        unit = torch.arange(F, device=device, dtype=torch.float32)

        def step(ids, mask, gidx):
            t = ids.reshape(-1)[mask.reshape(-1)].float()[:, None]
            for b in layers_want:
                G = torch.sin(t * 0.37 + unit[None, :] * 0.11 + b)
                V = torch.cos(t * 0.53 - unit[None, :] * 0.07 + 0.5 * b) * 2
                accs[b].update(torch.nn.functional.silu(G) * V, G, V, gidx)
    else:
        raise SystemExit(f'未知模式 {mode}')

    n_counted = 0
    last = 0.0
    with torch.inference_mode():
        for i in range(0, len(W), batch):
            ids = torch.tensor(W[i:i + batch], dtype=torch.long, device=device)
            off = torch.tensor(woff[i:i + batch], device=device)[:, None] + pos[None, :]
            s = torch.tensor(src[i:i + batch], device=device).long()[:, None].expand_as(off)
            mask = (pos[None, :] >= skip) & (ids != sep)
            gidx = ((s << C.SRC_SHIFT) | off.long()).reshape(-1)[mask.reshape(-1)]
            step(ids, mask, gidx)
            n_counted += int(mask.sum())
            if time.time() - last > 5 or i + batch >= len(W):
                _progress(min(i + batch, len(W)), len(W), n_counted, t0)
                last = time.time()
    t_fwd = time.time() - t0

    arrays = {}
    with torch.inference_mode():
        if sh['readout'] and mode == 'real':
            W_U, gamma = model.get_output_embeddings().weight, model.model.norm.weight
        for b in layers_want:
            d = accs[b].to_numpy()
            d['hist'] = d['hist'].astype(np.int32)
            if sh['readout']:
                if mode == 'real':
                    Wd = dec[b].mlp.down_proj.weight
                    top_ids, bot_ids = [], []
                    for s0 in range(0, F, 1024):
                        R = (W_U @ (gamma[:, None] * Wd[:, s0:s0 + 1024])).float()
                        top_ids.append(torch.topk(R, 10, dim=0).indices.T.cpu().numpy())
                        bot_ids.append(torch.topk(-R, 10, dim=0).indices.T.cpu().numpy())
                    d['readout_top'] = np.concatenate(top_ids).astype(np.int32)
                    d['readout_bot'] = np.concatenate(bot_ids).astype(np.int32)
                else:
                    d['readout_top'] = (np.arange(F)[:, None] * 7 + np.arange(10)[None, :]).astype(np.int32) % 1000
                    d['readout_bot'] = (np.arange(F)[:, None] * 11 + np.arange(10)[None, :]).astype(np.int32) % 1000
            for key, v in d.items():
                arrays[f'L{b}__{key}'] = v
    if device.startswith('cuda'):
        env['gpu_peak_gib'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
    out.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    data = buf.getvalue()
    (out / 'shard.npz').write_bytes(data)
    summary = {'schema': 'census_shard.v1', 'status': mode, 'job_id': job['job_id'], 'shard': sid,
               'layers': layers_want, 'n_tokens_counted': n_counted, 'expected_n': sh['expected_n'],
               'model_snapshot': snap['sha256'] if snap else None, 'shard_npz_sha256': hashlib.sha256(data).hexdigest(),
               'shard_npz_bytes': len(data), 'seconds_forward': round(t_fwd, 1), 'seconds_total': round(time.time() - t0, 1),
               'tok_per_s': round(n_counted / max(t_fwd, 1e-6), 1), 'env': env,
               'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               'census_sha256': hashlib.sha256((HERE / 'census.py').read_bytes()).hexdigest()}
    (out / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding='utf-8')
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='census_job.v1 分片 runner')
    ap.add_argument('--job', required=True)
    ap.add_argument('--shard', required=True)
    ap.add_argument('--tokens', required=True)
    ap.add_argument('--model', default=os.environ.get('AI2050_RUNNER_MODEL', ''))
    ap.add_argument('--out', default='outputs')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch', type=int, default=0)
    ap.add_argument('--mode', default='real', choices=['real', 'synthetic'])
    a = ap.parse_args(argv)
    if a.mode == 'real' and not a.model:
        raise SystemExit('真实模式需要 --model 指向本机模型目录')
    job = json.loads(Path(a.job).read_text(encoding='utf-8'))
    s = run(job, a.shard, Path(a.tokens), a.model, Path(a.out), device=a.device, batch=a.batch, mode=a.mode)
    print('@@result ' + json.dumps({k: s[k] for k in ('shard', 'n_tokens_counted', 'expected_n', 'tok_per_s', 'seconds_total')}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
