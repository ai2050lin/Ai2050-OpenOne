# -*- coding: utf-8 -*-
"""实时分析（live_trace.v1 / live_block.v1）：输入一段文本，在中心服务上跑一次真实前向传播，
逐层返回每个注意力头与 MLP 中间单元的响应，供「空间 · 实时分析」逐层步进观察每个 head 与 MLP。

数据真源：本机模型权重（AI2050_LIVE_MODEL_DIR，缺省 D:/develop/model/Qwen/Qwen3-4B ——
与自然语料普查 model_key=6be1034f0b55a6df-bf16 同款的 Qwen3-4B：36 块、32 头、8 KV 头、d_ff 9728）。
没有权重、或不是连接本机中心时，端点诚实返回 available=False / 503，前端显示空态，不做任何模拟。

端点：
  GET  /api/live/status        模型可用性、加载状态与结构参数（idle = 权重在但未加载，不自动加载）
  POST /api/live/load          手动加载模型（占 GPU）
  POST /api/live/unload        手动卸载模型（释放 GPU）
  POST /api/live/run           {text, max_tokens} → 一次前向的逐层观察（live_trace.v1；未加载时自动后台加载）
  GET  /api/live/block?block=  该块逐头 W_q/W_o 与 MLP gate/up/down 的均值池化参数缩略图
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
router = APIRouter(prefix='/api/live', tags=['live'])

DEFAULT_MODEL_DIR = 'D:/develop/model/Qwen/Qwen3-4B'
MAX_TOKENS_CAP = 32
TOP_MLP = 16
TOP_PAIRS = 3
THUMB_HEAD = (8, 8)      # 每头 W_q / W_o 缩略图网格（head_dim 方向 × d_model 方向）
THUMB_MLP = (8, 16)      # MLP gate/up/down 缩略图网格（d_ff 方向 × d_model 方向）

_state: Dict[str, Any] = {
    'status': 'idle',        # idle | loading | ready | error
    'error': '', 'progress': '',
    'model': None, 'tokenizer': None, 'device': '', 'dir': '', 'meta': {},
}
_LOCK = threading.Lock()          # 状态迁移与触发加载
_RUN_LOCK = threading.Lock()      # 前向串行（一块 GPU 一次一条）
_THREAD: Optional[threading.Thread] = None


def model_dir() -> Optional[Path]:
    cands = []
    env = os.environ.get('AI2050_LIVE_MODEL_DIR', '').strip()
    if env:
        cands.append(Path(env))
    cands.append(Path(DEFAULT_MODEL_DIR))
    native = os.environ.get('AI2050_NATIVE_DIR', 'D:/ai2050_native')
    cands.append(Path(native) / 'models' / 'hf' / 'qwen3-4b')
    for c in cands:
        try:
            if (c / 'config.json').exists():
                return c
        except OSError:
            continue
    return None


def _read_meta(d: Path) -> Dict[str, int]:
    cfg = json.loads((d / 'config.json').read_text(encoding='utf-8'))
    return {'n_layers': int(cfg['num_hidden_layers']), 'n_heads': int(cfg['num_attention_heads']),
            'n_kv_heads': int(cfg.get('num_key_value_heads') or cfg['num_attention_heads']),
            'head_dim': int(cfg.get('head_dim') or cfg['hidden_size'] // cfg['num_attention_heads']),
            'd_model': int(cfg['hidden_size']), 'd_ff': int(cfg['intermediate_size'])}


def _set(**kw) -> None:
    with _LOCK:
        _state.update(kw)


def _load_async() -> None:
    global _THREAD
    if _THREAD and _THREAD.is_alive():
        return
    d = model_dir()
    if d is None:
        _set(status='error', error='本机没有找到模型权重：设置 AI2050_LIVE_MODEL_DIR 指向含 config.json 的模型目录。')
        return
    _set(status='loading', progress='正在加载模型权重（首次约 10–30 秒）…', error='', dir=str(d))

    def work():
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            meta = _read_meta(d)
            _set(progress='读取 tokenizer…')
            tok = AutoTokenizer.from_pretrained(str(d))
            _set(progress='读取权重到显存（bf16）…')
            try:
                model = AutoModelForCausalLM.from_pretrained(str(d), dtype=torch.bfloat16, attn_implementation='eager')
            except TypeError:                                   # 旧版 transformers 用 torch_dtype
                model = AutoModelForCausalLM.from_pretrained(str(d), torch_dtype=torch.bfloat16, attn_implementation='eager')
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model.to(device).eval()
            _set(status='ready', model=model, tokenizer=tok, device=device, meta=meta,
                 progress='', error='')
        except Exception as exc:                                # 权重损坏 / 显存不足等：诚实报错
            _set(status='error', error=f'模型加载失败：{exc}')

    _THREAD = threading.Thread(target=work, daemon=True, name='live-analysis-load')
    _THREAD.start()


def _ensure_ready() -> None:
    st = _state['status']
    if st == 'ready':
        return
    if st == 'idle':
        _load_async()
        st = _state['status']
    if st == 'loading':
        raise HTTPException(503, '实时分析模型正在后台加载，请几秒后重试。')
    raise HTTPException(503, _state['error'] or '实时分析模型不可用。')


@router.get('/status')
def live_status() -> dict:
    d = model_dir()
    if d is None:
        return {'schema': 'live_status.v1', 'available': False, 'status': 'unavailable',
                'reason': '本机没有找到模型权重（AI2050_LIVE_MODEL_DIR），实时分析需要在本机中心上运行。',
                'model_dir': '', 'device': '', 'meta': {}}
    with _LOCK:
        snap = {'status': _state['status'], 'progress': _state['progress'], 'error': _state['error'],
                'device': _state['device'], 'dir': _state['dir'], 'meta': dict(_state['meta'])}
    out = {'schema': 'live_status.v1', 'available': True, 'model_dir': snap['dir'],
           'model_name': Path(snap['dir']).name if snap['dir'] else '', **snap}
    if snap['status'] == 'ready' and snap['device'] == 'cuda':   # 本服务进程当前的显存占用
        try:
            import torch
            out['vram_gb'] = round(torch.cuda.memory_allocated() / 1e9, 1)
        except Exception:
            pass
    return out


@router.post('/load')
def live_load() -> dict:
    """手动加载模型到 GPU（空间·实时分析界面按钮；加载后才能运行分析）。"""
    if model_dir() is None:
        raise HTTPException(503, '本机没有找到模型权重（AI2050_LIVE_MODEL_DIR），无法加载。')
    st = _state['status']
    if st == 'ready':
        return {'ok': True, 'status': 'ready', 'message': '模型已加载。'}
    if st == 'loading':
        return {'ok': True, 'status': 'loading', 'message': '模型正在加载中…'}
    _load_async()
    return {'ok': True, 'status': _state['status'], 'message': '开始后台加载模型。'}


@router.post('/unload')
def live_unload() -> dict:
    """手动卸载模型并释放 GPU 显存（前端会同步清掉旧 trace：卸载后参数缩略图无法继续取）。"""
    st = _state['status']
    if st == 'loading':
        raise HTTPException(409, '模型正在加载中，请等加载完成后再卸载。')
    if st != 'ready':
        return {'ok': True, 'status': st, 'message': '模型本来就没有加载。'}
    model = _state.get('model')
    try:
        import gc
        import torch
        device = _state.get('device')
        del model
        _set(model=None, tokenizer=None)
        gc.collect()
        if device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        _set(model=None, tokenizer=None)
        return {'ok': False, 'status': 'idle', 'message': f'卸载时出现问题（引用已清除）：{exc}'}
    _set(status='idle', progress='模型已卸载（GPU 显存已释放），需要时重新加载。', error='')
    return {'ok': True, 'status': 'idle', 'message': '模型已卸载，GPU 显存已释放。'}


class LiveRun(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    max_tokens: int = Field(24, ge=1, le=MAX_TOKENS_CAP)


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


@router.post('/run')
def live_run(b: LiveRun) -> dict:
    _ensure_ready()
    import torch
    model = _state['model']
    tok = _state['tokenizer']
    device = _state['device']
    meta = _state['meta']
    enc = tok(b.text, return_tensors='pt', add_special_tokens=True)
    ids = enc['input_ids'][0]
    truncated = len(ids) > b.max_tokens
    ids = ids[: b.max_tokens]
    T = int(len(ids))
    if T == 0:
        raise HTTPException(400, '这段文本没有产出任何 token。')
    H, KV, hd, d_model = meta['n_heads'], meta['n_kv_heads'], meta['head_dim'], meta['d_model']
    L = meta['n_layers']
    d_ff = meta['d_ff']

    layers = L and list(model.model.layers)
    if len(layers) != L:
        raise HTTPException(500, f'模型层数与配置不一致：config={L}，实际={len(layers)}')

    resid_in = [None] * L
    head_out = [None] * L
    mlp_top = [None] * L
    mlp_mean = [None] * L
    final_in = [None]

    def mk_resid(i):
        def hook(_m, args):
            x = args[0].detach().to(torch.float32).cpu().numpy()[0]        # [T, d_model]
            resid_in[i] = np.linalg.norm(x, axis=-1).round(4).tolist()
        return hook

    def mk_head(i):
        def hook(_m, args):
            x = args[0].detach().to(torch.float32).cpu().numpy()[0]        # [T, H*hd]
            x = x.reshape(T, H, hd)
            head_out[i] = np.linalg.norm(x, axis=-1).T.round(4).tolist()   # [H][T]
        return hook

    def mk_mlp(i):
        def hook(_m, args):
            x = args[0].detach().to(torch.float32).cpu().numpy()[0]        # [T, d_ff]
            k = min(TOP_MLP, d_ff)
            idx = np.argpartition(-np.abs(x), k - 1, axis=-1)[:, :k]
            val = np.take_along_axis(x, idx, axis=-1)
            order = np.argsort(-np.abs(val), axis=-1)
            idx = np.take_along_axis(idx, order, axis=-1)
            val = np.take_along_axis(val, order, axis=-1)
            mlp_top[i] = [[(int(u), _r(v)) for u, v in zip(row_i, row_v)] for row_i, row_v in zip(idx, val)]
            mlp_mean[i] = np.abs(x).mean(axis=-1).round(4).tolist()
        return hook

    def mk_final(_m, args):
        x = args[0].detach().to(torch.float32).cpu().numpy()[0]
        final_in[0] = np.linalg.norm(x, axis=-1).round(4).tolist()

    handles = []
    try:
        for i, layer in enumerate(layers):
            handles.append(layer.register_forward_pre_hook(mk_resid(i)))
            handles.append(layer.self_attn.o_proj.register_forward_pre_hook(mk_head(i)))
            handles.append(layer.mlp.down_proj.register_forward_pre_hook(mk_mlp(i)))
        handles.append(model.model.norm.register_forward_pre_hook(mk_final))
        t0 = time.perf_counter()
        with _RUN_LOCK, torch.no_grad():
            out = model(input_ids=ids.unsqueeze(0).to(device), use_cache=False, output_attentions=True)
            atts = out.attentions                                            # L × [1, H, T, T]
            logits = out.logits[0].detach().to(torch.float32).cpu()          # [T, V]
    finally:
        for h in handles:
            h.remove()

    probs = torch.softmax(logits, dim=-1)
    per_pos_top1 = [tok.decode([int(probs[t].argmax())]) for t in range(T)]
    k = min(8, probs.shape[-1])
    last_top = [(tok.decode([int(i)]), _r(float(probs[T - 1, i])), int(i)) for i in probs[T - 1].topk(k).indices]

    layer_docs = []
    for i in range(L):
        a = atts[i][0].detach().to(torch.float32).cpu().numpy()              # [H, T, T]
        attn_last = a[:, T - 1, :]
        pairs = []
        for h in range(H):
            flat = a[h].ravel()
            top = np.argsort(-flat)[:TOP_PAIRS]
            pairs.append([[int(t // T), int(t % T), _r(float(flat[t]), 3)] for t in top])
        layer_docs.append({
            'block': i,
            'resid_in': resid_in[i],
            'heads': {'out_norm': head_out[i], 'attn_last': attn_last.round(4).tolist(), 'top_pairs': pairs},
            'mlp': {'top': mlp_top[i], 'mean_abs': mlp_mean[i]},
        })

    return {'schema': 'live_trace.v1', 'model_name': Path(_state['dir']).name, 'model_dir': _state['dir'],
            'device': device, 'n_layers': L, 'n_heads': H, 'n_kv_heads': KV, 'head_dim': hd,
            'd_model': d_model, 'd_ff': d_ff,
            'tokens': [tok.decode([int(t)]) for t in ids], 'token_ids': [int(t) for t in ids],
            'n_tokens': T, 'truncated': truncated, 'max_tokens': b.max_tokens,
            'elapsed_ms': int((time.perf_counter() - t0) * 1000),
            'layers': layer_docs,
            'output': {'final_norm': final_in[0], 'per_pos_top1': per_pos_top1, 'last_top': last_top}}


@router.get('/block')
def live_block(block: int) -> dict:
    _ensure_ready()
    import torch
    model = _state['model']
    meta = _state['meta']
    if block < 0 or block >= meta['n_layers']:
        raise HTTPException(404, '块号超出范围')
    layer = model.model.layers[block]
    sa, mlp = layer.self_attn, layer.mlp
    H, KV, hd = meta['n_heads'], meta['n_kv_heads'], meta['head_dim']

    def thumb(w, grid=THUMB_MLP):
        """均值池化缩略图：g1×g2 网格上的带符号均值（float32 cpu）。"""
        with torch.no_grad():
            x = w.detach().to(torch.float32).cpu()
        g1, g2 = grid
        r1, r2 = x.shape[0] // g1, x.shape[1] // g2
        x = x[: g1 * r1, : g2 * r2].reshape(g1, r1, g2, r2).mean(dim=(1, 3))
        return [[_r(v, 3) for v in row] for row in x.tolist()]

    heads = []
    for h in range(H):
        q = sa.q_proj.weight[h * hd: (h + 1) * hd, :]                       # [hd, d_model]
        o = sa.o_proj.weight[:, h * hd: (h + 1) * hd].T                     # [hd, d_model] 转置成同向
        heads.append({'q': thumb(q, THUMB_HEAD), 'o': thumb(o, THUMB_HEAD)})
    mlp_doc = {'gate': thumb(mlp.gate_proj.weight), 'up': thumb(mlp.up_proj.weight),
               'down': thumb(mlp.down_proj.weight.T)}
    return {'schema': 'live_block.v1', 'block': block, 'n_heads': H, 'n_kv_heads': KV, 'head_dim': hd,
            'heads': heads, 'mlp': mlp_doc}
