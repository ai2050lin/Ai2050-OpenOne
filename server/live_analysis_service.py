# -*- coding: utf-8 -*-
"""实时分析（live_trace.v1 / live_block.v1）：输入一段文本，在中心服务上跑一次真实前向传播，
逐层返回每个注意力头与 MLP 中间单元的响应，供「空间 · 实时分析」逐层步进观察每个 head 与 MLP。

数据真源：AI2050_LIVE_MODEL_DIR 指定的本机权重，工作台同时提交中心登记的预期快照。
没有权重、或不是连接本机中心时，端点诚实返回 available=False / 503，前端显示空态，不做任何模拟。

端点：
  GET  /api/live/status        模型可用性、加载状态与结构参数（idle = 权重在但未加载，不自动加载）
  POST /api/live/load          手动加载模型（占 GPU）
  POST /api/live/unload        手动卸载模型（释放 GPU）
  POST /api/live/run           {text, max_tokens} → 一次前向的逐层观察（live_trace.v1；未加载时自动后台加载）
  GET  /api/live/block?block=  该块逐头 W_q/W_o 与 MLP gate/up/down 的均值池化参数缩略图
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
router = APIRouter(prefix='/api/live', tags=['live'])

MAX_TOKENS_CAP = 2048
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
_LOAD_LOCK = threading.Lock()
_THREAD: Optional[threading.Thread] = None
GIB = 2**30


def model_dir() -> Optional[Path]:
    env = os.environ.get('AI2050_LIVE_MODEL_DIR', '').strip()
    if env and (Path(env) / 'config.json').is_file():
        return Path(env).resolve()
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


@contextmanager
def _checkpoint_pread(directory, enabled):
    """Keep Windows sharded BF16 reads off the fragile torch mmap path."""
    if not enabled or os.name != 'nt':
        yield
        return
    from accelerate.utils import offload
    from transformers import modeling_utils

    def for_checkpoint(original):
        def open_file(filename, *args, **kwargs):
            if Path(filename).parent == directory:
                kwargs['backend'] = 'pread'
            return original(filename, *args, **kwargs)
        return open_file

    transformers_open, accelerate_open = modeling_utils.safe_open, offload.safe_open
    modeling_utils.safe_open = for_checkpoint(transformers_open)
    offload.safe_open = for_checkpoint(accelerate_open)
    try:
        yield
    finally:
        modeling_utils.safe_open, offload.safe_open = transformers_open, accelerate_open


def _load_async(expected_snapshot=None) -> None:
    global _THREAD
    if _THREAD and _THREAD.is_alive():
        return
    d = model_dir()
    if d is None:
        _set(status='error', error='本机没有找到模型权重：设置 AI2050_LIVE_MODEL_DIR 指向含 config.json 的模型目录。')
        return
    _set(status='loading', progress='正在核验本机权重快照…', error='', dir=str(d),
         model_snapshot=None, expected_snapshot=expected_snapshot)

    def work():
        lease = None
        try:
            from deploy.parameter_reader import model_snapshot
            identity = model_snapshot(d)['sha256']
            if expected_snapshot and identity!=expected_snapshot:
                raise ValueError('本机权重快照与中心选择不一致')
            if json.loads((d / 'config.json').read_bytes()).get('model_type')!='qwen3':
                raise ValueError('当前原生采集适配器仅核对过 Qwen3。')
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from deploy.storage_policy import checked_dir, storage_dir
            from shared.dist_protocol.mechanism_trace import GpuLease
            lease=GpuLease().acquire()
            if not torch.cuda.is_available():raise RuntimeError('真实长文本分析需要CUDA，未回退到CPU或合成数据。')
            meta = _read_meta(d)
            _set(progress='读取 tokenizer…')
            tok = AutoTokenizer.from_pretrained(str(d), local_files_only=True)
            shards = sorted(d.glob('*.safetensors'))
            weight_bytes = sum(shard.stat().st_size for shard in shards)
            gpu_free, _ = torch.cuda.mem_get_info(0)
            placement = {'mode':'cuda_resident','weight_bytes':weight_bytes,'available_cuda_bytes_at_load':gpu_free}
            options = {'device_map': {'': 0}}
            streamed = weight_bytes > gpu_free * 0.8
            if streamed:
                if not (d / 'model.safetensors.index.json').is_file():
                    raise RuntimeError('显存不足，逐层调度需要带索引的原生 BF16 safetensors。')
                # The original shards back lazy weights. Embedding and readout stay on CUDA.
                folder = checked_dir(storage_dir('scratch','live-offload'), GIB, operation='实时权重调度')
                options.update(device_map={
                    'model.embed_tokens':0, 'model.norm':0, 'model.rotary_emb':0, 'lm_head':0,
                    **{f'model.layers.{i}':'disk' for i in range(meta['n_layers'])},
                }, offload_folder=str(folder))
                placement['mode'] = 'bf16_layer_offload'
                placement['checkpoint_read'] = 'pread' if os.name == 'nt' else 'mmap'
            _set(progress='准备 BF16 权重与逐层调度…' if placement['mode']=='bf16_layer_offload' else '读取 BF16 权重到显存…')
            print(json.dumps({'live_stage':'load','placement':placement},ensure_ascii=False),flush=True)
            t0 = time.monotonic()
            with _checkpoint_pread(d.resolve(), streamed):
                model = AutoModelForCausalLM.from_pretrained(str(d.resolve()), dtype=torch.bfloat16,
                    attn_implementation='sdpa', local_files_only=True, **options).eval()
            device = 'cuda:0'
            model.requires_grad_(False)
            placement.update(elapsed_seconds=round(time.monotonic()-t0,3),
                             device_map={k:str(v) for k,v in model.hf_device_map.items()})
            print(json.dumps({'live_stage':'ready','placement':placement},ensure_ascii=False),flush=True)
            _state['gpu_lease']=lease
            _set(status='ready', model=model, tokenizer=tok, device=device, meta=meta,
                 placement=placement, progress='', error='', model_snapshot=identity)
        except Exception as exc:                                # 权重损坏 / 显存不足等：诚实报错
            if lease:lease.release()
            print(json.dumps({'live_stage':'load_error','reason':str(exc)},ensure_ascii=False),flush=True)
            _set(status='error', progress='', error=f'模型加载失败：{exc}')

    _THREAD = threading.Thread(target=work, daemon=True, name='live-analysis-load')
    _THREAD.start()


def _ensure_ready(expected_snapshot=None) -> None:
    with _LOAD_LOCK:
        st = _state['status']
        if st == 'ready':
            return
        if st == 'idle':
            _load_async(expected_snapshot)
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
                'device': _state['device'], 'dir': _state['dir'], 'meta': dict(_state['meta']),
                'model_snapshot': _state.get('model_snapshot') if _state['status']=='ready' else None}
    current_dir = Path(snap['dir']) if snap['status'] in ('loading', 'ready') and snap['dir'] else d
    try:
        if snap['status'] != 'ready' or not snap['meta']:
            snap['meta'] = _read_meta(current_dir)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return {'schema': 'live_status.v1', 'available': False, 'status': 'unavailable',
                'model_dir': str(current_dir), 'model_name': current_dir.name, 'meta': {},
                'reason': f'本机模型配置无法读取：{exc}'}
    out = {'schema': 'live_status.v1', 'available': True, 'model_dir': str(current_dir),
           'model_name': current_dir.name, 'snapshot_guard':True, **snap}
    if snap['status'] == 'ready' and snap['device'].startswith('cuda'):   # 本服务进程当前的显存占用
        try:
            import torch
            out['vram_gb'] = round(torch.cuda.memory_allocated() / 1e9, 1)
            out['placement'] = _state.get('placement')
        except Exception:
            pass
    return out


class LiveLoad(BaseModel):
    expected_snapshot: str | None = Field(None, pattern=r'^[a-f0-9]{64}$')


@router.post('/load')
def live_load(body: LiveLoad | None = None) -> dict:
    """手动加载模型到 GPU（空间·实时分析界面按钮；加载后才能运行分析）。"""
    if model_dir() is None:
        raise HTTPException(503, '本机没有找到模型权重（AI2050_LIVE_MODEL_DIR），无法加载。')
    expected = body.expected_snapshot if body else None
    with _LOAD_LOCK:
        st = _state['status']
        if st in ('ready','loading'):
            current = _state.get('model_snapshot') if st=='ready' else _state.get('expected_snapshot')
            if expected and current!=expected:
                raise HTTPException(409,'本机正在使用另一快照，请先卸载并核对模型目录。')
            return {'ok': True, 'status': st, 'message': '模型已加载。' if st=='ready' else '模型正在加载中…'}
        _load_async(expected)
    return {'ok': True, 'status': _state['status'], 'message': '开始后台加载模型。'}


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
        if str(device).startswith('cuda') and torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as exc:
        _set(model=None, tokenizer=None, status='idle', error=str(exc))
        lease=_state.pop('gpu_lease',None)
        if lease:lease.release()
        return {'ok': False, 'status': 'idle', 'message': f'卸载时出现问题（引用已清除）：{exc}'}
    _set(status='idle', progress='模型已卸载（GPU 显存已释放），需要时重新加载。', error='')
    lease=_state.pop('gpu_lease',None)
    if lease:lease.release()
    return {'ok': True, 'status': 'idle', 'message': '模型已卸载，GPU 显存已释放。'}


@router.post('/unload')
def safe_live_unload():
    if not _RUN_LOCK.acquire(blocking=False):raise HTTPException(409,'正在采集真实轨迹，请等待当前任务结束再卸载。')
    try:return live_unload()
    finally:_RUN_LOCK.release()


class LiveRun(BaseModel):
    text: str = Field(..., min_length=1, max_length=40000)
    max_tokens: int = Field(2048, ge=1, le=MAX_TOKENS_CAP)
    readout_positions: list[int] | None = None
    prompt_format: Literal['chat','rendered'] = 'chat'
    expected_snapshot: str | None = Field(None, pattern=r'^[a-f0-9]{64}$')


def _r(x: float, nd: int = 4) -> float:
    return round(float(x), nd)


def legacy_live_run(b: LiveRun) -> dict:
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


@router.post('/run')
def live_run(b:LiveRun)->dict:
    """Same native recorder as research runs, SDPA, explicit limits, durable trace."""
    if not _RUN_LOCK.acquire(blocking=False):raise HTTPException(409,'已有真实采集在执行。')
    try:
        _ensure_ready(b.expected_snapshot)
        from deploy.parameter_reader import model_snapshot
        snap=model_snapshot(_state['dir'])['sha256']
        if snap!=_state.get('model_snapshot') or b.expected_snapshot and snap!=b.expected_snapshot:
            raise HTTPException(409,'本机权重快照与已加载或所选模型不符，未执行前向。')
        from shared.dist_protocol.mechanism_trace import Recorder,digest
        from deploy.storage_policy import checked_dir
        from server.mechanism_trace_service import store
        import torch
        tok=_state['tokenizer'];model=_state['model']
        if model.config.model_type!='qwen3':raise HTTPException(422,'当前原生采集适配器仅核对过Qwen3。')
        prompt=b.text if b.prompt_format=='rendered' else tok.apply_chat_template(
            [{'role':'user','content':b.text}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        ids=tok.encode(prompt,add_special_tokens=False)
        if len(ids)>b.max_tokens:raise HTTPException(422,f'完整实际输入为{len(ids)} token，超过所选上限{b.max_tokens}；未截断、未执行。')
        positions=b.readout_positions or sorted({0,len(ids)//2,len(ids)-1})
        if len(positions)>32 or len(set(positions))!=len(positions) or any(type(p)is not int or p<0 or p>=len(ids) for p in positions):raise HTTPException(422,'需1–32个不重复且有效的模块读出位置。')
        if not positions:raise HTTPException(422,'至少一个模块读出位置。')
        positions=sorted(positions)
        item={'id':'live-'+str(time.time_ns()),'text':b.text,'prompt':prompt,'token_ids':ids,'split':'unregistered_live','scope':'实时观察'}
        trace_root=Path(os.environ.get('AI2050_LIVE_TRACE_DIR') or ROOT/'tests/glm5/result/live_traces')
        folder=checked_dir(trace_root/digest(item),GIB,operation='实时原生轨迹')
        import subprocess
        def git(*args):
            return subprocess.check_output(['git','-c',f'safe.directory={ROOT}',*args],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
        try:revision=git('rev-parse','HEAD');dirty=bool(git('status','--porcelain'))
        except (OSError,subprocess.SubprocessError):revision=None;dirty=None
        provenance={'git_sha':revision,'git_dirty':dirty,'model_path':_state['dir'],'model_revision':snap,'seed':None,
            'prompt_format':b.prompt_format,'placement':_state.get('placement'),
            'code_identity_status':'git revision plus working-tree changes' if revision else 'git identity unavailable; unregistered observation'}
        t0=time.perf_counter()
        print(json.dumps({'live_stage':'native_capture','tokens':len(ids),'positions':positions}),flush=True)
        streamed = provenance['placement']['mode'] == 'bf16_layer_offload'
        with _checkpoint_pread(Path(_state['dir']).resolve(), streamed), Recorder(model,positions,folder,deep=True) as rec:
            with torch.inference_mode():out=model(input_ids=torch.tensor([ids],device=_state['device']),use_cache=False,output_hidden_states=True,logits_to_keep=1)
        d,path=rec.finish(out,item,tok,snap,'LIVE-NATIVE-OBSERVATION',provenance)
        if max(v['attention_relative_l2'] for v in d['reconstruction'].values())>=.05:raise HTTPException(409,'注意力重算与实际输出误差超出冻结容差，轨迹保留但不展示为通过。')
        store().register(path)
        return {**d,'elapsed_ms':int((time.perf_counter()-t0)*1000),'research_status':'unregistered observation, no mechanism claim','truncated':False}
    finally:_RUN_LOCK.release()


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

    def weight(module, key):
        if module.weight.device.type != 'meta':
            return module.weight
        from safetensors import safe_open
        directory = Path(_state['dir'])
        checkpoint = json.loads((directory/'model.safetensors.index.json').read_text(encoding='utf-8'))
        key = f'model.layers.{block}.{key}.weight'
        with safe_open(str(directory/checkpoint['weight_map'][key]), framework='pt', device='cpu',
                       backend='pread' if os.name=='nt' else 'mmap') as shard:
            return shard.get_tensor(key)

    def thumb(w, grid=THUMB_MLP):
        """均值池化缩略图：g1×g2 网格上的带符号均值（float32 cpu）。"""
        with torch.no_grad():
            x = w.detach().to(torch.float32).cpu()
        g1, g2 = grid
        r1, r2 = x.shape[0] // g1, x.shape[1] // g2
        x = x[: g1 * r1, : g2 * r2].reshape(g1, r1, g2, r2).mean(dim=(1, 3))
        return [[_r(v, 3) for v in row] for row in x.tolist()]

    q_weight = weight(sa.q_proj,'self_attn.q_proj')
    o_weight = weight(sa.o_proj,'self_attn.o_proj')
    heads = []
    for h in range(H):
        q = q_weight[h * hd: (h + 1) * hd, :]
        o = o_weight[:, h * hd: (h + 1) * hd].T
        heads.append({'q': thumb(q, THUMB_HEAD), 'o': thumb(o, THUMB_HEAD)})
    mlp_doc = {'gate': thumb(weight(mlp.gate_proj,'mlp.gate_proj')),
               'up': thumb(weight(mlp.up_proj,'mlp.up_proj')),
               'down': thumb(weight(mlp.down_proj,'mlp.down_proj').T)}
    return {'schema': 'live_block.v1', 'block': block, 'n_heads': H, 'n_kv_heads': KV, 'head_dim': hd,
            'heads': heads, 'mlp': mlp_doc}
