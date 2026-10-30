# -*- coding: utf-8 -*-
"""神经元上下游电路（neuron_circuit.v1）：按 Goodfire 口径验证一个神经元时看它的「前后」——
即跨层的上下游神经元（前一层写入者 / 后一层读取者），而不是 MLP 内部（MLP 内神经元并行计算，无先后）。

打分全部来自本机模型权重（复用实时分析已加载的 Qwen3-4B，不重复占显存）：
  目标神经元（块 L, 单元 i）读取方向 r = gate/up 第 i 行（写入残差的方向 w = down 第 i 行）。
  上游（块 L−1 的写入者 j，写在 hs[L]）→ 块 L 注意力头 OV 中转或残差直连 → 目标读取：
      直连   score = cos(W_down_{L−1}[j], r)
      经头 h score = cos(W_down_{L−1}[j], W_V[:,kv] @ (W_O[:,h]^T @ r))
  下游（块 L+1 的读取者 j 的 gate/up 行 r_j）← 块 L+1 注意力头 OV 中转或残差直连 ← 目标写入：
      直连   score = cos(r_j, w)
      经头 h score = cos(r_j @ W_O[:,h], W_V[:,kv]^T @ w)
分数是余弦 / 线性路径强度，属于「电路假设」——读得出 ≠ 控得住，最终确认需要干预实验。

端点：
  GET /api/neurons/circuit?block=&unit=&k=&model_key=&run=
      → {available, upstream:[{unit,score,via,head,…}], downstream:[…]}
      普查覆盖时附带每个单元的分型与解释等级（仅作标签，不参与打分）。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

from server import live_analysis_service as las

router = APIRouter(prefix='/api/neurons', tags=['neurons'])


def _top(scores, k: int, exclude: Optional[int] = None) -> List[Dict[str, Any]]:
    """取正分前 k：负余弦 = 语义相反，不属于「前后一致」的验证范围。"""
    import torch
    out = []
    if scores.numel() == 0:
        return out
    vals, idx = torch.topk(scores, min(int(scores.numel()), k + 1))
    for v, j in zip(vals.tolist(), idx.tolist()):
        if j == exclude or v <= 0 or len(out) >= k:
            continue
        out.append({'unit': int(j), 'score': round(float(v), 5)})
    return out


def _circuit(model, L: int, i: int, k: int) -> Dict[str, Any]:
    import torch
    layers = model.model.layers
    n_layers = len(layers)
    dev = next(model.parameters()).device
    cfg = model.config
    n_heads = int(cfg.num_attention_heads)
    n_kv = int(getattr(cfg, 'num_key_value_heads', 0) or n_heads)
    hd = int(getattr(cfg, 'head_dim', 0) or cfg.hidden_size // n_heads)
    group = n_heads // n_kv

    lyr = layers[L].mlp
    r_g = lyr.gate_proj.weight[i].detach().float()          # 目标读取方向（gate 行，(F,H) 的行 i）
    r_u = lyr.up_proj.weight[i].detach().float()
    w_out = lyr.down_proj.weight[:, i].detach().float()     # 目标写入方向（down (H,F) 的列 i）

    def head_cols(h: int) -> slice:
        return slice(h * hd, (h + 1) * hd)

    def kv_cols(h: int) -> slice:
        kvh = h // group
        return slice(kvh * hd, (kvh + 1) * hd)

    upstream: List[Dict[str, Any]] = []
    downstream: List[Dict[str, Any]] = []
    if L >= 1:
        Wd = layers[L - 1].mlp.down_proj.weight.detach().float().t()  # (F, H) 上游写入矩阵（列 j = 神经元 j 的写出方向）
        norms = Wd.norm(dim=1).clamp_min(1e-8)
        cands = []
        for tag, r in (('gate', r_g), ('up', r_u)):
            for s in _top((Wd @ r) / norms, k, exclude=i):
                cands.append({**s, 'via': 'direct'})
        for h in range(n_heads):
            a = layers[L].self_attn.o_proj.weight.detach().float()[:, head_cols(h)].T @ r_g
            v = layers[L].self_attn.v_proj.weight.detach().float().T[:, kv_cols(h)] @ a
            for s in _top((Wd @ v) / norms, k, exclude=i):
                cands.append({**s, 'via': 'head', 'head': h})
        cands.sort(key=lambda x: -x['score'])
        seen = set()
        for c in cands:
            key = c['unit']
            if key in seen:
                continue
            seen.add(key)
            upstream.append(c)
            if len(upstream) >= k:
                break
        del Wd
    if L + 1 < n_layers:
        mlp = layers[L + 1].mlp
        R = torch.cat([mlp.gate_proj.weight.detach().float(),
                       mlp.up_proj.weight.detach().float()], dim=0)   # (2F, H) 下游读取矩阵
        rnorms = R.norm(dim=1).clamp_min(1e-8)
        cands = []
        dir_scores = (R @ w_out) / rnorms
        half = dir_scores.shape[0] // 2
        for s in _top(torch.maximum(dir_scores[:half], dir_scores[half:]), k, exclude=i):
            cands.append({**s, 'via': 'direct'})
        W_o = layers[L + 1].self_attn.o_proj.weight.detach().float()  # (H, heads*hd)
        W_v = layers[L + 1].self_attn.v_proj.weight.detach().float().T  # (H, kv*hd)
        RWO = R @ W_o                                                 # (2F, heads*hd)
        for h in range(n_heads):
            b = W_v[:, kv_cols(h)].T @ w_out                          # (hd,)
            s_h = (RWO[:, head_cols(h)] @ b) / rnorms
            for s in _top(torch.maximum(s_h[:half], s_h[half:]), k, exclude=i):
                cands.append({**s, 'via': 'head', 'head': h})
        cands.sort(key=lambda x: -x['score'])
        seen = set()
        for c in cands:
            key = c['unit']
            if key in seen:
                continue
            seen.add(key)
            downstream.append(c)
            if len(downstream) >= k:
                break
        del R, RWO
    torch.cuda.empty_cache() if dev == 'cuda' else None
    return {'upstream': upstream, 'downstream': downstream,
            'write_norm': round(float(w_out.norm()), 3),
            'read_norm': round(float(r_g.norm()), 3)}


def _labels(model_key: Optional[str], run: Optional[str], hits: List[Dict[str, Any]], block: int) -> None:
    """普查覆盖时给单元补分型/解释标签（只作展示，不参与打分）。"""
    if not model_key or not run or not hits:
        return
    try:
        from server import neuron_census_service as ncs
        base = ncs.layer_base(model_key, run, block)
        if not base:
            return
        from shared.dist_protocol.neuron_layers import TYPES
        tnames = [f'{a} {b}' for a, b in TYPES]
        cols = base.get('columns') or {}
        types = cols.get('type') or []
        grades = {}
        try:
            for (mk, b, u), recs in ncs.expl_index().items():
                if mk == model_key and b == block and recs:
                    grades[u] = recs[-1]['grade']
        except Exception:
            pass
        for h in hits:
            u = h['unit']
            if u < len(types):
                t = int(types[u])
                h['type_name'] = tnames[t] if t < len(tnames) else str(t)
            if u in grades:
                h['explained'] = grades[u]
    except Exception:
        return


@router.get('/circuit')
def neuron_circuit(block: int, unit: int, k: int = 5, model_key: str = '', run: str = '') -> dict:
    try:
        las._ensure_ready()
    except Exception as exc:                                    # 模型未加载 / 不在本机中心
        return {'schema': 'neuron_circuit.v1', 'available': False, 'reason': str(getattr(exc, 'detail', exc))}
    with las._LOCK:
        model, meta = las._state['model'], las._state['meta']
        device, model_name = las._state['device'], Path(las._state['dir'] or '').name
    n_layers, d_ff = int(meta['n_layers']), int(meta['d_ff'])
    if not (0 <= block < n_layers) or not (0 <= unit < d_ff):
        return {'schema': 'neuron_circuit.v1', 'available': False,
                'reason': f'块/单元越界：本模型 {n_layers} 块、每块 {d_ff} 个 MLP 单元。'}
    if model_key and run:                                       # 结构守卫：电路打分只对同一结构的普查有效
        try:
            from server import neuron_census_service as ncs
            man = ncs.census_manifest(model_key, run)
            if int(man.get('n_layers', -1)) != n_layers or int(man.get('d_ff', -1)) != d_ff:
                return {'schema': 'neuron_circuit.v1', 'available': False,
                        'reason': f'该普查是 {man.get("n_layers")} 块 / d_ff {man.get("d_ff")}，与本机权重（{n_layers} 块 / d_ff {d_ff}）不是同一结构；电路打分只对本机 Qwen3-4B 有效。'}
        except Exception:
            pass
    out = _circuit(model, block, unit, max(1, min(int(k), 12)))
    _labels(model_key or None, run or None, out['upstream'], block - 1)
    _labels(model_key or None, run or None, out['downstream'], block + 1)
    for side, b in (('upstream', block - 1), ('downstream', block + 1)):
        for h in out[side]:
            h['block'] = b
    return {'schema': 'neuron_circuit.v1', 'available': True, 'model_name': model_name, 'device': device,
            'block': block, 'unit': unit, 'k': k, 'n_layers': n_layers, 'd_ff': d_ff, **out}
