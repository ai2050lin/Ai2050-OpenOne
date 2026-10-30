# -*- coding: utf-8 -*-
"""神经元「前后锥」（neuron_bowtie.v1）：把 Goodfire 口径的「前后一致」验证做成一个可画的 3D 结构。

判据（Cammarata《Neural Networks Are Already Clean》的读法）：一个单元要算读得通，不能只看它自己的
读出标签，还要看
  ① 它检测的方向（读，r = gate 行）与它写出去的方向（写，w = down 列）是否同向 —— 自洽角 θ；
  ② 谁喂它：前一层写入者的写出方向，应当落在它的读方向上；
  ③ 谁用它：后一层读取者的读方向，应当落在它的写方向上。
②③ 若与 ① 同时成立，这个单元的「前后」是一致的；只在 ① 成立而在 ②③ 上发散，说明标签可能只是共现。

几何：读方向 r̂ 与写方向 ŵ 张开一个平面（张成空间是精确的二维，不是降维）：
  a1 = r̂，a2 = normalize(ŵ − (ŵ·r̂) r̂)        于是 ŵ 在该平面内 = (cosθ, sinθ)
  θ = 读/写夹角（自洽角）
每个邻居的方向 d̂（上游 = 它的写出方向，下游 = 它的读方向）投影到同一平面得 (u, v) = (d̂·a1, d̂·a2)：
  上游落在 (1, 0) 附近  ⇔  它写的正是目标读的
  下游落在 (cosθ, sinθ) 附近 ⇔ 它读的正是目标写的
所以「前后一致」在这个平面里就是一个收拢的蝴蝶结；发散则由 (u,v) 的横向偏离直接读出（off）。

标签（每个方向「是什么概念」）用 logit lens：r_j = W_U(γ ⊙ d̂)，取 top-k token。只是首阶读出，
不等于因果作用；分数同理是权重路径强度，最终确认需要干预实验。

端点：
  GET /api/neurons/bowtie?block=&unit=&k=&model_key=&run=
      → {available, target:{read,write,cos_rw,angle_deg,…}, upstream:[{u,v,align,off,tokens,…}],
         downstream:[…], consistency:{self_angle_deg,in_align,out_align,in_overlap,out_overlap,verdict}}
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter

from server import live_analysis_service as las
from server import neuron_circuit_service as ncs_circuit

router = APIRouter(prefix='/api/neurons', tags=['neurons'])


def _readouts(Wu, gamma, dirs, tok, n: int = 4) -> List[Dict[str, Any]]:
    """一批方向 → logit lens 的 top-n token（一次矩阵乘，方向已按列单位化）。"""
    import torch
    D = torch.stack(dirs, dim=1)                                  # (H, m)
    logits = (Wu @ (gamma.unsqueeze(1) * D).to(Wu.dtype)).float()  # (V, m)
    vals, ids = logits.topk(min(n, logits.shape[0]), dim=0)
    out = []
    for c in range(D.shape[1]):
        out.append({'tokens': [tok.decode([int(j)]) for j in ids[:, c].tolist()],
                    'logits': [round(float(v), 2) for v in vals[:, c].tolist()]})
    return out


def _unit(x):
    import torch
    return x / x.norm().clamp_min(1e-8)


def _cos(a, b) -> float:
    import torch
    return float(torch.dot(_unit(a), _unit(b)))


def _labels(model_key: Optional[str], run: Optional[str], hits: List[Dict[str, Any]], block: int) -> None:
    """普查覆盖时补分型 / 解释等级 / 普查自带的首位读出（只作标签，不参与打分）。"""
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
        ros = cols.get('ro') or []
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
            if u < len(ros):
                h['census_ro'] = ros[u]
            if u in grades:
                h['explained'] = grades[u]
    except Exception:
        return


def _overlap(a: List[str], b: List[str]) -> float:
    sa, sb = {x for x in a if x}, {x for x in b if x}
    if not sa or not sb:
        return 0.0
    return round(len(sa & sb) / len(sa | sb), 3)


@router.get('/bowtie')
def neuron_bowtie(block: int, unit: int, k: int = 6, model_key: str = '', run: str = '') -> dict:
    try:
        las._ensure_ready()
    except Exception as exc:
        return {'schema': 'neuron_bowtie.v1', 'available': False, 'reason': str(getattr(exc, 'detail', exc))}
    with las._LOCK:
        model, tok = las._state['model'], las._state['tokenizer']
        device, model_name = las._state['device'], Path(las._state['dir'] or '').name
    import torch
    meta = las._state['meta']
    n_layers, d_ff = int(meta['n_layers']), int(meta['d_ff'])
    if not (0 <= block < n_layers) or not (0 <= unit < d_ff):
        return {'schema': 'neuron_bowtie.v1', 'available': False,
                'reason': f'块/单元越界：本模型 {n_layers} 块、每块 {d_ff} 个 MLP 单元。'}
    if model_key and run:                                         # 结构守卫：与 circuit 同一口径
        try:
            from server import neuron_census_service as ncs
            man = ncs.census_manifest(model_key, run)
            if int(man.get('n_layers', -1)) != n_layers or int(man.get('d_ff', -1)) != d_ff:
                return {'schema': 'neuron_bowtie.v1', 'available': False,
                        'reason': f'该普查是 {man.get("n_layers")} 块 / d_ff {man.get("d_ff")}，与本机权重不是同一结构。'}
        except Exception:
            pass

    L, i = block, unit
    k = max(1, min(int(k), 10))
    layers = model.model.layers
    Wu = model.get_output_embeddings().weight.detach()
    gamma = model.model.norm.weight.detach().float()

    with torch.no_grad():
        r = layers[L].mlp.gate_proj.weight[i].detach().float()     # 读：这个单元检测的残差方向
        w = layers[L].mlp.down_proj.weight[:, i].detach().float()  # 写：它写回残差的方向
        rh, wh = _unit(r), _unit(w)
        cos_rw = float(torch.dot(rh, wh))
        theta = math.acos(max(-1.0, min(1.0, cos_rw)))
        a1 = rh
        t2 = wh - cos_rw * a1
        a2 = _unit(t2) if float(t2.norm()) > 1e-3 else _unit(torch.cat([r[1:], r[:1]]) if r.numel() > 1 else r)

        out = ncs_circuit._circuit(model, L, i, k)
        up_units, dn_units = out['upstream'], out['downstream']

        # 邻居方向：上游 = 前一层写入方向；下游 = 后一层读取方向
        up_dirs, dn_dirs = [], []
        if up_units:
            Wd = layers[L - 1].mlp.down_proj.weight.detach().float()
            up_dirs = [_unit(Wd[:, h['unit']]) for h in up_units]
        if dn_units:
            Rg = layers[L + 1].mlp.gate_proj.weight.detach().float()
            dn_dirs = [_unit(Rg[h['unit']]) for h in dn_units]

        r_up = layers[L].mlp.up_proj.weight[i].detach().float()
        reads = _readouts(Wu, gamma, [rh, wh, _unit(r_up)] + up_dirs + dn_dirs, tok, 4)
        tr_read, tr_write, tr_up = reads[0], reads[1], reads[2]
        n_up = len(up_dirs)

        up_nodes, dn_nodes = [], []
        for idx, h in enumerate(up_units):
            d = up_dirs[idx]
            u = float(torch.dot(d, a1)); v = float(torch.dot(d, a2))
            up_nodes.append({**h, 'block': L - 1, 'u': round(u, 4), 'v': round(v, 4),
                             'align': round(u, 4), 'off': round(abs(v), 4),
                             'tokens': reads[3 + idx]['tokens'],
                             'overlap': _overlap(reads[3 + idx]['tokens'], tr_read['tokens'])})
        sin_t = math.sin(theta)
        w_axis = a1 * cos_rw + a2 * sin_t                        # 写轴在平面内的坐标 (cosθ, sinθ)
        for idx, h in enumerate(dn_units):
            d = dn_dirs[idx]
            u = float(torch.dot(d, a1)); v = float(torch.dot(d, a2))
            align = float(torch.dot(d, w_axis))
            off = abs(v * cos_rw - u * sin_t)
            dn_nodes.append({**h, 'block': L + 1, 'u': round(u, 4), 'v': round(v, 4),
                             'align': round(align, 4), 'off': round(off, 4),
                             'tokens': reads[3 + n_up + idx]['tokens'],
                             'overlap': _overlap(reads[3 + n_up + idx]['tokens'], tr_write['tokens'])})

    _labels(model_key or None, run or None, up_nodes, L - 1)
    _labels(model_key or None, run or None, dn_nodes, L + 1)
    probe = [{'unit': i}]                                         # 目标自身只取普查标签
    _labels(model_key or None, run or None, probe, L)
    tgt_label = {k2: probe[0].get(k2) for k2 in ('type_name', 'explained', 'census_ro')}

    mean = lambda xs: round(sum(xs) / len(xs), 3) if xs else None
    in_align = mean([n['align'] for n in up_nodes])
    out_align = mean([n['align'] for n in dn_nodes])
    in_ov = mean([n['overlap'] for n in up_nodes])
    out_ov = mean([n['overlap'] for n in dn_nodes])
    deg = round(theta * 180 / 3.141592653589793, 1)
    if deg <= 30:
        self_say = f'读与写同向（{deg}°）：检测的概念和写出去的概念基本一致'
    elif deg <= 60:
        self_say = f'读与写部分偏离（{deg}°）：同一个概念域，但写入分量只覆盖检测方向的一部分'
    else:
        self_say = f'读与写明显不同向（{deg}°）：更像「检测 A、写出 B」的转换单元'
    verdict = (f'{self_say}；上游 {len(up_nodes)} 个写入者沿读轴平均对齐 {in_align if in_align is not None else "—"}'
               f'（token 重叠 {in_ov if in_ov is not None else "—"}），下游 {len(dn_nodes)} 个读取者沿写轴平均对齐'
               f' {out_align if out_align is not None else "—"}（token 重叠 {out_ov if out_ov is not None else "—"}）。')

    return {'schema': 'neuron_bowtie.v1', 'available': True, 'model_name': model_name, 'device': device,
            'block': L, 'unit': i, 'k': k, 'n_layers': n_layers, 'd_ff': d_ff,
            'target': {'block': L, 'unit': i, 'read': tr_read, 'write': tr_write, 'read_up': tr_up,
                       'cos_rw': round(cos_rw, 4), 'angle_deg': deg,
                       'write_axis': [round(cos_rw, 4), round(math.sin(theta), 4)], **tgt_label},
            'upstream': up_nodes, 'downstream': dn_nodes,
            'consistency': {'self_cos': round(cos_rw, 4), 'self_angle_deg': deg,
                            'in_align': in_align, 'out_align': out_align,
                            'in_overlap': in_ov, 'out_overlap': out_ov, 'verdict': verdict}}
