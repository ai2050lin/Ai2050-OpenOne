# -*- coding: utf-8 -*-
"""离线预计算跨层电路（neuron_circuit.v1）→ 附加包源文件。研发机执行（需要模型权重），服务器零计算。

背景：/api/neurons/circuit 按本机权重实时打分，但客户端（含网页版）是只读数据包模式，
正式服务器没有权重也没有该服务 → 离线显示「数据包里没有这项数据」。本脚本在研发机上
把电路打分批量预计算出来，写成 circuit_sources/<model_key>.circuit.v1.json.gz，上传到
服务器 <AI2050_DIST_DIR>/circuit_sources/ 后，datapack.build_extras 把它包成附加包
circuit:<model_key> 随数据包体系下发（deploy/datapack.py 的 pack_hint 已加 circuit 规则）。

覆盖集合：单元卡单元（card_units）∪ 每层按激活密度 top --per-layer（缺省 384）。
超出集合的单元离线不可用（客户端提示），实时模式（研发机 :5001）不受影响。

正确性：与 server/neuron_circuit_service._circuit 同一数学（gate/up 直连 + 注意力头 OV 中转，
正分 top-k、稳定排序、逐单元去重），并用 --verify 抽样直接调用 _circuit 比对。

用法（研发机，建议先停占用显存的服务）：
  .venv/Scripts/python.exe scripts/build_circuit_pack.py                 # 全部普查模型
  .venv/Scripts/python.exe scripts/build_circuit_pack.py --verify 3      # 抽样与实时实现比对
  .venv/Scripts/python.exe scripts/build_circuit_pack.py --gpu-gib 0     # 纯 CPU（慢）
产出：distributed_data/circuit_sources/<model_key>.circuit.v1.json.gz
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'deploy'))
sys.path.insert(0, str(ROOT))


def topk_rows(scores, k: int, exclude: int) -> list:
    """与 neuron_circuit_service._top 相同：正分前 k，排除自身。scores 任意设备，返回 CPU 列表。"""
    import torch
    if scores.numel() == 0:
        return []
    vals, idx = torch.topk(scores, min(int(scores.numel()), k + 1))
    vals, idx = vals.cpu().tolist(), idx.cpu().tolist()
    out = []
    for v, j in zip(vals, idx):
        if j == exclude or v <= 0 or len(out) >= k:
            continue
        out.append({'unit': int(j), 'score': round(float(v), 5)})
    return out


def merge_cands(cands: list, k: int) -> list:
    """与 _circuit 相同的合并：按分数降序（稳定排序），逐单元去重保最高，取前 k。"""
    cands = sorted(cands, key=lambda x: -x['score'])
    seen, out = set(), []
    for c in cands:
        if c['unit'] in seen:
            continue
        seen.add(c['unit'])
        out.append(c)
        if len(out) >= k:
            break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--model-key', default='', help='普查模型键（缺省：全部有普查结果的模型）')
    ap.add_argument('--out', default=str(ROOT / 'distributed_data' / 'circuit_sources'))
    ap.add_argument('--k', type=int, default=5)
    ap.add_argument('--per-layer', type=int, default=384, help='每层按激活密度补充的单元数上限')
    ap.add_argument('--verify', type=int, default=0, help='每模型抽样 n 个单元与实时 _circuit 比对')
    ap.add_argument('--gpu-gib', type=int, default=5, help='GPU 显存上限（GiB），0=纯 CPU')
    args = ap.parse_args()

    import torch
    from datapack import card_units, canonical
    from server import live_analysis_service as las

    d = las.model_dir()
    if d is None:
        raise SystemExit('找不到模型权重目录（AI2050_LIVE_MODEL_DIR / 缺省路径）')
    meta = las._read_meta(d)
    n_layers, d_ff = meta['n_layers'], meta['d_ff']
    n_heads, n_kv = meta['n_heads'], meta['n_kv_heads']
    hd, group = meta['head_dim'], n_heads // n_kv
    print(f'模型 {d.name}：{n_layers} 块 · d_model {meta["d_model"]} · d_ff {d_ff} · {n_heads} 头', flush=True)

    t0 = time.time()
    from transformers import AutoModelForCausalLM
    # 不用 device_map='auto'：accelerate 的 CPU offload 会把权重留在 meta 设备，直接访问 .weight 是空张量。
    # 改为手动分片：整模型载入 CPU，再把装得下的层逐个搬上 GPU（按实际空闲显存，留 1GiB 余量给 fp32 副本）。
    model = AutoModelForCausalLM.from_pretrained(str(d), dtype=torch.bfloat16, attn_implementation='eager')
    model.eval()
    if torch.cuda.is_available() and args.gpu_gib > 0:
        reserve = 1 * 2 ** 30
        moved = 0
        for lyr in model.model.layers:
            bytes_need = sum(p.numel() * p.element_size() for p in lyr.parameters())
            try:
                free, _total = torch.cuda.mem_get_info()
            except Exception:
                break
            if free - reserve < bytes_need:
                break
            lyr.to('cuda')
            moved += 1
        print(f'权重加载完成：{moved}/{len(model.model.layers)} 层在 GPU，其余 CPU（{time.time() - t0:.0f}s）', flush=True)
    else:
        print(f'权重加载完成：纯 CPU（{time.time() - t0:.0f}s）', flush=True)
    layers = model.model.layers

    from server import neuron_census_service as NS
    from server.neuron_circuit_service import _circuit, _labels

    keys = [args.model_key] if args.model_key else \
        sorted({r['model_key'] for r in NS.index().get('census', [])})
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for mk in keys:
        run = NS.pick_census(mk, None)['run']
        units = set(card_units(mk))
        # 每层按激活密度补足（用户可点层阵列的任何单元；密度来自逐层摘要）
        for block in range(n_layers):
            doc = NS.neuron_layer(model_key=mk, block=block, run=run)
            cols = (doc.get('columns') or {}) if not doc.get('pending') else {}
            dens = cols.get('dens') or []
            order = sorted(range(len(dens)), key=lambda j: -dens[j])[:args.per_layer]
            units.update((block, j) for j in order)
        by_block: dict = {}
        for b, u in sorted(units):
            if 0 <= b < n_layers and 0 <= u < d_ff:
                by_block.setdefault(b, []).append(u)
        total = sum(len(v) for v in by_block.values())
        print(f'[{mk}] run={run} 覆盖单元 {total}（{len(by_block)} 块）', flush=True)

        entries = {}
        t1 = time.time()
        done = 0
        for L, us in by_block.items():
            up_hits, down_hits = [], []                     # 供 _labels 打标签
            lyr = layers[L].mlp
            Wd = layers[L - 1].mlp.down_proj.weight.detach().float().t() if L >= 1 else None   # (F,H)
            norms = Wd.norm(dim=1).clamp_min(1e-8) if Wd is not None else None
            o_w = layers[L].self_attn.o_proj.weight.detach().float()      # (H, heads*hd)
            v_w = layers[L].self_attn.v_proj.weight.detach().float().T    # (H, kv*hd)
            if L + 1 < n_layers:
                mlp1 = layers[L + 1].mlp
                R = torch.cat([mlp1.gate_proj.weight.detach().float(),
                               mlp1.up_proj.weight.detach().float()], dim=0)                # (2F,H)
                rnorms = R.norm(dim=1).clamp_min(1e-8)
                W_o1 = layers[L + 1].self_attn.o_proj.weight.detach().float()
                W_v1 = layers[L + 1].self_attn.v_proj.weight.detach().float().T
                RWO = R @ W_o1                                                            # (2F, heads*hd)
                half = R.shape[0] // 2
            for i in us:
                r_g = lyr.gate_proj.weight[i].detach().float()
                r_u = lyr.up_proj.weight[i].detach().float()
                w_out = layers[L].mlp.down_proj.weight[:, i].detach().float()
                upstream = downstream = []
                if Wd is not None:
                    dev_o = o_w.device                     # 手动分片后 L-1/L 可能不同设备：逐运算对齐
                    cands = []
                    for r in (r_g.to(Wd.device), r_u.to(Wd.device)):
                        cands += [{'via': 'direct', **s} for s in topk_rows((Wd @ r) / norms, args.k, i)]
                    rg_o = r_g.to(dev_o)
                    for h in range(n_heads):
                        a = o_w[:, h * hd:(h + 1) * hd].T @ rg_o
                        v = v_w[:, (h // group) * hd:((h // group) + 1) * hd] @ a
                        cands += [{'via': 'head', 'head': h, **s}
                                  for s in topk_rows((Wd @ v.to(Wd.device)) / norms, args.k, i)]
                    upstream = merge_cands(cands, args.k)
                if L + 1 < n_layers:
                    dev_d = R.device
                    w_dn = w_out.to(dev_d)
                    cands = []
                    dir_scores = (R @ w_dn) / rnorms
                    cands += [{'via': 'direct', **s}
                              for s in topk_rows(torch.maximum(dir_scores[:half], dir_scores[half:]), args.k, i)]
                    for h in range(n_heads):
                        b = W_v1[:, (h // group) * hd:((h // group) + 1) * hd].T @ w_dn
                        s_h = (RWO[:, h * hd:(h + 1) * hd] @ b) / rnorms
                        cands += [{'via': 'head', 'head': h, **s}
                                  for s in topk_rows(torch.maximum(s_h[:half], s_h[half:]), args.k, i)]
                    downstream = merge_cands(cands, args.k)
                for h_ in upstream:
                    h_['block'] = L - 1
                for h_ in downstream:
                    h_['block'] = L + 1
                up_hits += upstream
                down_hits += downstream
                entries[canonical(f'/api/neurons/circuit?block={L}&unit={i}&k={args.k}'
                                  f'&model_key={mk}&run={run}')] = {
                    'schema': 'neuron_circuit.v1', 'available': True, 'model_name': d.name,
                    'device': 'precomputed', 'block': L, 'unit': i, 'k': args.k,
                    'n_layers': n_layers, 'd_ff': d_ff,
                    'upstream': upstream, 'downstream': downstream,
                    'write_norm': round(float(w_out.norm()), 3),
                    'read_norm': round(float(r_g.norm()), 3)}
                done += 1
                if done % 500 == 0:
                    print(f'  {done}/{total}（{time.time() - t1:.0f}s）', flush=True)
            del Wd, norms
            if L + 1 < n_layers:
                del R, RWO
            _labels(mk, run, up_hits, L - 1)               # 分型/解释标签（只作展示；原对象已进 entries）
            _labels(mk, run, down_hits, L + 1)
        print(f'[{mk}] 计算完成 {time.time() - t1:.0f}s', flush=True)

        # 标签是在 up_hits/down_hits 原对象上打的（同一 dict 引用已进 entries），此处无需回填。
        doc = {'schema': 'neuron_circuit_pack.v1', 'model_key': mk, 'run': run, 'k': args.k,
               'n_entries': len(entries), 'created_at': time.time(), 'entries': entries}

        if args.verify:
            import random
            random.seed(7)
            checked = bad = 0
            for L, us in by_block.items():
                for i in random.sample(us, min(args.verify, len(us))):
                    live = _circuit(model, L, i, args.k)
                    key = canonical(f'/api/neurons/circuit?block={L}&unit={i}&k={args.k}'
                                    f'&model_key={mk}&run={run}')
                    pre = entries[key]
                    for side in ('upstream', 'downstream'):
                        a = [(x['unit'], round(x['score'], 3), x['via'], x.get('head')) for x in live[side]]
                        b_ = [(x['unit'], round(x['score'], 3), x['via'], x.get('head')) for x in pre[side]]
                        if a != b_:
                            bad += 1
                            print(f'  MISMATCH {key} {side}\n    live={a[:3]}\n    pre ={b_[:3]}', flush=True)
                    checked += 1
            print(f'[{mk}] 抽样比对 {checked} 个单元，不一致 {bad} 处', flush=True)
            if bad:
                raise SystemExit('预计算与实时实现不一致，拒绝写出')

        name = re.sub(r'[^A-Za-z0-9._-]', '_', mk) + '.circuit.v1.json.gz'
        with gzip.open(out_dir / name, 'wt', encoding='utf-8') as fh:
            json.dump(doc, fh, ensure_ascii=False)
        size = (out_dir / name).stat().st_size
        print(f'[{mk}] → {out_dir / name}（{size / 2**20:.1f} MB，{len(entries)} 条）', flush=True)
    print('全部完成。上传服务器：<AI2050_DIST_DIR>/circuit_sources/ 后等 snapshot_worker 重建附加包。', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
