# -*- coding: utf-8 -*-
"""方法校准 C2（H3b 第 1 步，Phase 345）：充分性检验能否在 GPT-2 IOI 上找回已知电路。

背景：MA-H3（Phase 342–343）的“保留”曲线在 ≤10% 预算时恢复不了范围效应；MB-v1 的 GPT-2 IOI 上同样的
cf / mean / zero 保留曲线也找不回已知的小电路（exact 前 5% 只恢复 0.18）。所以先问：本项目的修补机制，
在“已知答案”的设定下，能不能复现论文的充分性结果？

已知答案（Wang et al. 2022）：IOI 电路 26 个头；其余头用 ABC 分布（三个名字都换掉）的均值消融、
MLP 全部保留时，电路保留了约 87% 的 logit 差（论文的 faithfulness）。
本脚本：
  1. 材料：calib_ioi_gpt2 的 BABA 模板，新种子；ABC 句 = 同模板、三个名字全换，按模板逐位置求均值（长度相同）。
  2. 保留集合 C 之外的头：c_proj 输入里该头的切片换成 ABC 的模板逐位置均值；MLP 全部保留。
  3. 指标：LD = 末位 logit(IO) − logit(S)；faithfulness = 平均 LD(C) / 平均 LD(全模型)（论文口径），
     另报 (LD(C) − LD(∅)) / (LD(M) − LD(∅))，∅ = 全部头均值消融。
  4. 集合：论文电路（26 头）；3 组随机 26 头；按 MB-v1 的 GPT-2 IOI 排序（exact、atp、atp_ig、dla、act_diff）取前 26 个头。
判据 CRITERIA 运行前固定，结果随判据写出。
用法：python scripts/calib_ioi_circuit.py --model D:/develop/model/gpt2-local
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]
import calib_ioi_gpt2 as C1  # noqa: E402

CIRCUIT = {   # Wang et al. 2022 图 2 的 26 个头
    'name_mover': ['9.9', '9.6', '10.0'],
    'backup_name_mover': ['10.10', '10.6', '10.2', '10.1', '11.2', '9.7', '9.0', '11.9'],
    'negative_name_mover': ['10.7', '11.10'],
    's_inhibition': ['7.3', '7.9', '8.6', '8.10'],
    'induction': ['5.5', '6.9', '5.8', '5.9'],
    'duplicate_token': ['0.1', '0.10', '3.0'],
    'previous_token': ['2.2', '4.11'],
}
CRITERIA = {
    'schema': 'calib_ioi_circuit.v1',
    'n_prompts': 240, 'seed': 1,
    'circuit_faith_min': 0.6,          # 论文约 0.87；低于 0.6 说明本项目的消融机制没有复现论文设定
    'random_margin_min': 0.3,          # 电路比 3 组随机 26 头中最好的高出至少 0.3
    'method_rule': '记录各方法前 26 个头的 faithfulness 与和论文电路的重合数；不设通过线（描述）',
    'rankings': 'research/mech_bench/MB-v1/gpt2/rankings.npz 的 rs_ioi|node|{method}（只取头，按分数降序）',
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='D:/develop/model/gpt2-local')
    ap.add_argument('--out', default='research/calibration/C2-ioi-circuit')
    a = ap.parse_args()
    from transformers import AutoTokenizer, GPT2LMHeadModel
    from provenance import write_result
    torch.manual_seed(0)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(a.model)
    model = GPT2LMHeadModel.from_pretrained(a.model, torch_dtype=torch.float32, attn_implementation='eager').to(dev).eval()
    L, H = model.config.n_layer, model.config.n_head
    dh = model.config.n_embd // H
    rows = C1.build(tok, CRITERIA['n_prompts'], CRITERIA['seed'])
    groups = {}
    for r in rows:
        groups.setdefault(r['template'], []).append(r)

    store, repl = {}, {}
    def hook(l):
        def pre(mod, inp):
            x = inp[0]
            store[l] = x.detach()
            if l in repl:
                mask, mean = repl[l]          # mask: [H] bool（True = 换成均值）
                if mask.any():
                    x = x.clone()
                    for h in torch.nonzero(mask).flatten().tolist():
                        x[..., h * dh:(h + 1) * dh] = mean[..., h * dh:(h + 1) * dh]
                    return (x,)
            return None
        return pre
    hooks = [model.transformer.h[l].attn.c_proj.register_forward_pre_hook(hook(l)) for l in range(L)]

    t0 = time.time()
    means = {}                                # 模板 → 层 → [t, n_embd] 的 ABC 均值
    with torch.no_grad():
        for t, gr in groups.items():
            ids = tok([r['corrupt'] for r in gr], return_tensors='pt').input_ids.to(dev)
            repl.clear(); model(ids)
            means[t] = {l: store[l].mean(0) for l in range(L)}

    def ld_with(keep: set) -> np.ndarray:
        out = []
        with torch.no_grad():
            for t, gr in groups.items():
                ids = tok([r['clean'] for r in gr], return_tensors='pt').input_ids.to(dev)
                repl.clear()
                for l in range(L):
                    mask = torch.tensor([f'{l}.{h}' not in keep for h in range(H)])
                    repl[l] = (mask, means[t][l])
                logits = model(ids).logits[:, -1]
                io = torch.tensor([tok.encode(r['io'])[0] for r in gr], device=dev)
                s = torch.tensor([tok.encode(r['s'])[0] for r in gr], device=dev)
                out.append((logits.gather(1, io[:, None]) - logits.gather(1, s[:, None])).squeeze(1).cpu().numpy())
        repl.clear()
        return np.concatenate(out)

    ALL = {f'{l}.{h}' for l in range(L) for h in range(H)}
    circuit = {h for v in CIRCUIT.values() for h in v}
    full, empty = ld_with(ALL), ld_with(set())
    def score(keep):
        ld = ld_with(keep)
        return {'n_heads': len(keep), 'ld': float(ld.mean()), 'faith': float(ld.mean() / full.mean()),
                'faith_norm': float((ld.mean() - empty.mean()) / (full.mean() - empty.mean())),
                'overlap_with_circuit': len(keep & circuit)}
    res = {'full_ld': float(full.mean()), 'empty_ld': float(empty.mean()), 'sets': {}}
    res['sets']['circuit'] = score(circuit)
    rng = random.Random(0)
    for i in range(3):
        res['sets'][f'random{i}'] = score(set(rng.sample(sorted(ALL), len(circuit))))
    z = np.load(ROOT / 'research/mech_bench/MB-v1/gpt2/rankings.npz')
    for m in ['exact', 'atp', 'atp_ig', 'dla', 'act_diff']:
        sc = z[f'rs_ioi|node|{m}'][:L * H]
        top = {f'{i // H}.{i % H}' for i in np.argsort(-sc)[:len(circuit)]}
        res['sets'][f'top26_{m}'] = {**score(top), 'heads': sorted(top, key=lambda x: tuple(map(int, x.split('.'))))}
    for hk in hooks:
        hk.remove()
    best_rand = max(res['sets'][f'random{i}']['faith'] for i in range(3))
    c = res['sets']['circuit']['faith']
    res['verdict'] = {'circuit_faith': c, 'best_random': best_rand,
                      'reproduces_paper': bool(c >= CRITERIA['circuit_faith_min'] and c - best_rand >= CRITERIA['random_margin_min'])}
    res.update({'criteria': CRITERIA, 'circuit': CIRCUIT, 'n_prompts': len(rows), 'seconds': round(time.time() - t0, 1),
                'torch': torch.__version__, 'device': dev})
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding='utf-8')
    print('C2', write_result('calib-ioi-circuit', res, model_path=a.model, seed=CRITERIA['seed'], tool='scripts/calib_ioi_circuit.py'))
    print(json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'heads'} for k, v in res['sets'].items()}, ensure_ascii=False, indent=1))
    print('verdict', res['verdict'])
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
