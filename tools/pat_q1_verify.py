# -*- coding: utf-8 -*-
"""Q1 独立复核 + 因果门。

三件事
------
V1 单头复核：抽 4 个「通过头」与 2 个随机头，逐格打印对三个实体的注意力质量与 argmax，
   人工可核对聚合没有算错（聚合口径出错是本项目最常见的假阳性来源）。
V2 因果门：把 top-30 头与 30 个随机头的注意力输出置零，看强制选择正确率掉多少。
   「能关闭」是 C2 判据；若 top-30 与随机头无差别，则本效应是分布式的、无可点名的承载者。
V3 冗余度：报告 top-30 消融后的下降幅度，用来区分「少数承载」与「全网分摊」。

用法
----
  .venv/Scripts/python.exe -X utf8 tools/pat_q1_verify.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (str(ROOT), str(ROOT / 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)

import provenance
from pat_q1_unbind import (FORMS, QUERIES, NEUTRAL, ROLES, build, encode, render,
                           triples, PERSONS, OBJECTS)

SEED = 20261101
N_TRIP = 24


def load_heads(k=None):
    """读最新一版 Q1 结果，返回 (前 k 个通过头, 全部通过头)。"""
    cand = sorted((ROOT / 'results').glob('2026-11-01/*_pat_q1_unbind_v2.json'))
    d = json.loads(cand[-1].read_text(encoding='utf-8'))
    allh = [tuple(x) for x in d['payload']['heads_all']]
    return (allh[:k] if k else allh), allh


def top_heads(path=None, k=30):
    return load_heads(k)[0]


class HeadMask:
    """把指定 (层,头) 的 o_proj 输入切片置零 —— 与项目既有 hook 口径一致。"""

    def __init__(self, model, meta, pairs):
        self.model = model
        self.dh = meta['head_dim']
        self.pairs = set(pairs)
        self.handles = []

    def __enter__(self):
        for li, layer in enumerate(self.model.model.layers):
            heads = [h for (l, h) in self.pairs if l == li]
            if not heads:
                continue
            dh = self.dh

            def hook(_m, args, _heads=tuple(heads), _dh=dh):
                x = args[0]
                x = x.clone()
                for h in _heads:
                    x[..., h * _dh:(h + 1) * _dh] = 0.0
                return (x,)

            self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(hook))
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()
        return False


@torch.no_grad()
def run_cell(model, tok, dev, trip, form, role, mask=None):
    ctx, ranges = build(trip, form)
    text = render(tok, ctx + QUERIES[role])
    ids, idx = encode(tok, text, ctx + QUERIES[role], ranges)
    t = torch.tensor([ids], device=dev)
    if mask is not None:
        with mask:
            out = model(input_ids=t, use_cache=False)
    else:
        out = model(input_ids=t, use_cache=False, output_attentions=True)
    lp = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    sc = {e: float(lp[tok(e, add_special_tokens=False)['input_ids'][0]]) for e in trip}
    return max(sc, key=sc.get), sc, (out.attentions if mask is None else None), idx, len(ids)


def main() -> int:
    from server import live_analysis_service as las

    t0 = time.time()
    las._load_async()
    while las._state['status'] in ('idle', 'loading'):
        if time.time() - t0 > 600:
            raise RuntimeError('加载超时')
        time.sleep(2.0)
    model, tok = las._state['model'], las._state['tokenizer']
    dev = next(model.parameters()).device
    meta = las._state['meta']

    trips = triples(N_TRIP)
    hs = top_heads()

    print('=== V1 单头复核（三元组 1，实体的注意力质量）===')
    trip = trips[0]
    probe = hs[:4] + [(10, 3), (5, 7)]
    for (l, h) in probe:
        tag = '通过' if (l, h) in hs else '随机'
        print('L%-2d H%-2d [%s]' % (l, h, tag))
        for form in FORMS:
            for role in ROLES:
                ctx, ranges = build(trip, form)
                text = render(tok, ctx + QUERIES[role])
                ids, idx = encode(tok, text, ctx + QUERIES[role], ranges)
                with torch.no_grad():
                    out = model(input_ids=torch.tensor([ids], device=dev),
                                use_cache=False, output_attentions=True)
                v = out.attentions[l][0, h, -1, :].float().cpu().numpy()
                m = {r: float(v[idx[r]].sum()) for r in ROLES}
                am = max(m, key=m.get)
                print('   %s %-9s 质量 %s  argmax=%-9s %s' % (
                    form, role, {k: round(x, 3) for k, x in m.items()}, am,
                    '' if am == role else '<< 不符'))

    print('=== V2 因果门（n=%d 三元组 x 6 格）===' % N_TRIP)
    rng = np.random.default_rng(SEED)
    allh = [(l, h) for l in range(meta['n_layers']) for h in range(meta['n_heads'])]
    ok_all = load_heads()[1]
    rand_h = [allh[i] for i in rng.choice(len(allh), size=len(ok_all), replace=False)]

    def acc(pairs):
        mask = HeadMask(model, meta, pairs) if pairs else None
        ok = n = 0
        for trip in trips:
            for form in FORMS:
                for role in ROLES:
                    want = {'agent': trip[0], 'recipient': trip[1], 'theme': trip[2]}[role]
                    pick, _, _, _, _ = run_cell(model, tok, dev, trip, form, role, mask)
                    ok += int(pick == want)
                    n += 1
        return ok / n, n

    a_base, n = acc([])
    a_top, _ = acc(hs)
    a_all, _ = acc(ok_all)
    a_rnd, _ = acc(rand_h)
    print('base=%.4f  top30=%.4f  all263=%.4f  rand263=%.4f  (n=%d)'
          % (a_base, a_top, a_all, a_rnd, n))
    drop, drop_all, drop_r = a_base - a_top, a_base - a_all, a_base - a_rnd
    v2 = ('all263 显著掉且远超随机' if drop_all >= 0.15 and drop_all >= drop_r + 0.10
          else 'all263 未显著掉（分布式冗余，无可点名承载集）')
    print('drop_top30=%.4f drop_all=%d头:%.4f drop_rand=%d头:%.4f => %s'
          % (drop, len(ok_all), drop_all, len(rand_h), drop_r, v2))

    out = {'n_trip': N_TRIP, 'acc_base': round(a_base, 4), 'acc_top30': round(a_top, 4),
           'acc_all_ok': round(a_all, 4), 'n_all_ok': len(ok_all),
           'acc_rand': round(a_rnd, 4), 'n_rand': len(rand_h),
           'drop_top30': round(drop, 4), 'drop_all_ok': round(drop_all, 4),
           'drop_rand': round(drop_r, 4), 'V2_verdict': v2,
           'top30': [list(x) for x in hs]}

    print('=== V3 剂量曲线（ok 集 vs 同规模随机集，k=消融头数）===')
    dose = []
    for k in (10, 30, 60, 100, 150, 200):
        ok_sel = ok_all[:k]
        rd_sel = [allh[i] for i in rng.choice(len(allh), size=k, replace=False)]
        ak, _ = acc(ok_sel)
        rk, _ = acc(rd_sel)
        dose.append({'k': k, 'acc_ok': round(ak, 4), 'acc_rand': round(rk, 4),
                     'gap': round(rk - ak, 4)})
        print('  k=%-3d  ok=%.4f  rand=%.4f  gap=%+.4f' % (k, ak, rk, rk - ak))
    mono = all(dose[i]['acc_ok'] <= dose[i + 1]['acc_ok'] + 1e-9 for i in range(len(dose) - 1))
    gap_max = max(d['gap'] for d in dose)
    print('  ok 集随 k 单调不增=%s  最大 gap=%+.4f（gap 为正=ok 集更脆）' % (mono, gap_max))
    out['dose'] = dose
    out['dose_monotone'] = bool(mono)
    out['gap_max'] = round(gap_max, 4)
    out['V3_verdict'] = ('ok 集显著更脆（有承载梯度）' if gap_max >= 0.10
                         else 'ok 集与随机同规模无差别（注意力落点非因果承载）')
    p = provenance.write_result('pat_q1_verify', out, seed=SEED, tool='tools/pat_q1_verify.py')
    print('落盘 %s' % p)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
