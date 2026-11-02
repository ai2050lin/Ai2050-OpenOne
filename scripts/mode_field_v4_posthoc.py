# -*- coding: utf-8 -*-
"""MF-v4 事后诊断（不进判定）：幅度匹配的随机单元对照。

问题：Qwen 第 33 / 35 层索引最小的约 128 个单元激活幅度明显更大（普查标准差 2.06 vs 0.80），选出的门控单元多在其中；
预注册的随机对照只按层匹配，多数抽到小幅度单元，H1 / H3a 的比较部分是「大单元 vs 小单元」。
这里按自然文本（MF-v1 确认集全部位置）上每个单元中间激活的标准差，为每个门控单元在同层抽取标准差最接近的单元
（排除门控单元，在最近的 10 个中随机取 1 个），组成 5 个幅度匹配集合：
  1  在 B 任务上重做单元修补，比较门控单元与幅度匹配集合的恢复；
  2  只写代码时的激活变化按标准差归一后比较。
"""
from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts')]

from shared.dist_protocol import mode_field as F  # noqa: E402
import mode_field_v1 as MF1  # noqa: E402
import mode_field_v3 as MF3  # noqa: E402
import mode_field_v4 as MF4  # noqa: E402
import xm_replicate_v1 as XM  # noqa: E402

STY = F.STYLES
CODE = STY.index('code')


def main(model_path='models/hf/qwen3-4b', int8=False):
    import torch
    crit = json.loads(MF4.PREREG.read_text(encoding='utf-8'))['criteria']
    rep = json.loads(MF4.report_path(model_path, int8).read_text(encoding='utf-8'))
    TL = rep['layers']
    off = {int(k): v for k, v in rep['unit_offsets'].items()}
    _, key = XM.key_of(model_path, int8)
    od = MF1.out_dir(key)
    U = np.load(od / 'mf_v4_units.npz')
    top, layer_of, rd = [int(x) for x in U['top']], U['layer_of'], U['rd_mean']
    z = MF1.load_fields(key)
    L = int(z['Hs'].shape[1]) - 1
    ls = rep['switch_layer']
    y, tok, sp, sh = z['style'], z['tok'], z['split'], z['shared'].astype(bool)
    D = sp == 'disc'
    Xs = z['Hs'][D[sh], ls].astype(np.float32)
    ys, ts = y[sh][D[sh]], tok[sh][D[sh]]
    V, SG = {}, {}
    for s in range(4):
        v = F.token_matched_direction(Xs, ys, ts, s)
        V[s] = v / np.linalg.norm(v)
        SG[s] = float((Xs @ V[s]).std())
    del Xs, z
    gc.collect()
    excl = json.loads((od / 'vocab.json').read_text(encoding='utf-8'))['excl']
    tok_, model = XM.load(model_path, int8)
    R = XM.Runner(tok_, model)
    C, ids = MF1.tokenize(tok_)
    EX = [torch.as_tensor(excl[s], device='cuda') for s in STY]
    cap, store = {}, {}
    st = {'inject': None, 'pos': None, 'capture': False, 'patch': None}

    def inj(mod, args, kwargs):
        if st['inject'] is None:
            return None
        h = (args[0] if args else kwargs['hidden_states']).clone()
        h[:, :st['pos']] += st['inject'][:, None, :].to(h.dtype)
        if args:
            return (h,) + tuple(args[1:]), kwargs
        kwargs['hidden_states'] = h
        return args, kwargs

    def dp_hook(l):
        def f(m, a):
            x = a[0]
            if st['capture']:
                cap[l] = x.detach().clone()
            if st['patch'] is not None and l in st['patch']:
                x = x.clone()
                for row, units in st['patch'][l].items():
                    x[row][:, units] = store[l][:, units].to(x.dtype)
                return (x,) + tuple(a[1:])
            return None
        return f
    hooks = [R.blocks[ls].register_forward_pre_hook(inj, with_kwargs=True)]
    hooks += [R.blocks[l].mlp.down_proj.register_forward_pre_hook(dp_hook(l)) for l in TL]
    # 1 自然文本上每个单元的标准差
    s1 = {l: 0 for l in TL}
    s2 = {l: 0 for l in TL}
    cnt = 0
    conf = [i for i, cc in enumerate(C) if cc['split'] == 'conf']
    for j in range(0, len(conf), 16):
        chunk = conf[j:j + 16]
        st['capture'] = True
        try:
            R.forward([ids[i] for i in chunk])
        finally:
            st['capture'] = False
        for l in TL:
            a = torch.cat([cap[l][b, :len(ids[i])].float() for b, i in enumerate(chunk)])
            s1[l] = s1[l] + a.sum(0)
            s2[l] = s2[l] + (a ** 2).sum(0)
        cnt += sum(len(ids[i]) for i in chunk)
    sd = np.concatenate([torch.sqrt(torch.clamp(s2[l] / cnt - (s1[l] / cnt) ** 2, min=0)).cpu().numpy() for l in TL])
    # 2 幅度匹配集合
    rng = np.random.default_rng(crit['seed'])
    topset = set(top)
    msets = []
    for _ in range(crit['n_random_sets']):
        s = []
        for i in top:
            pool = np.array([j for j in np.where(layer_of == layer_of[i])[0] if j not in topset and j not in s])
            near = pool[np.argsort(np.abs(sd[pool] - sd[i]))[:10]]
            s.append(int(rng.choice(near)))
        msets.append(s)
    norm_read = lambda units: float(np.mean(np.abs(rd[units]) / np.maximum(sd[units], 1e-6)))
    out = {'sd_top_mean': float(sd[top].mean()), 'sd_matched_mean': float(np.mean([sd[s].mean() for s in msets])),
           'sd_layer_matched_random_mean': float(np.mean([sd[s].mean() for s in U['rsets']])),
           'norm_read_top': norm_read(top), 'norm_read_matched': [norm_read(s) for s in msets]}

    def by_layer(units):
        o = {}
        for i in units:
            o.setdefault(int(layer_of[i]), []).append(int(i - off[int(layer_of[i])]))
        return {l: torch.as_tensor(v, device='cuda') for l, v in o.items()}
    variants = [None, by_layer(top)] + [by_layer(s) for s in msets]
    rest = {}
    vec = lambda s_: 6 * SG[s_] * V[s_]
    for g in [tuple(x) for x in crit['groups']]:
        p = STY.index(g[1])
        texts = [i for i, cc in enumerate(C) if cc['split'] == 'disc' and cc['style'] == g[0] and not MF3.in_A(cc['task'])]
        M = np.zeros((len(texts), 2 + len(variants)))
        for ti, i in enumerate(texts):
            n = max(2, round(0.6 * len(ids[i])))
            seq = ids[i][:n]
            st['pos'] = n
            st['inject'] = torch.as_tensor(np.stack([vec(p), vec(p) + vec(CODE)]), dtype=torch.float32, device='cuda')
            st['capture'] = True
            try:
                o = R.forward([seq] * 2)
            finally:
                st['capture'] = False
            lp = torch.log_softmax(o.logits[:, n - 1].float(), -1)
            M[ti, :2] = torch.logsumexp(lp[:, EX[p]], -1).cpu().numpy()
            for l in TL:
                store[l] = cap[l][0]
            st['inject'] = torch.as_tensor(np.stack([vec(p) + vec(CODE)] * len(variants)), dtype=torch.float32, device='cuda')
            pt = {}
            for row, spec in enumerate(variants):
                if spec is None:
                    continue
                for l, units in spec.items():
                    pt.setdefault(l, {})[row] = units
            st['patch'] = pt
            try:
                o = R.forward([seq] * len(variants))
            finally:
                st['patch'] = None
            lp = torch.log_softmax(o.logits[:, n - 1].float(), -1)
            M[ti, 2:] = torch.logsumexp(lp[:, EX[p]], -1).cpu().numpy()
        single, joint = M[:, 0], M[:, 2]
        gap = float((single - joint).mean())
        rest['%s:%s' % g] = {'top': float((M[:, 3] - joint).mean() / gap),
                             'matched': [float((M[:, 4 + j] - joint).mean() / gap) for j in range(len(msets))], 'gap': gap}
        print('group', g, {k: (round(v, 3) if isinstance(v, float) else [round(x, 3) for x in v]) for k, v in rest['%s:%s' % g].items()}, flush=True)
    for h in hooks:
        h.remove()
    out['restore_B'] = rest
    out['restore_top_mean'] = float(np.mean([r['top'] for r in rest.values()]))
    out['restore_matched_mean'] = [float(np.mean([r['matched'][j] for r in rest.values()])) for j in range(len(msets))]
    out['note'] = '事后诊断，不进 MF-v4 判定'
    print({k: v for k, v in out.items() if k != 'restore_B'}, flush=True)
    MF4.save(model_path, int8, {'posthoc_magnitude_matched': out})


if __name__ == '__main__':
    main(*(sys.argv[1:2] or []), int8='--int8' in sys.argv)
