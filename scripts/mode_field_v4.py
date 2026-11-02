# -*- coding: utf-8 -*-
"""MF-v4：后段 MLP 中「代码压过散文」的门控单元。

MF-v3（Phase 277）：同时写入代码与中文 / 英文说明时，散文效应被压掉，修补后段 MLP 整层可恢复（Qwen mlp27 / 33 / 35，GLM mlp25 / 38）。
本版在这些层的中间单元（down_proj 的输入 A）上找门控单元，材料与写入同 MF-v3（发现集文本，按任务分 A / B）。
  选择（只用 A）：末位直接作用差 Δ_i = (A_同时 − A_只散文)_i · (W_down[:, i] · u)，u = ∂log M_p / ∂(最终归一化输入)（冻结 RMS），
                三组合并取平均最负的 32 个单元。
  H1  B 上（三组恢复比例的平均）：只把这 32 个单元的中间激活（全部位置）换回「只写散文」的值，恢复 ≥ 整层修补（这些层全部单元）的 50%，
      且 ≥ 每个同层随机 32 单元集合的 3 倍。
  H2  B 上三组各自恢复 ≥ 0.2（同一批单元同时压住中文与英文说明）。
  H3  这些单元读代码方向：(a) 只写代码时末位激活变化的平均绝对值 ≥ 同层随机单元的 3 倍；
      (b) 自然文本（MF-v1 确认集共用 token 位置）上，得分 Σ sign_i A_i（sign 取只写代码时的变化方向）区分代码与散文位置的 AUC ≥ 0.7。
  描述：Qwen 的单元在神经元普查中的直接读出 token 与激活统计。
用法：python scripts/mode_field_v4.py prereg | run --model M [--int8] | analyze --model M [--int8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]

from shared.dist_protocol import mode_field as F  # noqa: E402
import mode_field_v1 as MF1  # noqa: E402
import mode_field_v3 as MF3  # noqa: E402
import xm_replicate_v1 as XM  # noqa: E402

PREREG = ROOT / 'research/mode_field/MF-v4.prereg.json'
STY = F.STYLES
CODE = STY.index('code')
CENSUS = Path('D:/ai2050_native/_census/6be1034f0b55a6df-bf16/2ca5f0a756d36c8a-20000000')


def prereg_doc():
    return {'id': 'MF-v4', 'schema': 'mode_field_prereg.v4', 'created': '2026-11-02',
            'basis': 'MF-v3（Phase 277）：代码压过散文由后段 MLP 计算；本版在 MF-v3 选出的层中找单元',
            'models': {'qwen3-4b': 'bf16（主）', 'glm4-9b-chat-hf': 'int8（复现）'},
            'materials': 'MF-v1 发现集文本（MF-v3 的 A / B 任务切分）；H3b 用 MF-v1 确认集文本',
            'unit_definition': 'MLP 中间单元 = down_proj 的输入坐标（SwiGLU：A = act(gate) ⊙ up）；W_down 列由 down_proj(I) − down_proj(0) 取得（int8 也适用）',
            'census': 'Qwen3-4B 普查运行 2ca5f0a756d36c8a-20000000（只作描述）',
            'power_note': '连续量阈值；每组 80 段（A / B 各 40）',
            'doc': __doc__, 'criteria': F.CRITERIA_V4}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（MF-v5）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def report_path(model_path, int8):
    d = ROOT / 'research/mode_field' / f'MF-v4-{Path(model_path).name}{"-int8" if int8 else ""}'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def save(model_path, int8, upd):
    p = report_path(model_path, int8)
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update(upd)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    return rep


def do_run(model_path, int8):
    import gc
    import torch
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    TL = crit['layers'][Path(model_path).name]
    snap, key = XM.key_of(model_path, int8)
    od = MF1.out_dir(key)
    z = MF1.load_fields(key)
    L = int(z['Hs'].shape[1]) - 1
    ls = round(0.5 * L)
    y, tok, sp, sh = z['style'], z['tok'], z['split'], z['shared'].astype(bool)
    D = sp == 'disc'
    Xs = z['Hs'][D[sh], ls].astype(np.float32)
    ys, ts = y[sh][D[sh]], tok[sh][D[sh]]
    V, SG = {}, {}
    for s in range(4):
        v = F.token_matched_direction(Xs, ys, ts, s)
        V[s] = v / np.linalg.norm(v)
        SG[s] = float((Xs @ V[s]).std())
    meta = {k_: z[k_] for k_ in ('text', 'pos', 'style', 'split', 'shared')}
    del Xs, z
    gc.collect()
    excl = json.loads((od / 'vocab.json').read_text(encoding='utf-8'))['excl']
    tok_, model = XM.load(model_path, int8)
    R = XM.Runner(tok_, model)
    C, ids = MF1.tokenize(tok_)
    EX = [torch.as_tensor(excl[s], device='cuda') for s in STY]
    Wu = R.model.lm_head.weight
    gw = R.model.model.norm.weight.float()
    eps = getattr(R.model.model.norm, 'variance_epsilon', getattr(R.model.model.norm, 'eps', 1e-6))
    t0 = time.time()
    # W_down 的转置：[I, d]
    WdT = {}
    for l in TL:
        dp = R.blocks[l].mlp.down_proj
        I_ = dp.in_features
        rows = []
        z0 = dp(torch.zeros(1, I_, device='cuda', dtype=torch.bfloat16)).float()
        for j in range(0, I_, 2048):
            e = torch.zeros(min(2048, I_ - j), I_, device='cuda', dtype=torch.bfloat16)
            e[torch.arange(len(e)), torch.arange(j, j + len(e))] = 1
            rows.append((dp(e).float() - z0).to(torch.bfloat16))
        WdT[l] = torch.cat(rows)
    sizes = [WdT[l].shape[0] for l in TL]
    layer_of = np.concatenate([[l] * n for l, n in zip(TL, sizes)])
    off = dict(zip(TL, np.concatenate([[0], np.cumsum(sizes)[:-1]]).tolist()))
    st = {'inject': None, 'pos': None, 'capture': False, 'patch': None}
    cap, store = {}, {}

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
            if st['patch'] is not None:
                x = x.clone()
                for row, units in st['patch'].get(l, {}).items():
                    if units is None:
                        x[row] = store[l].to(x.dtype)
                    else:
                        x[row][:, units] = store[l][:, units].to(x.dtype)
                return (x,) + tuple(a[1:])
            return None
        return f
    hooks = [R.blocks[ls].register_forward_pre_hook(inj, with_kwargs=True)]
    hooks += [R.blocks[l].mlp.down_proj.register_forward_pre_hook(dp_hook(l)) for l in TL]
    hooks.append(R.model.model.norm.register_forward_hook(lambda m, a, o: cap.__setitem__('pre', a[0].detach())))

    def run(seqs, need_u=None):
        i_, att = R._batch(seqs)
        hs = R.model.model(input_ids=i_, attention_mask=att).last_hidden_state
        last = len(seqs[0]) - 1
        lg = R.model.lm_head(hs[:, last]).float()
        lp = torch.log_softmax(lg, -1)
        ms = torch.stack([torch.logsumexp(lp[:, e], -1) for e in EX], -1)
        u = None
        if need_u is not None:
            row, p = need_u
            pr = lp[row].exp()
            e = EX[p]
            qw = pr[e] / pr[e].sum()
            grad = (qw.to(Wu.dtype) @ Wu[e]).float() - (pr.to(Wu.dtype) @ Wu).float()
            pre = cap['pre'][row, last].float()
            u = gw * grad / torch.sqrt((pre ** 2).mean() + eps)
        return ms.cpu().numpy(), u

    groups = [tuple(g) for g in crit['groups']]
    texts = {g: [i for i, cc in enumerate(C) if cc['split'] == 'disc' and cc['style'] == g[0]] for g in groups}
    Amask = {g: np.array([MF3.in_A(C[i]['task']) for i in texts[g]]) for g in groups}
    vec = lambda s: crit_k * SG[s] * V[s]
    crit_k = 6
    # ---------- 第一遍：直接作用差与只写代码的激活变化 ----------
    DEL = {g: np.zeros((len(texts[g]), len(layer_of)), np.float32) for g in groups}
    RD = []
    for g in groups:
        p = STY.index(g[1])
        for ti, i in enumerate(texts[g]):
            n = max(2, round(0.6 * len(ids[i])))
            st['pos'] = n
            st['inject'] = torch.as_tensor(np.stack([np.zeros_like(V[0]), vec(p), vec(CODE), vec(p) + vec(CODE)]), dtype=torch.float32, device='cuda')
            st['capture'] = True
            try:
                _, u = run([ids[i][:n]] * 4, need_u=(3, p))
            finally:
                st['capture'] = False
            for l in TL:
                a = cap[l][:, n - 1].float()                                 # [4, I]
                q = (WdT[l].float() @ u)                                     # [I]
                DEL[g][ti, off[l]:off[l] + a.shape[1]] = ((a[3] - a[1]) * q).cpu().numpy()
            RD.append(np.concatenate([(cap[l][2, n - 1].float() - cap[l][0, n - 1].float()).cpu().numpy() for l in TL]))
        print(f'@@progress pass1 {g} {round(time.time() - t0)}s', flush=True)
    st['inject'] = None
    inA = np.concatenate([Amask[g] for g in groups])
    allD = np.concatenate([DEL[g] for g in groups])
    top = F.select_units(allD, inA, crit['top_k'])
    rsets = F.random_units_matched(top, layer_of, crit['n_random_sets'], crit['seed'])
    RD = np.array(RD)
    rd_mean = RD.mean(0)
    print(f'@@progress top {[(int(layer_of[i]), int(i - off[layer_of[i]])) for i in top[:12]]}', flush=True)
    # ---------- 第二遍：单元修补 ----------
    def by_layer(units):
        out = {}
        for i in units:
            out.setdefault(int(layer_of[i]), []).append(int(i - off[layer_of[i]]))
        return out
    variants = [('joint', None), ('layers', {l: None for l in TL}), ('top', by_layer(top))] + \
               [(f'rand{j}', by_layer(s)) for j, s in enumerate(rsets)]
    MS = {}
    for g in groups:
        p = STY.index(g[1])
        M = np.zeros((len(texts[g]), 4 + len(variants)), np.float32)
        for ti, i in enumerate(texts[g]):
            n = max(2, round(0.6 * len(ids[i])))
            seq = ids[i][:n]
            st['pos'] = n
            st['inject'] = torch.as_tensor(np.stack([np.zeros_like(V[0]), vec(p), vec(CODE), vec(p) + vec(CODE)]), dtype=torch.float32, device='cuda')
            st['capture'] = True
            try:
                ms, _ = run([seq] * 4)
            finally:
                st['capture'] = False
            M[ti, :4] = ms[:, p]
            for l in TL:
                store[l] = cap[l][1]                                          # 只写散文的中间激活 [T, I]
            st['inject'] = torch.as_tensor(np.stack([vec(p) + vec(CODE)] * len(variants)), dtype=torch.float32, device='cuda')
            pt = {}
            for row, (_, spec) in enumerate(variants):
                if spec is None:
                    continue
                for l, units in spec.items():
                    pt.setdefault(l, {})[row] = units if units is None else torch.as_tensor(units, device='cuda')
            st['patch'] = pt
            try:
                ms, _ = run([seq] * len(variants))
            finally:
                st['patch'] = None
            M[ti, 4:] = ms[:, p]
        MS[g] = M
        np.save(od / f'mf_v4_{g[0]}_{g[1]}.npy', M)
        print(f'@@progress pass2 {g} {round(time.time() - t0)}s', flush=True)
    st['inject'] = None
    # ---------- H3b：自然文本上的单元得分 ----------
    sign = np.sign(rd_mean[top])
    conf = [i for i, cc in enumerate(C) if cc['split'] == 'conf']
    starts = np.concatenate([[0], np.cumsum([len(x) for x in ids])])
    sc, lab = [], []
    tl = by_layer(top)
    for j in range(0, len(conf), 16):
        chunk = conf[j:j + 16]
        st['capture'] = True
        try:
            run([ids[i] for i in chunk])
        finally:
            st['capture'] = False
        for b, i in enumerate(chunk):
            n = len(ids[i])
            shm = meta['shared'][starts[i]:starts[i] + n].astype(bool)
            vals = np.zeros(n)
            for l, units in tl.items():
                a = cap[l][b, :n][:, units].float().cpu().numpy()
                sg = np.array([sign[top.index(off[l] + u)] for u in units])
                vals += a @ sg
            sc += vals[shm].tolist()
            lab += [STY.index(C[i]['style']) == CODE] * int(shm.sum())
    sc, lab = np.array(sc), np.array(lab)
    for h in hooks:
        h.remove()
    np.savez(od / 'mf_v4_units.npz', top=np.array(top), layer_of=layer_of, rd_mean=rd_mean, rsets=np.array(rsets),
             DEL=allD, inA=inA, sc=sc, lab=lab)
    save(model_path, int8, {'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap,
                            'precision': 'int8' if int8 else 'bf16', 'criteria': crit, 'layers': TL, 'switch_layer': ls,
                            'variants': [v[0] for v in variants], 'unit_offsets': {str(k_): int(v) for k_, v in off.items()},
                            'seconds': round(time.time() - t0, 1)})


def census_cards(top, layer_of, off, model_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    out = []
    cache = {}
    for i in top:
        l, j = int(layer_of[i]), int(i - off[int(layer_of[i])])
        if l not in cache:
            cache[l] = np.load(CENSUS / f'L{l}.npz')
        c = cache[l]
        n = float(c['n'])
        mean = float(c['s1'][j] / n)
        out.append({'layer': l, 'unit': j, 'readout_top': [tok.decode([int(t)]) for t in c['readout_top'][j][:6]],
                    'readout_bot': [tok.decode([int(t)]) for t in c['readout_bot'][j][:6]],
                    'mean': mean, 'std': float(np.sqrt(max(c['s2'][j] / n - mean ** 2, 0))), 'amax': float(c['amax'][j]), 'amin': float(c['amin'][j])})
    return out


def do_analyze(model_path, int8):
    from provenance import write_result
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    _, key = XM.key_of(model_path, int8)
    od = MF1.out_dir(key)
    rep = json.loads(report_path(model_path, int8).read_text(encoding='utf-8'))
    U = np.load(od / 'mf_v4_units.npz')
    top, layer_of, rd = [int(x) for x in U['top']], U['layer_of'], U['rd_mean']
    off = {int(k_): v for k_, v in rep['unit_offsets'].items()}
    var = rep['variants']
    C, _ = MF1.tokenize(__import__('transformers').AutoTokenizer.from_pretrained(model_path))
    per = {}
    for g in [tuple(x) for x in crit['groups']]:
        M = np.load(od / f'mf_v4_{g[0]}_{g[1]}.npy')
        texts = [i for i, cc in enumerate(C) if cc['split'] == 'disc' and cc['style'] == g[0]]
        A = np.array([MF3.in_A(C[i]['task']) for i in texts])
        B = ~A
        single, joint = M[B, 1], M[B, 3]
        gap = float((single - joint).mean())
        rest = {v: float((M[B, 4 + j] - joint).mean() / gap) if gap > 0 else float('nan') for j, v in enumerate(var)}
        per['%s:%s' % g] = {'gap_B': gap, 'restore_B': rest,
                            'prose_retain_all': float((M[:, 3] - M[:, 0]).mean() / (M[:, 1] - M[:, 0]).mean())}
    mean = lambda v: float(np.mean([x['restore_B'][v] for x in per.values()]))
    h1 = F.unit_verdict(mean('layers'), mean('top'), [mean(f'rand{j}') for j in range(crit['n_random_sets'])], crit)
    h2 = {'per_group_top': {k_: x['restore_B']['top'] for k_, x in per.items()},
          'pass': bool(all(x['restore_B']['top'] >= crit['H2_group_restore_min'] for x in per.values()))}
    rs = U['rsets']
    rd_top = float(np.abs(rd[top]).mean())
    rd_rand = float(np.mean([np.abs(rd[s]).mean() for s in rs]))
    auc = F.auc(U['sc'][U['lab']], U['sc'][~U['lab']])
    h3 = {'read_top': rd_top, 'read_random': rd_rand, 'read_ratio': rd_top / max(rd_rand, 1e-12), 'auc_natural': auc,
          'n_positions': {'code': int(U['lab'].sum()), 'prose': int((~U['lab']).sum())},
          'pass': bool(rd_top >= crit['H3_read_ratio'] * rd_rand and auc >= crit['H3_auc_min'])}
    units = [{'layer': int(layer_of[i]), 'unit': int(i - off[int(layer_of[i])]), 'delta_A_mean': float(U['DEL'][U['inA']][:, i].mean()),
              'code_read': float(rd[i])} for i in top]
    upd = {'groups': per, 'H1': h1, 'H2': h2, 'H3': h3, 'top_units': units,
           'layer_counts': {str(l): int(sum(1 for u in units if u['layer'] == l)) for l in sorted(set(layer_of.tolist()))},
           'verdict': {'H1': h1['pass'], 'H2': h2['pass'], 'H3': h3['pass']}}
    if Path(model_path).name == 'qwen3-4b' and CENSUS.exists():
        upd['census_cards'] = census_cards(top, layer_of, off, model_path)
    rep = save(model_path, int8, upd)
    p = write_result(f'mf_v4_{Path(model_path).name}{"_int8" if int8 else ""}', rep, model_path=model_path, seed=crit['seed'],
                     tool='scripts/mode_field_v4.py')
    print(json.dumps(rep['verdict']), 'H1', {k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in h1.items()},
          '\nH2', h2['per_group_top'], '\nH3', {k_: v for k_, v in h3.items() if k_ != 'pass'}, '\nlayers', upd['layer_counts'], '\nprovenance →', p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run', 'analyze'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    ap.add_argument('--int8', action='store_true')
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': lambda: do_run(a.model, a.int8), 'analyze': lambda: do_analyze(a.model, a.int8)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
