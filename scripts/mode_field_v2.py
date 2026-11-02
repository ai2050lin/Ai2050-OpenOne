# -*- coding: utf-8 -*-
"""MF-v2：风格状态从哪些层的注意力汇集而来；四种风格是否各有一个单侧「打开」开关，同时打开两个时是叠加还是胜者通吃。

材料与采集沿用 MF-v1（scripts/mf_materials_v1.py；MF-v1 的 fields.npz / vocab.json），确认集 = 8 个未见任务。
A  汇集层段（注意力局部窗口）：在指定层上，把每个查询位置的注意力限制为当前及前 3 个位置（另保留首位置作注意力汇点），
   在读出层 r（网格层中最接近 0.75L）上，用发现集训练的 4 类读出器读「深共用」位置（当前与前 3 个 token 都是共用 token，
   因此局部窗口内没有任何风格专属 token）。条件：不改 / r 之前全部层 / r 之前的层均分 6 段、每次只改一段。
   H1  全部前段层改为局部窗口后，深共用位置（英文 / 论文 / 代码 3 对）两两准确率均值下降 ≥ 0.2。
   H2  下降最大的单段 ≥ 全部下降的 50%（汇集集中在一段层上；哪一段只描述）。
B  单侧开关与叠加（层 round(0.5L)，确认集各风格前 60% 前缀，全部前缀位置写入）：
   方向 v_s = 共用 token 配平的「风格 s − 其他」，σ_s 为发现集共用位置投影标准差。
   H3  12 个（目标 s × 来源 c ≠ s）中至少 10 个：Δlog M_s(+6σ) ≥ 1、Δ(+6) > Δ(+3) > 0、Δ(+6) 大于每个随机方向的 Δ(+6)。
   H4  12 个（来源 c，目标对 {a, b} ⊂ 其余三种）中至少 8 个：同时写入 6σ_a v_a + 6σ_b v_b 时，M_a、M_b 的变化都保留单独写入的 ≥ 50%。
   描述：四个方向的两两 cos；同时写入时第三种风格的变化。
用法：python scripts/mode_field_v2.py prereg | run --model M [--int8] | analyze --model M [--int8]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]

from shared.dist_protocol import mode_field as F  # noqa: E402
import mode_field_v1 as MF1  # noqa: E402
import path_scan_v1 as PS  # noqa: E402
import xm_replicate_v1 as XM  # noqa: E402

PREREG = ROOT / 'research/mode_field/MF-v2.prereg.json'
STY = F.STYLES
PAIRS3 = [(1, 2), (1, 3), (2, 3)]          # 英文–论文、英文–代码、论文–代码（中文没有深共用位置）


def prereg_doc():
    return {'id': 'MF-v2', 'schema': 'mode_field_prereg.v2', 'created': '2026-11-02',
            'basis': 'MF-v1（Phase 275）：风格是持续的上下文状态（共用 token 位置第 3 层起可读），写入只能单侧打开；H3 读出层在头上游的设计错误在此修正（读出层在被改层之后）',
            'models': {'qwen3-4b': 'bf16（主）', 'glm4-9b-chat-hf': 'int8（复现）'},
            'materials': 'MF-v1 材料与确认集（8 个未见任务）；读出器只用发现集训练，λ 取 MF-v1 H1 选定值',
            'attention': 'A 部分用 eager 注意力，在 self_attn 的前向钩子里替换加性因果掩码',
            'power_note': '连续量阈值与组合计数；深共用位置数沿用 MF-v1（确认集英文 190、论文 102、代码 62，Qwen）',
            'doc': __doc__, 'criteria': F.CRITERIA_V2}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（MF-v3）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def report_path(model_path, int8):
    d = ROOT / 'research/mode_field' / f'MF-v2-{Path(model_path).name}{"-int8" if int8 else ""}'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def save(model_path, int8, upd):
    p = report_path(model_path, int8)
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update(upd)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    return rep


def deep_mask(z, n):
    sh, pos = z['shared'].astype(bool), z['pos']
    I = np.arange(len(sh))
    d = sh.copy()
    for j in range(1, n):
        v = pos >= j
        d &= v & sh[np.where(v, I - j, 0)]
    return d


def do_run(model_path, int8):
    import torch
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    snap, key = XM.key_of(model_path, int8)
    od = MF1.out_dir(key)
    z = MF1.load_fields(key)
    v1 = json.loads(MF1.report_path(model_path, int8).read_text(encoding='utf-8'))
    lam = v1['H1']['lambda_rel']
    excl = json.loads((od / 'vocab.json').read_text(encoding='utf-8'))['excl']
    G = [int(x) for x in z['grid']]
    L = int(z['Hs'].shape[1]) - 1
    r = min(G, key=lambda g: abs(g - crit['readout_rel'] * L))
    y, tok, sp, sh = z['style'], z['tok'], z['split'], z['shared'].astype(bool)
    D = sp == 'disc'
    reader = F.ridge_multi(z['Hg'][D][:, G.index(r)].astype(np.float32), y[D], 4, lam)
    deep = deep_mask(z, crit['deep_n'])
    # B 的方向在加载模型之前算好，随后释放大数组（GLM 的 MF-v1 场约 5 GB，与模型同时驻留会耗尽内存）
    ls = round(crit['switch_rel'] * L)
    Xs = z['Hs'][D[sh], ls].astype(np.float32)
    ys, ts = y[sh][D[sh]], tok[sh][D[sh]]
    V, SG = {}, {}
    for s in range(4):
        v = F.token_matched_direction(Xs, ys, ts, s)
        V[s] = v / np.linalg.norm(v)
        SG[s] = float((Xs @ V[s]).std())
    del Xs
    n_tok, n_text = len(tok), len(set(z['text']))
    for k_ in ('Hg', 'Hs', 'M'):
        z.pop(k_)
    import gc
    gc.collect()
    tok_, model = PS.load(model_path, int8, eager=True)
    R = XM.Runner(tok_, model)
    C, ids = MF1.tokenize(tok_)
    assert len(ids) == n_text and sum(map(len, ids)) == n_tok, '分词与 MF-v1 不一致'
    starts = np.concatenate([[0], np.cumsum([len(i) for i in ids])])
    EX = [torch.as_tensor(excl[s], device='cuda') for s in STY]
    t0 = time.time()
    out = {'readout_layer': r, 'n_blocks': L}
    # ---------- A：汇集层段 ----------
    bands = [list(map(int, b)) for b in np.array_split(np.arange(r), crit['n_bands'])]
    conds = {'base': [], 'full': list(range(r)), **{f'band{i}': b for i, b in enumerate(bands)}}
    win = {'layers': set()}

    def mhook(li):
        def f(m, a, kw):
            if li in win['layers'] and kw.get('attention_mask') is not None:
                kw['attention_mask'] = F.window_mask(kw['attention_mask'], crit['window'])
                return a, kw
            return None
        return f
    hooks = [b.self_attn.register_forward_pre_hook(mhook(li), with_kwargs=True) for li, b in enumerate(R.blocks)]
    cap = {}
    hooks.append(R.blocks[r].register_forward_pre_hook(lambda m, a, kw: cap.__setitem__('h', a[0] if a else kw['hidden_states']), with_kwargs=True))
    texts = [i for i, c in enumerate(C) if c['split'] == 'conf' and c['style'] in ('en', 'paper', 'code')]
    rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in texts])
    sel = rows[deep[rows]]
    res = {}
    for name, layers in conds.items():
        win['layers'] = set(layers)
        Hr = np.zeros((len(tok), reader[0].shape[0]), np.float32)
        Ms = np.zeros((len(tok), 4), np.float32)
        for j in range(0, len(texts), 8):
            chunk = texts[j:j + 8]
            o = R.forward([ids[i] for i in chunk])
            lp = torch.log_softmax(o.logits.float(), -1)
            ms = torch.stack([torch.logsumexp(lp[..., e], -1) for e in EX], -1).cpu().numpy()
            h = cap['h'].float().cpu().numpy()
            for b, i in enumerate(chunk):
                n = len(ids[i])
                Hr[starts[i]:starts[i] + n] = h[b, :n]
                Ms[starts[i]:starts[i] + n] = ms[b, :n]
        S = F.scores(reader, Hr[sel])
        pa = F.pairwise_acc(S, y[sel], tok[sel])
        accs = {f'{STY[a]}-{STY[b]}': pa[f'{a}-{b}'] for a, b in PAIRS3}
        tm = Ms[sel, y[sel]]
        res[name] = {'layers': [min(layers), max(layers)] if layers else [], 'pair_acc': accs,
                     'mean_acc': float(np.nanmean(list(accs.values()))),
                     'margin': float(np.mean(S[np.arange(len(sel)), y[sel]] - np.where(np.eye(4, dtype=bool)[y[sel]], -np.inf, S).max(1))),
                     'true_style_log_mass': float(tm.mean())}
        print(f'@@progress A {name} {res[name]["layers"]} acc={res[name]["mean_acc"]:.3f} mass={res[name]["true_style_log_mass"]:.2f} '
              f'{round(time.time() - t0)}s', flush=True)
    win['layers'] = set()
    for h in hooks:
        h.remove()
    out['A'] = {'conditions': res, 'n_deep_positions': {STY[s]: int((y[sel] == s).sum()) for s in range(4)}, 'bands': bands}
    # ---------- B：单侧开关与叠加 ----------
    cosm = {f'{STY[a]}-{STY[b]}': float(V[a] @ V[b]) for a, b in combinations(range(4), 2)}
    rng = np.random.default_rng(crit['seed'])
    rand = [v / np.linalg.norm(v) for v in rng.normal(size=(crit['n_random'], len(V[0])))]
    sg_mean = float(np.mean(list(SG.values())))

    def mass(seqs, pos_rows, vecs):
        Vt = torch.as_tensor(np.stack(vecs), dtype=torch.float32, device='cuda')

        def hook(mod, args, kwargs):
            h = (args[0] if args else kwargs['hidden_states']).clone()
            for b, ps in enumerate(pos_rows):
                h[b, ps] += Vt[b].to(h.dtype)
            if args:
                return (h,) + tuple(args[1:]), kwargs
            kwargs['hidden_states'] = h
            return args, kwargs
        hk = R.blocks[ls].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            i_, att = R._batch(seqs)
            hs = R.model.model(input_ids=i_, attention_mask=att).last_hidden_state
        finally:
            hk.remove()
        last = torch.as_tensor([len(s) - 1 for s in seqs], device='cuda')
        lp = torch.log_softmax(R.model.lm_head(hs[torch.arange(len(seqs), device='cuda'), last]).float(), -1)
        return torch.stack([torch.logsumexp(lp[:, e], -1) for e in EX], -1).cpu().numpy()      # [B, 4]

    B = {}
    for c in range(4):
        tg = [s for s in range(4) if s != c]
        spec = [('base', None, 0)] + [(f'{STY[s]}+{k}', s, k) for s in tg for k in (3, 6)] + \
               [(f'rand{j}+{k}', ('r', j), k) for j in range(crit['n_random']) for k in (3, 6)] + \
               [(f'{STY[a]}&{STY[b]}', (a, b), 6) for a, b in combinations(tg, 2)]
        vals = []
        for i in [i for i, cc in enumerate(C) if cc['split'] == 'conf' and cc['style'] == STY[c]]:
            n = max(2, round(0.6 * len(ids[i])))
            vecs = []
            for _, s, k in spec:
                if s is None:
                    vecs.append(np.zeros_like(V[0]))
                elif isinstance(s, int):
                    vecs.append(k * SG[s] * V[s])
                elif s[0] == 'r':
                    vecs.append(k * sg_mean * rand[s[1]])
                else:
                    vecs.append(k * SG[s[0]] * V[s[0]] + k * SG[s[1]] * V[s[1]])
            m = mass([ids[i][:n]] * len(spec), [list(range(n))] * len(spec), vecs)
            vals.append(m - m[0])
        vals = np.array(vals)                                      # [n_text, n_spec, 4]
        B[STY[c]] = {'spec': [x[0] for x in spec], 'mean_delta': vals.mean(0).tolist()}
        np.save(od / f'mf_v2_B_{STY[c]}.npy', vals)
        print(f'@@progress B source={STY[c]} {round(time.time() - t0)}s', flush=True)
    out['B'] = {'layer': ls, 'sigma': {STY[s]: SG[s] for s in range(4)}, 'cos': cosm, 'by_source': B}
    save(model_path, int8, {'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap,
                            'precision': 'int8' if int8 else 'bf16', 'criteria': crit, 'run': out, 'seconds': round(time.time() - t0, 1)})


def do_analyze(model_path, int8):
    from provenance import write_result
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    _, key = XM.key_of(model_path, int8)
    od = MF1.out_dir(key)
    rep = json.loads(report_path(model_path, int8).read_text(encoding='utf-8'))
    A = rep['run']['A']['conditions']
    base = A['base']['mean_acc']
    full = base - A['full']['mean_acc']
    bd = [base - A[f'band{i}']['mean_acc'] for i in range(crit['n_bands'])]
    h1 = {'base_acc': base, 'full_window_acc': A['full']['mean_acc'], 'drop': full, 'pass': bool(full >= crit['H1_drop_min'])}
    h2 = {**F.band_localize(full, bd, crit), 'band_drops': bd, 'band_layers': rep['run']['A']['bands']}
    h3, h4 = {}, {}
    for c in range(4):
        vals = np.load(od / f'mf_v2_B_{STY[c]}.npy')
        spec = rep['run']['B']['by_source'][STY[c]]['spec']
        ix = {s: j for j, s in enumerate(spec)}
        tg = [s for s in range(4) if s != c]
        single = {}
        for s in tg:
            d_op = np.stack([np.zeros(len(vals)), vals[:, ix[f'{STY[s]}+3'], s], vals[:, ix[f'{STY[s]}+6'], s]], 1)
            d_r = np.stack([np.stack([np.zeros(len(vals)), vals[:, ix[f'rand{j}+3'], s], vals[:, ix[f'rand{j}+6'], s]], 1)
                            for j in range(crit['n_random'])])
            h3[f'{STY[s]}<-{STY[c]}'] = F.one_sided_verdict(d_op, d_r)
            single[s] = float(vals[:, ix[f'{STY[s]}+6'], s].mean())
        for a, b in combinations(tg, 2):
            j = ix[f'{STY[a]}&{STY[b]}']
            third = [s for s in tg if s not in (a, b)][0]
            h4[f'{STY[c]}:{STY[a]}&{STY[b]}'] = {**F.retain(float(vals[:, j, a].mean()), single[a], float(vals[:, j, b].mean()), single[b], crit),
                                                 'third_style_delta': float(vals[:, j, third].mean()),
                                                 'source_style_delta': float(vals[:, j, c].mean())}
    n3, n4 = sum(v['pass'] for v in h3.values()), sum(v['pass'] for v in h4.values())
    verdict = {'H1': h1['pass'], 'H2': h2['pass'], 'H3': bool(n3 >= crit['H3_min_combos']), 'H4': bool(n4 >= crit['H4_min_triples'])}
    rep = save(model_path, int8, {'H1': h1, 'H2': h2, 'H3': {'combos': h3, 'n_pass': n3}, 'H4': {'triples': h4, 'n_pass': n4}, 'verdict': verdict})
    p = write_result(f'mf_v2_{Path(model_path).name}{"_int8" if int8 else ""}', rep, model_path=model_path, seed=crit['seed'],
                     tool='scripts/mode_field_v2.py')
    print(json.dumps(verdict), f'drop={full:.3f} bands={np.round(bd, 3).tolist()} H3 {n3}/12 H4 {n4}/12', '\nprovenance →', p)


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
