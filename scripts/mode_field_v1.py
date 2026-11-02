# -*- coding: utf-8 -*-
"""MF-v1：风格模式状态与注意力维持（中文 / 英文 / 英文论文 / Python 代码）。

问题：风格是否作为一个由上下文计算出的持续状态存在于残差流？它能否被写入并改变下一个词的风格？由哪些注意力头维持？
它经 unembedding 是否成为一张「词表分区」偏置？在代码中，一次扰动之后模式能否自行恢复（「整体都是代码」的吸引子迹象）？
材料：scripts/mf_materials_v1.py，24 个算法任务 × 4 种风格 × 5 组标识符填充（480 段），按任务切分：发现 16 任务 / 确认 8 任务。
关键控制：只在跨风格共用 token 的位置判定可读出（嵌入层在这些位置完全相同）；写入方向按共用 token 配平求得。
  H1  确认集共用 token 位置上 6 个风格对的两两准确率都 ≥ 0.8（层与 λ 只按发现集分组交叉验证选定；hs[0] 按构造为机会水平）。
  H2  在 H1 层、确认集散文前缀（中 / 英 / 论文）全部位置沿「代码方向」写入，末位 log M_code（下一词落在代码专属词表的概率质量）
      steer_verdict 成立（三种来源都成立）。
  H3  逐头「代码模式写入」得分（头输出在下一层代码方向上的投影，按共用 token 配平的代码 − 其他差）前 10 头，在确认集代码文本全部位置置零：
      log M_code 平均下降与 H1 层读出间隔平均下降都 ≥ 同层随机 10 头集合最大下降的 3 倍。
  H4  H1 层代码方向经最终归一化权重与 unembedding 后，区分代码专属 token 与其他三种风格专属 token 的 AUC ≥ 0.8。
  H5  描述：确认集代码文本在 40% 处单个位置写入 −6σ 代码方向，log M_code 的变化沿后续 12 个位置的衰减与半衰期；与写入前缀全部位置对照。
  描述：逐层两两准确率、行为（下一词风格与文本风格一致的比例）、GLM4-9B（int8）复现。
用法：python scripts/mode_field_v1.py prereg | collect --model M [--int8] | analyze --model M [--int8] | steer ... | heads ... | analyze ...
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
from shared.dist_protocol import path_scan as P  # noqa: E402
import mf_materials_v1 as MAT  # noqa: E402
import xm_replicate_v1 as XM  # noqa: E402

PREREG = ROOT / 'research/mode_field/MF-v1.prereg.json'
STY = F.STYLES
CODE = STY.index('code')


def prereg_doc():
    C = MAT.corpus()
    return {'id': 'MF-v1', 'schema': 'mode_field_prereg.v1', 'created': '2026-11-02',
            'basis': '用户方案：注意力作为语言模式族的底层实体；先检验风格是否为上下文计算出的持续状态（Phase 275 方案 MF-v1）',
            'models': {'qwen3-4b': 'bf16（主）', 'glm4-9b-chat-hf': 'int8（复现）'},
            'materials': {'source': 'scripts/mf_materials_v1.py', 'split': MAT.split(), 'n_texts': len(C),
                          'sha256_texts': hashlib.sha256(json.dumps([c['text'] for c in C], ensure_ascii=False).encode()).hexdigest()},
            'shared_positions': '同一划分内，某 token 至少出现在两种风格中；H1 的风格对只用两者共有的 token',
            'mass': 'log M_s = log Σ_{w ∈ V_s} p(w | 前缀)，V_s 为发现集的风格专属词表（excl_min_count）',
            'power_note': 'H1 / H4 为连续量阈值；H2 每种来源 40 段确认文本；H3 为 40 段代码文本的连续量配对比较。AGENTS 13 的 McNemar 规划针对逐题二值，不适用',
            'doc': __doc__, 'criteria': F.CRITERIA}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（MF-v2）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def out_dir(key):
    return XM.out_dir(key).parent / 'MF-v1'


def report_path(model_path, int8):
    d = ROOT / 'research/mode_field' / f'MF-v1-{Path(model_path).name}{"-int8" if int8 else ""}'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def save(model_path, int8, upd):
    p = report_path(model_path, int8)
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update(upd)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    return rep


def tokenize(tok):
    C = MAT.corpus()
    ids = [tok(c['text'], add_special_tokens=False).input_ids for c in C]
    return C, ids


def vocab_and_shared(C, ids, crit):
    by = {s: [t for c, i in zip(C, ids) if c['split'] == 'disc' and c['style'] == s for t in i] for s in STY}
    excl = F.exclusive_vocab(by, crit['excl_min_count'])
    shared = {}
    for sp in ('disc', 'conf'):
        seen = {s: set(t for c, i in zip(C, ids) if c['split'] == sp and c['style'] == s for t in i) for s in STY}
        shared[sp] = {t for s in STY for t in seen[s] if sum(t in seen[o] for o in STY) >= 2}
    return excl, shared


def grid_layers(L):
    return list(range(0, L + 1, 3))


# ---------------- 采集 ----------------

def do_collect(model_path, int8):
    import torch
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    crit = pre['criteria']
    snap, key = XM.key_of(model_path, int8)
    od = out_dir(key)
    od.mkdir(parents=True, exist_ok=True)
    tok, model = XM.load(model_path, int8)
    R = XM.Runner(tok, model)
    L = R.L
    C, ids = tokenize(tok)
    excl, shared = vocab_and_shared(C, ids, crit)
    EX = [torch.as_tensor(excl[s], device='cuda') for s in STY]
    G = grid_layers(L)
    meta = {k_: [] for k_ in ('text', 'pos', 'tok', 'style', 'split', 'task', 'item', 'shared')}
    Hg, Hs, M = [], [], []
    t0 = time.time()
    for j in range(0, len(C), 16):
        chunk = list(range(j, min(j + 16, len(C))))
        o = R.forward([ids[i] for i in chunk], hidden=True)
        lp = torch.log_softmax(o.logits.float(), -1)
        mass = torch.stack([torch.logsumexp(lp[..., e], -1) for e in EX], -1).cpu().numpy()      # [B, T, 4]
        hs = torch.stack(o.hidden_states, 2)                                                     # [B, T, L+1, d]
        for b, i in enumerate(chunk):
            n = len(ids[i])
            c = C[i]
            sh = np.array([t in shared[c['split']] for t in ids[i]])
            for k_, v in (('text', [i] * n), ('pos', list(range(n))), ('tok', ids[i]), ('style', [STY.index(c['style'])] * n),
                          ('split', [c['split']] * n), ('task', [c['task']] * n), ('item', [c['item']] * n), ('shared', sh.tolist())):
                meta[k_] += v
            Hg.append(hs[b, :n][:, G].to(torch.float16).cpu().numpy())
            Hs.append(hs[b, :n][torch.as_tensor(sh, device='cuda')].to(torch.float16).cpu().numpy())
            M.append(mass[b, :n])
        del o, lp, hs
        if j % 96 == 0:
            print(f'@@progress collect {j}/{len(C)} {round(time.time() - t0)}s', flush=True)
    meta = {k_: np.array(v) for k_, v in meta.items()}
    np.savez(od / 'fields.npz', Hg=np.concatenate(Hg), Hs=np.concatenate(Hs), M=np.concatenate(M), grid=np.array(G), **meta)
    (od / 'vocab.json').write_text(json.dumps({'excl': excl, 'shared': {k_: sorted(v) for k_, v in shared.items()}}), encoding='utf-8')
    man = {'model': model_path, 'model_snapshot': snap, 'precision': 'int8' if int8 else 'bf16', 'n_blocks': L, 'n_positions': int(len(meta['tok'])),
           'n_shared': int(meta['shared'].sum()), 'excl_sizes': {s: len(excl[s]) for s in STY}, 'seconds': round(time.time() - t0, 1),
           'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip()}
    (od / 'manifest.json').write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps(man, ensure_ascii=False))


def load_fields(key):
    with np.load(out_dir(key) / 'fields.npz') as f:
        return {k_: f[k_] for k_ in f.files}


# ---------------- 分析一：H1 / H4 / 方向 ----------------

def do_analyze(model_path, int8):
    import torch
    from provenance import write_result
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    crit = pre['criteria']
    snap, key = XM.key_of(model_path, int8)
    od = out_dir(key)
    z = load_fields(key)
    G = [int(x) for x in z['grid']]
    L = int(z['Hs'].shape[1]) - 1
    y, tok, split, sh = z['style'], z['tok'], z['split'], z['shared'].astype(bool)
    D, Cf = split == 'disc', split == 'conf'
    task = z['task']
    t0 = time.time()
    res = {}
    # H1：层与 λ 由发现集分组交叉验证
    HgD = z['Hg'][D]
    Xs = {l: HgD[:, gi].astype(np.float32) for gi, l in enumerate(G)}
    sel = F.select_layer(Xs, y[D], tok[D], task[D], crit)
    l1, lam = sel['layer'], sel['lambda_rel']
    gi1 = G.index(l1)
    prof = {}
    for gi, l in enumerate(G):
        m = F.ridge_multi(Xs[l], y[D], 4, lam)
        prof[l] = F.pairwise_acc(F.scores(m, z['Hg'][Cf][:, gi].astype(np.float32)), y[Cf], tok[Cf])
    pa = prof[l1]
    names = {f'{a}-{b}': f'{STY[a]}-{STY[b]}' for a in range(4) for b in range(a + 1, 4)}
    res['H1'] = {'layer': l1, 'lambda_rel': lam, 'cv_acc': sel['cv_acc'], 'pair_acc': {names[k_]: v for k_, v in pa.items()},
                 'hs0_pair_acc': {names[k_]: v for k_, v in prof[0].items()},
                 'profile': {str(l): {names[k_]: v for k_, v in p.items()} for l, p in prof.items()},
                 'pass': bool(all(v >= crit['H1_pair_acc_min'] for v in pa.values()))}
    model1 = F.ridge_multi(Xs[l1], y[D], 4, lam)
    print(f'@@progress H1 layer={l1} pairs={ {names[k_]: round(v, 3) for k_, v in pa.items()} } {round(time.time() - t0)}s', flush=True)
    # 共用 token 位置的全层代码方向（只用发现集）
    shD = D[sh]
    Hs = z['Hs']
    ys, ts = y[sh], tok[sh]
    V = np.zeros((L + 1, Hs.shape[-1]), np.float32)
    SG = np.zeros(L + 1)
    for l in range(L + 1):
        X = Hs[shD, l].astype(np.float32)
        v = F.token_matched_direction(X, ys[shD], ts[shD], CODE)
        nv = np.linalg.norm(v)
        V[l] = v / nv if nv > 0 else v
        SG[l] = float((X @ V[l]).std())
    # 行为：下一个词的风格是否与文本风格一致（确认集）
    beh = {STY[s]: float((z['M'][Cf & (y == s)].argmax(1) == s).mean()) for s in range(4)}
    # H4：unembedding
    vocab = json.loads((od / 'vocab.json').read_text(encoding='utf-8'))
    excl = vocab['excl']
    wu, gw = unembed_weights(model_path, int8)
    h4 = {}
    for l in G:
        sc = wu @ (gw * V[l])
        h4[l] = F.auc(sc[excl['code']], sc[[t for s in STY if s != 'code' for t in excl[s]]])
    res['H4'] = {'layer': l1, 'auc': h4[l1], 'profile': {str(l): v for l, v in h4.items()}, 'pass': bool(h4[l1] >= crit['H4_auc_min'])}
    res['behavior_style_match'] = beh
    np.savez(od / 'directions.npz', V=V, SG=SG, W=model1[0], mu=model1[1], b=model1[2], layer=l1)
    rep = save(model_path, int8, {'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap,
                                  'precision': 'int8' if int8 else 'bf16', 'n_blocks': L, 'criteria': crit, 'H1': res['H1'], 'H4': res['H4'],
                                  'behavior_style_match': beh, 'excl_sizes': {s: len(excl[s]) for s in STY}})
    verdict = {h: rep[h]['pass'] for h in ('H1', 'H2', 'H3', 'H4') if h in rep and 'pass' in rep[h]}
    rep = save(model_path, int8, {'verdict': verdict})
    if all(h in rep for h in ('H2', 'H3', 'H5')):
        p = write_result(f'mf_v1_{Path(model_path).name}{"_int8" if int8 else ""}', rep, model_path=model_path, seed=crit['seed'],
                         tool='scripts/mode_field_v1.py')
        print('provenance →', p)
    print(json.dumps(verdict, ensure_ascii=False), f'H4 auc={h4[l1]:.3f} beh={beh}')


def unembed_weights(model_path, int8):
    """只读 lm_head 与最终归一化权重（不加载整个模型）。"""
    import torch
    from safetensors import safe_open
    idx = Path(model_path) / 'model.safetensors.index.json'
    files = sorted(set(json.loads(idx.read_text())['weight_map'].values())) if idx.exists() else ['model.safetensors']
    want = {'lm_head.weight': None, 'model.norm.weight': None, 'model.embed_tokens.weight': None}
    for fn in files:
        with safe_open(str(Path(model_path) / fn), framework='pt') as f:
            for k_ in f.keys():
                if k_ in want:
                    want[k_] = f.get_tensor(k_).float().numpy()
    wu = want['lm_head.weight'] if want['lm_head.weight'] is not None else want['model.embed_tokens.weight']
    return wu, want['model.norm.weight']


# ---------------- 干预：H2 / H5 ----------------

def do_steer(model_path, int8):
    import torch
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    crit = pre['criteria']
    snap, key = XM.key_of(model_path, int8)
    od = out_dir(key)
    dz = np.load(od / 'directions.npz')
    V, SG, l1 = dz['V'], dz['SG'], int(dz['layer'])
    tok, model = XM.load(model_path, int8)
    R = XM.Runner(tok, model)
    L = R.L
    C, ids = tokenize(tok)
    excl = json.loads((od / 'vocab.json').read_text(encoding='utf-8'))['excl']
    EXc = torch.as_tensor(excl['code'], device='cuda')
    rng = np.random.default_rng(crit['seed'])
    rand = [v / np.linalg.norm(v) for v in rng.normal(size=(crit['n_random'], V.shape[1]))]
    k = crit['k_grid']
    nk = len(k)
    t0 = time.time()

    def code_mass(seqs, layer, pos_rows, vecs, at):
        """at：每行要读的位置列表；返回 [B, len(at[0])] 的 log M_code。"""
        Vt = torch.as_tensor(np.stack(vecs), dtype=torch.float32, device='cuda')

        def hook(mod, args, kwargs):
            h = (args[0] if args else kwargs['hidden_states']).clone()
            for b, ps in enumerate(pos_rows):
                if ps:
                    h[b, ps] += Vt[b].to(h.dtype)
            if args:
                return (h,) + tuple(args[1:]), kwargs
            kwargs['hidden_states'] = h
            return args, kwargs
        hk = R.blocks[layer].register_forward_pre_hook(hook, with_kwargs=True)
        try:
            ids_, att = R._batch(seqs)
            hs = R.model.model(input_ids=ids_, attention_mask=att).last_hidden_state
        finally:
            hk.remove()
        rows = torch.arange(len(seqs), device='cuda')[:, None].expand(-1, len(at[0]))
        lp = torch.log_softmax(R.model.lm_head(hs[rows, torch.as_tensor(at, device='cuda')]).float(), -1)
        return torch.logsumexp(lp[..., EXc], -1).cpu().numpy()

    conf = [i for i, c in enumerate(C) if c['split'] == 'conf']
    out = {'H2': {}, 'H2_layers': {}, 'H5': {}}
    layers = sorted(set([l1] + list(range(6, L, 6))))
    for l in layers:
        vecs = [kk * SG[l] * V[l] for kk in k] + [kk * SG[l] * r for r in rand for kk in k]
        for src in crit['H2_sources']:
            D_ = []
            for i in [i for i in conf if C[i]['style'] == src]:
                n = max(2, round(0.6 * len(ids[i])))
                pre_ = ids[i][:n]
                m = code_mass([pre_] * len(vecs), l, [list(range(n))] * len(vecs), vecs, [[n - 1]] * len(vecs))[:, 0]
                D_.append(np.concatenate([m[j * nk:(j + 1) * nk] - m[j * nk + k.index(0)] for j in range(1 + crit['n_random'])]))
            D_ = np.array(D_)
            vv = P.steer_verdict(D_[:, :nk], np.stack([D_[:, (1 + j) * nk:(2 + j) * nk] for j in range(crit['n_random'])]), k, crit['min_swing'])
            out['H2_layers'].setdefault(str(l), {})[src] = vv
            if l == l1:
                out['H2'][src] = vv
            print(f'@@progress H2 L{l} {src} swing={vv["swing"]:.2f} ok={vv["supported_write"]} {round(time.time() - t0)}s', flush=True)
    out['H2']['pass'] = bool(all(out['H2'][s]['supported_write'] for s in crit['H2_sources']))
    out['H2']['layer'] = l1
    # H5：代码文本单点写入的衰减
    W_ = 12
    curves = {'point': [], 'prefix': [], 'random': []}
    for i in [i for i in conf if C[i]['style'] == 'code']:
        n = len(ids[i])
        t = round(0.4 * n)
        if t + W_ >= n:
            continue
        at = [list(range(t, t + W_ + 1))] * 4
        vecs = [np.zeros_like(V[l1]), -6 * SG[l1] * V[l1], -6 * SG[l1] * V[l1], -6 * SG[l1] * rand[0]]
        pos = [[], [t], list(range(t + 1)), [t]]
        m = code_mass([ids[i]] * 4, l1, pos, vecs, at)
        curves['point'].append(m[1] - m[0]); curves['prefix'].append(m[2] - m[0]); curves['random'].append(m[3] - m[0])
    cur = {k_: np.mean(v, 0).tolist() for k_, v in curves.items()}
    out['H5'] = {'layer': l1, 'offsets': list(range(W_ + 1)), 'mean_delta': cur, 'half_life_point': F.half_life(cur['point']),
                 'n_texts': len(curves['point'])}
    print(f'@@progress H5 point={np.round(cur["point"], 2).tolist()} hl={out["H5"]["half_life_point"]:.2f}', flush=True)
    save(model_path, int8, {'H2': out['H2'], 'H2_layers': out['H2_layers'], 'H5': out['H5'], 'seconds_steer': round(time.time() - t0, 1)})


# ---------------- 逐头：H3 ----------------

def do_heads(model_path, int8):
    import torch
    pre = json.loads(PREREG.read_text(encoding='utf-8'))
    crit = pre['criteria']
    snap, key = XM.key_of(model_path, int8)
    od = out_dir(key)
    dz = np.load(od / 'directions.npz')
    V, l1 = dz['V'], int(dz['layer'])
    Wr, mur, br = dz['W'], dz['mu'], dz['b']
    tok, model = XM.load(model_path, int8)
    R = XM.Runner(tok, model)
    L = R.L
    cfg = model.config
    H = cfg.num_attention_heads
    dh = getattr(cfg, 'head_dim', None) or cfg.hidden_size // H
    C, ids = tokenize(tok)
    excl, shared = vocab_and_shared(C, ids, crit)
    EXc = torch.as_tensor(excl['code'], device='cuda')
    t0 = time.time()
    U = []                                                    # U[l] = W_O^T v̂_{l+1}，[H·dh]
    for li, b in enumerate(R.blocks):
        op = b.self_attn.o_proj
        I = torch.eye(H * dh, device='cuda', dtype=torch.bfloat16)
        Wt = (op(I).float() - op(torch.zeros(1, H * dh, device='cuda', dtype=torch.bfloat16)).float())
        U.append(Wt @ torch.as_tensor(V[li + 1], device='cuda'))
        del Wt
    cap = {}
    hooks = [b.self_attn.o_proj.register_forward_pre_hook(lambda m, a, li=li: cap.__setitem__(li, a[0])) for li, b in enumerate(R.blocks)]
    abl = {'heads': None}

    def abl_hook(li):
        def f(m, a):
            if abl['heads'] is None or li not in abl['heads']:
                return None
            z = a[0].clone()
            for b_, hs_ in enumerate(abl['heads'][li]):
                for h in hs_:
                    z[b_, :, h * dh:(h + 1) * dh] = 0
            return (z,) + tuple(a[1:])
        return f
    hooks += [b.self_attn.o_proj.register_forward_pre_hook(abl_hook(li)) for li, b in enumerate(R.blocks)]
    # 1 发现集共用 token 位置的逐头写入得分
    disc = [i for i, c in enumerate(C) if c['split'] == 'disc']
    Sc, ys, ts = [], [], []
    for j in range(0, len(disc), 16):
        chunk = disc[j:j + 16]
        R.forward([ids[i] for i in chunk])
        for b, i in enumerate(chunk):
            n = len(ids[i])
            m = [p for p in range(n) if ids[i][p] in shared['disc']]
            if not m:
                continue
            mt = torch.as_tensor(m, device='cuda')
            row = torch.stack([(cap[li][b, mt].float() * U[li]).view(len(m), H, dh).sum(-1) for li in range(L)], 1)   # [m, L, H]
            Sc.append(row.reshape(len(m), -1).cpu().numpy())
            ys += [STY.index(C[i]['style'])] * len(m); ts += [ids[i][p] for p in m]
    Sc, ys, ts = np.concatenate(Sc), np.array(ys), np.array(ts)
    score = F.token_matched_direction(Sc, ys, ts, CODE)        # 每个头：共用 token 配平的「代码 − 其他」
    top = [int(i) for i in np.argsort(-score)[:crit['top_k']]]
    rsets = P.random_matched(top, H, crit['n_random_sets'], crit['seed'])
    print(f'@@progress H3 top={[(i // H, i % H, round(float(score[i]), 2)) for i in top]} {round(time.time() - t0)}s', flush=True)
    # 2 确认集代码文本：消融
    as_map = lambda flat: {li: [i % H for i in flat if i // H == li] for li in {i // H for i in flat}}
    variants = [None, as_map(top)] + [as_map(s) for s in rsets]
    conf_code = [i for i, c in enumerate(C) if c['split'] == 'conf' and c['style'] == 'code']
    hs_l = {}
    hk = R.blocks[l1].register_forward_pre_hook(lambda m, a, kw: hs_l.__setitem__('h', (a[0] if a else kw['hidden_states'])), with_kwargs=True) \
        if l1 < L else model.model.norm.register_forward_hook(lambda m, a, o: hs_l.__setitem__('h', o))   # hs[L] 为最终归一化之后
    Mc = np.zeros((len(variants), len(conf_code)))
    Mg = np.zeros((len(variants), len(conf_code)))
    for vi, var in enumerate(variants):
        for j in range(0, len(conf_code), 8):
            chunk = conf_code[j:j + 8]
            abl['heads'] = None if var is None else {li: [hs_ for _ in chunk] for li, hs_ in var.items()}
            try:
                o = R.forward([ids[i] for i in chunk])
            finally:
                abl['heads'] = None
            lp = torch.log_softmax(o.logits.float(), -1)
            mc = torch.logsumexp(lp[..., EXc], -1).cpu().numpy()
            h = hs_l['h'].float().cpu().numpy()
            for b, i in enumerate(chunk):
                n = len(ids[i])
                Mc[vi, j + b] = mc[b, 3:n].mean()
                m = [p for p in range(n) if ids[i][p] in shared['conf']]
                s = (h[b, m] - mur) @ Wr + br
                Mg[vi, j + b] = float((s[:, CODE] - np.delete(s, CODE, 1).max(1)).mean()) if m else np.nan
    hk.remove()
    for h in hooks:
        h.remove()
    drop = lambda A, v: float(np.nanmean(A[0] - A[v]))
    rM = [drop(Mc, 2 + j) for j in range(len(rsets))]
    rG = [drop(Mg, 2 + j) for j in range(len(rsets))]
    h3 = {'top_heads': [{'layer': i // H, 'head': i % H, 'score': float(score[i])} for i in top], 'random_sets': rsets,
          'mass_drop': drop(Mc, 1), 'mass_random_drops': rM, 'margin_drop': drop(Mg, 1), 'margin_random_drops': rG,
          'base_log_mass': float(Mc[0].mean()), 'base_margin': float(np.nanmean(Mg[0])),
          'score_positive_sum': float(score[score > 0].sum()), 'score_top_share': float(score[top].sum() / max(score[score > 0].sum(), 1e-30)),
          'heads_for_50': int(np.searchsorted(np.cumsum(np.sort(score[score > 0])[::-1]) / max(score[score > 0].sum(), 1e-30), 0.5) + 1)}
    h3['pass'] = bool(h3['mass_drop'] > 0 and h3['margin_drop'] > 0 and h3['mass_drop'] >= crit['H3_ratio_min'] * max(max(rM), 1e-9)
                      and h3['margin_drop'] >= crit['H3_ratio_min'] * max(max(rG), 1e-9))
    np.save(od / 'head_scores.npy', score)
    print(f'@@progress H3 mass {h3["mass_drop"]:.3f} vs {max(rM):.3f}; margin {h3["margin_drop"]:.3f} vs {max(rG):.3f} pass={h3["pass"]} '
          f'{round(time.time() - t0)}s', flush=True)
    save(model_path, int8, {'H3': h3, 'n_heads': H, 'seconds_heads': round(time.time() - t0, 1)})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'collect', 'analyze', 'steer', 'heads'])
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    ap.add_argument('--int8', action='store_true')
    a = ap.parse_args()
    {'prereg': do_prereg, 'collect': lambda: do_collect(a.model, a.int8), 'analyze': lambda: do_analyze(a.model, a.int8),
     'steer': lambda: do_steer(a.model, a.int8), 'heads': lambda: do_heads(a.model, a.int8)}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
