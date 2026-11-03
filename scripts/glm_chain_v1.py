# -*- coding: utf-8 -*-
"""GL-v1：在 GLM4-9B（bf16）上复现「代码压过散文」的门控链（第三个架构；GC-v1 + GC-v2 的方法合并为一次运行）。

GLM 与 Qwen / Gemma 的结构差异：注意力与 MLP 之前各一个 RMSNorm（权重即缩放，无 1 + w，无后置归一化），
gate 与 up 合并为 gate_up_proj（前 I 行 gate、后 I 行 up），SwiGLU；32 个头、2 个 KV 组（每组 16 头），头维 128，部分 RoPE。
bf16 约 18.7 GB，16 GB 显存放不下：嵌入表与前若干块的权重留在 CPU（锁页内存），执行该块时搬上 GPU、执行完换回 CPU 副本；
后三分之一的块（单元池）始终在 GPU。不使用任何量化。文本不加特殊前缀（与 MF-v1 / MF-v4 的 GLM 运行一致）。
材料同 MF-v1；写入范式同 MF-v4（块 round(L/2) 写入 6σ，文本前 60%，A / B 任务切分）；混合文本同 WS-v3 发现集。
  S0 方向：发现集共用 token 位置上的 token 配平代码方向 v_l（单元所在块的 hs[l]）。
  S1 单元：候选为块 ≥ round(2L/3) 的全部 MLP 中间单元；直接作用差 Δ_i = (A_同时 − A_只散文)_i ·（W_down 列对 log M_p 的梯度），
      A 任务上取最负 32 个。H1：B 任务上只把这 32 个单元换回「只散文」的值，散文恢复（三组平均）≥ 0.2，且 ≥ 3 × 同层随机 32 单元（5 组）。
  S2 读与编辑：读出向量 a_i = γ_post_attn ⊙ W_gate[i]（up 同理）。
      H2：对 v_l 的读分数 ≥ 同层随机集合（200 组）3 倍，p ≤ 0.01。
      H3：对冲突差值方向 d_l 的读分数 ≥ 3 倍，p ≤ 0.01（d_l：A 任务上块 l MLP 输入逐位置 RMS 归一化后「散文 + 代码」− 「只散文」的平均，位置 0 除外）。
      H4：从 32 个单元的 gate 与 up 行移除 d_l，B 任务恢复 ≥ 0.5 × H1 的修补恢复；同一单元移除随机方向、5 组随机单元移除 d_l，各 ≤ 编辑的 1/3。
      H5：该编辑后确认集自然文本 NLL 相对上升 ≤ 1%（全部与代码文本）。
      描述：移除 v_l 的编辑恢复（GC-v1 H3 的形式）、移除冲突差值前 4 个主方向的恢复、cos(d_l, v_l)。
  S3 上游头与同模式 QK：门控得分 G = Σ s_i g_i（s_i 由发现集自然文本定号），头的直接贡献经单元所在块的 MLP 输入归一化线性化；
      发现集上推向代码最大的 8 个头。在 WS-v3 发现集混合文本（md）上：
      H6：这 8 个头的查询加 +4σ 代码分量，结语位置对代码 key 的质量比 ≥ 1.5 × 原始，且 |log 效应| ≥ 3 × 随机方向。
      H7：代码段 key 移动 −4σ，代码位置对代码 key 的质量比 ≤ 0.8 × 原始，且 |log 效应| ≥ 3 × 随机方向。
被测模型：GLM4-9B-chat-hf（bf16）。与 Qwen / Gemma 的方向、单元、头不作跨模型对应，只比较规律是否成立。
用法：python scripts/glm_chain_v1.py prereg | run | analyze
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]

from shared.dist_protocol import weight_read as W  # noqa: E402
from shared.dist_protocol import mode_field as F  # noqa: E402
import mode_field_v1 as MF1  # noqa: E402
import mode_field_v3 as MF3  # noqa: E402
import fusion_v1 as FU1  # noqa: E402
import weight_space_v3 as WS3  # noqa: E402

PREREG = ROOT / 'research/weight_space/GL-v1.prereg.json'
MODEL = 'models/hf/glm4-9b-chat-hf'
STY = F.STYLES
CODE = STY.index('code')
GROUPS = [('zh', 'en'), ('en', 'zh'), ('paper', 'zh')]
RESERVE_GB = 4.5                       # 留给激活与 logits 的显存


def prereg_doc():
    return {'id': 'GL-v1', 'schema': 'glm_chain_prereg.v1', 'created': '2026-11-03',
            'basis': '门控链在 Qwen3-4B（Phase 278 / 284 / 285 / 287）与 Gemma-3-4B（Phase 290 / 291）上的结果；本版在第三个架构上复现拼图 17、22、25',
            'models': {'glm4-9b-chat-hf': 'bf16'}, 'precision_rule': '不使用任何量化；显存不足部分用 CPU 锁页内存逐块搬运',
            'materials': 'MF-v1 语料（发现 / 确认），MF-v4 的三组写入范式与 A / B 任务切分，WS-v3 的发现集混合文本',
            'differences': 'GC-v2 的 H2 参照 GC-v1 的修补恢复；本版 H4 参照同一次运行的 H1 修补恢复。旧 GLM int8 结果（Phase 273–279）不与本版合并',
            'power_note': '每组 B 任务约 40 段；随机集合 5 / 200 组；混合文本 md 约 160 段；连续量阈值',
            'doc': __doc__, 'criteria': W.CRITERIA_GL}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（GL-v2）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def report_path():
    d = ROOT / 'research/weight_space/GL-v1-glm4-9b-chat-hf'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def save(upd):
    p = report_path()
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update(upd)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    return rep


def load_offloaded(path, keep_gpu_from):
    """bf16 加载到 CPU；块 ≥ keep_gpu_from 上 GPU；更早的块与嵌入表留在锁页内存，执行时搬运。返回 (tok, model, 放置信息)。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=None).eval()
    lm = model.model
    blocks = lm.layers
    L = len(blocks)
    model.lm_head.to('cuda')
    lm.norm.to('cuda')
    lm.rotary_emb.to('cuda')
    lb = sum(p.numel() * p.element_size() for p in blocks[0].parameters())
    for l in range(L - 1, keep_gpu_from - 1, -1):
        blocks[l].to('cuda')
    free = torch.cuda.mem_get_info()[0]
    first_gpu = keep_gpu_from
    while first_gpu > 0 and free - lb > RESERVE_GB * 2 ** 30:
        first_gpu -= 1
        blocks[first_gpu].to('cuda')
        free = torch.cuda.mem_get_info()[0]

    def pin_module(mod):
        cpu = {}
        for n, p in mod.named_parameters():
            p.data = p.data.pin_memory()
            cpu[n] = p.data
        return cpu

    def make_offload(cpu):
        def pre(m, a, kw):
            for n, p in m.named_parameters():
                p.data = cpu[n].to('cuda', non_blocking=True)
            return None

        def post(m, a, kw, o):
            for n, p in m.named_parameters():
                p.data = cpu[n]
            return None
        return pre, post

    for l in range(first_gpu):
        pre, post = make_offload(pin_module(blocks[l]))
        blocks[l].register_forward_pre_hook(pre, with_kwargs=True)
        blocks[l].register_forward_hook(post, with_kwargs=True)
    emb = lm.embed_tokens
    pin_module(emb)
    emb.register_forward_pre_hook(lambda m, a: (a[0].cpu(),))
    emb.register_forward_hook(lambda m, a, o: o.to('cuda', non_blocking=True))
    gc.collect()
    info = {'gpu_blocks_from': first_gpu, 'cpu_blocks': list(range(first_gpu)), 'block_gb': round(lb / 2 ** 30, 3),
            'free_gb_after_load': round(torch.cuda.mem_get_info()[0] / 2 ** 30, 2)}
    return tok, model, info


def do_run():
    import torch
    import transformers.models.glm.modeling_glm as MG
    torch.set_grad_enabled(False)
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    snap = FU1.snapshot(MODEL)
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(MODEL)
    L, nh, nkv, hd, d = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.hidden_size
    I_ = cfg.intermediate_size
    grp = nh // nkv
    ls = round(0.5 * L)
    PL = list(range(round(crit['unit_pool_from_frac'] * L), L))
    tok, model, place = load_offloaded(MODEL, PL[0])
    print('@@placement', place, flush=True)
    lm = model.model
    blocks = lm.layers
    cu = lambda t: t.to('cuda')
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    mf1crit = json.loads(MF1.PREREG.read_text(encoding='utf-8'))['criteria']
    C, ids = MF1.tokenize(tok)
    excl, shared = MF1.vocab_and_shared(C, ids, mf1crit)
    shm = [np.array([t in shared[C[i]['split']] for t in ids[i]]) for i in range(len(C))]
    EX = [torch.as_tensor(excl[s], device='cuda') for s in STY]
    disc = [i for i, c in enumerate(C) if c['split'] == 'disc']
    conf = [i for i, c in enumerate(C) if c['split'] == 'conf']
    t0 = time.time()

    def batch(seqs):
        n = max(map(len, seqs))
        i_ = torch.full((len(seqs), n), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), n), dtype=torch.long)
        for b, s in enumerate(seqs):
            i_[b, :len(s)] = torch.as_tensor(s)
            att[b, :len(s)] = 1
        return i_.cuda(), att.cuda()

    gw_of = lambda mod: cu(mod.weight).float()
    eps_of = lambda mod: mod.variance_epsilon

    def directions(layers, with_inject=False):
        """发现集共用 token 位置上的 token 配平方向；with_inject 时另返回块 ls 输入上四种风格的方向与 σ。"""
        need = sorted(set(layers) | ({ls} if with_inject else set()))
        Xs = {l: [] for l in need}
        ys, ts = [], []
        for j in range(0, len(disc), 8):
            ch = disc[j:j + 8]
            i_, att = batch([ids[i] for i in ch])
            o = lm(input_ids=i_, attention_mask=att, output_hidden_states=True)
            for b, i in enumerate(ch):
                m = torch.as_tensor(shm[i], device='cuda')
                for l in need:
                    Xs[l].append(o.hidden_states[l][b, :len(ids[i])][m].float().cpu().numpy())
                ys += [STY.index(C[i]['style'])] * int(shm[i].sum())
                ts += [t for t, f in zip(ids[i], shm[i]) if f]
            del o
        ys, ts = np.array(ys), np.array(ts)
        vd, V, SG = {}, {}, {}
        for l in need:
            X = np.concatenate(Xs[l])
            Xs[l] = None
            vd[l] = W.unit(F.token_matched_direction(X, ys, ts, CODE))
            if with_inject and l == ls:
                for s in range(4):
                    v = F.token_matched_direction(X, ys, ts, s)
                    V[s] = v / np.linalg.norm(v)
                    SG[s] = float((X @ V[s]).std())
            del X
            gc.collect()
        return vd, V, SG, len(ys)

    # ---------- S0：方向 ----------
    vdir, V, SG, n_shared = directions(PL, with_inject=True)
    print(f'@@S0 directions n_shared={n_shared} {round(time.time() - t0)}s', flush=True)
    # ---------- S1：单元 ----------
    layer_of = np.concatenate([[l] * I_ for l in PL])
    off = {l: k * I_ for k, l in enumerate(PL)}
    st = {'inject': None, 'pos': None, 'capture': False, 'patch': None, 'capd': False}
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
            if st['patch'] is not None and l in st['patch']:
                x = x.clone()
                for row, units in st['patch'][l].items():
                    x[row][:, units] = store[l][:, units].to(x.dtype)
                return (x,) + tuple(a[1:])
            return None
        return f

    def pa_hook(l):
        def f(m, a):
            if st['capd']:
                x = a[0].float()
                cap[('d', l)] = x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + m.variance_epsilon)
            return None
        return f

    hooks = [blocks[ls].register_forward_pre_hook(inj, with_kwargs=True)]
    hooks += [blocks[l].mlp.down_proj.register_forward_pre_hook(dp_hook(l)) for l in PL]
    hooks += [blocks[l].post_attention_layernorm.register_forward_pre_hook(pa_hook(l)) for l in PL]
    hooks.append(lm.norm.register_forward_hook(lambda m, a, o: cap.__setitem__('pre', a[0].detach())))
    Wu = model.lm_head.weight
    gw = gw_of(lm.norm)

    def run(seqs, need_u=None):
        i_, att = batch(seqs)
        hs = lm(input_ids=i_, attention_mask=att).last_hidden_state
        last = len(seqs[0]) - 1
        lp = torch.log_softmax(model.lm_head(hs[:, last]).float(), -1)
        ms = torch.stack([torch.logsumexp(lp[:, e], -1) for e in EX], -1)
        u = None
        if need_u is not None:
            row, p = need_u
            pr = lp[row].exp()
            e = EX[p]
            qw = pr[e] / pr[e].sum()
            grad = (qw.to(Wu.dtype) @ Wu[e]).float() - (pr.to(Wu.dtype) @ Wu).float()
            pre = cap['pre'][row, last].float()
            u = gw * grad / torch.sqrt((pre ** 2).mean() + eps_of(lm.norm))
        return ms.cpu().numpy(), u

    vec = lambda s: crit['inject_k'] * SG[s] * V[s]
    texts = {g: [i for i in disc if C[i]['style'] == g[0]] for g in GROUPS}
    Amask = {g: np.array([MF3.in_A(C[i]['task']) for i in texts[g]]) for g in GROUPS}
    DEL = []
    dsum = {l: torch.zeros(d, device='cuda') for l in PL}
    dsamp = {l: [] for l in PL}
    nd = 0
    for g in GROUPS:
        p = STY.index(g[1])
        for i, a_ in zip(texts[g], Amask[g]):
            n = max(2, round(0.6 * len(ids[i])))
            st['pos'] = n
            st['inject'] = torch.as_tensor(np.stack([np.zeros(d), vec(p), vec(CODE), vec(p) + vec(CODE)]), dtype=torch.float32, device='cuda')
            st['capture'], st['capd'] = True, bool(a_)
            try:
                _, u = run([ids[i][:n]] * 4, need_u=(3, p))
            finally:
                st['capture'], st['capd'] = False, False
            row = np.zeros(len(layer_of), np.float32)
            for l in PL:
                a = cap[l][:, n - 1].float()
                q = (u @ blocks[l].mlp.down_proj.weight.float())
                row[off[l]:off[l] + I_] = ((a[3] - a[1]) * q).cpu().numpy()
                if a_:                                              # 冲突差值只用 A 任务（选单元的划分）
                    dl = cap[('d', l)][3, 1:n] - cap[('d', l)][1, 1:n]
                    dsum[l] += dl.sum(0)
                    dsamp[l].append(dl.cpu().numpy())
            if a_:
                nd += n - 1
            DEL.append(row)
    st['inject'] = None
    DEL = np.array(DEL)
    inA = np.concatenate([Amask[g] for g in GROUPS])
    top = F.select_units(DEL, inA, crit['top_k'])
    del DEL
    rsets = F.random_units_matched(top, layer_of, crit['n_random_sets'], crit['seed'])
    loc = lambda i: (int(layer_of[i]), int(i - off[int(layer_of[i])]))
    by_layer = lambda units: {l: [loc(i)[1] for i in units if loc(i)[0] == l] for l in PL}
    ddir, sub = {}, {}
    for l in PL:
        ddir[l] = W.unit((dsum[l] / nd).cpu().numpy())
        D_ = np.concatenate(dsamp[l])
        sub[l] = np.linalg.svd(D_, full_matrices=False)[2][:crit['sub_rank']]       # 不中心化：保留平均方向
        dsamp[l] = None
    del dsamp
    gc.collect()
    cos_dv = {str(l): float(ddir[l] @ vdir[l]) for l in PL}
    print(f'@@S1 top {[loc(i) for i in top[:10]]} {round(time.time() - t0)}s', flush=True)

    def measure(variants=None):
        """B 任务：返回 {组: [n, 2 + 变体数]}（只散文、同时、各修补变体）。"""
        out = {}
        for g in GROUPS:
            p = STY.index(g[1])
            M = []
            for i, a_ in zip(texts[g], Amask[g]):
                if a_:
                    continue
                n = max(2, round(0.6 * len(ids[i])))
                seq = ids[i][:n]
                st['pos'] = n
                st['inject'] = torch.as_tensor(np.stack([vec(p), vec(p) + vec(CODE)]), dtype=torch.float32, device='cuda')
                st['capture'] = variants is not None
                try:
                    ms, _ = run([seq] * 2)
                finally:
                    st['capture'] = False
                row = [ms[0, p], ms[1, p]]
                if variants:
                    for l in PL:
                        store[l] = cap[l][0]
                    st['inject'] = torch.as_tensor(np.stack([vec(p) + vec(CODE)] * len(variants)), dtype=torch.float32, device='cuda')
                    pt = {}
                    for r_, units in enumerate(variants):
                        for l, us in by_layer(units).items():
                            if us:
                                pt.setdefault(l, {})[r_] = torch.as_tensor(us, device='cuda')
                    st['patch'] = pt
                    try:
                        ms2, _ = run([seq] * len(variants))
                    finally:
                        st['patch'] = None
                    row += list(ms2[:, p])
                M.append(row)
            out['%s:%s' % g] = np.array(M)
        st['inject'] = None
        return out

    MP = measure([top] + rsets)
    print(f'@@S1 patch {round(time.time() - t0)}s', flush=True)
    # ---------- S2：读权重与编辑 ----------
    TLu = sorted({loc(i)[0] for i in top})
    gam = {l: gw_of(blocks[l].post_attention_layernorm).cpu().numpy() for l in PL}

    def rows_of(l, us):
        P = blocks[l].mlp.gate_up_proj.weight
        return P, list(us) + [I_ + u for u in us]

    def read_mean(units, dirs):
        vals = []
        for i in units:
            l, u_ = loc(i)
            P = blocks[l].mlp.gate_up_proj.weight
            Wg = P[u_].float().cpu().numpy()[None] * gam[l]
            Wup = P[I_ + u_].float().cpu().numpy()[None] * gam[l]
            vals.append(W.read_scores(Wg, Wup, dirs[l])[0])
        return float(np.mean(vals))

    rs_read = F.random_units_matched(top, layer_of, crit['n_random_sets_read'], crit['seed'])
    read_v = W.ratio_vs_random(read_mean(top, vdir), [read_mean(s, vdir) for s in rs_read])
    read_d = W.ratio_vs_random(read_mean(top, ddir), [read_mean(s, ddir) for s in rs_read])
    print('@@S2 read v', {k_: round(v, 4) for k_, v in read_v.items()}, 'd', {k_: round(v, 4) for k_, v in read_d.items()},
          'cos(d,v)', cos_dv, f'{round(time.time() - t0)}s', flush=True)

    def edit(units, dirs):
        saved = []
        for l, us in by_layer(units).items():
            if not us:
                continue
            P, rows = rows_of(l, us)
            saved.append((P, rows, P.data[rows].clone()))
            new = W.remove_subspace(P.data[rows].float().cpu().numpy(), gam[l], np.atleast_2d(dirs[l]))
            P.data[rows] = torch.as_tensor(new, dtype=P.dtype, device=P.device)
        return saved

    def restore(saved):
        for P, rows, old in reversed(saved):
            P.data[rows] = old

    def nll():
        out = {0: [], 1: []}
        for j in range(0, len(conf), 4):
            ch = conf[j:j + 4]
            i_, att = batch([ids[i] for i in ch])
            hs = lm(input_ids=i_, attention_mask=att).last_hidden_state
            for b, i in enumerate(ch):
                n = len(ids[i])
                lp = torch.log_softmax(model.lm_head(hs[b, :n - 1]).float(), -1).gather(-1, i_[b, 1:n, None])[..., 0]
                out[int(C[i]['style'] == 'code')].append(float(-lp.mean()))
        return {'0': float(np.mean(out[0])), '1': float(np.mean(out[1])), 'all': float(np.mean(out[0] + out[1]))}

    rng = np.random.default_rng(crit['seed'])
    rdir = {l: W.unit(rng.normal(0, 1, d)) for l in PL}
    ME = {'orig': measure()}
    NL = {'orig': nll()}
    plan = [('edit_d', top, ddir), ('edit_v', top, vdir), ('edit_sub4', top, sub), ('edit_randdir', top, rdir)] + \
           [(f'edit_randunits{j}', s_, ddir) for j, s_ in enumerate(rsets)]
    for name, units, dirs in plan:
        sv = edit(units, dirs)
        try:
            ME[name] = measure()
            if name in ('edit_d', 'edit_v', 'edit_sub4'):
                NL[name] = nll()
        finally:
            restore(sv)
        print(f'@@{name} {round(time.time() - t0)}s', flush=True)
    for h_ in hooks:
        h_.remove()
    save({'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'precision': 'bf16',
          'placement': place, 'n_blocks': L, 'n_heads': nh, 'n_kv': nkv, 'head_dim': hd, 'switch_layer': ls, 'unit_pool': PL,
          'n_shared_positions': n_shared, 'top_units': [list(loc(i)) for i in top],
          'random_sets': [[list(loc(i)) for i in s_] for s_ in rsets],
          'patch': {g: v.tolist() for g, v in MP.items()}, 'read_v': read_v, 'read_d': read_d, 'cos_d_v': cos_dv,
          'edits': {k_: {g: v.tolist() for g, v in d_.items()} for k_, d_ in ME.items()}, 'nll': NL})
    # ---------- S3：上游头 ----------
    uidx = {l: torch.as_tensor(by_layer(top)[l], device='cuda') for l in TLu}
    gcap, sgn = {}, {}
    hk = [blocks[l].mlp.gate_up_proj.register_forward_hook((lambda l: lambda m, a, o: gcap.__setitem__(l, o[..., uidx[l]].float()))(l)) for l in TLu]
    gs = {l: [[], []] for l in TLu}
    for j in range(0, len(disc), 8):
        ch = disc[j:j + 8]
        i_, att = batch([ids[i] for i in ch])
        lm(input_ids=i_, attention_mask=att)
        for b, i in enumerate(ch):
            m = torch.as_tensor(shm[i], device='cuda')
            for l in TLu:
                gs[l][int(C[i]['style'] == 'code')].append(gcap[l][b, :len(ids[i])][m].cpu().numpy())
    for l in TLu:
        sgn[l] = np.sign(np.concatenate(gs[l][1]).mean(0) - np.concatenate(gs[l][0]).mean(0))
    for h_ in hk:
        h_.remove()
    Sm = torch.stack([(torch.as_tensor(sgn[l], device='cuda', dtype=torch.float32)[:, None] *
                       blocks[l].mlp.gate_up_proj.weight[by_layer(top)[l]].float()).sum(0) * gw_of(blocks[l].post_attention_layernorm) for l in TLu])
    cc = {}

    def oproj_pre(lam):
        def f(m, a):
            zz = a[0].float().reshape(a[0].shape[0], a[0].shape[1], nh, hd)
            q = (Sm @ m.weight.float()).reshape(len(TLu), nh, hd)
            q = q * torch.as_tensor([1.0 if l >= lam else 0.0 for l in TLu], device='cuda')[:, None, None]
            cc[('a', lam)] = torch.einsum('bthd,khd->btkh', zz, q)
            return None
        return f

    def rms_pre(key_):
        def f(m, a):
            cc[key_] = torch.sqrt((a[0].float() ** 2).mean(-1) + m.variance_epsilon)
            return None
        return f

    LH = max(TLu)
    hk = [blocks[lam].self_attn.o_proj.register_forward_pre_hook(oproj_pre(lam)) for lam in range(LH + 1)]
    hk += [blocks[l].post_attention_layernorm.register_forward_pre_hook(rms_pre(('r', l))) for l in TLu]
    acc = [np.zeros((LH + 1, nh)), np.zeros((LH + 1, nh))]
    nn_ = [0, 0]
    for j in range(0, len(disc), 8):
        ch = disc[j:j + 8]
        i_, att = batch([ids[i] for i in ch])
        lm(input_ids=i_, attention_mask=att)
        rinv = torch.stack([1.0 / cc[('r', l)] for l in TLu], -1)
        for b, i in enumerate(ch):
            n = len(ids[i])
            m = torch.as_tensor(shm[i], device='cuda')
            c = int(C[i]['style'] == 'code')
            ri = rinv[b, :n][m]
            for lam in range(LH + 1):
                acc[c][lam] += (cc[('a', lam)][b, :n][m] * ri[:, :, None]).sum((0, 1)).cpu().numpy()
            nn_[c] += int(m.sum().item())
        cc.clear()
    for h_ in hk:
        h_.remove()
    Td = acc[1] / nn_[1] - acc[0] / nn_[0]
    sel = sorted([(lam, h) for lam in range(LH + 1) for h in range(nh)], key=lambda x: -Td[x])[:crit['n_heads']]
    rheads = W.matched_heads(sel, nh, crit['n_random_sets'], crit['seed'])
    print('@@S3 heads', sel, f'{round(time.time() - t0)}s', flush=True)
    sl = sorted({l for l, _ in sel})
    miss = [l for l in sl if l not in vdir]
    if miss:
        vd2, _, _, _ = directions(miss)
        vdir.update(vd2)
    # 同模式 QK
    ctl = {'collect': {}, 'store': {}, 'q': {}, 'k': {}, 'kmask': None, 'cap': False, 'xn': {}}

    def ws_attn(module, query, key_, value, attention_mask, scaling, dropout=0.0, **kw):
        k_ = MG.repeat_kv(key_, module.num_key_value_groups)
        v_ = MG.repeat_kv(value, module.num_key_value_groups)
        w = torch.matmul(query, k_.transpose(2, 3)) * scaling
        if attention_mask is not None:
            w = w + attention_mask
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        lam = module.layer_idx
        if ctl['collect'].get(lam):
            ctl['store'][lam] = w[:, ctl['collect'][lam]].float()
        return torch.matmul(w, v_).transpose(1, 2).contiguous(), None

    MG.eager_attention_forward = ws_attn
    model.set_attn_implementation('eager')

    def q_hook(l):
        def f(m, a, o):
            for h, dv in ctl['q'].get(l, []):
                o[..., h * hd:(h + 1) * hd] += dv
            return o
        return f

    def k_hook(l):
        def f(m, a, o):
            for g_, dv in ctl['k'].get(l, []):
                o[..., g_ * hd:(g_ + 1) * hd] += dv * ctl['kmask'][..., None]
            return o
        return f

    def ln_hook(l):
        def f(m, a, o):
            if ctl['cap']:
                ctl['xn'][l] = o.float()
        return f

    hk = [blocks[l].self_attn.q_proj.register_forward_hook(q_hook(l)) for l in sl]
    hk += [blocks[l].self_attn.k_proj.register_forward_hook(k_hook(l)) for l in sl]
    hk += [blocks[l].input_layernorm.register_forward_hook(ln_hook(l)) for l in sl]
    ctl['collect'] = {l: sorted({h for l2, h in sel if l2 == l}) for l in sl}
    col = {l: {h: i for i, h in enumerate(v)} for l, v in ctl['collect'].items()}
    rd0 = W.unit(np.random.default_rng(crit['seed'] + 1).normal(0, 1, d))
    udir, WQ, WK = {}, {}, {}
    for l in sl:
        at = blocks[l].self_attn
        WQ[l], WK[l] = cu(at.q_proj.weight).float(), cu(at.k_proj.weight).float()
        for nm, v in (('code', vdir[l]), ('rand', rd0)):
            u_ = gw_of(blocks[l].input_layernorm) * torch.as_tensor(v, device='cuda', dtype=torch.float32)
            udir[(l, nm)] = u_ / u_.norm()
    items = []
    for it in WS3.mixed_disc():
        if it['kind'] != 'md':
            continue
        text, segs = '', []
        for s_, seg in it['pieces']:
            segs.append((len(text), len(text) + len(s_), seg))
            text += s_
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        seg_of = np.array([next(sg for a, b, sg in segs if a <= s0 < b) if s0 < len(text) else segs[-1][2] for s0, _ in enc['offset_mapping']])
        items.append((enc['input_ids'], seg_of))
    proj = {k_: [] for k_ in udir}
    ctl['cap'] = True
    for seq, _ in items:
        i_, att = batch([seq])
        lm(input_ids=i_, attention_mask=att)
        for (l, nm), u_ in udir.items():
            proj[(l, nm)].append((ctl['xn'][l][0] @ u_).cpu().numpy())
    ctl['cap'] = False
    sig = {k_: float(np.concatenate(v).std()) for k_, v in proj.items()}

    def qk_run(qshift=None, kshift=None):
        out = {'closing': [], 'code': []}
        for seq, seg_of in items:
            codek = seg_of == 'code'
            ctl['q'], ctl['k'] = {}, {}
            for l, h in sel:
                if qshift:
                    nm, kk = qshift
                    dv = kk * sig[(l, nm)] * (WQ[l] @ udir[(l, nm)])[h * hd:(h + 1) * hd]
                    ctl['q'].setdefault(l, []).append((h, dv.to(torch.bfloat16)))
                if kshift:
                    nm, kk = kshift
                    g_ = h // grp
                    if all(g_ != g2 for g2, _ in ctl['k'].get(l, [])):
                        dv = kk * sig[(l, nm)] * (WK[l] @ udir[(l, nm)])[g_ * hd:(g_ + 1) * hd]
                        ctl['k'].setdefault(l, []).append((g_, dv.to(torch.bfloat16)))
            ctl['kmask'] = torch.as_tensor(codek[None], device='cuda', dtype=torch.bfloat16)
            i_, att = batch([seq])
            lm(input_ids=i_, attention_mask=att)
            A = {l: ctl['store'][l][0].cpu().numpy() for l in sl}
            for t in range(8, len(seq)):
                if seg_of[t] in out:
                    out[seg_of[t]].append(np.nanmean([W.mass_ratio(A[l][col[l][h], t], t, codek) for l, h in sel]))
        ctl['q'], ctl['k'] = {}, {}
        return {k_: float(np.nanmean(v)) for k_, v in out.items()}

    QK = {'orig': qk_run(), 'q_code_4': qk_run(qshift=('code', 4)), 'q_rand_4': qk_run(qshift=('rand', 4)),
          'k_code_-4': qk_run(kshift=('code', -4)), 'k_rand_-4': qk_run(kshift=('rand', -4))}
    for h_ in hk:
        h_.remove()
    print('@@S3 QK', QK, f'{round(time.time() - t0)}s', flush=True)
    save({'heads': [list(x) for x in sel], 'random_heads': [[list(x) for x in s_] for s_ in rheads],
          'T_disc_selected': [float(Td[x]) for x in sel], 'sigma': {f'{l}.{nm}': v for (l, nm), v in sig.items()}, 'qk': QK,
          'n_md_items': len(items), 'seconds': round(time.time() - t0, 1)})
    print('done', flush=True)


def do_analyze():
    from provenance import write_result
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    rep = json.loads(report_path().read_text(encoding='utf-8'))
    P = {g: np.array(v) for g, v in rep['patch'].items()}
    rest = lambda col_: float(np.mean([(P[g][:, col_] - P[g][:, 1]).mean() / (P[g][:, 0] - P[g][:, 1]).mean() for g in P]))
    r_top = rest(2)
    r_rand = [rest(3 + j) for j in range(crit['n_random_sets'])]
    h1 = W.over_random(r_top, r_rand, crit['H1_over_random'], crit['H1_patch_restore_min'])
    h1['per_group'] = {g: float((P[g][:, 2] - P[g][:, 1]).mean() / (P[g][:, 0] - P[g][:, 1]).mean()) for g in P}
    h2 = dict(rep['read_v'])
    h2['pass'] = bool(h2['ratio'] >= crit['H2_read_ratio_min'] and h2['p'] <= crit['H2_p_max'])
    h3 = dict(rep['read_d'])
    h3['pass'] = bool(h3['ratio'] >= crit['H3_read_ratio_min'] and h3['p'] <= crit['H3_p_max'])
    E = {k_: {g: np.array(v) for g, v in d_.items()} for k_, d_ in rep['edits'].items()}
    er = lambda nm: float(np.mean([(E[nm][g][:, 1] - E['orig'][g][:, 1]).mean() / (E['orig'][g][:, 0] - E['orig'][g][:, 1]).mean() for g in E['orig']]))
    per = lambda nm: {g: float((E[nm][g][:, 1] - E['orig'][g][:, 1]).mean() / (E['orig'][g][:, 0] - E['orig'][g][:, 1]).mean()) for g in E['orig']}
    e = er('edit_d')
    ctrl = [er('edit_randdir')] + [er(f'edit_randunits{j}') for j in range(crit['n_random_sets'])]
    h4 = {'edit_restore': e, 'per_group': per('edit_d'), 'patch_restore': r_top, 'frac_of_patch': e / r_top if r_top else float('nan'),
          'controls': ctrl, 'pass': bool(e >= crit['H4_restore_frac_min'] * r_top and max(ctrl) <= crit['H4_control_max_frac'] * e)}
    nl = rep['nll']
    rise = lambda nm: {k_: (nl[nm][k_] - nl['orig'][k_]) / nl['orig'][k_] for k_ in ('all', '1', '0')}
    r5 = rise('edit_d')
    h5 = {'rel_rise_all': r5['all'], 'rel_rise_code': r5['1'], 'rel_rise_prose': r5['0'],
          'pass': bool(r5['all'] <= crit['H5_nll_rise_max'] and r5['1'] <= crit['H5_nll_rise_max'])}
    Q = rep['qk']
    lq, lqr = np.log(Q['q_code_4']['closing'] / Q['orig']['closing']), np.log(Q['q_rand_4']['closing'] / Q['orig']['closing'])
    h6 = {'ratio': float(np.exp(lq)), 'random_ratio': float(np.exp(lqr)),
          'pass': bool(np.exp(lq) >= crit['H6_query_ratio_min'] and abs(lq) >= crit['over_random_log'] * abs(lqr))}
    lk, lkr = np.log(Q['k_code_-4']['code'] / Q['orig']['code']), np.log(Q['k_rand_-4']['code'] / Q['orig']['code'])
    h7 = {'ratio': float(np.exp(lk)), 'random_ratio': float(np.exp(lkr)),
          'pass': bool(np.exp(lk) <= crit['H7_key_ratio_max'] and abs(lk) >= crit['over_random_log'] * abs(lkr))}
    lc = {}
    for l, _ in rep['top_units']:
        lc[str(l)] = lc.get(str(l), 0) + 1
    desc = {'cos_d_v': rep['cos_d_v'], 'edit_v_restore': er('edit_v'), 'edit_v_per_group': per('edit_v'),
            'edit_v_nll_rise': rise('edit_v'), 'edit_sub4_restore': er('edit_sub4'), 'edit_sub4_per_group': per('edit_sub4'),
            'edit_sub4_nll_rise': rise('edit_sub4'), 'unit_layer_counts': lc}
    verdict = {f'H{i}': h['pass'] for i, h in enumerate([h1, h2, h3, h4, h5, h6, h7], 1)}
    rep = save({'criteria': crit, 'H1': h1, 'H2': h2, 'H3': h3, 'H4': h4, 'H5': h5, 'H6': h6, 'H7': h7, 'describe': desc, 'verdict': verdict})
    p = write_result('gl_v1_glm4-9b-chat-hf', rep, model_path=MODEL, seed=crit['seed'], tool='scripts/glm_chain_v1.py')
    print(json.dumps(verdict), '\nH1', h1, '\nH2', h2, '\nH3', h3, '\nH4', h4, '\nH5', h5, '\nH6', h6, '\nH7', h7,
          '\ndesc', desc, '\nheads', rep['heads'], '\nqk', Q, '\nprovenance →', p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run', 'analyze'])
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': do_run, 'analyze': do_analyze}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
