# -*- coding: utf-8 -*-
"""GC-v1：在 Gemma-3-4B-it 上复现「代码压过散文」的门控链（Phase 278 / 284 / 285 / 287 的方法）。

Gemma 与 Qwen 的结构差异：注意力与 MLP 前后各有一个 RMSNorm（权重形式 1 + w），MLP 为 GELU 门控，8 个头（4 个 KV 组，头维 256），嵌入乘 √d，需要 BOS。
材料同 MF-v1（文本前加 BOS）；状态以 float32 处理（Gemma 深层激活超出 float16 范围）。
  S0 方向：MF-v1 发现集共用 token 位置上各层 hs[l] 的 token 配平风格方向；写入层 ls = round(L/2)。
  S1 单元（MF-v4 方法）：在 ls 写入「散文」或「散文 + 代码」方向（6σ，文本前 60%），候选为块 ≥ round(2L/3) 的全部 MLP 中间单元；
      直接作用差 Δ_i = (A_同时 − A_只散文)_i ·（W_down 列经 post_feedforward_layernorm 线性化后对 log M_p 的梯度），A 任务上取最负 32 个。
      H1：B 任务上只把这 32 个单元的中间激活（全部位置）换回「只散文」的值，散文质量恢复（三组平均）≥ 0.2，且 ≥ 3 × 同层随机 32 单元（5 组）。
  S2 读权重与编辑（WS-v1 方法）：读出向量 a_i = (1 + w_pre_ffw) ⊙ W_gate[i]（up 同理）。
      H2：对 v_l 的读分数 ≥ 同层随机集合（200 组）的 3 倍，p ≤ 0.01。
      H3：移除 v_l 的封闭式编辑恢复 ≥ 0.5 × H1 的修补恢复；随机方向、5 组随机单元编辑的恢复各 ≤ 编辑的 1/3。
      H4：编辑后确认集自然文本 NLL 相对上升 ≤ 1%（全部与代码文本）。
  S3 上游头与同模式 QK（WS-v2 选头 + WS-v4 干预）：门控得分 G = Σ s_i g_i（s_i 由发现集自然文本定号），
      头的直接贡献经 post_attention_layernorm 线性化；发现集上推向代码最大的 8 个头。在 WS-v3 的发现集混合文本上：
      H5：只在这 8 个头的查询上加 +4σ 代码分量（q_norm 之前），md 结语位置对代码 key 的质量比 ≥ 1.5 × 原始，且 |log 效应| ≥ 3 × 随机方向。
      H6：代码段 key 移动 −4σ 后，md 代码位置对代码 key 的质量比 ≤ 0.8 × 原始，且 |log 效应| ≥ 3 × 随机方向。
被测模型：Gemma-3-4B-it（bf16，不使用量化）。与 Qwen 的方向、单元、头不作跨模型对应，只比较规律是否成立。
用法：python scripts/gemma_chain_v1.py prereg | run | analyze
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

PREREG = ROOT / 'research/weight_space/GC-v1.prereg.json'
MODEL = 'models/hf/gemma-3-4b-it'
STY = F.STYLES
CODE = STY.index('code')
GROUPS = [('zh', 'en'), ('en', 'zh'), ('paper', 'zh')]
NATIVE = Path('D:/ai2050_native/_weight_space')


def prereg_doc():
    return {'id': 'GC-v1', 'schema': 'gemma_chain_prereg.v1', 'created': '2026-11-02',
            'basis': 'Qwen3-4B 上的门控链：Phase 278（单元）、284（读权重与编辑）、285（上游头）、287（同模式 QK）；本版在另一架构上复现',
            'models': {'gemma-3-4b-it': 'bf16'}, 'precision_rule': '不使用任何量化',
            'materials': 'MF-v1 语料（发现 / 确认），MF-v4 的三组写入范式与 A / B 任务切分，WS-v3 的发现集混合文本',
            'differences_from_qwen': '候选单元池为后三分之一的块（Qwen 版由 MF-v3 整层修补选层）；H1 不与整层修补比较；H5 不要求严格单调',
            'power_note': '每组 B 任务约 40 段；随机集合 5 / 200 组；混合文本 md 160 段；连续量阈值',
            'doc': __doc__, 'criteria': W.CRITERIA_GC}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（GC-v2）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def report_path():
    d = ROOT / 'research/weight_space/GC-v1-gemma-3-4b-it'
    d.mkdir(parents=True, exist_ok=True)
    return d / 'report.json'


def save(upd):
    p = report_path()
    rep = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}
    rep.update(upd)
    p.write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float), encoding='utf-8')
    return rep


def do_run():
    import torch
    import transformers.models.gemma3.modeling_gemma3 as MG
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    snap = FU1.snapshot(MODEL)
    tok, model = FU1.load(MODEL)
    lm = model.model.language_model
    blocks = lm.layers
    tcfg = model.config.text_config
    L, nh, nkv, hd, d = tcfg.num_hidden_layers, tcfg.num_attention_heads, tcfg.num_key_value_heads, tcfg.head_dim, tcfg.hidden_size
    grp = nh // nkv
    ls = round(0.5 * L)
    bos = tok.bos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    mf1crit = json.loads(MF1.PREREG.read_text(encoding='utf-8'))['criteria']
    C, ids0 = MF1.tokenize(tok)
    ids = [[bos] + x for x in ids0]
    excl, shared = MF1.vocab_and_shared(C, ids0, mf1crit)
    shm = [np.array([False] + [t in shared[C[i]['split']] for t in ids0[i]]) for i in range(len(C))]
    EX = [torch.as_tensor(excl[s], device='cuda') for s in STY]
    t0 = time.time()

    def batch(seqs):
        n = max(map(len, seqs))
        i_ = torch.full((len(seqs), n), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), n), dtype=torch.long)
        for b, s in enumerate(seqs):
            i_[b, :len(s)] = torch.as_tensor(s)
            att[b, :len(s)] = 1
        return i_.cuda(), att.cuda()

    g1 = lambda mod: 1.0 + mod.weight.float()                      # Gemma RMSNorm 的有效缩放
    # ---------- S0：方向 ----------
    disc = [i for i, c in enumerate(C) if c['split'] == 'disc']
    conf = [i for i, c in enumerate(C) if c['split'] == 'conf']
    Xs = [[] for _ in range(L + 1)]
    ys, ts = [], []
    for j in range(0, len(disc), 8):
        ch = disc[j:j + 8]
        i_, att = batch([ids[i] for i in ch])
        o = lm(input_ids=i_, attention_mask=att, output_hidden_states=True)
        for b, i in enumerate(ch):
            m = torch.as_tensor(shm[i], device='cuda')
            n = len(ids[i])
            for l in range(L + 1):
                Xs[l].append(o.hidden_states[l][b, :n][m].float().cpu().numpy())
            ys += [STY.index(C[i]['style'])] * int(shm[i].sum())
            ts += [t for t, f in zip(ids[i], shm[i]) if f]
        del o
    ys, ts = np.array(ys), np.array(ts)
    vdir, V, SG = {}, {}, {}
    for l in range(L):
        X = np.concatenate(Xs[l])
        vdir[l] = W.unit(F.token_matched_direction(X, ys, ts, CODE))
        if l == ls:
            for s in range(4):
                v = F.token_matched_direction(X, ys, ts, s)
                V[s] = v / np.linalg.norm(v)
                SG[s] = float((X @ V[s]).std())
        Xs[l] = None
    del Xs
    gc.collect()
    print(f'@@S0 directions {round(time.time() - t0)}s', flush=True)
    # ---------- S1：单元 ----------
    PL = list(range(round(crit['unit_pool_from_frac'] * L), L))
    I_ = blocks[PL[0]].mlp.down_proj.in_features
    layer_of = np.concatenate([[l] * I_ for l in PL])
    off = {l: k * I_ for k, l in enumerate(PL)}
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
            if st['patch'] is not None and l in st['patch']:
                x = x.clone()
                for row, units in st['patch'][l].items():
                    x[row][:, units] = store[l][:, units].to(x.dtype)
                return (x,) + tuple(a[1:])
            return None
        return f

    def pf_hook(l):
        def f(m, a):
            if st['capture']:
                cap[('pf', l)] = torch.sqrt((a[0].float() ** 2).mean(-1) + m.eps)
            return None
        return f

    hooks = [blocks[ls].register_forward_pre_hook(inj, with_kwargs=True)]
    hooks += [blocks[l].mlp.down_proj.register_forward_pre_hook(dp_hook(l)) for l in PL]
    hooks += [blocks[l].post_feedforward_layernorm.register_forward_pre_hook(pf_hook(l)) for l in PL]
    hooks.append(lm.norm.register_forward_hook(lambda m, a, o: cap.__setitem__('pre', a[0].detach())))
    Wu = model.lm_head.weight
    gw = g1(lm.norm)

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
            u = gw * grad / torch.sqrt((pre ** 2).mean() + lm.norm.eps)
        return ms.cpu().numpy(), u

    vec = lambda s: crit['inject_k'] * SG[s] * V[s]
    texts = {g: [i for i in disc if C[i]['style'] == g[0]] for g in GROUPS}
    Amask = {g: np.array([MF3.in_A(C[i]['task']) for i in texts[g]]) for g in GROUPS}
    Wd32 = {l: blocks[l].mlp.down_proj.weight.float() for l in PL}
    DEL = []
    for g in GROUPS:
        p = STY.index(g[1])
        for i in texts[g]:
            n = max(2, round(0.6 * len(ids[i])))
            st['pos'] = n
            st['inject'] = torch.as_tensor(np.stack([np.zeros(d), vec(p), vec(CODE), vec(p) + vec(CODE)]), dtype=torch.float32, device='cuda')
            st['capture'] = True
            try:
                _, u = run([ids[i][:n]] * 4, need_u=(3, p))
            finally:
                st['capture'] = False
            row = np.zeros(len(layer_of), np.float32)
            for l in PL:
                a = cap[l][:, n - 1].float()
                q = Wd32[l].T @ (g1(blocks[l].post_feedforward_layernorm) * u) / cap[('pf', l)][3, n - 1]
                row[off[l]:off[l] + I_] = ((a[3] - a[1]) * q).cpu().numpy()
            DEL.append(row)
    st['inject'] = None
    del Wd32
    DEL = np.array(DEL)
    inA = np.concatenate([Amask[g] for g in GROUPS])
    top = F.select_units(DEL, inA, crit['top_k'])
    rsets = F.random_units_matched(top, layer_of, crit['n_random_sets'], crit['seed'])
    loc = lambda i: (int(layer_of[i]), int(i - off[int(layer_of[i])]))
    by_layer = lambda units: {l: [loc(i)[1] for i in units if loc(i)[0] == l] for l in PL}
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
    gam = {l: g1(blocks[l].pre_feedforward_layernorm).cpu().numpy() for l in PL}

    def read_mean(units):
        vals = []
        for i in units:
            l, u_ = loc(i)
            Wg = blocks[l].mlp.gate_proj.weight[u_].float().cpu().numpy()[None] * gam[l]
            Wup = blocks[l].mlp.up_proj.weight[u_].float().cpu().numpy()[None] * gam[l]
            vals.append(W.read_scores(Wg, Wup, vdir[l])[0])
        return float(np.mean(vals))

    rs_read = F.random_units_matched(top, layer_of, crit['n_random_sets_read'], crit['seed'])
    h2raw = W.ratio_vs_random(read_mean(top), [read_mean(s) for s in rs_read])
    print('@@S2 read', {k_: round(v, 4) for k_, v in h2raw.items()}, f'{round(time.time() - t0)}s', flush=True)

    def edit(units, dirs):
        saved = []
        for l, us in by_layer(units).items():
            if not us:
                continue
            for P in (blocks[l].mlp.gate_proj.weight, blocks[l].mlp.up_proj.weight):
                saved.append((P, us, P.data[us].clone()))
                P.data[us] = torch.as_tensor(W.remove_direction(P.data[us].float().cpu().numpy(), gam[l], dirs[l]), dtype=P.dtype, device=P.device)
        return saved

    def restore(saved):
        for P, us, old in reversed(saved):
            P.data[us] = old

    def nll():
        out = {0: [], 1: []}
        for j in range(0, len(conf), 8):
            ch = conf[j:j + 8]
            i_, att = batch([ids[i] for i in ch])
            lg = model.lm_head(lm(input_ids=i_, attention_mask=att).last_hidden_state).float()
            lp = torch.log_softmax(lg[:, :-1], -1).gather(-1, i_[:, 1:, None])[..., 0]
            for b, i in enumerate(ch):
                out[int(C[i]['style'] == 'code')].append(float(-lp[b, :len(ids[i]) - 1].mean()))
        return {'0': float(np.mean(out[0])), '1': float(np.mean(out[1])), 'all': float(np.mean(out[0] + out[1]))}

    rng = np.random.default_rng(crit['seed'])
    rdir = {l: W.unit(rng.normal(0, 1, d)) for l in PL}
    ME = {'orig': measure()}
    NL = {'orig': nll()}
    for name, units, dirs in [('edit', top, vdir), ('edit_randdir', top, rdir)] + [(f'edit_randunits{j}', s_, vdir) for j, s_ in enumerate(rsets)]:
        sv = edit(units, dirs)
        try:
            ME[name] = measure()
            if name == 'edit':
                NL[name] = nll()
        finally:
            restore(sv)
    for h_ in hooks:
        h_.remove()
    print(f'@@S2 edits {round(time.time() - t0)}s', flush=True)
    # ---------- S3：上游头 ----------
    uidx = {l: torch.as_tensor(by_layer(top)[l], device='cuda') for l in TLu}
    gcap, sgn = {}, {}
    hk = [blocks[l].mlp.gate_proj.register_forward_hook((lambda l: lambda m, a, o: gcap.__setitem__(l, o[..., uidx[l]].float()))(l)) for l in TLu]
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
                       blocks[l].mlp.gate_proj.weight[by_layer(top)[l]].float()).sum(0) * g1(blocks[l].pre_feedforward_layernorm) for l in TLu])
    cc = {}

    def oproj_pre(lam):
        def f(m, a):
            zz = a[0].float().reshape(a[0].shape[0], a[0].shape[1], nh, hd)
            q = ((Sm * g1(blocks[lam].post_attention_layernorm)) @ m.weight.float()).reshape(len(TLu), nh, hd)
            q = q * torch.as_tensor([1.0 if l >= lam else 0.0 for l in TLu], device='cuda')[:, None, None]
            cc[('a', lam)] = torch.einsum('bthd,khd->btkh', zz, q)
            return None
        return f

    def rms_pre(key_):
        def f(m, a):
            cc[key_] = torch.sqrt((a[0].float() ** 2).mean(-1) + m.eps)
            return None
        return f

    LH = max(TLu)
    hk = [blocks[lam].self_attn.o_proj.register_forward_pre_hook(oproj_pre(lam)) for lam in range(LH + 1)]
    hk += [blocks[lam].post_attention_layernorm.register_forward_pre_hook(rms_pre(('pa', lam))) for lam in range(LH + 1)]
    hk += [blocks[l].pre_feedforward_layernorm.register_forward_pre_hook(rms_pre(('r', l))) for l in TLu]
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
                pa = (1.0 / cc[('pa', lam)][b, :n][m])[:, None, None]
                acc[c][lam] += (cc[('a', lam)][b, :n][m] * ri[:, :, None] * pa).sum((0, 1)).cpu().numpy()
            nn_[c] += int(m.sum().item())
    for h_ in hk:
        h_.remove()
    Td = acc[1] / nn_[1] - acc[0] / nn_[0]
    sel = sorted([(lam, h) for lam in range(LH + 1) for h in range(nh)], key=lambda x: -Td[x])[:crit['n_heads']]
    rheads = W.matched_heads(sel, nh, crit['n_random_sets'], crit['seed'])
    print('@@S3 heads', sel, f'{round(time.time() - t0)}s', flush=True)
    # 同模式 QK
    ctl = {'collect': {}, 'store': {}, 'q': {}, 'k': {}, 'kmask': None, 'cap': False, 'xn': {}}

    def ws_attn(module, query, key_, value, attention_mask, dropout=0.0, scaling=None, softcap=None, **kw):
        k_ = MG.repeat_kv(key_, module.num_key_value_groups)
        v_ = MG.repeat_kv(value, module.num_key_value_groups)
        w = torch.matmul(query, k_.transpose(2, 3)) * (scaling if scaling is not None else module.head_dim ** -0.5)
        if softcap is not None:
            w = torch.tanh(w / softcap) * softcap
        if attention_mask is not None:
            w = w + attention_mask
        w = torch.nn.functional.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        lam = module.layer_idx
        if ctl['collect'].get(lam):
            ctl['store'][lam] = w[:, ctl['collect'][lam]].float()
        return torch.matmul(w, v_).transpose(1, 2).contiguous(), None

    MG.eager_attention_forward = ws_attn
    model.set_attn_implementation('eager')
    sl = sorted({l for l, _ in sel})

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
    udir = {}
    for l in sl:
        for nm, v in (('code', vdir[l]), ('rand', rd0)):
            u_ = g1(blocks[l].input_layernorm) * torch.as_tensor(v, device='cuda', dtype=torch.float32)
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
        seg_of = np.array(['bos'] + [next(sg for a, b, sg in segs if a <= s0 < b) if s0 < len(text) else segs[-1][2] for s0, _ in enc.offset_mapping])
        items.append(([bos] + enc.input_ids, seg_of))
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
                at = blocks[l].self_attn
                if qshift:
                    nm, kk = qshift
                    dv = kk * sig[(l, nm)] * (at.q_proj.weight.float() @ udir[(l, nm)])[h * hd:(h + 1) * hd]
                    ctl['q'].setdefault(l, []).append((h, dv.to(at.q_proj.weight.dtype)))
                if kshift:
                    nm, kk = kshift
                    g_ = h // grp
                    if all(g_ != g2 for g2, _ in ctl['k'].get(l, [])):
                        dv = kk * sig[(l, nm)] * (at.k_proj.weight.float() @ udir[(l, nm)])[g_ * hd:(g_ + 1) * hd]
                        ctl['k'].setdefault(l, []).append((g_, dv.to(at.k_proj.weight.dtype)))
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
    save({'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'precision': 'bf16',
          'n_blocks': L, 'n_heads': nh, 'head_dim': hd, 'switch_layer': ls, 'unit_pool': PL,
          'top_units': [list(loc(i)) for i in top], 'random_sets': [[list(loc(i)) for i in s_] for s_ in rsets],
          'patch': {g: v.tolist() for g, v in MP.items()}, 'read': h2raw,
          'edits': {k_: {g: v.tolist() for g, v in d_.items()} for k_, d_ in ME.items()}, 'nll': NL,
          'heads': [list(x) for x in sel], 'random_heads': [[list(x) for x in s_] for s_ in rheads],
          'T_disc_selected': [float(Td[x]) for x in sel], 'sigma': {f'{l}.{nm}': v for (l, nm), v in sig.items()}, 'qk': QK,
          'seconds': round(time.time() - t0, 1)})
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
    h2 = dict(rep['read'])
    h2['pass'] = bool(h2['ratio'] >= crit['H2_read_ratio_min'] and h2['p'] <= crit['H2_p_max'])
    E = {k_: {g: np.array(v) for g, v in d_.items()} for k_, d_ in rep['edits'].items()}

    def erest(name):
        return float(np.mean([(E[name][g][:, 1] - E['orig'][g][:, 1]).mean() / (E['orig'][g][:, 0] - E['orig'][g][:, 1]).mean() for g in E['orig']]))
    e = erest('edit')
    ctrl = [erest('edit_randdir')] + [erest(f'edit_randunits{j}') for j in range(crit['n_random_sets'])]
    h3 = {'edit_restore': e, 'patch_restore': r_top, 'frac_of_patch': e / r_top if r_top else float('nan'), 'controls': ctrl,
          'pass': bool(e >= crit['H3_restore_frac_min'] * r_top and max(ctrl) <= crit['H3_control_max_frac'] * e)}
    nl = rep['nll']
    rise = {k_: (nl['edit'][k_] - nl['orig'][k_]) / nl['orig'][k_] for k_ in ('all', '1', '0')}
    h4 = {'rel_rise_all': rise['all'], 'rel_rise_code': rise['1'], 'rel_rise_prose': rise['0'],
          'pass': bool(rise['all'] <= crit['H4_nll_rise_max'] and rise['1'] <= crit['H4_nll_rise_max'])}
    Q = rep['qk']
    lq, lqr = np.log(Q['q_code_4']['closing'] / Q['orig']['closing']), np.log(Q['q_rand_4']['closing'] / Q['orig']['closing'])
    h5 = {'ratio': float(np.exp(lq)), 'random_ratio': float(np.exp(lqr)),
          'pass': bool(np.exp(lq) >= crit['H5_query_ratio_min'] and abs(lq) >= crit['over_random_log'] * abs(lqr))}
    lk, lkr = np.log(Q['k_code_-4']['code'] / Q['orig']['code']), np.log(Q['k_rand_-4']['code'] / Q['orig']['code'])
    h6 = {'ratio': float(np.exp(lk)), 'random_ratio': float(np.exp(lkr)),
          'pass': bool(np.exp(lk) <= crit['H6_key_ratio_max'] and abs(lk) >= crit['over_random_log'] * abs(lkr))}
    lc = {}
    for l, _ in rep['top_units']:
        lc[str(l)] = lc.get(str(l), 0) + 1
    verdict = {f'H{i}': h['pass'] for i, h in enumerate([h1, h2, h3, h4, h5, h6], 1)}
    rep = save({'criteria': crit, 'H1': h1, 'H2': h2, 'H3': h3, 'H4': h4, 'H5': h5, 'H6': h6, 'unit_layer_counts': lc, 'verdict': verdict})
    p = write_result('gc_v1_gemma-3-4b-it', rep, model_path=MODEL, seed=crit['seed'], tool='scripts/gemma_chain_v1.py')
    print(json.dumps(verdict), '\nH1', h1, '\nH2', h2, '\nH3', h3, '\nH4', h4, '\nH5', h5, '\nH6', h6, '\nlayers', lc,
          '\nheads', rep['heads'], '\nqk', Q, '\nprovenance →', p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run', 'analyze'])
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': do_run, 'analyze': do_analyze}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
