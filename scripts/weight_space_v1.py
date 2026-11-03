# -*- coding: utf-8 -*-
"""WS-v1：用权重阅读解释「代码压过散文」的门控单元（Phase 278 MF-v4 的 32 个单元，Qwen3-4B 块 27 / 33 / 35）。

参考「权重空间编程」：先只读权重提出假设（零前向），再用少量前向与一次封闭式权重编辑判定。
  代码方向 v_l：MF-v1 发现集共用 token 位置上 hs[l] 的 token 配平方向（l = 单元所在块）。单元读出向量 a_i = γ_l ⊙ W_gate[i]（up 同理）。
  H1（只读权重）：32 个单元对 v_l 的读分数 sqrt(cos_gate² + cos_up²) 的平均 ≥ 同层随机 32 单元集合（200 组）的 3 倍，经验 p ≤ 0.01。
  H2（只读权重 → 前向判定）：注意力头 h（块 λ）对这些单元读出矩阵的组合分数 CS_h（只用 W_O、W_V、γ_in 与读出行，块 λ ≤ l 的单元）
      预测前向实测的 |T_h|：T_h = 头 h 对门控得分 Σ s_i g_i 的直接贡献在 MF-v1 确认集自然文本「代码 − 散文」共用 token 位置上的均值差
      （g_i = gate 预激活；s_i 由发现集文本定号；直接贡献 = o_h · Σ s_i a_i / RMS）。
      判据：Spearman ≥ 0.3，且高于范数基线 ‖W_OV‖_F 的 Spearman，且前 16 重合置换 p ≤ 0.05（全部 35 × 32 个头）。
  H3（写：封闭式编辑）：从 32 个单元的 gate 与 up 行中移除 v_l（a' = a − (a·v)v，改 32 × 2 × 2560 个权重），
      在 MF-v4 的写入范式（同一批 B 任务文本、同一写入强度）中，散文质量的恢复 ≥ MF-v4 激活修补恢复（三组平均）的 0.5 倍；
      对照：同一单元移除随机方向、5 组同层随机单元移除 v_l，恢复各 ≤ 真实编辑的 1/3。
  H4（损伤检查）：编辑后 MF-v1 确认集自然文本的平均下一词 NLL 相对上升 ≤ 1%（全部文本与代码文本分别）。
  描述：直接贡献分解（嵌入、各块 MLP、各头）与读出核对；前几个单元 W_down 列经最终归一化与 unembedding 的读出 token。
被测模型：Qwen3-4B（bf16，不使用量化；单元来自 MF-v4，只有 Qwen 的 bf16 结果）。
用法：python scripts/weight_space_v1.py prereg | run | analyze
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
import xm_replicate_v1 as XM  # noqa: E402

PREREG = ROOT / 'research/weight_space/WS-v1.prereg.json'
MODEL = 'models/hf/qwen3-4b'
STY = F.STYLES
CODE = STY.index('code')
TL = [27, 33, 35]
MF4 = ROOT / 'research/mode_field/MF-v4-qwen3-4b/report.json'


def prereg_doc():
    return {'id': 'WS-v1', 'schema': 'weight_space_prereg.v1', 'created': '2026-11-02',
            'basis': 'Phase 278（MF-v4）的 32 个门控单元；方法参考 Takehi 等「Sparks of Weight-Space Programming」（2026-11-02）',
            'models': {'qwen3-4b': 'bf16'}, 'precision_rule': '不使用任何量化',
            'units': 'research/mode_field/MF-v4-qwen3-4b/report.json 与原生 mf_v4_units.npz 的 top（MF-v4 只在 A 任务上选出）',
            'materials': 'MF-v1 发现集（定号、方向）与确认集（H2 真值、H4）；H3 用 MF-v4 的 B 任务文本与写入范式',
            'mf_v4_reference': 'H3 的参照恢复 = MF-v4 报告 groups[*].restore_B.top 的三组平均（已有结果，非本版重算）',
            'power_note': 'H1：200 组随机集合；H2：1120 个头、2000 次置换；H3 / H4 连续量阈值',
            'doc': __doc__, 'criteria': W.CRITERIA}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（WS-v2）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def report_path():
    d = ROOT / 'research/weight_space/WS-v1-qwen3-4b'
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
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    snap, key = XM.key_of(MODEL, False)
    od = MF1.out_dir(key)
    U = np.load(od / 'mf_v4_units.npz')
    top, layer_of = [int(x) for x in U['top']], U['layer_of']
    off = {int(k_): v for k_, v in json.loads(MF4.read_text(encoding='utf-8'))['unit_offsets'].items()}
    loc = lambda i: (int(layer_of[i]), int(i - off[int(layer_of[i])]))
    by_layer = lambda units: {l: [loc(i)[1] for i in units if loc(i)[0] == l] for l in TL}
    rsets = F.random_units_matched(top, layer_of, crit['n_random_sets'], crit['seed'])
    # ---------- 方向（只用发现集） ----------
    z = MF1.load_fields(key)
    L = int(z['Hs'].shape[1]) - 1
    ls = round(0.5 * L)
    y, tk, sp, sh = z['style'], z['tok'], z['split'], z['shared'].astype(bool)
    D = sp == 'disc'
    vdir, V, SG = {}, {}, {}
    for l in TL + [ls]:
        X = z['Hs'][D[sh], l].astype(np.float32)
        if l in TL:
            vdir[l] = W.unit(F.token_matched_direction(X, y[sh][D[sh]], tk[sh][D[sh]], CODE))
        if l == ls:
            for s in range(4):
                v = F.token_matched_direction(X, y[sh][D[sh]], tk[sh][D[sh]], s)
                V[s] = v / np.linalg.norm(v)
                SG[s] = float((X @ V[s]).std())
        del X
    shared_all = z['shared'].astype(bool)
    del z
    gc.collect()
    excl = json.loads((od / 'vocab.json').read_text(encoding='utf-8'))['excl']
    tok, model = XM.load(MODEL, False)
    R = XM.Runner(tok, model)
    C, ids = MF1.tokenize(tok)
    blocks = R.blocks
    t0 = time.time()
    cfg = model.config
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
    gam = {l: blocks[l].post_attention_layernorm.weight.float().cpu().numpy() for l in TL}
    Wg = {l: blocks[l].mlp.gate_proj.weight.float().cpu().numpy() for l in TL}
    Wu = {l: blocks[l].mlp.up_proj.weight.float().cpu().numpy() for l in TL}
    # ---------- H1：只读权重 ----------
    def read_mean(units):
        return float(np.mean([W.read_scores((Wg[l][[u]] * gam[l]), (Wu[l][[u]] * gam[l]), vdir[l])[0]
                              for l, u in map(loc, units)]))
    h1 = W.ratio_vs_random(read_mean(top), [read_mean(s) for s in rsets])
    per_unit_read = [{'layer': loc(i)[0], 'unit': loc(i)[1],
                      'cos_gate': float(W.cosines(Wg[loc(i)[0]][[loc(i)[1]]] * gam[loc(i)[0]], vdir[loc(i)[0]])[0]),
                      'cos_up': float(W.cosines(Wu[loc(i)[0]][[loc(i)[1]]] * gam[loc(i)[0]], vdir[loc(i)[0]])[0])} for i in top]
    print('@@H1', {k_: round(v, 4) for k_, v in h1.items()}, flush=True)
    # ---------- H2 预测：组合分数（只读权重） ----------
    def read_rows(units):
        bl = by_layer(units)
        return {l: np.concatenate([Wg[l][bl[l]] * gam[l], Wu[l][bl[l]] * gam[l]]) for l in TL if bl[l]}
    RR, RRr = read_rows(top), read_rows(rsets[0])
    CS, CSr, NB = np.zeros((TL[-1], nh)), np.zeros((TL[-1], nh)), np.zeros((TL[-1], nh))
    for lam in range(TL[-1]):
        at = blocks[lam].self_attn
        Wo = at.o_proj.weight.float().cpu().numpy()
        Wv = at.v_proj.weight.float().cpu().numpy() * blocks[lam].input_layernorm.weight.float().cpu().numpy()[None, :]
        Rl = np.concatenate([RR[l] for l in TL if l >= lam and l in RR])
        Rlr = np.concatenate([RRr[l] for l in TL if l >= lam and l in RRr])
        for h in range(nh):
            wo, wv = Wo[:, h * hd:(h + 1) * hd], Wv[(h // (nh // nkv)) * hd:(h // (nh // nkv) + 1) * hd]
            CS[lam, h] = W.composition_lowrank(Rl, wo, wv)
            CSr[lam, h] = W.composition_lowrank(Rlr, wo, wv)
            _, ro = np.linalg.qr(wo)
            NB[lam, h] = np.linalg.norm(ro @ wv)
    print(f'@@progress composition {round(time.time() - t0)}s', flush=True)
    # ---------- 前向：定号（发现集）与直接贡献（确认集） ----------
    starts = np.concatenate([[0], np.cumsum([len(x) for x in ids])])
    lab_code = lambda i: STY.index(C[i]['style']) == CODE
    st = {'mode': None}
    cap = {}
    tl_units = by_layer(top)
    uidx = {l: torch.as_tensor(tl_units[l], device='cuda') for l in TL}

    def gate_hook(l):
        def f(m, a, o):
            if st['mode'] == 'sign':
                cap[('g', l)] = o[..., uidx[l]].float()
        return f

    hooks = [blocks[l].mlp.gate_proj.register_forward_hook(gate_hook(l)) for l in TL]
    disc = [i for i, cc in enumerate(C) if cc['split'] == 'disc']
    conf = [i for i, cc in enumerate(C) if cc['split'] == 'conf']
    gs = {l: [[], []] for l in TL}                      # [散文, 代码]
    st['mode'] = 'sign'
    for j in range(0, len(disc), 16):
        ch = disc[j:j + 16]
        i_, att = R._batch([ids[i] for i in ch])
        R.model.model(input_ids=i_, attention_mask=att)
        for b, i in enumerate(ch):
            n = len(ids[i])
            m = torch.as_tensor(shared_all[starts[i]:starts[i] + n], device='cuda')
            for l in TL:
                gs[l][int(lab_code(i))].append(cap[('g', l)][b, :n][m].cpu().numpy())
    sgn = {l: np.sign(np.concatenate(gs[l][1]).mean(0) - np.concatenate(gs[l][0]).mean(0)) for l in TL}
    for h_ in hooks:
        h_.remove()
    S = {l: (sgn[l][:, None] * Wg[l][tl_units[l]] * gam[l]).sum(0) for l in TL if tl_units[l]}        # 带号读出和 [d]
    TLu = [l for l in TL if l in S]
    Sm = torch.as_tensor(np.stack([S[l] for l in TLu]), device='cuda')                                 # [k, d]
    Qh = {}
    for lam in range(TL[-1] + 1):
        Wo = blocks[lam].self_attn.o_proj.weight.float()                                               # [d, nh*hd]
        q = (Sm @ Wo).reshape(len(TLu), nh, hd)                                                       # [k, nh, hd]
        q = q * torch.as_tensor([1.0 if l >= lam else 0.0 for l in TLu], device='cuda')[:, None, None]
        Qh[lam] = q
    mmask = {lam: torch.as_tensor([1.0 if l > lam else 0.0 for l in TLu], device='cuda') for lam in range(TL[-1] + 1)}
    pos_cap = {}

    def oproj_pre(lam):
        def f(m, a):
            if st['mode'] == 'contrib':
                zz = a[0].float().reshape(a[0].shape[0], a[0].shape[1], nh, hd)
                pos_cap[('a', lam)] = torch.einsum('bthd,khd->btkh', zz, Qh[lam])
            return None
        return f

    def mlp_post(lam):
        def f(m, a, o):
            if st['mode'] == 'contrib':
                pos_cap[('m', lam)] = (o.float() @ Sm.T) * mmask[lam]
        return f

    def rms_pre(l):
        def f(m, a):
            if st['mode'] == 'contrib':
                x = a[0].float()
                pos_cap[('r', l)] = torch.sqrt((x ** 2).mean(-1) + m.variance_epsilon)
            return None
        return f

    emb_cap = {}
    hooks = [blocks[lam].self_attn.o_proj.register_forward_pre_hook(oproj_pre(lam)) for lam in range(TL[-1] + 1)]
    hooks += [blocks[lam].mlp.register_forward_hook(mlp_post(lam)) for lam in range(TL[-1])]
    hooks += [blocks[l].post_attention_layernorm.register_forward_pre_hook(rms_pre(l)) for l in TLu]
    hooks += [blocks[l].mlp.gate_proj.register_forward_hook(
        (lambda l: lambda m, a, o: pos_cap.__setitem__(('g', l), o[..., uidx[l]].float()) if st['mode'] == 'contrib' else None)(l)) for l in TLu]
    hooks.append(R.model.model.embed_tokens.register_forward_hook(lambda m, a, o: emb_cap.__setitem__('e', o.float()) if st['mode'] == 'contrib' else None))
    sg_t = {l: torch.as_tensor(sgn[l], device='cuda', dtype=torch.float32) for l in TLu}
    acc = {'head': [np.zeros((TL[-1] + 1, nh)), np.zeros((TL[-1] + 1, nh))], 'mlp': [np.zeros(TL[-1]), np.zeros(TL[-1])],
           'emb': [0.0, 0.0], 'total': [0.0, 0.0], 'n': [0, 0]}
    recon_err, recon_scale = [], []
    st['mode'] = 'contrib'
    for j in range(0, len(conf), 8):
        ch = conf[j:j + 8]
        i_, att = R._batch([ids[i] for i in ch])
        R.model.model(input_ids=i_, attention_mask=att)
        rinv = torch.stack([1.0 / pos_cap[('r', l)] for l in TLu], -1)                               # [b, t, k]
        for b, i in enumerate(ch):
            n = len(ids[i])
            m = torch.as_tensor(shared_all[starts[i]:starts[i] + n], device='cuda')
            c = int(lab_code(i))
            ri = rinv[b, :n][m]                                                                       # [p, k]
            hc = sum((pos_cap[('a', lam)][b, :n][m] * ri[:, :, None]).sum(1).sum(0).cpu().numpy()[None] *
                     np.eye(TL[-1] + 1)[lam][:, None] for lam in range(TL[-1] + 1))                    # [λ, nh]
            mc = np.array([(pos_cap[('m', lam)][b, :n][m] * ri).sum().item() for lam in range(TL[-1])])
            ec = ((emb_cap['e'][b, :n][m] @ Sm.T) * ri).sum().item()
            tot = sum((pos_cap[('g', l)][b, :n][m] * sg_t[l]).sum().item() for l in TLu)
            dsum = hc.sum() + mc.sum() + ec
            recon_err.append(abs(dsum - tot))
            recon_scale.append(abs(tot))
            acc['head'][c] += hc
            acc['mlp'][c] += mc
            acc['emb'][c] += ec
            acc['total'][c] += tot
            acc['n'][c] += int(m.sum().item())
    for h_ in hooks:
        h_.remove()
    st['mode'] = None
    mean = lambda k_: {c: acc[k_][c] / acc['n'][c] for c in (0, 1)}
    Th = mean('head')[1] - mean('head')[0]
    Tm = mean('mlp')[1] - mean('mlp')[0]
    Te = mean('emb')[1] - mean('emb')[0]
    Tt = mean('total')[1] - mean('total')[0]
    print(f'@@progress contributions {round(time.time() - t0)}s recon {np.sum(recon_err) / np.sum(recon_scale):.4f}', flush=True)
    # ---------- H3：封闭式编辑（MF-v4 写入范式，B 任务） ----------
    EX = [torch.as_tensor(excl[s], device='cuda') for s in STY]
    vec = lambda s: 6 * SG[s] * V[s]
    inj = {'v': None, 'pos': None}

    def inj_hook(mod, args, kwargs):
        if inj['v'] is None:
            return None
        h = (args[0] if args else kwargs['hidden_states']).clone()
        h[:, :inj['pos']] += inj['v'][:, None, :].to(h.dtype)
        if args:
            return (h,) + tuple(args[1:]), kwargs
        kwargs['hidden_states'] = h
        return args, kwargs

    hk = blocks[ls].register_forward_pre_hook(inj_hook, with_kwargs=True)
    groups = [tuple(g) for g in json.loads(MF4.read_text(encoding='utf-8'))['criteria']['groups']]
    btexts = {g: [i for i, cc in enumerate(C) if cc['split'] == 'disc' and cc['style'] == g[0] and not MF3.in_A(cc['task'])] for g in groups}

    def measure():
        out = {}
        for g in groups:
            p = STY.index(g[1])
            M = []
            for i in btexts[g]:
                n = max(2, round(0.6 * len(ids[i])))
                inj['pos'] = n
                inj['v'] = torch.as_tensor(np.stack([vec(p), vec(p) + vec(CODE)]), dtype=torch.float32, device='cuda')
                i_, att = R._batch([ids[i][:n]] * 2)
                lp = torch.log_softmax(R.model(input_ids=i_, attention_mask=att).logits[:, n - 1].float(), -1)
                M.append(torch.logsumexp(lp[:, EX[p]], -1).cpu().numpy())
            out['%s:%s' % g] = np.array(M)                                                             # [n, 2]：只散文、同时
        inj['v'] = None
        return out

    def nll():
        out = {0: [], 1: []}
        for j in range(0, len(conf), 8):
            ch = conf[j:j + 8]
            i_, att = R._batch([ids[i] for i in ch])
            lg = R.model(input_ids=i_, attention_mask=att).logits.float()
            lp = torch.log_softmax(lg[:, :-1], -1).gather(-1, i_[:, 1:, None])[..., 0]
            for b, i in enumerate(ch):
                n = len(ids[i])
                out[int(lab_code(i))].append(float(-lp[b, :n - 1].mean()))
        return out

    def edit(units, dirs):
        saved = []
        for l, us in by_layer(units).items():
            if not us:
                continue
            for P in (blocks[l].mlp.gate_proj.weight, blocks[l].mlp.up_proj.weight):
                rows = P.data[us].float().cpu().numpy()
                saved.append((P, us, P.data[us].clone()))
                P.data[us] = torch.as_tensor(W.remove_direction(rows, gam[l], dirs[l]), dtype=P.dtype, device=P.device)
        return saved

    def restore(saved):
        for P, us, old in saved:
            P.data[us] = old

    rng = np.random.default_rng(crit['seed'])
    rdir = {l: W.unit(rng.normal(0, 1, len(vdir[l]))) for l in TL}
    res_m, res_nll = {'orig': measure()}, {'orig': nll()}
    for name, units, dirs in [('edit', top, vdir), ('edit_randdir', top, rdir)] + \
                             [(f'edit_randunits{j}', rsets[j], vdir) for j in range(5)]:
        sv = edit(units, dirs)
        try:
            res_m[name] = measure()
            if name in ('edit', 'edit_randdir'):
                res_nll[name] = nll()
        finally:
            restore(sv)
        print(f'@@progress {name} {round(time.time() - t0)}s', flush=True)
    hk.remove()
    nd = R.model.model.norm.weight.float()
    Wdn = {l: blocks[l].mlp.down_proj.weight.float() for l in TL}
    readout = []
    for i in top[:8]:
        l, u = loc(i)
        lg = R.model.lm_head.weight.float() @ (nd * Wdn[l][:, u])
        readout.append({'layer': l, 'unit': u, 'up': [tok.decode([int(t)]) for t in lg.topk(6).indices],
                        'down': [tok.decode([int(t)]) for t in (-lg).topk(6).indices]})
    od2 = Path('D:/ai2050_native/_weight_space') / snap[:16] / 'WS-v1'
    od2.mkdir(parents=True, exist_ok=True)
    np.savez(od2 / 'ws_v1.npz', CS=CS, CSr=CSr, NB=NB, Th=Th, Tm=Tm,
             **{f'M_{k_}_{g.replace(":", "_")}': v for k_, d_ in res_m.items() for g, v in d_.items()})
    save({'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'precision': 'bf16',
          'n_heads': nh, 'n_kv': nkv, 'head_dim': hd, 'switch_layer': ls, 'H1_raw': h1, 'per_unit_read': per_unit_read,
          'contrib': {'Th': Th.tolist(), 'Tm': Tm.tolist(), 'Te': Te, 'Tt': Tt,
                      'recon_rel_err': float(np.sum(recon_err) / np.sum(recon_scale)), 'n_pos': acc['n']},
          'nll': {k_: {str(c): float(np.mean(v[c])) for c in (0, 1)} | {'all': float(np.mean(v[0] + v[1]))} for k_, v in res_nll.items()},
          'unit_signs': {str(l): sgn[l].tolist() for l in TL}, 'unit_readout': readout, 'seconds': round(time.time() - t0, 1)})
    print('done', round(time.time() - t0, 1), flush=True)


def do_analyze():
    from provenance import write_result
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    rep = json.loads(report_path().read_text(encoding='utf-8'))
    snap = rep['model_snapshot']
    Z = np.load(Path('D:/ai2050_native/_weight_space') / snap[:16] / 'WS-v1' / 'ws_v1.npz')
    CS, CSr, NB = Z['CS'], Z['CSr'], Z['NB']
    Th = np.array(rep['contrib']['Th'])[:CS.shape[0]]
    truth = np.abs(Th).ravel()
    h2 = W.head_verdict(CS.ravel(), NB.ravel(), truth, crit)
    h2['spearman_random_units_cs'] = W.spearman(CSr.ravel(), truth)
    mf4 = json.loads(MF4.read_text(encoding='utf-8'))['groups']
    groups = list(mf4.keys())
    rest = {}
    for name in ['edit', 'edit_randdir'] + [f'edit_randunits{j}' for j in range(5)]:
        r = {}
        for g in groups:
            o, e = Z[f'M_orig_{g.replace(":", "_")}'], Z[f'M_{name}_{g.replace(":", "_")}']
            r[g] = float((e[:, 1] - o[:, 1]).mean() / (o[:, 0] - o[:, 1]).mean())
        rest[name] = {'per_group': r, 'mean': float(np.mean(list(r.values())))}
    ref = float(np.mean([mf4[g]['restore_B']['top'] for g in groups]))
    e = rest['edit']['mean']
    ctrl = [rest['edit_randdir']['mean']] + [rest[f'edit_randunits{j}']['mean'] for j in range(5)]
    h3 = {'edit_restore': e, 'mf4_patch_restore': ref, 'frac_of_patch': e / ref, 'controls': ctrl,
          'pass': bool(e >= crit['H3_restore_frac_min'] * ref and max(ctrl) <= crit['H3_control_max_frac'] * e)}
    nl = rep['nll']
    rise = {k_: (nl['edit'][k_] - nl['orig'][k_]) / nl['orig'][k_] for k_ in ('all', '1', '0')}
    h4 = {'rel_rise_all': rise['all'], 'rel_rise_code': rise['1'], 'rel_rise_prose': rise['0'],
          'randdir_rise_all': (nl['edit_randdir']['all'] - nl['orig']['all']) / nl['orig']['all'],
          'pass': bool(rise['all'] <= crit['H4_nll_rise_max'] and rise['1'] <= crit['H4_nll_rise_max'])}
    h1 = dict(rep['H1_raw'])
    h1['pass'] = bool(h1['ratio'] >= crit['H1_read_ratio_min'] and h1['p'] <= crit['H1_p_max'])
    c = rep['contrib']
    tot = c['Tt']
    decomp = {'total_contrast': tot, 'heads_sum': float(np.sum(c['Th'])), 'mlp_sum': float(np.sum(c['Tm'])), 'emb': c['Te'],
              'top_heads': [{'layer': int(k_ // CS.shape[1]), 'head': int(k_ % CS.shape[1]), 'T': float(Th.ravel()[k_]),
                             'CS': float(CS.ravel()[k_]), 'CS_rank': int((CS.ravel() > CS.ravel()[k_]).sum()) + 1}
                            for k_ in np.argsort(-truth)[:12]],
              'top_mlps': sorted([{'layer': j, 'T': float(v)} for j, v in enumerate(c['Tm'])], key=lambda d: -abs(d['T']))[:8]}
    verdict = {'H1': h1['pass'], 'H2': h2['pass'], 'H3': h3['pass'], 'H4': h4['pass']}
    rep = save({'criteria': crit, 'H1': h1, 'H2': h2, 'H3': h3, 'H3_detail': rest, 'H4': h4, 'decomposition': decomp, 'verdict': verdict})
    p = write_result('ws_v1_qwen3-4b', rep, model_path=MODEL, seed=crit['seed'], tool='scripts/weight_space_v1.py')
    print(json.dumps(verdict), '\nH1', h1, '\nH2', {k_: v for k_, v in h2.items()}, '\nH3', h3, '\nH4', h4,
          '\ndecomp', json.dumps(decomp, ensure_ascii=False)[:1500], '\nprovenance →', p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run', 'analyze'])
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': do_run, 'analyze': do_analyze}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
