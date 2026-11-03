# -*- coding: utf-8 -*-
"""GC-v2：Gemma-3-4B 门控单元实际读入的方向——冲突时代码信号传到单元所在层后的方向。

接 Phase 290（GC-v1）：Gemma 的 32 个门控单元修补恢复 0.84，但只弱地读自然文本代码方向（2.45 倍），移除它只得修补效应的 4%。
  冲突差值方向 d_l：A 任务（选单元用的发现划分）上，同一文本在块 17 只写散文与写散文 + 代码两种条件下，
      单元所在块 MLP 输入（pre_feedforward_layernorm 的输入，逐位置除以 RMS）之差，在写入的全部位置（BOS 除外）上取平均。
      它是写入的代码方向经块 17 → l 传播后的样子，只依赖 A 任务。
  H1（读权重）：32 个单元的读分数（gate 与 up 行 ×(1 + w_pre_ffw) 对 d_l）≥ 同层随机集合（200 组）的 3 倍，p ≤ 0.01。
  H2（编辑）：从 32 个单元的 gate 与 up 行中移除 d_l（γ = 0 维度不动），B 任务三组平均的散文恢复 ≥ 0.5 × GC-v1 的修补恢复（0.835）；
      对照：同一单元移除随机方向、5 组随机单元（GC-v1 的 random_sets）移除 d_l，恢复各 ≤ 编辑的 1/3。
  H3：编辑后 MF-v1 确认集自然文本 NLL 相对上升 ≤ 1%（全部与代码文本）。
  描述：d_l 与自然代码方向 v_l 的余弦；移除冲突差值前 4 个主方向（逐位置差值的 PCA）的编辑恢复。
被测模型：Gemma-3-4B-it（bf16，不使用量化）。
用法：python scripts/gemma_chain_v2.py prereg | run | analyze
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
import gemma_chain_v1 as GC1  # noqa: E402

PREREG = ROOT / 'research/weight_space/GC-v2.prereg.json'
MODEL = GC1.MODEL
STY = F.STYLES
CODE = STY.index('code')
GROUPS = GC1.GROUPS


def prereg_doc():
    return {'id': 'GC-v2', 'schema': 'gemma_chain_prereg.v2', 'created': '2026-11-02',
            'basis': 'Phase 290（GC-v1）：Gemma 门控单元可修补（0.84）但不线性读自然代码方向；本版检验它们读的是否是写入后传播的代码方向',
            'models': {'gemma-3-4b-it': 'bf16'}, 'precision_rule': '不使用任何量化',
            'units': 'research/weight_space/GC-v1-gemma-3-4b-it/report.json 的 top_units 与 random_sets；修补恢复参照 GC-v1 的 H1 值',
            'materials': 'MF-v1 语料；MF-v4 写入范式（A 任务求 d_l，B 任务测恢复）',
            'power_note': '每组 B 任务约 40 段；200 / 5 组随机集合；连续量阈值',
            'doc': __doc__, 'criteria': W.CRITERIA_GC2}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（GC-v3）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def report_path():
    d = ROOT / 'research/weight_space/GC-v2-gemma-3-4b-it'
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
    gc1 = json.loads(GC1.report_path().read_text(encoding='utf-8'))
    top = [tuple(x) for x in gc1['top_units']]
    rsets = [[tuple(x) for x in s_] for s_ in gc1['random_sets']]
    snap = FU1.snapshot(MODEL)
    tok, model = FU1.load(MODEL)
    lm = model.model.language_model
    blocks = lm.layers
    tcfg = model.config.text_config
    L, d = tcfg.num_hidden_layers, tcfg.hidden_size
    ls = round(0.5 * L)
    bos = tok.bos_token_id
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    mf1crit = json.loads(MF1.PREREG.read_text(encoding='utf-8'))['criteria']
    C, ids0 = MF1.tokenize(tok)
    ids = [[bos] + x for x in ids0]
    excl, shared = MF1.vocab_and_shared(C, ids0, mf1crit)
    shm = [np.array([False] + [t in shared[C[i]['split']] for t in ids0[i]]) for i in range(len(C))]
    EX = [torch.as_tensor(excl[s], device='cuda') for s in STY]
    g1 = lambda mod: 1.0 + mod.weight.float()
    t0 = time.time()

    def batch(seqs):
        n = max(map(len, seqs))
        i_ = torch.full((len(seqs), n), pad, dtype=torch.long)
        att = torch.zeros((len(seqs), n), dtype=torch.long)
        for b, s in enumerate(seqs):
            i_[b, :len(s)] = torch.as_tensor(s)
            att[b, :len(s)] = 1
        return i_.cuda(), att.cuda()

    TLu = sorted({l for l, _ in top} | {l for s_ in rsets for l, _ in s_})
    disc = [i for i, c in enumerate(C) if c['split'] == 'disc']
    conf = [i for i, c in enumerate(C) if c['split'] == 'conf']
    need = sorted(set(TLu) | {ls})
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
    vdir, V, SG = {}, {}, {}
    for l in need:
        X = np.concatenate(Xs[l])
        vdir[l] = W.unit(F.token_matched_direction(X, ys, ts, CODE))
        if l == ls:
            for s in range(4):
                v = F.token_matched_direction(X, ys, ts, s)
                V[s] = v / np.linalg.norm(v)
                SG[s] = float((X @ V[s]).std())
    del Xs
    gc.collect()
    print(f'@@S0 {round(time.time() - t0)}s', flush=True)
    st = {'inject': None, 'pos': None, 'cap': False}
    cap = {}

    def inj(mod, args, kwargs):
        if st['inject'] is None:
            return None
        h = (args[0] if args else kwargs['hidden_states']).clone()
        h[:, :st['pos']] += st['inject'][:, None, :].to(h.dtype)
        if args:
            return (h,) + tuple(args[1:]), kwargs
        kwargs['hidden_states'] = h
        return args, kwargs

    def pre_ffw(l):
        def f(m, a):
            if st['cap']:
                x = a[0].float()
                cap[l] = x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + m.eps)
            return None
        return f

    hooks = [blocks[ls].register_forward_pre_hook(inj, with_kwargs=True)]
    hooks += [blocks[l].pre_feedforward_layernorm.register_forward_pre_hook(pre_ffw(l)) for l in TLu]
    vec = lambda s: 6 * SG[s] * V[s]
    texts = {g: [i for i in disc if C[i]['style'] == g[0]] for g in GROUPS}
    # 冲突差值方向（只用 A 任务）
    dsum = {l: torch.zeros(d, device='cuda') for l in TLu}
    dsamp = {l: [] for l in TLu}
    nd = 0
    for g in GROUPS:
        p = STY.index(g[1])
        for i in texts[g]:
            if not MF3.in_A(C[i]['task']):
                continue
            n = max(2, round(0.6 * len(ids[i])))
            st['pos'], st['cap'] = n, True
            st['inject'] = torch.as_tensor(np.stack([vec(p), vec(p) + vec(CODE)]), dtype=torch.float32, device='cuda')
            i_, att = batch([ids[i][:n]] * 2)
            lm(input_ids=i_, attention_mask=att)
            st['cap'] = False
            for l in TLu:
                dl = cap[l][1, 1:n] - cap[l][0, 1:n]
                dsum[l] += dl.sum(0)
                dsamp[l].append(dl.cpu().numpy())
            nd += n - 1
    st['inject'] = None
    ddir = {l: W.unit((dsum[l] / nd).cpu().numpy()) for l in TLu}
    sub = {}
    for l in TLu:
        D_ = np.concatenate(dsamp[l])
        _, _, vt = np.linalg.svd(D_ - 0 * D_.mean(0), full_matrices=False)       # 不中心化：保留平均方向
        sub[l] = vt[:crit['sub_rank']]
    del dsamp
    cos_dv = {str(l): float(ddir[l] @ vdir[l]) for l in TLu}
    print('@@cos(d, v)', cos_dv, f'{round(time.time() - t0)}s', flush=True)
    gam = {l: g1(blocks[l].pre_feedforward_layernorm).cpu().numpy() for l in TLu}

    def read_mean(units):
        vals = []
        for l, u_ in units:
            a_g = blocks[l].mlp.gate_proj.weight[u_].float().cpu().numpy()[None] * gam[l]
            a_u = blocks[l].mlp.up_proj.weight[u_].float().cpu().numpy()[None] * gam[l]
            vals.append(W.read_scores(a_g, a_u, ddir[l])[0])
        return float(np.mean(vals))

    pool = {l: blocks[l].mlp.down_proj.in_features for l in TLu}
    layer_of = np.concatenate([[l] * pool[l] for l in TLu])
    off = dict(zip(TLu, np.concatenate([[0], np.cumsum([pool[l] for l in TLu])[:-1]]).tolist()))
    gid = [off[l] + u_ for l, u_ in top]
    rr = F.random_units_matched(gid, layer_of, crit['n_random_sets_read'], crit['seed'])
    back = lambda i: (int(layer_of[i]), int(i - off[int(layer_of[i])]))
    h1raw = W.ratio_vs_random(read_mean(top), [read_mean([back(i) for i in s_]) for s_ in rr])
    h1raw['natural_dir_read_ratio_gc1'] = gc1['read']['ratio']
    print('@@H1', {k_: round(v, 4) for k_, v in h1raw.items()}, flush=True)

    def measure():
        out = {}
        for g in GROUPS:
            p = STY.index(g[1])
            M = []
            for i in texts[g]:
                if MF3.in_A(C[i]['task']):
                    continue
                n = max(2, round(0.6 * len(ids[i])))
                st['pos'] = n
                st['inject'] = torch.as_tensor(np.stack([vec(p), vec(p) + vec(CODE)]), dtype=torch.float32, device='cuda')
                i_, att = batch([ids[i][:n]] * 2)
                lp = torch.log_softmax(model.lm_head(lm(input_ids=i_, attention_mask=att).last_hidden_state[:, n - 1]).float(), -1)
                M.append(torch.logsumexp(lp[:, EX[p]], -1).cpu().numpy())
            out['%s:%s' % g] = np.array(M)
        st['inject'] = None
        return out

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

    def edit(units, dirs):
        saved = []
        by = {}
        for l, u_ in units:
            by.setdefault(l, []).append(u_)
        for l, us in by.items():
            for P in (blocks[l].mlp.gate_proj.weight, blocks[l].mlp.up_proj.weight):
                saved.append((P, us, P.data[us].clone()))
                new = W.remove_subspace(P.data[us].float().cpu().numpy(), gam[l], np.atleast_2d(dirs[l]))
                P.data[us] = torch.as_tensor(new, dtype=P.dtype, device=P.device)
        return saved

    def restore(saved):
        for P, us, old in reversed(saved):
            P.data[us] = old

    rng = np.random.default_rng(crit['seed'])
    rdir = {l: W.unit(rng.normal(0, 1, d)) for l in TLu}
    ME, NL = {'orig': measure()}, {'orig': nll()}
    for name, units, dirs in [('edit_d', top, ddir), ('edit_sub4', top, sub), ('edit_randdir', top, rdir)] + \
                             [(f'edit_randunits{j}', s_, ddir) for j, s_ in enumerate(rsets)]:
        sv = edit(units, dirs)
        try:
            ME[name] = measure()
            if name in ('edit_d', 'edit_sub4'):
                NL[name] = nll()
        finally:
            restore(sv)
        print(f'@@{name} {round(time.time() - t0)}s', flush=True)
    for h_ in hooks:
        h_.remove()
    save({'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'precision': 'bf16',
          'switch_layer': ls, 'unit_layers': TLu, 'cos_d_v': cos_dv, 'read': h1raw, 'patch_restore_gc1': gc1['H1']['value'],
          'edits': {k_: {g: v.tolist() for g, v in d_.items()} for k_, d_ in ME.items()}, 'nll': NL,
          'seconds': round(time.time() - t0, 1)})
    print('done', flush=True)


def do_analyze():
    from provenance import write_result
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    rep = json.loads(report_path().read_text(encoding='utf-8'))
    h1 = dict(rep['read'])
    h1['pass'] = bool(h1['ratio'] >= crit['H1_read_ratio_min'] and h1['p'] <= crit['H1_p_max'])
    E = {k_: {g: np.array(v) for g, v in d_.items()} for k_, d_ in rep['edits'].items()}
    er = lambda nm: float(np.mean([(E[nm][g][:, 1] - E['orig'][g][:, 1]).mean() / (E['orig'][g][:, 0] - E['orig'][g][:, 1]).mean() for g in E['orig']]))
    per = lambda nm: {g: float((E[nm][g][:, 1] - E['orig'][g][:, 1]).mean() / (E['orig'][g][:, 0] - E['orig'][g][:, 1]).mean()) for g in E['orig']}
    ref = rep['patch_restore_gc1']
    e = er('edit_d')
    ctrl = [er('edit_randdir')] + [er(f'edit_randunits{j}') for j in range(crit['n_random_sets'])]
    h2 = {'edit_restore': e, 'per_group': per('edit_d'), 'patch_restore': ref, 'frac_of_patch': e / ref, 'controls': ctrl,
          'pass': bool(e >= crit['H2_restore_frac_min'] * ref and max(ctrl) <= crit['H2_control_max_frac'] * e)}
    nl = rep['nll']
    rise = {k_: (nl['edit_d'][k_] - nl['orig'][k_]) / nl['orig'][k_] for k_ in ('all', '1', '0')}
    h3 = {'rel_rise_all': rise['all'], 'rel_rise_code': rise['1'], 'rel_rise_prose': rise['0'],
          'pass': bool(rise['all'] <= crit['H3_nll_rise_max'] and rise['1'] <= crit['H3_nll_rise_max'])}
    desc = {'cos_d_v': rep['cos_d_v'], 'edit_sub4_restore': er('edit_sub4'), 'edit_sub4_per_group': per('edit_sub4'),
            'edit_sub4_nll_rise_all': (nl['edit_sub4']['all'] - nl['orig']['all']) / nl['orig']['all']}
    verdict = {'H1': h1['pass'], 'H2': h2['pass'], 'H3': h3['pass']}
    rep = save({'criteria': crit, 'H1': h1, 'H2': h2, 'H3': h3, 'describe': desc, 'verdict': verdict})
    p = write_result('gc_v2_gemma-3-4b-it', rep, model_path=MODEL, seed=crit['seed'], tool='scripts/gemma_chain_v2.py')
    print(json.dumps(verdict), '\nH1', h1, '\nH2', h2, '\nH3', h3, '\ndesc', desc, '\nprovenance →', p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run', 'analyze'])
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': do_run, 'analyze': do_analyze}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
