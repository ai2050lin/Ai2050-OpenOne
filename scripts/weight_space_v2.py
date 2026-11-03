# -*- coding: utf-8 -*-
"""WS-v2：门控单元的上游注意力头——封闭式写出编辑的总效应与注意力模式。

接 Phase 284（WS-v1）：门控得分「代码 − 散文」差约一半由注意力头直接写入，集中在块 32 的少数头。
  选择（只用发现集）：按 WS-v1 的直接贡献分解（门控得分 G = Σ s_i g_i，s_i 沿用 WS-v1 由发现集定的号），
      在 MF-v1 发现集自然文本上取「代码 − 散文」直接贡献最大（推向代码）的 8 个头（块 ≤ 34）。
  编辑：从这 8 个头的写出矩阵 W_O^h 中移除写向代码方向 v_{λ+1}（hs[λ+1] 的 token 配平方向）的分量，W_O' = (I − v vᵀ) W_O。
      对照：5 组同块随机头做同样编辑。描述性：改为移除写向门控读出方向 span{S_l}（l ≥ λ）的分量。
  H1（自然文本，总效应）：确认集门控得分「代码 − 散文」差下降 ≥ 20%，且 ≥ 随机头集合平均下降的 3 倍。
  H2（MF-v4 写入范式，B 任务）：散文质量恢复 ≥ 0.1，且 ≥ 随机头集合平均的 3 倍。
  H3（损伤）：确认集自然文本平均 NLL 相对上升 ≤ 1%（全部与代码文本分别）。
  描述：选中头的直接贡献（确认集）与总效应对照；注意力局部性（代码位置 t ≥ 24，去掉位置 0 后落在最近 16 个位置的比例）与选中头最关注的 token。
被测模型：Qwen3-4B（bf16，不使用量化）。
用法：python scripts/weight_space_v2.py prereg | run | analyze
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts'), str(ROOT / 'tools')]

from shared.dist_protocol import weight_read as W  # noqa: E402
from shared.dist_protocol import mode_field as F  # noqa: E402
import mode_field_v1 as MF1  # noqa: E402
import mode_field_v3 as MF3  # noqa: E402
import xm_replicate_v1 as XM  # noqa: E402
import weight_space_v1 as WS1  # noqa: E402

PREREG = ROOT / 'research/weight_space/WS-v2.prereg.json'
MODEL = WS1.MODEL
STY = F.STYLES
CODE = STY.index('code')
TL = WS1.TL
NATIVE = Path('D:/ai2050_native/_weight_space')


def prereg_doc():
    return {'id': 'WS-v2', 'schema': 'weight_space_prereg.v2', 'created': '2026-11-02',
            'basis': 'Phase 284（WS-v1）：门控得分差约一半由注意力头直接写入（块 32 KV 组 6 等）；本版对上游头做封闭式写出编辑',
            'models': {'qwen3-4b': 'bf16'}, 'precision_rule': '不使用任何量化',
            'units_and_signs': 'MF-v4 的 32 个单元；符号沿用 research/weight_space/WS-v1-qwen3-4b/report.json 的 unit_signs（发现集定号）',
            'selection': 'WS-v1 的确认集逐头贡献已公开；本版重新只在发现集上选头，确认集只用于判定',
            'mf_v4_paradigm': '与 WS-v1 H3 相同的 B 任务文本与写入强度',
            'power_note': 'H1：确认集 160 段文本的共用 token 位置；H2：三组 B 任务；5 组随机头对照；连续量阈值',
            'doc': __doc__, 'criteria': W.CRITERIA_V2}


def do_prereg():
    PREREG.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(prereg_doc(), ensure_ascii=False, indent=1) + '\n'
    if PREREG.exists() and PREREG.read_text(encoding='utf-8') != text:
        raise SystemExit('预注册文件已存在且内容不同：改动须升版本（WS-v3）')
    PREREG.write_text(text, encoding='utf-8', newline='\n')
    sha = hashlib.sha256(text.encode('utf-8')).hexdigest()
    PREREG.with_suffix('.sha256').write_text(sha + '\n', encoding='utf-8', newline='\n')
    print('prereg', sha)


def report_path():
    d = ROOT / 'research/weight_space/WS-v2-qwen3-4b'
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
    ws1 = json.loads(WS1.report_path().read_text(encoding='utf-8'))
    U = np.load(od / 'mf_v4_units.npz')
    top, layer_of = [int(x) for x in U['top']], U['layer_of']
    off = {int(k_): v for k_, v in json.loads(WS1.MF4.read_text(encoding='utf-8'))['unit_offsets'].items()}
    tl_units = {l: [int(i - off[l]) for i in top if int(layer_of[i]) == l] for l in TL}
    sgn = {l: np.array(ws1['unit_signs'][str(l)]) for l in TL}
    # ---------- 方向（只用发现集） ----------
    z = MF1.load_fields(key)
    L = int(z['Hs'].shape[1]) - 1
    ls = round(0.5 * L)
    y, tk, sp, sh = z['style'], z['tok'], z['split'], z['shared'].astype(bool)
    D = sp == 'disc'
    vdir, V, SG = {}, {}, {}
    for l in range(1, L + 1):
        X = z['Hs'][D[sh], l].astype(np.float32)
        vdir[l] = W.unit(F.token_matched_direction(X, y[sh][D[sh]], tk[sh][D[sh]], CODE))
        if l == ls:
            for s in range(4):
                v = F.token_matched_direction(X, y[sh][D[sh]], tk[sh][D[sh]], s)
                V[s] = v / np.linalg.norm(v)
                SG[s] = float((X @ V[s]).std())
        del X
    shared_all = sh
    del z
    gc.collect()
    excl = json.loads((od / 'vocab.json').read_text(encoding='utf-8'))['excl']
    tok, model = XM.load(MODEL, False)
    R = XM.Runner(tok, model)
    C, ids = MF1.tokenize(tok)
    blocks = R.blocks
    cfg = model.config
    nh, hd = cfg.num_attention_heads, getattr(cfg, 'head_dim', cfg.hidden_size // cfg.num_attention_heads)
    t0 = time.time()
    starts = np.concatenate([[0], np.cumsum([len(x) for x in ids])])
    is_code = lambda i: STY.index(C[i]['style']) == CODE
    disc = [i for i, cc in enumerate(C) if cc['split'] == 'disc']
    conf = [i for i, cc in enumerate(C) if cc['split'] == 'conf']
    gam = {l: blocks[l].post_attention_layernorm.weight.float() for l in TL}
    S = {l: (torch.as_tensor(sgn[l], device='cuda', dtype=torch.float32)[:, None] *
             blocks[l].mlp.gate_proj.weight[tl_units[l]].float() * gam[l]).sum(0) for l in TL}
    Sm = torch.stack([S[l] for l in TL])                                                    # [3, d]
    uidx = {l: torch.as_tensor(tl_units[l], device='cuda') for l in TL}
    sg_t = {l: torch.as_tensor(sgn[l], device='cuda', dtype=torch.float32) for l in TL}
    st = {'contrib': False, 'gate': False}
    cap = {}

    def oproj_pre(lam):
        def f(m, a):
            if st['contrib']:
                zz = a[0].float().reshape(a[0].shape[0], a[0].shape[1], nh, hd)
                q = (Sm @ m.weight.float()).reshape(len(TL), nh, hd)
                q = q * torch.as_tensor([1.0 if l >= lam else 0.0 for l in TL], device='cuda')[:, None, None]
                cap[('a', lam)] = torch.einsum('bthd,khd->btkh', zz, q)
            return None
        return f

    def rms_pre(l):
        def f(m, a):
            if st['contrib']:
                cap[('r', l)] = torch.sqrt((a[0].float() ** 2).mean(-1) + m.variance_epsilon)
            return None
        return f

    def gate_post(l):
        def f(m, a, o):
            if st['gate']:
                cap[('g', l)] = (o[..., uidx[l]].float() * sg_t[l]).sum(-1)
        return f

    hooks = [blocks[lam].self_attn.o_proj.register_forward_pre_hook(oproj_pre(lam)) for lam in range(TL[-1] + 1)]
    hooks += [blocks[l].post_attention_layernorm.register_forward_pre_hook(rms_pre(l)) for l in TL]
    hooks += [blocks[l].mlp.gate_proj.register_forward_hook(gate_post(l)) for l in TL]

    def contrib(texts):
        """逐头直接贡献的「代码 − 散文」均值差 [块, 头]，以及门控得分差。"""
        acc = [np.zeros((TL[-1] + 1, nh)), np.zeros((TL[-1] + 1, nh))]
        g, n = [0.0, 0.0], [0, 0]
        st['contrib'] = st['gate'] = True
        for j in range(0, len(texts), 8):
            ch = texts[j:j + 8]
            i_, att = R._batch([ids[i] for i in ch])
            R.model.model(input_ids=i_, attention_mask=att)
            rinv = torch.stack([1.0 / cap[('r', l)] for l in TL], -1)
            for b, i in enumerate(ch):
                k = len(ids[i])
                m = torch.as_tensor(shared_all[starts[i]:starts[i] + k], device='cuda')
                c = int(is_code(i))
                ri = rinv[b, :k][m]
                for lam in range(TL[-1] + 1):
                    acc[c][lam] += (cap[('a', lam)][b, :k][m] * ri[:, :, None]).sum((0, 1)).cpu().numpy()
                g[c] += sum(cap[('g', l)][b, :k][m].sum().item() for l in TL)
                n[c] += int(m.sum().item())
        st['contrib'] = st['gate'] = False
        return acc[1] / n[1] - acc[0] / n[0], g[1] / n[1] - g[0] / n[0]

    def gate_contrast(texts):
        g, n = [0.0, 0.0], [0, 0]
        st['gate'] = True
        for j in range(0, len(texts), 16):
            ch = texts[j:j + 16]
            i_, att = R._batch([ids[i] for i in ch])
            R.model.model(input_ids=i_, attention_mask=att)
            for b, i in enumerate(ch):
                k = len(ids[i])
                m = torch.as_tensor(shared_all[starts[i]:starts[i] + k], device='cuda')
                c = int(is_code(i))
                g[c] += sum(cap[('g', l)][b, :k][m].sum().item() for l in TL)
                n[c] += int(m.sum().item())
        st['gate'] = False
        return g[1] / n[1] - g[0] / n[0]

    Td, Gd = contrib(disc)
    cand = [(lam, h) for lam in range(TL[-1]) for h in range(nh)]
    sel = sorted(cand, key=lambda x: -Td[x])[:crit['n_heads']]
    rsets = W.matched_heads(sel, nh, crit['n_random_sets'], crit['seed'])
    Tc, Gc = contrib(conf)
    print('@@sel', sel, 'disc G', round(Gd, 2), 'conf G', round(Gc, 2), f'{round(time.time() - t0)}s', flush=True)
    # ---------- 编辑与测量 ----------
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

    hooks.append(blocks[ls].register_forward_pre_hook(inj_hook, with_kwargs=True))
    groups = [tuple(g) for g in json.loads(WS1.MF4.read_text(encoding='utf-8'))['criteria']['groups']]
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
            out['%s:%s' % g] = np.array(M)
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
                out[int(is_code(i))].append(float(-lp[b, :len(ids[i]) - 1].mean()))
        return {'0': float(np.mean(out[0])), '1': float(np.mean(out[1])), 'all': float(np.mean(out[0] + out[1]))}

    def edit(heads, mode):
        saved = []
        for lam, h in heads:
            P = blocks[lam].self_attn.o_proj.weight
            sl = slice(h * hd, (h + 1) * hd)
            old = P.data[:, sl].clone()
            Vw = vdir[lam + 1][None] if mode == 'code' else np.stack([S[l].cpu().numpy() for l in TL if l >= lam])
            P.data[:, sl] = torch.as_tensor(W.remove_write(old.float().cpu().numpy(), Vw), dtype=P.dtype, device=P.device)
            saved.append((P, sl, old))
        return saved

    def restore(saved):
        for P, sl, old in reversed(saved):
            P.data[:, sl] = old

    res = {'orig': {'G': gate_contrast(conf), 'M': measure(), 'nll': nll()}}
    for name, heads, mode in [('sel_code', sel, 'code'), ('sel_gate', sel, 'gate')] + \
                             [(f'rand{j}_code', s_, 'code') for j, s_ in enumerate(rsets)]:
        sv = edit(heads, mode)
        try:
            res[name] = {'G': gate_contrast(conf), 'M': measure()}
            if name.startswith('sel'):
                res[name]['nll'] = nll()
        finally:
            restore(sv)
        print(f'@@progress {name} G {res[name]["G"]:.2f} {round(time.time() - t0)}s', flush=True)
    for h_ in hooks:
        h_.remove()
    # ---------- 描述：注意力模式（只读，确认集代码文本） ----------
    model.set_attn_implementation('eager')
    W_ = crit['locality_window']
    loc = {'sel': [], 'rand': []}
    tokmass = {hh: Counter() for hh in sel}
    rheads = [x for s_ in rsets for x in s_]
    for j in range(0, len(conf), 4):
        ch = [i for i in conf[j:j + 4] if is_code(i)]
        if not ch:
            continue
        i_, att = R._batch([ids[i] for i in ch])
        o = R.model.model(input_ids=i_, attention_mask=att, output_attentions=True)
        for b, i in enumerate(ch):
            k = len(ids[i])
            m = shared_all[starts[i]:starts[i] + k]
            for t in range(24, k):
                if not m[t]:
                    continue
                for grp, hs_ in (('sel', sel), ('rand', rheads)):
                    for lam, h in hs_:
                        row = o.attentions[lam][b, h, t, :t + 1].float().cpu().numpy()
                        loc[grp].append(W.locality(row, t, W_))
                        if grp == 'sel':
                            for p_ in np.argsort(-row[1:])[:3] + 1:
                                tokmass[(lam, h)][tok.decode([ids[i][p_]])] += float(row[p_])
        del o
    model.set_attn_implementation('sdpa')
    (NATIVE / snap[:16] / 'WS-v2').mkdir(parents=True, exist_ok=True)
    np.savez(NATIVE / snap[:16] / 'WS-v2' / 'ws_v2.npz', Td=Td, Tc=Tc)
    save({'prereg_sha256': PREREG.with_suffix('.sha256').read_text().strip(), 'model_snapshot': snap, 'precision': 'bf16',
          'selected': [list(x) for x in sel], 'random_sets': [[list(x) for x in s_] for s_ in rsets],
          'T_disc_selected': [float(Td[x]) for x in sel], 'T_conf_selected': [float(Tc[x]) for x in sel],
          'G_disc': Gd, 'G_conf_decomp': Gc, 'heads_conf_sum': float(Tc.sum()),
          'results': {k_: {'G': v['G'], 'M': {g: m.tolist() for g, m in v['M'].items()}, **({'nll': v['nll']} if 'nll' in v else {})}
                      for k_, v in res.items()},
          'locality': {g: {'mean': float(np.nanmean(v)), 'n': len(v)} for g, v in loc.items()},
          'top_tokens': {f'{lam}.{h}': [[t_, round(w_, 2)] for t_, w_ in c_.most_common(10)] for (lam, h), c_ in tokmass.items()},
          'seconds': round(time.time() - t0, 1)})
    print('done', round(time.time() - t0, 1), flush=True)


def do_analyze():
    from provenance import write_result
    crit = json.loads(PREREG.read_text(encoding='utf-8'))['criteria']
    rep = json.loads(report_path().read_text(encoding='utf-8'))
    R_ = rep['results']
    g0 = R_['orig']['G']
    red = lambda k_: 1 - R_[k_]['G'] / g0
    rnames = [k_ for k_ in R_ if k_.startswith('rand')]
    h1 = W.over_random(red('sel_code'), [red(k_) for k_ in rnames], crit['H1_over_random'], crit['H1_reduction_min'])
    groups = list(R_['orig']['M'].keys())

    def restore(k_):
        r = {}
        for g in groups:
            o, e = np.array(R_['orig']['M'][g]), np.array(R_[k_]['M'][g])
            r[g] = float((e[:, 1] - o[:, 1]).mean() / (o[:, 0] - o[:, 1]).mean())
        return float(np.mean(list(r.values()))), r
    h2 = W.over_random(restore('sel_code')[0], [restore(k_)[0] for k_ in rnames], crit['H2_over_random'], crit['H2_restore_min'])
    h2['per_group'] = restore('sel_code')[1]
    nl = R_['orig']['nll']
    rise = {k_: (R_['sel_code']['nll'][k_] - nl[k_]) / nl[k_] for k_ in ('all', '1', '0')}
    h3 = {'rel_rise_all': rise['all'], 'rel_rise_code': rise['1'], 'rel_rise_prose': rise['0'],
          'pass': bool(rise['all'] <= crit['H3_nll_rise_max'] and rise['1'] <= crit['H3_nll_rise_max'])}
    desc = {'direct_selected_conf': float(np.sum(rep['T_conf_selected'])), 'G_conf': g0,
            'direct_share': float(np.sum(rep['T_conf_selected']) / g0),
            'gate_dir_edit': {'reduction': red('sel_gate'), 'restore': restore('sel_gate')[0],
                              'nll_rise_all': (R_['sel_gate']['nll']['all'] - nl['all']) / nl['all']}}
    verdict = {'H1': h1['pass'], 'H2': h2['pass'], 'H3': h3['pass']}
    rep = save({'criteria': crit, 'H1': h1, 'H2': h2, 'H3': h3, 'describe': desc, 'verdict': verdict})
    p = write_result('ws_v2_qwen3-4b', rep, model_path=MODEL, seed=crit['seed'], tool='scripts/weight_space_v2.py')
    print(json.dumps(verdict), '\nsel', rep['selected'], '\nH1', h1, '\nH2', h2, '\nH3', h3, '\ndesc', desc,
          '\nlocality', rep['locality'], '\nprovenance →', p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['prereg', 'run', 'analyze'])
    a = ap.parse_args()
    {'prereg': do_prereg, 'run': do_run, 'analyze': do_analyze}[a.cmd]()
    return 0


if __name__ == '__main__':
    sys.exit(main())
