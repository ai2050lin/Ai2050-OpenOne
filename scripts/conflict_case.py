# -*- coding: utf-8 -*-
"""CF-v1：上下文与记忆冲突在 Qwen3-4B 上的完整案例（目标一：注意力的模式路由与优先级；目标二：相关神经元）。

材料：shared/dist_protocol/conflict/CF-v1.json。每个事实（国家 → 首都）配一个反事实首都；冲突提示先给反事实陈述再问同一事实，
线索（cue）改变陈述的可信度、重复次数或语义（否定对照）。
指标：d = log P(上下文答案) − log P(记忆答案)（完整答案对数概率）；d > 0 记为上下文胜。

预注册（写于运行之前，判据在 CRITERIA，随结果写出）：
  0. 行为门：无上下文时 log P(记忆答案) > log P(反事实答案)，未通过的事实不进入分析。
  1. 赛跑曲线：最后位置逐层 logit lens（中间层先过最终 RMSNorm），首 token 的上下文 − 记忆差；决出层 = 之后各层与最终同号的最小层。
  2. 逐头直接作用（DLA）：头输出 / rms(最终残差) ⊙ γ · (W_U[c] − W_U[m])；逐头总效应：该头在全部位置换成均值后首 token 差的变化。
     两者并列报告，并报告前 20 的重叠（C1 教训：单头消融会被下游补偿误导）。
  3. 头电路（只用 discovery 选头，heldout 上检验）：按 discovery 上平均 DLA（朝上下文方向为正）取前 K 个头。
     必要性：在全部位置均值消融这 K 个头 → heldout 上平均 d 的下降比例，对照 R 组随机 K 头；
     充分性（最后位置）：只在最后位置把其余全部头换成均值 → heldout 上首 token 差的保留比例。
  判据：「头电路必要」当且仅当下降比例 ≥ min_drop 且大于全部随机组；「最后位置充分」当且仅当保留比例 ≥ min_keep。
  4. 线索优先级表：每个 cue 的上下文胜率、平均 d、决出层中位数、该 cue 下 DLA 前 5 的头；
     否定对照（negated）：若上下文胜率仍 ≥ 0.5，记为「复制而非理解」的候选证据。
  5. 神经元：最后位置逐单元 DLA（A_i · W_down[:, i] 经同一读出），discovery 选前 30，heldout 同号复现比例；DLA 闭合检查
     （嵌入 + 全部头 + 全部 MLP 的直接作用之和 ≈ 实际首 token 差）。
用法：python scripts/conflict_case.py --model models/hf/qwen3-4b --out research/mechanism/CF-v1-qwen3-4b
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CRITERIA = {'schema': 'conflict_case.v1', 'K': 20, 'R': 5, 'min_drop': 0.25, 'min_keep': 0.5, 'top_neurons': 30,
            'seed': 0, 'split_threshold': 154}


def split_of(set_id, subject):
    return 'discovery' if hashlib.sha256((set_id + subject).encode('utf-8')).digest()[0] < CRITERIA['split_threshold'] else 'heldout'


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='models/hf/qwen3-4b')
    ap.add_argument('--set', default='shared/dist_protocol/conflict/CF-v1.json')
    ap.add_argument('--out', default='research/mechanism/CF-v1-qwen3-4b')
    args = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from shared.dist_protocol.census_runner import model_snapshot
    t0 = time.time()
    torch.set_grad_enabled(False)
    spec = json.loads(Path(args.set).read_text(encoding='utf-8'))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map='cuda').eval()
    cfg = model.config
    NL, NH, HD = cfg.num_hidden_layers, cfg.num_attention_heads, cfg.head_dim
    layers = model.model.layers
    WU = model.lm_head.weight
    gamma = model.model.norm.weight.float()
    enc = lambda s: tok(s, add_special_tokens=False).input_ids

    # ---- 材料 ----
    facts = spec['facts']
    first = lambda s: enc(s)[0]
    for i, f in enumerate(facts):
        k = 7
        while first(facts[(i + k) % len(facts)]['answer']) == first(f['answer']):
            k += 1
        f['cf'] = facts[(i + k) % len(facts)]['answer']
        f['split'] = split_of(spec['set_id'], f['subject'])

    def prompt(f, cue):
        st = spec['statement'].format(S=f['subject'], O=f['cf'])
        neg = spec['negated_statement'].format(S=f['subject'], O=f['cf'])
        q = spec['query'].format(S=f['subject'])
        return spec['cues'][cue].replace('{STMT}', st).replace('{NEG}', neg).replace('{QUERY}', q)

    @torch.no_grad()
    def ans_lp(p, a):
        pi, ai = enc(p), enc(a)
        lg = model(input_ids=torch.as_tensor([pi + ai], device='cuda')).logits[0].float()
        lp = torch.log_softmax(lg[len(pi) - 1:-1], -1)
        return float(lp.gather(-1, torch.as_tensor(ai, device='cuda')[:, None]).sum())

    def d_full(p, f):
        return ans_lp(p, f['cf']) - ans_lp(p, f['answer'])

    gate = []
    for f in facts:
        q = spec['query'].format(S=f['subject'])
        f['gate_margin'] = -d_full(q, f)
        f['gated'] = f['gate_margin'] > 0
        gate.append(f['gated'])
    G = [f for f in facts if f['gated']]

    # ---- 批量前向工具（右侧补齐，取每条最后一个真实位置）----
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    def batch(prompts):
        ids = [enc(p) for p in prompts]
        T = max(map(len, ids))
        x = torch.full((len(ids), T), pad, dtype=torch.long)
        m = torch.zeros((len(ids), T), dtype=torch.long)
        for i, s in enumerate(ids):
            x[i, :len(s)] = torch.as_tensor(s); m[i, :len(s)] = 1
        return x.cuda(), m.cuda(), torch.as_tensor([len(s) - 1 for s in ids], device='cuda')

    cap = {'z': {}, 'A': {}, 'xf': None}
    ctl = {'abl': {}, 'where': 'all', 'last': None, 'mean_all': None, 'mean_last': None, 'capture': False}
    def opre(mod, a, l):
        z = a[0]
        if ctl['capture']:
            cap['z'][l] = z
        hs = ctl['abl'].get(l)
        if hs:
            z = z.clone()
            for h in hs:
                sl = slice(h * HD, (h + 1) * HD)
                if ctl['where'] == 'all':
                    z[:, :, sl] = ctl['mean_all'][l][sl].to(z.dtype)
                else:
                    b = torch.arange(z.shape[0], device=z.device)
                    z[b, ctl['last'], sl] = ctl['mean_last'][l][sl].to(z.dtype)
            return (z,)
    def dpre(mod, a, l):
        if ctl['capture']:
            cap['A'][l] = a[0]
    hooks = []
    for l, blk in enumerate(layers):
        hooks.append(blk.self_attn.o_proj.register_forward_pre_hook(lambda m, a, l=l: opre(m, a, l)))
        hooks.append(blk.mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: dpre(m, a, l)))
    hooks.append(model.model.norm.register_forward_pre_hook(lambda m, a: cap.__setitem__('xf', a[0]) if ctl['capture'] else None))

    def toks2(fs):
        return (torch.as_tensor([first(f['cf']) for f in fs], device='cuda'), torch.as_tensor([first(f['answer']) for f in fs], device='cuda'))

    @torch.no_grad()
    def run(fs, cue, capture=False, hidden=False):
        x, m, last = batch([prompt(f, cue) for f in fs])
        ctl['capture'], ctl['last'] = capture, last
        out = model(input_ids=x, attention_mask=m, output_hidden_states=hidden)
        ctl['capture'] = False
        b = torch.arange(len(fs), device='cuda')
        c, mm = toks2(fs)
        lg = out.logits[b, last].float()
        return out, x, m, last, (lg.gather(1, c[:, None]) - lg.gather(1, mm[:, None]))[:, 0]

    # ---- 1/4. 每个 cue：完整答案 d、赛跑曲线、DLA ----
    def readout_vec(fs, last):
        xf = cap['xf'][torch.arange(len(fs), device='cuda'), last].float()
        rms = xf.pow(2).mean(-1, keepdim=True).add(cfg.rms_norm_eps).sqrt()
        c, mm = toks2(fs)
        return (WU[c].float() - WU[mm].float()) * gamma / rms             # [n, d]

    def head_dla(fs, last, v):
        b = torch.arange(len(fs), device='cuda')
        out = torch.zeros(len(fs), NL, NH, device='cuda')
        for l in range(NL):
            z = cap['z'][l][b, last].float()                            # [n, NH*HD]
            Wo = layers[l].self_attn.o_proj.weight.float()              # [d, NH*HD]
            for h in range(NH):
                sl = slice(h * HD, (h + 1) * HD)
                out[:, l, h] = ((z[:, sl] @ Wo[:, sl].T) * v).sum(-1)
        return out

    def neuron_dla(fs, last, v):
        b = torch.arange(len(fs), device='cuda')
        out = []
        for l in range(NL):
            A = cap['A'][l][b, last].float()                            # [n, d_ff]
            u = v @ layers[l].mlp.down_proj.weight.float()              # [n, d_ff]
            out.append(A * u)
        return torch.stack(out, 1)                                      # [n, NL, d_ff]

    cues = list(spec['cues'])
    per_cue, store = {}, {}
    for cue in cues:
        d = [d_full(prompt(f, cue), f) for f in G]
        out, x, m, last, d1 = run(G, cue, capture=True, hidden=True)
        b = torch.arange(len(G), device='cuda')
        c, mm = toks2(G)
        race = []
        with torch.no_grad():
            for l, hsl in enumerate(out.hidden_states):
                h = hsl[b, last]
                h = h if l == NL else model.model.norm(h)
                rc = (h.float() * (WU[c].float() - WU[mm].float())).sum(-1)
                race.append(rc.cpu().numpy())
        race = np.stack(race, 1)                                        # [n, NL+1]
        dec = []
        for r in race:
            s = np.sign(r[-1]); k = len(r) - 1
            while k > 0 and np.sign(r[k - 1]) == s:
                k -= 1
            dec.append(k)
        v = readout_vec(G, last)
        hd = head_dla(G, last, v)
        store[cue] = {'d': np.array(d), 'race': race, 'decisive': np.array(dec), 'head_dla': hd.cpu().numpy(), 'd_first': d1.cpu().numpy()}
        if cue == 'plain':
            nd = neuron_dla(G, last, v)
            emb = (out.hidden_states[0][b, last].float() * v).sum(-1)
            closure = (emb + hd.sum((1, 2)) + nd.sum((1, 2))).cpu().numpy()
            store[cue]['neuron_dla'] = nd.cpu().numpy()
            store[cue]['closure'] = closure
            # 均值（供消融）：全部真实位置 / 最后位置
            ctl['mean_all'] = {l: (cap['z'][l].float() * m[..., None]).sum((0, 1)) / m.sum() for l in range(NL)}
            ctl['mean_last'] = {l: cap['z'][l][b, last].float().mean(0) for l in range(NL)}
        cap['z'], cap['A'] = {}, {}
        hdm = store[cue]['head_dla'].mean(0).reshape(-1)
        top5 = np.argsort(-hdm)[:5]
        per_cue[cue] = {'n': len(G), 'context_win_rate': float(np.mean(np.array(d) > 0)), 'mean_d': float(np.mean(d)),
                        'decisive_layer_median': float(np.median(dec)),
                        'race_mean': race.mean(0).tolist(),
                        'top5_heads_dla': [{'head': f'{i // NH}.{i % NH}', 'dla': float(hdm[i])} for i in top5]}
        print('@@cue', cue, json.dumps({k: per_cue[cue][k] for k in ('context_win_rate', 'mean_d', 'decisive_layer_median')}), flush=True)

    # ---- 2/3. 头：总效应、选头、必要性与充分性 ----
    disc = [i for i, f in enumerate(G) if f['split'] == 'discovery']
    held = [i for i, f in enumerate(G) if f['split'] == 'heldout']
    Gd, Gh = [G[i] for i in disc], [G[i] for i in held]
    dla_d = store['plain']['head_dla'][disc].mean(0)                    # [NL, NH]
    _, _, _, _, base_d = run(Gd, 'plain')
    tot = np.zeros((NL, NH))
    for l in range(NL):
        for h in range(NH):
            ctl.update(abl={l: [h]}, where='all')
            _, _, _, _, dd = run(Gd, 'plain')
            tot[l, h] = float((dd - base_d).mean())                     # 负 = 去掉后上下文优势变小
    ctl['abl'] = {}
    K, R = CRITERIA['K'], CRITERIA['R']
    names = [f'{l}.{h}' for l in range(NL) for h in range(NH)]
    top = np.argsort(-dla_d.reshape(-1))[:K]
    top_eff = np.argsort(tot.reshape(-1))[:K]
    def as_abl(idx):
        a = {}
        for i in idx:
            a.setdefault(int(i) // NH, []).append(int(i) % NH)
        return a
    _, _, _, _, hb = run(Gh, 'plain')
    d_h_full = np.array([d_full(prompt(f, 'plain'), f) for f in Gh])
    def abl_full(idx, where):
        ctl.update(abl=as_abl(idx), where=where)
        r = np.array([d_full(prompt(f, 'plain'), f) for f in Gh])
        ctl['abl'] = {}
        return r
    drop = lambda r: float((d_h_full.mean() - r.mean()) / d_h_full.mean())
    nec_top = drop(abl_full(top, 'all'))
    rng = np.random.default_rng(CRITERIA['seed'])
    nec_rand = [drop(abl_full(rng.choice(NL * NH, K, replace=False), 'all')) for _ in range(R)]
    rest = np.setdiff1d(np.arange(NL * NH), top)
    ctl.update(abl=as_abl(rest), where='last')
    _, _, _, _, hs_ = run(Gh, 'plain')
    ctl['abl'] = {}
    keep = float(hs_.mean() / hb.mean())
    heads = {
        'dla_discovery': dla_d.tolist(), 'total_effect_discovery': tot.tolist(),
        'top_dla': [{'head': names[i], 'dla': float(dla_d.reshape(-1)[i]), 'total_effect': float(tot.reshape(-1)[i])} for i in top],
        'top_total_effect': [{'head': names[i], 'total_effect': float(tot.reshape(-1)[i]), 'dla': float(dla_d.reshape(-1)[i])} for i in top_eff],
        'overlap_top_dla_vs_total': int(len(set(top) & set(top_eff))),
        'spearman_dla_vs_total': float(np.corrcoef(np.argsort(np.argsort(dla_d.reshape(-1))), np.argsort(np.argsort(-tot.reshape(-1))))[0, 1]),
        'necessity': {'drop_top': nec_top, 'drop_random': nec_rand,
                      'necessary': bool(nec_top >= CRITERIA['min_drop'] and nec_top > max(nec_rand))},
        'sufficiency_last_position': {'keep_fraction': keep, 'sufficient': bool(keep >= CRITERIA['min_keep'])},
        'heldout_mean_d': float(d_h_full.mean()),
    }

    # ---- v2（看到 v1 结果后、运行之前写定；v1 的必要性 / 充分性按平均 d 归一化，平均 d < 0 时符号失效）----
    # 三组头只用 discovery（plain）选出：C = DLA 最正的 K 个（直接写上下文答案）；M = DLA 最负的 K 个（直接写记忆答案）；
    # U = 总效应最负的 K 个（去掉后更偏记忆，按定义在上游传递上下文）。heldout 上，对 plain 与 repeat2 两个 cue，
    # Δ = 平均 d（消融后）− 平均 d（完整），预期 Δ_C < 0、Δ_U < 0、Δ_M > 0。
    # 某组「有作用」当且仅当 Δ 符号符合预期且 |Δ| 大于全部 R 组随机 K 头的 |Δ|。
    # 层级检验：只消融 U，看 heldout plain 上 C 组 DLA 之和的变化；下降 ≥ hier_min 记为「U → C 的上下游关系成立」。
    CRIT2 = {'schema': 'conflict_case.v2', 'K': K, 'R': R, 'cues': ['plain', 'repeat2'], 'hier_min': 0.25}
    groups = {'C_context_writers': top, 'M_memory_writers': np.argsort(dla_d.reshape(-1))[:K], 'U_upstream': top_eff}
    expect = {'C_context_writers': -1, 'M_memory_writers': 1, 'U_upstream': -1}
    rsets = [rng.choice(NL * NH, K, replace=False) for _ in range(R)]
    def mean_d(cue, idx=None):
        if idx is not None:
            ctl.update(abl=as_abl(idx), where='all')
        r = float(np.mean([d_full(prompt(f, cue), f) for f in Gh]))
        ctl['abl'] = {}
        return r
    v2 = {'criteria': CRIT2, 'groups': {g: [names[i] for i in idx] for g, idx in groups.items()}, 'by_cue': {}}
    for cue in CRIT2['cues']:
        base = mean_d(cue)
        rnd = [mean_d(cue, idx) - base for idx in rsets]
        row = {'heldout_mean_d': base, 'random_delta': rnd}
        for g, idx in groups.items():
            dlt = mean_d(cue, idx) - base
            row[g] = {'delta': dlt, 'effect': bool(np.sign(dlt) == expect[g] and abs(dlt) > max(abs(x) for x in rnd))}
        v2['by_cue'][cue] = row
    def c_dla(idx=None):
        if idx is not None:
            ctl.update(abl=as_abl(idx), where='all')
        _, _, _, last_h, _ = run(Gh, 'plain', capture=True)
        ctl['abl'] = {}
        hdl = head_dla(Gh, last_h, v_clean)
        cap['z'], cap['A'] = {}, {}
        return float(hdl.reshape(len(Gh), -1)[:, groups['C_context_writers']].sum(1).mean())
    _, _, _, last_h, _ = run(Gh, 'plain', capture=True)
    v_clean = readout_vec(Gh, last_h)
    cap['z'], cap['A'] = {}, {}
    c0, c1 = c_dla(), c_dla(groups['U_upstream'])
    c_rand = [c_dla(idx) for idx in rsets]
    v2['hierarchy'] = {'C_dla_clean': c0, 'C_dla_after_U_ablation': c1, 'drop': (c0 - c1) / c0,
                       'C_dla_after_random': c_rand, 'holds': bool((c0 - c1) / c0 >= CRIT2['hier_min'] and c1 < min(c_rand))}

    # ---- 5. 神经元 ----
    nd = store['plain']['neuron_dla']
    nd_d, nd_h = nd[disc].mean(0), nd[held].mean(0)
    flat = np.argsort(-np.abs(nd_d).reshape(-1))[:CRITERIA['top_neurons']]
    dff = nd.shape[2]
    neurons = {'top': [{'layer': int(i // dff), 'unit': int(i % dff), 'dla_discovery': float(nd_d.reshape(-1)[i]),
                        'dla_heldout': float(nd_h.reshape(-1)[i])} for i in flat],
               'sign_replication_heldout': float(np.mean(np.sign(nd_d.reshape(-1)[flat]) == np.sign(nd_h.reshape(-1)[flat]))),
               'mlp_total_by_layer': nd.sum(2).mean(0).tolist(), 'attn_total_by_layer': store['plain']['head_dla'].sum(2).mean(0).tolist()}
    closure = {'actual_first_token_d_mean': float(store['plain']['d_first'].mean()), 'dla_sum_mean': float(store['plain']['closure'].mean()),
               'max_abs_gap': float(np.abs(store['plain']['closure'] - store['plain']['d_first']).max())}
    share = {'attention': float(store['plain']['head_dla'].sum((1, 2)).mean()), 'mlp': float(nd.sum((1, 2)).mean())}
    for hk in hooks:
        hk.remove()

    neg = per_cue.get('negated', {})
    report = {
        'criteria': CRITERIA, 'set': spec['set_id'], 'set_sha256': hashlib.sha256(Path(args.set).read_bytes()).hexdigest(),
        'model': args.model, 'model_snapshot': model_snapshot(args.model)['sha256'], 'dtype': 'bfloat16 (readouts in float32)',
        'n_facts': len(facts), 'n_gated': len(G), 'n_discovery': len(Gd), 'n_heldout': len(Gh),
        'gate': [{'subject': f['subject'], 'answer': f['answer'], 'counterfactual': f['cf'], 'split': f['split'],
                  'gate_margin': f['gate_margin'], 'gated': f['gated']} for f in facts],
        'priority_table': per_cue,
        'negated_control': {'context_win_rate': neg.get('context_win_rate'),
                            'copying_candidate': bool(neg.get('context_win_rate', 0) >= 0.5)},
        'heads': heads, 'v2': v2, 'neurons': neurons, 'dla_closure': closure, 'component_share_plain': share,
        'examples': [{'cue': c, 'prompt': prompt(G[0], c), 'context_answer': G[0]['cf'], 'memory_answer': G[0]['answer']} for c in cues],
        'n_layers': NL, 'n_heads': NH, 'torch': torch.__version__, 'seconds': round(time.time() - t0, 1),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding='utf-8')
    np.savez_compressed(out / 'arrays.npz', **{f'{c}_race': store[c]['race'] for c in cues},
                        **{f'{c}_head_dla': store[c]['head_dla'] for c in cues}, head_total_effect=tot)
    print(json.dumps({'gated': f'{len(G)}/{len(facts)}', 'necessity': heads['necessity'], 'sufficiency': heads['sufficiency_last_position'],
                      'overlap': heads['overlap_top_dla_vs_total'], 'v2': {c: {g: v for g, v in r.items() if g.startswith(('C_', 'M_', 'U_'))} for c, r in v2['by_cue'].items()}, 'hier': v2['hierarchy'], 'closure': closure, 'share': share,
                      'neuron_rep': neurons['sign_replication_heldout']}, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
