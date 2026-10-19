# -*- coding: utf-8 -*-
"""INT-Q-v2 预注册干预（research/interventions/INT-Q-v2.prereg.json）：多位置 / 多层的「疑问」方向充分性与必要性。

继承 INT-Q-v1 的方向、probe、材料与判据结构（scripts/run_intervention_q.py 的函数），新增：
  - 位置模式 final_only / final_and_after（句末及其后全部提示 token，不含答案候选）；
  - 必要性的逐位置基准 μ_D(k)：GR-Q discovery 陈述句套认知 probe 的无干预前向中，相对句末偏移 k 处在 d̂ 上投影的均值；
  - 按配置网格逐层运行，Bonferroni 校正后的显著性水平取自预注册。

用法：
    python scripts/run_intervention_q2.py --model qwen3-4b --out research/interventions/INT-Q-v2-qwen3-4b.json [--limit 4] [--configs 0,1]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from shared.dist_protocol import entangle as E  # noqa: E402
from shared.dist_protocol.runner_v3 import module_device, model_snapshot  # noqa: E402
import run_intervention_q as V1  # noqa: E402

PRE_V1 = ROOT / 'research/interventions/INT-Q-v1.prereg.json'
PRE_V2 = ROOT / 'research/interventions/INT-Q-v2.prereg.json'
UNC, CER = V1.UNC, V1.CER


def check_sha(p: Path):
    sha = V1.file_sha(p)
    want = p.with_suffix('.sha256').read_text(encoding='utf-8').split()[0]
    if sha != want:
        raise SystemExit(f'{p.name} 哈希 {sha} 与登记的 {want} 不符')
    return sha


class Steer:
    """在某个块的输出（= hs[layer]）上，对 positions 中每个位置加位移；fn(H) → (P, D)，H 为第 0 行在这些位置的干预前状态。"""

    def __init__(self):
        self.positions, self.fn, self.H, self.handle = None, None, None, None

    def attach(self, block):
        if self.handle is not None:
            self.handle.remove()
        self.handle = block.register_forward_hook(self.hook)

    def hook(self, mod, args, out):
        if self.fn is None:
            return None
        hs = out[0] if isinstance(out, tuple) else out
        idx = list(self.positions)
        H = hs[0, idx].float()
        self.H = H.detach().cpu()
        delta = self.fn(self.H)
        if delta is None:
            return None
        hs = hs.clone()
        hs[:, idx] = (hs[:, idx].float() + delta.to(hs.device)).to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs


def subsample(P, rng):
    def pick(rows, fam, n):
        sel = [r for r in rows if r['family'] == fam]
        if len(sel) <= n:
            return sel
        return [sel[i] for i in sorted(rng.choice(len(sel), n, replace=False))]
    return {'decl_cross': pick(P['decl_cross'], 'GR-NEG', 30) + pick(P['decl_cross'], 'KN-ISA', 30),
            'interrog_cross': pick(P['interrog_cross'], 'GR-NEG', 30) + pick(P['interrog_cross'], 'KN-ISA', 18),
            'selectivity': pick(P['selectivity'], 'KN-ISA', 30)}


def main(argv=None) -> int:
    import torch
    from scipy import stats
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=None, help='每池最多条目数（只用于自检）')
    ap.add_argument('--configs', default=None, help='只跑网格中的这些下标（自检用），如 0,1')
    args = ap.parse_args(argv)
    sha1, sha2 = check_sha(PRE_V1), check_sha(PRE_V2)
    pre1 = json.loads(PRE_V1.read_text(encoding='utf-8'))
    pre2 = json.loads(PRE_V2.read_text(encoding='utf-8'))
    cfg = pre2['configs'][args.model]
    grid = cfg['grid']
    if args.configs:
        grid = [grid[int(i)] for i in args.configs.split(',')]
    alpha = eval(cfg['alpha_per_test'].split('（')[0])  # 预注册中的 "0.01 / 3" 等
    n_rand = int(cfg['n_random'])
    t0 = time.time()

    P_all, gq, gn = V1.materials(pre1)
    if args.model == 'glm4-9b':
        P = subsample(P_all, np.random.default_rng(20261019))
    else:
        P = {k: P_all[k] for k in ('decl_cross', 'interrog_cross', 'decl_within', 'interrog_within', 'selectivity')}
    if args.limit:
        P = {k: v[:args.limit] for k, v in P.items()}
    gq_by = {it['id']: it for it in gq['items'] if not it.get('canary')}
    gn_by = {it['id']: it for it in gn['items'] if not it.get('canary')}
    # μ_D(k) 校准材料：GR-Q discovery 陈述句（套认知 probe）
    calib = [V1_gr(it, pre1) for it in gq['items'] if not it.get('canary') and it['split'] == 'discovery' and it['cond']['mood'] == 'decl']
    if args.limit:
        calib = calib[:max(args.limit, 8)]

    fq = E.NativeField(V1.native_dirs('GR-Q', 'TG-GR-Q-v7', cfg['native_key']))
    fn_ = E.NativeField(V1.native_dirs('GR-NEG', 'TG-GR-NEG-v6', cfg['native_key']))
    m = {'model_id': args.model, 'path': cfg['path'], 'placement': cfg['placement']}
    tok, model, path = V1.load_model(m)
    decoder = model.get_decoder() if hasattr(model, 'get_decoder') else model.model
    blocks = list(decoder.layers)
    steer = Steer()
    dev_in = module_device(model.get_input_embeddings()) or 'cuda'
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

    def prep(text, prompt):
        te = tok(text, return_offsets_mapping=True)
        pos = max(i for i, (a, b) in enumerate(te['offset_mapping']) if b > a)
        pids = tok(prompt)['input_ids']
        if list(pids[:pos + 1]) != list(te['input_ids'][:pos + 1]):
            return None, None
        return pids, pos

    def score(pids, positions, cands, fns):
        """每个条件单独一次前向（批 = 全部候选，与 runner_v3 相同）；fns[c](H) → (P, D) 或 None（无干预）。"""
        seqs = [tok(c, add_special_tokens=False)['input_ids'] for c in cands]
        n_p = len(pids)
        width = n_p + max(len(s) for s in seqs)
        ids = torch.full((len(cands), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(cands), width), dtype=torch.long)
        for k, s in enumerate(seqs):
            ids[k, :n_p + len(s)] = torch.tensor(list(pids) + s)
            mask[k, :n_p + len(s)] = 1
        ids, mask = ids.to(dev_in), mask.to(dev_in)
        out = np.zeros((len(fns), len(cands)))
        H0 = None
        steer.positions = positions
        for ci, f in enumerate(fns):
            steer.fn = f
            try:
                with torch.no_grad():
                    lp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), dim=-1)
            finally:
                steer.fn = None
            if ci == 0:
                H0 = steer.H
            for k, s in enumerate(seqs):
                q = torch.arange(n_p - 1, n_p - 1 + len(s), device=lp.device)
                out[ci, k] = float(lp[k, q, torch.tensor(s, device=lp.device)].sum().item())
        return out, H0

    beh = V1.gq_behavior(args.model, gq)
    # 模型级自检：零干预 g 与正式运行一致（层无关；用校准材料中的 GR-Q 条目）
    steer.attach(blocks[0])
    diffs = []
    for it in calib[:12]:
        pids, pos = prep(it['text'], it['prompt'])
        lp, _ = score(pids, [pos], [UNC, CER], [lambda H: None])
        if it['id'] in beh:
            diffs.append(abs((lp[0, 0] - lp[0, 1]) - beh[it['id']]))
    result = {'schema': 'intervention_result.v2', 'prereg_id': pre2['id'], 'prereg_sha256': sha2, 'inherits_sha256': sha1,
              'model_id': args.model, 'model_snapshot': model_snapshot(path), 'alpha_per_test': alpha, 'n_random': n_rand,
              'limit': args.limit, 'pool_sizes': {k: len(v) for k, v in P.items()},
              'self_checks': {'gq_margin_absdiff_max': max(diffs) if diffs else None, 'gq_margin_n': len(diffs)},
              'configs': []}
    T = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32)  # noqa: E731
    rngR = np.random.default_rng(int(pre1['directions']['random_controls']['seed']))

    for g in grid:
        L, mode = int(g['layer_hs']), g['positions']
        steer.attach(blocks[L - 1])
        d, _, XQ, idsQ, _, _ = V1.class_mean_diff(fq, gq_by, L, 'mood', 'interrog', 'decl')
        p, _, _, _, _, _ = V1.class_mean_diff(fn_, gn_by, L, 'polarity', 'neg', 'aff')
        nd = float(np.linalg.norm(d))
        dhat, phat = d / nd, p / np.linalg.norm(p)
        R = np.random.default_rng(int(pre1['directions']['random_controls']['seed'])).standard_normal((n_rand, d.shape[0])).astype(np.float32)
        Rhat = R / np.linalg.norm(R, axis=1, keepdims=True)
        native_final = {x: XQ[i] for i, x in enumerate(idsQ)}
        dh_t = T(dhat)

        def positions_for(pids, pos):
            return [pos] if mode == 'final_only' else list(range(pos, len(pids)))

        # μ_D(k) 校准 + 原生全场自检
        sums, cnts, cos = {}, {}, []
        for it in calib:
            pids, pos = prep(it['text'], it['prompt'])
            if pids is None:
                continue
            ps = positions_for(pids, pos)
            _, H = score(pids, ps, [UNC, CER], [lambda H: None])
            proj = (H @ dh_t).numpy()
            for k, v in enumerate(proj):
                sums[k] = sums.get(k, 0.0) + float(v)
                cnts[k] = cnts.get(k, 0) + 1
            if it['id'] in native_final:
                v = native_final[it['id']]
                h = H[0].numpy()
                cos.append(float(h @ v / (np.linalg.norm(h) * np.linalg.norm(v) + 1e-9)))
        kmax = max(sums)
        mu = np.array([sums[k] / cnts[k] for k in range(kmax + 1)])

        def mu_of(n):
            return T([mu[min(k, kmax)] for k in range(n)])

        suff_vecs = [T(d), T(p / np.linalg.norm(p) * nd)] + [T(r * nd) for r in Rhat]
        suff_names = ['base', 'd@1', 'p'] + [f'r{k}' for k in range(n_rand)]
        suff_fns = [lambda H: None] + [(lambda v: (lambda H: v.expand(H.shape[0], -1)))(v) for v in suff_vecs]
        units = [T(dhat), T(phat)] + [T(r) for r in Rhat]
        nec_names = ['base', 'abl_d', 'abl_p'] + [f'r{k}' for k in range(n_rand)]

        def nec_f(u):
            def f(H):
                mv = (H @ dh_t) - mu_of(H.shape[0])          # 每个位置在 d̂ 上相对 μ_D(k) 的偏离
                return -mv[:, None] * u[None, :]
            return f
        nec_fns = [lambda H: None] + [nec_f(u) for u in units]

        pools = {}

        def run_pool(name, names, fns):
            rows = []
            for it in P.get(name, []):
                pids, pos = prep(it['text'], it['prompt'])
                if pids is None:
                    continue
                lp, _ = score(pids, positions_for(pids, pos), [UNC, CER], fns)
                rows.append({'id': it['id'], 'family': it['family'], 'filler': it['filler'], 'g': (lp[:, 0] - lp[:, 1]).tolist()})
            pools[name] = {'conditions': names, 'items': rows}
            print(f'[int-q2] {args.model} L{L} {mode} {name}: {len(rows)} 条 · {time.time() - t0:.0f}s', flush=True)

        run_pool('decl_cross', suff_names, suff_fns)
        run_pool('interrog_cross', nec_names, nec_fns)
        if 'decl_within' in P:
            run_pool('decl_within', suff_names, suff_fns)
            run_pool('interrog_within', nec_names, nec_fns)
        sel = []
        for it in P['selectivity']:
            pids, pos = prep(it['text'], it['cat_prompt'])
            if pids is None:
                continue
            lp, _ = score(pids, positions_for(pids, pos), [it['Y']] + it['cat_contrast'], suff_fns[:2])
            sel.append({'id': it['id'], 'filler': it['filler'], 'm_cat': (lp[:, 0] - lp[:, 1:].max(1)).tolist()})
        pools['selectivity'] = {'conditions': ['base', 'd@1'], 'items': sel}
        entry = {'layer_hs': L, 'positions': mode, 'role': g['role'], 'norm_d': nd, 'cos_d_p': float(dhat @ phat),
                 'mu_D': mu.tolist(), 'native_cos_min': min(cos) if cos else None, 'native_cos_n': len(cos),
                 'pools': pools, 'evaluation': evaluate(pools, stats, alpha)}
        result['configs'].append(entry)
        ev = entry['evaluation']
        print(f'[int-q2] {args.model} L{L} {mode}: suff Δ={ev["sufficiency_primary"]["mean_delta"]["d@1"]:+.3f} '
              f'({ev["sufficiency_primary"]["verdict"]}) nec drop={ev["necessity_primary"]["mean_drop"]["abl_d"]:+.3f} '
              f'({ev["necessity_primary"]["verdict"]}) native_cos={entry["native_cos_min"]}', flush=True)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        result['seconds'] = round(time.time() - t0, 1)
        Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'self_checks': result['self_checks'],
                      'configs': [{'L': c['layer_hs'], 'pos': c['positions'], 'native_cos_min': c['native_cos_min'],
                                   **c['evaluation']['verdicts']} for c in result['configs']]}, ensure_ascii=False, indent=1))
    return 0


def V1_gr(it, pre1):
    spec_slots = _slots('GR-Q')[it['filler_id']]
    s = it['text']
    return {'id': it['id'], 'family': 'GR-Q', 'filler': it['filler_id'], 'split': it['split'], 'cond': it['cond'], 'text': s,
            'prompt': s + pre1['probes']['epistemic_GR'].replace('{text}', '').replace('{S}', spec_slots['S']).replace('{A}', spec_slots['A'])}


_SLOTS = {}


def _slots(fam):
    if fam not in _SLOTS:
        spec = json.loads((ROOT / 'shared/dist_protocol/grammars' / f'{fam}.json').read_text(encoding='utf-8'))
        _SLOTS[fam] = {f['id']: f['slots'] for f in spec['fillers']}
    return _SLOTS[fam]


def evaluate(pools, stats, alpha):
    """与 INT-Q-v1 相同的判据结构，显著性水平取预注册的 alpha_per_test。"""
    def G(name):
        return np.array([r['g'] for r in pools[name]['items']]) if pools.get(name, {}).get('items') else None

    def fsign(delta, name):
        by = {}
        for v, r in zip(delta, pools[name]['items']):
            by.setdefault((r['family'], r['filler']), []).append(v)
        means = np.array([np.mean(v) for v in by.values()])
        k = int((means > 0).sum())
        return {'n_fillers': len(means), 'n_positive': k,
                'p_sign': float(stats.binomtest(k, len(means), 0.5, alternative='greater').pvalue)}

    out = {'validity': {}, 'verdicts': {}}
    for tag, (q, dl) in {'cross': ('interrog_cross', 'decl_cross'), 'within': ('interrog_within', 'decl_within')}.items():
        gq_, gd_ = G(q), G(dl)
        if gq_ is None or gd_ is None:
            continue
        diff = float(gq_[:, 0].mean() - gd_[:, 0].mean())
        out['validity'][tag] = {'mean_g_interrog': float(gq_[:, 0].mean()), 'mean_g_decl': float(gd_[:, 0].mean()), 'diff': diff, 'valid': diff > 0}

    def test(name, tag, kind):
        g = G(name)
        names = pools[name]['conditions']
        D = (g - g[:, [0]]) if kind == 'suff' else (g[:, [0]] - g)
        means = {n: float(D[:, i].mean()) for i, n in enumerate(names)}
        key, ctrl = ('d@1', 'p') if kind == 'suff' else ('abl_d', 'abl_p')
        dd = D[:, names.index(key)]
        w = stats.wilcoxon(dd, alternative='greater') if np.any(dd != 0) else None
        rnd = [means[n] for n in names if n.startswith('r')]
        crit = [bool(means[key] > 0 and w is not None and w.pvalue < alpha), bool(means[key] > max(rnd)), bool(means[key] > means[ctrl])]
        valid = out['validity'].get(tag, {}).get('valid', False)
        return {'n': int(len(g)), ('mean_delta' if kind == 'suff' else 'mean_drop'): means,
                'wilcoxon_p': float(w.pvalue) if w else None, 'random_max': max(rnd), 'random_mean': float(np.mean(rnd)),
                'criteria': crit, 'verdict': '不判定' if not valid else ('通过' if all(crit) else '未通过'),
                'filler_robustness': fsign(dd, name), 'base_mean_g': float(g[:, 0].mean())}

    out['sufficiency_primary'] = test('decl_cross', 'cross', 'suff')
    out['necessity_primary'] = test('interrog_cross', 'cross', 'nec')
    if G('decl_within') is not None:
        out['sufficiency_secondary'] = test('decl_within', 'within', 'suff')
        out['necessity_secondary'] = test('interrog_within', 'within', 'nec')
    sel = np.array([r['m_cat'] for r in pools['selectivity']['items']])
    dcat = float((sel[:, 1] - sel[:, 0]).mean())
    dg = out['sufficiency_primary']['mean_delta']['d@1']
    out['selectivity'] = {'n': int(len(sel)), 'mean_delta_m_cat': dcat, 'limit': 0.5 * dg,
                          'verdict': ('通过' if abs(dcat) <= 0.5 * dg else '未通过') if dg > 0 else '不判定'}
    for k in ('sufficiency_primary', 'necessity_primary', 'sufficiency_secondary', 'necessity_secondary', 'selectivity'):
        if k in out:
            out['verdicts'][k] = out[k]['verdict']
    out['verdicts']['causal_use_in_scope'] = '是' if out['verdicts']['sufficiency_primary'] == '通过' and out['verdicts']['necessity_primary'] == '通过' else '否'
    return out


if __name__ == '__main__':
    raise SystemExit(main())
