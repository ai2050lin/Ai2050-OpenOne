# -*- coding: utf-8 -*-
"""INT-N-v1 预注册消融（research/interventions/INT-N-v1.prereg.json）：语气选择性 MLP 单元的必要性 / 充分性。

单元清单冻结在 research/interventions/INT-N-v1.units.json。在提示中文本最后一个 token 的位置，把所选单元送入
down_proj 的值替换为另一水平的类均值（TG-GR-Q v10 原生 MLP 中间单元，discovery，final 读出）。
每条材料：一次与 runner_v3 相同批形状的无干预前向（自检），以及把全部条件与一个无干预条件堆叠的单次前向
（效应 = 同批干预 − 同批无干预）；只在答案位置计算 lm_head。

用法：python scripts/run_intervention_n.py --model qwen3-4b --out research/interventions/INT-N-v1-qwen3-4b.json [--limit 4]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts'))

from shared.dist_protocol import neurons as N  # noqa: E402
from shared.dist_protocol.pipeline import parse_blob  # noqa: E402
from shared.dist_protocol.runner_v3 import module_device, model_snapshot  # noqa: E402
import run_intervention_q as V1  # noqa: E402

PRE = ROOT / 'research/interventions/INT-N-v1.prereg.json'
PRE_Q = ROOT / 'research/interventions/INT-Q-v1.prereg.json'
UNITS = ROOT / 'research/interventions/INT-N-v1.units.json'
NATIVE = Path(os.environ.get('AI2050_NATIVE_DIR', 'D:/ai2050_native'))
DESIGN = 'e2395ae124e0'
SEGS = ('浅', '中', '深')
UNC, CER = V1.UNC, V1.CER


def behavior_g(model_id, gq):
    """TG-GR-Q v10 正式运行的 margin 换算成 g（疑问句 m = g，陈述句 m = −g）。"""
    db = sqlite3.connect(f'file:{ROOT / "distributed_data/dist.sqlite3"}?mode=ro', uri=True)
    mood = {it['id']: it['cond']['mood'] for it in gq['items'] if not it.get('canary')}
    out = {}
    for sha, sj, ds in db.execute("SELECT sha, summary_json, design_sha FROM results WHERE tm_id='TG-GR-Q' AND model_id=? AND kind='real'", (model_id,)):
        if not ds.startswith(DESIGN):
            continue
        s = json.loads(sj)
        files = parse_blob((ROOT / 'distributed_data/objects' / sha[:2] / sha).read_bytes(), s.get('_manifest') or [])
        for b in json.loads(files['behavior.json'].decode('utf-8')).get('items', []):
            if b.get('margin') is not None and b['id'] in mood:
                out[b['id']] = b['margin'] if mood[b['id']] == 'interrog' else -b['margin']
    return out


def main(argv=None) -> int:
    import torch
    from scipy import stats
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args(argv)
    pre = json.loads(PRE.read_text(encoding='utf-8'))
    sha = V1.file_sha(PRE)
    if sha != PRE.with_suffix('.sha256').read_text(encoding='utf-8').split()[0]:
        raise SystemExit('INT-N-v1 预注册文件哈希不符')
    ub = UNITS.read_bytes()
    if hashlib.sha256(ub).hexdigest() != pre['units']['sha256_raw'] and \
            hashlib.sha256(ub.replace(b'\r\n', b'\n')).hexdigest() != pre['units']['sha256_lf']:
        raise SystemExit('单元清单哈希不符')
    units = json.loads(ub.decode('utf-8'))['models'][args.model]
    m = next(x for x in pre['models'] if x['model_id'] == args.model)
    preq = json.loads(PRE_Q.read_text(encoding='utf-8'))
    t0 = time.time()

    # ---- 目标值：v10 原生 MLP 单元，discovery，final 读出的类均值 ----
    gq10 = json.loads((ROOT / 'distributed_data/bundles/TG-GR-Q/v10/corpus.json').read_text(encoding='utf-8'))
    items10 = {it['id']: it for it in gq10['items'] if not it.get('canary')}
    dirs = sorted(glob.glob(str(NATIVE / 'GR-Q/TG-GR-Q-v10/*' / m['native_key'] / 'shard-*')))
    field = N.MlpField(dirs)
    disc = np.array([items10[i]['split'] == 'discovery' for i in field.ids])
    isq = np.array([items10[i]['cond']['mood'] == 'interrog' for i in field.ids])
    d_ff = field.d_ff
    seg_units = {s: [(u['block'], u['unit']) for u in units['segments'][s]['units']] for s in SEGS}
    rand_sets = {}
    for si, s in enumerate(SEGS):
        rng = np.random.default_rng(20261019 + si)
        per_block = {}
        for b, j in seg_units[s]:
            per_block.setdefault(b, []).append(j)
        sets = []
        for _ in range(20):
            chosen = []
            for b in sorted(per_block):
                pool = np.setdiff1d(np.arange(d_ff), np.array(per_block[b]))
                chosen += [(b, int(j)) for j in rng.choice(pool, len(per_block[b]), replace=False)]
            sets.append(chosen)
        rand_sets[s] = sets
    need = {}
    for s in SEGS:
        for b, j in seg_units[s] + [x for st in rand_sets[s] for x in st]:
            need.setdefault(b, set()).add(j)
    means = {}            # (block, unit) → {'decl': ..., 'interrog': ...}
    native_A = {}         # 自检：GR-Q 条目在这些块的 A
    for b in sorted(need):
        G, V = field.GV('final', b)
        A = N.silu(G) * V
        for j in need[b]:
            means[(b, j)] = {'decl': float(A[disc & ~isq, j].mean()), 'interrog': float(A[disc & isq, j].mean())}
        native_A[b] = {i: A[k] for k, i in enumerate(field.ids)}

    # ---- 材料与模型 ----
    P, _, _ = V1.materials(preq)
    if args.limit:
        P = {k: v[:args.limit] for k, v in P.items()}
    tok, model, path = V1.load_model({'model_id': args.model, 'path': m['path'], 'placement': m['placement'].split('（')[0].strip()})
    decoder = model.get_decoder() if hasattr(model, 'get_decoder') else model.model
    blocks = list(decoder.layers)
    lm_head = model.get_output_embeddings()
    head_dev = module_device(lm_head) or 'cuda'
    dev_in = module_device(model.get_input_embeddings()) or 'cuda'
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    ctx = {'on': False, 'pos': None, 'plan': {}, 'cap': {}}

    def make_hook(b):
        def pre_hook(mod, hargs):
            if not ctx['on']:
                return None
            x = hargs[0]
            p = ctx['pos']
            ctx['cap'][b] = x[0, p].detach().float().cpu()
            spec = ctx['plan'].get(b)
            if spec is None:
                return None
            rows, cols, vals = spec
            x = x.clone()
            x[rows.to(x.device), p, cols.to(x.device)] = vals.to(x.device, x.dtype)
            return (x,) + tuple(hargs[1:])
        return pre_hook
    for b in sorted(need):
        blocks[b].mlp.down_proj.register_forward_pre_hook(make_hook(b))

    def prep(text, prompt):
        te = tok(text, return_offsets_mapping=True)
        pos = max(i for i, (a, c) in enumerate(te['offset_mapping']) if c > a)
        pids = tok(prompt)['input_ids']
        if list(pids[:pos + 1]) != list(te['input_ids'][:pos + 1]):
            return None, None
        return pids, pos

    def batch(pids, cands, n_cond):
        seqs = [tok(c, add_special_tokens=False)['input_ids'] for c in cands]
        n_p = len(pids)
        width = n_p + max(len(s) for s in seqs)
        R = n_cond * len(cands)
        ids = torch.full((R, width), pad_id, dtype=torch.long)
        mask = torch.zeros((R, width), dtype=torch.long)
        for c in range(n_cond):
            for k, s in enumerate(seqs):
                r = c * len(cands) + k
                ids[r, :n_p + len(s)] = torch.tensor(list(pids) + s)
                mask[r, :n_p + len(s)] = 1
        return ids, mask, seqs, n_p

    def runner_base(pids, pos, cands):
        """与 runner_v3.answer_logprobs 相同：候选单独成批、完整 logits；同时捕获 down_proj 输入（自检）。"""
        ids, mask, seqs, n_p = batch(pids, cands, 1)
        ctx.update(on=True, pos=pos, plan={}, cap={})
        try:
            with torch.no_grad():
                lp = torch.log_softmax(model(input_ids=ids.to(dev_in), attention_mask=mask.to(dev_in)).logits.float(), dim=-1)
        finally:
            ctx['on'] = False
        out = [float(lp[k, torch.arange(n_p - 1, n_p - 1 + len(s), device=lp.device), torch.tensor(s, device=lp.device)].sum())
               for k, s in enumerate(seqs)]
        return np.array(out), dict(ctx['cap'])

    def stacked(pids, pos, cands, conds):
        """conds：每个条件为 [(block, unit, value)]（空表 = 无干预）；返回 (条件数, 候选数) 的对数概率。"""
        ids, mask, seqs, n_p = batch(pids, cands, len(conds))
        plan = {}
        for c, spec in enumerate(conds):
            for b, j, v in spec:
                for k in range(len(cands)):
                    plan.setdefault(b, ([], [], []))
                    plan[b][0].append(c * len(cands) + k); plan[b][1].append(j); plan[b][2].append(v)
        ctx.update(on=True, pos=pos, cap={},
                   plan={b: (torch.tensor(r), torch.tensor(c_), torch.tensor(v)) for b, (r, c_, v) in plan.items()})
        try:
            with torch.no_grad():
                h = decoder(input_ids=ids.to(dev_in), attention_mask=mask.to(dev_in)).last_hidden_state
        finally:
            ctx['on'] = False
        out = np.zeros((len(conds), len(cands)))
        for c in range(len(conds)):
            for k, s in enumerate(seqs):
                r = c * len(cands) + k
                q = torch.arange(n_p - 1, n_p - 1 + len(s), device=h.device)
                lg = lm_head(h[r, q].to(head_dev)).float()
                lp = torch.log_softmax(lg, dim=-1)
                out[c, k] = float(lp[torch.arange(len(s)), torch.tensor(s, device=lp.device)].sum())
        return out

    def conds_for(level):
        cs, names = [[]], ['base']
        for s in SEGS:
            cs.append([(b, j, means[(b, j)][level]) for b, j in seg_units[s]]); names.append(f'{s}:target')
            for k, st in enumerate(rand_sets[s]):
                cs.append([(b, j, means[(b, j)][level]) for b, j in st]); names.append(f'{s}:r{k}')
        return cs, names

    beh = behavior_g(args.model, gq10)
    checks = {'g_runner_vs_behavior': [], 'down_input_rel_err': [], 'stacked_vs_runner_base': []}
    pools = {}

    def run_pool(name, level):
        conds, names = conds_for(level)
        rows = []
        for it in P[name]:
            pids, pos = prep(it['text'], it['prompt'])
            if pids is None:
                continue
            lp_r, cap = runner_base(pids, pos, [UNC, CER])
            g_r = float(lp_r[0] - lp_r[1])
            if it['id'] in beh:
                checks['g_runner_vs_behavior'].append(abs(g_r - beh[it['id']]))
            if it['id'] in native_A[next(iter(native_A))]:
                for b, v in cap.items():
                    ref = native_A[b][it['id']]
                    checks['down_input_rel_err'].append(float(np.linalg.norm(v.numpy() - ref) / (np.linalg.norm(ref) + 1e-9)))
            lp = stacked(pids, pos, [UNC, CER], conds)
            g = lp[:, 0] - lp[:, 1]
            checks['stacked_vs_runner_base'].append(float(g[0] - g_r))
            rows.append({'id': it['id'], 'family': it['family'], 'filler': it['filler'], 'g_runner_base': g_r, 'g': g.tolist()})
        pools[name] = {'conditions': names, 'level': level, 'items': rows}
        print(f'[int-n] {args.model} {name}: {len(rows)} 条 · {time.time() - t0:.0f}s', flush=True)

    run_pool('interrog_cross', 'decl')
    run_pool('interrog_within', 'decl')
    run_pool('decl_cross', 'interrog')
    run_pool('decl_within', 'interrog')
    sel_rows = []
    sconds = [[]] + [[(b, j, means[(b, j)]['interrog']) for b, j in seg_units[s]] for s in SEGS]
    for it in P['selectivity']:
        pids, pos = prep(it['text'], it['cat_prompt'])
        if pids is None:
            continue
        lp = stacked(pids, pos, [it['Y']] + it['cat_contrast'], sconds)
        sel_rows.append({'id': it['id'], 'm_cat': (lp[:, 0] - lp[:, 1:].max(1)).tolist()})
    pools['selectivity'] = {'conditions': ['base'] + [f'{s}:target' for s in SEGS], 'items': sel_rows}

    def q(a):
        a = np.array(a)
        return {'n': int(len(a)), 'max': float(a.max()) if len(a) else None, 'median': float(np.median(a)) if len(a) else None}
    out = {'schema': 'intervention_result.n1', 'prereg_id': pre['id'], 'prereg_sha256': sha, 'model_id': args.model,
           'model_snapshot': model_snapshot(path), 'limit': args.limit, 'units': {s: seg_units[s] for s in SEGS},
           'self_checks': {'g_runner_vs_behavior': q(checks['g_runner_vs_behavior']), 'down_input_rel_err': q(checks['down_input_rel_err']),
                           'stacked_vs_runner_base': {'n': len(checks['stacked_vs_runner_base']),
                                                      'abs_median': float(np.median(np.abs(checks['stacked_vs_runner_base']))) if checks['stacked_vs_runner_base'] else None,
                                                      'abs_max': float(np.max(np.abs(checks['stacked_vs_runner_base']))) if checks['stacked_vs_runner_base'] else None}},
           'pools': pools, 'seconds': round(time.time() - t0, 1)}
    out['evaluation'] = evaluate(pools, stats, pre)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'self_checks': out['self_checks'], 'verdicts': out['evaluation']['verdicts']}, ensure_ascii=False, indent=1))
    return 0


def evaluate(pools, stats, pre):
    alpha = 0.01 / 3
    E = {'validity': {}, 'segments': {}, 'verdicts': {}}

    def G(name):
        return np.array([r['g'] for r in pools[name]['items']])
    for tag, (qn, dn) in {'cross': ('interrog_cross', 'decl_cross'), 'within': ('interrog_within', 'decl_within')}.items():
        diff = float(G(qn)[:, 0].mean() - G(dn)[:, 0].mean())
        E['validity'][tag] = {'diff': diff, 'valid': diff > 0}

    def test(name, s, kind, tag):
        g = G(name)
        names = pools[name]['conditions']
        D = (g[:, [0]] - g) if kind == 'nec' else (g - g[:, [0]])
        ti = names.index(f'{s}:target')
        dd = D[:, ti]
        rnd = [float(D[:, i].mean()) for i, n in enumerate(names) if n.startswith(f'{s}:r')]
        w = stats.wilcoxon(dd, alternative='greater') if np.any(dd != 0) else None
        crit = [bool(dd.mean() > 0 and w is not None and w.pvalue < alpha), bool(dd.mean() > max(rnd))]
        return {'n': int(len(dd)), 'mean': float(dd.mean()), 'wilcoxon_p': float(w.pvalue) if w else None,
                'random_max': max(rnd), 'random_mean': float(np.mean(rnd)),
                'ratio_to_random_mean': float(dd.mean() / np.mean(rnd)) if np.mean(rnd) > 0 else None,
                'criteria': crit, 'verdict': '不判定' if not E['validity'][tag]['valid'] else ('通过' if all(crit) else '未通过')}
    sel = np.array([r['m_cat'] for r in pools['selectivity']['items']])
    for si, s in enumerate(SEGS):
        e = {'necessity_primary': test('interrog_cross', s, 'nec', 'cross'), 'necessity_secondary': test('interrog_within', s, 'nec', 'within'),
             'sufficiency_primary': test('decl_cross', s, 'suff', 'cross'), 'sufficiency_secondary': test('decl_within', s, 'suff', 'within')}
        dcat = float((sel[:, si + 1] - sel[:, 0]).mean())
        dg = e['sufficiency_primary']['mean']
        e['selectivity'] = {'mean_delta_m_cat': dcat, 'limit': 0.5 * dg,
                            'verdict': ('通过' if abs(dcat) <= 0.5 * dg else '未通过') if dg > 0 else '不判定'}
        E['segments'][s] = e
        E['verdicts'][s] = {k: v['verdict'] for k, v in e.items()}
        E['verdicts'][s]['causal_use_in_scope'] = '是' if e['necessity_primary']['verdict'] == '通过' and e['sufficiency_primary']['verdict'] == '通过' else '否'
    return E


if __name__ == '__main__':
    raise SystemExit(main())
