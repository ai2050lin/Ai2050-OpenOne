# -*- coding: utf-8 -*-
"""SP-v1 特异性实验运行器（判据与材料在 shared/dist_protocol/mech_spec.py）。

  prepare  只用 MB-v1 的 discovery 划分与定义窗口：主干、迭代移除、单节点加回，生成排序
  eval     --split confirm（已暴露，只做开发核对）或 sealed（全新材料，只评测一次）
  report   逐模型判定
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
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tools'))
import mech_bench_run as R  # noqa: E402
from shared.dist_protocol import mech_bench as B  # noqa: E402
from shared.dist_protocol import mech_spec as S  # noqa: E402
from shared.dist_protocol.census_runner import model_snapshot  # noqa: E402

OUT = ROOT / 'research' / 'mech_bench' / 'SP-v1'
MB = ROOT / 'research' / 'mech_bench' / 'MB-v1'


def code_hash():
    parts = [Path(S.__file__), Path(__file__), Path(B.__file__), Path(R.__file__)]
    return hashlib.sha256(b''.join(p.read_bytes() for p in parts)).hexdigest()[:16]


def windows(tok, lo, hi):
    """定义窗口取语料末尾倒数 8 万到 4 万 token，评测窗口取最后 4 万，互不重叠。"""
    from transformers import AutoTokenizer
    src = AutoTokenizer.from_pretrained(R.TOK_FOR_NATURAL)
    n, seq = S.CRITERIA['natural_windows'], S.CRITERIA['natural_seq']
    out = []
    for name in ('wikipedia-en', 'fineweb-edu-en'):
        stream = np.load(f'{R.NATURAL}/tokens_{name}.npz')['tokens']
        ids = tok.encode(src.decode(stream[lo:hi].tolist()), add_special_tokens=False)
        out += [ids[i * seq:(i + 1) * seq] for i in range(n // 2)]
    return torch.tensor(out)


@torch.no_grad()
def natural_means(eng, wins, batch):
    sums = {'z': {l: 0 for l in range(eng.L)}, 'm': {l: 0 for l in range(eng.L)}}
    steps = 0
    for s in range(0, len(wins), batch):
        cap = eng.capture(wins[s:s + batch])
        steps += 1
        for l in range(eng.L):
            sums['z'][l] = sums['z'][l] + cap['z'][l].double().mean(dim=(0, 1))
            sums['m'][l] = sums['m'][l] + cap['m'][l].double().mean(dim=(0, 1))
    return {k: {l: (v[l] / steps).float() for l in v} for k, v in sums.items()}


@torch.no_grad()
def kl_of_sets(eng, wins, means, sets, batch):
    kl = {k: 0.0 for k in sets}
    steps = 0
    for s in range(0, len(wins), batch):
        w = wins[s:s + batch]
        steps += 1
        eng.clear()
        base = torch.log_softmax(eng.forward_logits(w, keep=0).float(), -1)
        p = base.exp()
        for key, keep in sets.items():
            eng.set_nodes(keep, means)
            lp = torch.log_softmax(eng.forward_logits(w, keep=0).float(), -1)
            kl[key] += float((p * (base - lp)).sum(-1).mean())
            del lp
        eng.clear()
    return {k: v / steps for k, v in kl.items()}


@torch.no_grad()
def group_scores(eng, items, batch, dev):
    """子样本上的迭代移除与单节点加回。"""
    sub = items[::max(1, len(items) // S.CRITERIA['n_sub'])][:S.CRITERIA['n_sub']]
    packs = []
    for idx in R.batches(sub, batch):
        clean, corrupt, a, b = R.tensors(sub, idx, dev)
        packs.append((clean, a, b, eng.capture(corrupt)))
    n = eng.n_nodes

    def effect(alive):
        eff = np.zeros(len(alive))
        for clean, a, b, cf in packs:
            keep = torch.zeros(n, dtype=torch.bool)
            keep[torch.as_tensor(alive.copy())] = True
            eng.set_nodes(keep, cf)
            base = eng.ld(clean, a, b)
            for j, c in enumerate(alive):
                keep[int(c)] = False
                eng.set_nodes(keep, cf)
                eff[j] += float((base - eng.ld(clean, a, b)).sum())
                keep[int(c)] = True
        eng.clear()
        return np.abs(eff) / len(sub)

    order = S.iterative_order(n, effect)
    add = np.zeros(n)
    none = torch.zeros(n, dtype=torch.bool)
    for clean, a, b, cf in packs:
        eng.set_nodes(none, cf)
        base = eng.ld(clean, a, b)
        for c in range(n):
            keep = none.clone()
            keep[c] = True
            eng.set_nodes(keep, cf)
            add[c] += float((eng.ld(clean, a, b) - base).sum())
    eng.clear()
    return order, np.abs(add) / len(sub), len(sub)


def prepare(args):
    dev = 'cuda'
    mb = MB / args.tag
    disc = json.loads((mb / 'discovery.json').read_text(encoding='utf-8'))
    z = np.load(mb / 'rankings.npz')
    if hashlib.sha256((mb / 'rankings.npz').read_bytes()).hexdigest() != disc['rankings_sha256']:
        raise SystemExit('MB-v1 排序哈希不一致')
    model, tok, dtype = R.load(args.model, dev)
    eng = R.Engine(model, dev)
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    n = eng.n_nodes
    t0 = time.time()
    wins = windows(tok, -80000, -40000).to(dev)
    means = natural_means(eng, wins, args.nat_batch)
    sets = {}
    for c in range(n):
        keep = torch.ones(n, dtype=torch.bool)
        keep[c] = False
        sets[c] = keep
    kl = kl_of_sets(eng, wins, means, sets, args.nat_batch)
    natural = np.array([kl[c] for c in range(n)])
    backbone = S.backbone_of(natural)
    meta = {'schema': S.SCHEMA, 'model': str(args.model), 'model_snapshot': model_snapshot(args.model)['sha256'], 'dtype': dtype, 'L': eng.L, 'H': eng.H,
            'design_sha': S.design_sha(code_hash()), 'code': code_hash(), 'mb_rankings_sha256': disc['rankings_sha256'],
            'backbone_n': int(len(backbone)), 'natural_seconds': round(time.time() - t0, 1), 'tasks': {}}
    print('natural', meta['natural_seconds'], 'backbone top', backbone[:8].tolist(), flush=True)
    arrays = {'natural_kl': natural, 'backbone': backbone}
    ig = {t: z[f'{t}|node|atp_ig'] for t in B.TASKS}
    ex = {t: z[f'{t}|node|exact'] for t in B.TASKS}
    mats = R.materials(tok, 'discovery')
    for task in B.TASKS:
        t0 = time.time()
        it, add, n_sub = group_scores(eng, mats[task]['items'], args.batch, dev)
        o_ig, o_ex = S.order_of(ig[task]), S.order_of(ex[task])
        arrays[f'{task}|atp_ig'], arrays[f'{task}|exact'] = o_ig, o_ex
        arrays[f'{task}|ig_contrast'] = S.order_of(S.contrast_score(ig, task))
        arrays[f'{task}|exact_contrast'] = S.order_of(S.contrast_score(ex, task))
        arrays[f'{task}|ig_nobb'] = S.excluding(o_ig, backbone)
        arrays[f'{task}|exact_nobb'] = S.excluding(o_ex, backbone)
        arrays[f'{task}|exact_iter'] = it
        arrays[f'{task}|exact_denoise'] = S.order_of(add)
        meta['tasks'][task] = {'n_sub': n_sub, 'seconds': round(time.time() - t0, 1)}
        print(task, meta['tasks'][task], 'iter top', it[:8].tolist(), flush=True)
    np.savez_compressed(out / 'orders.npz', **arrays)
    meta['orders_sha256'] = hashlib.sha256((out / 'orders.npz').read_bytes()).hexdigest()
    (out / 'prepare.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')


def plans(eng, orders, backbone, task, rng):
    n, ks = eng.n_nodes, S.ladder_counts(eng.n_nodes)
    groups = R.group_of(eng)
    plan = []
    for src in B.TASKS:
        tag = 'self' if src == task else src
        for k in ks:
            for m in S.REMOVE_METHODS:
                keep = ~R.keep_from(orders[f'{src}|{m}'], k, n)
                for abl in S.ABLATIONS:
                    plan.append((S.rkey(abl, m, k, tag), keep, abl))
    rand = [rng.permutation(n) for _ in range(S.CRITERIA['n_draws'])]
    for k in ks:
        for d, order in enumerate(rand):
            plan.append((S.rkey('cf', 'random', k, draw=d), ~R.keep_from(order, k, n), 'cf'))
            plan.append((S.rkey('cf', 'matched', k, draw=d), ~R.matched_keep(orders[f'{task}|ig_contrast'], k, groups, rng), 'cf'))
    base = torch.zeros(n, dtype=torch.bool)
    base[torch.as_tensor(backbone.copy())] = True
    free = np.setdiff1d(np.arange(n), backbone)
    bk = [k for k in S.BB_K if k <= len(free)]
    for k in bk:
        for m in S.BB_METHODS:
            order = S.excluding(orders[f'{task}|{m}'], backbone)
            plan.append((S.bkey(m, k), base | R.keep_from(order, k, n), 'cf'))
        for d in range(S.CRITERIA['n_draws']):
            plan.append((S.bkey('random', k, d), base | R.keep_from(rng.permutation(free), k, n), 'cf'))
    plan.append(('bb_only', base, 'cf'))
    for b, k in zip(B.BUDGETS, B.budget_counts(n, B.BUDGETS)):
        for m in S.GROUP_METHODS:
            plan.append((S.gkey(m, b), R.keep_from(orders[f'{task}|{m}'], k, n), 'cf'))
    return plan


def eval_task(eng, task, items, orders, backbone, means, batch, rng):
    plan = plans(eng, orders, backbone, task, rng)
    n = len(items)
    raw = {k: np.zeros(n, dtype=np.float32) for k in [p[0] for p in plan] + ['clean', 'corrupt', 'empty|cf', 'empty|mean']}
    consts = {'mean': R.constant_repl(means[task], eng, 'mean')}
    none = torch.zeros(eng.n_nodes, dtype=torch.bool)
    for idx in R.batches(items, batch):
        clean, corrupt, a, b = R.tensors(items, idx, eng.device)
        repl = {'cf': eng.capture(corrupt), **consts}
        raw['clean'][idx] = eng.ld(clean, a, b).cpu().numpy()
        raw['corrupt'][idx] = eng.ld(corrupt, a, b).cpu().numpy()
        for abl in S.ABLATIONS:
            eng.set_nodes(none, repl[abl])
            raw[f'empty|{abl}'][idx] = eng.ld(clean, a, b).cpu().numpy()
        for key, keep, abl in plan:
            eng.set_nodes(keep, repl[abl])
            raw[key][idx] = eng.ld(clean, a, b).cpu().numpy()
        eng.clear()
    return raw


def evaluate(args):
    dev = 'cuda'
    out = OUT / args.tag
    meta0 = json.loads((out / 'prepare.json').read_text(encoding='utf-8'))
    if hashlib.sha256((out / 'orders.npz').read_bytes()).hexdigest() != meta0['orders_sha256']:
        raise SystemExit('orders.npz 哈希不一致')
    if meta0['code'] != code_hash():
        raise SystemExit('代码已变，排序须重新生成')
    if args.split == 'sealed' and (out / 'eval_sealed.json').exists():
        raise SystemExit('sealed 已评测过，不再重开')
    z = np.load(out / 'orders.npz')
    orders = {k: z[k] for k in z.files if '|' in k}
    backbone = z['backbone']
    means = torch.load(MB / args.tag / 'means.pt', weights_only=False)
    model, tok, _ = R.load(args.model, dev)
    eng = R.Engine(model, dev)
    enc = lambda s: tok.encode(s, add_special_tokens=False)
    meta = {'split': args.split, 'tasks': {}, 'orders_sha256': meta0['orders_sha256']}
    for task in B.TASKS:
        mat = S.build_sealed(task, S.CRITERIA['n_sealed'], enc) if args.split == 'sealed' else B.build_items(task, 'confirm', B.CRITERIA['n_confirm'], enc)
        t0 = time.time()
        rng = np.random.default_rng(int(hashlib.sha256(f'{task}|{args.split}|spec'.encode()).hexdigest()[:8], 16))
        raw = eval_task(eng, task, mat['items'], orders, backbone, means, args.batch, rng)
        np.savez_compressed(out / f'raw_{args.split}_{task}.npz', **raw, clusters=np.array([i['family'] for i in mat['items']]))
        meta['tasks'][task] = {'n': len(mat['items']), 'dropped': mat['dropped'], 'n_families': mat['n_families'],
                               'gate': B.behavior_gate(raw['clean'], raw['corrupt']), 'seconds': round(time.time() - t0, 1)}
        print(task, meta['tasks'][task], flush=True)
    t0 = time.time()
    wins = windows(tok, -40000, None).to(dev)
    nat_means = natural_means(eng, wins, args.nat_batch)
    n = eng.n_nodes
    sets = {}
    for k in S.ladder_counts(n):
        for task in B.TASKS:
            for m in S.REMOVE_METHODS:
                sets[f'{task}|{m}|{k}'] = ~R.keep_from(orders[f'{task}|{m}'], k, n)
        rng = np.random.default_rng(k)
        for d in range(S.CRITERIA['n_draws']):
            sets[f'random|{d}|{k}'] = ~R.keep_from(rng.permutation(n), k, n)
    meta['natural_kl'] = kl_of_sets(eng, wins, nat_means, sets, args.nat_batch)
    meta['natural_seconds'] = round(time.time() - t0, 1)
    (out / f'eval_{args.split}.json').write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding='utf-8')


def report(args):
    from power_analysis import check_phase
    from provenance import provenance
    out = OUT / args.tag
    prep = json.loads((out / 'prepare.json').read_text(encoding='utf-8'))
    ev = json.loads((out / f'eval_{args.split}.json').read_text(encoding='utf-8'))
    n_nodes = prep['L'] * prep['H'] + prep['L']
    ks = S.ladder_counts(n_nodes)
    pk = S.primary_k(n_nodes)
    raws, clusters = {}, {}
    for task in B.TASKS:
        d = np.load(out / f'raw_{args.split}_{task}.npz')
        raws[task] = {k: d[k].astype(np.float64) for k in d.files if k != 'clusters'}
        clusters[task] = d['clusters'].tolist()
    nat = ev['natural_kl']
    nd = S.CRITERIA['n_draws']
    rand_kl = {k: float(np.mean([nat[f'random|{d}|{k}'] for d in range(nd)])) for k in ks}
    rep = {'schema': S.SCHEMA, 'criteria': S.CRITERIA, 'split': args.split, 'model': prep['model'], 'model_snapshot': prep['model_snapshot'], 'dtype': prep['dtype'],
           'design_sha': prep['design_sha'], 'orders_sha256': prep['orders_sha256'], 'primary_k': pk, 'ladder': ks, 'random_kl': rand_kl, 'backbone_n': prep['backbone_n'],
           'tasks': {}, 'official_mib': False, 'provenance': provenance(model_path=prep['model'], seed=0, tool='mech_spec_run')}
    ladder, primary = {}, {}
    for m in S.REMOVE_METHODS:
        primary[m] = {}
        for task in B.TASKS:
            rows = {str(k): S.spec_cell(raws, clusters, task, m, k, natural=nat.get(f'{task}|{m}|{k}'), natural_random=rand_kl[k], n_boot=args.n_boot) for k in ks}
            ladder.setdefault(task, {})[m] = rows
            primary[m][task] = rows[str(pk)]
    mean_cells = {m: {t: S.spec_cell(raws, clusters, t, m, pk, abl='mean', natural=nat.get(f'{t}|{m}|{pk}'), natural_random=rand_kl[pk], n_boot=args.n_boot)
                      for t in B.TASKS} for m in S.REMOVE_METHODS}
    bk = [k for k in S.BB_K if k <= n_nodes - prep['backbone_n']]
    for task in B.TASKS:
        raw, cl = raws[task], clusters[task]
        meta = ev['tasks'][task]
        ident = B.identity_gate(raw[S.gkey('atp_ig', 1.0)], raw['corrupt'], raw['clean'], raw['empty|cf'])
        dr = lambda m, k: float(np.mean([S.drop_of(raw[S.rkey('cf', m, k, draw=d)], raw['clean'], raw['empty|cf']) for d in range(nd)]))
        rep['tasks'][task] = {**meta, 'power': check_phase(S.CRITERIA['expected_delta'], meta['n'], S.CRITERIA['p_d']), 'identity': ident,
                              'ladder': ladder[task], 'random_drop': {str(k): dr('random', k) for k in ks}, 'matched_drop': {str(k): dr('matched', k) for k in ks},
                              'backbone': {m: S.bb_score(raw, cl, bk, m, n_boot=args.n_boot) for m in S.BB_METHODS},
                              'group': S.group_score(raw, cl, n_boot=args.n_boot)}
    gates = all(t['gate']['pass'] and t['identity']['status'] == 'pass' for t in rep['tasks'].values())
    rep['gates_pass'] = gates
    rep['primary'] = primary
    rep['summary_cf'] = S.summarize(primary) if gates else {'status': '门未通过，不判定'}
    rep['summary_mean'] = S.summarize(mean_cells) if gates else {}
    rep['mean_cells'] = mean_cells
    (out / f'report_{args.split}.json').write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding='utf-8')
    print('gates', gates, 'primary k', pk, 'random kl', {k: round(v, 3) for k, v in rand_kl.items()})
    r2 = lambda c, k: round(c.get(k, 0), 2)
    for m in S.REMOVE_METHODS:
        print(m, rep['summary_cf'].get(m), {t: (primary[m][t].get('status'), r2(primary[m][t], 'target_drop'), r2(primary[m][t], 'worst_off'), r2(primary[m][t], 'natural_ratio')) for t in B.TASKS})
        print('   mean', rep['summary_mean'].get(m), {t: (mean_cells[m][t].get('status'), r2(mean_cells[m][t], 'target_drop'), r2(mean_cells[m][t], 'worst_off')) for t in B.TASKS})
    for t, v in rep['tasks'].items():
        print(t, 'id', v['identity']['status'], 'acc', round(v['gate']['clean_acc'], 3), 'gap', round(v['gate']['gap'], 2), 'fam', v['n_families'], 'rand drop', round(v['random_drop'][str(pk)], 3))
        g = v['group']
        print('   group', g.get('status'), {k: round(x, 3) for k, x in g.get('auc', {}).items()}, {k: (round(g[k]['closed'], 2), g[k]['verdict']) for k in ('exact_iter', 'exact_denoise') if k in g})
        print('   bb', {m: (x.get('status'), round(x.get('missing', 0), 2), x.get('k_f80'), round(x.get('auc', 0), 3), round(x.get('random_auc', 0), 3), x.get('beats_random')) for m, x in v['backbone'].items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=['prepare', 'eval', 'report'])
    ap.add_argument('--model', default='')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--split', default='confirm', choices=['confirm', 'sealed'])
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--nat-batch', type=int, default=4)
    ap.add_argument('--n-boot', type=int, default=S.CRITERIA['n_boot'])
    args = ap.parse_args()
    {'prepare': prepare, 'eval': evaluate, 'report': report}[args.step](args)


if __name__ == '__main__':
    main()
