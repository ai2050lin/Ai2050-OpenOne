# -*- coding: utf-8 -*-
"""INT-Q 预注册干预检验（research/interventions/INT-Q-v1.prereg.json）：「疑问」方向的充分性 / 必要性 / 选择性。

方向 d 来自 GR-Q 原生全场（discovery，final 读出，hs[layer]）；在提示中文本最后一个 token 的 hs[layer] 上干预，
用认知 probe 的完整答案对数概率差 g = log P(不确定) − log P(确定) 度量。判据只读预注册文件，不在本脚本中设定。

用法：
    python scripts/run_intervention_q.py --prereg research/interventions/INT-Q-v1.prereg.json --model qwen3-4b \
        --out research/interventions/INT-Q-v1-qwen3-4b.json [--limit 8]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shared.dist_protocol import entangle as E  # noqa: E402
from shared.dist_protocol.runner_v3 import parse_max_memory, module_device, model_snapshot  # noqa: E402

BUNDLES = ROOT / 'distributed_data' / 'bundles'
NATIVE = Path(os.environ.get('AI2050_NATIVE_DIR', 'D:/ai2050_native'))
UNC, CER = '不确定', '确定'


def file_sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def corpus(tm: str, version: int) -> dict:
    return json.loads((BUNDLES / tm / f'v{version}' / 'corpus.json').read_text(encoding='utf-8'))


def native_dirs(family: str, tm_dir: str, key: str):
    dirs = sorted(glob.glob(str(NATIVE / family / tm_dir / '*' / key / 'shard-*')))
    if not dirs:
        raise FileNotFoundError(f'没有原生全场：{family}/{tm_dir}/*/{key}')
    return dirs


def class_mean_diff(field, items_by_id, layer, factor, hi, lo, split='discovery'):
    X = field.X('final', layer)
    ids = field.ids
    sel_hi = [i for i, x in enumerate(ids) if items_by_id.get(x, {}).get('split') == split and items_by_id[x]['cond'].get(factor) == hi]
    sel_lo = [i for i, x in enumerate(ids) if items_by_id.get(x, {}).get('split') == split and items_by_id[x]['cond'].get(factor) == lo]
    return X[sel_hi].mean(0) - X[sel_lo].mean(0), X[sel_lo], X, ids, len(sel_hi), len(sel_lo)


def materials(pre):
    gq = corpus('TG-GR-Q', 7)
    gn = corpus('TG-GR-NEG', 6)
    kn = corpus('TG-KN-ISA', 3)
    held = ('lockbox', 'lockbox_frame')

    def gr(it, fam):
        s = it['text']
        fl = it['filler_slots']
        return {'id': it['id'], 'family': fam, 'filler': it['filler_id'], 'split': it['split'], 'cond': it['cond'], 'text': s,
                'prompt': s + pre['probes']['epistemic_GR'].replace('{text}', '').replace('{S}', fl['S']).replace('{A}', fl['A'])}

    def kni(it):
        fl = it['filler_slots']
        return {'id': it['id'], 'family': 'KN-ISA', 'filler': it['filler_id'], 'split': it['split'], 'cond': it['cond'],
                'text': it['text'], 'Y': fl['Y'], 'X': fl['X'],
                'prompt': it['text'] + pre['probes']['epistemic_KN'].replace('{text}', '').replace('{X}', fl['X']).replace('{Y}', fl['Y']),
                'cat_prompt': it['text'] + pre['probes']['category_KN'].split('；')[0].replace('{text}', '').replace('{X}', fl['X'])}

    # 填充槽位：语料条目不带槽位，按文法补上
    for c, g in ((gq, 'GR-Q'), (gn, 'GR-NEG'), (kn, 'KN-ISA')):
        spec = json.loads((ROOT / 'shared/dist_protocol/grammars' / f'{g}.json').read_text(encoding='utf-8'))
        slots = {f['id']: f['slots'] for f in spec['fillers']}
        for it in c['items']:
            if not it.get('canary'):
                it['filler_slots'] = slots[it['filler_id']]
    P = {'decl_cross': [], 'interrog_cross': [], 'decl_within': [], 'interrog_within': [], 'selectivity': []}
    for it in gn['items']:
        if it.get('canary') or it['split'] not in held:
            continue
        P['decl_cross' if it['cond']['mood'] == 'decl' else 'interrog_cross'].append(gr(it, 'GR-NEG'))
    kn_stmt = {'decl', 'scene', 'concess', 'topic'}
    y_by_split = {}
    for it in kn['items']:
        if not it.get('canary'):
            y_by_split.setdefault(it['split'], set()).add(it['filler_slots']['Y'])
    for it in kn['items']:
        if it.get('canary') or it['split'] not in held:
            continue
        r = kni(it)
        r['cat_contrast'] = sorted(y_by_split[it['split']] - {r['Y']})
        if it['cond']['frame'] in kn_stmt:
            P['decl_cross'].append(r)
            P['selectivity'].append(r)
        elif it['cond']['frame'] == 'interrog':
            P['interrog_cross'].append(r)
    for it in gq['items']:
        if it.get('canary') or it['split'] not in held:
            continue
        P['decl_within' if it['cond']['mood'] == 'decl' else 'interrog_within'].append(gr(it, 'GR-Q'))
    return P, gq, gn


def load_model(m):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = str(ROOT / m['path'])
    tok = AutoTokenizer.from_pretrained(path)
    dtype = torch.bfloat16
    if 'device_map=auto' in m['placement']:
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from transformers import AutoConfig
        mm = parse_max_memory(m['placement'].split('max_memory')[1].strip())
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(path), torch_dtype=dtype)
        model.tie_weights()
        model = load_checkpoint_and_dispatch(model, path, device_map='auto', max_memory=mm, dtype=dtype,
                                             offload_folder=os.environ.get('AI2050_OFFLOAD_DIR', 'D:/ai2050_offload/' + m['model_id']),
                                             offload_buffers=True,
                                             no_split_module_classes=list(getattr(model, '_no_split_modules', None) or []))
    else:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, device_map='cuda')
    model.eval()
    return tok, model, path


class Steer:
    """在第 layer-1 个块的输出（= hs[layer]）、位置 pos 上，对批内每一行加上一行位移。
    deltas_fn(h) → (R, D) float32 张量；h 为第 0 行在 pos 处的干预前状态（各行在 pos 之前完全相同）。"""

    def __init__(self, block):
        self.pos, self.fn, self.h = None, None, None
        block.register_forward_hook(self.hook)

    def hook(self, mod, args, out):
        if self.fn is None:
            return None
        hs = out[0] if isinstance(out, tuple) else out
        h = hs[0, self.pos].float()
        self.h = h.detach().cpu().numpy()
        delta = self.fn(h).to(hs.device)
        hs = hs.clone()
        hs[:, self.pos] = (hs[:, self.pos].float() + delta).to(hs.dtype)
        return (hs,) + tuple(out[1:]) if isinstance(out, tuple) else hs


def main(argv=None) -> int:
    import torch
    from scipy import stats
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--prereg', required=True)
    ap.add_argument('--model', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=None, help='每池最多条目数（只用于自检，正式运行不设）')
    args = ap.parse_args(argv)

    pre_p = Path(args.prereg)
    pre = json.loads(pre_p.read_text(encoding='utf-8'))
    sha = file_sha(pre_p)
    want = (pre_p.with_suffix('.sha256')).read_text(encoding='utf-8').split()[0]
    if sha != want:
        raise SystemExit(f'预注册文件哈希 {sha} 与登记的 {want} 不符')
    m = next(x for x in pre['models'] if x['model_id'] == args.model)
    L = int(m['layer_hs'])
    t0 = time.time()

    # ---- 方向 ----
    P, gq, gn = materials(pre)
    gq_by = {it['id']: it for it in gq['items'] if not it.get('canary')}
    gn_by = {it['id']: it for it in gn['items'] if not it.get('canary')}
    fq = E.NativeField(native_dirs('GR-Q', 'TG-GR-Q-v7', m['native_key']))
    d, Xdecl, XQ, idsQ, nq_hi, nq_lo = class_mean_diff(fq, gq_by, L, 'mood', 'interrog', 'decl')
    fn_ = E.NativeField(native_dirs('GR-NEG', 'TG-GR-NEG-v6', m['native_key']))
    p, _, _, _, np_hi, np_lo = class_mean_diff(fn_, gn_by, L, 'polarity', 'neg', 'aff')
    nd = float(np.linalg.norm(d))
    dhat = d / nd
    mu_D = float((Xdecl @ dhat).mean())
    p_s = p / np.linalg.norm(p) * nd
    phat = p / np.linalg.norm(p)
    rc = pre['directions']['random_controls']
    rng = np.random.default_rng(int(rc['seed']))
    R = rng.standard_normal((int(rc['n']), d.shape[0])).astype(np.float32)
    Rhat = R / np.linalg.norm(R, axis=1, keepdims=True)
    native_final = {x: XQ[i] for i, x in enumerate(idsQ)}

    tok, model, path = load_model(m)
    decoder = model.get_decoder() if hasattr(model, 'get_decoder') else model.model
    blocks = list(decoder.layers)
    steer = Steer(blocks[L - 1])
    dev_in = module_device(model.get_input_embeddings()) or 'cuda'
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)
    T = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731

    def prep(text, prompt):
        te = tok(text, return_offsets_mapping=True)
        off = te['offset_mapping']
        pos = max(i for i, (a, b) in enumerate(off) if b > a)
        pids = tok(prompt)['input_ids']
        if list(pids[:pos + 1]) != list(te['input_ids'][:pos + 1]):
            return None, None
        return pids, pos

    def score(pids, pos, cands, fn):
        """返回 (C, 候选数) 的完整答案对数概率；fn(h) 给出每个条件的位移 (C, D)。
        每个条件单独一次前向（批 = 该条件下的全部候选，与 runner_v3.answer_logprobs 的批形状相同）：
        bf16 下不同批形状的 logit 会相差一个量化步长（约 0.1–0.25 nats），堆叠全部条件会让零干预结果偏离正式运行。"""
        seqs = [tok(c, add_special_tokens=False)['input_ids'] for c in cands]
        n_p = len(pids)
        C = fn(torch.zeros(d.shape[0])).shape[0]
        width = n_p + max(len(s) for s in seqs)
        ids = torch.full((len(cands), width), pad_id, dtype=torch.long)
        mask = torch.zeros((len(cands), width), dtype=torch.long)
        for k, s in enumerate(seqs):
            ids[k, :n_p + len(s)] = torch.tensor(list(pids) + s)
            mask[k, :n_p + len(s)] = 1
        ids, mask = ids.to(dev_in), mask.to(dev_in)
        out = np.zeros((C, len(cands)))
        h0 = None
        steer.pos = pos
        for ci in range(C):
            steer.fn = lambda h, ci=ci: fn(h.cpu())[ci:ci + 1].expand(len(cands), -1)
            try:
                with torch.no_grad():
                    lp = torch.log_softmax(model(input_ids=ids, attention_mask=mask).logits.float(), dim=-1)
            finally:
                steer.fn = None
            if ci == 0:
                h0 = steer.h
            for k, s in enumerate(seqs):
                q = torch.arange(n_p - 1, n_p - 1 + len(s), device=lp.device)
                out[ci, k] = float(lp[k, q, torch.tensor(s, device=lp.device)].sum().item())
        return out, h0

    suff_names = ['base', 'd@0.5', 'd@1', 'd@2', 'p'] + [f'r{k}' for k in range(len(R))]
    suff_D = torch.stack([T(np.zeros_like(d)), T(0.5 * d), T(d), T(2 * d), T(p_s)] + [T(r / np.linalg.norm(r) * nd) for r in R])
    nec_names = ['base', 'abl_d', 'abl_p'] + [f'r{k}' for k in range(len(R))]
    U = torch.stack([T(dhat), T(phat)] + [T(r) for r in Rhat])
    dh_t = T(dhat)

    def nec_fn(h):
        mv = float(h.float() @ dh_t) - mu_D
        return torch.cat([torch.zeros(1, U.shape[1]), -mv * U])

    res = {'excluded_straddle': {}, 'pools': {}}
    checks = {'native_cos': [], 'gq_margin_diff': []}
    beh_margin = gq_behavior(args.model, gq)

    def run_pool(name, names, fn):
        rows = []
        items = P[name][:args.limit] if args.limit else P[name]
        for it in items:
            pids, pos = prep(it['text'], it['prompt'])
            if pids is None:
                res['excluded_straddle'][name] = res['excluded_straddle'].get(name, 0) + 1
                continue
            lp, h = score(pids, pos, [UNC, CER], fn)
            g = lp[:, 0] - lp[:, 1]
            rows.append({'id': it['id'], 'family': it['family'], 'filler': it['filler'], 'g': g.tolist()})
            if it['id'] in native_final:
                v = native_final[it['id']]
                checks['native_cos'].append(float(h @ v / (np.linalg.norm(h) * np.linalg.norm(v) + 1e-9)))
            if it['id'] in beh_margin:
                mb = beh_margin[it['id']]
                checks['gq_margin_diff'].append(abs(g[0] - mb))
        res['pools'][name] = {'conditions': names, 'items': rows}
        print(f'[int-q] {args.model} {name}: {len(rows)} 条 · {time.time() - t0:.0f}s', flush=True)

    run_pool('decl_cross', suff_names, lambda h: suff_D)
    run_pool('decl_within', suff_names, lambda h: suff_D)
    run_pool('interrog_cross', nec_names, nec_fn)
    run_pool('interrog_within', nec_names, nec_fn)

    # 选择性：类别 probe
    sel = []
    for it in (P['selectivity'][:args.limit] if args.limit else P['selectivity']):
        pids, pos = prep(it['text'], it['cat_prompt'])
        if pids is None:
            res['excluded_straddle']['selectivity'] = res['excluded_straddle'].get('selectivity', 0) + 1
            continue
        cands = [it['Y']] + it['cat_contrast']
        lp, _ = score(pids, pos, cands, lambda h: suff_D[[0, 2]])
        mcat = lp[:, 0] - lp[:, 1:].max(1)
        sel.append({'id': it['id'], 'filler': it['filler'], 'm_cat': mcat.tolist()})
    res['pools']['selectivity'] = {'conditions': ['base', 'd@1'], 'items': sel}
    print(f'[int-q] {args.model} selectivity: {len(sel)} 条 · {time.time() - t0:.0f}s', flush=True)

    out = {'schema': 'intervention_result.v1', 'prereg_id': pre['id'], 'prereg_sha256': sha, 'model_id': args.model,
           'model_snapshot': model_snapshot(path), 'layer_hs': L, 'limit': args.limit,
           'direction': {'norm_d': nd, 'mu_D': mu_D, 'n_interrog': nq_hi, 'n_decl': nq_lo, 'n_neg': np_hi, 'n_aff': np_lo,
                         'cos_d_p': float(dhat @ phat)},
           'self_checks': {'native_cos_min': min(checks['native_cos']) if checks['native_cos'] else None,
                           'native_cos_n': len(checks['native_cos']),
                           'gq_margin_absdiff_max': max(checks['gq_margin_diff']) if checks['gq_margin_diff'] else None,
                           'gq_margin_n': len(checks['gq_margin_diff'])},
           'torch': torch.__version__, 'seconds': round(time.time() - t0, 1), **res}
    out['evaluation'] = evaluate(out, stats)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding='utf-8')
    print(json.dumps({'self_checks': out['self_checks'], 'evaluation': out['evaluation']['verdicts']}, ensure_ascii=False, indent=1))
    return 0


def gq_behavior(model_id, gq):
    """正式运行 behavior.json 中 GR-Q 条目的 margin 换算成 g：疑问句 m = g，陈述句 m = −g。"""
    import sqlite3
    from shared.dist_protocol.pipeline import parse_blob
    db = sqlite3.connect(f'file:{ROOT / "distributed_data/dist.sqlite3"}?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    mood = {it['id']: it['cond']['mood'] for it in gq['items'] if not it.get('canary')}
    out = {}
    for r in db.execute("SELECT sha, summary_json, design_sha FROM results WHERE tm_id='TG-GR-Q' AND model_id=? AND kind='real'", (model_id,)):
        if not r['design_sha'].startswith('5d4d3dfbf5b6'):
            continue
        s = json.loads(r['summary_json'])
        p = ROOT / 'distributed_data/objects' / r['sha'][:2] / r['sha']
        files = parse_blob(p.read_bytes(), s.get('_manifest') or [])
        for b in json.loads(files['behavior.json'].decode('utf-8')).get('items', []):
            if b.get('margin') is not None and b['id'] in mood:
                out[b['id']] = b['margin'] if mood[b['id']] == 'interrog' else -b['margin']
    return out


def evaluate(out, stats):
    """按预注册判据评估（INT-Q-v1 tests.*.pass_if），不引入新判据。"""
    pools = out['pools']

    def arr(pool):
        return np.array([r['g'] for r in pools[pool]['items']]) if pools[pool]['items'] else np.zeros((0, 1))

    def filler_sign(delta, pool):
        by = {}
        for v, r in zip(delta, pools[pool]['items']):
            by.setdefault((r['family'], r['filler']), []).append(v)
        means = np.array([np.mean(v) for v in by.values()])
        k = int((means > 0).sum())
        return {'n_fillers': len(means), 'n_positive': k,
                'p_sign': float(stats.binomtest(k, len(means), 0.5, alternative='greater').pvalue) if len(means) else None}

    E_ = {'pools': {}, 'verdicts': {}}
    G = {k: arr(k) for k in ('decl_cross', 'decl_within', 'interrog_cross', 'interrog_within')}
    validity = {}
    for tag, (q, dl) in {'cross': ('interrog_cross', 'decl_cross'), 'within': ('interrog_within', 'decl_within')}.items():
        diff = float(G[q][:, 0].mean() - G[dl][:, 0].mean()) if len(G[q]) and len(G[dl]) else None
        validity[tag] = {'mean_g_interrog': float(G[q][:, 0].mean()), 'mean_g_decl': float(G[dl][:, 0].mean()),
                         'diff': diff, 'valid': diff is not None and diff > 0}
    E_['validity'] = validity

    def suff(pool, tag):
        g = G[pool]
        names = pools[pool]['conditions']
        D = g - g[:, [0]]
        means = {n: float(D[:, i].mean()) for i, n in enumerate(names)}
        dd = D[:, names.index('d@1')]
        w = stats.wilcoxon(dd, alternative='greater') if np.any(dd != 0) else None
        rnd = [means[n] for n in names if n.startswith('r')]
        crit = [bool(means['d@1'] > 0 and w is not None and w.pvalue < 0.01), bool(means['d@1'] > max(rnd)), bool(means['d@1'] > means['p'])]
        verdict = '不判定' if not validity[tag]['valid'] else ('通过' if all(crit) else '未通过')
        return {'n': int(len(g)), 'mean_delta': means, 'wilcoxon_p': float(w.pvalue) if w else None,
                'random_max': max(rnd), 'random_mean': float(np.mean(rnd)), 'criteria': crit, 'verdict': verdict,
                'filler_robustness': filler_sign(dd, pool), 'base_mean_g': float(g[:, 0].mean())}

    def nec(pool, tag):
        g = G[pool]
        names = pools[pool]['conditions']
        Dr = g[:, [0]] - g
        means = {n: float(Dr[:, i].mean()) for i, n in enumerate(names)}
        dd = Dr[:, names.index('abl_d')]
        w = stats.wilcoxon(dd, alternative='greater') if np.any(dd != 0) else None
        rnd = [means[n] for n in names if n.startswith('r')]
        crit = [bool(means['abl_d'] > 0 and w is not None and w.pvalue < 0.01), bool(means['abl_d'] > max(rnd)), bool(means['abl_d'] > means['abl_p'])]
        verdict = '不判定' if not validity[tag]['valid'] else ('通过' if all(crit) else '未通过')
        return {'n': int(len(g)), 'mean_drop': means, 'wilcoxon_p': float(w.pvalue) if w else None,
                'random_max': max(rnd), 'random_mean': float(np.mean(rnd)), 'criteria': crit, 'verdict': verdict,
                'filler_robustness': filler_sign(dd, pool), 'base_mean_g': float(g[:, 0].mean())}

    E_['pools']['sufficiency_primary'] = suff('decl_cross', 'cross')
    E_['pools']['sufficiency_secondary'] = suff('decl_within', 'within')
    E_['pools']['necessity_primary'] = nec('interrog_cross', 'cross')
    E_['pools']['necessity_secondary'] = nec('interrog_within', 'within')
    sel = np.array([r['m_cat'] for r in pools['selectivity']['items']])
    dcat = float((sel[:, 1] - sel[:, 0]).mean()) if len(sel) else None
    dg = E_['pools']['sufficiency_primary']['mean_delta']['d@1']
    E_['pools']['selectivity'] = {'n': int(len(sel)), 'mean_delta_m_cat': dcat, 'limit': 0.5 * dg,
                                  'base_cat_acc': float((sel[:, 0] > 0).mean()) if len(sel) else None,
                                  'verdict': ('通过' if dcat is not None and abs(dcat) <= 0.5 * dg else '未通过') if dg > 0 else '不判定'}
    for k, v in E_['pools'].items():
        E_['verdicts'][k] = v['verdict']
    sp, npv = E_['verdicts']['sufficiency_primary'], E_['verdicts']['necessity_primary']
    E_['verdicts']['causal_use_in_scope'] = '是' if sp == '通过' and npv == '通过' else '否'
    return E_


if __name__ == '__main__':
    raise SystemExit(main())
