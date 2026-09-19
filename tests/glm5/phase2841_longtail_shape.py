"""Phase 2841 (LPF Delta-II/III): long-tail shape verdict.

2840 established the head-level carrying capacity is long-tailed
(S4=25.3%, S16=40.2%, S32=73.7%) but only sampled 4 rungs.  This phase
scans every single head individually (ablate one head, matched
protocol) at 5 key layers -- L22/L23 (causal peaks, 2838/2839),
L26 (mid), L28 (alignment peak, 2839), L33 (2833 peak) -- giving a
32-point per-layer carrying distribution, then adjudicates its shape:

  power law   log(drop) vs log(rank)  linear fit R2_p
  exponential log(drop) vs rank      linear fit R2_e

Prereg (frozen before any readout):
  E1  shape_resolvable iff |R2_p - R2_e| >= 0.02 for >= 4/5 layers
  E2  power_dominant iff >= 4/5 layers have R2_p > R2_e
  E3  concentration reported: for each layer,
      c8 = mean(top-8 drops) / sum(all 32 drops)
  C1  head_additivity iff joint_all32_drop(L22) / max(sum_singles_L22,
      1e-9) in [0.7, 1.3]  (head-level analogue of 2839 layer ratio)
  verdict: longtail_shape = power_law_dominant iff E2;
           exponential_dominant iff E1 and (>= 4/5 layers R2_e > R2_p);
           mixed otherwise
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2841' / 'longtail_shape'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2841
SCAN_LAYERS = [22, 23, 26, 28, 33]
LAST = 35

PREREG = {
    'E1': 'shape_resolvable iff |R2_p - R2_e| >= 0.02 for >= 4/5 layers',
    'E2': 'power_dominant iff >= 4/5 layers R2_p > R2_e',
    'E3': 'concentration c8 = mean(top8 drops)/sum(all32 drops) reported '
          'per layer',
    'C1': 'head_additivity iff joint_all32_drop(L22)/sum_singles_L22 '
          'in [0.7, 1.3]',
    'verdict': 'power_law_dominant iff E2; exponential_dominant iff E1 '
               'and >= 4/5 layers R2_e > R2_p; mixed otherwise',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def fit_r2(x, y):
    """linear fit R2 of y on x (numpy polyfit, degree 1)."""
    if len(x) < 3:
        return float('nan')
    c = np.polyfit(x, y, 1)
    pred = np.polyval(c, x)
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED}
    fc.save(OUT / 'execution.json', execution)

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()

    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            tc[t] = int(ids[0])
        return tc[t]

    # ---------- ablation control: one head (or list) in one layer ----------
    abl = {'groups': None}

    def make_abl_hook(li):
        def pre_hook(module, args):
            g = abl['groups']
            if g is None or li not in g:
                return None
            a = args[0]
            a2 = a.clone()
            v = a2.view(*a2.shape[:-1], 32, a2.shape[-1] // 32)
            v[..., list(g[li]), :] = 0
            return (a2,) + tuple(args[1:])
        return pre_hook

    for li, layer in enumerate(model.model.layers):
        layer.self_attn.o_proj.register_forward_pre_hook(make_abl_hook(li))

    cap = {'attn': {}, 'mlp': {}}

    def make_out_hook(kind, li):
        def hook(module, args, output):
            o = output[0] if isinstance(output, tuple) else output
            cap[kind].setdefault(li, []).append(
                o[0].detach().float().cpu().numpy())
        return hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(
            make_out_hook('attn', li)))
        handles.append(layer.mlp.register_forward_hook(
            make_out_hook('mlp', li)))

    def clear_cap():
        for d in ('attn', 'mlp'):
            for li in cap[d]:
                del cap[d][li][:]

    def forward_run(tokens, pos):
        clear_cap()
        with torch.no_grad():
            out = model(torch.tensor([tokens], device='cuda'),
                        output_hidden_states=True)
            hs = np.stack([h[0, pos, :].float().cpu().numpy()
                           for h in out.hidden_states])
        attn = np.stack([cap['attn'][li][0][pos] for li in range(36)])
        mlp = np.stack([cap['mlp'][li][0][pos] for li in range(36)])
        return hs, attn, mlp

    # ---------- targets (identical to 2837-2840) ----------
    all_words = [w for v in CATS.values() for w in v]
    single_tok = []
    for w in all_words:
        try:
            tid(w)
            single_tok.append(w)
        except AssertionError:
            pass
    targets = {}
    for cat in CAT_WORDS:
        cs = [w for w in CATS[cat] if w in single_tok]
        targets[cat] = cs[:2]
    target_list = [(cat, w) for cat in CAT_WORDS for w in targets[cat]]

    Erows = {w: W_U[tid(w)].astype(np.float64) for w in single_tok}
    cents = []
    for cat in CAT_WORDS:
        ws = [w for w in CATS[cat] if w in single_tok]
        cents.append(np.stack([Erows[w] for w in ws]).mean(0))
    Cm = np.stack(cents)
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    dW_unit = np.stack([unit(dW[i]) for i in range(10)])

    rng = np.random.default_rng(SEED)
    vocab_size = W_U.shape[0]
    word_tids = set(tc.values())
    null_tids = []
    while len(null_tids) < len(target_list):
        r = int(rng.integers(0, vocab_size))
        if r not in word_tids and r > 0:
            null_tids.append(r)
    func_tid = tid('the')

    def conds_for(i, cat, w):
        w_tid = tid(w)
        same_cat = [x for x in targets[cat] if x != w]
        if not same_cat:
            same_cat = [x for x in CATS[cat] if x != w
                        and x in single_tok][:1]
        return {'same': [tid(same_cat[0]), w_tid],
                'func': [func_tid, w_tid],
                'null': [null_tids[i], w_tid]}

    def measure_cls(i, cat, w):
        cdir = dW_unit[CAT_WORDS.index(cat)]
        w_tid = tid(w)
        conds = conds_for(i, cat, w)
        hs_iso, _, _ = forward_run([w_tid], 0)
        iso0 = hs_iso[0]
        ds = {}
        for cn, toks in conds.items():
            hs, attn, mlp = forward_run(toks, 1)
            raw_fin = hs[LAST] + attn[LAST] + mlp[LAST]
            ds[cn] = raw_fin - iso0
        d_spec = ds['same'] - 0.5 * (ds['func'] + ds['null'])
        n2 = float(np.linalg.norm(d_spec))
        return float(abs(d_spec @ cdir) / max(n2, 1e-30))

    def mean_cls():
        vals = [measure_cls(i, cat, w)
                for i, (cat, w) in enumerate(target_list)]
        return float(np.mean(vals))

    cls_base_m = mean_cls()
    print('P2841 base mean_cls=%.6f' % cls_base_m, flush=True)

    # ---------- single-head scan ----------
    drops = {}          # (li, head) -> drop
    for li in SCAN_LAYERS:
        for h in range(32):
            abl['groups'] = {li: [h]}
            m = mean_cls()
            abl['groups'] = None
            drops[(li, h)] = 1.0 - m / max(cls_base_m, 1e-30)
        print('P2841 layer L%d scanned (top5 %s)' % (
            li, np.round(sorted(
                [drops[(li, h)] for h in range(32)], reverse=True)[:5],
                4).tolist()), flush=True)

    # ---------- C1: joint all-32-head ablation at L22 vs sum singles ----------
    abl['groups'] = {22: list(range(32))}
    joint22 = 1.0 - mean_cls() / max(cls_base_m, 1e-30)
    abl['groups'] = None
    sum_single22 = float(sum(drops[(22, h)] for h in range(32)))
    ratio22 = joint22 / max(sum_single22, 1e-9)

    # ---------- shape fits ----------
    layer_fits = {}
    for li in SCAN_LAYERS:
        d = np.array([drops[(li, h)] for h in range(32)])
        d_sorted = np.sort(d)[::-1]
        d_pos = d_sorted[d_sorted > 1e-9]
        n_pos = int(len(d_pos))
        ranks = np.arange(1, n_pos + 1, dtype=np.float64)
        r2_p = fit_r2(np.log(ranks), np.log(d_pos))
        r2_e = fit_r2(ranks.astype(np.float64), np.log(d_pos))
        c8 = float(np.mean(d_sorted[:8]) / max(np.sum(d_sorted), 1e-30))
        layer_fits[li] = {'n_pos': n_pos, 'R2_power': round(r2_p, 4),
                          'R2_exp': round(r2_e, 4), 'c8_conc': round(c8, 4),
                          'sum_drop': round(float(np.sum(d)), 4),
                          'max_drop': round(float(d_sorted[0]), 4),
                          'min_drop': round(float(d_sorted[-1]), 4)}

    resolvable = [li for li in SCAN_LAYERS
                  if abs(layer_fits[li]['R2_power']
                         - layer_fits[li]['R2_exp']) >= 0.02]
    n_power = sum(1 for li in SCAN_LAYERS
                  if layer_fits[li]['R2_power'] > layer_fits[li]['R2_exp'])
    n_exp = sum(1 for li in SCAN_LAYERS
                if layer_fits[li]['R2_exp'] > layer_fits[li]['R2_power'])
    e1 = len(resolvable) >= 4
    e2 = n_power >= 4
    if e2:
        shape = 'power_law_dominant'
    elif e1 and n_exp >= 4:
        shape = 'exponential_dominant'
    else:
        shape = 'mixed'

    v = {
        'cls_base_mean': round(cls_base_m, 6),
        'E1_shape_resolvable': bool(e1),
        'n_layers_resolvable': len(resolvable),
        'E2_power_dominant': bool(e2),
        'n_layers_power': n_power,
        'n_layers_exp': n_exp,
        'C1_head_additivity_ratio': round(ratio22, 4),
        'C1_head_additivity': bool(0.7 <= ratio22 <= 1.3),
        'joint22_drop': round(joint22, 4),
        'sum_single22_drop': round(sum_single22, 4),
        'longtail_shape': shape,
        'layer_fits': {str(li): layer_fits[li] for li in SCAN_LAYERS},
    }

    result = {'phase': 2841, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)

    fc.npz(OUT / 'head_drop_map.npz',
           **{('drops_L%d' % li): np.array(
               [drops[(li, h)] for h in range(32)], dtype=np.float32)
              for li in SCAN_LAYERS})

    elapsed = time.monotonic() - t0
    cc.ledger('phase2841', elapsed)
    print('P2841 VERDICT %s' % json.dumps(v), flush=True)
    print('P2841 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
