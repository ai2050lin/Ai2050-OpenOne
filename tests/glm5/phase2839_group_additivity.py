"""Phase 2839 (LPF Delta-II/III): group-ablation additivity + dual-
spectrum misalignment quantification.

2838 found the causal peak at L22-23 (7.4%/7.2%) but no single layer
carries >10%, and the alignment spectrum (share peak L30/L33/L35) is
misaligned with the causal spectrum.  This phase tests:

  (A) additivity: is the joint ablation damage of the {L22,L23} group
      equal to the sum of single-layer damages (independent), larger
      (synergistic cross-layer loop), or smaller (overlapping
      redundancy)?  Plus the 16-layer all-top-4 upper bound.
  (B) misalignment: Spearman correlation between the per-layer
      alignment curve (sum of top-4 |share| per layer, from pass-1
      share map) and the per-layer causal curve (2838-protocol single
      drops re-measured in-run for L22/L23 and taken from the in-run
      scan for the rest? NO - causal curve is NOT re-scanned here;
      misalignment uses only the alignment curve argmax vs the causal
      argmax from 2838's registered result plus the in-run singles).

Prereg (frozen before any readout):
  A1  additivity_ratio = drop(G{22,23}) / (drop(L22) + drop(L23)),
      both measured in-run with matched protocol:
      independent  iff 0.7 <= ratio <= 1.3
      synergistic  iff ratio > 1.3
      overlapping  iff ratio < 0.7
  A2  distributed_sum_substantial iff drop(G_all_16layers) > 0.15
  B1  dual_spectrum_misaligned iff |spearman(align_curve, caus_curve)|
      < 0.5, where align_curve = per-layer sum of top-4 |share|
      (in-run pass 1) and caus_curve = 2838 registered drop curve
      (immutable result.json).
  verdict_a: regime = independent / synergistic / overlapping (A1),
             with A2 as the upper-bound descriptor
  verdict_b: misaligned_confirmed iff B1 else partially_aligned
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
OUT = BASE / 'phase2839' / 'group_additivity'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2838_RESULT = BASE / 'phase2838' / 'decisive_layer_neuron' / 'result.json'
SEED = 2839
SCAN_LAYERS = list(range(20, 36))
LAST = 35
TOP_K = 4

PREREG = {
    'A1': 'additivity_ratio = drop(G22,23) / (drop(L22)+drop(L23)) '
          'in-run matched protocol; independent iff 0.7<=r<=1.3, '
          'synergistic iff r>1.3, overlapping iff r<0.7',
    'A2': 'distributed_sum_substantial iff drop(G_all16) > 0.15',
    'B1': 'dual_spectrum_misaligned iff |spearman(align_curve, '
          'caus_curve_2838)| < 0.5',
    'verdict_a': 'regime = A1 class, A2 upper-bound descriptor',
    'verdict_b': 'misaligned_confirmed iff B1 else partially_aligned',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())
    DOMAIN_OF = exec2806['domain_of']
    res2838 = json.loads(SRC_2838_RESULT.read_text(encoding='utf-8'))
    top_heads_2838 = {int(k): v for k, v in
                      res2838['top_heads_per_layer'].items()}

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED,
                 'top_heads_source': 'phase2838 result.json (immutable)'}
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

    W3 = {}
    for li in SCAN_LAYERS:
        Wo = model.model.layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)
        W3[li] = Wo.reshape(Wo.shape[0], 32, Wo.shape[1] // 32)

    # ---------- multi-layer ablation control ----------
    abl = {'groups': None, 'coords': None}  # groups: {li: [heads]}

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

    for li in SCAN_LAYERS:
        model.model.layers[li].self_attn.o_proj \
            .register_forward_pre_hook(make_abl_hook(li))

    cap = {'attn': {}, 'mlp': {}, 'opj': {}}
    cap['opj_on'] = False

    def make_out_hook(kind, li):
        def hook(module, args, output):
            o = output[0] if isinstance(output, tuple) else output
            cap[kind].setdefault(li, []).append(
                o[0].detach().float().cpu().numpy())
        return hook

    def make_opj_hook(li):
        def pre_hook(module, args):
            if cap['opj_on']:
                cap['opj'].setdefault(li, []).append(
                    args[0].detach()[0].float().cpu().numpy())
        return pre_hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(
            make_out_hook('attn', li)))
        handles.append(layer.mlp.register_forward_hook(
            make_out_hook('mlp', li)))
        if li in SCAN_LAYERS:
            handles.append(layer.self_attn.o_proj
                           .register_forward_pre_hook(make_opj_hook(li)))

    def clear_cap():
        for d in ('attn', 'mlp', 'opj'):
            for li in cap[d]:
                del cap[d][li][:]

    def forward_capture(tokens, pos):
        clear_cap()
        cap['opj_on'] = True
        with torch.no_grad():
            out = model(torch.tensor([tokens], device='cuda'),
                        output_hidden_states=True)
            hs = np.stack([h[0, pos, :].float().cpu().numpy()
                           for h in out.hidden_states])
        cap['opj_on'] = False
        attn = np.stack([cap['attn'][li][0][pos] for li in range(36)])
        mlp = np.stack([cap['mlp'][li][0][pos] for li in range(36)])
        opj = {li: cap['opj'][li][0][pos] for li in SCAN_LAYERS}
        return hs, attn, mlp, opj

    def forward_abl(tokens, pos):
        clear_cap()
        with torch.no_grad():
            out = model(torch.tensor([tokens], device='cuda'),
                        output_hidden_states=True)
            hs = np.stack([h[0, pos, :].float().cpu().numpy()
                           for h in out.hidden_states])
        attn = np.stack([cap['attn'][li][0][pos] for li in range(36)])
        mlp = np.stack([cap['mlp'][li][0][pos] for li in range(36)])
        return hs, attn, mlp

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
        hs_iso, _, _, _ = forward_capture([w_tid], 0)
        iso0 = hs_iso[0]
        ds = {}
        for cn, toks in conds.items():
            hs, attn, mlp = forward_abl(toks, 1)
            raw_fin = hs[LAST] + attn[LAST] + mlp[LAST]
            ds[cn] = raw_fin - iso0
        d_spec = ds['same'] - 0.5 * (ds['func'] + ds['null'])
        n2 = float(np.linalg.norm(d_spec))
        return float(abs(d_spec @ cdir) / max(n2, 1e-30))

    # ---------- pass 1: base cls + alignment curve ----------
    cls_base_list = []
    share_map = {li: [] for li in SCAN_LAYERS}
    for i, (cat, w) in enumerate(target_list):
        w_tid = tid(w)
        cdir = dW_unit[CAT_WORDS.index(cat)]
        conds = conds_for(i, cat, w)
        hs_iso, _, _, opj_iso = forward_capture([w_tid], 0)
        iso0 = hs_iso[0]
        ds = {}
        opj_conds = {}
        for cn, toks in conds.items():
            hs, attn, mlp, opj = forward_capture(toks, 1)
            raw_fin = hs[LAST] + attn[LAST] + mlp[LAST]
            ds[cn] = raw_fin - iso0
            opj_conds[cn] = opj
        d_spec = ds['same'] - 0.5 * (ds['func'] + ds['null'])
        cls_base_list.append(float(abs(d_spec @ cdir)
                                   / max(np.linalg.norm(d_spec), 1e-30)))
        dh = {cn: {li: np.einsum('dkh,kh->dk', W3[li],
                                 opj_conds[cn][li].astype(np.float64)
                                 .reshape(32, 128)) for li in SCAN_LAYERS}
              for cn in opj_conds}
        dh_iso = {li: np.einsum('dkh,kh->dk', W3[li],
                                opj_iso[li].astype(np.float64)
                                .reshape(32, 128)) for li in SCAN_LAYERS}
        for li in SCAN_LAYERS:
            d_spec_h = ((dh['same'][li] - dh_iso[li]) - 0.5 * (
                (dh['func'][li] - dh_iso[li])
                + (dh['null'][li] - dh_iso[li])))
            n2 = float(np.linalg.norm(d_spec_h.sum(axis=1)))
            sh = d_spec_h.T @ cdir / max(n2, 1e-30)
            share_map[li].append(sh)
        print('P2839 pass1 [%d/%d] %s cls=%.4f'
              % (i + 1, len(target_list), w, cls_base_list[-1]),
              flush=True)

    cls_base_m = float(np.mean(cls_base_list))
    share_mean = {li: np.mean(share_map[li], axis=0) for li in SCAN_LAYERS}
    align_curve = np.array([
        float(np.abs(share_mean[li])[
            np.argsort(-np.abs(share_mean[li]))[:TOP_K]].sum())
        for li in SCAN_LAYERS])

    # ---------- arm A: group ablations (matched protocol) ----------
    def group_drop(groups):
        abl['groups'] = groups
        cls_vals = []
        for i, (cat, w) in enumerate(target_list):
            cls_vals.append(measure_cls(i, cat, w))
        abl['groups'] = None
        return 1.0 - float(np.mean(cls_vals)) / max(cls_base_m, 1e-30)

    drop22 = group_drop({22: top_heads_2838[22]})
    drop23 = group_drop({23: top_heads_2838[23]})
    drop1223 = group_drop({22: top_heads_2838[22],
                           23: top_heads_2838[23]})
    g_all = {li: top_heads_2838[li] for li in SCAN_LAYERS}
    drop_all = group_drop(g_all)
    print('P2839 drops: L22=%.4f L23=%.4f G22+23=%.4f G_all=%.4f'
          % (drop22, drop23, drop1223, drop_all), flush=True)

    denom = drop22 + drop23
    ratio = drop1223 / denom if denom > 1e-9 else float('nan')
    if 0.7 <= ratio <= 1.3:
        regime = 'independent'
    elif ratio > 1.3:
        regime = 'synergistic'
    else:
        regime = 'overlapping'
    a2 = drop_all > 0.15

    # ---------- arm B: dual-spectrum misalignment ----------
    caus_curve = np.array([res2838['drop_curve'][str(li)]
                           for li in SCAN_LAYERS])
    rho = spearman(align_curve, caus_curve)
    b1 = abs(rho) < 0.5
    align_peak = SCAN_LAYERS[int(np.argmax(align_curve))]
    caus_peak = SCAN_LAYERS[int(np.argmax(caus_curve))]

    v = {
        'cls_base_mean': round(cls_base_m, 6),
        'drop_L22': round(float(drop22), 4),
        'drop_L23': round(float(drop23), 4),
        'drop_G22_23': round(float(drop1223), 4),
        'drop_G_all16': round(float(drop_all), 4),
        'sum_singles': round(float(denom), 4),
        'additivity_ratio': round(float(ratio), 4),
        'regime': regime,
        'A2_distributed_sum_substantial': bool(a2),
        'align_peak_layer': align_peak,
        'caus_peak_layer_2838': caus_peak,
        'spearman_align_caus': round(float(rho), 4),
        'B1_dual_spectrum_misaligned': bool(b1),
        'final_verdict_b': ('misaligned_confirmed' if b1
                            else 'partially_aligned'),
    }

    result = {'phase': 2839, 'prereg': PREREG, 'verdict': v,
              'align_curve': {str(li): round(float(align_curve[j]), 5)
                              for j, li in enumerate(SCAN_LAYERS)},
              'caus_curve_2838': {str(li): round(float(caus_curve[j]), 5)
                                  for j, li in enumerate(SCAN_LAYERS)}}
    fc.save(OUT / 'result.json', result)

    fc.npz(OUT / 'spectra.npz',
           align_curve=align_curve.astype(np.float32),
           caus_curve_2838=caus_curve.astype(np.float32),
           **{('share_L%d' % li): share_mean[li].astype(np.float32)
              for li in SCAN_LAYERS})

    elapsed = time.monotonic() - t0
    cc.ledger('phase2839', elapsed)
    print('P2839 VERDICT %s' % json.dumps(v), flush=True)
    print('P2839 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
