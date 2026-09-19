"""Phase 2838 (LPF Delta-II/III): decisive-layer location + first
coordinate-level causal test.

2837 showed the L35 write-head cluster is NOT necessary for either
attribute modulation or induction copy (decoupled, 2.3% damage).
Hypothesis: the decisive layer sits earlier (2836 A2 predicted ~L30),
and within a head the causal unit is a small set of input coordinates.

Protocol:
  Pass 1 (base, capture): 20 words x {iso, same, func, null}; capture
  hidden states, per-layer attn/mlp outputs, per-layer o_proj inputs.
  Build per-layer per-head separated-share map; base cls_spec uses
  ctrl = 0.5*(func+null), raw final state (2836 discipline).
  Pass 2 (layer scan): for each layer l in 20..35, ablate that layer's
  own top-4 heads (by |share|) at that layer only; re-measure cls_spec
  with iso/same/func/null ALL under ablation (matched protocol)
  -> drop(l).
  Pass 3 (coordinates): at peak layer l*, top head h*: rank the 128
  input coordinates by mean |(W3[h][:,k].cdir) * a_k| over words;
  jointly ablate top-8 coords (and random-8 control) at (l*, h*),
  full matched protocol -> drops.

Prereg (frozen before any readout):
  D1  layer_resolved iff argmax_l drop(l) in [25, 35] and
      max_l drop(l) > 0.10
  D2  peak_damaging iff drop(peak) > 3 * drop(L35) (measured in-run)
  D3  decisive_before_final iff peak_layer <= 34
  verdict_a: decisive_layer_located iff D1 and D2 and D3
  N1  coordinates_carry iff drop(top8 coords at (l*,h*)) >=
      0.5 * drop(full h* head ablation at l*, measured in-run)
  verdict_b: coordinate_causal_unit iff N1 else distributed
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
OUT = BASE / 'phase2838' / 'decisive_layer_neuron'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2838
SCAN_LAYERS = list(range(20, 36))
LAST = 35
TOP_K = 4

PREREG = {
    'D1': 'layer_resolved iff argmax_l drop(l) in [25,35] and '
          'max_l drop(l) > 0.10',
    'D2': 'peak_damaging iff drop(peak) > 3 * drop(L35) measured in-run',
    'D3': 'decisive_before_final iff peak_layer <= 34',
    'verdict_a': 'decisive_layer_located iff D1 and D2 and D3',
    'N1': 'coordinates_carry iff drop(top8 coords) >= 0.5 * '
          'drop(full h* head) at (l*, h*), both measured in-run',
    'verdict_b': 'coordinate_causal_unit iff N1 else distributed',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())
    DOMAIN_OF = exec2806['domain_of']

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED,
                 'scan_layers': SCAN_LAYERS, 'top_k': TOP_K}
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

    # ---------- ablation control ----------
    abl = {'layer': None, 'heads': None, 'coords': None}

    def make_abl_hook(li):
        def pre_hook(module, args):
            if abl['layer'] != li:
                return None
            a = args[0]
            a2 = a.clone()
            v = a2.view(*a2.shape[:-1], 32, a2.shape[-1] // 32)
            if abl['heads'] is not None:
                v[..., list(abl['heads']), :] = 0
            if abl['coords'] is not None:
                for (h, k) in abl['coords']:
                    v[..., h, k] = 0
            return (a2,) + tuple(args[1:])
        return pre_hook

    for li in SCAN_LAYERS:
        model.model.layers[li].self_attn.o_proj \
            .register_forward_pre_hook(make_abl_hook(li))

    # ---------- capture ----------
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

    # ---------- targets ----------
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

    def measure_cls(i, cat, w, capture=False):
        """cls_spec under current ablation state; ctrl=0.5(func+null),
        iso re-measured under same state.  Optionally return opj of
        the 'same' forward (base state only)."""
        cdir = dW_unit[CAT_WORDS.index(cat)]
        w_tid = tid(w)
        conds = conds_for(i, cat, w)
        hs_iso, _, _, opj_iso = forward_capture([w_tid], 0)
        iso0 = hs_iso[0]
        ds = {}
        opj_same = None
        for cn, toks in conds.items():
            hs, attn, mlp, opj = forward_capture(toks, 1)
            raw_fin = hs[LAST] + attn[LAST] + mlp[LAST]
            ds[cn] = raw_fin - iso0
            if cn == 'same':
                opj_same = opj
        d_spec = ds['same'] - 0.5 * (ds['func'] + ds['null'])
        n2 = float(np.linalg.norm(d_spec))
        cls = float(abs(d_spec @ cdir) / max(n2, 1e-30))
        return cls, opj_iso, opj_same

    # ---------- pass 1: base capture + share map ----------
    share_map = {li: [] for li in SCAN_LAYERS}
    cls_base_list = []
    for i, (cat, w) in enumerate(target_list):
        cls, opj_iso, opj_same = measure_cls(i, cat, w, capture=True)
        cls_base_list.append(cls)
        dh_iso = {li: np.einsum('dkh,kh->dk', W3[li],
                                opj_iso[li].astype(np.float64)
                                .reshape(32, 128)) for li in SCAN_LAYERS}
        dh_same = {li: np.einsum('dkh,kh->dk', W3[li],
                                 opj_same[li].astype(np.float64)
                                 .reshape(32, 128)) for li in SCAN_LAYERS}
        # func/null opj not captured in measure_cls; capture separately
        conds = conds_for(i, cat, w)
        dh_fn = {}
        for cn in ('func', 'null'):
            hs, attn, mlp, opj_c = forward_capture(conds[cn], 1)
            dh_fn[cn] = {li: np.einsum('dkh,kh->dk', W3[li],
                                       opj_c[li].astype(np.float64)
                                       .reshape(32, 128))
                         for li in SCAN_LAYERS}
        cdir = dW_unit[CAT_WORDS.index(cat)]
        for li in SCAN_LAYERS:
            d_spec_h = ((dh_same[li] - dh_iso[li]) - 0.5 * (
                (dh_fn['func'][li] - dh_iso[li])
                + (dh_fn['null'][li] - dh_iso[li])))
            n2 = float(np.linalg.norm(d_spec_h.sum(axis=1)))
            sh = d_spec_h.T @ cdir / max(n2, 1e-30)
            share_map[li].append(sh)
        print('P2838 pass1 [%d/%d] %s cls_base=%.4f'
              % (i + 1, len(target_list), w, cls), flush=True)

    cls_base_m = float(np.mean(cls_base_list))
    share_mean = {li: np.mean(share_map[li], axis=0) for li in SCAN_LAYERS}
    top_heads = {li: list(np.argsort(-np.abs(share_mean[li]))[:TOP_K])
                 for li in SCAN_LAYERS}

    # ---------- pass 2: layer scan (matched protocol) ----------
    drop = {}
    for li in SCAN_LAYERS:
        abl['layer'] = li
        abl['heads'] = top_heads[li]
        abl['coords'] = None
        cls_vals = []
        for i, (cat, w) in enumerate(target_list):
            cls, _, _ = measure_cls(i, cat, w)
            cls_vals.append(cls)
        cls_abl = float(np.mean(cls_vals))
        drop[li] = 1.0 - cls_abl / max(cls_base_m, 1e-30)
        print('P2838 pass2 L%d top4=%s drop=%.4f'
              % (li, top_heads[li], drop[li]), flush=True)
    abl['layer'] = None
    abl['heads'] = None

    peak_layer = int(max(drop, key=drop.get))
    d1 = (25 <= peak_layer <= 35) and max(drop.values()) > 0.10
    d2 = drop[peak_layer] > 3 * max(drop[LAST], 1e-30)
    d3 = peak_layer <= 34
    verdict_a = ('decisive_layer_located' if (d1 and d2 and d3)
                 else 'not_located')

    # ---------- pass 3: coordinate-level causal test ----------
    l_star = peak_layer
    h_star = int(top_heads[l_star][0])
    coord_scores = np.zeros((len(target_list), 128))
    for i, (cat, w) in enumerate(target_list):
        cdir = dW_unit[CAT_WORDS.index(cat)]
        cls_r, opj_iso_r, opj_same_r = measure_cls(i, cat, w)
        a = opj_same_r[l_star].astype(np.float64).reshape(32, 128)[h_star]
        w_row = W3[l_star][:, h_star, :]  # (2560, 128)
        proj = np.abs(w_row.T @ cdir)  # (128,)
        coord_scores[i] = proj * np.abs(a)
    coord_rank = np.argsort(-coord_scores.mean(axis=0))
    top8 = [(h_star, int(k)) for k in coord_rank[:8]]
    rand8 = [(h_star, int(k)) for k in
             rng.choice(128, size=8, replace=False).tolist()]

    def drop_for_coords(coords):
        abl['layer'] = l_star
        abl['heads'] = None
        abl['coords'] = coords
        cls_vals = []
        for i, (cat, w) in enumerate(target_list):
            cls, _, _ = measure_cls(i, cat, w)
            cls_vals.append(cls)
        abl['coords'] = None
        return 1.0 - float(np.mean(cls_vals)) / max(cls_base_m, 1e-30)

    drop_top8 = drop_for_coords(top8)
    drop_rand8 = drop_for_coords(rand8)

    abl['layer'] = l_star
    abl['heads'] = [h_star]
    abl['coords'] = None
    cls_vals = []
    for i, (cat, w) in enumerate(target_list):
        cls, _, _ = measure_cls(i, cat, w)
        cls_vals.append(cls)
    abl['heads'] = None
    abl['layer'] = None
    full_head_drop = 1.0 - float(np.mean(cls_vals)) / max(cls_base_m, 1e-30)

    n1 = drop_top8 >= 0.5 * max(full_head_drop, 1e-30)
    verdict_b = 'coordinate_causal_unit' if n1 else 'distributed'

    v = {
        'peak_layer': peak_layer,
        'peak_drop': round(float(drop[peak_layer]), 4),
        'drop_L35': round(float(drop[LAST]), 4),
        'L35_top4_recovered': sorted(int(h) for h in top_heads[LAST]),
        'D1_layer_resolved': bool(d1),
        'D2_peak_damaging': bool(d2),
        'D3_decisive_before_final': bool(d3),
        'final_verdict_a': verdict_a,
        'l_star': l_star,
        'h_star': 'L%d h%d' % (l_star, h_star),
        'top8_coords': [[h, k] for h, k in top8],
        'drop_top8_coords': round(float(drop_top8), 4),
        'drop_rand8_coords': round(float(drop_rand8), 4),
        'full_head_drop': round(float(full_head_drop), 4),
        'N1_coordinates_carry': bool(n1),
        'final_verdict_b': verdict_b,
    }

    result = {'phase': 2838, 'prereg': PREREG, 'verdict': v,
              'drop_curve': {str(li): round(float(drop[li]), 5)
                             for li in SCAN_LAYERS},
              'top_heads_per_layer': {str(li): [int(h) for h in top_heads[li]]
                                      for li in SCAN_LAYERS}}
    fc.save(OUT / 'result.json', result)

    npz = {('share_L%d' % li): share_mean[li].astype(np.float32)
           for li in SCAN_LAYERS}
    npz['drop_curve'] = np.array([drop[li] for li in SCAN_LAYERS],
                                 dtype=np.float32)
    npz['coord_scores_mean'] = coord_scores.mean(axis=0).astype(np.float32)
    fc.npz(OUT / 'layer_coord_map.npz', **npz)

    elapsed = time.monotonic() - t0
    cc.ledger('phase2838', elapsed)
    print('P2838 VERDICT %s' % json.dumps(v), flush=True)
    print('P2838 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
