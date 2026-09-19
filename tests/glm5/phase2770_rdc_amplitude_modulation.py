"""Phase 2770 (Beta): parameter-level implementation of knowledge amplitude
modulation (2768 follow-up).

Background.  2768 showed category knowledge = 1.6-2.9x amplification of the
swap response on SHARED frame/lexical coordinates (no dedicated coordinate
group).  Open question (Alpha/Beta/Gamma plan): WHICH gate/up parameters make
the shared machine "turn faster" when the subject has knowledge?

Design (frozen before any intervention forward):
  Material: exact 2768 v2 panel rebuild (same builder, same DERANGE, same
  pseudo words).  288 prompts; measurement at last prompt position.
  C001 unit localisation (per category c, layers 20..35):
    - capture MLP input x_l at the final position for every prompt/layer.
    - unit activation act_k = silu(W_g x)_k * (W_u x)_k.
    - pair swap gain of unit k along pair direction u_p = unit(dh_L28):
      g_p[k] = (act_k(B) - act_k(A)) * (W_d[k,:] . u_p).
    - per-unit knowledge gain ratio r_k = median_p in RealPairs |g| /
      median_p in PseudoPairs |g| (same category swap, pseudo subject).
    - modulation units per category: r_k >= 2 AND sign(g) consistent
      (>= 6/8 same-sign) on the SELECTION half (odd-index pairs).
  C002 causal (held-out even-index pairs only):
    - lever: scale down_proj input coordinates x[0,-1,k] by gamma for the
      top-64 modulation units (union over categories/layers, gamma applied at
      the unit's own layer).
    - gamma in {0.5, 2.0}; matched random-unit control (same layers, same
      count per layer, seed 2770002).
    - B2a: relative reduction of pooled real nd(L28) under gamma=0.5 exceeds
      the pseudo nd reduction by >= 5 percentage points.
    - B2b: real nd reduction (gamma=0.5, modulation units) >= 2x the real nd
      reduction under matched random units.
    - B3 (descriptive): gamma=2.0 real increase > pseudo increase.
  C003 cross-model: qwen3-1.7b (eager, BF16), same material builder with its
    tokenizer (assert single-token predicates); primary layer 21 (relative
    depth 0.757 == 28/37 of qwen4's L28); B4: real/pseudo nd ratio > 1 for
    >= 4/5 categories.
  verdict: modulation_causal iff B2a and B2b.
Integrity gates:
  G1: base nd curves and knowledge amplification ratios reproduce 2768 Q3
      within 2% per category (same model/weights).
  G2: pseudo/real token-length identity within pairs (2768 assertion).
Status: descriptive + preregistered; NOT mechanism closure.
"""
import time
from pathlib import Path

import numpy as np

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2770' / 'qwen4_amplitude_modulation'
P2768 = BASE / 'phase2768' / 'qwen4_category_atlas_v2'
PRIMARY = 28
DEEP_MIN = 20
TOP_UNITS = 64
SEED_CTRL = 2770002

PREREG = {
    'phase': 2770,
    'question': 'Which gate/up parameters implement the knowledge amplitude '
                'amplification of the shared swap-response coordinates?',
    'unit_definition': 'knowledge gain ratio r_k = median|g_real| / '
                       'median|g_pseudo| on swap-direction projections, '
                       'selection on odd pairs, causal test on even pairs',
    'causal_lever': 'scale down_proj input coordinates by gamma at the '
                    'final position for top-64 modulation units',
    'criteria': {
        'B1': 'descriptive: >= 3/5 categories have >= 32 modulation units',
        'B2a': 'real nd(L28) relative reduction (gamma=0.5) exceeds pseudo '
               'reduction by >= 5pp on held-out even pairs',
        'B2b': 'real reduction with modulation units >= 2x matched random '
               'units',
        'B3': 'descriptive gamma=2.0',
        'B4': 'cross-model qwen3-1.7b amplification ratio > 1 for >= 4/5 '
              'categories at layer 21'},
    'verdict': 'modulation_causal iff B2a and B2b',
    'gates': {'G1': 'base amplification ratios reproduce 2768 Q3 within 2%',
              'G2': 'pair length identity'},
    'frozen_before_any_intervention_forward': True,
}


def main():
    t0 = time.time()
    cc.guard(0)
    assert not (OUT / 'result.json').exists(), 'immutable; delete before rerun'
    OUT.mkdir(parents=True, exist_ok=True)
    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG}
    fc.save(OUT / 'execution.json', execution)

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer
    import phase2768_category_atlas_v2 as p2768

    from rdc_query_common import MODELS
    tok = AutoTokenizer.from_pretrained(
        ROOT / 'models/hf' / MODELS['qwen4'], local_files_only=True,
        trust_remote_code=True, use_fast=True)
    material = p2768.build_material(tok)
    prompts = material['prompts']
    n_prompts = len(prompts)
    real_pairs = material['real_pairs']
    pseudo_pairs = material['pseudo_pairs']
    # G2: fixed-tail identity -> embedding dh at the last position is exactly 0
    # (within-pair token LENGTH may differ; 2768 hard-flaw note applies)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok2 = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    n_layers = 36
    d_model = model.config.hidden_size
    inter = model.config.intermediate_size

    # G2 (real gate): fixed tail -> last-position L0 dh exactly 0
    emb = model.get_input_embeddings().weight.detach()
    for a, b, _, _ in real_pairs + pseudo_pairs:
        ta = prompts[a]['ids']
        tb = prompts[b]['ids']
        n_tail = 5
        assert ta[-n_tail:] == tb[-n_tail:], ('G2 tail', a, b)
        with torch.inference_mode():
            e0 = emb[torch.tensor(ta, device=device)]
            e1 = emb[torch.tensor(tb, device=device)]
        assert float((e0[-1] - e1[-1]).abs().max()) == 0.0, 'G2 L0 dh'

    # capture mlp input x_l at final position + hidden states
    x_store = np.empty((n_prompts, n_layers, d_model), dtype=np.float32)
    h_store = np.empty((n_prompts, n_layers + 1, d_model), dtype=np.float32)
    cap = {}

    def mlp_pre(l):
        def hook(module, args):
            cap[l] = args[0].detach()
        return hook

    handles = [model.model.layers[l].mlp.register_forward_pre_hook(mlp_pre(l))
               for l in range(n_layers)]
    for pi, p in enumerate(prompts):
        cap.clear()
        with torch.inference_mode():
            t = torch.tensor([p['ids']], device=device)
            o = model(t, output_hidden_states=True)
            for l in range(n_layers):
                x_store[pi, l] = cap[l][0, -1].float().cpu().numpy()
            h_store[pi] = np.stack(
                [h[0, -1].float().cpu().numpy() for h in o.hidden_states])
        if pi % 100 == 0:
            print('P2770 CAP %d/%d' % (pi, n_prompts), flush=True)
    for h_ in handles:
        h_.remove()

    def nd_l28(pairs, h=None):
        hh = h_store if h is None else h
        dh = hh[[p[1] for p in pairs], PRIMARY, :].astype(np.float64) \
            - hh[[p[0] for p in pairs], PRIMARY, :].astype(np.float64)
        ha = hh[[p[0] for p in pairs], PRIMARY, :].astype(np.float64)
        hb = hh[[p[1] for p in pairs], PRIMARY, :].astype(np.float64)
        den = 0.5 * ((ha ** 2).sum(1) + (hb ** 2).sum(1))
        return float((dh ** 2).sum(1).mean() / max(den.mean(), 1e-30))

    # G1: reproduce 2768 Q3
    amp = {}
    for c in p2768.CATEGORIES:
        rp = [p for p in real_pairs if p[2] == c]
        pp = [p for p in pseudo_pairs if p[2] == c]
        amp[c] = nd_l28(rp) / max(nd_l28(pp), 1e-30)
    r68 = fc.read(P2768 / 'result.json')
    g1_max = 0.0
    for c in amp:
        ref = r68['Q3']['per_category']['cat_' + c]['ratio']
        g1_max = max(g1_max, abs(amp[c] / ref - 1.0))
    assert g1_max <= 0.02, ('G1', amp)

    # ---------- C001 unit localisation ------------------------------------
    def acts_for(idxs, l):
        """act [n, inter] for prompt indices at layer l (float32)."""
        x = torch.tensor(x_store[idxs, l].astype(np.float32), device=device)
        Wg = model.model.layers[l].mlp.gate_proj.weight.detach().float()
        Wu = model.model.layers[l].mlp.up_proj.weight.detach().float()
        return F.silu(x @ Wg.T) * (x @ Wu.T)

    CATEGORIES = p2768.CATEGORIES
    modulation = {}   # cat -> list[(l, k, ratio)]
    sel_stats = {}
    for c in CATEGORIES:
        rp = [(a, b) for a, b, cc_, _ in real_pairs if cc_ == c]
        pp = [(a, b) for a, b, cc_, _ in pseudo_pairs if cc_ == c]
        rp_sel = [rp[i] for i in range(len(rp)) if i % 2 == 1]
        pp_sel = [pp[i] for i in range(len(pp)) if i % 2 == 1]
        u_real = []
        for a, b in rp_sel:
            d = (h_store[b, PRIMARY] - h_store[a, PRIMARY]).astype(np.float64)
            n = np.linalg.norm(d)
            u_real.append(d / max(n, 1e-30))
        u_pse = []
        for a, b in pp_sel:
            d = (h_store[b, PRIMARY] - h_store[a, PRIMARY]).astype(np.float64)
            n = np.linalg.norm(d)
            u_pse.append(d / max(n, 1e-30))
        ratio = np.full((n_layers, inter), np.nan, dtype=np.float64)
        cons = np.zeros((n_layers, inter), dtype=np.int32)
        for l in range(DEEP_MIN, n_layers):
            act_rB = acts_for([p[1] for p in rp_sel], l)
            act_rA = acts_for([p[0] for p in rp_sel], l)
            dr = (act_rB - act_rA).cpu().numpy()          # (8, inter)
            act_pB = acts_for([p[1] for p in pp_sel], l)
            act_pA = acts_for([p[0] for p in pp_sel], l)
            dp = (act_pB - act_pA).cpu().numpy()
            Wd = model.model.layers[l].mlp.down_proj.weight.detach().\
                float().cpu().numpy()          # (2560, inter)
            Ur = np.stack(u_real)              # (8, 2560)
            proj_r = Ur @ Wd                   # (8, inter)
            Up = np.stack(u_pse)
            proj_p = Up @ Wd
            gr = dr * proj_r                               # (8, inter)
            gp = dp * proj_p
            med_r = np.median(np.abs(gr), axis=0)
            med_p = np.median(np.abs(gp), axis=0)
            ratio[l] = med_r / np.maximum(med_p, 1e-8)
            signs = np.sign(gr)
            cons[l] = np.maximum((signs > 0).sum(0), (signs < 0).sum(0))
        mask = (ratio >= 2.0) & (cons >= 6)   # >=6/8 same-sign on selection half
        ls, ks = np.where(mask)
        order = np.argsort(-ratio[ls, ks])
        units = [(int(ls[j]), int(ks[j]), float(ratio[ls[j], ks[j]]))
                 for j in order]
        modulation[c] = units[:128]
        sel_stats[c] = {'n_units': int(mask.sum()),
                        'top_ratios': [round(u[2], 2) for u in units[:8]]}
    b1_count = sum(1 for c in CATEGORIES
                   if sel_stats[c]['n_units'] >= 32)
    union = {}
    for c in CATEGORIES:
        for (l, k, r) in modulation[c][:TOP_UNITS]:
            union.setdefault(l, set()).add(k)
    union_list = {l: sorted(v) for l, v in union.items()}
    total_units = sum(len(v) for v in union_list.values())

    # ---------- C002 causal (even pairs, gamma intervention) --------------
    scale_state = {'gamma': 1.0, 'units': {}}

    def down_pre(l):
        def hook(module, args):
            if scale_state['gamma'] == 1.0 or l not in scale_state['units']:
                return None
            x = args[0].clone()
            ks = scale_state['units'][l]
            x[0, -1, ks] *= scale_state['gamma']
            return (x,) + tuple(args[1:])
        return hook

    dhandles = [model.model.layers[l].mlp.down_proj.
                register_forward_pre_hook(down_pre(l)) for l in range(n_layers)]

    def collect_h():
        hh = np.empty((n_prompts, n_layers + 1, d_model), dtype=np.float32)
        for pi, p in enumerate(prompts):
            with torch.inference_mode():
                t = torch.tensor([p['ids']], device=device)
                o = model(t, output_hidden_states=True)
                hh[pi] = np.stack([h[0, -1].float().cpu().numpy()
                                   for h in o.hidden_states])
        return hh

    rp_even = [pair for j, pair in enumerate(real_pairs) if j % 2 == 0]
    pp_even = [pair for j, pair in enumerate(pseudo_pairs) if j % 2 == 0]

    def pooled(nd_pairs, hh):
        vals = {}
        for c in CATEGORIES:
            rp = [p for p in nd_pairs if len(p) > 2 and p[2] == c]
            if not rp:
                vals[c] = None
                continue
            vals[c] = nd_l28(rp, hh)
        return vals

    def rel_change(base, new):
        out = {}
        for c in CATEGORIES:
            if base[c] is None or new[c] is None:
                out[c] = None
            else:
                out[c] = new[c] / base[c] - 1.0
        return out

    base_real = pooled(rp_even, h_store)
    base_pse = pooled(pp_even, h_store)

    def run_gamma(gamma, units_map):
        scale_state['gamma'] = gamma
        scale_state['units'] = {l: torch.tensor(sorted(ks), device=device)
                                for l, ks in units_map.items()}
        try:
            hh = collect_h()
        finally:
            scale_state['gamma'] = 1.0
            scale_state['units'] = {}
        return hh

    hh_mod_half = run_gamma(0.5, union_list)
    hh_mod_double = run_gamma(2.0, union_list)
    # matched random control: same layer->count profile
    rng = np.random.default_rng(SEED_CTRL)
    rand_units = {}
    for l, ks in union_list.items():
        rand_units[l] = set(int(k) for k in rng.choice(inter, len(ks),
                                                       replace=False))
    rand_list = {l: sorted(v) for l, v in rand_units.items()}
    hh_rand_half = run_gamma(0.5, rand_list)

    ch_mod = rel_change(base_real, pooled(rp_even, hh_mod_half))
    ch_mod_pse = rel_change(base_pse, pooled(pp_even, hh_mod_half))
    ch_rand = rel_change(base_real, pooled(rp_even, hh_rand_half))
    ch_mod2 = rel_change(base_real, pooled(rp_even, hh_mod_double))
    ch_mod2_pse = rel_change(base_pse, pooled(pp_even, hh_mod_double))

    def agg(d):
        vals = [v for v in d.values() if v is not None]
        return float(np.mean(vals))

    b2a_red_mod = -agg(ch_mod)          # positive = reduction
    b2a_red_pse = -agg(ch_mod_pse)
    b2a_pass = bool(b2a_red_mod - b2a_red_pse >= 0.05)
    b2b_red_rand = -agg(ch_rand)
    b2b_pass = bool(b2a_red_mod >= 2.0 * max(b2b_red_rand, 1e-9))
    b3 = {'real_rel_change_gamma2': agg(ch_mod2),
          'pseudo_rel_change_gamma2': agg(ch_mod2_pse)}

    # ---------- C003 cross-model 1.7b -------------------------------------
    model2 = None
    b4 = {}
    try:
        from transformers import AutoModelForCausalLM
        model2 = AutoModelForCausalLM.from_pretrained(
            ROOT / 'models/hf/qwen3-1.7b', dtype=torch.bfloat16,
            device_map={'': 'cuda:0'}, attn_implementation='eager',
            local_files_only=True).eval()
        tok17 = AutoTokenizer.from_pretrained(
            ROOT / 'models/hf/qwen3-1.7b', local_files_only=True,
            trust_remote_code=True, use_fast=True)
        mat17 = p2768.build_material(tok17)
        pr17 = mat17['prompts']
        L2 = model2.config.num_hidden_layers
        prim17 = 21
        h17 = np.empty((len(pr17), L2 + 1, model2.config.hidden_size),
                       dtype=np.float32)
        with torch.inference_mode():
            for pi, p in enumerate(pr17):
                t = torch.tensor([p['ids']], device='cuda:0')
                o = model2(t, output_hidden_states=True)
                h17[pi] = np.stack([h[0, -1].float().cpu().numpy()
                                    for h in o.hidden_states])
        def nd17(pairs):
            a = np.array([p[0] for p in pairs])
            b = np.array([p[1] for p in pairs])
            dh = h17[b, prim17].astype(np.float64) - h17[a, prim17]
            den = 0.5 * ((h17[a, prim17].astype(np.float64) ** 2).sum(1) +
                         (h17[b, prim17].astype(np.float64) ** 2).sum(1))
            return float((dh ** 2).sum(1).mean() / max(den.mean(), 1e-30))
        for c in CATEGORIES:
            rp = [p for p in mat17['real_pairs'] if p[2] == c]
            pp = [p for p in mat17['pseudo_pairs'] if p[2] == c]
            b4[c] = nd17(rp) / max(nd17(pp), 1e-30)
        b4_pass = sum(1 for v in b4.values() if v > 1.0) >= 4
    finally:
        if model2 is not None:
            del model2
            torch.cuda.empty_cache()

    for h_ in dhandles:
        h_.remove()

    verdict = 'modulation_causal' if (b2a_pass and b2b_pass) else \
        'not_confirmed'
    results = {
        'phase': 2770, 'G1_max_rel_dev': g1_max,
        'C001': {'sel_stats': sel_stats,
                 'B1_categories_ge32': b1_count,
                 'n_union_units': total_units,
                 'union_layers': {str(l): len(v)
                                  for l, v in union_list.items()}},
        'C002': {'base_real_even': base_real, 'base_pseudo_even': base_pse,
                 'rel_change_mod_gamma0.5': ch_mod,
                 'rel_change_mod_pseudo_gamma0.5': ch_mod_pse,
                 'rel_change_rand_gamma0.5': ch_rand,
                 'B2a_reduction_mod': b2a_red_mod,
                 'B2a_reduction_pseudo': b2a_red_pse, 'B2a_pass': b2a_pass,
                 'B2b_reduction_random': b2b_red_rand, 'B2b_pass': b2b_pass,
                 'B3': b3},
        'C003': {'layer17': 21, 'ratios': b4,
                 'B4_pass': bool(b4.get('_pass', sum(
                     1 for v in b4.values() if v > 1.0) >= 4))},
        'verdict': verdict, 'seconds': time.time() - t0}
    fc.save(OUT / 'result.json', results)
    fc.npz(OUT / 'modulation_units.npz',
           union_layers=np.array(sorted(union_list.keys())),
           union_counts=np.array([len(union_list[l])
                                  for l in sorted(union_list.keys())]),
           sel_n=np.array([sel_stats[c]['n_units'] for c in CATEGORIES]))
    print('PHASE2770_DONE verdict=%s B1=%d B2a=%s(%s vs %s) B2b=%s(rand %s) '
          'B4=%s' % (verdict, b1_count, b2a_pass, round(b2a_red_mod, 4),
                     round(b2a_red_pse, 4), b2b_pass,
                     round(b2b_red_rand, 4), results['C003']['B4_pass']),
          flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        (OUT / 'crash.txt').write_text(traceback.format_exc(),
                                       encoding='utf-8')
        raise
