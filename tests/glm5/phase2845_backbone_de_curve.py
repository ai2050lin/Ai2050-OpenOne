"""Phase 2845 (LPF Delta-II/III): non-gate backbone homogeneity + DE
state formation curve + DE multi-category hub matrix.

2844 closed the gate line: attention backbone (100%) = small gate pool
(~9%, additive, saturating) + non-gate bulk (~91%).  This phase opens
the bulk and the DE hub:

Arm A -- sample 20 non-gate heads uniformly from layers 20-35
  (excluding the 15 registered gate heads of phase2844).  For each:
  (a) 2842-style source decomposition of its cls_spec contribution in
      the 2-token window [cond, w] (pos0 = cond token, pos1 = w self),
      (b) single-head QK clamp drop (2843/2844 pipeline, head clamped
      to its own func/null self-mass).
  Prereg:
  H1  backbone_isomorphic_sources iff >= 60% of sampled heads are
      self_dominant (self share >= 0.9 of |cdir source contributions|)
  H2  non_gate_small iff mean |clamp drop| of sampled heads < 0.02
  verdict_a: backbone_isomorphic iff H1 and H2; heterogeneous else

Arm B -- DE-state formation curve: for [cond, DE, w], the cdir
  projection delta_de(l) = cdir . (h_DE_same(l) - 0.5(h_DE_func(l) +
  h_DE_null(l))) for every layer l = 0..35 (DE position, input states
  via hidden_states at that position).  formation layer = first l with
  mean delta_de(l) >= 0.5 * max_l mean delta_de(l).
  B1  early_formation iff formation_layer <= 24
  B2  consistent_at_formation iff >= 70% of words have delta_de > 0
      at the formation layer
  verdict_b: hub_early_formation iff B1 and B2

Arm C -- DE multi-category hub: delta_de projected on all 10 category
  directions dW_unit (80 words x 10 directions).
  C1  multi_category_hub iff >= 8/10 directions have >= 70% positive
      words
  C2  selectivity = mean over words of max_dir / sum_dir |delta_de|
      (descriptive)
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
OUT = BASE / 'phase2845' / 'backbone_de_curve'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2844_RESULT = BASE / 'phase2844' / 'joint_gate_ladder' / 'result.json'
SEED = 2845
N_SAMPLE = 20
MAX_WORDS = 8
LAST = 35
GATE_LAYERS = [22, 23, 26, 28, 33]

PREREG = {
    'H1': 'backbone_isomorphic_sources iff >= 60% of 20 sampled '
          'non-gate heads are self_dominant (self share >= 0.9)',
    'H2': 'non_gate_small iff mean |clamp drop| of sampled heads < 0.02',
    'verdict_a': 'backbone_isomorphic iff H1 and H2; heterogeneous else',
    'B1': 'early_formation iff formation_layer (first l with mean '
          'delta_de >= 0.5*peak) <= 24',
    'B2': 'consistent_at_formation iff >= 70% words delta_de > 0 at '
          'formation layer',
    'verdict_b': 'hub_early_formation iff B1 and B2',
    'C1': 'multi_category_hub iff >= 8/10 directions have >= 70% '
          'positive words (DE delta projection per direction)',
    'C2': 'selectivity = mean max_dir/sum_dir |delta_de| (descriptive)',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())
    res2844 = json.loads(SRC_2844_RESULT.read_text(encoding='utf-8'))
    gate_keys = set(res2844['verdict']['rank_order']) \
        if 'rank_order' in res2844['verdict'] else set()

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED,
                 'gate_keys_excluded': sorted(gate_keys)}
    fc.save(OUT / 'execution.json', execution)

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()

    hd = int(model.config.head_dim)
    nh = int(model.config.num_attention_heads)
    n_kv = int(model.config.num_key_value_heads)
    group = nh // n_kv

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

    cap = {'sain': {}, 'attn': {}, 'mlp': {}}

    def make_out_hook(kind, li):
        def hook(module, args, output):
            o = output[0] if isinstance(output, tuple) else output
            cap[kind].setdefault(li, []).append(
                o[0].detach().float().cpu().numpy())
        return hook

    def make_in_hook(li):
        def pre_hook(module, args, kwargs):
            x = args[0] if args else kwargs['hidden_states']
            cap['sain'].setdefault(li, []).append(
                x.detach()[0].float().cpu().numpy())
        return pre_hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.register_forward_hook(
            make_out_hook('attn', li)))
        handles.append(layer.mlp.register_forward_hook(
            make_out_hook('mlp', li)))
        handles.append(layer.self_attn.register_forward_pre_hook(
            make_in_hook(li), with_kwargs=True))

    def clear_cap():
        for d in ('sain', 'attn', 'mlp'):
            for li in cap[d]:
                del cap[d][li][:]

    def forward_run(tokens, pos):
        clear_cap()
        with torch.no_grad():
            out = model(torch.tensor([tokens], device='cuda'),
                        output_hidden_states=True, output_attentions=True)
            hs = np.stack([h[0, pos, :].float().cpu().numpy()
                           for h in out.hidden_states])
        attn = np.stack([cap['attn'][li][0][pos] for li in range(36)])
        mlp = np.stack([cap['mlp'][li][0][pos] for li in range(36)])
        aw = {li: out.attentions[li][0].float().cpu().numpy()
              for li in range(20, 36)}
        sain = {li: cap['sain'][li][0] for li in range(20, 36)}
        return hs, attn, mlp, aw, sain

    # ---------- targets (identical construction to 2843/2844) ----------
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
        targets[cat] = [w for w in CATS[cat]
                        if w in single_tok][:MAX_WORDS]
    target_list = [(cat, w) for cat in CAT_WORDS for w in targets[cat]]
    n_words = len(target_list)

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
    null_tids = {}
    while len(null_tids) < n_words:
        r = int(rng.integers(0, vocab_size))
        if r not in word_tids and r > 0:
            null_tids[len(null_tids)] = r
    func_tid = tid('the')
    de_tid = tid('的')

    # ---------- Arm A: sample 20 non-gate heads from L20-35 ----------
    rng2 = np.random.default_rng(SEED + 1)
    pool = []
    for li in range(20, 36):
        for h in range(nh):
            k = 'L%d_h%d' % (li, h)
            if k not in gate_keys:
                pool.append((li, h))
    sample_idx = rng2.choice(len(pool), N_SAMPLE, replace=False)
    SAMPLED = [pool[int(i)] for i in sample_idx]

    OV = {}
    for li in range(20, 36):
        Wo = model.model.layers[li].self_attn.o_proj.weight \
            .detach().float().cpu().numpy().astype(np.float64)
        Wo3 = Wo.reshape(Wo.shape[0], nh, hd)
        for h in range(nh):
            OV[(li, h)] = Wo3[:, h, :]          # (4096, 128)

    def conds2_for(i, cat, w):
        w_tid = tid(w)
        same_cat = [x for x in targets[cat] if x != w]
        if not same_cat:
            same_cat = [x for x in CATS[cat] if x != w
                        and x in single_tok][:1]
        return {'same': [tid(same_cat[0]), w_tid],
                'func': [func_tid, w_tid],
                'null': [null_tids[i], w_tid]}

    def conds3_for(i, cat, w):
        c2 = conds2_for(i, cat, w)
        return {cn: [t[0], de_tid, t[1]] for cn, t in c2.items()}

    def make_patched(sa):
        orig = sa.forward
        holder = {'map': {}}
        scaling = sa.scaling
        import transformers.models.qwen3.modeling_qwen3 as q3

        def forward(hidden_states, position_embeddings,
                    attention_mask=None, past_key_values=None, **kw):
            if not holder['map']:
                return orig(hidden_states, position_embeddings,
                            attention_mask, past_key_values, **kw)
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, hd)
            q = sa.q_norm(sa.q_proj(hidden_states)
                          .view(hidden_shape)).transpose(1, 2)
            k = sa.k_norm(sa.k_proj(hidden_states)
                          .view(hidden_shape)).transpose(1, 2)
            v = sa.v_proj(hidden_states).view(hidden_shape) \
                .transpose(1, 2)
            cos, sin = position_embeddings
            q, k = q3.apply_rotary_pos_emb(q, k, cos, sin)
            k = q3.repeat_kv(k, sa.num_key_value_groups)
            v = q3.repeat_kv(v, sa.num_key_value_groups)
            aw = torch.matmul(q, k.transpose(2, 3)) * scaling
            L = aw.shape[-1]
            causal = torch.full((L, L), torch.finfo(aw.dtype).min,
                                device=aw.device, dtype=aw.dtype).triu(1)
            aw = aw + causal
            for h, blist in holder['map'].items():
                for (qi, ki, b) in blist:
                    aw[0, h, qi, ki] = aw[0, h, qi, ki] + b
            aw = torch.softmax(aw, dim=-1)
            out = torch.matmul(aw, v).transpose(1, 2) \
                .reshape(*input_shape, -1)
            out = sa.o_proj(out)
            return out, aw
        return forward, holder

    patches = {}
    for li in range(20, 36):
        sa = model.model.layers[li].self_attn
        fwd, holder = make_patched(sa)
        sa.forward = fwd
        patches[li] = holder

    def bias_for(aw2, li, h):
        A11 = float(aw2['same'][li][h][1, 1])
        t = 0.5 * (float(aw2['func'][li][h][1, 1])
                   + float(aw2['null'][li][h][1, 1]))
        t = min(max(t, 0.01), 0.99)
        A11c = min(max(A11, 1e-4), 0.9999)
        b = float(np.log((t / (1.0 - t)) * (1.0 - A11c) / A11c))
        return float(np.clip(b, -20.0, 20.0))

    vproj = {li: model.model.layers[li].self_attn.v_proj
             for li in range(20, 36)}

    self_shares = []          # per sampled head: mean self share
    drops_head = {k: [] for k in range(N_SAMPLE)}
    clamp_resid = []
    de_curve_words = []       # per word: (36,) mean-candidate curve
    de_curves_10 = []         # per word: (36, 10) all-layer direction matrix

    for i, (cat, w) in enumerate(target_list):
        cdir = dW_unit[CAT_WORDS.index(cat)]
        w_tid = tid(w)
        c2 = conds2_for(i, cat, w)
        hs_iso, attn_iso, mlp_iso, _, _ = forward_run([w_tid], 0)
        iso0 = hs_iso[LAST] + attn_iso[LAST] + mlp_iso[LAST]
        raw2, aw2, sain2 = {}, {}, {}
        for cn, toks in c2.items():
            hs, attn, mlp, awc, sac = forward_run(toks, 1)
            raw2[cn] = (hs[LAST] + attn[LAST] + mlp[LAST]) - iso0
            aw2[cn] = awc
            sain2[cn] = sac
        d_spec_full = raw2['same'] - 0.5 * (raw2['func'] + raw2['null'])
        nfull = max(float(np.linalg.norm(d_spec_full)), 1e-30)
        cls_base = float(abs(d_spec_full @ cdir)) / nfull

        # ---- source decomposition per sampled head (from base runs)
        for si, (li, h) in enumerate(SAMPLED):
            kv = h // group
            s_dir = {}
            for cn in c2:
                vin = sain2[cn][li][:2]                    # (2, 4096)
                v = vproj[li](torch.tensor(
                    vin, device='cuda',
                    dtype=next(vproj[li].parameters()).dtype)
                ).detach().float().cpu().numpy().astype(np.float64)
                Vh = v.reshape(2, n_kv, hd)[:, kv, :]      # (2, 128)
                Wm = Vh @ OV[(li, h)].T                    # (2, 4096)
                a = aw2[cn][li][h][1, :2]                  # (2,)
                s_dir[cn] = a[:, None] * Wm                # (2, 4096)
            ssp = s_dir['same'] - 0.5 * (s_dir['func'] + s_dir['null'])
            s0 = float(ssp[0] @ cdir)
            s1 = float(ssp[1] @ cdir)
            denom = abs(s0) + abs(s1)
            self_shares.append(
                float(s1 / denom) if denom > 1e-30 else 0.0)

        # ---- single-head clamp drops
        for si, (li, h) in enumerate(SAMPLED):
            b = bias_for(aw2, li, h)
            patches[li]['map'] = {h: [(1, 1, b)]}
            hs, attn, mlp, awc, _ = forward_run(c2['same'], 1)
            patches[li]['map'] = {}
            clamp_resid.append(
                abs(float(awc[li][h][1, 1])
                    - 0.5 * (float(aw2['func'][li][h][1, 1])
                             + float(aw2['null'][li][h][1, 1]))))
            dsc = (hs[LAST] + attn[LAST] + mlp[LAST]) - iso0 \
                - 0.5 * (raw2['func'] + raw2['null'])
            cls_c = float(abs(dsc @ cdir)) / nfull
            drops_head[si].append(
                (cls_base - cls_c) / max(cls_base, 1e-30))

        # ---- Arm B/C: DE-state all-layer curve + 10-direction matrix
        c3 = conds3_for(i, cat, w)
        proj = {}
        for cn, toks in c3.items():
            hs, _, _, _, _ = forward_run(toks, 1)   # pos1 = DE token
            S = hs[:36] @ dW_unit.T                       # (36, 10)
            proj[cn] = S                                  # (36, 10)
        D = proj['same'] - 0.5 * (proj['func'] + proj['null'])
        de_curves_10.append(D)                            # (36, 10)
        de_curve_words.append(D[:, CAT_WORDS.index(cat)])
        if (i + 1) % 10 == 0:
            print('P2845 words [%d/%d]' % (i + 1, n_words), flush=True)

    # ---------- verdicts ----------
    per_head_share = np.zeros(N_SAMPLE)
    for si in range(N_SAMPLE):
        vals = [self_shares[si + j * N_SAMPLE]
                for j in range(n_words)]
        per_head_share[si] = float(np.mean(vals))
    frac_self_dom = float(np.mean(per_head_share >= 0.9))
    h1 = frac_self_dom >= 0.60

    mean_drop = np.array([float(np.mean(drops_head[si]))
                          for si in range(N_SAMPLE)])
    mean_abs_drop = float(np.mean(np.abs(mean_drop)))
    h2 = mean_abs_drop < 0.02
    verdict_a = 'backbone_isomorphic' if (h1 and h2) else 'heterogeneous'

    curves = np.stack(de_curve_words)               # (n_words, 36)
    curve_mean = curves.mean(0)
    peak = float(curve_mean.max())
    form_layer = int(next(l for l in range(36)
                          if curve_mean[l] >= 0.5 * peak))
    b1 = form_layer <= 24
    frac_pos_form = float(np.mean(curves[:, form_layer] > 0))
    b2 = frac_pos_form >= 0.70
    verdict_b = ('hub_early_formation' if (b1 and b2)
                 else 'late_or_inconsistent')

    dmat = np.stack([D[22] for D in de_curves_10])  # (n_words, 10) @ L22
    frac_pos_dir = np.array([float(np.mean(dmat[:, di] > 0))
                             for di in range(10)])
    c1 = bool(np.sum(frac_pos_dir >= 0.70) >= 8)
    sums = np.abs(dmat).sum(1)
    c2_sel = float(np.mean(np.abs(dmat).max(1)
                           / np.maximum(sums, 1e-30)))

    v = {
        'n_words': n_words,
        'n_sampled_heads': N_SAMPLE,
        'sampled_heads': ['L%d_h%d' % t for t in SAMPLED],
        'frac_self_dominant': round(frac_self_dom, 4),
        'H1_backbone_isomorphic_sources': bool(h1),
        'mean_abs_clamp_drop': round(mean_abs_drop, 5),
        'H2_non_gate_small': bool(h2),
        'verdict_a': verdict_a,
        'per_head_self_share': {('L%d_h%d' % SAMPLED[si]):
                                round(float(per_head_share[si]), 4)
                                for si in range(N_SAMPLE)},
        'per_head_mean_drop': {('L%d_h%d' % SAMPLED[si]):
                               round(float(mean_drop[si]), 5)
                               for si in range(N_SAMPLE)},
        'clamp_max_resid': round(float(np.max(clamp_resid)), 5),
        'formation_layer': form_layer,
        'B1_early_formation': bool(b1),
        'frac_pos_at_formation': round(frac_pos_form, 4),
        'B2_consistent_at_formation': bool(b2),
        'verdict_b': verdict_b,
        'de_curve_mean_peak_layer': int(np.argmax(curve_mean)),
        'de_curve_mean_peak_val': round(peak, 6),
        'frac_pos_per_direction': {CAT_WORDS[di]:
                                   round(float(frac_pos_dir[di]), 4)
                                   for di in range(10)},
        'C1_multi_category_hub': bool(c1),
        'C2_selectivity_mean': round(c2_sel, 4),
    }

    result = {'phase': 2845, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)

    fc.npz(OUT / 'backbone_de.npz',
           per_head_self_share=per_head_share.astype(np.float64),
           per_head_mean_drop=mean_drop.astype(np.float64),
           de_curves=curves.astype(np.float64),
           de_curves_10=np.stack(de_curves_10).astype(np.float64),
           de_dir_matrix=dmat.astype(np.float64))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2845', elapsed)
    print('P2845 VERDICT %s' % json.dumps(v), flush=True)
    print('P2845 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
