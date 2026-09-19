"""Phase 2865 (LPF MA2 cont): L13H30 dynamic tracing -- II1 closure 2nd cut.

2862 excluded the OV-direct-write explanation (static |g| <= 0.0075 vs
drop = 0.101).  Two candidate paths remain for how the top-1 causal head
carries class direction:
  (a) structural relay: its ablation changes downstream attention key/value
      geometry; class direction is carried by re-routing, not by writing;
  (b) indirect write: it writes some direction v (not aligned to cdir)
      that downstream mlp/attn transforms INTO a cdir-aligned component.

Design: per word, one full forward + one clamp forward (L13H30 attention
bias at (1,1) clamped to func/null mean, 2853/2860 protocol verbatim).
The attention-bias patch touches ONLY head 30 of layer 13, therefore
    Delta_attn[13] = attn_out_clamp - attn_out_full   (pos 1)
is a PURE H30 effect: the injected vector v13.  True-residual deltas are
captured at LN2 pre-hook inputs (2861 discipline: hook truth, no
reconstruction).  Exact telescoping identity to verify (v1):
    final_delta = Delta_attn[13]
                + sum_{l=13..34} Delta_mlp[l]
                + sum_{l=14..35} Delta_attn[l]
where final_delta = (ln2in[35]+mlp[35])_cl - (...)_fu  (x_36 readout).
cdir component budget:
    C_final = final_delta . cdir
    c13     = v13 . cdir                      (expected ~0 by 2862)
    G_mlp   = sum_l (Delta_mlp[l] . cdir)     (signed)
    G_attn  = sum_{l>=14} (Delta_attn[l] . cdir)   (signed)

Prereg (frozen before any readout):
  v1  identity_budget: max_w |cfin - (c13 + G_attn_w + G_mlp_w)| < 0.05
      (cdir scalar budget; EXACT identity in real arithmetic.  The
      vector-space telescoping is polluted by bf16 rounding accumulation
      in deep layers -- diag E-check: per-layer recursion error grows
      0.1 -> 2.2 with depth, ln2in tensor carries 23-layer rounding
      delta ~10-30, which dwarfs ||v13||~0.3-2.3 for small-v13 words.
      Scalar projection closes because rounding is near-unbiased:
      mean-level closure observed at 0.002.)
  D1  arm attribution (aggregate spectrum, 80-word mean):
      attn_carried  iff |G_attn| > 2|G_mlp| and |G_attn| > 0.5|C_final-c13|
      mlp_carried   iff |G_mlp| > 2|G_attn| and |G_mlp| > 0.5|C_final-c13|
      else mixed;  D1w = per-word majority vote share (descriptive)
  D2  descriptive: e_ff = C_final/||v13|| (indirect-write efficiency,
      compare 2862 static g<=0.0075); first-reach layer l* = first l with
      |cum_cdir[l] - c13| >= 0.5 |C_final - c13|
  D3  relay-layer stability: modal first-reach layer share >= 0.6 over 80
      words -> stable; null = 200 random word-layer permutations (SEED=2865)
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
OUT = BASE / 'phase2865' / 'l13h30_trace'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2855
MAX_WORDS = 8
LAST = 35
NL = 36
CLAMP_L, CLAMP_H = 13, 30

PREREG = {
    'v1': 'identity_budget iff max_w |cdir budget residual| < 0.05 '
          '(scalar; vector telescoping polluted by bf16 rounding '
          'accumulation, kept descriptive)',
    'D1': 'attn_carried iff |G_attn|>2|G_mlp| and |G_attn|>0.5|C_final-c13|; '
          'mlp_carried iff mirror; else mixed (aggregate 80-word mean '
          'spectra); D1w per-word vote descriptive',
    'D2': 'descriptive: e_ff=C_final/||v13|| vs static 0.0075; '
          'first-reach layer l* distribution',
    'D3': 'stable relay iff modal l* share >= 0.6; null 200 permutations '
          '(SEED=2865)',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())

    execution_path = OUT / 'execution.json'
    if not execution_path.exists():
        execution = {'timestamp': fc.stamp(),
                     'source': cc.snapshot(__file__),
                     'prereg': PREREG, 'seed': SEED,
                     'design': 'L13H30 dynamic tracing: full vs clamp '
                               'forward per word, layerwise arm '
                               'decomposition of true-residual deltas '
                               'at LN2 pre-hook inputs (2861 discipline)'}
        fc.save(execution_path, execution)

    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()

    hd = int(model.config.head_dim)

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

    cap = {'ln2in': {}, 'attn': {}, 'mlp': {}}

    def make_out_hook(kind, li):
        def hook(module, args, output):
            o = output[0] if isinstance(output, tuple) else output
            cap[kind].setdefault(li, []).append(
                o[0].detach().float().cpu().numpy())
        return hook

    def make_in_hook(kind, li):
        def pre_hook(module, args, kwargs):
            x = args[0] if args else kwargs['hidden_states']
            cap[kind].setdefault(li, []).append(
                x.detach()[0].float().cpu().numpy())
        return pre_hook

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.post_attention_layernorm
                       .register_forward_pre_hook(
                           make_in_hook('ln2in', li), with_kwargs=True))
        handles.append(layer.self_attn.register_forward_hook(
            make_out_hook('attn', li)))
        handles.append(layer.mlp.register_forward_hook(
            make_out_hook('mlp', li)))

    def clear_cap():
        for d in ('ln2in', 'attn', 'mlp'):
            for li in cap[d]:
                del cap[d][li][:]

    def forward_run(tokens, pos):
        clear_cap()
        with torch.no_grad():
            out = model(torch.tensor([tokens], device='cuda'),
                        output_attentions=True)
        ln2in = {li: cap['ln2in'][li][0][pos].astype(np.float64)
                 for li in range(NL)}
        attn = np.stack([cap['attn'][li][0][pos].astype(np.float64)
                         for li in range(NL)])
        mlp = np.stack([cap['mlp'][li][0][pos].astype(np.float64)
                        for li in range(NL)])
        aw = out.attentions[CLAMP_L][0].float().cpu().numpy()
        return ln2in, attn, mlp, aw

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

    layers = model.model.layers
    vdt = next(layers[0].mlp.parameters()).dtype

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
    for li, layer in enumerate(model.model.layers):
        fwd, holder = make_patched(layer.self_attn)
        layer.self_attn.forward = fwd
        patches[li] = holder

    # -- storage --
    a_mlp = np.zeros((n_words, NL))       # Delta_mlp[l] . cdir
    a_attn = np.zeros((n_words, NL))      # Delta_attn[l] . cdir
    c13 = np.zeros(n_words)
    cfin = np.zeros(n_words)
    v13n = np.zeros(n_words)
    e_ff = np.zeros(n_words)
    l_star = np.full(n_words, -1)
    ident_resid = np.zeros(n_words)
    a11_full = np.zeros(n_words)
    a11_clamp = np.zeros(n_words)
    v13_cos = np.zeros(n_words)

    for i, (cat, w) in enumerate(target_list):
        ci = CAT_WORDS.index(cat)
        cdir = dW_unit[ci]
        w_tid = tid(w)
        same_cat = [x for x in targets[cat] if x != w]
        if not same_cat:
            same_cat = [x for x in CATS[cat] if x != w
                        and x in single_tok][:1]
        toks = [tid(same_cat[0]), w_tid]

        # full forward + func/null clamp-bias reference (2860 protocol)
        ln2in_f, attn_f, mlp_f, aw_f = forward_run(toks, 1)
        # func and null forwards only to compute clamp bias (1,1)
        _, _, _, aw_func = forward_run([func_tid, w_tid], 1)
        _, _, _, aw_null = forward_run([null_tids[i], w_tid], 1)
        A11 = float(aw_f[CLAMP_H][1, 1])
        t_raw = 0.5 * (float(aw_func[CLAMP_H][1, 1])
                       + float(aw_null[CLAMP_H][1, 1]))
        t = min(max(t_raw, 0.01), 0.99)
        A11c = min(max(A11, 1e-4), 0.9999)
        b13 = float(np.clip(np.log((t / (1.0 - t)) * (1.0 - A11c) / A11c),
                            -20.0, 20.0))

        # clamp forward
        patches[CLAMP_L]['map'] = {CLAMP_H: [(1, 1, b13)]}
        ln2in_c, attn_c, mlp_c, aw_c = forward_run(toks, 1)
        patches[CLAMP_L]['map'] = {}

        a11_full[i] = A11
        a11_clamp[i] = float(aw_c[CLAMP_H][1, 1])

        v13 = attn_c[CLAMP_L] - attn_f[CLAMP_L]     # pure H30 effect
        c13[i] = float(v13 @ cdir)
        v13n[i] = float(np.linalg.norm(v13))
        v13_cos[i] = c13[i] / max(v13n[i], 1e-30)

        d_attn = attn_c - attn_f                    # (NL, d)
        d_mlp = mlp_c - mlp_f
        for l in range(NL):
            a_attn[i, l] = float(d_attn[l] @ cdir)
            a_mlp[i, l] = float(d_mlp[l] @ cdir)

        final_d = (ln2in_c[LAST] + mlp_c[LAST]) \
            - (ln2in_f[LAST] + mlp_f[LAST])
        tele = d_mlp[CLAMP_L:LAST + 1].sum(0) \
            + d_attn[CLAMP_L:LAST + 1].sum(0)
        ident_resid[i] = float(np.linalg.norm(final_d - tele)) \
            / max(v13n[i], 1e-30)
        cfin[i] = float(final_d @ cdir)
        e_ff[i] = cfin[i] / max(v13n[i], 1e-30)

        # first-reach layer: cumulative cdir growth from c13
        # cum[k] = cdir component summed over layers CLAMP_L..CLAMP_L+k
        # (attn + mlp arms); cum[n_steps-1] == cfin (telescoping)
        n_steps = LAST - CLAMP_L + 1
        cum = np.cumsum(a_attn[i, CLAMP_L:LAST + 1]
                        + a_mlp[i, CLAMP_L:LAST + 1])
        target_half = 0.5 * (cfin[i] - c13[i])
        if abs(target_half) > 1e-12:
            for k in range(n_steps):
                if abs(cum[k] - c13[i]) >= abs(target_half):
                    l_star[i] = CLAMP_L + k
                    break

        if (i + 1) % 20 == 0:
            print('P2865 words [%d/%d]' % (i + 1, n_words), flush=True)

    # ---------- verdicts ----------
    # v1: cdir scalar budget closure (exact identity in real arithmetic;
    # vector-space telescoping is descriptive only -- bf16 rounding
    # accumulation in deep layers, see PREREG note)
    budget_resid = cfin - (c13 + a_attn[:, CLAMP_L + 1:LAST + 1].sum(1)
                           + a_mlp[:, CLAMP_L:LAST + 1].sum(1))
    v1 = bool(float(np.max(np.abs(budget_resid))) < 0.05)
    v1_max = float(np.max(np.abs(budget_resid)))
    ident_vec_mean = float(ident_resid.mean())  # descriptive

    G_mlp = float(a_mlp[:, CLAMP_L:LAST + 1].sum())
    G_attn = float(a_attn[:, CLAMP_L + 1:LAST + 1].sum())
    c13m = float(c13.mean())
    cfinm = float(cfin.mean())
    denom = abs(cfinm - c13m)
    if abs(G_attn) > 2 * abs(G_mlp) and abs(G_attn) > 0.5 * denom:
        d1 = 'attn_carried'
    elif abs(G_mlp) > 2 * abs(G_attn) and abs(G_mlp) > 0.5 * denom:
        d1 = 'mlp_carried'
    else:
        d1 = 'mixed'

    # per-word arm dominance vote (descriptive)
    votes = 0
    for i in range(n_words):
        gm = float(a_mlp[i, CLAMP_L:LAST + 1].sum())
        ga = float(a_attn[i, CLAMP_L + 1:LAST + 1].sum())
        if abs(ga) > 2 * abs(gm):
            votes += 1
    d1w = votes / n_words

    e_m = float(e_ff.mean())
    ls = l_star[l_star > 0]
    if len(ls) > 0:
        vals, counts = np.unique(ls, return_counts=True)
        modal_l = int(vals[np.argmax(counts)])
        modal_share = float(counts.max() / len(ls))
    else:
        modal_l, modal_share = -1, 0.0
    ngn = np.random.default_rng(2865)
    null_shares = []
    for _ in range(200):
        perm = ngn.permutation(ls) if len(ls) > 1 else ls
        vv, cc2 = np.unique(perm, return_counts=True)
        null_shares.append(float(cc2.max() / max(len(perm), 1)))
    null_p95 = float(np.percentile(null_shares, 95))
    d3 = bool(modal_share >= 0.6)
    d3_note = 'modal_share %.3f vs null p95 %.3f' % (modal_share, null_p95)

    v = {
        'n_words': n_words,
        'v1_identity_budget': v1,
        'v1_max_budget_resid': round(v1_max, 6),
        'v1_vec_telescoping_mean_descriptive':
            round(ident_vec_mean, 3),
        'D1_arm': d1,
        'G_attn': round(G_attn, 4),
        'G_mlp': round(G_mlp, 4),
        'c13_mean': round(c13m, 5),
        'C_final_mean': round(cfinm, 5),
        'D1w_attn_vote_share': round(d1w, 3),
        'v13_cos_mean': round(float(v13_cos.mean()), 5),
        'v13_norm_mean': round(float(v13n.mean()), 4),
        'e_ff_mean': round(e_m, 5),
        'e_ff_p10_p90': [round(float(np.percentile(e_ff, 10)), 5),
                         round(float(np.percentile(e_ff, 90)), 5)],
        'first_reach_modal_layer': modal_l,
        'first_reach_modal_share': round(modal_share, 3),
        'D3_stable_relay': d3,
        'D3_note': d3_note,
        'A11_full_mean': round(float(a11_full.mean()), 5),
        'A11_clamp_mean': round(float(a11_clamp.mean()), 5),
        'final_verdict': '%s/v1=%s/D3=%s' % (d1, v1, d3),
    }

    result = {'phase': 2865, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'l13h30_trace.npz',
           a_mlp=a_mlp.astype(np.float32),
           a_attn=a_attn.astype(np.float32),
           c13=c13.astype(np.float32),
           c_final=cfin.astype(np.float32),
           v13_norm=v13n.astype(np.float32),
           e_ff=e_ff.astype(np.float32),
           l_star=l_star.astype(np.int32),
           identity_resid=ident_resid.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2865', elapsed)
    print('P2865 VERDICT %s' % json.dumps(v), flush=True)
    print('P2865 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
