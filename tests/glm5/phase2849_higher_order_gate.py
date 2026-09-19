"""Phase 2849 (LPF MA2 cont): higher-order interaction decomposition +
gate-closure causal test.

From 2848: early-cluster pairs are additive (ratios ~1.0) but the
5-head joint is sub-additive (0.65) -> saturation must come from
higher-order interactions.  And clamping the FORMER L13 h30 reduces
amplifier self-gates (3/4 heads >10%).  Two closure questions:
  G-order  at what subset size does saturation appear?  Full subset
           enumeration C(5,k) with in-run single-head denominators.
  G-gate   is the gate drop CAUSALLY sufficient?  Clamp the 4
           amplifier self-gates directly to their measured values
           under L13-h30 clamp (logit bias to a target A11), and test
           how much of the L13 h30 behavioral drop is reproduced.

Prereg (frozen before any readout):
  G1  higher_order_saturation iff mean ratio(3) < mean ratio(2) - 0.10
      AND mean ratio(5) < mean ratio(2) - 0.20
      (ratio(k) = joint_k / sum of in-run singles over the subset)
  G2  gate_sufficient_interface iff joint gate-clamp drop (own
      direction) >= 0.6 * L13h30 clamp drop
  verdict: closure_confirmed iff G2 (G1 descriptive decomposition)
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
OUT = BASE / 'phase2849' / 'higher_order_gate'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2849
MAX_WORDS = 8
LAST = 35
NL = 36
NH = 32

EARLY = [(13, 30), (2, 31), (5, 25), (5, 26), (4, 4)]
LATE = [(22, 28), (23, 29), (26, 4), (34, 15)]

PREREG = {
    'G1': 'higher_order_saturation iff aggregate mean ratio(3) < '
          'ratio(2) - 0.10 AND ratio(5) < ratio(2) - 0.20 '
          '(ratio(k) = mean joint_k / mean sum-of-in-run-singles; '
          'Gen1 word-level ratio was denominator-pathological, '
          'verdict voided and recomputed on aggregate ratio)',
    'G2': 'gate_sufficient_interface iff joint gate-clamp drop '
          '(own dir) >= 0.6 * L13h30 clamp drop',
    'verdict': 'closure_confirmed iff G2',
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
                     'design': 'C(5,k) full enumeration + amplifier '
                               'gate closure, 80 words'}
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
        attn = np.stack([cap['attn'][li][0][pos] for li in range(NL)])
        mlp = np.stack([cap['mlp'][li][0][pos] for li in range(NL)])
        aw = {li: out.attentions[li][0].float().cpu().numpy()
              for li in range(NL)}
        sain = {li: cap['sain'][li][0] for li in range(NL)}
        return hs, attn, mlp, aw, sain

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

    def set_maps(spec):
        for li in patches:
            patches[li]['map'] = {}
        for (li, h, b) in spec:
            patches[li]['map'][h] = [(1, 1, b)]

    def bias_to(aw2, li, h, target):
        """logit bias moving A[1,1] to `target`."""
        A11 = float(aw2['same'][li][h][1, 1])
        t = min(max(target, 0.01), 0.99)
        A11c = min(max(A11, 1e-4), 0.9999)
        b = float(np.log((t / (1.0 - t)) * (1.0 - A11c) / A11c))
        return float(np.clip(b, -20.0, 20.0))

    def bias_for(aw2, li, h):
        """clamp A[1,1] to the matched func/null mean (2843 style)."""
        t = 0.5 * (float(aw2['func'][li][h][1, 1])
                   + float(aw2['null'][li][h][1, 1]))
        return bias_to(aw2, li, h, t)

    def bias_match(aw2, awt, li, h):
        """logit bias moving A[1,1] to its value in awt (clamped ref)."""
        return bias_to(aw2, li, h, float(awt[li][h][1, 1]))

    from itertools import combinations

    SUBSETS = []
    for k in range(2, 6):
        SUBSETS.extend(combinations(range(5), k))   # 10+10+5+1

    singles_w = np.zeros((n_words, 5))
    joint_w = np.zeros((n_words, len(SUBSETS)))
    drop1330_w = np.zeros(n_words)
    gate_w = np.zeros((n_words, len(LATE)))
    gate_joint_w = np.zeros(n_words)
    resid_all = []
    resid_gate = []

    for i, (cat, w) in enumerate(target_list):
        ci = CAT_WORDS.index(cat)
        w_tid = tid(w)
        same_cat = [x for x in targets[cat] if x != w]
        if not same_cat:
            same_cat = [x for x in CATS[cat] if x != w
                        and x in single_tok][:1]
        c2 = {'same': [tid(same_cat[0]), w_tid],
              'func': [func_tid, w_tid],
              'null': [null_tids[i], w_tid]}

        hs_iso, attn_iso, mlp_iso, _, _ = forward_run([w_tid], 0)
        iso0 = hs_iso[LAST] + attn_iso[LAST] + mlp_iso[LAST]
        raw2, aw2 = {}, {}
        for cn, toks in c2.items():
            hs, attn, mlp, awc, _ = forward_run(toks, 1)
            raw2[cn] = (hs[LAST] + attn[LAST] + mlp[LAST]) - iso0
            aw2[cn] = awc
        d_spec_full = raw2['same'] \
            - 0.5 * (raw2['func'] + raw2['null'])
        nfull = max(float(np.linalg.norm(d_spec_full)), 1e-30)
        base_own = abs(float(d_spec_full @ dW_unit[ci])) / nfull

        def clamp_run(spec):
            set_maps(spec)
            hs, attn, mlp, awc, _ = forward_run(c2['same'], 1)
            set_maps([])
            for (li, h, b) in spec:
                resid_all.append(
                    abs(float(awc[li][h][1, 1])
                        - 0.5 * (float(aw2['func'][li][h][1, 1])
                                 + float(aw2['null'][li][h][1, 1]))))
            dsc = (hs[LAST] + attn[LAST] + mlp[LAST]) - iso0 \
                - 0.5 * (raw2['func'] + raw2['null'])
            return abs(float(dsc @ dW_unit[ci])) / nfull, awc

        # L13 h30 clamp (reference + gate targets)
        b13 = bias_for(aw2, 13, 30)
        cls_c13, awc13 = clamp_run([(13, 30, b13)])
        drop1330_w[i] = (base_own - cls_c13) / max(base_own, 1e-30)

        # gate targets: amplifier A11 under L13h30 clamp
        gate_spec = []
        for j, (li, h) in enumerate(LATE):
            a_cl = float(awc13[li][h][1, 1])
            gate_w[i, j] = (float(aw2['same'][li][h][1, 1]) - a_cl) \
                / max(float(aw2['same'][li][h][1, 1]), 1e-30)
            gate_spec.append((li, h, bias_match(aw2, awc13, li, h)))

        # singles
        singles = []
        for j, (li, h) in enumerate(EARLY):
            if (li, h) == (13, 30):
                singles.append(drop1330_w[i])
                continue
            b = bias_for(aw2, li, h)
            cls_c, _ = clamp_run([(li, h, b)])
            singles.append((base_own - cls_c) / max(base_own, 1e-30))
        singles_w[i] = singles

        # full subset enumeration
        for si, sub in enumerate(SUBSETS):
            spec = []
            for j in sub:
                li, h = EARLY[j]
                if (li, h) == (13, 30):
                    spec.append((13, 30, b13))
                else:
                    spec.append((li, h, bias_for(aw2, li, h)))
            cls_c, _ = clamp_run(spec)
            joint_w[i, si] = (base_own - cls_c) / max(base_own, 1e-30)

        # gate closure: clamp all 4 amplifier gates to post-former levels
        set_maps(gate_spec)
        hs, attn, mlp, awc_g, _ = forward_run(c2['same'], 1)
        set_maps([])
        for (li, h, b) in gate_spec:
            resid_gate.append(abs(float(awc_g[li][h][1, 1])
                                  - float(awc13[li][h][1, 1])))
        dsc = (hs[LAST] + attn[LAST] + mlp[LAST]) - iso0 \
            - 0.5 * (raw2['func'] + raw2['null'])
        cls_c = abs(float(dsc @ dW_unit[ci])) / nfull
        gate_joint_w[i] = (base_own - cls_c) / max(base_own, 1e-30)

        if (i + 1) % 10 == 0:
            print('P2849 words [%d/%d]' % (i + 1, n_words), flush=True)

    # ---------- verdicts ----------
    # aggregate ratio (robust): mean joint / mean sum-of-singles;
    # word-level median ratio reported as secondary (denominator-
    # pathological entries with sum-singles <= 0.05 excluded).
    ratio_by_k = {}
    rmed_by_k = {}
    for k in range(2, 6):
        idxs = [si for si, sub in enumerate(SUBSETS) if len(sub) == k]
        denom = np.zeros((n_words, len(idxs)))
        for q, si in enumerate(idxs):
            denom[:, q] = singles_w[:, list(SUBSETS[si])].sum(1)
        J = joint_w[:, idxs]
        ratio_by_k[k] = float(J.mean() / max(denom.mean(), 1e-30))
        valid = denom > 0.05
        if valid.sum() > 20:
            rmed_by_k[k] = float(np.median(
                (J / np.maximum(denom, 1e-30))[valid]))
        else:
            rmed_by_k[k] = None
    g1 = bool(ratio_by_k[3] < ratio_by_k[2] - 0.10
              and ratio_by_k[5] < ratio_by_k[2] - 0.20)

    g2_val = float(np.mean(gate_joint_w)) / max(
        float(np.mean(drop1330_w)), 1e-30)
    g2 = bool(g2_val >= 0.6)

    v = {
        'n_words': n_words,
        'ratio_by_k': {str(k): round(ratio_by_k[k], 4)
                       for k in ratio_by_k},
        'ratio_median_by_k': {str(k): (round(rmed_by_k[k], 4)
                                       if rmed_by_k[k] is not None
                                       else None)
                              for k in rmed_by_k},
        'G1_higher_order_saturation': g1,
        'mean_drop_L13h30': round(float(np.mean(drop1330_w)), 5),
        'mean_singles': [round(float(x), 5)
                         for x in singles_w.mean(0)],
        'G2_gate_sufficient_interface': g2,
        'gate_joint_drop': round(float(np.mean(gate_joint_w)), 5),
        'gate_closure_ratio': round(g2_val, 4),
        'gate_rel_drops_mean': [round(float(x), 4)
                                for x in gate_w.mean(0)],
        'max_resid_former_clamps': round(float(np.max(resid_all)), 5)
            if resid_all else None,
        'max_resid_gate_vs_target': round(float(np.max(resid_gate)), 5)
            if resid_gate else None,
        'final_verdict': 'closure_confirmed' if g2 else
            ('decomposed_only' if g1 else 'neither'),
    }

    result = {'phase': 2849, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'higher_order.npz',
           singles=singles_w.astype(np.float32),
           joint=joint_w.astype(np.float32),
           drop1330=drop1330_w.astype(np.float32),
           gate_joint=gate_joint_w.astype(np.float32),
           gate_rel=gate_w.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2849', elapsed)
    print('P2849 VERDICT %s' % json.dumps(v), flush=True)
    print('P2849 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
