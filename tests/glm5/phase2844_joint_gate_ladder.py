"""Phase 2844 (LPF Delta-II/III): joint gate clamp ladder + DE anchoring.

2843 showed single gates are real (dose-response rho=0.75) but small
(1.47% mean).  This phase closes the gate line:

Arm A -- joint clamp ladder: cumulative clamp of the 15 target heads
  (ranked by 2841 registered single-head drops; rungs at 1/3/6/10/15),
  each head clamped to its own func/null self-mass via exact logit
  bias (2843 pipeline); plus joint clamp of 5 random control heads.
  Prereg:
  J1  collective_substantial iff joint_top15_drop >= 0.15
  J2  gate_additive iff joint_top15_drop / max(sum_singles,1e-9)
      in [0.7, 1.3]   (singles re-measured in-run, same words)
  J3  monotone iff drop(1) < drop(3) < drop(6) < drop(10) < drop(15)
  J4  rand_joint_small iff joint_rand5_drop < 0.05
  verdict_a: collective_gate_causal iff J1 and J3 and J4;
             redundant_gate_pool iff J3 and (not J1); mixed otherwise

Arm B -- DE-anchoring: in [cond, DE, w], is the DE token's state at
  L22 input conditioned by the category word (inheritance via earlier
  attention)?  delta_de = cdir . (h_DE_same - 0.5(h_DE_func +
  h_DE_null)); sign consistency over 80 words; ratio vs the same
  delta computed for w's state.
  A2  anchor_inheritance iff >= 70% of words have delta_de > 0
  A3  ratio |delta_de| / |delta_w| reported
C1  observational: Spearman(dgate_dir, cls_base_dir) reported
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
OUT = BASE / 'phase2844' / 'joint_gate_ladder'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2841_NPZ = BASE / 'phase2841' / 'longtail_shape' / 'head_drop_map.npz'
SEED = 2844
SCAN_LAYERS = [22, 23, 26, 28, 33]
TOPK = 3
N_RAND = 1
MAX_WORDS = 8
LAST = 35
RUNGS = [1, 3, 6, 10, 15]

PREREG = {
    'J1': 'collective_substantial iff joint_top15_drop >= 0.15',
    'J2': 'gate_additive iff joint_top15_drop/sum_singles in [0.7,1.3]',
    'J3': 'monotone iff drop(1)<drop(3)<drop(6)<drop(10)<drop(15)',
    'J4': 'rand_joint_small iff joint_rand5_drop < 0.05',
    'verdict_a': 'collective_gate_causal iff J1 and J3 and J4; '
                 'redundant_gate_pool iff J3 and (not J1); mixed else',
    'A2': 'anchor_inheritance iff >= 70% of words delta_de > 0 '
          "(cdir projection of DE-state at L22 input, same vs ctrl)",
    'A3': 'ratio mean |delta_de| / |delta_w| reported',
    'C1': 'observational Spearman(dgate_dir, cls_base_dir) reported',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())

    z41 = np.load(SRC_2841_NPZ)
    HEADS = {li: [int(h) for h in np.argsort(-z41['drops_L%d' % li])[:TOPK]]
             for li in SCAN_LAYERS}
    rng0 = np.random.default_rng(SEED)
    RAND = {li: [int(h) for h in rng0.choice(
        [h for h in range(32) if h not in HEADS[li]], N_RAND,
        replace=False)] for li in SCAN_LAYERS}
    # 15-head rank order by 2841 registered drops
    all15 = [('L%d_h%d' % (li, h), float(z41['drops_L%d' % li][h]))
             for li in SCAN_LAYERS for h in HEADS[li]]
    all15.sort(key=lambda kv: -kv[1])
    rank_order = [k for k, _ in all15]
    key2lh = {k: (int(k.split('_')[0][1:]), int(k.split('_h')[1]))
              for k in rank_order}
    rand_keys = [('L%d_h%d' % (li, h)) for li in SCAN_LAYERS
                 for h in RAND[li]]

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED,
                 'rank_order': rank_order}
    fc.save(OUT / 'execution.json', execution)

    import torch
    import transformers.models.qwen3.modeling_qwen3 as q3
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()
    n_kv = int(model.config.num_key_value_heads)

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

    patches = {}

    def make_patched(sa):
        orig = sa.forward
        holder = {'map': {}}
        scaling = sa.scaling
        hd = sa.head_dim

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

    for li in SCAN_LAYERS:
        sa = model.model.layers[li].self_attn
        fwd, holder = make_patched(sa)
        sa.forward = fwd
        patches[li] = holder

    cap = {'attn': {}, 'mlp': {}, 'sain': {}}

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
        for d in ('attn', 'mlp', 'sain'):
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
              for li in SCAN_LAYERS}
        sain = {li: cap['sain'][li][0] for li in SCAN_LAYERS}
        return hs, attn, mlp, aw, sain

    # ---------- targets (identical construction to 2843) ----------
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

    rng = np.random.default_rng(SEED + 1)
    vocab_size = W_U.shape[0]
    word_tids = set(tc.values())
    null_tids = {}
    while len(null_tids) < n_words:
        r = int(rng.integers(0, vocab_size))
        if r not in word_tids and r > 0:
            null_tids[len(null_tids)] = r
    func_tid = tid('the')
    de_tid = tid('的')

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

    def bias_for(aw2, li, h):
        A11 = float(aw2['same'][li][h][1, 1])
        t = 0.5 * (float(aw2['func'][li][h][1, 1])
                   + float(aw2['null'][li][h][1, 1]))
        t = min(max(t, 0.01), 0.99)
        A11c = min(max(A11, 1e-4), 0.9999)
        b = float(np.log((t / (1.0 - t)) * (1.0 - A11c) / A11c))
        return float(np.clip(b, -20.0, 20.0)), t

    cls_base_w = []
    singles = {k: [] for k in rank_order}
    rung_drops = {r: [] for r in RUNGS}
    joint_rand_drops = []
    dde_w = []
    ddw_w = []
    dgate_dir_words = {cat: [] for cat in CAT_WORDS}
    clamp_resid = []

    for i, (cat, w) in enumerate(target_list):
        cdir = dW_unit[CAT_WORDS.index(cat)]
        w_tid = tid(w)
        c2 = conds2_for(i, cat, w)
        hs_iso, _, _, _, _ = forward_run([w_tid], 0)
        iso0 = hs_iso[0]
        raw2, aw2 = {}, {}
        for cn, toks in c2.items():
            hs, attn, mlp, awc, _ = forward_run(toks, 1)
            raw2[cn] = (hs[LAST] + attn[LAST] + mlp[LAST]) - iso0
            aw2[cn] = awc
        d_spec_full = raw2['same'] - 0.5 * (raw2['func'] + raw2['null'])
        nfull = max(float(np.linalg.norm(d_spec_full)), 1e-30)
        cls_base = float(abs(d_spec_full @ cdir)) / nfull
        cls_base_w.append(cls_base)
        dgate_dir_words[cat].append(
            float(np.mean([aw2['same'][22][h][1, 1]
                           - 0.5 * (aw2['func'][22][h][1, 1]
                                    + aw2['null'][22][h][1, 1])
                           for h in HEADS[22]])))

        biases = {}
        for k in rank_order:
            li, h = key2lh[k]
            biases[k] = bias_for(aw2, li, h)

        # in-run singles
        for k in rank_order:
            li, h = key2lh[k]
            b, t = biases[k]
            patches[li]['map'] = {h: [(1, 1, b)]}
            hs, attn, mlp, awc, _ = forward_run(c2['same'], 1)
            patches[li]['map'] = {}
            dsc = ((hs[LAST] + attn[LAST] + mlp[LAST]) - iso0) \
                - 0.5 * (raw2['func'] + raw2['null'])
            cls_c = float(abs(dsc @ cdir)) / nfull
            singles[k].append((cls_base - cls_c) / max(cls_base, 1e-30))
            clamp_resid.append(abs(float(awc[li][h][1, 1]) - t))

        # joint rungs (cumulative)
        for r in RUNGS:
            rkeys = rank_order[:r]
            m = {}
            for k in rkeys:
                li, h = key2lh[k]
                b, _ = biases[k]
                m.setdefault(li, []).append((h, 1, 1, b))
            for li in SCAN_LAYERS:
                patches[li]['map'] = {h: [(1, 1, bb)]
                                      for (h, _, _, bb) in m.get(li, [])}
            hs, attn, mlp, awc, _ = forward_run(c2['same'], 1)
            for li in SCAN_LAYERS:
                patches[li]['map'] = {}
            dsc = ((hs[LAST] + attn[LAST] + mlp[LAST]) - iso0) \
                - 0.5 * (raw2['func'] + raw2['null'])
            cls_c = float(abs(dsc @ cdir)) / nfull
            rung_drops[r].append((cls_base - cls_c) / max(cls_base, 1e-30))

        # joint random control (all 5 rand heads together)
        m = {}
        for k in rand_keys:
            li, h = key2lh.get(k, (int(k.split('_')[0][1:]),
                                   int(k.split('_h')[1])))
            b, _ = bias_for(aw2, li, h)
            m.setdefault(li, []).append((h, 1, 1, b))
        for li in SCAN_LAYERS:
            patches[li]['map'] = {h: [(1, 1, bb)]
                                  for (h, _, _, bb) in m.get(li, [])}
        hs, attn, mlp, awc, _ = forward_run(c2['same'], 1)
        for li in SCAN_LAYERS:
            patches[li]['map'] = {}
        dsc = ((hs[LAST] + attn[LAST] + mlp[LAST]) - iso0) \
            - 0.5 * (raw2['func'] + raw2['null'])
        cls_c = float(abs(dsc @ cdir)) / nfull
        joint_rand_drops.append((cls_base - cls_c) / max(cls_base, 1e-30))

        # Arm B: DE-state conditioning at L22 input (3tok)
        c3 = conds3_for(i, cat, w)
        st = {}
        for cn, toks in c3.items():
            hs, attn, mlp, awc, sain = forward_run(toks, 2)
            st[cn] = sain[22]                     # (3, 4096) L22 input
        d_de = float(st['same'][1] @ cdir) \
            - 0.5 * (float(st['func'][1] @ cdir)
                     + float(st['null'][1] @ cdir))
        d_w = float(st['same'][2] @ cdir) \
            - 0.5 * (float(st['func'][2] @ cdir)
                     + float(st['null'][2] @ cdir))
        dde_w.append(d_de)
        ddw_w.append(d_w)
        if (i + 1) % 10 == 0:
            print('P2844 words [%d/%d]' % (i + 1, n_words), flush=True)

    # ---------- verdicts ----------
    mean_single = {k: float(np.mean(singles[k])) for k in rank_order}
    sum_singles = float(sum(mean_single.values()))
    rung_means = {r: float(np.mean(rung_drops[r])) for r in RUNGS}
    joint15 = rung_means[15]
    joint_rand = float(np.mean(joint_rand_drops))
    ratio = joint15 / max(sum_singles, 1e-9)

    j1 = joint15 >= 0.15
    j2 = 0.7 <= ratio <= 1.3
    vals = [rung_means[r] for r in RUNGS]
    j3 = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    j4 = joint_rand < 0.05
    if j1 and j3 and j4:
        verdict_a = 'collective_gate_causal'
    elif j3 and (not j1):
        verdict_a = 'redundant_gate_pool'
    else:
        verdict_a = 'mixed'

    dde_arr = np.array(dde_w)
    frac_pos = float(np.mean(dde_arr > 0))
    a2 = frac_pos >= 0.70
    a3 = float(np.mean(np.abs(dde_arr)) / max(np.mean(np.abs(ddw_w)), 1e-30))

    # C1: per-direction dgate vs cls_base
    dgate_dir = {cat: float(np.mean(v)) for cat, v in
                 dgate_dir_words.items()}
    base_dir = {}
    for cat in CAT_WORDS:
        ws = [idx for idx, (c, _) in enumerate(target_list) if c == cat]
        base_dir[cat] = float(np.mean([cls_base_w[j] for j in ws]))
    c1 = spearman(np.array([dgate_dir[c] for c in CAT_WORDS]),
                  np.array([base_dir[c] for c in CAT_WORDS]))

    v = {
        'n_words': n_words,
        'cls_base_mean': round(float(np.mean(cls_base_w)), 6),
        'clamp_max_resid': round(float(np.max(clamp_resid)), 5),
        'rung_drops': {str(r): round(rung_means[r], 4) for r in RUNGS},
        'sum_singles': round(sum_singles, 4),
        'joint_top15_drop': round(joint15, 4),
        'joint_rand5_drop': round(joint_rand, 4),
        'additivity_ratio': round(ratio, 4),
        'J1_collective_substantial': bool(j1),
        'J2_gate_additive': bool(j2),
        'J3_monotone': bool(j3),
        'J4_rand_joint_small': bool(j4),
        'verdict_a': verdict_a,
        'de_anchor_frac_pos': round(frac_pos, 4),
        'A2_anchor_inheritance': bool(a2),
        'A3_de_over_w_ratio': round(a3, 4),
        'C1_spearman_dgate_clsbase': round(c1, 4),
        'dgate_per_dir': {k: round(x, 5) for k, x in
                          sorted(dgate_dir.items(), key=lambda kv: -kv[1])},
        'clsbase_per_dir': {k: round(x, 5) for k, x in
                            sorted(base_dir.items(),
                                   key=lambda kv: -kv[1])},
    }

    result = {'phase': 2844, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)

    fc.npz(OUT / 'ladder.npz',
           rung_curve=np.array([rung_means[r] for r in RUNGS],
                               dtype=np.float64),
           singles=np.array([mean_single[k] for k in rank_order],
                            dtype=np.float64),
           dde=dde_arr.astype(np.float64),
           ddw=np.array(ddw_w, dtype=np.float64))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2844', elapsed)
    print('P2844 VERDICT %s' % json.dumps(v), flush=True)
    print('P2844 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
