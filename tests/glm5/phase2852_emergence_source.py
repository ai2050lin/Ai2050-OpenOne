"""Phase 2852 (LPF MA2 cont): cdir emergence source localization.

From 2851: within L30-35 both attn and mlp increment deltas project
NEGATIVE on cdir (attn -0.065, mlp -0.59 signed sums) while the
residual displacement cdir component GROWS 0.37 -> 1.76 in the same
span.  Deep layers are a negative source, so the positive source of
the emergence must live in L14-29 component writes carried by the
residual stream.  This phase localizes it.

Arms (80 words, matched protocol, 5 forwards/word; clamp L13h30 to
base awc, identical to 2847-2851):
  A  per-layer attn/mlp increment cdir profile L0..35 (signed),
     residual displacement profile for the identity audit.
  B  per-head write-delta cdir projection for ALL L14..29 layers
     (2851 recipe: a_c*(V_c@OV) - a_b*(V_b@OV)); source-layer
     selection and concentration stats computed at verdict time
     from the full measurement (no data-dependent measuring).
  C  prereg concentration verdicts S1/S2 + identity audit S3.

Prereg (frozen before any readout; execution.json written first):
  S1  layered concentration iff the top-3 layers of L14-29 (by
      |attn+mlp increment cdir|, word-mean) carry >=60% of the
      window total absolute increment
  S2  head localization iff the top-5 (layer,head) pairs across
      the 3 source layers carry >=50% of the combined absolute
      head cdir increment
  S3  identity audit: |sum increments L14-34 - profile delta
      (35 vs 14)| / |profile delta| < 5%
  verdict: source_mechanism = layered_writers iff S1 and S2,
           else diffuse_field
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
OUT = BASE / 'phase2852' / 'emergence_source'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2852
MAX_WORDS = 8
LAST = 35
NL = 36
NH = 32
WIN_LO, WIN_HI = 14, 30          # L14..29 inclusive

CENSUS_FRONT = {(13, 30), (2, 31), (5, 25), (5, 26), (4, 4), (22, 28),
                (23, 29), (23, 7), (26, 4), (34, 15)}

PREREG = {
    'S1': 'layered concentration iff top-3 layers of L14-29 by '
          '|attn+mlp increment cdir| carry >=60% of window total '
          'absolute increment',
    'S2': 'head localization iff top-5 (layer,head) pairs across '
          'the 3 source layers carry >=50% of combined absolute '
          'head cdir increment',
    'S3': 'identity audit: |sum increments L14-34 - profile delta '
          '(35 vs 14)| / |profile delta| < 5%',
    'verdict': 'source_mechanism = layered_writers iff S1 and S2, '
               'else diffuse_field',
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
                     'design': 'L14-29 per-layer/per-head cdir '
                               'increment attribution, 80 words'}
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
            # hidden_states has NL+1 entries; hs[l] is the state AFTER
            # layer l-1 (hs[0] = embeddings), pos-sliced -> (NL+1, dim)
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

    OV = {}
    for li in range(NL):
        Wo = model.model.layers[li].self_attn.o_proj.weight \
            .detach().float().cpu().numpy().astype(np.float64)
        Wo3 = Wo.reshape(Wo.shape[0], nh, hd)
        for h in range(nh):
            OV[(li, h)] = Wo3[:, h, :]
    vproj = {li: model.model.layers[li].self_attn.v_proj
             for li in range(NL)}

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

    def bias_for(aw2, li, h):
        A11 = float(aw2['same'][li][h][1, 1])
        t = 0.5 * (float(aw2['func'][li][h][1, 1])
                   + float(aw2['null'][li][h][1, 1]))
        t = min(max(t, 0.01), 0.99)
        A11c = min(max(A11, 1e-4), 0.9999)
        b = float(np.log((t / (1.0 - t)) * (1.0 - A11c) / A11c))
        return float(np.clip(b, -20.0, 20.0))

    # accumulators
    delta_cdir = np.zeros((n_words, NL))       # residual displacement
    delta_norm = np.zeros((n_words, NL))
    dhs_cdir = np.zeros((n_words, NL))         # attn increment, signed
    dmlp_cdir = np.zeros((n_words, NL))        # mlp increment, signed
    head_cdir = np.zeros((n_words, WIN_HI - WIN_LO, NH))  # L14..29
    resid_all = []

    for i, (cat, w) in enumerate(target_list):
        ci = CAT_WORDS.index(cat)
        cdir = dW_unit[ci]
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
        raw2, aw2, sain2 = {}, {}, {}
        for cn, toks in c2.items():
            hs, attn, mlp, awc, sac = forward_run(toks, 1)
            raw2[cn] = (hs[LAST] + attn[LAST] + mlp[LAST]) - iso0
            aw2[cn] = awc
            sain2[cn] = sac
            if cn == 'same':
                hs_b2, attn_b2, mlp_b2 = hs, attn, mlp

        b13 = bias_for(aw2, 13, 30)
        set_maps([(13, 30, b13)])
        hs_c, attn_c, mlp_c, awc_c, sain_c = forward_run(c2['same'], 1)
        set_maps([])
        resid_all.append(abs(float(awc_c[13][30][1, 1])
                             - 0.5 * (float(aw2['func'][13][30][1, 1])
                                      + float(aw2['null'][13][30][1, 1]))))

        # arm A: profiles (hs is pos-sliced; NO extra [1] index)
        for l in range(NL):
            dl = hs_c[l] - hs_b2[l]
            delta_norm[i, l] = float(np.linalg.norm(dl))
            delta_cdir[i, l] = float(dl @ cdir)
            dha = attn_c[l] - attn_b2[l]
            dhm = mlp_c[l] - mlp_b2[l]
            dhs_cdir[i, l] = float(dha @ cdir)
            dmlp_cdir[i, l] = float(dhm @ cdir)

        # arm B: per-head write delta cdir projection, L14..29
        for q, l in enumerate(range(WIN_LO, WIN_HI)):
            vin_b = sain2['same'][l][1:2]
            vin_c = sain_c[l][1:2]
            vdt = next(vproj[l].parameters()).dtype
            vb = vproj[l](torch.tensor(vin_b, device='cuda', dtype=vdt)
                          ).detach().float().cpu().numpy() \
                .astype(np.float64)
            vc = vproj[l](torch.tensor(vin_c, device='cuda', dtype=vdt)
                          ).detach().float().cpu().numpy() \
                .astype(np.float64)
            for h in range(NH):
                kv = h // group
                Vb = vb.reshape(1, n_kv, hd)[0, kv]
                Vc = vc.reshape(1, n_kv, hd)[0, kv]
                a_b = float(aw2['same'][l][h][1, 1])
                a_c = float(awc_c[l][h][1, 1])
                dww = a_c * (Vc @ OV[(l, h)].T) \
                    - a_b * (Vb @ OV[(l, h)].T)
                head_cdir[i, q, h] = float(dww @ cdir)

        if (i + 1) % 20 == 0:
            print('P2852 words [%d/%d]' % (i + 1, n_words), flush=True)

    # ---------- verdicts ----------
    inc_attn = dhs_cdir.mean(0)               # signed (36,)
    inc_mlp = dmlp_cdir.mean(0)
    inc_layer = inc_attn + inc_mlp            # layer l writes -> hs[l+1]
    prof = delta_cdir.mean(0)                 # signed (36,)

    # S1: layer concentration inside L14..29
    win = inc_layer[WIN_LO:WIN_HI]
    abs_win = np.abs(win)
    tot_abs = float(abs_win.sum())
    order3 = np.argsort(-abs_win)[:3]
    src_layers = [int(WIN_LO + t) for t in order3]
    s1_share = float(abs_win[order3].sum() / tot_abs) \
        if tot_abs > 1e-9 else 0.0
    s1 = bool(s1_share >= 0.6)

    # S2: head concentration across the 3 source layers
    hc = head_cdir.mean(0)                    # (16, 32)
    q_idx = [l - WIN_LO for l in src_layers]
    hc_sel = hc[q_idx]                        # (3, 32)
    flat = np.abs(hc_sel).reshape(-1)
    tot_head = float(flat.sum())
    o5 = np.argsort(-flat)[:5]
    s2_share = float(flat[o5].sum() / tot_head) \
        if tot_head > 1e-9 else 0.0
    s2 = bool(s2_share >= 0.5)
    top5_pairs = [(int(src_layers[t // NH]), int(t % NH),
                   round(float(hc_sel[t // NH, t % NH]), 4))
                  for t in o5]
    n_front = sum(1 for (l, h, _) in top5_pairs
                  if (l, h) in CENSUS_FRONT)

    # S3: identity audit; layer l writes drive hs[l+1], so
    # prof[35]-prof[14] = sum over l=14..34 of (attn+mlp increments)
    prof_delta = float(prof[LAST] - prof[WIN_LO])
    inc_sum = float(inc_layer[WIN_LO:LAST].sum())
    rel_err = abs(inc_sum - prof_delta) / max(abs(prof_delta), 1e-12)
    s3 = bool(rel_err < 0.05)

    deep_neg = float(inc_layer[30:LAST].sum())     # L30..34 total
    deep_attn = float(inc_attn[30:LAST].sum())
    deep_mlp = float(inc_mlp[30:LAST].sum())

    k_verdict = 'layered_writers' if (s1 and s2) else 'diffuse_field'

    v = {
        'n_words': n_words,
        'S1_layer_concentration': s1,
        'S1_share': round(s1_share, 4),
        'source_layers': src_layers,
        'S2_head_localization': s2,
        'S2_share': round(s2_share, 4),
        'top5_head_pairs': [{'layer': l, 'head': h, 'cdir_proj': p}
                            for (l, h, p) in top5_pairs],
        'n_front_in_top5': n_front,
        'S3_identity_closed': s3,
        'S3_rel_err': round(rel_err, 5),
        'profile_delta_L14_35': round(prof_delta, 4),
        'inc_sum_L14_34': round(inc_sum, 4),
        'deep_neg_total_L30_34': round(deep_neg, 4),
        'deep_attn_L30_34': round(deep_attn, 4),
        'deep_mlp_L30_34': round(deep_mlp, 4),
        'inc_layer_profile': [round(float(x), 4) for x in inc_layer],
        'inc_attn_profile': [round(float(x), 4) for x in inc_attn],
        'inc_mlp_profile': [round(float(x), 4) for x in inc_mlp],
        'delta_cdir_profile': [round(float(x), 3) for x in prof],
        'delta_norm_profile': [round(float(x), 2)
                               for x in delta_norm.mean(0)],
        'max_resid': round(float(np.max(resid_all)), 5)
            if resid_all else None,
        'final_verdict': k_verdict,
    }

    result = {'phase': 2852, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'source.npz',
           delta_cdir=delta_cdir.astype(np.float32),
           delta_norm=delta_norm.astype(np.float32),
           dhs_cdir=dhs_cdir.astype(np.float32),
           dmlp_cdir=dmlp_cdir.astype(np.float32),
           head_cdir=head_cdir.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2852', elapsed)
    print('P2852 VERDICT %s' % json.dumps(v), flush=True)
    print('P2852 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
