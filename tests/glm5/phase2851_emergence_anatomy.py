"""Phase 2851 (LPF MA2 cont): cdir emergence anatomy at L30-35.

From 2850: clamping the FORMER L13 h30 causes a global geometry
perturbation; the cdir component of the displacement stays ~0.04
through L14-29 then jumps to 0.29..0.63 across L30-35.  No single
head aligns with the FORMER's write direction, so the emergence is
not a discrete hand-off.  Hypothesis under test:

  TRANSMISSION AMPLIFICATION: amplifiers (2846 census: large direct
  cdir write, e.g. L34 h15 s1=0.47) read their own (displaced) state
  and re-write it through cdir-aligned OV channels -- the displaced
  geometry is TRANSMITTED into cdir space by the readout lattice,
  not by any single consumer.

Arms (80 words, matched protocol, 5 forwards/word):
  A  displacement rotation profile: delta(l) = residual displacement
     at pos1 (raw hs, 2845 lesson); report a_l = delta.cdir/||delta||
     and ||delta(l)|| across all 36 layers (residual caliber;
     2850's sain-caliber profile compared as caliber sensitivity).
  B  per-layer increment attribution: attn/mlp residual increments
     projected on cdir, L14..35; within L30-35, per-head write
     delta projected on cdir (absolute) -> top contributor heads.
  C  transmission gain: for top amplifier heads, gain_h =
     (dWrite_h . cdir) / (delta(l-1) . cdir).

Prereg (frozen before any readout):
  K1  transmission_amplification iff >= 60% of the total cdirchg
      increment (L14->35, mean over words) is contributed by L30-35
      attention write deltas AND >= 3 of the top-5 contributor heads
      are 2846 census front-edge/amplifier heads
  K2  gain_reported (descriptive): gain_h for the top-3 gain heads
  verdict: emergence_mechanism = transmission_lattice iff K1,
           else diffuse_unresolved
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
OUT = BASE / 'phase2851' / 'emergence_anatomy'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2851
MAX_WORDS = 8
LAST = 35
NL = 36
NH = 32

CENSUS_FRONT = {(13, 30), (2, 31), (5, 25), (5, 26), (4, 4), (22, 28),
                (23, 29), (23, 7), (26, 4), (34, 15)}

PREREG = {
    'K1': 'transmission_amplification iff >=60% of total cdirchg '
          'increment L14->35 from L30-35 attn write deltas AND >=3/5 '
          'top contributor heads are census front-edge heads',
    'K2': 'gain_h descriptive for top-3 gain heads',
    'verdict': 'emergence_mechanism = transmission_lattice iff K1',
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
                     'design': 'delta rotation profile + increment '
                               'attribution + transmission gain, 80 words'}
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
    delta_norm = np.zeros((n_words, NL))       # arm A (residual hs)
    delta_cdir = np.zeros((n_words, NL))       # signed projection
    dhs_norm = np.zeros((n_words, NL))
    dhs_cdir = np.zeros((n_words, NL))         # signed
    dmlp_cdir = np.zeros((n_words, NL))
    dhs_sain_norm = np.zeros((n_words, NL))    # sain caliber comparison
    dhs_sain_cdir = np.zeros((n_words, NL))
    head_cdir = np.zeros((n_words, 6, NH))     # arm B: L30..35 x 32
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

        # arm A: residual displacement profile; base residual/attn/mlp
        # states are the 'same' run saved from the raw2 loop above
        # (clamped run hs_c vs base run hs_b2, both pos1).
        for l in range(NL):
            dl = hs_c[l] - hs_b2[l]   # forward_run returns pos-sliced hs
            delta_norm[i, l] = float(np.linalg.norm(dl))
            delta_cdir[i, l] = float(dl @ cdir)
            # sain caliber comparison
            dsn = sain_c[l][1] - sain2['same'][l][1]
            dhs_sain_norm[i, l] = float(np.linalg.norm(dsn))
            dhs_sain_cdir[i, l] = abs(float(dsn @ cdir))
            dha = attn_c[l] - attn_b2[l]
            dhm = mlp_c[l] - mlp_b2[l]
            dhs_norm[i, l] = float(np.linalg.norm(dha))
            dhs_cdir[i, l] = float(dha @ cdir)
            dmlp_cdir[i, l] = float(dhm @ cdir)

        # arm B: per-head write delta cdir projection, L30..35
        for q, l in enumerate(range(30, 36)):
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

        if (i + 1) % 10 == 0:
            print('P2851 words [%d/%d]' % (i + 1, n_words), flush=True)

    # ---------- verdicts ----------
    # total cdirchg increment L14->35 from attn write deltas
    inc_attn = dhs_cdir.mean(0)                # signed (36,)
    cdirchg_mean = np.abs(delta_cdir).mean(0)  # (36,) residual caliber
    total_inc = float(inc_attn[14:].sum())
    late_inc = float(inc_attn[30:].sum())
    # direct division; denominator is |total| ~0.32, no degeneracy.
    # (max(x,1e-30) guard was wrong: for negative x it returns 1e-30
    #  and fabricates 1e28 ratios -- fixed before result registration)
    frac_late = (late_inc / total_inc) if abs(total_inc) > 1e-6 \
        else float('nan')
    frac_late_abs = float(np.abs(inc_attn[30:]).sum()
                          / max(np.abs(inc_attn[14:]).sum(), 1e-30))

    # top contributor heads (L30-35)
    hc = head_cdir.mean(0)                     # (6, 32)
    flat = np.abs(hc).reshape(-1)
    order = np.argsort(-flat)[:5]
    top5 = [(int(30 + t // NH), int(t % NH), round(float(hc[t // NH,
              t % NH]), 4)) for t in order]
    n_front = sum(1 for (l, h, _) in top5
                  if (l, h) in CENSUS_FRONT)

    # arm A rotation profile
    a_l = delta_cdir.mean(0) / np.maximum(delta_norm.mean(0), 1e-30)
    dn_mean = delta_norm.mean(0)
    dc_mean = np.abs(delta_cdir).mean(0)

    # transmission gain: top-3 heads by |head_cdir|
    gains = {}
    for (l, h, val) in top5[:3]:
        if val == 0:
            continue
        den = float(np.mean(delta_cdir[:, l - 1]))
        gain = float(hc[l - 30, h]) / den if abs(den) > 1e-6 \
            else float('nan')
        gains['L%dh%d' % (l, h)] = round(gain, 3)

    k1 = bool(frac_late >= 0.6 and n_front >= 3)

    v = {
        'n_words': n_words,
        'K1_transmission_amplification': k1,
        'frac_late_attn_cdir': round(frac_late, 4)
            if np.isfinite(frac_late) else None,
        'frac_late_attn_cdir_abs': round(frac_late_abs, 4),
        'attn_inc_cdir_L14_29': round(float(inc_attn[14:30].sum()), 4),
        'attn_inc_cdir_L30_35': round(late_inc, 4),
        'top5_heads': [{'layer': l, 'head': h, 'cdir_proj': v}
                       for (l, h, v) in top5],
        'n_front_in_top5': n_front,
        'K2_gains': gains,
        'delta_norm_profile': [round(float(x), 2) for x in dn_mean],
        'delta_cdir_profile': [round(float(x), 3) for x in dc_mean],
        'align_profile': [round(float(x), 4) for x in a_l],
        'sain_cdir_profile': [round(float(x), 3)
                              for x in dhs_sain_cdir.mean(0)],
        'mlp_inc_cdir_L30_35': round(float(dmlp_cdir[30:].mean(0).sum()),
                                     4),
        'max_resid': round(float(np.max(resid_all)), 5)
            if resid_all else None,
        'final_verdict': 'transmission_lattice' if k1
            else 'diffuse_unresolved',
    }

    result = {'phase': 2851, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'emergence.npz',
           delta_norm=delta_norm.astype(np.float32),
           delta_cdir=delta_cdir.astype(np.float32),
           dhs_cdir=dhs_cdir.astype(np.float32),
           dmlp_cdir=dmlp_cdir.astype(np.float32),
           head_cdir=head_cdir.astype(np.float32),
           dhs_sain_cdir=dhs_sain_cdir.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2851', elapsed)
    print('P2851 VERDICT %s' % json.dumps(v), flush=True)
    print('P2851 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
