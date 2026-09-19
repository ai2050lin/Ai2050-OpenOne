"""Phase 2848 (LPF MA2 cont / ATLAS_PLAN frontline 2): lateral
inhibition quantification + former->amplifier interface + collaborator
pairs.

Motivated by 2847: clamping L13 h30 (a FORMER: no direct cdir write,
9.3x self-binding gate) raised alignment on 4/10 competing directions
(negative drops) and its damage materializes only at L34.  Three open
questions:
  H-lateral  is the rise of competing directions real lateral
             inhibition (specific to formers) or a generic clamp
             artifact (random heads also raise competitors)?
  H-interf   do formers feed amplifiers?  Does clamping L13 h30
             weaken the amplifiers' self-binding gates / shift their
             input states (L13 -> L22-23 interface)?
  H-pairs    which early-cluster pairs are the sub-additive
             collaborators (2847 F3 ratio 0.65)?

Arms (80 words, matched func/null protocol):
  A  10-direction alignment delta (absolute, small-denominator-safe)
     under L13 h30 clamp vs 3 random control heads' clamps.
  B  per-amplifier self-gate A11 and input-state displacement under
     L13 h30 clamp.
  C  all 10 early-cluster pairs: joint clamp vs sum of singles.

Prereg (frozen before any readout):
  H1  lateral_inhibition_confirmed iff >= 3 non-own directions show
      mean alignment delta > +0.02 with word-level frac>0.6 under
      L13 h30 clamp AND the same statistic for random control heads
      is < 1/3 of the L13 h30 values
  H2  interface_confirmed iff >= 3 of 4 amplifiers show relative
      A11 (self-mass) reduction > 10% under L13 h30 clamp OR input
      displacement at their layer > 2x the L13 clamp residual scale
  H3  collaborator_pairs iff >= 2 of 10 pairs have joint/sum < 0.7
  verdict: mechanism_trio_confirmed iff H1 AND H2
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
OUT = BASE / 'phase2848' / 'lateral_inhibition'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2848
MAX_WORDS = 8
LAST = 35
NL = 36
NH = 32

EARLY = [(13, 30), (2, 31), (5, 25), (5, 26), (4, 4)]
LATE = [(22, 28), (23, 29), (26, 4), (34, 15)]

PREREG = {
    'H1': 'lateral_inhibition_confirmed iff >=3 non-own directions '
          'mean align delta > +0.02 & frac_words>0.6 under L13h30 '
          'clamp AND random-control same stat < 1/3 of it',
    'H2': 'interface_confirmed iff >=3/4 amplifiers A11 rel reduction '
          '>10% under L13h30 clamp OR input disp at their layer > 2x '
          'L13 residual scale (0.02)',
    'H3': 'collaborator_pairs iff >=2 of 10 early pairs joint/sum<0.7',
    'verdict': 'mechanism_trio_confirmed iff H1 AND H2',
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
                     'design': 'lateral inhibition + interface + '
                               'collaborator pairs, 80 words'}
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

    banned = set(EARLY) | set(LATE)
    while True:
        ctrl = [(int(rng.integers(0, NL)), int(rng.integers(0, NH)))
                for _ in range(3)]
        if len(set(ctrl)) == 3 and not (banned & set(ctrl)):
            break

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
    delta1330 = np.zeros((n_words, 10))    # arm A target head
    delta_ctrl = np.zeros((n_words, 3, 10))
    gate_drop = np.zeros((n_words, len(LATE)))   # arm B A11 rel change
    disp_late = np.zeros((n_words, len(LATE), NL))
    pair_joint = np.zeros((n_words, 10))
    pair_sum = np.zeros((n_words, 10))
    resid_all = []

    PAIRS = [(i, j) for i in range(len(EARLY))
             for j in range(i + 1, len(EARLY))]

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
        raw2, aw2, sain2 = {}, {}, {}
        for cn, toks in c2.items():
            hs, attn, mlp, awc, sac = forward_run(toks, 1)
            raw2[cn] = (hs[LAST] + attn[LAST] + mlp[LAST]) - iso0
            aw2[cn] = awc
            sain2[cn] = sac
        d_spec_full = raw2['same'] \
            - 0.5 * (raw2['func'] + raw2['null'])
        nfull = max(float(np.linalg.norm(d_spec_full)), 1e-30)
        base_d = np.array([abs(float(d_spec_full @ dW_unit[d]))
                           for d in range(10)])

        sain_same_ref = {l: sain2['same'][l][1].copy()
                         for l in range(NL)}

        def clamp_run(spec):
            set_maps(spec)
            hs, attn, mlp, awc, sac = forward_run(c2['same'], 1)
            set_maps([])
            return hs, attn, mlp, awc, sac

        def align_delta(spec, drop1330=False):
            hs_c, attn_c, mlp_c, awc_c, sac = clamp_run(spec)
            for (li, h, b) in spec:
                resid_all.append(
                    abs(float(awc_c[li][h][1, 1])
                        - 0.5 * (float(aw2['func'][li][h][1, 1])
                                 + float(aw2['null'][li][h][1, 1]))))
            dsc = (hs_c[LAST] + attn_c[LAST] + mlp_c[LAST]) - iso0 \
                - 0.5 * (raw2['func'] + raw2['null'])
            dd = np.array([abs(float(dsc @ dW_unit[d]))
                           - base_d[d] for d in range(10)]) / nfull
            return dd, awc_c, sac

        # arm A: L13 h30
        b13 = bias_for(aw2, 13, 30)
        dd13, awc13, sac13 = align_delta([(13, 30, b13)])
        delta1330[i] = dd13

        # arm B: amplifier gates + input displacement under L13h30 clamp
        for j, (li, h) in enumerate(LATE):
            a_base = float(aw2['same'][li][h][1, 1])
            a_cl = float(awc13[li][h][1, 1])
            gate_drop[i, j] = (a_base - a_cl) / max(a_base, 1e-30)
            dv = sac[li][1] - sain_same_ref[li]
            disp_late[i, j] = np.array(
                [float(np.linalg.norm(sac[l][1] - sain_same_ref[l]))
                 for l in range(NL)])

        # arm A controls: random heads
        for k, (li, h) in enumerate(ctrl):
            b = bias_for(aw2, li, h)
            dd, _, _ = align_delta([(li, h, b)])
            delta_ctrl[i, k] = dd

        # singles for pair sums (reuse 2847 formula: own-direction drop)
        base_own = base_d[ci] / nfull
        singles = {}
        for (li, h) in EARLY:
            b = bias_for(aw2, li, h)
            hs_c, attn_c, mlp_c, _, _ = clamp_run([(li, h, b)])
            dsc = (hs_c[LAST] + attn_c[LAST] + mlp_c[LAST]) - iso0 \
                - 0.5 * (raw2['func'] + raw2['null'])
            cls_c = abs(float(dsc @ dW_unit[ci])) / nfull
            singles[(li, h)] = (base_own - cls_c) / max(base_own, 1e-30)

        # arm C: pairs
        for p, (a, b) in enumerate(PAIRS):
            ha, hb = EARLY[a], EARLY[b]
            spec = [(ha[0], ha[1], bias_for(aw2, ha[0], ha[1])),
                    (hb[0], hb[1], bias_for(aw2, hb[0], hb[1]))]
            hs_c, attn_c, mlp_c, _, _ = clamp_run(spec)
            dsc = (hs_c[LAST] + attn_c[LAST] + mlp_c[LAST]) - iso0 \
                - 0.5 * (raw2['func'] + raw2['null'])
            cls_c = abs(float(dsc @ dW_unit[ci])) / nfull
            joint = (base_own - cls_c) / max(base_own, 1e-30)
            pair_joint[i, p] = joint
            pair_sum[i, p] = singles[ha] + singles[hb]

        if (i + 1) % 10 == 0:
            print('P2848 words [%d/%d]' % (i + 1, n_words), flush=True)

    # ---------- verdicts ----------
    own_mask = np.zeros(10, dtype=bool)
    # word-to-own-direction mapping varies per word; recompute masks
    own_idx = np.array([CAT_WORDS.index(cat)
                        for cat, w in target_list])
    dd13 = delta1330                       # (80, 10)
    dd_ctrl = delta_ctrl.mean(1)           # (80, 10)

    frac_pos = np.zeros(10)
    mean_dd = np.zeros(10)
    mean_dd_ctrl = np.zeros(10)
    for d in range(10):
        rows = own_idx != d                # exclude each word's own dir
        # statistic per direction over words whose own dir != d
        vals = dd13[rows, d]
        frac_pos[d] = float(np.mean(vals > 0.02))
        mean_dd[d] = float(np.mean(vals))
        mean_dd_ctrl[d] = float(np.mean(dd_ctrl[rows, d]))

    n_lat = int(np.sum((mean_dd > 0.02) & (frac_pos > 0.6)))
    h1 = bool(n_lat >= 3
              and np.mean(mean_dd[mean_dd > 0.02])
              > 3.0 * max(np.mean(mean_dd_ctrl), 1e-30))

    gate_rel = gate_drop.mean(0)           # (4,)
    disp_at_layer = np.array([
        float(np.mean(disp_late[:, j, LATE[j][0]]))
        for j in range(len(LATE))])
    h2 = bool(np.sum(gate_rel > 0.10) >= 3
              or np.min(disp_at_layer) > 0.04)

    ratios = pair_joint.mean(0) / np.maximum(pair_sum.mean(0), 1e-30)
    n_pairs = int(np.sum(ratios < 0.7))
    h3 = bool(n_pairs >= 2)

    v = {
        'n_words': n_words,
        'control_heads': [[int(l), int(h)] for l, h in ctrl],
        'H1_lateral_inhibition': h1,
        'n_directions_risen': n_lat,
        'mean_dd_per_direction': [round(float(x), 5)
                                  for x in mean_dd],
        'frac_pos_per_direction': [round(float(x), 3)
                                   for x in frac_pos],
        'mean_dd_ctrl_per_direction': [round(float(x), 5)
                                       for x in mean_dd_ctrl],
        'H2_interface': h2,
        'amplifier_gate_rel_drop': [round(float(x), 4)
                                    for x in gate_rel],
        'amplifier_input_disp_at_layer': [round(float(x), 5)
                                          for x in disp_at_layer],
        'H3_collaborator_pairs': h3,
        'n_strong_pairs': n_pairs,
        'pair_ratios': [round(float(x), 4) for x in ratios],
        'max_resid': round(float(np.max(resid_all)), 5)
            if resid_all else None,
        'final_verdict': 'mechanism_trio_confirmed' if (h1 and h2)
            else ('partial' if (h1 or h2 or h3) else 'not_confirmed'),
    }

    result = {'phase': 2848, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'lateral.npz',
           delta1330=delta1330.astype(np.float32),
           delta_ctrl=delta_ctrl.astype(np.float32),
           gate_drop=gate_drop.astype(np.float32),
           disp_late=disp_late.astype(np.float32),
           pair_joint=pair_joint.astype(np.float32),
           pair_sum=pair_sum.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2848', elapsed)
    print('P2848 VERDICT %s' % json.dumps(v), flush=True)
    print('P2848 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
