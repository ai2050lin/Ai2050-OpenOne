"""Phase 2843 (LPF Delta-II/III): gate causal clamp + domain scan (scaled).

2842 found the write-head interface is a QK self-binding gate
(write_interface=self, 4/5).  This phase upgrades sample scale and
adjudicates the gate's causal sufficiency and domain universality:

Scale: 15 target heads (top-3 per layer from 2841 npz: L22/23/26/28/33)
  + 5 random-head controls; up to 8 single-token words per category
  (10 categories, 2806 framework); 2-token AND 3-token windows
  (3tok: [cond, DE, w], source positions 0/1/2).

Arm A (per-head causal clamp): for each head, clamp A[h, q, q]
  (self-attention mass at query position) in the SAME condition down
  to its own func/null mean t via exact per-head logit bias
  b = ln[(t/(1-t))*(1-A)/A]; re-measure cls_spec matched protocol.
  Prereg:
  G1  gate_causal_sufficient iff mean_h drop_gate >= 0.30
  G2  gate_specific iff mean drop_gate(target15)
      >= mean drop_gate(rand5) + 0.15
  G3  dose_response: Spearman(delta_gate_h, drop_gate_h) reported
  verdict_a: gate_causal iff G1 and G2

Arm B (domain + 3-token):
  D1  gate_universal iff delta_gate > 0 in >= 8/10 directions
      (L22 top-3 mean, delta_gate = A_s - 0.5(A_f + A_n) self mass)
  T1  self_survives_3tok iff mean |share_pos2| > 2*|share_pos0|
      and > 2*|share_pos1| (15-head mean decomposition, L22 h28 plus
      all 15 heads' aggregate)
  verdict_b: gate_universal_and_self_anchored iff D1 and T1
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
OUT = BASE / 'phase2843' / 'gate_causal_clamp'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2841_NPZ = BASE / 'phase2841' / 'longtail_shape' / 'head_drop_map.npz'
SEED = 2843
SCAN_LAYERS = [22, 23, 26, 28, 33]
TOPK = 3
N_RAND = 1          # random control heads per layer
MAX_WORDS = 8
LAST = 35
DE_TID = None       # set after tokenizer

PREREG = {
    'G1': 'gate_causal_sufficient iff mean_h drop_gate >= 0.30',
    'G2': 'gate_specific iff mean drop_gate(target15) >= '
          'mean drop_gate(rand5) + 0.15',
    'G3': 'dose_response Spearman(delta_gate_h, drop_gate_h) reported',
    'verdict_a': 'gate_causal iff G1 and G2',
    'D1': 'gate_universal iff delta_gate > 0 in >= 8/10 directions '
          '(L22 top-3 mean self-mass)',
    'T1': 'self_survives_3tok iff mean |share_pos2| > 2*|share_pos0| '
          'and > 2*|share_pos1| (15-head mean, 3-token window)',
    'verdict_b': 'gate_universal_and_self_anchored iff D1 and T1',
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

    # 15 target heads from 2841 registered npz (top-3 per layer)
    z41 = np.load(SRC_2841_NPZ)
    HEADS = {}
    for li in SCAN_LAYERS:
        d = z41['drops_L%d' % li]
        HEADS[li] = [int(h) for h in np.argsort(-d)[:TOPK]]
    rng0 = np.random.default_rng(SEED)
    RAND = {li: [int(h) for h in rng0.choice(
        [h for h in range(32) if h not in HEADS[li]], N_RAND,
        replace=False)] for li in SCAN_LAYERS}
    ALL_HEADS = {li: HEADS[li] + RAND[li] for li in SCAN_LAYERS}
    tgt_keys = [('L%d_h%d' % (li, h)) for li in SCAN_LAYERS
                for h in HEADS[li]]
    rnd_keys = [('L%d_h%d' % (li, h)) for li in SCAN_LAYERS
                for h in RAND[li]]

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED,
                 'target_heads': HEADS, 'rand_heads': RAND}
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
    group = 32 // n_kv

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

    # ---------- per-layer patched attention (exact replica + bias) ----------
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

    NEED_LAYERS = sorted(set(SCAN_LAYERS))

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
              for li in NEED_LAYERS}
        sain = {li: cap['sain'][li][0] for li in NEED_LAYERS}
        return hs, attn, mlp, aw, sain

    # ---------- targets: all single-token words per category (cap 8) ----------
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
        cs = [w for w in CATS[cat] if w in single_tok][:MAX_WORDS]
        targets[cat] = cs
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
        return {cn: [toks[0], de_tid, toks[1]] for cn, toks in c2.items()}

    # OV slices + vproj for all scanned heads
    OV = {}
    vproj = {}
    for li in SCAN_LAYERS:
        Wo = model.model.layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)
        Wo3 = Wo.reshape(Wo.shape[0], 32, Wo.shape[1] // 32)
        for h in ALL_HEADS[li]:
            OV[(li, h)] = Wo3[:, h, :]
        vproj[li] = model.model.layers[li].self_attn.v_proj

    def head_keys(li):
        return [(li, h) for h in ALL_HEADS[li]]

    def vs_of(sain, li):
        v = vproj[li](torch.tensor(sain[li], device='cuda',
                                   dtype=torch.bfloat16)) \
            .detach().float().cpu().numpy().astype(np.float64)
        return v.reshape(v.shape[0], n_kv, -1)          # (S, 8, 128)

    # ---------- accumulators ----------
    cls_base_w = []            # per-word cls (2tok base)
    drop_gate = {k: [] for k in tgt_keys + rnd_keys}
    clamp_resid = []
    dgate_words = {li: {h: [] for h in HEADS[li]} for li in SCAN_LAYERS}
    sh3 = {li: {h: [[], [], []] for h in HEADS[li]} for li in SCAN_LAYERS}

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

        # per-head gate delta (same-cond self mass minus ctrl)
        for li in SCAN_LAYERS:
            for h in HEADS[li]:
                a = aw2
                g = (float(a['same'][li][h][1, 1])
                     - 0.5 * (float(a['func'][li][h][1, 1])
                              + float(a['null'][li][h][1, 1])))
                dgate_words[li][h].append(g)

        # per-head clamp: bias self mass to own ctrl level, same cond only
        for li in SCAN_LAYERS:
            for h in ALL_HEADS[li]:
                A11 = float(aw2['same'][li][h][1, 1])
                t = 0.5 * (float(aw2['func'][li][h][1, 1])
                           + float(aw2['null'][li][h][1, 1]))
                t = min(max(t, 0.01), 0.99)
                A11c = min(max(A11, 1e-4), 0.9999)
                b = float(np.log((t / (1.0 - t)) * (1.0 - A11c) / A11c))
                b = float(np.clip(b, -20.0, 20.0))
                patches[li]['map'] = {h: [(1, 1, b)]}
                hs, attn, mlp, awc, _ = forward_run(c2['same'], 1)
                patches[li]['map'] = {}
                raw_c = (hs[LAST] + attn[LAST] + mlp[LAST]) - iso0
                dsc = raw_c - 0.5 * (raw2['func'] + raw2['null'])
                cls_c = float(abs(dsc @ cdir)) / nfull
                drop_gate['L%d_h%d' % (li, h)].append(
                    (cls_base - cls_c) / max(cls_base, 1e-30))
                clamp_resid.append(abs(float(awc[li][h][1, 1]) - t))

        # 3-token window: decomposition for target heads
        c3 = conds3_for(i, cat, w)
        hs_iso3, _, _, _, _ = forward_run([w_tid], 0)
        iso3 = hs_iso3[0]
        raw3, v3, aw3 = {}, {}, {}
        for cn, toks in c3.items():
            hs, attn, mlp, awc, sain = forward_run(toks, 2)
            raw3[cn] = (hs[LAST] + attn[LAST] + mlp[LAST]) - iso3
            v3[cn] = {li: vs_of(sain, li) for li in SCAN_LAYERS}
            aw3[cn] = {li: awc[li] for li in SCAN_LAYERS}
        dfull3 = raw3['same'] - 0.5 * (raw3['func'] + raw3['null'])
        nf3 = max(float(np.linalg.norm(dfull3)), 1e-30)
        for li in SCAN_LAYERS:
            for h in HEADS[li]:
                kv = h // group
                Cs = {}
                for cn in c3:
                    Vh = v3[cn][li][:, kv, :]               # (3,128)
                    Wm = Vh @ OV[(li, h)].T                 # (3,4096)
                    Cs[cn] = aw3[cn][li][h][2, :][:, None] * Wm
                Csp = Cs['same'] - 0.5 * (Cs['func'] + Cs['null'])
                for j in range(3):
                    sh3[li][h][j].append(float(Csp[j] @ cdir) / nf3)
        if (i + 1) % 10 == 0:
            print('P2843 words [%d/%d] cls_base=%.4f' % (
                i + 1, n_words, float(np.mean(cls_base_w))), flush=True)

    # ---------- Arm A verdicts ----------
    dg_t = {k: float(np.mean(drop_gate[k])) for k in tgt_keys}
    dg_r = {k: float(np.mean(drop_gate[k])) for k in rnd_keys}
    mean_t = float(np.mean(list(dg_t.values())))
    mean_r = float(np.mean(list(dg_r.values())))
    g1 = mean_t >= 0.30
    g2 = mean_t >= mean_r + 0.15
    # dose-response across 15 target heads
    dgate_h = []
    for li in SCAN_LAYERS:
        for h in HEADS[li]:
            dgate_h.append(float(np.mean(dgate_words[li][h])))
    rho = spearman(np.array(dgate_h), np.array(list(dg_t.values())))

    # ---------- Arm B verdicts ----------
    dgate_dir = {}
    for ci, cat in enumerate(CAT_WORDS):
        ws = [idx for idx, (cat2, _) in enumerate(target_list)
              if cat2 == cat]
        vals = []
        for li in SCAN_LAYERS[:1]:      # L22 top-3 mean
            for h in HEADS[li]:
                vals.append(float(np.mean([dgate_words[li][h][j]
                                           for j in ws])))
        dgate_dir[cat] = float(np.mean(vals))
    n_pos_dir = sum(1 for v in dgate_dir.values() if v > 0)
    d1 = n_pos_dir >= 8

    # 3-token aggregate: mean over 15 heads of mean shares
    s0 = float(np.mean([np.mean(sh3[li][h][0]) for li in SCAN_LAYERS
                        for h in HEADS[li]]))
    s1 = float(np.mean([np.mean(sh3[li][h][1]) for li in SCAN_LAYERS
                        for h in HEADS[li]]))
    s2 = float(np.mean([np.mean(sh3[li][h][2]) for li in SCAN_LAYERS
                        for h in HEADS[li]]))
    t1 = abs(s2) > 2 * abs(s0) and abs(s2) > 2 * abs(s1)

    v = {
        'n_words': n_words,
        'cls_base_mean': round(float(np.mean(cls_base_w)), 6),
        'clamp_max_resid': round(float(np.max(clamp_resid)), 5),
        'mean_drop_gate_target15': round(mean_t, 4),
        'mean_drop_gate_rand5': round(mean_r, 4),
        'G1_gate_causal_sufficient': bool(g1),
        'G2_gate_specific': bool(g2),
        'G3_spearman_dose': round(rho, 4),
        'drop_gate_per_head': {k: round(x, 4) for k, x in
                               sorted(dg_t.items(), key=lambda kv: -kv[1])},
        'drop_gate_rand': {k: round(x, 4) for k, x in dg_r.items()},
        'verdict_a': 'gate_causal' if (g1 and g2) else
                     ('gate_partial' if g1 or g2 else 'not_sufficient'),
        'delta_gate_per_dir': {k: round(x, 5) for k, x in
                               sorted(dgate_dir.items(),
                                      key=lambda kv: -kv[1])},
        'n_dir_positive': n_pos_dir,
        'D1_gate_universal': bool(d1),
        'share3_pos0': round(s0, 5), 'share3_pos1': round(s1, 5),
        'share3_pos2': round(s2, 5),
        'T1_self_survives_3tok': bool(t1),
        'verdict_b': 'gate_universal_and_self_anchored'
                     if (d1 and t1) else 'partial',
    }

    result = {'phase': 2843, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)

    fc.npz(OUT / 'gate_maps.npz',
           dgate_dir=np.array([dgate_dir[c] for c in CAT_WORDS],
                              dtype=np.float64),
           **{('dropgate_%s' % k): np.float64(x)
              for k, x in list(dg_t.items()) + list(dg_r.items())})

    elapsed = time.monotonic() - t0
    cc.ledger('phase2843', elapsed)
    print('P2843 VERDICT %s' % json.dumps(v), flush=True)
    print('P2843 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
