"""Phase 2855 (LPF MA2 cont): L29-31 gain-window anatomy.

From 2854 (post-erratum): the cdir component accumulation is a
passive transmission band; the ">1 gain window" at L29-31
(g_emp 1.68/1.28/0.90) shapes the emergence curve.  Open question:
where does the window gain live -- LN2 (post_attention_layernorm),
the SwiGLU itself, or which FF neurons?

Arms (80 words, matched protocol, EPS=1.0, window L26..35):
  A  three-spectrum decomposition per layer:
     g_comp = [mlp(LN2(x+e*cdir)) - mlp(LN2(x))].cdir / e
     g_ln   = [LN2(x+e*cdir) - LN2(x)].cdir / e
     g_mlp  = [mlp(ln_x + dln) - mlp(ln_x)].cdir / e   (chained)
     with x = sain+attn at pos1 (base run), ln_x = LN2(x),
     dln = LN2(x+e*cdir) - LN2(x).  Baseline out_b = hook mlp output
     (= mlp(ln_x)) -- same function, 2854 LN2 lesson applied.
  B  SwiGLU neuron-level decomposition of the linearized response:
     m = silu'(g) .* (Wg.dln) .* u + silu(g) .* (Wu.dln)  (ff-dim)
     c_n = (Wd^T.cdir)_n * m_n ;  top-32 / top-8 share of total |c_n|
  C  linearity check: Pearson r(linearized response, numerical
     response) per layer.

Prereg (frozen before any readout):
  W1  window source: swiglu_intrinsic iff >=2 of L29-31 have
      |g_mlp| >= 1.2 with sign(g_mlp)=sign(g_comp); ln2_mediated
      iff >=2 of L29-31 have |g_ln - 1| >= 0.2; else flat
  W2  neuron_localized iff word-mean top-32 share >= 0.5 pooled
      over L29-31 (9728 FF neurons); else neuron_diffuse
  L1  linear_ok iff mean Pearson r over L29-31 >= 0.9
  verdict: window_mechanism = W1 verdict + '/' + W2 verdict
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
OUT = BASE / 'phase2855' / 'window_anatomy'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2855
MAX_WORDS = 8
LAST = 35
NL = 36
EPS = 1.0
WIN_LO, WIN_HI = 26, 36          # anatomy window L26..35
W1_LO, W1_HI = 29, 32            # focus L29..31
TOPK = 32

PREREG = {
    'W1': 'window source: swiglu_intrinsic iff >=2 of L29-31 have '
          '|g_mlp|>=1.2 and sign(g_mlp)=sign(g_comp); ln2_mediated '
          'iff >=2 of L29-31 have |g_ln-1|>=0.2; else flat',
    'W2': 'neuron_localized iff word-mean top-32 share >= 0.5 '
          'pooled over L29-31; else neuron_diffuse',
    'L1': 'linear_ok iff mean Pearson r(linearized, numerical) '
          'over L29-31 >= 0.9',
    'verdict': 'window_mechanism = W1 + "/" + W2 (+ L1 flag)',
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
                     'design': 'L26-35 LN2/SwiGLU gain decomposition + '
                               'neuron top-k, EPS=1.0, 80 words',
                     'eps': EPS}
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

    layers = model.model.layers
    vdt = next(layers[0].mlp.parameters()).dtype

    def ln2_call(li, x):
        t = torch.tensor(x[None, :], device='cuda', dtype=vdt)
        with torch.no_grad():
            o = layers[li].post_attention_layernorm(t)
        return o[0].detach().float().cpu().numpy().astype(np.float64)

    def mlp_raw(li, X):
        # raw SwiGLU, NO LN2 (caller applies LN2 explicitly)
        t = torch.tensor(X[None, :], device='cuda', dtype=vdt)
        with torch.no_grad():
            o = layers[li].mlp(t)
        return o[0].detach().float().cpu().numpy().astype(np.float64)

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

    n_win = WIN_HI - WIN_LO
    g_comp = np.zeros((n_words, n_win))
    g_ln = np.zeros((n_words, n_win))
    g_mlp = np.zeros((n_words, n_win))
    top32_share = np.zeros((n_words, n_win))
    top8_share = np.zeros((n_words, n_win))
    lin_r = np.zeros((n_words, n_win))
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

        cdir_t = torch.tensor(cdir, device='cuda', dtype=vdt)
        for q, l in enumerate(range(WIN_LO, WIN_HI)):
            x_in = sain2['same'][l][1] + attn_b2[l]
            ln_x = ln2_call(l, x_in)                 # LN2(x)
            dln = ln2_call(l, x_in + EPS * cdir) - ln_x
            out_b = mlp_b2[l]                        # = mlp(ln_x), hook

            out_comp = mlp_raw(l, ln2_call(l, x_in + EPS * cdir))
            g_comp[i, q] = float((out_comp - out_b) @ cdir) / EPS
            g_ln[i, q] = float(dln @ cdir) / EPS
            out_m = mlp_raw(l, ln_x + dln)
            g_mlp[i, q] = float((out_m - out_b) @ cdir) / EPS

            # neuron-level linearized decomposition (torch, on-graph)
            m = layers[l].mlp
            h_t = torch.tensor(ln_x[None, :], device='cuda', dtype=vdt)
            d_t = torch.tensor(dln[None, :], device='cuda', dtype=vdt)
            with torch.no_grad():
                pg = (m.gate_proj.weight @ h_t[0])   # (ff,)
                pu = (m.up_proj.weight @ h_t[0])
                dg = (m.gate_proj.weight @ d_t[0])
                du = (m.up_proj.weight @ d_t[0])
                sg = torch.sigmoid(pg)
                silu_p = sg * (1.0 + pg * (1.0 - sg))
                mvec = silu_p * dg * pu + sg * du    # (ff,)
                rlin = (m.down_proj.weight @ mvec)   # (d,)
                cvec = (m.down_proj.weight.T @ cdir_t) * mvec
            rlin_np = rlin.detach().float().cpu().numpy().astype(
                np.float64)
            cvec_np = cvec.detach().float().cpu().numpy().astype(
                np.float64)
            num = out_m - out_b
            sx = np.std(rlin_np)
            sy = np.std(num)
            lin_r[i, q] = float(np.corrcoef(rlin_np, num)[0, 1]) \
                if sx > 1e-9 and sy > 1e-9 else 0.0
            a = np.abs(cvec_np)
            tot = float(a.sum())
            if tot > 1e-12:
                srt = np.sort(a)[::-1]
                top32_share[i, q] = float(srt[:TOPK].sum() / tot)
                top8_share[i, q] = float(srt[:8].sum() / tot)
            else:
                top32_share[i, q] = 0.0
                top8_share[i, q] = 0.0

        if (i + 1) % 20 == 0:
            print('P2855 words [%d/%d]' % (i + 1, n_words), flush=True)

    # ---------- verdicts ----------
    def wm(a):
        return a.mean(0)

    q29, q31 = W1_LO - WIN_LO, W1_HI - WIN_LO     # 3..5 inclusive
    gcm = wm(g_comp)
    gmm = wm(g_mlp)
    glm = wm(g_ln)

    n_swiglu = sum(1 for q in range(q29, q31 + 1)
                   if abs(gmm[q]) >= 1.2
                   and np.sign(gmm[q]) == np.sign(gcm[q]))
    n_ln = sum(1 for q in range(q29, q31 + 1)
               if abs(glm[q] - 1.0) >= 0.2)
    if n_swiglu >= 2:
        w1 = 'swiglu_intrinsic'
    elif n_ln >= 2:
        w1 = 'ln2_mediated'
    else:
        w1 = 'flat'

    w2_share = float(top32_share[:, q29:q31 + 1].mean())
    w2 = 'neuron_localized' if w2_share >= 0.5 else 'neuron_diffuse'
    l1 = bool(lin_r[:, q29:q31 + 1].mean() >= 0.9)

    v = {
        'n_words': n_words,
        'W1_source': w1,
        'n_swiglu_layers': n_swiglu,
        'n_ln2_layers': n_ln,
        'W2_neuron': w2,
        'W2_top32_share_mean': round(w2_share, 4),
        'W2_top8_share_mean': round(
            float(top8_share[:, q29:q31 + 1].mean()), 4),
        'L1_linear_ok': l1,
        'L1_mean_r': round(float(lin_r[:, q29:q31 + 1].mean()), 4),
        'g_comp_L26_35': [round(float(x), 4) for x in gcm],
        'g_ln_L26_35': [round(float(x), 4) for x in glm],
        'g_mlp_L26_35': [round(float(x), 4) for x in gmm],
        'top32_share_L26_35': [round(float(x), 4)
                               for x in top32_share.mean(0)],
        'max_resid': round(float(np.max(resid_all)), 5)
            if resid_all else None,
        'final_verdict': '%s/%s' % (w1, w2),
    }

    result = {'phase': 2855, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'window.npz',
           g_comp=g_comp.astype(np.float32),
           g_ln=g_ln.astype(np.float32),
           g_mlp=g_mlp.astype(np.float32),
           top32_share=top32_share.astype(np.float32),
           top8_share=top8_share.astype(np.float32),
           lin_r=lin_r.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2855', elapsed)
    print('P2855 VERDICT %s' % json.dumps(v), flush=True)
    print('P2855 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
