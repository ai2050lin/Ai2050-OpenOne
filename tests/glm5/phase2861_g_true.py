"""Phase 2861 (LPF MA2 cont): true-residual workpoint re-measurement.

2860 root-caused hard-wire #4 (workpoint mixing): x_in = sain+attn was
LN1(x)+attn_out (pseudo-residual), NOT the true LN2 input x+attn_out;
2853 g_jac / 2855 g_comp mixed hook truth (true residual) with rebuild
(pseudo residual), inflating the "deep amplifier ~7" artifact.

This phase re-measures the window anatomy at the TRUE workpoint:
  ln2in_true = pre-hook on post_attention_layernorm (true LN2 input)
  mlpin_true = pre-hook on mlp                      (true mlp input)
  out_b_true = mlp_raw(mlpin_true)  (explicit rebuild, same workpoint)
  g_ln_true     = [LN2(ln2in+e*cdir) - LN2(ln2in)].cdir / e
  g_comp_true   = [mlp(LN2(ln2in+e*cdir)) - out_b_true].cdir / e
  g_direct_true = [mlp(mlpin+e*cdir)      - out_b_true].cdir / e
Built-in consistency checks (bf16 determinism, expected ~0):
  v1 = max|LN2(ln2in_true) - mlpin_true|   (LN2 rebuild == hook input)
  v2 = max|mlp_raw(mlpin_true) - mlpout_true| (mlp rebuild == hook out)

Protocol verbatim from 2860/2855 otherwise (SEED=2855 vocab, 80 words,
L26-35, EPS=1.0, L13H30 clamp arm).  Prereg frozen before readout:
  E1  2860 criteria form at the true workpoint: swiglu_intrinsic iff
      >=6/10 layers |ratio-1|<=0.3, ratio=g_direct/g_comp signed with
      abs guard 1e-6; ln2_rewrites iff >=6/10 layers |ratio|<=0.5;
      else mixed
  E2  window large-gain existence at the true workpoint (2853 T1
      threshold 1.2): no_amplification iff max_L|g_comp_true_mean|<1.2
      AND max_L|g_direct_true_mean|<1.2; else amplification_present
  E3  descriptive: Pearson r(g_direct_true_mean_L26_35,
      g_emp_2853_L26_35); g_emp from 2853 result.json
      T2_g_emp_L14_35 (clamp-state measured ratio, uncorrupted by #4)
  E4  descriptive: per-layer |g_comp_true_mean - g_comp_2860_mean|
      (pseudo- vs true-workpoint sensitivity of the compound arm)
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
OUT = BASE / 'phase2861' / 'g_true'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2853 = BASE / 'phase2853' / 'mlp_transmission' / 'result.json'
SRC_2860 = BASE / 'phase2860' / 'g_direct' / 'result.json'
SEED = 2855
MAX_WORDS = 8
LAST = 35
NL = 36
EPS = 1.0
WIN_LO, WIN_HI = 26, 36

PREREG = {
    'E1': '2860 criteria form at true workpoint: swiglu_intrinsic iff '
          '>=6/10 layers |ratio-1|<=0.3, ratio=g_direct/g_comp signed '
          '(abs guard 1e-6); ln2_rewrites iff >=6/10 layers '
          '|ratio|<=0.5; else mixed',
    'E2': 'no_amplification iff max_L|g_comp_true_mean|<1.2 AND '
          'max_L|g_direct_true_mean|<1.2 (2853 T1 threshold); else '
          'amplification_present',
    'E3': 'descriptive: Pearson r(g_direct_true_mean_L26_35, '
          'g_emp_2853_L26_35)',
    'E4': 'descriptive: per-layer |g_comp_true_mean - '
          'g_comp_2860_mean|',
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
                     'design': 'true-residual workpoint (LN2/mlp '
                               'pre-hook) g_ln/g_comp/g_direct, 80 '
                               'words L26-35 EPS=1.0, batched mlp, '
                               '2860 protocol otherwise verbatim'}
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

    cap = {'sain': {}, 'attn': {}, 'mlp': {},
           'ln2in': {}, 'mlpin': {}}

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
        handles.append(layer.self_attn.register_forward_hook(
            make_out_hook('attn', li)))
        handles.append(layer.mlp.register_forward_hook(
            make_out_hook('mlp', li)))
        handles.append(layer.self_attn.register_forward_pre_hook(
            make_in_hook('sain', li), with_kwargs=True))
        handles.append(layer.post_attention_layernorm
                       .register_forward_pre_hook(
                           make_in_hook('ln2in', li), with_kwargs=True))
        handles.append(layer.mlp.register_forward_pre_hook(
            make_in_hook('mlpin', li), with_kwargs=True))

    def clear_cap():
        for d in ('sain', 'attn', 'mlp', 'ln2in', 'mlpin'):
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
        ln2in = np.stack([cap['ln2in'][li][0][pos] for li in range(NL)])
        mlpin = np.stack([cap['mlpin'][li][0][pos] for li in range(NL)])
        aw = {li: out.attentions[li][0].float().cpu().numpy()
              for li in range(NL)}
        sain = {li: cap['sain'][li][0] for li in range(NL)}
        return hs, attn, mlp, aw, sain, ln2in, mlpin

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

    def mlp_batch(li, X):
        # raw SwiGLU, NO LN2 (caller applies LN2 explicitly); rows batch
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
    g_direct = np.zeros((n_words, n_win))
    g_ln = np.zeros((n_words, n_win))
    dln_norm = np.zeros((n_words, n_win))
    dln_cos = np.zeros((n_words, n_win))
    v1_all = []
    v2_all = []
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

        hs_iso, attn_iso, mlp_iso, _, _, _, _ = forward_run([w_tid], 0)
        iso0 = hs_iso[LAST] + attn_iso[LAST] + mlp_iso[LAST]
        raw2, aw2, sain2 = {}, {}, {}
        for cn, toks in c2.items():
            hs, attn, mlp, awc, sac, l2i, mpi = forward_run(toks, 1)
            raw2[cn] = (hs[LAST] + attn[LAST] + mlp[LAST]) - iso0
            aw2[cn] = awc
            sain2[cn] = sac
            if cn == 'same':
                attn_b2 = attn
                ln2in_b = l2i
                mlpin_b = mpi
                mlpout_b = mlp

        b13 = bias_for(aw2, 13, 30)
        patches[13]['map'] = {30: [(1, 1, b13)]}
        _, _, _, awc_c, _, _, _ = forward_run(c2['same'], 1)
        patches[13]['map'] = {}
        resid_all.append(abs(float(awc_c[13][30][1, 1])
                             - 0.5 * (float(aw2['func'][13][30][1, 1])
                                      + float(aw2['null'][13][30][1, 1]))))

        for q, l in enumerate(range(WIN_LO, WIN_HI)):
            ln_x = ln2_call(l, ln2in_b[l])
            dln = ln2_call(l, ln2in_b[l] + EPS * cdir) - ln_x
            outs = mlp_batch(l, np.stack([ln_x,
                                          ln_x + dln,
                                          mlpin_b[l] + EPS * cdir]))
            out_b, out_comp, out_direct = outs
            g_comp[i, q] = float((out_comp - out_b) @ cdir) / EPS
            g_direct[i, q] = float((out_direct - out_b) @ cdir) / EPS
            g_ln[i, q] = float(dln @ cdir) / EPS
            nd = float(np.linalg.norm(dln))
            dln_norm[i, q] = nd / EPS
            dln_cos[i, q] = float(dln @ cdir) / nd if nd > 1e-12 else 0.0
            v1_all.append(float(np.abs(ln_x - mlpin_b[l]).max()))
            v2_all.append(float(np.abs(out_b - mlpout_b[l]).max()))

        if (i + 1) % 20 == 0:
            print('P2861 words [%d/%d]' % (i + 1, n_words), flush=True)

    # ---------- verdicts ----------
    gcm = g_comp.mean(0)
    gdm = g_direct.mean(0)
    glm_ = g_ln.mean(0)
    dnm = dln_norm.mean(0)
    dcm = dln_cos.mean(0)

    n_close, n_rew = 0, 0
    ratios = []
    for q in range(n_win):
        if abs(gcm[q]) > 1e-6:
            ratio = float(gdm[q] / gcm[q])
        else:
            ratio = float('nan')
        ratios.append(ratio)
        if np.isfinite(ratio):
            if abs(ratio - 1.0) <= 0.3:
                n_close += 1
            if abs(ratio) <= 0.5:
                n_rew += 1
    if n_close >= 6:
        e1 = 'swiglu_intrinsic'
    elif n_rew >= 6:
        e1 = 'ln2_rewrites'
    else:
        e1 = 'mixed'

    mag_c = float(np.max(np.abs(gcm)))
    mag_d = float(np.max(np.abs(gdm)))
    e2 = 'no_amplification' if (mag_c < 1.2 and mag_d < 1.2) \
        else 'amplification_present'

    r2853 = json.loads(SRC_2853.read_text(encoding='utf-8'))
    g_emp = np.array(r2853['verdict']['T2_g_emp_L14_35'],
                     dtype=np.float64)
    assert g_emp.shape == (NL - 14,), g_emp.shape
    ge = g_emp[WIN_LO - 14: WIN_HI - 14]
    ok = np.isfinite(ge) & np.isfinite(gdm)
    e3_r = float(np.corrcoef(gdm[ok], ge[ok])[0, 1]) if ok.sum() >= 3 \
        else float('nan')

    gcm2860 = np.array(json.loads(SRC_2860.read_text(
        encoding='utf-8'))['verdict']['g_comp_mean_L26_35'])
    e4 = np.abs(gcm - gcm2860)

    v = {
        'n_words': n_words,
        'E1_source_true': e1,
        'n_close_layers': n_close,
        'n_rewrite_layers': n_rew,
        'E2_amplification': e2,
        'max_abs_g_comp_true': round(mag_c, 4),
        'max_abs_g_direct_true': round(mag_d, 4),
        'E3_r_direct_vs_emp': None if not np.isfinite(e3_r)
                              else round(e3_r, 4),
        'E4_max_abs_diff_vs_pseudo': round(float(e4.max()), 4),
        'ratio_direct_over_comp': [None if not np.isfinite(r)
                                   else round(r, 4) for r in ratios],
        'g_comp_true_mean_L26_35': [round(float(x), 4) for x in gcm],
        'g_direct_true_mean_L26_35': [round(float(x), 4) for x in gdm],
        'g_ln_true_mean_L26_35': [round(float(x), 4) for x in glm_],
        'g_diff_true_mean_L26_35': [round(float(a - b), 4)
                                    for a, b in zip(gcm, gdm)],
        'dln_norm_mean_L26_35': [round(float(x), 4) for x in dnm],
        'dln_cos_mean_L26_35': [round(float(x), 4) for x in dcm],
        'chk_v1_max': round(float(np.max(v1_all)), 6),
        'chk_v2_max': round(float(np.max(v2_all)), 6),
        'max_resid': round(float(np.max(resid_all)), 5),
        'final_verdict': '%s/%s' % (e1, e2),
    }

    result = {'phase': 2861, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'g_true.npz',
           g_comp=g_comp.astype(np.float32),
           g_direct=g_direct.astype(np.float32),
           g_ln=g_ln.astype(np.float32),
           dln_norm=dln_norm.astype(np.float32),
           dln_cos=dln_cos.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2861', elapsed)
    print('P2861 VERDICT %s' % json.dumps(v), flush=True)
    print('P2861 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
