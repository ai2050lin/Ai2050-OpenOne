"""Phase 2853 (LPF MA2 cont): MLP cdir transmission gain spectrum.

From 2852: the cdir "emergence" is carried by deep MLP layers
TRANSMITTING an already-displaced state (no positive-source layer;
mlp increments dominate: -1.15 over L30-34 vs attn +0.08).  The
open question is the per-layer transmission gain: do deep MLPs
merely pass the cdir offset through (passive, g ~ 1) or actively
amplify it (active, g >= 1.5)?

Measurement (80 words, matched protocol, 5 forwards/word + 46
perturbation MLP calls/word):
  x_in(l)  = sain_same[l][pos1] + attn_base[l]  (mlp input, base run)
  g_jac(l) = [mlp(x_in + EPS*cdir) - mlp(x_in)] . cdir / EPS
             (numerical Jacobian quadratic form along cdir)
  din_cdir(l)  = [(sain_c+attn_c) - (sain_b+attn_b)] . cdir  (pos1)
  dout_cdir(l) = [mlp_c - mlp_b] . cdir                      (= 2852)
  g_emp(l) = mean(dout_cdir) / mean(din_cdir)   [ratio of means]
  linearity diagnostic: same with EPS2 = 0.5 over L26..35.

dtype note: mlp params are bf16; at |x|~30 one bf16 ulp ~0.125, so
EPS = 1.0 (not 0.1) is preregistered to stay above quantization
noise; EPS2 = 0.5 is kept as a linearity check only.

Prereg (frozen before any readout):
  T1  active_amplification iff max over L26-35 of word-mean
      g_jac(EPS=1.0) >= 1.5;  passive_transmission iff median
      |g_jac| over L26-35 in [0.5, 1.5);  else nonlinear_unresolved
  T2  descriptive: g_emp spectrum L14-35; Pearson r(g_jac, g_emp)
      over L26-35; din/dout spectra; linearity ratio median
  verdict = T1 three-way
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
OUT = BASE / 'phase2853' / 'mlp_transmission'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2853
MAX_WORDS = 8
LAST = 35
NL = 36
EPS = 1.0
EPS2 = 0.5
WIN_LO = 26                      # T1 window L26..35

PREREG = {
    'T1': 'active_amplification iff max word-mean g_jac(EPS=1.0) '
          'over L26-35 >= 1.5; passive_transmission iff median '
          '|g_jac| over L26-35 in [0.5,1.5); else nonlinear_unresolved',
    'T2': 'g_emp = mean dout.cdir / mean din.cdir per layer; '
          'Pearson r(g_jac, g_emp) over L26-35; linearity ratio '
          'g(EPS2=0.5)/g(EPS=1.0) median over L26-35; descriptive',
    'verdict': 'T1 three-way',
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
                     'design': 'MLP cdir Jacobian gain spectrum, '
                               'EPS=1.0 (+EPS2=0.5 check), 80 words',
                     'eps': EPS, 'eps2': EPS2}
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

    mlps = {li: model.model.layers[li].mlp for li in range(NL)}
    vdt = next(mlps[0].parameters()).dtype

    def mlp_call(li, x_vec):
        t = torch.tensor(x_vec[None, :], device='cuda', dtype=vdt)
        with torch.no_grad():
            o = mlps[li](t)
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

    # accumulators
    g_jac = np.zeros((n_words, NL))      # EPS=1.0, all layers
    g_jac05 = np.zeros((n_words, NL - WIN_LO))   # EPS2=0.5, L26..35
    din_cdir = np.zeros((n_words, NL))   # mlp-input delta . cdir
    dout_cdir = np.zeros((n_words, NL))  # mlp-output delta . cdir
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

        for l in range(NL):
            # mlp input at pos1: sain (hook, seq-major) + attn out
            x_in = sain2['same'][l][1] + attn_b2[l]
            out_b = mlp_b2[l]                    # hook-captured base out
            din = (sain_c[l][1] + attn_c[l]) \
                - (sain2['same'][l][1] + attn_b2[l])
            dout = mlp_c[l] - mlp_b2[l]
            din_cdir[i, l] = float(din @ cdir)
            dout_cdir[i, l] = float(dout @ cdir)
            out_p = mlp_call(l, x_in + EPS * cdir)
            g_jac[i, l] = float((out_p - out_b) @ cdir) / EPS
            if l >= WIN_LO:
                out_p2 = mlp_call(l, x_in + EPS2 * cdir)
                g_jac05[i, l - WIN_LO] = \
                    float((out_p2 - out_b) @ cdir) / EPS2

        if (i + 1) % 20 == 0:
            print('P2853 words [%d/%d]' % (i + 1, n_words), flush=True)

    # ---------- verdicts ----------
    gj = g_jac.mean(0)                       # (36,)
    late = gj[WIN_LO:]                       # L26..35
    late_abs = np.abs(late)

    if float(late_abs.max()) >= 1.5:
        t1_verdict = 'active_amplification'
    elif 0.5 <= float(np.median(late_abs)) < 1.5:
        t1_verdict = 'passive_transmission'
    else:
        t1_verdict = 'nonlinear_unresolved'

    dout_m = dout_cdir.mean(0)
    din_m = din_cdir.mean(0)
    g_emp = np.full(NL, np.nan)
    for l in range(NL):
        if abs(din_m[l]) > 1e-6:
            g_emp[l] = dout_m[l] / din_m[l]

    x = gj[WIN_LO:]
    y = g_emp[WIN_LO:]
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() >= 3 and np.std(x[ok]) > 1e-9 and np.std(y[ok]) > 1e-9:
        r_jac_emp = float(np.corrcoef(x[ok], y[ok])[0, 1])
    else:
        r_jac_emp = None

    g05 = g_jac05.mean(0)
    ok2 = np.abs(gj[WIN_LO:]) > 1e-6
    lin_ratio = float(np.median(g05[ok2] / gj[WIN_LO:][ok2])) \
        if ok2.sum() >= 3 else None

    v = {
        'n_words': n_words,
        'T1_verdict': t1_verdict,
        'g_jac_late_L26_35': [round(float(x), 4) for x in late],
        'g_jac_max_late': round(float(late_abs.max()), 4),
        'g_jac_median_abs_late': round(float(np.median(late_abs)), 4),
        'T2_g_emp_L14_35': [round(float(v_), 4)
                            if np.isfinite(v_) else None
                            for v_ in g_emp[14:]],
        'T2_r_jac_emp': round(r_jac_emp, 4) if r_jac_emp is not None
            else None,
        'T2_lin_ratio_median': round(lin_ratio, 4)
            if lin_ratio is not None else None,
        'din_cdir_profile': [round(float(x), 4) for x in din_m],
        'dout_cdir_profile': [round(float(x), 4) for x in dout_m],
        'g_jac_full': [round(float(x), 4) for x in gj],
        'max_resid': round(float(np.max(resid_all)), 5)
            if resid_all else None,
    }

    result = {'phase': 2853, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'transmission.npz',
           g_jac=g_jac.astype(np.float32),
           g_jac05=g_jac05.astype(np.float32),
           din_cdir=din_cdir.astype(np.float32),
           dout_cdir=dout_cdir.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2853', elapsed)
    print('P2853 VERDICT %s' % json.dumps(v), flush=True)
    print('P2853 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
