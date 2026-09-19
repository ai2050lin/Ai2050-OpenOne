"""Phase 2854 (LPF MA2 cont): g_jac specificity controls + operating
point shift.

From 2853: deep MLPs show a huge small-signal cdir gain
(g_jac = cdir.J.cdir up to 51 at L34, base workpoint) while the
actual clamped-state transmission is mild (g_emp 0.2-1.7).  Two
open confounds, both fixed here:

  CONF-1 (missing null): g_jac(cdir) may sit inside the generic
      J-spectrum background.  Control: N_DIR=16 random unit
      directions per word, same protocol -> null distribution per
      layer; z = (g_cdir - median(null)) / (1.4826*MAD(null)).
  CONF-2 (workpoint): g_jac was measured at the BASE workpoint.
      Re-measure at the CLAMPED workpoint (x_in' = sain_c + attn_c)
      -> g'.  Workpoint-shift hypothesis: g' << g explains why the
      experienced gain g_emp is mild (saturation brake).

Batching: mlp is elementwise, so one batched call with rows
[cdir, r1..r16] evaluates all directions at once (2 batched calls
per word per layer).

Prereg (frozen before any readout):
  Z1  cdir_specific iff >= 6 of L26-35 layers have z >= 3
      (null = 80 words x 16 random dirs, base workpoint, EPS=1.0)
  Z2  operating_point_shift iff >= 6 of L26-35 layers have
      g'_cdir / g_cdir < 0.5   (means over words)
  verdict: saturated_cdir_lattice iff Z1 and Z2
           active_cdir_lattice   iff Z1 and not Z2
           anisotropic_background otherwise
  D1  descriptive: consistency of g_cdir_base with 2853's g_jac
      (Pearson r over L26-35, loaded from 2853 result.json)
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
OUT = BASE / 'phase2854' / 'gain_specificity'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2853 = BASE / 'phase2853' / 'mlp_transmission' / 'result.json'
SEED = 2854
MAX_WORDS = 8
LAST = 35
NL = 36
EPS = 1.0
N_DIR = 16
WIN_LO, WIN_HI = 26, 36          # T-window L26..35

PREREG = {
    'Z1': 'cdir_specific iff >=6 of L26-35 layers have '
          'z=(g_cdir-median(null))/(1.4826*MAD(null)) >= 3, '
          'null = 80 words x 16 random unit dirs, base workpoint',
    'Z2': 'operating_point_shift iff >=6 of L26-35 layers have '
          'g_clamp_cdir / g_base_cdir < 0.5 (mean over words)',
    'verdict': 'saturated_cdir_lattice iff Z1 and Z2; '
               'active_cdir_lattice iff Z1 and not Z2; '
               'anisotropic_background otherwise',
    'D1': 'Pearson r(g_cdir_base, 2853 g_jac) over L26-35, '
          'consistency check, descriptive',
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
                     'design': 'random-direction null + clamped '
                               'workpoint gain, EPS=1.0, 80 words',
                     'eps': EPS, 'n_dir': N_DIR}
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

    def mlp_batch(li, X):
        # MUST go through post_attention_layernorm: the real forward
        # computes mlp(LN2(h)) (Qwen3MLP has no internal LN).  Gen1
        # omitted LN2 -> baseline mismatch mlp_raw(x+e.d) vs
        # mlp(LN2(h)) -> fabricated 5-51 "gains" (erratum vs 2853).
        t = torch.tensor(X, device='cuda', dtype=vdt)
        with torch.no_grad():
            h = model.model.layers[li].post_attention_layernorm(t)
            o = mlps[li](h)
        return o.detach().float().cpu().numpy().astype(np.float64)

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
    g_cdir_base = np.zeros((n_words, NL))
    g_cdir_clamp = np.zeros((n_words, NL))
    g_rand_base = np.zeros((n_words, NL, N_DIR))
    g_rand_clamp = np.zeros((n_words, NL, N_DIR))
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

        # word-specific random unit directions
        rw = np.random.default_rng(SEED * 1000 + i)
        dirs = rw.standard_normal((N_DIR, hs_b2.shape[1]))
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
        all_dirs = np.vstack([cdir[None, :], dirs])   # (1+N_DIR, dim)

        for l in range(NL):
            x_base = sain2['same'][l][1] + attn_b2[l]
            out_b = mlp_b2[l]
            out_c = mlp_c[l]
            x_clamp = sain_c[l][1] + attn_c[l]

            Xb = x_base[None, :] + EPS * all_dirs
            ob = mlp_batch(l, Xb)                  # (1+N_DIR, dim)
            resp = ((ob - out_b[None, :]) * all_dirs).sum(1)
            g_cdir_base[i, l] = resp[0] / EPS
            g_rand_base[i, l] = resp[1:] / EPS

            Xc = x_clamp[None, :] + EPS * all_dirs
            oc = mlp_batch(l, Xc)
            resp2 = ((oc - out_c[None, :]) * all_dirs).sum(1)
            g_cdir_clamp[i, l] = resp2[0] / EPS
            g_rand_clamp[i, l] = resp2[1:] / EPS

        if (i + 1) % 20 == 0:
            print('P2854 words [%d/%d]' % (i + 1, n_words), flush=True)

    # ---------- verdicts ----------
    gcb = g_cdir_base.mean(0)                # (36,)
    gcc = g_cdir_clamp.mean(0)
    grb = g_rand_base.mean(0)                # (36, N_DIR) mean over words

    z_late = np.zeros(WIN_HI - WIN_LO)
    for q, l in enumerate(range(WIN_LO, WIN_HI)):
        null = g_rand_base[:, l, :].reshape(-1)   # 80 x 16 pooled
        med = float(np.median(null))
        mad = float(np.median(np.abs(null - med)))
        denom = 1.4826 * mad
        z_late[q] = (gcb[l] - med) / denom if denom > 1e-9 else np.inf

    z1 = bool((z_late >= 3.0).sum() >= 6)

    # ratio: signed denominators -- direct division with abs guard
    # (max(x,1e-9) returns 1e-9 for negative x -> 1e8 artifacts,
    #  5th occurrence of the guard-vs-negative-denominator lesson)
    ratio = np.full(WIN_HI - WIN_LO, np.nan)
    for q in range(WIN_HI - WIN_LO):
        if abs(gcb[WIN_LO + q]) > 1e-6:
            ratio[q] = gcc[WIN_LO + q] / gcb[WIN_LO + q]
    valid = np.isfinite(ratio)
    z2 = bool(valid.sum() >= 8 and (ratio[valid] < 0.5).sum() >= 6)

    if z1 and z2:
        verdict = 'saturated_cdir_lattice'
    elif z1:
        verdict = 'active_cdir_lattice'
    else:
        verdict = 'anisotropic_background'

    # D1 consistency with 2853
    r_2853 = None
    try:
        v53 = json.loads(SRC_2853.read_text(encoding='utf-8'))['verdict']
        gj53 = np.array(v53['g_jac_late_L26_35'], dtype=np.float64)
        ok = np.isfinite(gj53) & np.isfinite(gcb[WIN_LO:])
        if ok.sum() >= 3:
            r_2853 = float(np.corrcoef(gj53[ok], gcb[WIN_LO:][ok])[0, 1])
    except Exception:
        pass

    v = {
        'n_words': n_words,
        'Z1_cdir_specific': z1,
        'z_cdir_L26_35': [round(float(x), 2) for x in z_late],
        'z_min_late': round(float(z_late.min()), 2),
        'z_median_late': round(float(np.median(z_late)), 2),
        'Z2_operating_point_shift': z2,
        'ratio_clamp_over_base_L26_35': [round(float(x), 4)
                                         if np.isfinite(x) else None
                                         for x in ratio],
        'g_cdir_base_L26_35': [round(float(x), 3)
                               for x in gcb[WIN_LO:]],
        'g_cdir_clamp_L26_35': [round(float(x), 3)
                                for x in gcc[WIN_LO:]],
        'g_rand_base_mean_L26_35': [round(float(x), 3)
                                    for x in grb[WIN_LO:].mean(1)],
        'D1_r_vs_2853': round(r_2853, 4) if r_2853 is not None else None,
        'max_resid': round(float(np.max(resid_all)), 5)
            if resid_all else None,
        'final_verdict': verdict,
    }

    result = {'phase': 2854, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'specificity.npz',
           g_cdir_base=g_cdir_base.astype(np.float32),
           g_cdir_clamp=g_cdir_clamp.astype(np.float32),
           g_rand_base=g_rand_base.astype(np.float32),
           g_rand_clamp=g_rand_clamp.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2854', elapsed)
    print('P2854 VERDICT %s' % json.dumps(v), flush=True)
    print('P2854 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
