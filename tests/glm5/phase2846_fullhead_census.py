"""Phase 2846 (LPF MA2 / ATLAS_PLAN frontline 2): full-head causal census.

36 layers x 32 heads = 1152 heads, 80 words.  For every head:
  (a) causal: single-head QK clamp drop (clamped to its own func/null
      self-mass; 2843/2844/2845 pipeline) measured on all 80 words;
  (b) attribution (free, from base-run captures): 2842-style source
      decomposition of the direct cdir write (pos0 = cond token,
      pos1 = w self) in the 2-token window.
Checkpointing: after each layer finishes all 80 words, the layer's
drops/shares are saved to census_L{l}.npz; a rerun skips completed
layers (resume-safe for the ~90 min budget).

Prereg (frozen before any readout):
  C1  exponential_dominant_full iff exponential fit beats power-law
      fit (R^2) for >= 60% of the 36 layers (per-layer 32-head
      mean-drop distribution)
  C2  concentrated_front iff top-64 heads (5.6% of 1152) carry
      >= 25% of the total positive load
      (load = mean_w max(drop, 0), summed over heads)
  C3  share_predicts_necessity_at_scale iff Spearman(mean direct
      cdir write, mean drop) over all 1152 heads >= 0.3
  verdict: census_complete (always, if all blocks measured);
           shape + concentration + mapping reported per C1-C3
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
OUT = BASE / 'phase2846' / 'fullhead_census'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SEED = 2846
MAX_WORDS = 8
LAST = 35
NL = 36
NH = 32

PREREG = {
    'C1': 'exponential_dominant_full iff >= 60% of 36 layers have '
          'exp fit R^2 > power-law fit R^2 (32-head mean-drop dist)',
    'C2': 'concentrated_front iff top-64 heads carry >= 25% of total '
          'positive load (mean_w max(drop,0))',
    'C3': 'share_predicts_necessity_at_scale iff Spearman(mean direct '
          'cdir write, mean drop) over 1152 heads >= 0.3',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def r2(y, yhat):
    y = np.asarray(y, dtype=np.float64)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / max(ss_tot, 1e-30)


def fit_shapes(vals):
    """vals: 32 per-head mean drops (may contain negatives). Fit on
    descending-sorted values: exponential = log-linear in rank,
    power-law = log-log in rank."""
    v = np.sort(np.asarray(vals, dtype=np.float64))[::-1]
    x = np.arange(1, len(v) + 1, dtype=np.float64)
    pos = v > 0
    if pos.sum() < 4:
        return {'r2_exp': 0.0, 'r2_pow': 0.0}
    lx = np.log(x[pos])
    lv = v[pos]
    b1 = np.polyfit(lx, lv, 1)
    yhat_e = np.polyval(b1, lx)
    r2_exp = r2(lv, yhat_e)
    b2 = np.polyfit(lx, np.log(lv), 1)
    yhat_p = np.exp(np.polyval(b2, lx))
    r2_pow = r2(lv, yhat_p)
    return {'r2_exp': float(r2_exp), 'r2_pow': float(r2_pow)}


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

    execution_path = OUT / 'execution.json'
    if not execution_path.exists():
        execution = {'timestamp': fc.stamp(),
                     'source': cc.snapshot(__file__),
                     'prereg': PREREG, 'seed': SEED,
                     'design': '1152 heads x 80 words, per-layer '
                               'checkpoints, resume-safe'}
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

    # ---------- targets (identical construction to 2843-2845) ----------
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

    def conds2_for(i, cat, w):
        w_tid = tid(w)
        same_cat = [x for x in targets[cat] if x != w]
        if not same_cat:
            same_cat = [x for x in CATS[cat] if x != w
                        and x in single_tok][:1]
        return {'same': [tid(same_cat[0]), w_tid],
                'func': [func_tid, w_tid],
                'null': [null_tids[i], w_tid]}

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

    # resume state
    drops_all = np.full((n_words, NL, NH), np.nan)   # (80, 36, 32)
    s0_all = np.full((n_words, NL, NH), np.nan)
    s1_all = np.full((n_words, NL, NH), np.nan)
    done_layers = set()
    for li in range(NL):
        pth = OUT / ('census_L%d.npz' % li)
        if pth.exists():
            z = np.load(pth)
            drops_all[:, li, :] = z['drops']
            s0_all[:, li, :] = z['s0']
            s1_all[:, li, :] = z['s1']
            done_layers.add(li)
    if done_layers:
        print('P2846 resume: %d layers already done %s'
              % (len(done_layers), sorted(done_layers)), flush=True)

    clamp_resid = []
    layers_done_this_run = []

    for li in range(NL):
        if li in done_layers:
            continue
        dL = np.zeros((n_words, NH))
        s0L = np.zeros((n_words, NH))
        s1L = np.zeros((n_words, NH))
        for i, (cat, w) in enumerate(target_list):
            cdir = dW_unit[CAT_WORDS.index(cat)]
            w_tid = tid(w)
            c2 = conds2_for(i, cat, w)
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
            cls_base = float(abs(d_spec_full @ cdir)) / nfull

            # attribution (free): direct cdir write per head
            vdt = next(vproj[li].parameters()).dtype
            Vc = {}
            for cn in c2:
                vin = sain2[cn][li][:2]
                Vc[cn] = vproj[li](torch.tensor(
                    vin, device='cuda', dtype=vdt)
                ).detach().float().cpu().numpy()
            for h in range(NH):
                kv = h // group
                s_dir = {}
                for cn in c2:
                    Vh = Vc[cn].reshape(2, n_kv, hd)[:, kv, :]
                    s_dir[cn] = aw2[cn][li][h][1, :2][:, None] \
                        * (Vh @ OV[(li, h)].T)
                ssp = s_dir['same'] \
                    - 0.5 * (s_dir['func'] + s_dir['null'])
                s0L[i, h] = float(ssp[0] @ cdir)
                s1L[i, h] = float(ssp[1] @ cdir)

            # causal clamp per head in this layer
            for h in range(NH):
                b = bias_for(aw2, li, h)
                patches[li]['map'] = {h: [(1, 1, b)]}
                hs, attn, mlp, awc, _ = forward_run(c2['same'], 1)
                patches[li]['map'] = {}
                clamp_resid.append(
                    abs(float(awc[li][h][1, 1])
                        - 0.5 * (float(aw2['func'][li][h][1, 1])
                                 + float(aw2['null'][li][h][1, 1]))))
                dsc = (hs[LAST] + attn[LAST] + mlp[LAST]) - iso0 \
                    - 0.5 * (raw2['func'] + raw2['null'])
                cls_c = float(abs(dsc @ cdir)) / nfull
                dL[i, h] = (cls_base - cls_c) / max(cls_base, 1e-30)
            if (i + 1) % 5 == 0:
                print('P2846 L%d words [%d/%d]'
                      % (li, i + 1, n_words), flush=True)
        drops_all[:, li, :] = dL
        s0_all[:, li, :] = s0L
        s1_all[:, li, :] = s1L
        np.savez(OUT / ('census_L%d.npz' % li),
                 drops=dL.astype(np.float32),
                 s0=s0L.astype(np.float32),
                 s1=s1L.astype(np.float32))
        layers_done_this_run.append(li)
        print('P2846 layer %d checkpointed (max_resid %.4f)'
              % (li, max(clamp_resid[-NH:]) if clamp_resid else 0.0),
              flush=True)

    # ---------- verdicts ----------
    mean_drop = drops_all.mean(0).reshape(NL * NH)     # (1152,)
    mean_s0 = s0_all.mean(0).reshape(NL * NH)
    mean_s1 = s1_all.mean(0).reshape(NL * NH)

    shape_fits = {li: fit_shapes(drops_all[:, li, :].mean(0))
                  for li in range(NL)}
    n_exp = sum(1 for f in shape_fits.values()
                if f['r2_exp'] > f['r2_pow'])
    c1 = n_exp >= int(0.6 * NL)
    shape_verdict = ('exponential_dominant_full' if c1
                     else 'mixed_or_powerlaw')

    load = np.maximum(mean_drop, 0.0)
    total_load = float(load.sum())
    order = np.argsort(-load)
    top64 = float(load[order[:64]].sum())
    frac_top64 = top64 / max(total_load, 1e-30)
    c2 = frac_top64 >= 0.25

    direct = mean_s0 + mean_s1
    c3_val = spearman(direct, mean_drop)
    c3 = c3_val >= 0.3

    v = {
        'n_words': n_words,
        'n_heads_measured': int(NL * NH),
        'layers_done_this_run': layers_done_this_run,
        'clamp_max_resid': round(float(np.max(clamp_resid)), 5)
            if clamp_resid else None,
        'n_layers_exp_beats_pow': n_exp,
        'C1_exponential_dominant_full': bool(c1),
        'shape_verdict': shape_verdict,
        'frac_top64_load': round(frac_top64, 4),
        'C2_concentrated_front': bool(c2),
        'C3_spearman_direct_drop': round(c3_val, 4),
        'C3_share_predicts_necessity': bool(c3),
        'mean_abs_drop_overall': round(
            float(np.mean(np.abs(mean_drop))), 5),
        'frac_negative_drop_heads': round(
            float(np.mean(mean_drop < 0)), 4),
        'top10_heads_by_load': [
            {'layer': int(k // NH), 'head': int(k % NH),
             'load': round(float(load[k]), 5)}
            for k in order[:10]],
    }

    result = {'phase': 2846, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'census_full.npz',
           mean_drop=mean_drop.astype(np.float64),
           mean_s0=mean_s0.astype(np.float64),
           mean_s1=mean_s1.astype(np.float64),
           drops_all=drops_all.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2846', elapsed)
    print('P2846 VERDICT %s' % json.dumps(v), flush=True)
    print('P2846 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
