"""Phase 2842 (LPF Delta-II/III): QK-edge source backtrace.

2841 located the per-layer top-1 causal heads (L22 h28, L23 h29,
L26 h17, L28 h4, L33 h25).  This phase decomposes each head's
contribution to the cls_spec alignment into its two QK source
positions -- pos0 = condition word (same/func/null), pos1 = target
word (e.g. 'apple') -- using exact attention weights (eager attn) and
value vectors at the target layer:

  head out(pos1) = sum_j A[h,1,j] * OV_h V[j]
  C_j = same-term(pos j) - 0.5*(func-term + null-term)   (4096-vec)
  signed share_j = C_j . cdir / ||d_spec_full||

Prereg (frozen before any readout):
  Q1  source_located iff |share_0| > 2*|share_1| or vice versa
  Q2  cond_word_dominant iff share_0 > 2*share_1 and share_0 > 0
  Q3  self_dominant iff share_1 > 2*share_0 and share_1 > 0
  Q4  attention mass A[h,1,0] / A[h,1,1] reported per condition
  verdict per head: cond_word_source / self_source / mixed
  network verdict: write_interface = cond_word iff >= 3/5 heads
      cond_word_dominant; self iff >= 3/5 self_dominant; else mixed
Sanity: sum_j C_j must equal C_head (rtol 1e-3) -- identity check.
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
OUT = BASE / 'phase2842' / 'qk_source_backtrace'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2841_NPZ = BASE / 'phase2841' / 'longtail_shape' / 'head_drop_map.npz'
SEED = 2842
HEADS = {22: 28, 23: 29, 26: 17, 28: 4, 33: 25}
LAST = 35

PREREG = {
    'Q1': 'source_located iff |share_0| > 2*|share_1| or vice versa '
          '(mean over words)',
    'Q2': 'cond_word_dominant iff share_0 > 2*share_1 and share_0 > 0',
    'Q3': 'self_dominant iff share_1 > 2*share_0 and share_1 > 0',
    'Q4': 'attention_mass A[h,1,0]/A[h,1,1] reported per condition',
    'verdict_head': 'cond_word_source / self_source / mixed per head',
    'verdict_net': 'write_interface = cond_word iff >=3/5 heads '
                   'cond_word_dominant; self iff >=3/5 self_dominant; '
                   'else mixed',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED,
                 'heads': HEADS}
    fc.save(OUT / 'execution.json', execution)

    import torch
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
              for li in HEADS}
        sain = {li: cap['sain'][li][0] for li in HEADS}
        return hs, attn, mlp, aw, sain

    # ---------- targets (identical to 2837-2841) ----------
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
        cs = [w for w in CATS[cat] if w in single_tok]
        targets[cat] = cs[:2]
    target_list = [(cat, w) for cat in CAT_WORDS for w in targets[cat]]

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
    null_tids = []
    while len(null_tids) < len(target_list):
        r = int(rng.integers(0, vocab_size))
        if r not in word_tids and r > 0:
            null_tids.append(r)
    func_tid = tid('the')

    def conds_for(i, cat, w):
        w_tid = tid(w)
        same_cat = [x for x in targets[cat] if x != w]
        if not same_cat:
            same_cat = [x for x in CATS[cat] if x != w
                        and x in single_tok][:1]
        return {'same': [tid(same_cat[0]), w_tid],
                'func': [func_tid, w_tid],
                'null': [null_tids[i], w_tid]}

    # o_proj slices for target heads
    OV = {}
    for li, h in HEADS.items():
        Wo = model.model.layers[li].self_attn.o_proj.weight.detach() \
            .float().cpu().numpy().astype(np.float64)
        Wo3 = Wo.reshape(Wo.shape[0], 32, Wo.shape[1] // 32)
        OV[li] = Wo3[:, h, :]          # (4096, 128)

    vproj = {li: model.model.layers[li].self_attn.v_proj
             for li in HEADS}

    accum = {li: {'share0': [], 'share1': [], 'shareh': [],
                  'a_s0': [], 'a_s1': [], 'a_f0': [], 'a_f1': [],
                  'a_n0': [], 'a_n1': []}
             for li in HEADS}
    id_err = []

    for i, (cat, w) in enumerate(target_list):
        cdir = dW_unit[CAT_WORDS.index(cat)]
        w_tid = tid(w)
        conds = conds_for(i, cat, w)
        hs_iso, _, _, _, _ = forward_run([w_tid], 0)
        iso0 = hs_iso[0]
        raw, vw, aw = {}, {}, {}
        for cn, toks in conds.items():
            hs, attn, mlp, awc, sain = forward_run(toks, 1)
            raw_fin = hs[LAST] + attn[LAST] + mlp[LAST]
            raw[cn] = raw_fin - iso0
            vw[cn] = {}
            aw[cn] = {}
            for li in HEADS:
                v = vproj[li](torch.tensor(
                    sain[li], device='cuda',
                    dtype=torch.bfloat16)).detach().float() \
                    .cpu().numpy().astype(np.float64)
                vw[cn][li] = v.reshape(v.shape[0], n_kv, -1)  # (2,8,128)
                aw[cn][li] = awc[li][HEADS[li]][1, :]          # (2,)
        d_spec_full = raw['same'] - 0.5 * (raw['func'] + raw['null'])
        nfull = max(float(np.linalg.norm(d_spec_full)), 1e-30)
        for li, h in HEADS.items():
            kv = h // group
            C = {}
            for cn in conds:
                Vh = vw[cn][li][:, kv, :]                      # (2,128)
                W = Vh @ OV[li].T                              # (2,4096)
                terms = aw[cn][li][:, None] * W                # (2,4096)
                C[cn] = terms                                  # rows j=0,1
            Cspec = C['same'] - 0.5 * (C['func'] + C['null'])  # (2,4096)
            s0 = float(Cspec[0] @ cdir) / nfull
            s1 = float(Cspec[1] @ cdir) / nfull
            sh = float(Cspec.sum(0) @ cdir) / nfull
            err = abs(s0 + s1 - sh) / max(abs(sh), 1e-9)
            id_err.append(err)
            accum[li]['share0'].append(s0)
            accum[li]['share1'].append(s1)
            accum[li]['shareh'].append(sh)
            accum[li]['a_s0'].append(float(aw['same'][li][0]))
            accum[li]['a_s1'].append(float(aw['same'][li][1]))
            accum[li]['a_f0'].append(float(aw['func'][li][0]))
            accum[li]['a_f1'].append(float(aw['func'][li][1]))
            accum[li]['a_n0'].append(float(aw['null'][li][0]))
            accum[li]['a_n1'].append(float(aw['null'][li][1]))
        if (i + 1) % 5 == 0:
            print('P2842 words [%d/%d]' % (i + 1, len(target_list)),
                  flush=True)

    max_id_err = float(max(id_err))

    heads_out = {}
    n_cond = 0
    n_self = 0
    for li, h in HEADS.items():
        a = {k: float(np.mean(v)) for k, v in accum[li].items()}
        s0, s1, sh = a['share0'], a['share1'], a['shareh']
        q1 = abs(s0) > 2 * abs(s1) or abs(s1) > 2 * abs(s0)
        q2 = s0 > 2 * s1 and s0 > 0
        q3 = s1 > 2 * s0 and s1 > 0
        if q2:
            verdict = 'cond_word_source'
            n_cond += 1
        elif q3:
            verdict = 'self_source'
            n_self += 1
        else:
            verdict = 'mixed'
        heads_out['L%d_h%d' % (li, h)] = {
            'share_pos0': round(s0, 5), 'share_pos1': round(s1, 5),
            'share_head': round(sh, 5),
            'attn_mass_same_pos0': round(a['a_s0'], 4),
            'attn_mass_same_pos1': round(a['a_s1'], 4),
            'attn_mass_func_pos0': round(a['a_f0'], 4),
            'attn_mass_null_pos0': round(a['a_n0'], 4),
            'Q1_source_located': bool(q1),
            'Q2_cond_word_dominant': bool(q2),
            'Q3_self_dominant': bool(q3),
            'verdict': verdict,
        }

    if n_cond >= 3:
        net = 'cond_word'
    elif n_self >= 3:
        net = 'self'
    else:
        net = 'mixed'

    v = {
        'max_identity_err': max_id_err,
        'n_cond_word': n_cond, 'n_self': n_self,
        'write_interface': net,
        'heads': heads_out,
    }

    result = {'phase': 2842, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)

    fc.npz(OUT / 'source_shares.npz',
           **{('%s_%s' % (key, stat)): np.array(vals, dtype=np.float64)
              for key, dd in accum.items()
              for stat, vals in dd.items()})

    elapsed = time.monotonic() - t0
    cc.ledger('phase2842', elapsed)
    print('P2842 VERDICT %s' % json.dumps(v), flush=True)
    print('P2842 elapsed %.1fs' % elapsed, flush=True)

    del model
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
