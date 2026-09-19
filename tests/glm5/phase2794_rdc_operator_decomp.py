"""Phase 2794 (LPF-7): operator decomposition of the word-position
pipeline + corrected context scrub.

2793 proved the word-position property readout is causally
self-sufficient (bit-exact immunity to post-word context; raw
material = the token embedding alone).  This phase asks WHICH
operators do the construction: per-layer zeroing of attention vs MLP
at the word position (Arm A), and a corrected context test with the
content word BEFORE the target plus a long-prefix condition (Arm B).

Arms:
  A  12 sentences (6 neutral + 6 swap_late, same set as 2790/2791):
     for each layer l in 0..35 zero EITHER the word-position
     attention output OR the word-position MLP output; read margin
     at l*=30.  Curves attn_drop[l], mlp_drop[l].
  B  6 words x 4 sentence conditions:
     full_new  "Compared to the {nxt}, the {w} was seen at the
                market yesterday"   (content word BEFORE target)
     scrub_new same with {nxt} -> item
     long_pre  "In the garden behind the old wooden fence, the {w}
                was seen yesterday"
     min       "The {w}"

Prereg (frozen before any forward):
  P-L  mlp_dominant iff sum_l |mlp_drop(l)| >= 3 x sum_l
       |attn_drop(l)| (mean over the 12 sentences).
  P-M  attention_participates iff max_l attn_drop(l) >= 0.5.
  P-N  prefix_content_word_effect iff
       |mean(margin_full_new - margin_scrub_new)| >= 0.5.
  P-O  long_prefix_immunity iff
       |mean(margin_long_pre - margin_min)| < 0.5.
  D    descriptive: full per-layer/per-operator tables; per-word
       Arm-B table.
verdict: families independent.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2794' / 'qwen4_operator_decomp'

WORDS = ['apple', 'dog', 'gold', 'Japan', 'car', 'ocean']
PROPS = {
    'apple': ['fruit', 'red', 'plant'],
    'dog': ['animal', 'mammal', 'pet'],
    'gold': ['metal', 'yellow', 'precious'],
    'Japan': ['country', 'Japanese', 'Tokyo'],
    'car': ['vehicle', 'drive', 'wheels'],
    'ocean': ['water', 'sea', 'salt'],
}
RIVAL = {w: [p for w2 in WORDS if w2 != w for p in PROPS[w2]]
         for w in WORDS}
L_STAR = 30
N_LAYERS = 36

PREREG = {
    'P-L': 'mlp_dominant iff sum_l |mlp_drop| >= 3 x sum_l '
           '|attn_drop| (word-position zeroing, margin at l*=30)',
    'P-M': 'attention_participates iff max_l attn_drop >= 0.5',
    'P-N': 'prefix_content_word_effect iff '
           '|mean(full_new - scrub_new)| >= 0.5',
    'P-O': 'long_prefix_immunity iff '
           '|mean(long_pre - min)| < 0.5',
    'verdict': 'families independent',
}


def main():
    import torch
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'words': WORDS, 'l_star': L_STAR}
    fc.save(OUT / 'execution.json', execution)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok2 = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    final_norm = model.model.norm

    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            tc[t] = int(ids[0])
        return tc[t]

    def lens_np(h):
        with torch.inference_mode():
            hn = final_norm(
                torch.tensor(h, device=device).unsqueeze(0))
            return W_U @ hn[0].float().cpu().numpy()

    def margin_z(z, w):
        zp = float(np.mean([z[tid(pr)] for pr in PROPS[w]]))
        zr = float(np.mean([z[tid(r)] for r in RIVAL[w]]))
        return zp - zr

    def make_zero_hook(p):
        def hook(module, inp, outp):
            if isinstance(outp, tuple):
                t = outp[0].clone()
                t[0, p, :] = 0.0
                return (t,) + outp[1:]
            t = outp.clone()
            t[0, p, :] = 0.0
            return t
        return hook

    def forward(ids, zero_spec, p):
        hs_list = []
        for kind, l in zero_spec:
            mod = (model.model.layers[l].self_attn if kind == 'attn'
                   else model.model.layers[l].mlp)
            hs_list.append(mod.register_forward_hook(make_zero_hook(p)))
        try:
            with torch.inference_mode():
                o = model(torch.tensor([ids], device=device),
                          output_hidden_states=True)
        finally:
            for h in hs_list:
                h.remove()
        return o.hidden_states[L_STAR][0, p].float().cpu().numpy()

    # ---------- Arm A: per-layer attn vs mlp zeroing ----------
    bsents = []
    for i, w in enumerate(WORDS):
        nxt = WORDS[(i + 1) % 6]
        bsents.append((w, 'The', ' ' + w, '', 1, 'neutral'))
        bsents.append((w, 'Unlike the', ' ' + w,
                       ', the ' + nxt + ' was seen at the market '
                       'yesterday', 2, 'swap_late'))
    assert len(bsents) == 12

    attn_drop = {l: [] for l in range(N_LAYERS)}
    mlp_drop = {l: [] for l in range(N_LAYERS)}
    for (w, pre, span, suf, p, tag) in bsents:
        ids = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(span, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        h0 = forward(ids, [], p)
        m0 = margin_z(lens_np(h0), w)
        for l in range(N_LAYERS):
            ha = forward(ids, [('attn', l)], p)
            ma = margin_z(lens_np(ha), w)
            attn_drop[l].append(m0 - ma)
            hm = forward(ids, [('mlp', l)], p)
            mm = margin_z(lens_np(hm), w)
            mlp_drop[l].append(m0 - mm)
        print('P2794 A %s(%s) m0=%.2f done' % (w, tag, m0), flush=True)

    attn_mean = {l: float(np.mean(attn_drop[l]))
                 for l in range(N_LAYERS)}
    mlp_mean = {l: float(np.mean(mlp_drop[l]))
                for l in range(N_LAYERS)}
    sum_a = float(np.sum(np.abs(list(attn_mean.values()))))
    sum_m = float(np.sum(np.abs(list(mlp_mean.values()))))
    max_a = float(np.max(np.abs(list(attn_mean.values()))))
    p_l = bool(sum_m >= 3 * sum_a)
    p_m = bool(max_a >= 0.5)
    print('P2794 sum|attn|=%.3f sum|mlp|=%.3f max|attn|=%.3f '
          'P-L=%s P-M=%s' % (sum_a, sum_m, max_a, p_l, p_m),
          flush=True)

    # ---------- Arm B: corrected scrub + long prefix ----------
    conds = {}
    for i, w in enumerate(WORDS):
        nxt = WORDS[(i + 1) % 6]
        conds[w] = {
            'full_new': ('Compared to the', ' ' + w,
                         ', the ' + nxt + ' was seen at the market '
                         'yesterday'),
            'scrub_new': ('Compared to the', ' ' + w,
                          ', the item was seen at the market '
                          'yesterday'),
            'long_pre': ('In the garden behind the old wooden fence',
                         ' ' + w, ' was seen yesterday'),
            'min': ('The', ' ' + w, ''),
        }
    B = {w: {} for w in WORDS}
    for w in WORDS:
        for tag, (pre, span, suf) in conds[w].items():
            ids = tok(pre, add_special_tokens=False)['input_ids'] + \
                tok(span, add_special_tokens=False)['input_ids'] + \
                tok(suf, add_special_tokens=False)['input_ids']
            p = len(tok(pre, add_special_tokens=False)['input_ids'])
            with torch.inference_mode():
                o = model(torch.tensor([ids], device=device),
                          output_hidden_states=True)
            z = lens_np(o.hidden_states[L_STAR][0, p].float().cpu()
                        .numpy())
            B[w][tag] = margin_z(z, w)
        print('P2794 B %s %s' % (w, {t: round(v, 2) for t, v
                                     in B[w].items()}), flush=True)

    fn = np.array([B[w]['full_new'] for w in WORDS])
    sn = np.array([B[w]['scrub_new'] for w in WORDS])
    lp = np.array([B[w]['long_pre'] for w in WORDS])
    mn = np.array([B[w]['min'] for w in WORDS])
    d_pre = float(np.mean(fn - sn))
    d_long = float(np.mean(lp - mn))
    p_n = bool(abs(d_pre) >= 0.5)
    p_o = bool(abs(d_long) < 0.5)
    print('P2794 full_new=%.3f scrub_new=%.3f d_pre=%.3f | '
          'long=%.3f min=%.3f d_long=%.3f P-N=%s P-O=%s'
          % (fn.mean(), sn.mean(), d_pre, lp.mean(), mn.mean(),
             d_long, p_n, p_o), flush=True)

    verdict = {
        'mlp_dominant': p_l, 'attention_participates': p_m,
        'prefix_content_word_effect': p_n,
        'long_prefix_immunity': p_o,
        'sum_abs_attn': sum_a, 'sum_abs_mlp': sum_m,
        'max_abs_attn': max_a,
        'attn_drop_mean': attn_mean, 'mlp_drop_mean': mlp_mean,
        'd_prefix_content': d_pre, 'd_long_prefix': d_long,
        'arm_b_margins': B,
    }
    result = {'phase': 2794, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'operator_curves.npz',
           layers=np.arange(N_LAYERS, dtype=np.int64),
           attn=np.array([attn_mean[l] for l in range(N_LAYERS)]),
           mlp=np.array([mlp_mean[l] for l in range(N_LAYERS)]),
           words=np.array(WORDS, dtype=np.str_))
    print('P2794 VERDICT %s' % json.dumps(
        {k: v for k, v in verdict.items()
         if k not in ('arm_b_margins', 'attn_drop_mean',
                      'mlp_drop_mean')}), flush=True)


if __name__ == '__main__':
    main()
