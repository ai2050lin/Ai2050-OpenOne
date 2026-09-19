"""Phase 2792 (LPF-5): purify pipeline quantification.

2791 replaced the injection model with the purify model: the token
embedding already carries a huge property prior (margin 25.1 vs peak
readout 7.96), L0 compresses it, early layers L4-17 establish rival
separation (bundled C3 zeroing flips 4/12 margins negative), mid
layers shape lens readability.  Three closed questions remain:
  (A) WHERE in the layer stack does rival sorting first become
      correct (sort milestone) vs where does lens readability emerge
      (2790: L24+)?  Full-layer margin + hard-sort margin
      margin_sort(l) = mean_prop [z(prop) - max_RIVAL z(r)].
  (B) Is early rival separation itself single-layer critical
      (per-layer zeroing inside L4-17) or distributed?
  (C) Is the embedding prior the SAME semantic content as the mid
      readout (top-50 overlap of property tokens)?

Prereg (frozen before any forward):
  P-F  sort_before_readable iff the global margin_sort curve has a
       milestone l_clean (first layer with global mean > 0 that
       stays > 0 through L35) AND l_clean <= 20.
  P-G  early_separation_localized iff max over l in 4..17 of the
       single-layer word-position MLP-zero drop (at l* = 30 readout)
       is >= 2.0.
  P-H  prior_readout_same_source iff >= 5/6 words have >= 1
       property token present in BOTH the embedding lens top-50 and
       the l* lens top-50 (per-word mean residual at l*).
  D    descriptive: full purify-ratio curve margin(l)/margin_emb;
       per-layer early drop table; per-word top-50 overlaps.
verdict: sort_before_readable iff P-F; localized iff P-G;
         same_source iff P-H (independent families).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2792' / 'qwen4_purify_pipeline'

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
EARLY_RANGE = list(range(4, 18))

PREREG = {
    'P-F': 'sort_before_readable iff global margin_sort curve has '
           'milestone l_clean (first l with mean>0 staying >0 '
           'through L35) AND l_clean <= 20',
    'P-G': 'early_separation_localized iff max single-layer drop '
           'over l in 4..17 >= 2.0 at l* readout',
    'P-H': 'prior_readout_same_source iff >= 5/6 words have >= 1 '
           'property token in BOTH embedding lens top-50 and l* '
           'lens top-50',
    'verdict': 'sort_before_readable iff P-F; localized iff P-G; '
               'same_source iff P-H',
}


def main():
    import torch
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'words': WORDS,
                 'early_range': EARLY_RANGE, 'l_star': L_STAR}
    fc.save(OUT / 'execution.json', execution)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok2 = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    final_norm = model.model.norm
    emb_table = model.model.embed_tokens.weight.detach().float().cpu()

    tc = {}
    multi = []

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                multi.append(t)
            tc[t] = int(ids[0])
        return tc[t]

    def lens_np(h):
        with torch.inference_mode():
            hn = final_norm(
                torch.tensor(h, device=device).unsqueeze(0))
            return W_U @ hn[0].float().cpu().numpy()

    def margins_z(z, w):
        zp = float(np.mean([z[tid(pr)] for pr in PROPS[w]]))
        zr = float(np.mean([z[tid(r)] for r in RIVAL[w]]))
        zmax = float(max(z[tid(r)] for r in RIVAL[w]))
        return zp - zr, zp - zmax

    margin_emb, sort_emb = {}, {}
    top50_emb = {}
    for w in WORDS:
        z = lens_np(emb_table[tid(w)].numpy())
        margin_emb[w], sort_emb[w] = margins_z(z, w)
        top = np.argsort(-z)[:50]
        top50_emb[w] = set(tok.decode([int(t)]).strip().lower()
                           for t in top)
    print('P2792 margin_emb %s' % json.dumps(
        {w: round(margin_emb[w], 2) for w in WORDS}), flush=True)

    hooks = []

    def make_hook(p):
        def hook(module, inp, outp):
            t = outp
            assert isinstance(t, torch.Tensor), type(t)
            t = t.clone()
            t[0, p, :] = 0.0
            return t
        return hook

    def forward(ids, zero_layers, p):
        hs_list = []
        for l in zero_layers:
            hs_list.append(
                model.model.layers[l].mlp.register_forward_hook(
                    make_hook(p)))
        try:
            with torch.inference_mode():
                o = model(torch.tensor([ids], device=device),
                          output_hidden_states=True)
        finally:
            for h in hs_list:
                h.remove()
        return o.hidden_states

    occs = []
    for i in range(len(WORDS)):
        for j in range(len(WORDS)):
            if j <= i:
                continue
            A, B = WORDS[i], WORDS[j]
            occs.append((A, 'Unlike the', ' ' + A,
                         ', the ' + B + ' was seen at the market '
                         'yesterday'))
            occs.append((B, 'Unlike the', ' ' + B,
                         ', the ' + A + ' was seen at the market '
                         'yesterday'))
            occs.append((A, 'Unlike the ' + A + ', the', ' ' + A,
                         ' was seen at the market yesterday'))
            occs.append((B, 'Unlike the ' + B + ', the', ' ' + B,
                         ' was seen at the market yesterday'))
    for w in WORDS:
        occs.append((w, 'The', ' ' + w,
                     ' is loved by children everywhere'))
        occs.append((w, 'Everyone remembers the', ' ' + w, ''))
    assert len(occs) == 72

    # ---------- Arm A: full-layer purify curve ----------
    mg = np.zeros((37, 6))   # margin mean per layer per word
    sg = np.zeros((37, 6))   # hard-sort margin
    words_a = []
    for k, (w, pre, span, suf) in enumerate(occs):
        ids = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(span, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        p = len(tok(pre, add_special_tokens=False)['input_ids'])
        hs = forward(ids, [], p)
        wi = WORDS.index(w)
        for l in range(37):
            z = lens_np(hs[l][0, p].float().cpu().numpy())
            m, s = margins_z(z, w)
            mg[l, wi] += m
            sg[l, wi] += s
        words_a.append(w)
        if k % 24 == 0:
            print('P2792 A occ %d/72' % k, flush=True)
    words_a = np.array(words_a)
    cnt = np.array([np.sum(words_a == w) for w in WORDS])
    mg /= cnt[None, :]
    sg /= cnt[None, :]

    gm = mg.mean(axis=1)
    gs = sg.mean(axis=1)
    l_clean = None
    for l in range(37):
        if gs[l] > 0 and np.all(gs[l:] > 0):
            l_clean = l
            break
    p_f = bool(l_clean is not None and l_clean <= 20)
    print('P2792 l_clean=%s gs[L0..L35]=%s'
          % (l_clean, np.round(gs[::4], 2).tolist()), flush=True)
    print('P2792 P-F sort_before_readable=%s' % p_f, flush=True)

    # ---------- Arm B: per-layer early zeroing ----------
    bsents = []
    for i, w in enumerate(WORDS):
        nxt = WORDS[(i + 1) % 6]
        bsents.append((w, 'The', ' ' + w, '', 1, 'neutral'))
        bsents.append((w, 'Unlike the', ' ' + w,
                       ', the ' + nxt + ' was seen at the market '
                       'yesterday', 2, 'swap_late'))
    assert len(bsents) == 12

    dropB = {l: [] for l in EARLY_RANGE}
    decB = {l: [] for l in EARLY_RANGE}   # rival-vs-target decision
    for (w, pre, span, suf, p, tag) in bsents:
        ids = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(span, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        hs = forward(ids, [], p)
        z0 = lens_np(hs[L_STAR][0, p].float().cpu().numpy())
        m0, _ = margins_z(z0, w)
        zt0 = float(np.mean([z0[tid(pr)] for pr in PROPS[w]]))
        zr0 = float(np.mean([z0[tid(r)] for r in RIVAL[w]]))
        for l in EARLY_RANGE:
            hs1 = forward(ids, [l], p)
            z1 = lens_np(hs1[L_STAR][0, p].float().cpu().numpy())
            m1, _ = margins_z(z1, w)
            zt1 = float(np.mean([z1[tid(pr)] for pr in PROPS[w]]))
            zr1 = float(np.mean([z1[tid(r)] for r in RIVAL[w]]))
            dropB[l].append(m0 - m1)
            decB[l].append(1.0 if abs(zr1 - zr0) > abs(zt1 - zt0)
                           else 0.0)
        print('P2792 B %s(%s) m0=%.2f' % (w, tag, m0), flush=True)

    drop_mean = {l: float(np.mean(dropB[l])) for l in EARLY_RANGE}
    l_crit = max(drop_mean, key=drop_mean.get)
    max_drop = drop_mean[l_crit]
    p_g = bool(max_drop >= 2.0)
    print('P2792 l_crit=%d max_drop=%.3f P-G=%s'
          % (l_crit, max_drop, p_g), flush=True)

    # ---------- Arm C: prior vs readout content overlap ----------
    M_lstar = {w: [] for w in WORDS}
    for k, (w, pre, span, suf) in enumerate(occs):
        ids = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(span, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        p = len(tok(pre, add_special_tokens=False)['input_ids'])
        hs = forward(ids, [], p)
        M_lstar[w].append(
            hs[L_STAR][0, p].float().cpu().numpy().copy())
    same_source = {}
    for w in WORDS:
        hm = np.mean(np.stack(M_lstar[w]), axis=0)
        z = lens_np(hm)
        top = np.argsort(-z)[:50]
        top50_l = set(tok.decode([int(t)]).strip().lower()
                      for t in top)
        prop_hits_emb = [pr for pr in PROPS[w]
                         if any(pr.lower() in t
                                for t in top50_emb[w])]
        prop_hits_l = [pr for pr in PROPS[w]
                       if any(pr.lower() in t for t in top50_l)]
        common_props = [pr for pr in PROPS[w]
                        if pr in prop_hits_emb and pr in prop_hits_l]
        ov = len(top50_emb[w] & top50_l)
        same_source[w] = {'overlap_count': ov,
                          'prop_hits_emb': prop_hits_emb,
                          'prop_hits_lstar': prop_hits_l,
                          'props_in_both': common_props}
    n_h = sum(1 for w in WORDS if same_source[w]['props_in_both'])
    p_h = bool(n_h >= 5)
    print('P2792 P-H same_source=%s (%d/6)' % (p_h, n_h), flush=True)

    verdict = {
        'sort_before_readable': p_f, 'l_clean': l_clean,
        'early_separation_localized': p_g,
        'l_crit': int(l_crit), 'max_early_drop': max_drop,
        'prior_readout_same_source': p_h, 'n_same_source': n_h,
        'margin_emb': margin_emb, 'sort_emb': sort_emb,
        'early_drop_curve': drop_mean,
        'early_rival_dom_frac': {l: float(np.mean(decB[l]))
                                 for l in EARLY_RANGE},
        'same_source_detail': same_source,
    }
    result = {'phase': 2792, 'prereg': PREREG, 'verdict': verdict,
              'margin_curve': {'L%d' % l: {w: float(mg[l, i])
                                           for i, w in
                                           enumerate(WORDS)}
                               for l in range(37)},
              'sort_curve': {'L%d' % l: {w: float(sg[l, i])
                                         for i, w in
                                         enumerate(WORDS)}
                             for l in range(37)},
              'multi_token_words': multi}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'purify_curve.npz',
           layers=np.arange(37, dtype=np.int64),
           margin=mg, sort_margin=sg,
           words=np.array(WORDS, dtype=np.str_))
    print('P2792 VERDICT %s' % json.dumps(
        {k: v for k, v in verdict.items()
         if k not in ('same_source_detail',)}), flush=True)


if __name__ == '__main__':
    main()
