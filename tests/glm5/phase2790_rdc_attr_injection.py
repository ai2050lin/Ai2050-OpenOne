"""Phase 2790 (LPF-3): attribute injection localization.

2789 established the in-situ dwelling code (within-cos 0.961) and
found property emergence at L24, peak at L31, dip at L35 (5-layer
sampling).  Question now: WHERE does the property component come
from, and is it separable from the word-identity component?

Arms:
  A  full-layer (0..36) lens margin curve at the word position over
     the frozen 2788/2789 panel (72 occurrences, 6 words).
     margin(w, prop) = z(prop) - mean(z(rival props of other 5 words)).
     l* = argmax over prereg range L20..L35 of global mean margin.
  B  MLP-zero ablation at the WORD POSITION ONLY, one layer at a
     time, l = 0..l*-1, on 12 sentences (per word: neutral "The w"
     + one swap_late sentence with w as first noun, pos=2).
     Readout: lens margin at l* (word position) + word-token logit
     (specificity control).
  C  identity/property separability: per word, PCA of 12 in-situ
     residuals at l_c (saved layer nearest l*); remove PC1; lens
     top-50 property coverage (cf. 2789 coverage without removal).

Prereg (frozen before any forward):
  P1  attr_injection_localized iff mean margin drop over prereg
      window L20..L29 >= 2.0 AND mean drop over early controls
      L4..L17 < 1.0.
  P2  attr_identity_separable iff >= 4/6 words keep >= 1 prereg
      property token in lens top-50 after PC1 removal at l_c.
  D   descriptive: per-layer drop curve; specificity ratio
      (word-logit drop / margin drop over window); early-layer
      two-wave signature (L8..L14 drop > 1.0 noted if present).
verdict: localized iff P1; separable iff P2 (independent).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2790' / 'qwen4_attr_injection'

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
SAVE_LAYERS = [22, 24, 26, 28, 30, 31, 32, 35]
WINDOW = list(range(20, 30))
EARLY = list(range(4, 18))

PREREG = {
    'A': 'full-layer lens margin curve at word position over the '
         'frozen 72-occurrence panel; margin = z(prop) - mean(z('
         'rival props)); l* = argmax over L20..L35 of global mean',
    'P1': 'attr_injection_localized iff mean margin drop over '
          'window L20..L29 >= 2.0 AND mean drop over early '
          'controls L4..L17 < 1.0 (MLP zeroed at word position '
          'only, readout at l*)',
    'P2': 'attr_identity_separable iff >= 4/6 words keep >= 1 '
          'property in lens top-50 after per-word PC1 removal at '
          'l_c (nearest saved layer to l*)',
    'verdict': 'localized iff P1; separable iff P2 (independent)',
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
                 'window': WINDOW, 'early': EARLY}
    fc.save(OUT / 'execution.json', execution)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok2 = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    final_norm = model.model.norm
    n_layers = len(model.model.layers)
    assert n_layers == 36

    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            tc[t] = int(ids[0])
        return tc[t]

    def lens(h):
        with torch.inference_mode():
            hn = final_norm(
                torch.tensor(h, device=device).unsqueeze(0))
            return W_U @ hn[0].float().cpu().numpy()

    # ---------- sentence panel (same construction as 2789) ----------
    occs = []
    for i in range(len(WORDS)):
        for j in range(len(WORDS)):
            if j <= i:
                continue
            A, B = WORDS[i], WORDS[j]
            occs.append((A, 'Unlike the', ' ' + A,
                         ', the ' + B + ' was seen at the market '
                         'yesterday', 'swap_late'))
            occs.append((B, 'Unlike the', ' ' + B,
                         ', the ' + A + ' was seen at the market '
                         'yesterday', 'swap_late'))
            occs.append((A, 'Unlike the ' + A + ', the', ' ' + A,
                         ' was seen at the market yesterday',
                         'swap_early'))
            occs.append((B, 'Unlike the ' + B + ', the', ' ' + B,
                         ' was seen at the market yesterday',
                         'swap_early'))
    for w in WORDS:
        occs.append((w, 'The', ' ' + w,
                     ' is loved by children everywhere', 'subject'))
        occs.append((w, 'Everyone remembers the', ' ' + w, '',
                     'final'))
    assert len(occs) == 72

    ids_l, pos_l = [], []
    for (w, pre, span, suf, tag) in occs:
        ids = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(span, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        ids_l.append(ids)
        pos_l.append(len(tok(pre,
                             add_special_tokens=False)['input_ids']))
    words = np.array([o[0] for o in occs])
    pos = np.array(pos_l)

    def forward_hidden(ids, want_all=False):
        with torch.inference_mode():
            o = model(torch.tensor([ids], device=device),
                      output_hidden_states=True)
        hs = o.hidden_states
        if want_all:
            return [hs[l][0].float().cpu().numpy().copy()
                    for l in range(len(hs))]
        return [hs[l][0, pos_l[len(ids_l.index(ids))] if False else 0]
                for l in []] or hs

    # ---------- Arm A: full-layer margin curve ----------
    H_save = {l: [] for l in SAVE_LAYERS}
    margin_curve = {l: {} for l in range(37)}
    lens_z = {l: {} for l in range(37)}   # per-word mean lens margin
    for k in range(len(occs)):
        ids = ids_l[k]
        p = pos_l[k]
        with torch.inference_mode():
            o = model(torch.tensor([ids], device=device),
                      output_hidden_states=True)
        hs = o.hidden_states
        w = words[k]
        for l in range(37):
            z = lens(hs[l][0, p].float().cpu().numpy())
            zp = np.mean([z[tid(pr)] for pr in PROPS[w]])
            zr = np.mean([z[tid(r)] for r in RIVAL[w]])
            margin_curve[l].setdefault(w, []).append(float(zp - zr))
        for l in SAVE_LAYERS:
            H_save[l].append(hs[l][0, p].float().cpu().numpy().copy())
        if k % 24 == 0:
            print('P2790 A occ %d/72' % k, flush=True)

    for l in range(37):
        for w in WORDS:
            lens_z[l][w] = float(np.mean(margin_curve[l][w]))
    glob = [np.mean([lens_z[l][w] for w in WORDS])
            for l in range(20, 36)]
    l_star = 20 + int(np.argmax(glob))
    print('P2790 l*=%d (global margin %.3f)'
          % (l_star, glob[l_star - 20]), flush=True)

    # ---------- Arm B: word-position MLP-zero ablation ----------
    # 12 sentences: per word neutral + swap_late with w as first noun
    bsents = []
    for i, w in enumerate(WORDS):
        nxt = WORDS[(i + 1) % 6]
        bsents.append((w, 'The', ' ' + w, '', 1, 'neutral'))
        bsents.append((w, 'Unlike the', ' ' + w,
                       ', the ' + nxt + ' was seen at the market '
                       'yesterday', 2, 'swap_late'))
    assert len(bsents) == 12

    def margin_at(h, w):
        z = lens(h)
        zp = np.mean([z[tid(pr)] for pr in PROPS[w]])
        zr = np.mean([z[tid(r)] for r in RIVAL[w]])
        return float(zp - zr), float(z[tid(w)])

    hooks = []

    def make_hook(p):
        def hook(module, inp, outp):
            t = outp
            assert isinstance(t, torch.Tensor), type(t)
            t = t.clone()
            t[0, p, :] = 0.0
            return t
        return hook

    def run_ablated(ids, p, l):
        h = model.model.layers[l].mlp.register_forward_hook(
            make_hook(p))
        hooks.append(h)
        try:
            with torch.inference_mode():
                o = model(torch.tensor([ids], device=device),
                          output_hidden_states=True)
        finally:
            for hh in hooks:
                hh.remove()
            hooks.clear()
        return o.hidden_states[l_star][0, p].float().cpu().numpy()

    drop = {l: {} for l in range(0, 36)}     # margin drop per layer
    wdrop = {l: {} for l in range(0, 36)}    # word-logit drop
    for (w, pre, span, suf, p, tag) in bsents:
        ids = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(span, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        with torch.inference_mode():
            o = model(torch.tensor([ids], device=device),
                      output_hidden_states=True)
        h0 = o.hidden_states[l_star][0, p].float().cpu().numpy()
        m0, logit0 = margin_at(h0, w)
        for l in range(0, l_star):
            h1 = run_ablated(ids, p, l)
            m1, logit1 = margin_at(h1, w)
            drop[l].setdefault(w, []).append(m0 - m1)
            wdrop[l].setdefault(w, []).append(logit0 - logit1)
        print('P2790 B %s(%s) m0=%.2f done' % (w, tag, m0), flush=True)

    drop_mean = {l: float(np.mean([v for w in WORDS
                                   for v in drop[l][w]]))
                 for l in range(0, l_star)}
    wdrop_mean = {l: float(np.mean([v for w in WORDS
                                    for v in wdrop[l][w]]))
                  for l in range(0, l_star)}
    window_drop = float(np.mean([drop_mean[l] for l in WINDOW
                                 if l < l_star]))
    early_drop = float(np.mean([drop_mean[l] for l in EARLY]))
    spec_ratio = float(
        np.mean([wdrop_mean[l] for l in WINDOW if l < l_star])
        / max(window_drop, 1e-9))
    p1 = bool(window_drop >= 2.0 and early_drop < 1.0)
    print('P2790 window_drop=%.3f early_drop=%.3f spec=%.2f P1=%s'
          % (window_drop, early_drop, spec_ratio, p1), flush=True)

    # ---------- Arm C: PC1 removal separability ----------
    l_c = min(SAVE_LAYERS, key=lambda l: abs(l - l_star))

    def unit(x):
        return x / max(np.linalg.norm(x), 1e-9)

    def coverage_top50(z, w):
        top = np.argsort(-z)[:50]
        toks = [tok.decode([int(t)]).strip().lower() for t in top]
        return [pr for pr in PROPS[w]
                if any(pr.lower() in t for t in toks)]

    cov_keep, cov_after = {}, {}
    for l in SAVE_LAYERS:
        M = np.stack(H_save[l])                  # (72, d)
        cov_keep[l], cov_after[l] = {}, {}
        for w in WORDS:
            idx = np.where(words == w)[0]
            Hw = np.stack([H_save[l][k] for k in idx])  # (12, d)
            U, S, Vh = np.linalg.svd(Hw - Hw.mean(0, keepdims=True),
                                     full_matrices=False)
            pc1 = Vh[0]
            hm = Hw.mean(0)
            z0 = lens(hm)
            cov_keep[l][w] = coverage_top50(z0, w)
            resid = hm - float(hm @ pc1) * pc1
            z1 = lens(resid)
            cov_after[l][w] = coverage_top50(z1, w)
    n_keep = sum(1 for w in WORDS if cov_keep[l_c][w])
    n_after = sum(1 for w in WORDS if cov_after[l_c][w])
    p2 = bool(n_after >= 4)
    print('P2790 l_c=%d keep=%d/6 after_PC1=%d/6 P2=%s'
          % (l_c, n_keep, n_after, p2), flush=True)

    verdict = {
        'attr_injection_localized': p1,
        'attr_identity_separable': p2,
        'l_star': int(l_star), 'l_c': int(l_c),
        'window_drop': window_drop, 'early_drop': early_drop,
        'specificity_ratio': spec_ratio,
        'n_covered_keep': n_keep, 'n_covered_after_pc1': n_after,
        'global_margin_by_layer': {str(l): float(g) for l, g
                                   in zip(range(20, 36), glob)},
        'early_wave_note': {str(l): drop_mean.get(l)
                            for l in range(8, 15)},
    }
    result = {'phase': 2790, 'prereg': PREREG, 'verdict': verdict,
              'drop_curve': {str(l): drop_mean[l]
                             for l in range(0, l_star)},
              'word_logit_drop_curve': {str(l): wdrop_mean[l]
                                        for l in range(0, l_star)},
              'lens_z': {str(l): lens_z[l] for l in range(37)},
              'coverage_keep': {str(l): cov_keep[l]
                                for l in SAVE_LAYERS},
              'coverage_after': {str(l): cov_after[l]
                                 for l in SAVE_LAYERS},
              'multi_token_props': [t for t in tc
                                    if len(tok(
                                        ' ' + t,
                                        add_special_tokens=False
                                    )['input_ids']) != 1],
              }
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'ablation_curve.npz',
           layers=np.array(sorted(drop_mean.keys()),
                           dtype=np.int64),
           drop=np.array([drop_mean[l]
                          for l in sorted(drop_mean.keys())],
                         dtype=np.float64),
           words=words.astype(np.str_))
    print('P2790 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
