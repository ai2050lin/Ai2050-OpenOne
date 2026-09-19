"""Phase 2791 (LPF-4): L0 source decomposition + competition readout.

2790/2790b left two hard gaps: (iii) the L0-dominated single-layer
drop was not verified against the embedding itself, and (v) the car
margin RISE under C1 was seen in only 2 sentences.  This phase closes
both.

Arm A  L0 source split (all 72 occurrences, word position):
   margin_emb(w)   = lens of the raw token embedding (sentence-free)
   margin_h1(w)    = lens of hs[1] (= layer-0 full output)
   margin_h1_noMLP0(w) = same with word-position MLP[0] zeroed
   Split: prior P = margin_emb; total L0 gain G = h1 - emb;
   causal MLP0 share M = h1 - h1_noMLP0; residual R = h1_noMLP0 - emb.
Arm B  competition readout (12 sentences, 4 arms):
   none / C1 window L20-29 / L24 single / C3 early L4-17
   raw z saved for the 3 target props and 15 rivals at l*=30;
   per sentence-word: d_target, d_rival, d_margin decomposition.

Prereg (frozen before any forward):
  P-A  embedding_prior iff >= 5/6 words margin_emb >= 0.5
  P-B  L0_gain iff global mean(margin_h1 - margin_emb) >= 1.0
  P-C  mlp0_causal iff mean(margin_h1 - margin_h1_noMLP0) >= 0.5
       AND >= 5/6 words positive
  P-D  competition_dominant iff among sentence-word pairs with
       d_margin > +0.5 under C1, fraction with |d_rival| > |d_target|
       is >= 0.8
  P-E  rise_common iff >= 3/12 sentence-word pairs have
       d_margin > +0.5 under C1
verdict: L0_source_quantified iff P-A AND P-B; mlp0_causal iff P-C;
         competition story iff P-D AND P-E (independent families).
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2791' / 'qwen4_l0_competition'

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
L_STAR = 30          # frozen from phase 2790
ARMS_B = {
    'none': [],
    'C1_window': list(range(20, 30)),
    'L24_single': [24],
    'C3_early': list(range(4, 18)),
}

PREREG = {
    'P-A': 'embedding_prior iff >= 5/6 words margin_emb >= 0.5',
    'P-B': 'L0_gain iff global mean(margin_h1 - margin_emb) >= 1.0',
    'P-C': 'mlp0_causal iff mean(h1 - h1_noMLP0) >= 0.5 AND >= 5/6 '
           'words positive',
    'P-D': 'competition_dominant iff among d_margin > +0.5 pairs '
           'under C1, fraction with |d_rival| > |d_target| >= 0.8',
    'P-E': 'rise_common iff >= 3/12 pairs with d_margin > +0.5 '
           'under C1',
    'verdict': 'L0_source_quantified iff P-A AND P-B; mlp0_causal '
               'iff P-C; competition iff P-D AND P-E',
}


def main():
    import torch
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'words': WORDS, 'arms_b': ARMS_B,
                 'l_star': L_STAR}
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

    def margin_z(z, w):
        zp = np.mean([z[tid(pr)] for pr in PROPS[w]])
        zr = np.mean([z[tid(r)] for r in RIVAL[w]])
        return float(zp - zr)

    # ---------- Arm A: L0 source split ----------
    margin_emb = {}
    for w in WORDS:
        e = emb_table[tid(w)].numpy()
        margin_emb[w] = margin_z(lens_np(e), w)
    n_pa = sum(1 for w in WORDS if margin_emb[w] >= 0.5)
    p_a = bool(n_pa >= 5)
    print('P2791 margin_emb %s' % json.dumps(
        {w: round(margin_emb[w], 2) for w in WORDS}), flush=True)
    print('P2791 P-A embedding_prior=%s (%d/6)' % (p_a, n_pa),
          flush=True)

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

    h1_rows = []       # margin_h1 per occurrence
    h1n_rows = []      # margin_h1_noMLP0
    words_a = []
    for k, (w, pre, span, suf) in enumerate(occs):
        ids = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(span, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        p = len(tok(pre, add_special_tokens=False)['input_ids'])
        hs = forward(ids, [], p)
        z1 = lens_np(hs[1][0, p].float().cpu().numpy())
        hs2 = forward(ids, [0], p)
        z1n = lens_np(hs2[1][0, p].float().cpu().numpy())
        h1_rows.append(margin_z(z1, w))
        h1n_rows.append(margin_z(z1n, w))
        words_a.append(w)
        if k % 24 == 0:
            print('P2791 A occ %d/72' % k, flush=True)

    words_a = np.array(words_a)
    h1_mean = {w: float(np.mean(np.array(h1_rows)[words_a == w]))
               for w in WORDS}
    h1n_mean = {w: float(np.mean(np.array(h1n_rows)[words_a == w]))
                for w in WORDS}
    emb_arr = np.array([margin_emb[w] for w in WORDS])
    h1_arr = np.array([h1_mean[w] for w in WORDS])
    h1n_arr = np.array([h1n_mean[w] for w in WORDS])
    g_gain = float(np.mean(h1_arr - emb_arr))
    m_share = float(np.mean(h1_arr - h1n_arr))
    n_pos_m = int(np.sum((h1_arr - h1n_arr) > 0))
    p_b = bool(g_gain >= 1.0)
    p_c = bool(m_share >= 0.5 and n_pos_m >= 5)
    print('P2791 L0 split: emb=%.3f h1=%.3f h1_noMLP0=%.3f '
          'G=%.3f M=%.3f P-B=%s P-C=%s (%d/6 pos)'
          % (emb_arr.mean(), h1_arr.mean(), h1n_arr.mean(),
             g_gain, m_share, p_b, p_c, n_pos_m), flush=True)

    # ---------- Arm B: competition readout ----------
    bsents = []
    for i, w in enumerate(WORDS):
        nxt = WORDS[(i + 1) % 6]
        bsents.append((w, 'The', ' ' + w, '', 1, 'neutral'))
        bsents.append((w, 'Unlike the', ' ' + w,
                       ', the ' + nxt + ' was seen at the market '
                       'yesterday', 2, 'swap_late'))
    assert len(bsents) == 12

    comp = {}
    for (w, pre, span, suf, p, tag) in bsents:
        ids = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(span, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        row = {'word': w, 'tag': tag}
        for arm, lz in ARMS_B.items():
            hs = forward(ids, lz, p)
            z = lens_np(hs[L_STAR][0, p].float().cpu().numpy())
            zt = float(np.mean([z[tid(pr)] for pr in PROPS[w]]))
            zr = float(np.mean([z[tid(r)] for r in RIVAL[w]]))
            row[arm] = {'z_target': zt, 'z_rival': zr,
                        'margin': zt - zr}
        comp[w + '|' + tag] = row
        print('P2791 B %s(%s) margins %s'
              % (w, tag,
                 {a: round(row[a]['margin'], 2) for a in ARMS_B}),
              flush=True)

    pairs = []
    for k, row in comp.items():
        d_m = row['C1_window']['margin'] - row['none']['margin']
        d_t = row['C1_window']['z_target'] - row['none']['z_target']
        d_r = row['C1_window']['z_rival'] - row['none']['z_rival']
        pairs.append({'key': k, 'word': row['word'],
                      'd_margin': d_m, 'd_target': d_t,
                      'd_rival': d_r})
    rises = [q for q in pairs if q['d_margin'] > 0.5]
    p_e = bool(len(rises) >= 3)
    if rises:
        frac = float(np.mean([1.0 if abs(q['d_rival']) > abs(q['d_target'])
                              else 0.0 for q in rises]))
    else:
        frac = 0.0
    p_d = bool(frac >= 0.8)
    print('P2791 rises=%d/12 frac_rival_gt_target=%.2f P-D=%s P-E=%s'
          % (len(rises), frac, p_d, p_e), flush=True)

    verdict = {
        'embedding_prior': p_a, 'L0_gain': p_b, 'mlp0_causal': p_c,
        'competition_dominant': p_d, 'rise_common': p_e,
        'L0_source_quantified': bool(p_a and p_b),
        'margin_emb': margin_emb, 'margin_h1_mean': h1_mean,
        'margin_h1_noMLP0_mean': h1n_mean,
        'global': {'emb': float(emb_arr.mean()),
                   'h1': float(h1_arr.mean()),
                   'h1_noMLP0': float(h1n_arr.mean()),
                   'G_gain': g_gain, 'M_mlp0': m_share,
                   'n_pos_mlp0': n_pos_m},
        'rises': rises, 'frac_rival_gt_target': frac,
    }
    result = {'phase': 2791, 'prereg': PREREG, 'verdict': verdict,
              'competition_pairs': pairs,
              'multi_token_words': multi}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'l0_split.npz',
           words=np.array(WORDS, dtype=np.str_),
           margin_emb=emb_arr, margin_h1=h1_arr,
           margin_h1_noMLP0=h1n_arr)
    print('P2791 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
