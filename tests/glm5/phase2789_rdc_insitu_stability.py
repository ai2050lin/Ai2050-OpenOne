"""Phase 2789 (LPF-2): in-situ residual encoding stability.

Motivated by 2788: the TRANSPORTED packet (span-deletion v_row at
final position) is position-DEPENDENT (within-cos 0.24, pos-corr
-0.81), yet within-word >> between-word at every layer.  The user's
hypothesis shifts to the word's OWN position: fixed weights + the word
at any position should leave a STABLE property encoding in the
residual stream AT THE WORD'S POSITION (MLP-injected), which attention
then transports selectively.

Design (same 42 frozen sentences / 72 occurrences as 2788):
  Measure h_l AT THE WORD'S OWN TOKEN POSITION for
  l in {8,16,24,31,35} in each sentence containing the word.
  Neutral baseline: "The {word}" (2 tokens) for each word.
  Readout: proper logit lens z = W_U @ RMSNorm(h) per layer.

Prereg (frozen before any forward):
  P1  in_situ_stable iff at L35 within-word mean cos (in-situ
      residuals across different sentences/positions) >= 0.5 AND
      within - between >= 0.15.
  P2  property_emergence_layer = smallest l where >= 4/6 words have
      >= 1 preregistered property token in top-50 of lens readout
      averaged over the word's occurrences.
  D   descriptive: layer curve; neutral-context cos; position corr.
verdict: in_situ_stable iff P1.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2789' / 'qwen4_insitu_stability'

LAYERS = [8, 16, 24, 31, 35]
WORDS = ['apple', 'dog', 'gold', 'Japan', 'car', 'ocean']
PROPS = {
    'apple': ['fruit', 'red', 'plant'],
    'dog': ['animal', 'mammal', 'pet'],
    'gold': ['metal', 'yellow', 'precious'],
    'Japan': ['country', 'Japanese', 'Tokyo'],
    'car': ['vehicle', 'drive', 'wheels'],
    'ocean': ['water', 'sea', 'salt'],
}

PREREG = {
    'P1': 'in_situ_stable iff L35 within-word in-situ mean cos >= 0.5 '
          'AND within - between >= 0.15',
    'P2': 'property_emergence_layer = smallest l with >= 4/6 words '
          'property-covered in lens top-50',
    'verdict': 'in_situ_stable iff P1',
}


def main():
    import torch
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'words': WORDS}
    fc.save(OUT / 'execution.json', execution)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok2 = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    final_norm = model.model.norm

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
    n_occ = len(occs)
    assert n_occ == 72

    ids_l, pos_l = [], []
    for (w, pre, span, suf, tag) in occs:
        ids = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(span, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        ids_l.append(ids)
        pos_l.append(len(tok(pre,
                             add_special_tokens=False)['input_ids']))
    # neutral baselines
    neu_ids = {w: tok('The ' + w,
                      add_special_tokens=False)['input_ids']
               for w in WORDS}
    neu_pos = {w: 1 for w in WORDS}

    def hs_all(ids):
        with torch.inference_mode():
            o = model(torch.tensor([ids], device=device),
                      output_hidden_states=True)
        return [o.hidden_states[l][0].float().cpu().numpy().copy()
                for l in LAYERS]

    # in-situ residuals at the word's own position
    H = {l: [] for l in LAYERS}
    for k in range(n_occ):
        hs = hs_all(ids_l[k])
        p = pos_l[k]
        for li, l in enumerate(LAYERS):
            H[l].append(hs[li][p])
        if k % 24 == 0:
            print('P2789 occ %d/%d' % (k, n_occ), flush=True)

    # neutral residuals (word position = 1)
    NEU = {l: {} for l in LAYERS}
    for w in WORDS:
        hs = hs_all(neu_ids[w])
        for li, l in enumerate(LAYERS):
            NEU[l][w] = hs[li][neu_pos[w]]

    words = np.array([o[0] for o in occs])
    pos = np.array(pos_l)

    def unit(x):
        return x / np.linalg.norm(x)

    stats = {}
    for l in LAYERS:
        M = np.stack([unit(h) for h in H[l]])
        C = M @ M.T
        wmask = (words[:, None] == words[None, :])
        iu = np.triu_indices(n_occ, k=1)
        within = [C[a, b] for a, b in zip(*iu) if wmask[a, b]]
        between = [C[a, b] for a, b in zip(*iu) if not wmask[a, b]]
        stats[l] = {'within_mean': float(np.mean(within)),
                    'between_mean': float(np.mean(between)),
                    'within_min': float(np.min(within))}
        print('P2789 L%d in-situ within=%.4f between=%.4f (min %.3f)'
              % (l, stats[l]['within_mean'],
                 stats[l]['between_mean'],
                 stats[l]['within_min']), flush=True)

    within35 = stats[35]['within_mean']
    gap35 = stats[35]['within_mean'] - stats[35]['between_mean']
    p1 = bool(within35 >= 0.5 and gap35 >= 0.15)

    # lens readout per word per layer (mean over occurrences, normed)
    def lens(h):
        with torch.inference_mode():
            hn = final_norm(torch.tensor(h, device=device).unsqueeze(0))
            z = (hn @ W_U.T if False else
                 W_U @ hn[0].float().cpu().numpy())
        return z

    coverage = {}
    readout = {}
    for l in LAYERS:
        cov_l = {}
        for w in WORDS:
            idx = np.where(words == w)[0]
            hm = np.mean(np.stack([H[l][k] for k in idx]), axis=0)
            z = lens(hm)
            top = np.argsort(-z)[:50]
            toks = [tok.decode([int(t)]).strip().lower() for t in top]
            hit = [p for p in PROPS[w]
                   if any(p.lower() == t or p.lower() in t
                          for t in toks)]
            cov_l[w] = hit
            if l == 35:
                readout[w] = [tok.decode([int(t)]).strip()
                              for t in np.argsort(-z)[:10]]
        coverage[l] = cov_l
        n_cov = sum(1 for w in WORDS if cov_l[w])
        print('P2789 L%d property-covered %d/6' % (l, n_cov),
              flush=True)

    emerge = next((l for l in LAYERS
                   if sum(1 for w in WORDS if coverage[l][w]) >= 4),
                  None)

    # neutral context cos (L35): sentence in-situ vs neutral
    neu_cos = {}
    for w in WORDS:
        idx = np.where(words == w)[0]
        cs = [float(unit(H[35][k]) @ unit(NEU[35][w])) for k in idx]
        neu_cos[w] = float(np.mean(cs))

    # position corr
    M35 = np.stack([unit(h) for h in H[35]])
    C35 = M35 @ M35.T
    iu = np.triu_indices(n_occ, k=1)
    pw = [(float(C35[a, b]), int(abs(int(pos[a]) - int(pos[b]))))
          for a, b in zip(*iu) if words[a] == words[b]]
    r = float(np.corrcoef([p[1] for p in pw], [p[0] for p in pw])[0, 1])

    verdict = {
        'in_situ_stable': p1,
        'within35': within35, 'between35': stats[35]['between_mean'],
        'gap35': gap35,
        'layer_curve': {str(l): stats[l]['within_mean'] for l in LAYERS},
        'property_emergence_layer': emerge,
        'coverage35': coverage[35],
        'n_covered35': sum(1 for w in WORDS if coverage[35][w]),
        'neutral_cos35': neu_cos,
        'cos_pos_corr': r,
    }
    result = {'phase': 2789, 'prereg': PREREG, 'verdict': verdict,
              'readout35': readout,
              'stats': {str(l): stats[l] for l in LAYERS}}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'insitu_stats.npz',
           words=words.astype(np.str_), pos=pos.astype(np.int64))
    print('P2789 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
