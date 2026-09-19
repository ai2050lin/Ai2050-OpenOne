"""Phase 2788 (LPF-1): content-packet position invariance.

USER-DIRECTED PARADIGM (2026-09-16): weights are FIXED but the same
word participates at ARBITRARY positions, yet stable regularities
emerge (apple is a plant, apple's color...).  The mechanism must be a
POSITION-INVARIANT content packet transported to the readout position
by attention.  Span-deletion v_row (established instrument) measures
exactly that packet: v_row = h(with word) - h(without word) at the
final position.

Design (generation-side: no wrong/right rows, no repair):
  Words (6, property-rich): apple, dog, gold, Japan, car, ocean.
  Swap frames: for each unordered pair {A,B} two sentences
    S1 "Unlike the {B}, the {A} was seen at the market yesterday"
    S2 "Unlike the {A}, the {B} was seen at the market yesterday"
    -> each word occurs early (pos ~3) and late (pos ~6), 10
    occurrences per word.
  Extras per word: subject-initial ("The {w} is loved by children
    everywhere") and sentence-final ("Everyone remembers the {w}")
    -> +2 occurrences, widening position range.
  42 sentences, 60 word-occurrences total.
  Layers scanned: {8, 16, 24, 31, 35} (hidden_states after layer l).

Prereg (frozen before any forward):
  G1   tokenization roundtrip on 6 spot checks; every occurrence has
       v_norm > 0 at every layer.
  P1   content_packet_stable iff at L35 the mean within-word
       cross-occurrence cos >= 0.5 AND within-word minus between-word
       mean cos >= 0.15.
  P2   emergence layer = smallest l in {8,16,24,31,35} with within
       mean cos >= 0.5 (descriptive curve recorded).
  P3   descriptive property readout: top-10 unembedding tokens of each
       word's mean L35 packet; a word counts as property-covered if
       any of its preregistered property tokens appears in top-50:
       apple {fruit, red, plant}, dog {animal, mammal, pet},
       gold {metal, yellow, precious}, Japan {country, japanese,
       tokyo}, car {vehicle, drive, wheels}, ocean {water, sea, salt}.
  D    descriptive: within-word cos vs absolute-position spread.
verdict: content_packet_stable iff P1.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2788' / 'qwen4_position_invariance'

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
    'G1': 'roundtrip spot checks + all occurrence v_norms > 0',
    'P1': 'content_packet_stable iff L35 within-word mean cos >= 0.5 '
          'AND (within - between) >= 0.15',
    'P2': 'emergence layer = smallest layer with within mean >= 0.5',
    'P3': 'descriptive property readout, coverage per PROPS in top-50',
    'verdict': 'content_packet_stable iff P1',
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

    # build occurrences: (word, pre, span, suf, tag)
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
    assert n_occ == 72, n_occ  # 15 pairs x 4 + 12 extras

    # tokenization + roundtrip spot checks
    ids_f_l, ids_w_l, pos_l = [], [], []
    for (w, pre, span, suf, tag) in occs:
        ids_f = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(span, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        ids_w = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        ids_f_l.append(ids_f)
        ids_w_l.append(ids_w)
        pos_l.append(len(tok(pre, add_special_tokens=False)['input_ids']))
    rt = [tok.decode(ids_f_l[k]) ==
          occs[k][1] + occs[k][2] + occs[k][3] for k in range(6)]
    assert all(rt), ('G1 roundtrip', rt)
    print('P2788 G1_OK roundtrip 6/6', flush=True)

    # forward with hidden states
    def hs_all(ids):
        with torch.inference_mode():
            o = model(torch.tensor([ids], device=device),
                      output_hidden_states=True)
        return [o.hidden_states[l][0, -1].float().cpu().numpy().copy()
                for l in LAYERS]

    V = {l: [] for l in LAYERS}
    norms = {l: [] for l in LAYERS}
    for k in range(n_occ):
        hf = hs_all(ids_f_l[k])
        hw = hs_all(ids_w_l[k])
        for li, l in enumerate(LAYERS):
            v = hf[li] - hw[li]
            n = float(np.linalg.norm(v))
            assert n > 0, ('G1 degenerate', k, l)
            norms[l].append(n)
            V[l].append(v / n)
        if k % 24 == 0:
            print('P2788 occ %d/%d' % (k, n_occ), flush=True)

    words = np.array([o[0] for o in occs])
    pos = np.array(pos_l)
    tags = np.array([o[4] for o in occs])

    # cos statistics per layer
    stats = {}
    for l in LAYERS:
        M = np.stack(V[l])                       # (72, d)
        C = M @ M.T
        wmask = (words[:, None] == words[None, :])
        iu = np.triu_indices(n_occ, k=1)
        within = [C[a, b] for a, b in zip(*iu) if wmask[a, b]]
        between = [C[a, b] for a, b in zip(*iu) if not wmask[a, b]]
        stats[l] = {'within_mean': float(np.mean(within)),
                    'between_mean': float(np.mean(between)),
                    'within_min': float(np.min(within)),
                    'n_within': len(within)}
        print('P2788 L%d within=%.4f between=%.4f (min %.3f)'
              % (l, stats[l]['within_mean'], stats[l]['between_mean'],
                 stats[l]['within_min']), flush=True)

    within35 = stats[35]['within_mean']
    gap35 = stats[35]['within_mean'] - stats[35]['between_mean']
    p1 = bool(within35 >= 0.5 and gap35 >= 0.15)
    emerge = next((l for l in LAYERS
                   if stats[l]['within_mean'] >= 0.5), None)

    # P3 property readout of mean packets at L35
    readout = {}
    covered = {}
    for w in WORDS:
        idx = np.where(words == w)[0]
        c = np.mean(np.stack([V[35][k] for k in idx]), axis=0)
        c = c / np.linalg.norm(c)
        z = W_U @ c
        top = np.argsort(-z)[:50]
        toks = [tok.decode([int(t)]).strip() for t in top]
        readout[w] = toks[:10]
        pl = PROPS[w]
        covered[w] = [p for p in pl if any(
            p.lower() == t.lower() or p.lower() in t.lower()
            for t in toks)]
        print('P2788 P3 %s covered=%s top5=%s'
              % (w, covered[w], toks[:5]), flush=True)
    n_covered = int(sum(1 for w in WORDS if covered[w]))

    # descriptive: within cos vs position spread (L35)
    M35 = np.stack(V[35])
    C35 = M35 @ M35.T
    iu = np.triu_indices(n_occ, k=1)
    pairs_within = [(float(C35[a, b]), int(abs(pos[a] - pos[b])),
                     str(tags[a]) + '/' + str(tags[b]))
                    for a, b in zip(*iu) if words[a] == words[b]]
    posd = np.array([p[1] for p in pairs_within])
    cosm = np.array([p[0] for p in pairs_within])
    r = float(np.corrcoef(posd, cosm)[0, 1])

    verdict = {
        'content_packet_stable': p1,
        'within35': within35, 'between35': stats[35]['between_mean'],
        'gap35': gap35,
        'emergence_layer': emerge,
        'layer_curve': {str(l): stats[l]['within_mean'] for l in LAYERS},
        'property_covered_words': n_covered,
        'covered': covered,
        'cos_pos_corr': r,
        'n_occ': n_occ,
    }
    result = {'phase': 2788, 'prereg': PREREG, 'verdict': verdict,
              'readout': readout,
              'stats': {str(l): stats[l] for l in LAYERS},
              'occurrences': [{'word': o[0], 'tag': o[4], 'pos': int(p)}
                              for o, p in zip(occs, pos)]}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'invariance_stats.npz',
           words=words.astype(np.str_), pos=pos.astype(np.int64),
           tags=tags.astype(np.str_),
           within35=np.array([p[0] for p in pairs_within]),
           posdiff=posd,
           norms35=np.array(norms[35]))
    print('P2788 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
