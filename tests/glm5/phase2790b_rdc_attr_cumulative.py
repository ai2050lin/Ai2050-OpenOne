"""Phase 2790b (LPF-3b): cumulative MLP-zero ablation.

2790 found single-layer word-position MLP zeroing barely moves the
property margin (window drop 0.235 vs baseline ~8) -> property is
NOT single-layer injected.  If injection is distributed-cumulative
over the L20..L29 window, zeroing MANY layers at once must collapse
the margin.  2790b is that discriminating test.

Arms (12 sentences, word position only, readout at l*=30 lens):
  C1  zero MLP[20..29] together   (prereg window cumulative)
  C2  zero MLP[0..29] together    (everything before readout)
  C3  zero MLP[4..17] together    (early control cumulative)
  C4  zero MLP[30..35] together   (at/after readout layer)

Prereg (frozen before any forward):
  P1b  distributed_cumulative_injection iff C1 margin drop >= 4.0
       AND C1 word-logit drop < C1 margin drop (specificity).
  P2b  window_specificity_cumulative iff C3 drop < 0.5 * C1 drop.
  D    descriptive: full arm table incl. C2 collapse and C4.
verdict: P1b and P2b independent.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2790b' / 'qwen4_attr_cumulative'

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
L_STAR = 30   # frozen from phase 2790 (execution-locked)

ARMS = {
    'C1_window': list(range(20, 30)),
    'C2_all_pre': list(range(0, 30)),
    'C3_early': list(range(4, 18)),
    'C4_post': list(range(30, 36)),
}

PREREG = {
    'P1b': 'distributed_cumulative_injection iff C1 margin drop '
           '>= 4.0 AND C1 word-logit drop < C1 margin drop',
    'P2b': 'window_specificity_cumulative iff C3 drop < 0.5 * C1 '
           'drop',
    'readout': 'l*=30 frozen from phase 2790 result',
    'verdict': 'P1b and P2b independent',
}


def main():
    import torch
    from transformers import AutoTokenizer
    OUT.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'arms': ARMS, 'l_star': L_STAR}
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

    def lens(h):
        with torch.inference_mode():
            hn = final_norm(
                torch.tensor(h, device=device).unsqueeze(0))
            return W_U @ hn[0].float().cpu().numpy()

    def margin_at(h, w):
        z = lens(h)
        zp = np.mean([z[tid(pr)] for pr in PROPS[w]])
        zr = np.mean([z[tid(r)] for r in RIVAL[w]])
        return float(zp - zr), float(z[tid(w)])

    bsents = []
    for i, w in enumerate(WORDS):
        nxt = WORDS[(i + 1) % 6]
        bsents.append((w, 'The', ' ' + w, '', 1, 'neutral'))
        bsents.append((w, 'Unlike the', ' ' + w,
                       ', the ' + nxt + ' was seen at the market '
                       'yesterday', 2, 'swap_late'))
    assert len(bsents) == 12

    hooks = []

    def make_hook(p):
        def hook(module, inp, outp):
            t = outp
            assert isinstance(t, torch.Tensor), type(t)
            t = t.clone()
            t[0, p, :] = 0.0
            return t
        return hook

    def run(ids, p, layers_zero):
        hs_list = []
        for l in layers_zero:
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
        return o.hidden_states[L_STAR][0, p].float().cpu().numpy()

    res = {}
    base_rows = []
    for (w, pre, span, suf, p, tag) in bsents:
        ids = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(span, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        h0 = run(ids, p, [])
        m0, logit0 = margin_at(h0, w)
        base_rows.append({'word': w, 'tag': tag, 'm0': m0,
                          'logit0': logit0})
        row = {}
        for arm, lz in ARMS.items():
            h1 = run(ids, p, lz)
            m1, logit1 = margin_at(h1, w)
            row[arm] = {'margin_drop': m0 - m1,
                        'word_logit_drop': logit0 - logit1}
        res[w + '|' + tag] = row
        print('P2790b %s(%s) m0=%.2f %s'
              % (w, tag, m0,
                 ' '.join('%s:%.2f' % (a, row[a]['margin_drop'])
                          for a in ARMS)), flush=True)

    agg = {}
    for arm in ARMS:
        md = np.mean([res[k][arm]['margin_drop'] for k in res])
        wd = np.mean([res[k][arm]['word_logit_drop'] for k in res])
        agg[arm] = {'margin_drop': float(md),
                    'word_logit_drop': float(wd)}
        print('P2790b %s margin_drop=%.3f word_logit_drop=%.3f'
              % (arm, md, wd), flush=True)

    c1, c3 = agg['C1_window'], agg['C3_early']
    p1b = bool(c1['margin_drop'] >= 4.0 and
               c1['word_logit_drop'] < c1['margin_drop'])
    p2b = bool(c3['margin_drop'] < 0.5 * c1['margin_drop'])

    verdict = {
        'distributed_cumulative_injection': p1b,
        'window_specificity_cumulative': p2b,
        'arms': agg,
    }
    result = {'phase': '2790b', 'prereg': PREREG, 'verdict': verdict,
              'per_sentence': res, 'base_rows': base_rows}
    fc.save(OUT / 'result.json', result)
    print('P2790b VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
