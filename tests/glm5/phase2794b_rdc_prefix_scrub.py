"""Phase 2794b (LPF-7b): corrected prefix-content scrub.

2794 Arm B repeated the 2793 template mistake: the full_new
template placed {w} BEFORE {nxt} ("Compared to the apple, the dog
..."), so the content word was again post-target and d_pre=0 was
just the causal-immunity replay.  2794b puts the content word
strictly BEFORE the target, and additionally splits the long-prefix
modulation (-1.79 in 2794) into content vs length.

Conditions (6 words, word position readout at l*=30, frozen):
  pre_content  "Compared to the {nxt}, the {w} was seen at the
                market yesterday"   (content word strictly before)
  scrub_pre    "Compared to the item, the {w} ..." (scratched)
  long_pre     "In the garden behind the old wooden fence, the {w}
                was seen yesterday"
  long_scrub   "In the place behind the old wooden object, the {w}
                was seen yesterday"
  min          "The {w}"

Prereg (frozen before any forward):
  P-Q  prefix_content_modulation iff |mean(pre_content -
       scrub_pre)| >= 0.5
  P-R  long_prefix_content_driven iff |mean(long_pre - long_scrub)|
       >= 0.5
  D    descriptive: full condition table per word; sanity check
       that pre_content differs from scrub_pre in tokens before the
       target (assert pre-token-count equal, content differs).
verdict: P-Q and P-R independent.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2794b' / 'qwen4_prefix_scrub'

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

PREREG = {
    'P-Q': 'prefix_content_modulation iff |mean(pre_content - '
           'scrub_pre)| >= 0.5 (content word strictly before target)',
    'P-R': 'long_prefix_content_driven iff |mean(long_pre - '
           'long_scrub)| >= 0.5',
    'verdict': 'P-Q and P-R independent',
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

    conds = {}
    for i, w in enumerate(WORDS):
        nxt = WORDS[(i + 1) % 6]
        conds[w] = {
            'pre_content': ('Compared to the ' + nxt + ', the',
                            ' ' + w,
                            ' was seen at the market yesterday'),
            'scrub_pre': ('Compared to the item, the', ' ' + w,
                          ' was seen at the market yesterday'),
            'long_pre': ('In the garden behind the old wooden fence',
                         ' ' + w, ' was seen yesterday'),
            'long_scrub': ('In the place behind the old wooden '
                           'object', ' ' + w,
                           ' was seen yesterday'),
            'min': ('The', ' ' + w, ''),
        }

    # sanity: within each (pre_content, scrub_pre) pair the prefix
    # token count is identical (structure kept) and the prefix
    # content genuinely differs (scrub effective).
    for w in WORDS:
        pre_toks = []
        for tag, (pre, span, suf) in conds[w].items():
            if tag in ('pre_content', 'scrub_pre'):
                pre_toks.append(
                    tok(pre, add_special_tokens=False)['input_ids'])
        assert len(pre_toks[0]) == len(pre_toks[1]), \
            (w, len(pre_toks[0]), len(pre_toks[1]))
        assert pre_toks[0] != pre_toks[1], (w, 'scrub ineffective')
    print('P2794b sanity OK (pre lengths equal, contents differ)',
          flush=True)

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
        print('P2794b %s %s' % (w, {t: round(v, 3) for t, v
                                    in B[w].items()}), flush=True)

    pc = np.array([B[w]['pre_content'] for w in WORDS])
    sp = np.array([B[w]['scrub_pre'] for w in WORDS])
    lp = np.array([B[w]['long_pre'] for w in WORDS])
    ls = np.array([B[w]['long_scrub'] for w in WORDS])
    mn = np.array([B[w]['min'] for w in WORDS])
    d_q = float(np.mean(pc - sp))
    d_r = float(np.mean(lp - ls))
    d_long_min = float(np.mean(lp - mn))
    p_q = bool(abs(d_q) >= 0.5)
    p_r = bool(abs(d_r) >= 0.5)
    print('P2794b pre_content=%.3f scrub_pre=%.3f d_q=%.3f | '
          'long_pre=%.3f long_scrub=%.3f d_r=%.3f | '
          'min=%.3f d_long_min=%.3f P-Q=%s P-R=%s'
          % (pc.mean(), sp.mean(), d_q, lp.mean(), ls.mean(), d_r,
             mn.mean(), d_long_min, p_q, p_r), flush=True)

    verdict = {
        'prefix_content_modulation': p_q,
        'long_prefix_content_driven': p_r,
        'd_prefix_content': d_q, 'd_long_content': d_r,
        'd_long_vs_min': d_long_min,
        'margins': B,
    }
    result = {'phase': '2794b', 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'prefix_margins.npz',
           words=np.array(WORDS, dtype=np.str_),
           pre_content=pc, scrub_pre=sp, long_pre=lp,
           long_scrub=ls, min_m=mn)
    print('P2794b VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
