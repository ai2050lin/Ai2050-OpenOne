"""Phase 2793 (LPF-6): prior decomposition + construction sourcing.

2792 downgraded "purify" to "construct": the embedding prior is huge
but only 3/6 words share readout content -> suspected metric-coupling
component.  Two closed questions:
  Arm A  decompose the prior: is margin_emb driven by a global
         metric coupling between E and W_U (background alignment)
         or by property-specific alignment (true semantics)?
         Pure weight-space computation, zero forwards.
  Arm B  source the construction: how much of the l*=30 property
         margin at the word position comes from sentence context
         (co-occurring content words) vs the word itself?
         Compare full panel sentence vs structure-kept scrubbed
         sentence (content words -> item/location/market->location)
         vs minimal "The {w}".

Prereg (frozen before any forward):
  P-I  semantic_component_significant iff attribute-vs-rival cos gap
       (RMSNorm directions) >= 3 x background std over random token
       pairs AND > 0.  Otherwise coupling_dominant.
  P-J  context_semantic iff mean(margin_full - margin_scrub) >= 1.0
  P-K  word_self_sufficient iff mean(margin_scrub - margin_min)
       >= 1.0 AND mean(margin_scrub / margin_full) >= 0.8
  D    descriptive: background cos distribution; per-word A/B tables.
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
OUT = BASE / 'phase2793' / 'qwen4_prior_context'

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
SEM_TOKENS = sorted({t for prs in PROPS.values() for t in prs} |
                    {t for v in RIVAL.values() for t in v})
L_STAR = 30
N_BG = 300
BG_SEED = 2793

PREREG = {
    'P-I': 'semantic_component_significant iff attr-vs-rival cos '
           'gap >= 3 x background std (random token pairs, L2 '
           'directions) AND > 0; else coupling_dominant',
    'P-J': 'context_semantic iff mean(margin_full - margin_scrub) '
           '>= 1.0 (structure kept, content words scrubbed)',
    'P-K': 'word_self_sufficient iff mean(scrub - min) >= 1.0 AND '
           'mean(scrub/full) >= 0.8',
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
                 'prereg': PREREG, 'words': WORDS,
                 'sem_tokens': SEM_TOKENS, 'n_bg': N_BG,
                 'bg_seed': BG_SEED, 'l_star': L_STAR}
    fc.save(OUT / 'execution.json', execution)

    from phase2662_symmetric_mapping_contract import load_native
    model, tok2 = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    final_norm = model.model.norm
    E = model.model.embed_tokens.weight.detach().float().cpu().numpy()
    vocab = E.shape[0]

    def unit(x):
        n = np.linalg.norm(x)
        return x / max(n, 1e-9)

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

    # ---------- Arm A: metric coupling decomposition ----------
    Wu_dirs = {t: unit(W_U[tid(t)]) for t in SEM_TOKENS}

    rng = np.random.default_rng(BG_SEED)
    banned = set(tc.values())
    bg_ids = []
    while len(bg_ids) < N_BG:
        i = int(rng.integers(0, vocab))
        if i not in banned:
            bg_ids.append(i)
            banned.add(i)
    U_dirs = np.stack([unit(E[i]) for i in bg_ids])   # (300, d)
    V_dirs = np.stack([unit(W_U[i]) for i in bg_ids])  # (300, d)
    bg_cos = U_dirs @ V_dirs.T   # random E-dir x random W_U-dir
    bg_abs_mean = float(np.mean(np.abs(bg_cos)))
    bg_std = float(np.std(bg_cos))

    attr_gap = {}
    for w in WORDS:
        u = unit(E[tid(w)])
        ca = float(np.mean([float(u @ Wu_dirs[pr]) for pr in PROPS[w]]))
        cr = float(np.mean([float(u @ Wu_dirs[r]) for r in RIVAL[w]]))
        attr_gap[w] = ca - cr
    gap_mean = float(np.mean(list(attr_gap.values())))
    p_i = bool(gap_mean >= 3 * bg_std and gap_mean > 0)
    print('P2793 A bg|cos|=%.4f std=%.4f attr_gap=%.4f '
          'per_word=%s P-I=%s'
          % (bg_abs_mean, bg_std, gap_mean,
             {w: round(v, 4) for w, v in attr_gap.items()}, p_i),
          flush=True)

    # ---------- Arm B: context scrubbing ----------
    bsents = []
    for i, w in enumerate(WORDS):
        nxt = WORDS[(i + 1) % 6]
        full = (('Unlike the', ' ' + w,
                 ', the ' + nxt + ' was seen at the market '
                 'yesterday'))
        scrub_pre, scrub_span, scrub_suf = (
            'Unlike the', ' ' + w,
            ', the item was seen at the location yesterday')
        bsents.append((w, full, 'full'))
        bsents.append((w, (scrub_pre, scrub_span, scrub_suf),
                       'scrub'))
        bsents.append((w, ('The', ' ' + w, ''), 'min'))
    assert len(bsents) == 18

    res = {}
    for (w, (pre, span, suf), tag) in bsents:
        ids = tok(pre, add_special_tokens=False)['input_ids'] + \
            tok(span, add_special_tokens=False)['input_ids'] + \
            tok(suf, add_special_tokens=False)['input_ids']
        p = len(tok(pre, add_special_tokens=False)['input_ids'])
        with torch.inference_mode():
            o = model(torch.tensor([ids], device=device),
                      output_hidden_states=True)
        z = lens_np(o.hidden_states[L_STAR][0, p].float().cpu()
                    .numpy())
        res.setdefault(w, {})[tag] = margin_z(z, w)
        print('P2793 B %s %s margin=%.2f' % (w, tag,
                                             res[w][tag]), flush=True)

    full = np.array([res[w]['full'] for w in WORDS])
    scrub = np.array([res[w]['scrub'] for w in WORDS])
    mini = np.array([res[w]['min'] for w in WORDS])
    d_ctx = float(np.mean(full - scrub))
    d_self = float(np.mean(scrub - mini))
    ratio = float(np.mean(scrub / np.maximum(full, 1e-9)))
    p_j = bool(d_ctx >= 1.0)
    p_k = bool(d_self >= 1.0 and ratio >= 0.8)
    print('P2793 full=%.3f scrub=%.3f min=%.3f d_ctx=%.3f '
          'd_self=%.3f ratio=%.2f P-J=%s P-K=%s'
          % (full.mean(), scrub.mean(), mini.mean(), d_ctx, d_self,
             ratio, p_j, p_k), flush=True)

    verdict = {
        'semantic_component_significant': p_i,
        'coupling_dominant': bool(not p_i),
        'context_semantic': p_j, 'word_self_sufficient': p_k,
        'bg_cos_abs_mean': bg_abs_mean, 'bg_cos_std': bg_std,
        'attr_gap': attr_gap, 'attr_gap_mean': gap_mean,
        'margins': {w: res[w] for w in WORDS},
        'd_context': d_ctx, 'd_self': d_self, 'scrub_full_ratio':
            ratio,
    }
    result = {'phase': 2793, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'prior_context.npz',
           words=np.array(WORDS, dtype=np.str_),
           margin_full=full, margin_scrub=scrub,
           margin_min=mini, attr_gap=np.array([attr_gap[w]
                                               for w in WORDS]))
    print('P2793 VERDICT %s' % json.dumps(verdict), flush=True)


if __name__ == '__main__':
    main()
