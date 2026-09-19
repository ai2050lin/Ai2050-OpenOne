"""Phase 2771 (Gamma): framework-subspace projection, category-knowledge
manifold geometry, cross-model two-cluster replication (2768/2769 follow-up).

Background.  2768: knowledge = 1.6-2.9x amplitude modulation on shared
frame/lexical swap coordinates; top-|dh| candidates are a frame/lexical
machine.  Open questions (Gamma plan): after projecting out the pseudo-word
frame subspace, do knowledge-specific residual coordinates exist?  What is
the full category-knowledge manifold geometry (5 categories + colour + size)?
When does the category word become lens-readable?  Does the two-cluster
polar geometry replicate in qwen3-1.7b?

Design (frozen before any forward):
  Material: exact 2768 v2 panel rebuild (same builder).  Measurement at the
  last prompt position, all 37 hidden states.
  C001 frame projection:
    - pseudo dh at L28 (80 pairs) -> SVD basis F, k = smallest k with 90%
      energy, capped at 16.
    - projected real residual r = dh_real - F F^T dh_real.
    - residual candidates per category: top-32 |r|, stable >= 60% of 16
      pairs; R1: >= 2/5 categories have >= 3 stable residual candidates.
    - projected pooled real nd curve; R2: median(L24..35)/max(L0..19) > 1.
  C002 manifold geometry (7 directions at L28):
    - delta_c = mean real dh over the 16 pairs per category; delta_color /
      delta_size = mean att_real dh.
    - 7x7 cosine matrix; descriptive cluster structure (does colour/size
      attach to the biological or the physical cluster, or separate?).
    - R3 (confirmatory of 2767/2768 with single-wrong-class data): intra-
      biological mean cosine > mean bio-vs-physical cosine of delta
      directions.
  C003 lens readability trajectory:
    - z_l = N(h_l) @ W_U.T; 5-way margin of the true category token vs the
      other 4 category tokens at the predicate-predicting position.
    - readable layer = first l >= 8 with margin > 0; R4: >= 4/5 categories
      have a readable layer for >= 12/16 entities.
  C004 cross-model: qwen3-1.7b (eager, BF16), same builder, proportional
    layer 21; R5: mean intra-biological cosine > mean bio-vs-physical
    cosine of category mean states.
  verdict: manifold_structured iff R2 and R3 (projection/readability are
  reported descriptively).
Status: descriptive + preregistered; NOT mechanism closure.
"""
import time
from pathlib import Path

import numpy as np

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2771' / 'qwen4_manifold_gamma'
PRIMARY = 28
TOPK = 32
STABILITY = 0.6

PREREG = {
    'phase': 2771,
    'question': 'After removing the pseudo-frame subspace, do knowledge-'
                'specific residual coordinates exist; what is the category '
                'knowledge manifold geometry; when is the category word '
                'lens-readable; does the two-cluster geometry replicate in '
                'qwen3-1.7b?',
    'projection': 'SVD basis of 80 pseudo dh at L28, 90% energy cap 16; '
                  'residual = real dh - F F^T dh',
    'criteria': {'R1': '>= 2/5 categories have >= 3 stable residual '
                       'candidates at L28 (descriptive-weak)',
                 'R2': 'projected real nd median(L24..35)/max(L0..19) > 1',
                 'R3': 'intra-biological mean cosine > bio-vs-physical mean '
                       'cosine of delta directions at L28',
                 'R4': '>= 4/5 categories lens-readable (margin>0) for '
                       '>= 12/16 entities',
                 'R5': '1.7b category mean states: intra_bio mean > cross '
                       'mean at layer 21'},
    'verdict': 'manifold_structured iff R2 and R3',
    'frozen_before_any_forward': True,
}


def jaccard(a, b):
    s1, s2 = set(a.tolist()), set(b.tolist())
    if not s1 and not s2:
        return 0.0
    return len(s1 & s2) / max(len(s1 | s2), 1)


def main():
    t0 = time.time()
    cc.guard(0)
    assert not (OUT / 'result.json').exists(), 'immutable; delete before rerun'
    OUT.mkdir(parents=True, exist_ok=True)
    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG}
    fc.save(OUT / 'execution.json', execution)

    import torch
    from transformers import AutoTokenizer
    import phase2768_category_atlas_v2 as p2768
    from rdc_query_common import MODELS

    tok = AutoTokenizer.from_pretrained(
        ROOT / 'models/hf' / MODELS['qwen4'], local_files_only=True,
        trust_remote_code=True, use_fast=True)
    material = p2768.build_material(tok)
    prompts = material['prompts']
    n_prompts = len(prompts)
    CATEGORIES = p2768.CATEGORIES

    from phase2662_symmetric_mapping_contract import load_native
    model, tok2 = load_native('qwen4')
    model.eval()
    device = next(model.parameters()).device
    n_layers = 36
    d_model = model.config.hidden_size

    h_all = np.empty((n_prompts, n_layers + 1, d_model), dtype=np.float32)
    for pi, p in enumerate(prompts):
        with torch.inference_mode():
            t = torch.tensor([p['ids']], device=device)
            o = model(t, output_hidden_states=True)
            h_all[pi] = np.stack(
                [h[0, -1].float().cpu().numpy() for h in o.hidden_states])
        if pi % 100 == 0:
            print('P2771 CAP %d/%d' % (pi, n_prompts), flush=True)

    real_pairs = material['real_pairs']
    pseudo_pairs = material['pseudo_pairs']
    att_pairs = material['att_real_pairs']

    def dh(pairs, l):
        return (h_all[[p[1] for p in pairs], l].astype(np.float64)
                - h_all[[p[0] for p in pairs], l].astype(np.float64))

    # ---------- C001 frame projection --------------------------------------
    dh_pse28 = dh(pseudo_pairs, PRIMARY)          # (80, 2560)
    U, S, Vt = np.linalg.svd(dh_pse28, full_matrices=False)
    energy = np.cumsum(S ** 2) / (S ** 2).sum()
    k = int(np.searchsorted(energy, 0.9) + 1)
    k = min(k, 16)
    F = Vt[:k].T                                   # (2560, k)
    results_proj = {'k_frame': k, 'energy': float(energy[k - 1])}

    def cand_coords(res):
        top = np.argsort(-np.abs(res), axis=1)[:, :TOPK]
        count = np.zeros(res.shape[1], dtype=np.int64)
        for j in range(res.shape[0]):
            count[top[j]] += 1
        return np.where(count >= STABILITY * res.shape[0])[0]

    residual_cands = {}
    r68 = fc.read(BASE / 'phase2768' / 'qwen4_category_atlas_v2' /
                  'result.json')
    for c in CATEGORIES:
        rp = [p for p in real_pairs if p[2] == c]
        dhr = dh(rp, PRIMARY)
        res = dhr - (dhr @ F) @ F.T
        residual_cands[c] = cand_coords(res)
    r1_count = sum(1 for c in CATEGORIES if len(residual_cands[c]) >= 3)
    r1_pass = bool(r1_count >= 2)

    # projected pooled real nd curve
    nd_proj = np.empty(37)
    nd_raw = np.empty(37)
    for l in range(37):
        dhr = dh(real_pairs, l)
        ha = h_all[[p[0] for p in real_pairs], l].astype(np.float64)
        hb = h_all[[p[1] for p in real_pairs], l].astype(np.float64)
        den = 0.5 * ((ha ** 2).sum(1) + (hb ** 2).sum(1)).mean()
        res = dhr - (dhr @ F) @ F.T
        nd_proj[l] = (res ** 2).sum(1).mean() / max(den, 1e-30)
        nd_raw[l] = (dhr ** 2).sum(1).mean() / max(den, 1e-30)
    r2_ratio = float(np.median(nd_proj[24:36]) / max(nd_proj[0:20].max(),
                                                     1e-30))
    r2_pass = bool(r2_ratio > 1.0)

    # ---------- C002 manifold geometry --------------------------------------
    deltas = {}
    for c in CATEGORIES:
        rp = [p for p in real_pairs if p[2] == c]
        deltas[c] = dh(rp, PRIMARY).mean(0)
    att_color = [p for p in att_pairs if p[2] == 'color']
    att_size = [p for p in att_pairs if p[2] == 'size']
    deltas['color'] = dh(att_color, PRIMARY).mean(0)
    deltas['size'] = dh(att_size, PRIMARY).mean(0)
    names = CATEGORIES + ['color', 'size']
    M = np.stack([deltas[n] / max(np.linalg.norm(deltas[n]), 1e-30)
                  for n in names])
    cos = M @ M.T
    bio = ['fruit', 'plant', 'animal']
    phys = ['solid', 'liquid']
    intra_bio = [float(cos[names.index(a), names.index(b)])
                 for a in bio for b in bio if a < b]
    cross = [float(cos[names.index(a), names.index(b)])
             for a in bio for b in phys]
    r3_pass = bool(np.mean(intra_bio) > np.mean(cross))
    color_attach = {n: float(cos[names.index('color'), names.index(n)])
                    for n in CATEGORIES}
    size_attach = {n: float(cos[names.index('size'), names.index(n)])
                   for n in CATEGORIES}

    # ---------- C003 lens readability ---------------------------------------
    nrm = model.model.norm
    W_lm = model.lm_head.weight.detach().float()
    pool_ids = material['pool_ids']
    readable = {}
    for c in CATEGORIES:
        tid = pool_ids[c]
        others = [pool_ids[o] for o in CATEGORIES if o != c]
        cnt = 0
        layers = []
        idxs = [pi for pi, p in enumerate(prompts)
                if p['kind'] == 'real_true' and p['meta']['cat'] == c]
        for pi in idxs:
            got = None
            with torch.inference_mode():
                t = torch.tensor([prompts[pi]['ids']], device=device)
                o = model(t, output_hidden_states=True)
                pos = list(prompts[pi]['ids']).index(tid)
                for l in range(8, 37):
                    n = nrm(o.hidden_states[l][0, pos - 1].float())
                    z = (n[None, :] @ W_lm.T)[0].float().cpu().numpy()
                    if z[tid] > max(z[o_] for o_ in others):
                        got = l
                        break
            if got is not None:
                cnt += 1
                layers.append(got)
        readable[c] = {'n_readable': cnt, 'n': len(idxs),
                       'layers_median': float(np.median(layers))
                       if layers else None}
    r4_pass = bool(sum(1 for c in CATEGORIES
                       if readable[c]['n_readable'] >= 12) >= 4)

    # ---------- C004 cross-model --------------------------------------------
    b5 = {}
    model2 = None
    try:
        from transformers import AutoModelForCausalLM
        model2 = AutoModelForCausalLM.from_pretrained(
            ROOT / 'models/hf/qwen3-1.7b', dtype=torch.bfloat16,
            device_map={'': 'cuda:0'}, attn_implementation='eager',
            local_files_only=True).eval()
        tok17 = AutoTokenizer.from_pretrained(
            ROOT / 'models/hf/qwen3-1.7b', local_files_only=True,
            trust_remote_code=True, use_fast=True)
        mat17 = p2768.build_material(tok17)
        pr17 = mat17['prompts']
        L2 = model2.config.num_hidden_layers
        prim17 = 21
        h17 = np.empty((len(pr17), L2 + 1, model2.config.hidden_size),
                       dtype=np.float32)
        with torch.inference_mode():
            for pi, p in enumerate(pr17):
                t = torch.tensor([p['ids']], device='cuda:0')
                o = model2(t, output_hidden_states=True)
                h17[pi] = np.stack([h[0, -1].float().cpu().numpy()
                                    for h in o.hidden_states])
        means17 = {}
        for c in CATEGORIES:
            idxs = [pi for pi, p in enumerate(pr17)
                    if p['kind'] == 'real_true' and p['meta']['cat'] == c]
            m = h17[idxs, prim17].astype(np.float64).mean(0)
            means17[c] = m / np.linalg.norm(m)
        names5 = CATEGORIES
        cos17 = np.array([[float(means17[a] @ means17[b]) for b in names5]
                          for a in names5])
        ib = [cos17[names5.index(a), names5.index(b)] for a in bio
              for b in bio if a < b]
        cr = [cos17[names5.index(a), names5.index(b)] for a in bio
              for b in phys]
        b5 = {'matrix': cos17.tolist(), 'intra_bio': ib, 'cross': cr,
              'intra_bio_mean': float(np.mean(ib)),
              'cross_mean': float(np.mean(cr))}
        r5_pass = bool(np.mean(ib) > np.mean(cr))
    finally:
        if model2 is not None:
            del model2
            torch.cuda.empty_cache()

    verdict = 'manifold_structured' if (r2_pass and r3_pass) else \
        'not_confirmed'
    results = {
        'phase': 2771,
        'C001': {'k_frame': k, 'residual_candidates':
                 {c: residual_cands[c].tolist() for c in CATEGORIES},
                 'R1_count': r1_count, 'R1_pass': r1_pass,
                 'nd_proj_curve': nd_proj.tolist(),
                 'nd_raw_curve': nd_raw.tolist(),
                 'R2_ratio': r2_ratio, 'R2_pass': r2_pass},
        'C002': {'cos_matrix_7x7': cos.tolist(), 'names': names,
                 'intra_bio': intra_bio, 'cross_bio_phys': cross,
                 'R3_pass': r3_pass, 'color_attach': color_attach,
                 'size_attach': size_attach},
        'C003': {'readable': readable, 'R4_pass': r4_pass},
        'C004': {'layer17': 21, **b5, 'R5_pass': r5_pass},
        'verdict': verdict, 'seconds': time.time() - t0}
    fc.save(OUT / 'result.json', results)
    fc.npz(OUT / 'manifold_stats.npz',
           cos7=cos, nd_proj=nd_proj, nd_raw=nd_raw,
           residual_fruit=residual_cands['fruit'],
           residual_plant=residual_cands['plant'],
           residual_animal=residual_cands['animal'],
           residual_solid=residual_cands['solid'],
           residual_liquid=residual_cands['liquid'])
    print('PHASE2771_DONE verdict=%s R1=%d R2=%s(%.3f) R3=%s R4=%s R5=%s' %
          (verdict, r1_count, r2_pass, r2_ratio, r3_pass, r4_pass, r5_pass),
          flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        (OUT / 'crash.txt').write_text(traceback.format_exc(),
                                       encoding='utf-8')
        raise
