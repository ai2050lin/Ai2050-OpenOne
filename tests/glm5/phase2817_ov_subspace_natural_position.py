"""Phase 2817 (LPF-30): OV SUBSPACE ALIGNMENT + NATURAL-CONTEXT POSITION.

2816 left two registered limitations:
 (1) o_proj single-column test is a LOWER BOUND - multi-head combination
     writes could hide below the per-column max;
 (2) position test used only function-word fillers - natural-context
     position stability unknown.

Arm A (zero-forward, per-head OV column-space): for layer L head h,
  W_h = W_O[:, h*hd:(h+1)*hd] in R^(d_model x d_head); attention head h
  writes delta = W_h * a_h into the residual stream, so its reachable
  write subspace is Col(W_h).  Semantic content = how much of each
  class direction unitD_i (2807 protocol) lies inside Col(W_h):
      proj_len(L,h,i) = || P_Col(W_h) unitD_i ||
  computed via G = W_h^T W_h, c = W_h^T unitD_i,  proj^2 = c^T G^-1 c
  (0 <= proj <= 1; d_model-dim projection theorem).
  Null: 30 random unit vectors in R^2560 pushed through the SAME
  subspace (shape-matched; theory mean sqrt(d_head/d_model) = 0.224).
  P-A1 attn_ov_subspace_semantic: exists (L,h) with
       max_i proj_len >= 0.40 AND that head's random q95 < 0.40
  (0.40 ~= 1.8x theoretical random mean, above any 128-dim subspace
  q95 under full-rank null; frozen before readout.)
  Descriptive: best head's projected write direction read out through
  the unembedding (mechanism evidence, not a verdict).

Arm B (CUDA, qwen3-4b bf16): natural-context position stability.
  Same 8 single-token targets as 2816; 7 sentences per target with a
  FIXED tail '{w} is here' and natural growing prefixes so the
  target's first token sits at exact positions 0..6:
    0 '{w} is here'
    1 'the {w} is here'
    2 'and the {w} is here'
    3 'and then the {w} is here'
    4 'and then a small {w} is here'
    5 'and then a very small {w} is here'
    6 'and then a very small shiny {w} is here'
  Metrics identical to 2816 (class profile via unitD, full state,
  identity channel), read at the target's first token.
  P-P1n semantics_position_invariant_natural: mean_{L in 4..35}
       profile stability >= 0.80
  P-P3 position_effect_context_dependent: prof_stab_natural -
       prof_stab_functional(2816 result) >= 0.05

Prereg frozen before any readout; verdicts only from the P-* above.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import rdc_construction_common as cc
import rdc_feature_common as fc

ROOT, BASE = cc.ROOT, cc.BASE
OUT = BASE / 'phase2817' / 'ov_subspace_natural_position'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SRC_2816 = BASE / 'phase2816' / 'layers_semantics_position'
SEED = 2817
N_RAND_VEC = 30
THRESH_OV = 0.40
THRESH_POS = 0.80
THRESH_CTX = 0.05
LAYER_LO, LAYER_HI = 4, 35

TARGETS = ['apple', 'gold', 'eagle', 'sedan', 'Spain', 'soup', 'cave',
           'sock']
# natural prefix growth; first word sentence-initial (no space), the
# rest mid-sentence (leading space).  Target position = len(prefix).
PREFIX_WORDS = {
    0: [],
    1: ['the'],
    2: ['and', ' the'],
    3: ['and', ' then', ' the'],
    4: ['and', ' then', ' a', ' small'],
    5: ['and', ' then', ' a', ' very', ' small'],
    6: ['and', ' then', ' a', ' very', ' small', ' shiny'],
}
TAIL_WORDS = [' is', ' here']
N_POS = 7

PREREG = {
    'P-A1': 'attn_ov_subspace_semantic iff exists (L,head) with '
            'max_i ||P_Col(W_h) unitD_i|| >= 0.40 AND that head random '
            'q95 < 0.40 (30 shape-matched random unit vectors)',
    'P-P1n': 'semantics_position_invariant_natural iff mean_{L in '
             '4..35} profile stability (mean pairwise cos across 7 '
             'natural-prefix positions, 8 words) >= 0.80',
    'P-P3': 'position_effect_context_dependent iff prof_stab_natural '
            '- prof_stab_functional(2816) >= 0.05',
    'verdict': 'ov_subspace_semantic = P-A1; '
               'semantics_position_invariant_natural = P-P1n; '
               'position_effect_context_dependent = P-P3',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)

    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())
    res2811 = json.loads((SRC_2811 / 'result.json').read_text(
        encoding='utf-8'))
    res2816 = json.loads((SRC_2816 / 'result.json').read_text(
        encoding='utf-8'))
    prof_func_ref = float(
        res2816['verdict']['prof_stab_mean_4_35'])

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED,
                 'thresh_ov': THRESH_OV, 'thresh_pos': THRESH_POS,
                 'thresh_ctx': THRESH_CTX,
                 'layer_window': [LAYER_LO, LAYER_HI],
                 'targets': TARGETS,
                 'prefix_words': {str(k): v for k, v in
                                  PREFIX_WORDS.items()},
                 'tail_words': TAIL_WORDS,
                 'n_rand_vec': N_RAND_VEC,
                 'prof_stab_functional_ref_2816': prof_func_ref,
                 'note': 'Arm A zero-forward per-head OV column-space '
                         'projection (fixes 2816 limitation 1); Arm B '
                         'CUDA natural-prefix position stability '
                         '(fixes 2816 limitation 2): full-sentence BPE '
                         'encoding, target first-token positions 0..6 '
                         'exact, span-mean readout (1-2 token targets)'}
    fc.save(OUT / 'execution.json', execution)

    # ---------- tensors + gates ----------
    from safetensors import safe_open
    mdir = ROOT / 'models' / 'hf' / 'qwen3-4b'
    index = json.loads((mdir / 'model.safetensors.index.json')
                       .read_text(encoding='utf-8'))['weight_map']

    def read_tensor(name):
        with safe_open(str(mdir / index[name]), framework='pt') as f:
            return f.get_tensor(name).float().numpy()

    Etab = read_tensor('model.embed_tokens.weight')
    g = read_tensor('model.norm.weight').astype(np.float64)
    try:
        Wu = read_tensor('lm_head.weight')
        tie = False
    except KeyError:
        Wu = Etab
        tie = True
    cfg = json.loads((mdir / 'config.json').read_text(encoding='utf-8'))
    eps = float(cfg.get('rms_norm_eps', 1e-6))
    n_layers = int(cfg.get('num_hidden_layers', 36))
    n_heads = int(cfg.get('num_attention_heads', 32))
    # head_dim from the actual o_proj shape (config head_dim may be
    # explicit and differ from d_model // n_heads under GQA-style QK)
    Wo0 = read_tensor('model.layers.0.self_attn.o_proj.weight')
    d_model = Wo0.shape[0]
    hd = Wo0.shape[1] // n_heads
    assert Wo0.shape[1] % n_heads == 0 and Wo0.shape[0] == Etab.shape[1]
    cfg_hd = int(cfg.get('head_dim', 0) or 0)
    if cfg_hd:
        assert cfg_hd == hd, (cfg_hd, hd)
    del Wo0

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(mdir), local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tc = {}

    def tid(t):
        # exact replica of the 2811/2816 tokenizer rule: prefer the
        # space-prefixed form, fall back to the bare form
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, t
            tc[t] = int(ids[0])
        return tc[t]

    h7 = np.load(SRC_2807 / 'heldout.npz')
    b11 = np.load(SRC_2811 / 'battery.npz')
    atlas_words = [w for v in CATS.values() for w in v]
    cent = {c: np.stack([Wu[tid(w)].astype(np.float64)
                         for w in CATS[c]]).mean(0) for c in CAT_WORDS}
    Cm = np.stack([cent[c] for c in CAT_WORDS])
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    gate_dW = float(np.abs(dW - h7['dW_class'].astype(np.float64)).max())
    Z_eval = np.stack([(Etab[tid(w)].astype(np.float64)
                        / np.sqrt((Etab[tid(w)] ** 2).mean() + eps)
                        * g) for w in res2811['eval_words']])
    gate_Z = float(np.abs(Z_eval - b11['Z_eval'].astype(np.float64)).max())
    print('P2817 gates: dW=%.2e Zeval=%.2e n_layers=%d heads=%d hd=%d '
          'tie=%s' % (gate_dW, gate_Z, n_layers, n_heads, hd, tie),
          flush=True)
    assert gate_dW < 1e-6 and gate_Z < 1e-4
    assert Wu.shape == Etab.shape

    unitD = np.stack([unit(dW[i]) for i in range(10)])
    unitD_f = unitD.astype(np.float32)
    rng = np.random.default_rng(SEED)
    R = rng.standard_normal((Etab.shape[1], N_RAND_VEC)).astype(
        np.float32)
    R /= np.linalg.norm(R, axis=0, keepdims=True)

    # ---------- Arm A: per-head OV column-space projection ----------
    proj = np.zeros((n_layers, n_heads, 10), dtype=np.float32)
    rq95 = np.zeros((n_layers, n_heads), dtype=np.float32)
    eye_hd = np.eye(hd, dtype=np.float32)
    for L in range(n_layers):
        Wo = read_tensor('model.layers.%d.self_attn.o_proj.weight'
                         % L).astype(np.float32)
        assert Wo.shape == (Etab.shape[1], n_heads * hd), Wo.shape
        for h in range(n_heads):
            Wh = Wo[:, h * hd:(h + 1) * hd]
            G = Wh.T @ Wh
            jit = 1e-6 * float(np.trace(G)) / hd
            Gj = G + eye_hd * jit
            C = Wh.T @ unitD_f.T                     # (hd, 10)
            X = np.linalg.solve(Gj, C)
            p2 = np.maximum((C * X).sum(0), 0.0)
            proj[L, h] = np.sqrt(np.minimum(p2, 1.0))
            Cr = Wh.T @ R                           # (hd, 30)
            Xr = np.linalg.solve(Gj, Cr)
            pr2 = np.maximum((Cr * Xr).sum(0), 0.0)
            rq95[L, h] = float(np.quantile(
                np.sqrt(np.minimum(pr2, 1.0)), 0.95))
        if L % 6 == 0:
            print('P2817 Arm A layer %d max_proj=%.4f rq95_med=%.4f'
                  % (L, float(proj[L].max()), float(np.median(rq95[L]))),
                  flush=True)
        del Wo
    pass_mask = (proj.max(axis=2) >= THRESH_OV) & (rq95 < THRESH_OV)
    n_pass_heads = int(pass_mask.sum())
    p_a1 = n_pass_heads > 0
    bi, bh, bc = np.unravel_index(int(np.argmax(proj)),
                                  proj.shape)
    best_info = {'layer': int(bi), 'head': int(bh),
                 'class': CAT_WORDS[int(bc)],
                 'proj_len': round(float(proj[bi, bh, bc]), 4),
                 'rand_q95': round(float(rq95[bi, bh]), 4),
                 'head_proj_max': round(float(proj[bi, bh].max()), 4)}
    print('P2817 Arm A done: pass_heads=%d P-A1=%s best=%s'
          % (n_pass_heads, p_a1, json.dumps(best_info)), flush=True)

    # best head readout (descriptive): projected write direction
    Wo = read_tensor('model.layers.%d.self_attn.o_proj.weight'
                     % int(bi)).astype(np.float32)
    Wh = Wo[:, int(bh) * hd:(int(bh) + 1) * hd]
    Gj = Wh.T @ Wh + eye_hd * (1e-6 * float(np.trace(Wh.T @ Wh)) / hd)
    c = Wh.T @ unitD_f[int(bc)]
    what = Wh @ np.linalg.solve(Gj, c)
    what = unit(what.astype(np.float64))
    top_ids = np.argsort(-(Wu @ what))[:8].tolist()
    head_readout = {'info': best_info,
                    'top_unembed': [tok.decode([t]).strip()
                                    for t in top_ids]}
    del Wo, Wh
    print('P2817 best-head readout %s' % json.dumps(head_readout),
          flush=True)

    # ---------- Arm B: natural-context position (CUDA) ----------
    import torch
    from transformers import AutoModelForCausalLM
    for w in TARGETS:
        tid(w)
    tail_ids = tok(' is here', add_special_tokens=False)['input_ids']
    assert len(tail_ids) == 2, tail_ids
    seqs, meta = [], []
    for w in TARGETS:
        for p in range(N_POS):
            words = PREFIX_WORDS[p]
            prefix_str = ' '.join(t.strip() for t in words)
            pre_ids = (tok(prefix_str, add_special_tokens=False)
                       ['input_ids']) if prefix_str else []
            assert len(pre_ids) == p, (w, p, prefix_str, pre_ids)
            full_str = ((prefix_str + ' ' if prefix_str else '')
                        + w + ' is here')
            ids = tok(full_str, add_special_tokens=False)['input_ids']
            assert ids[:len(pre_ids)] == pre_ids, (w, p)
            span_lo = len(pre_ids)
            span_hi = len(ids) - len(tail_ids)
            assert 1 <= span_hi - span_lo <= 2, (w, p, span_hi - span_lo)
            assert ids[span_hi:] == tail_ids, (w, p)
            seqs.append(ids)
            meta.append((w, p, span_lo, span_hi))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None \
        else tok.eos_token_id
    maxlen = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), maxlen), int(pad_id),
                           dtype=torch.long)
    attn_mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        attn_mask[i, :len(s)] = 1
    model = AutoModelForCausalLM.from_pretrained(
        str(mdir), torch_dtype=torch.bfloat16, device_map='auto')
    model.eval()
    t_load = time.monotonic() - t0
    with torch.no_grad():
        out = model(input_ids=input_ids.to(model.device),
                    attention_mask=attn_mask.to(model.device),
                    output_hidden_states=True)
    HS = [h.float().cpu().numpy() for h in out.hidden_states]
    del model, out
    torch.cuda.empty_cache()
    print('P2817 Arm B forward done (%d seqs, load+run %.1fs)'
          % (len(seqs), time.monotonic() - t0), flush=True)

    zt = {w: unit((Etab[tid(w)].astype(np.float64)
                   / np.sqrt((Etab[tid(w)] ** 2).mean() + eps) * g))
          for w in TARGETS}
    pos_pairs = [(a, b) for a in range(N_POS) for b in range(a + 1,
                                                           N_POS)]
    prof_stab, full_stab, id_cv = [], [], []
    for L in range(len(HS)):
        H = HS[L]
        ps, fs, cvs = [], [], []
        for wi, w in enumerate(TARGETS):
            idxs = [wi * N_POS + p for p in range(N_POS)]
            hs = np.stack([H[i, meta[i][2]:meta[i][3], :]
                           .astype(np.float64).mean(0) for i in idxs])
            prof = hs @ unitD.T
            pn = prof / np.maximum(np.linalg.norm(prof, axis=1,
                                                  keepdims=True), 1e-30)
            pcs = [float(pn[a] @ pn[b]) for a, b in pos_pairs]
            hn = hs / np.maximum(np.linalg.norm(hs, axis=1,
                                                keepdims=True), 1e-30)
            fcs = [float(hn[a] @ hn[b]) for a, b in pos_pairs]
            idc = hs @ zt[w]
            ps.append(float(np.mean(pcs)))
            fs.append(float(np.mean(fcs)))
            cvs.append(float(np.std(idc)
                             / max(abs(float(np.mean(idc))), 1e-9)))
        prof_stab.append(float(np.mean(ps)))
        full_stab.append(float(np.mean(fs)))
        id_cv.append(float(np.mean(cvs)))
    win = range(LAYER_LO, LAYER_HI + 1)
    m_prof = float(np.mean([prof_stab[L] for L in win]))
    m_full = float(np.mean([full_stab[L] for L in win]))
    gap_ctx = m_prof - prof_func_ref
    p_p1n = bool(m_prof >= THRESH_POS)
    p_p3 = bool(gap_ctx >= THRESH_CTX)
    print('P2817 Arm B prof_stab mean[4..35]=%.4f full=%.4f '
          'gap_ctx_vs_2816=%.4f P-P1n=%s P-P3=%s'
          % (m_prof, m_full, gap_ctx, p_p1n, p_p3), flush=True)
    print('P2817 curves prof %s' % json.dumps(
        [round(x, 3) for x in prof_stab]), flush=True)

    verdict = {
        'n_layers': n_layers, 'n_heads': n_heads, 'head_dim': hd,
        'n_pass_heads': n_pass_heads,
        'attn_ov_subspace_semantic': bool(p_a1),
        'best_head': best_info, 'best_head_readout': head_readout,
        'proj_stats': {
            'overall_max': round(float(proj.max()), 4),
            'rq95_max': round(float(rq95.max()), 4),
            'rq95_median': round(float(np.median(rq95)), 4)},
        'prof_stab_natural_mean_4_35': round(m_prof, 4),
        'full_stab_natural_mean_4_35': round(m_full, 4),
        'id_cv_natural_mean_4_35': round(float(np.mean(id_cv[4:36])), 4),
        'prof_stab_functional_ref_2816': prof_func_ref,
        'gap_context': round(gap_ctx, 4),
        'prof_stab_natural_curve': [round(x, 4) for x in prof_stab],
        'full_stab_natural_curve': [round(x, 4) for x in full_stab],
        'id_cv_natural_curve': [round(x, 4) for x in id_cv],
        'semantics_position_invariant_natural': p_p1n,
        'position_effect_context_dependent': p_p3,
        'load_run_seconds': round(t_load, 1),
    }
    result = {'phase': 2817, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'ovpos.npz',
           head_proj=proj, head_rand_q95=rq95,
           prof_stab_natural=np.array(prof_stab, dtype=np.float32),
           full_stab_natural=np.array(full_stab, dtype=np.float32),
           id_cv_natural=np.array(id_cv, dtype=np.float32),
           unitD=unitD.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2817', elapsed)
    print('P2817 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2817 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
