"""Phase 2816 (LPF-29): SEMANTICS IN LAYERS vs RULES — position-fixed
parameters, position-free meanings.

User questions (2026-09-17):
 (a) embeddings carry much semantics; how much LANGUAGE do the layer
     PARAMETERS carry — pure rules, or semantics+rules mixed?
 (b) in autoregression a semantic can sit at ANY position, but layer
     parameters are position-FIXED — what relates the two?

Arm W (zero-forward, all 37 layers): neuron-level semantic-write
census.  For layer L, every MLP neuron j writes w_j = W_down[:, j]
into the residual stream; every attention input dim j writes
o_j = W_O[:, j] (per-head OV write components).  Semantic content in
parameters = alignment of these write vectors with the 10 class
contrast directions unitD (2807 protocol).  Random-vector baseline per
matrix gives the shape-matched null.
  P-W1 mlp_neuron_semantic: exists L with max_j cos(W_down[:,j],
       unitD_i) >= 0.30 AND that layer's random q95 < 0.30
  P-W2 attn_ov_semantic: same for o_proj
Lower bound note: single-column test misses combination-coded writes.

Arm P (CUDA, qwen3-4b bf16, device_map auto): position-permutation
stability.  8 single-token targets x 7 slots in a fixed 7-content-token
function-word frame (same filler multiset every sentence; target slides
slots 0..6, RoPE positions 0..6).  Channels per layer L (0..37):
  class profile  P[w,p,L] = [cos(h, unitD_i)]_i   (10-dim, signed)
  identity       s[w,p,L]  = cos(h, z_word)       (scalar)
  full state     h
  P-P1 semantics_position_invariant: mean_{L in 4..35} profile
       stability (mean pairwise cos across positions) >= 0.80
  P-P2 profile > full: mean_{L in 4..35} (profile_stab - full_stab)
       >= 0.05
Synthesis targets: are layers rules (position-general ops) that read /
write semantic channels (position-free content in the residual stream)?

Prereg (frozen before any readout):
  P-W1/P-W2/P-P1/P-P2 as above.
  verdict: layers_rules_semantics_mixed = P-W1 OR P-W2;
           semantics_position_invariant = P-P1 AND P-P2
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
OUT = BASE / 'phase2816' / 'layers_semantics_position'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SEED = 2816
N_RAND_VEC = 30
THRESH = 0.30
LAYER_LO, LAYER_HI = 4, 35

TARGETS = ['apple', 'gold', 'eagle', 'sedan', 'Spain', 'soup', 'cave',
           'sock']
FILLERS = ['the', 'of', 'and', 'to', 'a', 'in']

PREREG = {
    'P-W1': 'mlp_neuron_semantic iff exists layer L with '
            'max_j cos(W_down[:,j], unitD_i) >= 0.30 AND that layer '
            'random-vector q95 < 0.30',
    'P-W2': 'attn_ov_semantic: same criterion for o_proj',
    'P-P1': 'semantics_position_invariant iff mean_{L in 4..35} '
            'profile stability (mean pairwise cos across 7 positions, '
            '8 words) >= 0.80',
    'P-P2': 'profile > full iff mean_{L in 4..35} (profile_stab - '
            'full_stab) >= 0.05',
    'verdict': 'layers_rules_semantics_mixed = P-W1 OR P-W2; '
               'semantics_position_invariant = P-P1 AND P-P2',
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

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED, 'thresh': THRESH,
                 'layer_window': [LAYER_LO, LAYER_HI],
                 'targets': TARGETS, 'fillers': FILLERS,
                 'n_rand_vec': N_RAND_VEC,
                 'note': 'Arm W zero-forward write census (single-column '
                         'lower bound); Arm P CUDA position-permutation, '
                         'add_special_tokens=False, raw content ids, '
                         'RoPE positions 0..6'}
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

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(mdir), local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, t
            tc[t] = int(ids[0])
        return tc[t]

    eval_words = res2811['eval_words']
    h7 = np.load(SRC_2807 / 'heldout.npz')
    b11 = np.load(SRC_2811 / 'battery.npz')
    atlas_words = [w for v in CATS.values() for w in v]
    cent = {c: np.stack([Wu[tid(w)].astype(np.float64)
                         for w in CATS[c]]).mean(0) for c in CAT_WORDS}
    Cm = np.stack([cent[c] for c in CAT_WORDS])
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    gate_dW = float(np.abs(dW - h7['dW_class'].astype(np.float64)).max())
    Z_eval = np.stack([(Etab[tid(w)].astype(np.float64)
                        / np.sqrt((Etab[tid(w)] ** 2).mean() + eps) * g)
                       for w in eval_words])
    gate_Z = float(np.abs(Z_eval - b11['Z_eval'].astype(np.float64)).max())
    print('P2816 gates: dW=%.2e Zeval=%.2e n_layers=%d tie=%s'
          % (gate_dW, gate_Z, n_layers, tie), flush=True)
    assert gate_dW < 1e-6 and gate_Z < 1e-4

    unitD = np.stack([unit(dW[i]) for i in range(10)])
    rng = np.random.default_rng(SEED)
    rand_vs = np.stack([unit(rng.standard_normal(Etab.shape[1]))
                        for _ in range(N_RAND_VEC)])

    # ---------- Arm W: write census ----------
    layer_rows = []
    best = {'down': (0.0, None), 'ov': (0.0, None)}
    for L in range(n_layers):
        row = {'layer': L}
        for kind, name in (('down', 'model.layers.%d.mlp.down_proj.weight'
                            % L),
                           ('ov', 'model.layers.%d.self_attn.o_proj.weight'
                            % L)):
            try:
                W = read_tensor(name).astype(np.float32)
            except KeyError:
                row[kind + '_present'] = False
                continue
            cn = np.linalg.norm(W, axis=0)
            cn = np.maximum(cn, 1e-12)
            Wn = W / cn
            class_max = []
            for i in range(10):
                colcos = np.abs(Wn.T @ unitD[i].astype(np.float32))
                class_max.append((float(colcos.max()), int(colcos.argmax())))
            cm_val = max(m for m, _ in class_max)
            cm_cls = CAT_WORDS[int(np.argmax([m for m, _ in class_max]))]
            rnd = [float(np.abs(Wn.T @ v).max()) for v in rand_vs]
            rq95 = float(np.quantile(rnd, 0.95))
            row[kind + '_max'] = round(cm_val, 4)
            row[kind + '_argmax_class'] = cm_cls
            row[kind + '_rand_q95'] = round(rq95, 4)
            row[kind + '_pass'] = bool(cm_val >= THRESH and rq95 < THRESH)
            if kind in best and cm_val > best[kind][0]:
                j = class_max[int(np.argmax([m for m, _ in class_max]))][1]
                best[kind] = (cm_val, {'layer': L, 'neuron': j,
                                       'cos': round(cm_val, 4),
                                       'class': cm_cls})
            del W, Wn
        layer_rows.append(row)
        if L % 6 == 0:
            print('P2816 Arm W layer %d %s' % (L, json.dumps(row)),
                  flush=True)
    p_w1 = any(r.get('down_pass', False) for r in layer_rows)
    p_w2 = any(r.get('ov_pass', False) for r in layer_rows)
    n_down_pass = sum(1 for r in layer_rows if r.get('down_pass'))
    n_ov_pass = sum(1 for r in layer_rows if r.get('ov_pass'))
    print('P2816 Arm W done: down_pass_layers=%d ov_pass_layers=%d '
          'best_down=%s best_ov=%s'
          % (n_down_pass, n_ov_pass, json.dumps(best['down']),
             json.dumps(best['ov'])), flush=True)
    readouts = {}
    for kind in ('down', 'ov'):
        info = best[kind][1]
        if info is None:
            continue
        L, j = info['layer'], info['neuron']
        name = ('model.layers.%d.mlp.down_proj.weight' % L if kind
                == 'down' else
                'model.layers.%d.self_attn.o_proj.weight' % L)
        W = read_tensor(name).astype(np.float32)
        v = W[:, j]
        top_ids = np.argsort(-(Wu @ v.astype(np.float64)))[:8].tolist()
        readouts[kind] = {'info': info,
                          'top_unembed': [tok.decode([t]).strip()
                                          for t in top_ids]}
        del W
    print('P2816 best-neuron readouts %s' % json.dumps(readouts),
          flush=True)

    # ---------- Arm P: position permutation (CUDA) ----------
    import torch
    from transformers import AutoModelForCausalLM
    for w in TARGETS + FILLERS:
        tid(w)
    seqs, meta = [], []
    for w in TARGETS:
        for p in range(7):
            content = FILLERS[:p] + [w] + FILLERS[p:]
            ids = [tid(t) for t in content]
            seqs.append(ids)
            meta.append((w, p))
    input_ids = torch.tensor(seqs, dtype=torch.long)
    model = AutoModelForCausalLM.from_pretrained(
        str(mdir), torch_dtype=torch.bfloat16, device_map='auto')
    model.eval()
    t_load = time.monotonic() - t0
    with torch.no_grad():
        out = model(input_ids=input_ids.to(model.device),
                    output_hidden_states=True)
    HS = [h.float().cpu().numpy() for h in out.hidden_states]
    del model, out
    torch.cuda.empty_cache()
    print('P2816 Arm P forward done (%d seqs, load+run %.1fs)'
          % (len(seqs), time.monotonic() - t0), flush=True)

    zt = {w: unit((Etab[tid(w)].astype(np.float64)
                   / np.sqrt((Etab[tid(w)] ** 2).mean() + eps) * g))
          for w in TARGETS}
    n_pos = 7
    pos_pairs = [(a, b) for a in range(n_pos) for b in range(a + 1,
                                                           n_pos)]
    prof_stab, full_stab, id_cv = [], [], []
    for L in range(len(HS)):
        H = HS[L]
        ps, fs, cvs = [], [], []
        for wi, w in enumerate(TARGETS):
            idxs = [wi * n_pos + p for p in range(n_pos)]
            hs = np.stack([H[i, meta[i][1], :].astype(np.float64)
                           for i in idxs])
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
    m_gap = m_prof - m_full
    p_p1 = bool(m_prof >= 0.80)
    p_p2 = bool(m_gap >= 0.05)
    print('P2816 Arm P prof_stab mean[4..35]=%.4f full_stab=%.4f gap='
          '%.4f id_cv=%.3f P-P1=%s P-P2=%s'
          % (m_prof, m_full, m_gap, float(np.mean(id_cv[4:36])), p_p1,
             p_p2), flush=True)
    print('P2816 curves prof %s' % json.dumps(
        [round(x, 3) for x in prof_stab]), flush=True)
    print('P2816 curves full %s' % json.dumps(
        [round(x, 3) for x in full_stab]), flush=True)

    verdict = {
        'n_layers': n_layers,
        'n_down_pass_layers': n_down_pass, 'n_ov_pass_layers': n_ov_pass,
        'layer_rows': layer_rows, 'best_neuron_readouts': readouts,
        'mlp_neuron_semantic': p_w1, 'attn_ov_semantic': p_w2,
        'prof_stab_mean_4_35': round(m_prof, 4),
        'full_stab_mean_4_35': round(m_full, 4),
        'gap_mean_4_35': round(m_gap, 4),
        'id_cv_mean_4_35': round(float(np.mean(id_cv[4:36])), 4),
        'prof_stab_curve': [round(x, 4) for x in prof_stab],
        'full_stab_curve': [round(x, 4) for x in full_stab],
        'id_cv_curve': [round(x, 4) for x in id_cv],
        'semantics_position_invariant': bool(p_p1 and p_p2),
        'layers_rules_semantics_mixed': bool(p_w1 or p_w2),
        'load_run_seconds': round(t_load, 1),
    }
    result = {'phase': 2816, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'channels.npz',
           prof_stab=np.array(prof_stab, dtype=np.float32),
           full_stab=np.array(full_stab, dtype=np.float32),
           id_cv=np.array(id_cv, dtype=np.float32),
           unitD=unitD.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2816', elapsed)
    print('P2816 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2816 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
