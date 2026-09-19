"""Phase 2829 = GAMMA (LPF-42): CROSS-SCALE 14B VERIFICATION.

User plan step 3: does the 4B "write-head reuse x entity key" structure
exist in 14B?  Local Qwen3-14B found (40L x 40H, hidden 5120,
tie_word_embeddings=False).  GPU 16GB < 28GB -> device_map='auto'
mixed offload; forwards are slow, budget accordingly.

Replicates 2824/2825 core on 14B:
  - full 48-entity capture, 9-direction measured spectrum (1600 heads)
  - arms: A_raw (apple top-20 plain), B_diff (diff-selected top-20
    plain), C_orth (diff heads + orthogonal keys)
  - colour margins red/black for all 48 entities
Structural correspondence checks:
  - last-layer mega write head (4B: L35 h0 c=1.301)
  - bundle head (4B: L29 h27, apple/sky = 27.9)

Prereg (frozen):
  G1: spectrum_signif: apple red-spectrum top-20 mean c > null q95
  G2: triple_efficacy: C_orth d_margin(apple) >= 1.0
  G3: orth_isolation: C_orth spill(47 ents) < 0.7 x A_raw spill
  G4: last_layer_head: red-spectrum argmax head in the last layer
      AND its c > 5 x null q95
  G5: bundle_head_exists: some head with >= 6/9 positive directions
      for apple AND apple/sky ratio > 5 in its strongest direction
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
OUT = BASE / 'phase2829' / 'gamma_14b'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
MODEL_DIR = ROOT / 'models' / 'hf' / 'Qwen3-14B'
SEED = 2829
N_HEADS_EDIT = 20
N_NULL = 100
COLOR_WORDS = ['red', 'green', 'blue', 'black', 'white', 'yellow',
               'brown', 'pink', 'gray', 'purple', 'orange', 'crimson']
RED_ENTS = ['apple', 'cherry', 'strawberry', 'tomato', 'blood']
CTRL_ENTS = ['sky', 'grass', 'coal', 'banana']
ENTS48 = ['apple', 'cherry', 'lemon', 'banana', 'grape', 'bread',
          'dog', 'cat', 'mouse', 'elephant', 'lion', 'bee', 'horse',
          'ox', 'whale', 'shark', 'kitten', 'puppy', 'rock', 'stone',
          'feather', 'pillow', 'sponge', 'steel', 'brick', 'balloon',
          'bubble', 'hammer', 'nail', 'car', 'plane', 'rocket',
          'trumpet', 'piano', 'cup', 'sun', 'moon', 'ice', 'cloud',
          'tree', 'mountain', 'river',
          'strawberry', 'tomato', 'blood', 'sky', 'grass', 'coal']
DOMAINS = ['size', 'weight', 'temperature', 'speed', 'hardness',
           'taste', 'loudness', 'shape']
PAIR = {'size': ('big', 'small'), 'weight': ('heavy', 'light'),
        'temperature': ('hot', 'cold'), 'speed': ('fast', 'slow'),
        'hardness': ('hard', 'soft'), 'taste': ('sweet', 'bitter'),
        'loudness': ('loud', 'quiet'), 'shape': ('round', 'sharp')}
ADJ_ALL = sorted({a for p in PAIR.values() for a in p})
DIR_NAMES = DOMAINS + ['color_red']

PREREG = {
    'G1': 'spectrum_signif: apple red-spectrum top-20 mean c > null q95',
    'G2': 'triple_efficacy: C_orth d_margin(apple) >= 1.0',
    'G3': 'orth_isolation: C_orth spill(47) < 0.7 x A_raw spill',
    'G4': 'last_layer_head: red argmax head in last layer AND '
          'c > 5 x null q95',
    'G5': 'bundle_head_exists: some head >= 6/9 positive dirs for '
          'apple AND apple/sky ratio > 5 in strongest direction',
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

    execution = {
        'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
        'prereg': PREREG, 'seed': SEED,
        'n_heads_edit': N_HEADS_EDIT, 'model': 'Qwen3-14B',
        'entities': ENTS48, 'dir_names': DIR_NAMES,
        'note': 'GAMMA: cross-scale verification of write-head reuse '
                'x entity key on Qwen3-14B (mixed offload)'}
    fc.save(OUT / 'execution.json', execution)

    # ---------- tensors + gates ----------
    from safetensors import safe_open
    mdir = MODEL_DIR
    index = json.loads((mdir / 'model.safetensors.index.json')
                       .read_text(encoding='utf-8'))['weight_map']

    def read_tensor(name):
        with safe_open(str(mdir / index[name]), framework='pt') as f:
            return f.get_tensor(name).float().numpy()

    g = read_tensor('model.norm.weight').astype(np.float64)
    Wu = read_tensor('lm_head.weight')
    cfg = json.loads((mdir / 'config.json').read_text(encoding='utf-8'))
    eps = float(cfg.get('rms_norm_eps', 1e-6))
    n_layers = int(cfg.get('num_hidden_layers', 40))
    n_heads = int(cfg.get('num_attention_heads', 40))

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

    # 14B-adapted gate + directions.  Measured: embed_tokens rows are
    # ~orthogonal to lm_head rows (cos ~ 0.005, untied).  The 4B
    # z(w)=rmsnorm(E_w)*g back-solve does NOT represent the 14B readout
    # direction.  Hidden-space readout direction is u_w = g * Wu_row(w)
    # since logit_w = <rmsnorm(h)*g, Wu_row(w)> = <h, g*Wu_row(w)>.
    def u(t):
        return g * Wu[tid(t)].astype(np.float64)

    for w in COLOR_WORDS + ADJ_ALL + ['The', ' is']:
        tid(w)
    others_red = [w for w in COLOR_WORDS if w != 'red']
    d_red_cand = unit(u('red') - np.stack(
        [u(w) for w in others_red]).mean(0))
    gate_dir = float(min(float(d_red_cand @ (u('red') - u(w)))
                         for w in others_red))
    assert gate_dir > 0, 'red direction fails separation'
    for d, p in PAIR.items():
        dd = unit(u(p[0]) - u(p[1]))
        assert float(dd @ (u(p[0]) - u(p[1]))) > 0, d
    print('P2829 gate dir min margin %.3f' % gate_dir, flush=True)

    dWc = {}
    for c in ['red', 'black']:
        others = [w for w in COLOR_WORDS if w != c]
        dWc[c] = unit(u(c) - np.stack([u(w) for w in others]).mean(0))
    dD = {d: unit(u(p[0]) - u(p[1])) for d, p in PAIR.items()}
    dirs9 = [dD[d] for d in DOMAINS] + [dWc['red']]
    DMAT9 = np.stack(dirs9, axis=1)  # (5120, 9)

    # free numpy-side big arrays BEFORE model load (RAM 33.7GB total;
    # loader peak otherwise segfaults)
    import gc
    del Wu, g
    gc.collect()

    # ---------- CUDA (mixed offload) ----------
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(mdir), dtype=torch.bfloat16, device_map='auto',
        max_memory={0: '12GiB', 'cpu': '21GiB'},
        low_cpu_mem_usage=True)
    model.eval()
    dev = model.device
    print('P2829 model loaded first_device=%s' % str(dev), flush=True)

    ent_tok = {s: tok(' ' + s, add_special_tokens=False)['input_ids']
               for s in ENTS48}
    k_seqs = [[tid('The')] + ent_tok[s] + [tid(' is')] for s in ENTS48]
    k_ids = torch.tensor(k_seqs, dtype=torch.long)
    k_mask = torch.ones_like(k_ids)
    col_id = {c: tid(c) for c in ['red', 'black']}
    ap_row = ENTS48.index('apple')

    def forward():
        with torch.no_grad():
            out = model(input_ids=k_ids.to(dev),
                        attention_mask=k_mask.to(dev))
        return out.logits[:, -1, :].float().cpu().numpy()

    def margins(lg):
        return {s: float(lg[i, col_id['black']] - lg[i, col_id['red']])
                for i, s in enumerate(ENTS48)}

    # ---------- capture all 48 ----------
    store = {}

    def make_cap(Lr):
        def cap(mod, args):
            store[Lr] = args[0][:, 1:3, :].detach().float().cpu() \
                .numpy()
            return None
        return cap

    hooks = [model.model.layers[Lr].self_attn.o_proj
             .register_forward_pre_hook(make_cap(Lr))
             for Lr in range(n_layers)]
    forward()
    for h in hooks:
        h.remove()
    print('P2829 captured %d layers' % len(store), flush=True)

    # ---------- spectrum + null in one pass ----------
    Wo0 = read_tensor('model.layers.0.self_attn.o_proj.weight')
    d_model, hd_total = Wo0.shape
    hd = hd_total // n_heads
    del Wo0
    ne = len(ENTS48)
    rng = np.random.default_rng(SEED)
    U = rng.standard_normal((N_NULL, d_model))
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    spec9 = np.zeros((n_layers, ne, n_heads, 9), dtype=np.float32)
    null_parts = []
    for Lr in range(n_layers):
        W = read_tensor(
            'model.layers.%d.self_attn.o_proj.weight' % Lr)
        W3 = W.astype(np.float64).reshape(d_model, n_heads, hd)
        K3 = store[Lr].mean(axis=1).reshape(ne, n_heads, hd)
        delta = np.einsum('dhk,ehk->edh', W3, K3)
        spec9[Lr] = np.einsum('edh,dr->ehr', delta, DMAT9) \
            .astype(np.float32)
        d_ap = delta[ap_row]  # (5120, 40)
        null_parts.append(d_ap.T @ U.T)  # (40, 100)
        del W, W3, delta
        if Lr % 10 == 0:
            print('P2829 spec L%d' % Lr, flush=True)
    null_pool = np.concatenate(null_parts, axis=0).ravel()
    null_q95 = float(np.quantile(null_pool, 0.95))
    print('P2829 null q95 %.4f' % null_q95, flush=True)

    red_spec = spec9[:, :, :, 8].astype(np.float64)  # (L, e, h)
    c_ap = red_spec[:, ap_row, :]
    top_raw = sorted([(int(Lr), int(h), float(c_ap[Lr, h]))
                      for Lr in range(n_layers) for h in range(n_heads)
                      if c_ap[Lr, h] > 0],
                     key=lambda t: -t[2])[:N_HEADS_EDIT]
    ctrl_rows = [ENTS48.index(s) for s in CTRL_ENTS]
    c_ctrl = red_spec[:, ctrl_rows, :].mean(axis=1)
    diff = c_ap - c_ctrl
    top_diff = sorted([(int(Lr), int(h), float(diff[Lr, h]))
                       for Lr in range(n_layers) for h in range(n_heads)
                       if c_ap[Lr, h] > 0],
                      key=lambda t: -t[2])[:N_HEADS_EDIT]
    heads_raw = [(Lr, h) for (Lr, h, _) in top_raw]
    heads_diff = [(Lr, h) for (Lr, h, _) in top_diff]
    overlap = len(set(heads_raw) & set(heads_diff))
    am = np.unravel_index(np.argmax(c_ap), c_ap.shape)
    print('P2829 red argmax L%d h%d c=%.3f overlap %d/20'
          % (am[0], am[1], c_ap[am], overlap), flush=True)

    # ---------- edit arms (hook-equivalent rank-1) ----------
    # 14B mixed offload leaves some o_proj weights on the meta device
    # (cannot read/write .data).  A rank-1 weight edit W += outer(d,
    # key/kn2) is mathematically equivalent to a forward hook on
    # o_proj: output += (input_h . key / kn2) * d.  The hook version
    # needs no weight access; removal restores exactly.  c_{l,h} is
    # taken from the already-measured spec9 (same quantity).
    def edit_hooks(heads, orth):
        hs = []
        for (Lr, h) in heads:
            sl = slice(h * hd, (h + 1) * hd)
            key = store[Lr][ap_row, :, h * hd:(h + 1) * hd].mean(axis=0)
            kn2 = float(key @ key)
            if kn2 < 1e-12:
                continue
            keep = 1.0
            if orth:
                k2 = key.copy()
                for cr in ctrl_rows:
                    kc = store[Lr][cr, :, h * hd:(h + 1) * hd] \
                        .mean(axis=0)
                    n2 = float(kc @ kc)
                    if n2 < 1e-12:
                        continue
                    k2 = k2 - (k2 @ kc) / n2 * kc
                kn2b = float(k2 @ k2)
                if kn2b < 0.05 * kn2:
                    continue
                keep = float(k2 @ key) / kn2
                key, kn2 = k2, kn2b
            c = float(spec9[Lr, ap_row, h, 8]) * keep
            d = c * (dWc['black'] - dWc['red'])
            d_t = torch.tensor(d.astype(np.float32))
            k_t = torch.tensor((key / kn2).astype(np.float32))

            def make_hook(k_t, d_t, sl):
                def hook(mod, args, output):
                    x = args[0][..., sl]
                    proj = x @ k_t.to(device=x.device,
                                      dtype=x.dtype)
                    return output + proj.unsqueeze(-1) * d_t.to(
                        device=output.device, dtype=output.dtype)
                return hook

            hs.append(model.model.layers[Lr].self_attn.o_proj
                      .register_forward_hook(make_hook(k_t, d_t, sl)))
        return hs

    m0 = margins(forward())

    def run(heads, orth):
        hs = edit_hooks(heads, orth=orth)
        m = margins(forward())
        for h in hs:
            h.remove()
        return m

    print('P2829 arm A_raw...', flush=True)
    m_raw = run(heads_raw, orth=False)
    print('P2829 arm C_orth...', flush=True)
    m_orth = run(heads_diff, orth=True)
    m_rest = margins(forward())
    ok_rest = max(abs(m_rest[s] - m0[s]) for s in ENTS48) < 0.1

    d_raw = {s: m_raw[s] - m0[s] for s in ENTS48}
    d_orth = {s: m_orth[s] - m0[s] for s in ENTS48}
    spill = lambda dm: float(np.mean([abs(dm[s]) for s in ENTS48
                                      if s != 'apple']))
    focus = RED_ENTS + CTRL_ENTS

    # ---------- structural checks ----------
    hs_apple = spec9[:, ap_row, :, :].astype(np.float64).sum(axis=0)
    hs_sky = spec9[:, ENTS48.index('sky'), :, :].astype(np.float64) \
        .sum(axis=0)
    bundle = None
    for h in range(n_heads):
        npos = int((hs_apple[h] > 0).sum())
        if npos < 6:
            continue
        dstar = int(np.argmax(hs_apple[h]))
        ratio = float(hs_apple[h, dstar]
                      / max(abs(hs_sky[h, dstar]), 1e-9))
        if ratio > 5 and (bundle is None
                          or ratio > bundle['ratio']):
            bundle = {'head': int(h), 'n_pos_dirs': npos,
                      'strongest_dir': DIR_NAMES[dstar],
                      'apple': round(float(hs_apple[h, dstar]), 3),
                      'sky': round(float(hs_sky[h, dstar]), 3),
                      'ratio': round(ratio, 1)}
    last = int(n_layers - 1)
    g4_val = float(c_ap[last, :].max())
    g4_head = int(np.argmax(c_ap[last, :]))

    g1 = bool(np.mean([t[2] for t in top_raw]) > null_q95)
    g2 = bool(d_orth['apple'] >= 1.0)
    g3 = bool(spill(d_orth) < 0.7 * max(spill(d_raw), 1e-9))
    g4 = bool(am[0] == last and g4_val > 5 * null_q95)
    g5 = bool(bundle is not None)

    verdict = {
        'null_q95': round(null_q95, 4),
        'config': {'layers': n_layers, 'heads': n_heads,
                   'hidden': d_model, 'hd': hd},
        'baseline': {s: round(m0[s], 3) for s in focus},
        'top_raw': [{'L': Lr, 'h': h, 'c': round(c, 3)}
                    for (Lr, h, c) in top_raw[:10]],
        'overlap_raw_diff': overlap,
        'deltas': {
            'A_raw': {s: round(d_raw[s], 3) for s in focus},
            'C_orth': {s: round(d_orth[s], 3) for s in focus}},
        'spill': {'A_raw': round(spill(d_raw), 3),
                  'C_orth': round(spill(d_orth), 3)},
        'red_argmax': {'L': int(am[0]), 'h': int(am[1]),
                       'c': round(float(c_ap[am]), 3)},
        'last_layer_max': {'L': last, 'h': g4_head,
                           'c': round(g4_val, 3)},
        'bundle_head': bundle,
        'G1_signif': g1, 'G2_triple_efficacy': g2,
        'G3_orth_isolation': g3, 'G4_last_layer_head': g4,
        'G5_bundle_head': g5,
        'restore_ok': bool(ok_rest),
        'gate': {'dir_margin_min': round(gate_dir, 4),
                 'note': '14B readout directions from untied lm_head '
                         '(u_w = g*Wu_row); 4B heldout dW_class '
                         'incompatible (2560-dim, tied-embed)'},
        'load_run_seconds': round(time.monotonic() - t0, 1),
    }
    result = {'phase': 2829, 'plan': 'GAMMA', 'prereg': PREREG,
              'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'spec9_14b.npz', spec9=spec9)
    print('P2829 VERDICT %s' % json.dumps(verdict), flush=True)
    elapsed = time.monotonic() - t0
    cc.ledger('phase2829', elapsed)
    print('P2829 elapsed %.1fs G1=%s G2=%s G3=%s G4=%s G5=%s'
          % (elapsed, g1, g2, g3, g4, g5), flush=True)


if __name__ == '__main__':
    main()
