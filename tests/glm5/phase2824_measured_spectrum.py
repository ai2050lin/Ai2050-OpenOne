"""Phase 2824 (LPF-37): MEASURED WRITE SPECTRUM -> CONCRETE CIRCUIT.

User goal (2026-09-17 06:22): the CONCRETE mechanism.  2823 showed
geometric census != actual writes (head level).  This phase measures
the FULL 1152-head write spectrum under apple context and turns it
into (a) a concrete head-level circuit map for apple->red, (b) a
measured-top editing operator.

Capture: one forward, hooks on all 36 o_proj inputs; keep rows for
apple / cherry / sky / grass.  Per layer & head:
  delta_{l,h} = W_O^{l,h} @ a_h(apple ctx),  c_{l,h} = <delta, dW_red>
Conditions:
  H_meas: rank-1 red->black on top-20 heads with measured c > 0
  H_geo:  same operator on geometric-census top-20 late heads
          (2819 best_val order; expected to include wrong-sign heads)
  C_cols: 2822 V1 top20/layer column redirect (replicate)
  COMBO:  C_cols + H_meas
Prereg (frozen):
  P1: d apple(H_meas) > d apple(H_geo)   (measured selection wins)
  P2: d apple(COMBO) >= d_cols + d_heads - 0.2 AND > max(parts)
  P3: pearson(c_apple spectrum, c_sky spectrum) < 0.5 over 1152
  P4: |d sky(H_meas)| < 0.2  (rank-1 isolation)
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
OUT = BASE / 'phase2824' / 'measured_spectrum'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SRC_2819 = BASE / 'phase2819' / 'knowledge_edit_locus'
SEED = 2824
N_HEADS_EDIT = 20
COLOR_WORDS = ['red', 'green', 'blue', 'black', 'white', 'yellow',
               'brown', 'pink', 'gray', 'purple', 'orange', 'crimson']
COLOURS = ['red', 'black', 'purple', 'blue', 'green', 'yellow']
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

PREREG = {
    'P1': 'measured_selection: d apple(H_meas, top-20 measured '
          'c>0 heads) > d apple(H_geo, geometric top-20 late heads)',
    'P2': 'combo_additive: d apple(COMBO) >= d_cols + d_heads - 0.2 '
          'AND > max(d_cols, d_heads)',
    'P3': 'head_specificity: pearson(c_apple, c_sky) over 1152 '
          'heads < 0.5',
    'P4': 'rank1_isolation: |d sky(H_meas)| < 0.2',
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
    res2819 = json.loads((SRC_2819 / 'result.json').read_text(
        encoding='utf-8'))

    execution = {
        'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
        'prereg': PREREG, 'seed': SEED,
        'n_heads_edit': N_HEADS_EDIT, 'entities': ENTS48,
        'note': 'full 1152-head measured write spectrum under apple '
                'context; measured-top rank-1 editing vs geometric '
                'selection; combo with 2822 column redirect'}
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
    except KeyError:
        Wu = Etab
    cfg = json.loads((mdir / 'config.json').read_text(encoding='utf-8'))
    eps = float(cfg.get('rms_norm_eps', 1e-6))
    n_layers = int(cfg.get('num_hidden_layers', 36))
    n_heads = int(cfg.get('num_attention_heads', 32))

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

    h7 = np.load(SRC_2807 / 'heldout.npz')
    b11 = np.load(SRC_2811 / 'battery.npz')
    cent = {c: np.stack([Wu[tid(w)].astype(np.float64)
                         for w in CATS[c]]).mean(0) for c in CAT_WORDS}
    Cm = np.stack([cent[c] for c in CAT_WORDS])
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    gate_dW = float(np.abs(dW - h7['dW_class'].astype(np.float64)).max())
    Z_eval = np.stack([(Etab[tid(w)].astype(np.float64)
                        / np.sqrt((Etab[tid(w)] ** 2).mean() + eps) * g)
                       for w in res2811['eval_words']])
    gate_Z = float(np.abs(Z_eval - b11['Z_eval'].astype(np.float64)).max())
    assert gate_dW < 1e-6 and gate_Z < 1e-4

    def zw(t):
        e = Etab[tid(t)].astype(np.float64)
        return e / np.sqrt((e ** 2).mean() + eps) * g

    for w in COLOR_WORDS + COLOURS + ['The', ' is']:
        tid(w)
    dWc = {}
    for c in ['red', 'black']:
        others = [w for w in COLOR_WORDS if w != c]
        dWc[c] = unit(zw(c) - np.stack([zw(w) for w in others]).mean(0))
    r_np = dWc['red'].astype(np.float32).astype(np.float64)
    b_np = dWc['black'].astype(np.float32).astype(np.float64)

    # ---------- CUDA ----------
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(mdir), dtype=torch.bfloat16, device_map='auto')
    model.eval()
    dev = model.device

    ent_tok = {s: tok(' ' + s, add_special_tokens=False)['input_ids']
               for s in ENTS48}
    k_seqs = [[tid('The')] + ent_tok[s] + [tid(' is')] for s in ENTS48]
    k_ids = torch.tensor(k_seqs, dtype=torch.long)
    k_mask = torch.ones_like(k_ids)
    col_id = {c: tid(c) for c in COLOURS}

    def forward():
        with torch.no_grad():
            out = model(input_ids=k_ids.to(dev),
                        attention_mask=k_mask.to(dev))
        return out.logits[:, -1, :].float().cpu().numpy()

    def colour_readout(lg):
        red_i = [ENTS48.index(s) for s in RED_ENTS + CTRL_ENTS]
        return {s: float(lg[i, col_id['black']] - lg[i, col_id['red']])
                for i, s in zip(red_i, RED_ENTS + CTRL_ENTS)}

    # ---------- capture: all 36 o_proj inputs, 4 rows ----------
    keep = [ENTS48.index(s) for s in ['apple', 'cherry', 'sky',
                                      'grass']]
    store = {}

    def make_cap(L):
        def cap(mod, args):
            store[L] = args[0][keep].detach().float().cpu().numpy()
            return None
        return cap

    hooks = [model.model.layers[L].self_attn.o_proj
             .register_forward_pre_hook(make_cap(L))
             for L in range(n_layers)]
    forward()
    for h in hooks:
        h.remove()
    print('P2824 captured %d layers' % len(store), flush=True)

    # ---------- measured spectrum: c[l,h] for 4 entities ----------
    Wo0 = read_tensor('model.layers.0.self_attn.o_proj.weight')
    d_model, hd_total = Wo0.shape
    hd = hd_total // n_heads
    del Wo0
    spec = np.zeros((n_layers, n_heads, 4), dtype=np.float64)
    dnorm = np.zeros((n_layers, n_heads), dtype=np.float64)
    for L in range(n_layers):
        W = read_tensor('model.layers.%d.self_attn.o_proj.weight' % L)
        W3 = W.astype(np.float64).reshape(d_model, n_heads, hd)
        for k in range(4):
            keys = store[L][k, 1:3, :].mean(axis=0)  # (4096,)
            K3 = keys.reshape(n_heads, hd)
            delta = np.einsum('dkh,kh->dk', W3, K3)  # (2560, 32)
            spec[L, :, k] = delta.T @ r_np
            if k == 0:
                dnorm[L] = np.linalg.norm(delta, axis=0)
        del W, W3
    c_ap = spec[:, :, 0].ravel()
    c_sky = spec[:, :, 2].ravel()
    c_gr = spec[:, :, 3].ravel()
    p3_r = float(np.corrcoef(c_ap, c_sky)[0, 1])
    pos_n = int((c_ap > 0).sum())
    top_meas = sorted([(int(L), int(h), float(spec[L, h, 0]))
                       for L in range(n_layers)
                       for h in range(n_heads) if spec[L, h, 0] > 0],
                      key=lambda t: -t[2])[:N_HEADS_EDIT]
    by_layer = {}
    for (L, h, c) in top_meas:
        by_layer.setdefault(L, []).append(h)
    print('P2824 spectrum pos %d/1152 corr_apple_sky %.3f top %s'
          % (pos_n, p3_r, json.dumps(
              {'L%d' % L: [round(c, 3) for (ll, hh, c) in top_meas
                           if ll == L] for L in sorted(by_layer)})),
          flush=True)

    # ---------- edit operators ----------
    def rank1_heads(heads):
        """heads = [(L, h)]; rank-1 V1 redirect using measured key."""
        backups = []
        for (L, h) in heads:
            W = model.model.layers[L].self_attn.o_proj.weight.data
            sl = slice(h * hd, (h + 1) * hd)
            key = store[L][0, 1:3, h * hd:(h + 1) * hd].mean(axis=0)
            kn2 = float(key @ key)
            if kn2 < 1e-12:
                continue
            delta = W[:, sl].float().cpu().numpy().astype(
                np.float64) @ key
            c = float(delta @ r_np)
            delta_new = delta - c * r_np + c * b_np
            d = delta_new - delta
            backups.append((L, sl, W[:, sl].clone()))
            upd = torch.tensor(
                np.outer(d.astype(np.float32),
                         (key / kn2).astype(np.float32)),
                device=W.device, dtype=W.dtype)
            W[:, sl] = W[:, sl] + upd
        return backups

    def col_redirect(cols):
        backups = []
        for (L, j, comp) in cols:
            W = model.model.layers[L].mlp.down_proj.weight.data
            backups.append((L, j, W[:, j].clone()))
            w = W[:, j].float().cpu().numpy().astype(np.float64)
            w = w - comp * r_np + comp * b_np
            W[:, j] = torch.tensor(w.astype(np.float32),
                                   device=W.device)
        return backups

    def restore(heads_bk, cols_bk):
        for (L, sl, orig) in heads_bk:
            model.model.layers[L].self_attn.o_proj.weight.data[:, sl] \
                = orig
        for (L, j, orig) in cols_bk:
            model.model.layers[L].mlp.down_proj.weight.data[:, j] = \
                orig

    # census for column redirect (same as 2822)
    census_cols = []
    for L in range(26, 36):
        name = 'model.layers.%d.mlp.down_proj.weight' % L
        with safe_open(str(mdir / index[name]), framework='pt') as f:
            W = f.get_tensor(name).float().numpy().astype(np.float32)
        Wn = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True),
                            1e-12)
        cr = Wn.T @ r_np.astype(np.float32)
        order = np.argsort(-np.abs(cr))[:20]
        census_cols += [(L, int(j), float(cr[j])) for j in order]
        del W, Wn
    by_l = {}
    for (L, j, c) in census_cols:
        by_l.setdefault(L, []).append((L, j, c))
    cols20 = []
    for L in sorted(by_l):
        cols20 += sorted(by_l[L], key=lambda t: -abs(t[2]))[:20]
    # wait: per-layer top20 of 200 needs top20 per layer
    cols20 = []
    for L in sorted(by_l):
        cols20 += sorted(by_l[L], key=lambda t: -abs(t[2]))[:20]

    ov_rows = res2819['verdict']['K2']['ov_rows']
    late = [r for r in ov_rows if r['layer'] >= 30
            and r['best_val'] > r['rand_q95']]
    late.sort(key=lambda r: -r['best_val'])
    geo_heads = [(int(r['layer']), int(r['head']))
                 for r in late[:N_HEADS_EDIT]]
    meas_heads = [(L, h) for (L, h, c) in top_meas]

    cmar0 = colour_readout(forward())
    print('P2824 baseline %s' % json.dumps(
        {k: round(v, 2) for k, v in cmar0.items()}), flush=True)

    def run_cond(heads=None, cols=None):
        hb = rank1_heads(heads) if heads else []
        cb = col_redirect(cols) if cols else []
        lg = forward()
        restore(hb, cb)
        return colour_readout(lg)

    m_meas = run_cond(heads=meas_heads)
    m_geo = run_cond(heads=geo_heads)
    m_cols = run_cond(cols=cols20)
    m_combo = run_cond(heads=meas_heads, cols=cols20)
    m_rest = colour_readout(forward())
    ok_rest = max(abs(m_rest[s] - cmar0[s])
                  for s in RED_ENTS + CTRL_ENTS) < 0.1

    d = lambda m, s: round(m[s] - cmar0[s], 3)
    d_meas = m_meas['apple'] - cmar0['apple']
    d_geo = m_geo['apple'] - cmar0['apple']
    d_cols = m_cols['apple'] - cmar0['apple']
    d_combo = m_combo['apple'] - cmar0['apple']
    p1 = bool(d_meas > d_geo)
    p2 = bool(d_combo >= d_cols + d_meas - 0.2
              and d_combo > max(d_cols, d_meas))
    p4 = bool(abs(m_meas['sky'] - cmar0['sky']) < 0.2)
    print('P2824 dapple meas %.3f geo %.3f cols %.3f combo %.3f '
          'dsky_meas %.3f restore %s P1=%s P2=%s P4=%s'
          % (d_meas, d_geo, d_cols, d_combo,
             m_meas['sky'] - cmar0['sky'], ok_rest, p1, p2, p4),
          flush=True)

    verdict = {
        'baseline': {k: round(v, 3) for k, v in cmar0.items()},
        'spectrum': {'n_pos': pos_n, 'corr_apple_sky': round(p3_r, 3),
                     'corr_apple_grass': round(float(np.corrcoef(
                         c_ap, c_gr)[0, 1]), 3),
                     'top_measured': [{'L': L, 'h': h,
                                       'c': round(c, 3)}
                                      for (L, h, c) in top_meas],
                     'geo_heads': geo_heads},
        'deltas': {'meas': {s: d(m_meas, s)
                            for s in RED_ENTS + CTRL_ENTS},
                   'geo': {s: d(m_geo, s)
                           for s in RED_ENTS + CTRL_ENTS},
                   'cols': {s: d(m_cols, s)
                            for s in RED_ENTS + CTRL_ENTS},
                   'combo': {s: d(m_combo, s)
                             for s in RED_ENTS + CTRL_ENTS}},
        'P1_measured_wins': p1,
        'P2_combo_additive': p2,
        'P3_head_specificity': bool(p3_r < 0.5),
        'P4_rank1_isolation': p4,
        'restore_ok': bool(ok_rest),
        'gate': {'dW': gate_dW, 'Zeval': gate_Z},
        'load_run_seconds': round(time.monotonic() - t0, 1),
    }
    result = {'phase': 2824, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'spectrum.npz',
           spec=spec.astype(np.float32),
           dnorm=dnorm.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2824', elapsed)
    print('P2824 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2824 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
