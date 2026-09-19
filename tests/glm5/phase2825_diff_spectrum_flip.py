"""Phase 2825 (LPF-38): DIFFERENTIAL SPECTRUM -> ENTITY-CONDITIONED WRITE + TRUE FLIP.

User goal (2026-09-17 06:54 "hao de, jixu"): continue.  2824 delivered the
measured head circuit but P4 failed: top-20 head rank-1 edit spills to
other entities (sky +0.688, banana +2.188) because conditioning lives in
the key (input side) while the write direction (output side) is shared.
This phase attacks the spill with two output-side mechanisms and then
pushes for a TRUE flip.

Mechanisms:
  A_raw : top-20 by c_apple, plain key   (2824 replication; spills)
  B_diff: top-20 by diff = c_apple - mean(c_ctrl), ctrl = sky/grass/coal/
          banana.  Heads that write red FOR APPLE and NOT for others.
  C_orth: same heads as B_diff but key orthogonalised: k' = k_apple -
          sum_c (k.k_hat_c) k_hat_c  -> rank-1 responds to apple, not ctrl.
  FLIP  : cols20 (2822 V1) + best head arm + emb-edit apple row
          (e += beta*(dWb-dWr)/g at apple token positions; token-private
          so zero structural spill to other entities).

Full 48-entity spectrum captured this run (spec[L,e,h]) and full
48-entity delta-margin matrix per arm -> selectivity profile.

Prereg (frozen):
  P1: diff_selection_isolation: |d_sky(B_diff)| < |d_sky(A_raw)|
      AND d_apple(B_diff) >= 0.5*d_apple(A_raw)
  P2: orth_key_isolation: |d_sky(C_orth)| < |d_sky(A_raw)|
      AND d_apple(C_orth) >= 0.5*d_apple(A_raw)
  P3: banana_spill_fixed: min(|d_banana(B_diff)|, |d_banana(C_orth)|) < 1.0
      (2824 A_raw spilled 2.188)
  P4: true_flip: exists (arm in {cols+B_diff, cols+C_orth}) x
      beta in [1.0, 2.0, 4.0] with margin_black_minus_red(apple) > 0
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
OUT = BASE / 'phase2825' / 'diff_spectrum_flip'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SEED = 2825
N_HEADS_EDIT = 20
BETAS = [1.0, 2.0, 4.0]
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
    'P1': 'diff_selection_isolation: |d_sky(B_diff)| < |d_sky(A_raw)| '
          'AND d_apple(B_diff) >= 0.5*d_apple(A_raw)',
    'P2': 'orth_key_isolation: |d_sky(C_orth)| < |d_sky(A_raw)| '
          'AND d_apple(C_orth) >= 0.5*d_apple(A_raw)',
    'P3': 'banana_spill_fixed: min(|d_banana(B_diff)|, '
          '|d_banana(C_orth)|) < 1.0',
    'P4': 'true_flip: exists (arm in {cols+B_diff, cols+C_orth}) x '
          'beta in [1.0, 2.0, 4.0] with margin_black_minus_red(apple) > 0',
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
        'prereg': PREREG, 'seed': SEED, 'betas': BETAS,
        'n_heads_edit': N_HEADS_EDIT, 'entities': ENTS48,
        'note': 'full 48-entity measured spectrum; differential head '
                'selection; orthogonal-key rank-1; flip冲击 via cols + '
                'heads + emb-edit apple row'}
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
    ap_id = tid(' apple')
    ap_row = ENTS48.index('apple')

    def forward():
        with torch.no_grad():
            out = model(input_ids=k_ids.to(dev),
                        attention_mask=k_mask.to(dev))
        return out.logits[:, -1, :].float().cpu().numpy()

    def margins(lg):
        return {s: float(lg[i, col_id['black']] - lg[i, col_id['red']])
                for i, s in enumerate(ENTS48)}

    # ---------- capture: all 36 o_proj inputs, all 48 rows x pos[1:3] ----------
    store = {}

    def make_cap(L):
        def cap(mod, args):
            store[L] = args[0][:, 1:3, :].detach().float().cpu().numpy()
            return None
        return cap

    hooks = [model.model.layers[L].self_attn.o_proj
             .register_forward_pre_hook(make_cap(L))
             for L in range(n_layers)]
    forward()
    for h in hooks:
        h.remove()
    print('P2825 captured %d layers x %d ents' % (len(store), len(ENTS48)),
          flush=True)

    # ---------- full spectrum: spec[L, e, h] ----------
    Wo0 = read_tensor('model.layers.0.self_attn.o_proj.weight')
    d_model, hd_total = Wo0.shape
    hd = hd_total // n_heads
    del Wo0
    spec = np.zeros((n_layers, len(ENTS48), n_heads), dtype=np.float64)
    for L in range(n_layers):
        W = read_tensor('model.layers.%d.self_attn.o_proj.weight' % L)
        W3 = W.astype(np.float64).reshape(d_model, n_heads, hd)
        K3 = store[L].mean(axis=1).reshape(len(ENTS48), n_heads, hd)
        delta = np.einsum('dhk,ehk->edh', W3, K3)
        spec[L] = np.einsum('edh,d->eh', delta, r_np)
        del W, W3, delta
    c_ap = spec[:, ap_row, :]  # (36, 32)
    ctrl_rows = [ENTS48.index(s) for s in CTRL_ENTS]
    c_ctrl = spec[:, ctrl_rows, :].mean(axis=1)  # (36, 32)

    # cross-check vs 2824 (apple, sky, grass rows)
    z24 = np.load(BASE / 'phase2824' / 'measured_spectrum' / 'spectrum.npz')
    s24 = z24['spec']  # (36, 32, 4) apple/cherry/sky/grass
    xchk = {
        'apple': float(np.abs(s24[:, :, 0] - c_ap).max()),
        'sky': float(np.abs(s24[:, :, 2]
                            - spec[:, ENTS48.index('sky'), :]).max()),
        'grass': float(np.abs(s24[:, :, 3]
                              - spec[:, ENTS48.index('grass'), :]).max()),
    }
    print('P2825 cross2824 %s' % json.dumps(
        {k: round(v, 6) for k, v in xchk.items()}), flush=True)

    # ---------- head selection ----------
    diff_spec = c_ap - c_ctrl
    top_raw = sorted([(int(L), int(h), float(c_ap[L, h]))
                      for L in range(n_layers) for h in range(n_heads)
                      if c_ap[L, h] > 0],
                     key=lambda t: -t[2])[:N_HEADS_EDIT]
    top_diff = sorted([(int(L), int(h), float(diff_spec[L, h]))
                       for L in range(n_layers) for h in range(n_heads)
                       if c_ap[L, h] > 0],
                      key=lambda t: -t[2])[:N_HEADS_EDIT]
    heads_raw = [(L, h) for (L, h, c) in top_raw]
    heads_diff = [(L, h) for (L, h, c) in top_diff]
    overlap = len(set(heads_raw) & set(heads_diff))
    print('P2825 heads overlap %d/20 diff_top %s' % (overlap, json.dumps(
        [{'L': L, 'h': h, 'diff': round(c, 3)} for (L, h, c) in
         top_diff[:8]])), flush=True)

    # ---------- edit operators ----------
    def rank1(heads, orth=False):
        backups = []
        keep_ratios = []
        for (L, h) in heads:
            W = model.model.layers[L].self_attn.o_proj.weight.data
            sl = slice(h * hd, (h + 1) * hd)
            key = store[L][ap_row, :, h * hd:(h + 1) * hd].mean(axis=0)
            kn2 = float(key @ key)
            if kn2 < 1e-12:
                continue
            if orth:
                khat = key / np.sqrt(kn2)
                k2 = key.copy()
                for cr in ctrl_rows:
                    kc = store[L][cr, :, h * hd:(h + 1) * hd].mean(axis=0)
                    n2 = float(kc @ kc)
                    if n2 < 1e-12:
                        continue
                    k2 = k2 - (k2 @ kc) / n2 * kc
                kn2b = float(k2 @ k2)
                if kn2b < 0.05 * kn2:
                    continue
                keep_ratios.append(float(k2 @ key) / kn2)
                key, kn2 = k2, kn2b
            delta = W[:, sl].float().cpu().numpy().astype(
                np.float64) @ key
            c = float(delta @ r_np)
            d = c * (b_np - r_np)
            backups.append((L, sl, W[:, sl].clone()))
            upd = torch.tensor(
                np.outer(d.astype(np.float32),
                         (key / kn2).astype(np.float32)),
                device=W.device, dtype=W.dtype)
            W[:, sl] = W[:, sl] + upd
        return backups, keep_ratios

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

    # ---------- arm runner ----------
    m0 = margins(forward())

    def run_heads(heads, orth=False, cols=None, beta=None):
        hb, kr = rank1(heads, orth=orth) if heads else ([], [])
        cb = col_redirect(cols) if cols else []
        if beta is not None:
            hnd = model.model.layers[0].register_forward_pre_hook(
                make_emb(beta))
            lg = forward()
            hnd.remove()
        else:
            lg = forward()
        restore(hb, cb)
        return margins(lg), kr

    g_t = torch.tensor(((b_np - r_np) / g).astype(np.float32))

    def make_emb(beta):
        def emb_hook(mod, args):
            emb = args[0]
            mask = (k_ids.to(emb.device) == ap_id)
            if bool(mask.any()):
                out = emb.clone()
                dlt = (beta * g_t).to(device=emb.device,
                                      dtype=emb.dtype)
                out[mask] = out[mask] + dlt
                return out
            return None
        return emb_hook

    m_raw, _ = run_heads(heads_raw)
    m_diff, _ = run_heads(heads_diff)
    m_orth, kr = run_heads(heads_diff, orth=True)

    def dvec(m):
        return {s: round(m[s] - m0[s], 3) for s in ENTS48}

    d_raw, d_diff, d_orth = dvec(m_raw), dvec(m_diff), dvec(m_orth)
    spill = lambda dm, skip: float(np.mean(
        [abs(dm[s]) for s in ENTS48 if s != skip]))
    sel = lambda dm: round(abs(dm['apple'])
                           / max(spill(dm, 'apple'), 1e-9), 2)
    focus = RED_ENTS + CTRL_ENTS

    # ---------- flip shock ----------
    flip_log = []
    p4 = False
    best_flip = None
    for arm_name, arm_heads, arm_orth in [('B_diff', heads_diff, False),
                                          ('C_orth', heads_diff, True)]:
        for beta in BETAS:
            m, _ = run_heads(arm_heads, orth=arm_orth, cols=cols20,
                             beta=beta)
            rec = {'arm': arm_name, 'beta': beta,
                   'apple': round(m['apple'], 3),
                   'cherry': round(m['cherry'], 3),
                   'sky': round(m['sky'], 3),
                   'banana': round(m['banana'], 3),
                   'coal': round(m['coal'], 3)}
            flip_log.append(rec)
            if m['apple'] > 0:
                p4 = True
                if best_flip is None:
                    best_flip = rec
    print('P2825 flip %s' % json.dumps(flip_log), flush=True)

    m_rest = margins(forward())
    ok_rest = max(abs(m_rest[s] - m0[s]) for s in ENTS48) < 0.1

    p1 = bool(abs(d_diff['sky']) < abs(d_raw['sky'])
              and d_diff['apple'] >= 0.5 * d_raw['apple'])
    p2 = bool(abs(d_orth['sky']) < abs(d_raw['sky'])
              and d_orth['apple'] >= 0.5 * d_raw['apple'])
    p3 = bool(min(abs(d_diff['banana']),
                  abs(d_orth['banana'])) < 1.0)

    verdict = {
        'baseline': {s: round(m0[s], 3) for s in focus},
        'cross2824': {k: round(v, 6) for k, v in xchk.items()},
        'head_selection': {
            'overlap_raw_diff': overlap,
            'top_diff': [{'L': L, 'h': h, 'diff': round(c, 3)}
                         for (L, h, c) in top_diff],
            'orth_key_keep_ratio': [round(float(x), 4)
                                    for x in kr]},
        'deltas': {
            'A_raw': {s: d_raw[s] for s in focus},
            'B_diff': {s: d_diff[s] for s in focus},
            'C_orth': {s: d_orth[s] for s in focus}},
        'selectivity_apple_over_others': {
            'A_raw': sel(d_raw), 'B_diff': sel(d_diff),
            'C_orth': sel(d_orth)},
        'spill_mean_others': {
            'A_raw': round(spill(d_raw, 'apple'), 3),
            'B_diff': round(spill(d_diff, 'apple'), 3),
            'C_orth': round(spill(d_orth, 'apple'), 3)},
        'full_delta_matrix': {
            'A_raw': d_raw, 'B_diff': d_diff, 'C_orth': d_orth},
        'flip_log': flip_log,
        'best_flip': best_flip,
        'P1_diff_isolation': p1,
        'P2_orth_isolation': p2,
        'P3_banana_fixed': p3,
        'P4_true_flip': p4,
        'restore_ok': bool(ok_rest),
        'gate': {'dW': gate_dW, 'Zeval': gate_Z},
        'load_run_seconds': round(time.monotonic() - t0, 1),
    }
    result = {'phase': 2825, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'spec48.npz',
           spec=spec.astype(np.float32))

    print('P2825 focus raw %s' % json.dumps(
        {s: d_raw[s] for s in focus}), flush=True)
    print('P2825 focus diff %s' % json.dumps(
        {s: d_diff[s] for s in focus}), flush=True)
    print('P2825 focus orth %s' % json.dumps(
        {s: d_orth[s] for s in focus}), flush=True)
    print('P2825 sel %s P1=%s P2=%s P3=%s P4=%s restore=%s'
          % (json.dumps(verdict['selectivity_apple_over_others']),
             p1, p2, p3, p4, ok_rest), flush=True)

    elapsed = time.monotonic() - t0
    cc.ledger('phase2825', elapsed)
    print('P2825 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2825 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
