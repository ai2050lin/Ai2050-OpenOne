"""Phase 2827 = ALPHA (LPF-40): DOMAIN x ENTITY SPECTRUM MATRIX +
PER-DOMAIN TRIPLE EDITOR + PATHWAY DIVISION MAP.

User plan (2026-09-17 07:44): Phase Alpha of the staged grand plan.
Sub-tasks in ONE phase:
  (1) full 48-entity x 1152-head x 9-direction measured spectrum
  (2) per-domain triple (diff selection + orthogonal key + flip) edit
  (3) pathway division: which domains ride OV heads vs MLP columns
  (4) universal attribute head L29 h27 entity-conditioning check

Targets (positive pole of each pair): size->elephant, weight->elephant,
temperature->sun, speed->rocket, hardness->steel, taste->apple,
loudness->trumpet, shape->balloon.  color_red (apple) carried over
from 2824/2825 for the division map.

Arms per domain:
  head3    : diff-selected top-10 heads (c_target - mean(c_others)),
             orthogonal keys (ctrl = sky/grass/coal/banana), flip
  headplain: same heads, plain target key, flip
  cols     : per-layer top-20 columns by |proj dD_pos| (L26-35),
             flip on comp>0 columns

Prereg (frozen):
  A1: spectrum_signif: for ALL 9 domains, target-entity top-10 head
      mean c > null q95 (null = 100 random dirs x 48-entity deltas)
  A2: triple_efficacy: >=7/8 new domains have head3 d_margin(target)
      < -0.3 (correct direction, meaningful size)
  A3: orth_isolation: head3 mean|d_margin(others47)| < 0.7 x
      headplain's, in >=6/8 domains
  A4: division_structure: per-domain head3-vs-cols efficacy ratio;
      ratio_max/ratio_min > 3 across the 8 new domains
  A5: universal_head_conditioned: L29 h27 color c[apple]/c[sky] > 3
      AND >=5/9 domains have max/min c over 48 entities > 3
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
OUT = BASE / 'phase2827' / 'alpha_matrix'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SEED = 2827
N_HEADS_EDIT = 10
N_COLS_PER_LAYER = 20
CENSUS_L0 = 26
N_NULL = 100
COLOR_WORDS = ['red', 'green', 'blue', 'black', 'white', 'yellow',
               'brown', 'pink', 'gray', 'purple', 'orange', 'crimson']
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
TARGET = {'size': 'elephant', 'weight': 'elephant',
          'temperature': 'sun', 'speed': 'rocket', 'hardness': 'steel',
          'taste': 'apple', 'loudness': 'trumpet', 'shape': 'balloon'}
CTRL_ENTS = ['sky', 'grass', 'coal', 'banana']
DIR_NAMES = DOMAINS + ['color_red']

PREREG = {
    'A1': 'spectrum_signif: ALL 9 domains, target top-10 head mean c '
          '> null q95 (100 random dirs x 48-entity deltas)',
    'A2': 'triple_efficacy: >=7/8 domains head3 d_margin(target) < -0.3',
    'A3': 'orth_isolation: head3 spill < 0.7 x headplain spill in '
          '>=6/8 domains',
    'A4': 'division_structure: head3/cols efficacy ratio max/min > 3 '
          'across 8 domains',
    'A5': 'universal_head_conditioned: c29h27color[apple]/c[sky] > 3 '
          'AND >=5/9 domains entity max/min > 3',
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
        'n_heads_edit': N_HEADS_EDIT, 'n_cols_per_layer':
            N_COLS_PER_LAYER, 'census_l0': CENSUS_L0,
        'targets': TARGET, 'ctrl_ents': CTRL_ENTS,
        'dir_names': DIR_NAMES, 'pairs': PAIR, 'entities': ENTS48,
        'note': 'ALPHA: full domain x entity spectrum matrix; per-domain '
                'triple editor; pathway division map; universal head '
                'conditioning check'}
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

    for w in COLOR_WORDS + ADJ_ALL + ['The', ' is']:
        tid(w)
    others_red = [w for w in COLOR_WORDS if w != 'red']
    d_red = unit(zw('red') - np.stack(
        [zw(w) for w in others_red]).mean(0))
    dD = {d: unit(zw(p[0]) - zw(p[1])) for d, p in PAIR.items()}
    dirs = [dD[d] for d in DOMAINS] + [d_red]
    DMAT = np.stack(dirs, axis=1)  # (2560, 9)

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
    adj_id = {a: tid(a) for a in ADJ_ALL}

    def forward():
        with torch.no_grad():
            out = model(input_ids=k_ids.to(dev),
                        attention_mask=k_mask.to(dev))
        return out.logits[:, -1, :].float().cpu().numpy()

    # ---------- capture all 48 entities ----------
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
    print('P2827 captured %d layers' % len(store), flush=True)

    # ---------- full spectrum spec[L, e, h, d9] ----------
    Wo0 = read_tensor('model.layers.0.self_attn.o_proj.weight')
    d_model, hd_total = Wo0.shape
    hd = hd_total // n_heads
    del Wo0
    ne = len(ENTS48)
    spec = np.zeros((n_layers, ne, n_heads, 9), dtype=np.float32)
    for L in range(n_layers):
        W = read_tensor('model.layers.%d.self_attn.o_proj.weight' % L)
        W3 = W.astype(np.float64).reshape(d_model, n_heads, hd)
        K3 = store[L].mean(axis=1).reshape(ne, n_heads, hd)
        delta = np.einsum('dhk,ehk->edh', W3, K3)  # (ne, 2560, 32)
        spec[L] = np.einsum('edh,dr->ehr', delta,
                            DMAT).astype(np.float32)  # (ne, 32, 9)
        del W, W3, delta
    print('P2827 spec done %s' % (spec.shape,), flush=True)

    # ---------- random null (48-entity deltas x 100 dirs) ----------
    rng = np.random.default_rng(SEED)
    U = rng.standard_normal((N_NULL, d_model))
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    null_parts = []
    for L in range(n_layers):
        W = read_tensor('model.layers.%d.self_attn.o_proj.weight' % L)
        W3 = W.astype(np.float64).reshape(d_model, n_heads, hd)
        K3 = store[L].mean(axis=1).reshape(ne, n_heads, hd)
        delta = np.einsum('dhk,ehk->edh', W3, K3)
        c = np.einsum('edh,dr->ehr', delta, U.T.astype(np.float64))
        null_parts.append(c.ravel())
        del W, W3, delta
    null_pool = np.concatenate(null_parts)
    null_q95 = float(np.quantile(null_pool, 0.95))
    print('P2827 null q95 %.4f' % null_q95, flush=True)

    # ---------- A5: universal head L29 h27 ----------
    c29 = spec[29, :, 27, :].astype(np.float64)  # (48, 9)
    univ = {}
    for di, name in enumerate(DIR_NAMES):
        col = c29[:, di]
        univ[name] = {
            'apple': round(float(col[ENTS48.index('apple')]), 3),
            'sky': round(float(col[ENTS48.index('sky')]), 3),
            'max_min': round(float(col.max()
                                   / max(abs(col.min()), 1e-9)), 2)}
    a5_ratio = float(c29[ENTS48.index('apple'), 8]
                     / max(abs(c29[ENTS48.index('sky'), 8]), 1e-9))
    n_cond = int(sum(1 for name in DIR_NAMES
                     if univ[name]['max_min'] > 3))
    a5 = bool(a5_ratio > 3 and n_cond >= 5)
    print('P2827 univ %s ratio %.2f ncond %d' % (
        json.dumps({k: v['apple'] for k, v in univ.items()}),
        a5_ratio, n_cond), flush=True)

    # ---------- A1: target top-10 significance ----------
    a1 = {}
    for name in DOMAINS:
        di = DIR_NAMES.index(name)
        e = ENTS48.index(TARGET[name])
        col = spec[:, e, :, di].astype(np.float64)
        top = sorted([float(col[L, h]) for L in range(n_layers)
                      for h in range(n_heads)], reverse=True)[:10]
        a1[name] = bool(np.mean(top) > null_q95)
    col_red = spec[:, ENTS48.index('apple'), :, 8].astype(np.float64)
    top_red = sorted([float(col_red[L, h]) for L in range(n_layers)
                      for h in range(n_heads)], reverse=True)[:10]
    a1['color_red'] = bool(np.mean(top_red) > null_q95)
    p_a1 = bool(all(a1.values()))
    print('P2827 A1 %s' % json.dumps(a1), flush=True)

    # ---------- baseline margins ----------
    def margin_fn(dom):
        p, n = PAIR[dom]
        return lambda lg: {s: float(lg[i, adj_id[p]]
                                    - lg[i, adj_id[n]])
                           for i, s in enumerate(ENTS48)}

    m_base_all = forward()
    m_base = {dom: margin_fn(dom)(m_base_all) for dom in DOMAINS}
    print('P2827 baseline target margins %s' % json.dumps(
        {dom: round(m_base[dom][TARGET[dom]], 2)
         for dom in DOMAINS}), flush=True)

    # ---------- per-domain editors ----------
    ctrl_rows = [ENTS48.index(s) for s in CTRL_ENTS]
    arm_results = {}

    def diff_heads(dom):
        di = DIR_NAMES.index(dom)
        e = ENTS48.index(TARGET[dom])
        col = spec[:, e, :, di].astype(np.float64)
        others = [spec[:, x, :, di].astype(np.float64)
                  for x in range(ne) if x != e]
        diff = col - np.mean(others, axis=0)
        top = sorted([(int(L), int(h), float(diff[L, h]))
                      for L in range(n_layers)
                      for h in range(n_heads) if col[L, h] > 0],
                     key=lambda t: -t[2])[:N_HEADS_EDIT]
        return [(L, h) for (L, h, _) in top]

    def rank1_flip(dom, heads, orth):
        dpos = dD[dom]
        dneg = -dD[dom]
        backups = []
        for (L, h) in heads:
            W = model.model.layers[L].self_attn.o_proj.weight.data
            sl = slice(h * hd, (h + 1) * hd)
            e = ENTS48.index(TARGET[dom])
            key = store[L][e, :, h * hd:(h + 1) * hd].mean(axis=0)
            kn2 = float(key @ key)
            if kn2 < 1e-12:
                continue
            if orth:
                k2 = key.copy()
                for cr in ctrl_rows:
                    kc = store[L][cr, :, h * hd:(h + 1) * hd].mean(
                        axis=0)
                    n2 = float(kc @ kc)
                    if n2 < 1e-12:
                        continue
                    k2 = k2 - (k2 @ kc) / n2 * kc
                kn2b = float(k2 @ k2)
                if kn2b < 0.05 * kn2:
                    continue
                key, kn2 = k2, kn2b
            delta = W[:, sl].float().cpu().numpy().astype(
                np.float64) @ key
            c = float(delta @ dpos)
            d = -2.0 * c * dpos
            backups.append((L, sl, W[:, sl].clone()))
            upd = torch.tensor(
                np.outer(d.astype(np.float32),
                         (key / kn2).astype(np.float32)),
                device=W.device, dtype=W.dtype)
            W[:, sl] = W[:, sl] + upd
        return backups

    def col_flip(dom):
        dpos = dD[dom].astype(np.float32)
        backups = []
        for L in range(CENSUS_L0, n_layers):
            name = 'model.layers.%d.mlp.down_proj.weight' % L
            with safe_open(str(mdir / index[name]), framework='pt') as f:
                W = f.get_tensor(name).float().numpy().astype(np.float32)
            Wn = W / np.maximum(np.linalg.norm(W, axis=0,
                                               keepdims=True), 1e-12)
            cp = Wn.T @ dpos
            order = np.argsort(-np.abs(cp))[:N_COLS_PER_LAYER]
            for j in order:
                comp = float(cp[j])
                if comp <= 0:
                    continue
                Wl = model.model.layers[L].mlp.down_proj.weight.data
                backups.append((L, int(j), Wl[:, int(j)].clone()))
                w = Wl[:, int(j)].float().cpu().numpy().astype(
                    np.float64)
                w = w - 2.0 * comp * dpos.astype(np.float64)
                Wl[:, int(j)] = torch.tensor(
                    w.astype(np.float32), device=Wl.device)
            del W, Wn
        return backups

    def restore(heads_bk, cols_bk):
        for (L, sl, orig) in heads_bk:
            model.model.layers[L].self_attn.o_proj.weight.data[:, sl] \
                = orig
        for (L, j, orig) in cols_bk:
            model.model.layers[L].mlp.down_proj.weight.data[:, j] = \
                orig

    for dom in DOMAINS:
        heads = diff_heads(dom)
        eff = {}
        # head3
        hb = rank1_flip(dom, heads, orth=True)
        lg = forward()
        restore(hb, [])
        m = margin_fn(dom)(lg)
        eff['head3'] = {s: m[s] - m_base[dom][s] for s in ENTS48}
        # headplain
        hb = rank1_flip(dom, heads, orth=False)
        lg = forward()
        restore(hb, [])
        m = margin_fn(dom)(lg)
        eff['headplain'] = {s: m[s] - m_base[dom][s] for s in ENTS48}
        # cols
        cb = col_flip(dom)
        lg = forward()
        restore([], cb)
        m = margin_fn(dom)(lg)
        eff['cols'] = {s: m[s] - m_base[dom][s] for s in ENTS48}
        arm_results[dom] = eff
        tgt = TARGET[dom]
        spill3 = float(np.mean([abs(v) for k, v in
                                eff['head3'].items() if k != tgt]))
        spillp = float(np.mean([abs(v) for k, v in
                                eff['headplain'].items() if k != tgt]))
        print('P2827 %s tgt=%s head3 %.3f headplain %.3f cols %.3f '
              'spill3 %.3f spillp %.3f' % (
                  dom, tgt, eff['head3'][tgt], eff['headplain'][tgt],
                  eff['cols'][tgt], spill3, spillp), flush=True)

    m_rest_all = forward()
    ok_rest = True
    for dom in DOMAINS:
        fn = margin_fn(dom)
        mr = fn(m_rest_all)
        if max(abs(mr[s] - m_base[dom][s]) for s in ENTS48) >= 0.1:
            ok_rest = False
    print('P2827 restore_ok %s' % ok_rest, flush=True)

    # ---------- verdicts ----------
    a2 = {}
    for dom in DOMAINS:
        tgt = TARGET[dom]
        a2[dom] = bool(arm_results[dom]['head3'][tgt] < -0.3)
    p_a2 = bool(sum(a2.values()) >= 7)
    a3 = {}
    for dom in DOMAINS:
        tgt = TARGET[dom]
        s3 = float(np.mean([abs(v) for k, v in
                            arm_results[dom]['head3'].items()
                            if k != tgt]))
        sp = float(np.mean([abs(v) for k, v in
                            arm_results[dom]['headplain'].items()
                            if k != tgt]))
        a3[dom] = bool(s3 < 0.7 * max(sp, 1e-9))
    p_a3 = bool(sum(a3.values()) >= 6)
    ratios = {}
    for dom in DOMAINS:
        tgt = TARGET[dom]
        eh = abs(arm_results[dom]['head3'][tgt])
        ec = abs(arm_results[dom]['cols'][tgt])
        ratios[dom] = round(eh / max(ec, 1e-9), 3)
    r_vals = [v for v in ratios.values() if v > 0]
    p_a4 = bool(max(r_vals) / max(min(r_vals), 1e-9) > 3)

    verdict = {
        'null_q95': round(null_q95, 4),
        'A1_signif': a1,
        'baseline_target_margins': {
            dom: round(m_base[dom][TARGET[dom]], 2)
            for dom in DOMAINS},
        'arms_target_delta': {
            dom: {arm: round(arm_results[dom][arm][TARGET[dom]], 3)
                  for arm in ['head3', 'headplain', 'cols']}
            for dom in DOMAINS},
        'arms_spill': {
            dom: {arm: round(float(np.mean(
                [abs(v) for k, v in arm_results[dom][arm].items()
                 if k != TARGET[dom]])), 3)
                for arm in ['head3', 'headplain', 'cols']}
            for dom in DOMAINS},
        'head_col_ratio': ratios,
        'universal_head_L29h27': univ,
        'universal_head_ratio': round(a5_ratio, 2),
        'universal_head_n_cond': n_cond,
        'A1_pass': p_a1, 'A2_pass': p_a2,
        'A3_pass': p_a3, 'A4_pass': p_a4, 'A5_pass': a5,
        'A2_per_domain': a2, 'A3_per_domain': a3,
        'restore_ok': bool(ok_rest),
        'gate': {'dW': gate_dW, 'Zeval': gate_Z},
        'load_run_seconds': round(time.monotonic() - t0, 1),
    }
    result = {'phase': 2827, 'plan': 'ALPHA', 'prereg': PREREG,
              'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'spec_full.npz',
           spec=spec)  # (36, 48, 32, 9) float32
    print('P2827 VERDICT %s' % json.dumps(verdict), flush=True)
    elapsed = time.monotonic() - t0
    cc.ledger('phase2827', elapsed)
    print('P2827 elapsed %.1fs A1=%s A2=%s A3=%s A4=%s A5=%s'
          % (elapsed, p_a1, p_a2, p_a3, p_a4, a5), flush=True)


if __name__ == '__main__':
    main()
