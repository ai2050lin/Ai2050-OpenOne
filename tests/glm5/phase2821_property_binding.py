"""Phase 2821 (LPF-34): PROPERTY-BINDING GLOBAL MECHANISM.

User directive (2026-09-17 05:26): find the WHOLE-system mechanism
for object attributes (colour / size / weight / ...) — how entities x
properties compute, not one property at a time.  Goal: the law that
lets few parameters yield infinite combinations.

Core hypothesis: the attribute system is FACTORED COMPUTATION.
Margin matrix M[s, d] (entity s, domain d) should decompose
additively: M = alpha_s + beta_d + eps.  If true, N entities x D
domains need N+D factors, not N*D storage slots — the quantitative
mechanism of "few parameters, infinite combinations".

Arm A (additivity): M (20 entities x 5 domains: size/weight/
temperature/speed/hardness; adj-pair margins big-small / heavy-light
/ hot-cold / fast-slow / hard-soft at "The {s} is").
  A0 semantic_truth: sign agreement >= 0.80 on preregistered
     confident cells (50 cells, table frozen below)
  A1 additive_binding: two-way ANOVA additive share
     1 - ||M - row - col + total||^2 / ||M - total||^2 >= 0.80
     AND > q95 of 200 column-wise row-permutation nulls
Arm B (domain census): dD_d for 5 domains (+2819-protocol colour ref
  red-vs-11); cos of every down_proj column L26..35 with dD_d
  (SIGNED, 2820 antagonism lesson); per-domain top-5 cols per layer.
  A2 domain_separation: leakage(d->d') = median |cos| of domain d's
     top-5 cols (best layer) against dD_d' < 0.30 for ALL d != d'
  A3 late_stage: mean layer of each domain's top-5 columns >= 28
Arm C (cross-domain causal replication of 2820):
  B1 shared_write_path_size: mlp-edit big->small on size top-4 cols
     gives spillover r_mlp = |dM(cherry)|/|dM(elephant)| >= 0.5
  B2 private_entity_row_size: emb-edit elephant big->small beta=2
     leaves |dM| < 0.1 for all other 19 entities (size margin)

Prereg frozen before any readout.  Entities: all single-token
(probe_2821_tok); adjectives all single-token.
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
OUT = BASE / 'phase2821' / 'property_binding'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SRC_2819 = BASE / 'phase2819' / 'knowledge_edit_locus'
SEED = 2821
N_PERM = 200
CENSUS_L0, CENSUS_L1 = 26, 35
TOPK = 5

ENTS = ['elephant', 'mouse', 'lion', 'cherry', 'grape', 'banana',
        'rock', 'feather', 'pillow', 'sponge', 'steel', 'ice', 'sun',
        'car', 'plane', 'rocket', 'cloud', 'tree', 'mountain', 'cup']
DOMAINS = ['size', 'weight', 'temperature', 'speed', 'hardness']
PAIR = {'size': ('big', 'small'), 'weight': ('heavy', 'light'),
        'temperature': ('hot', 'cold'), 'speed': ('fast', 'slow'),
        'hardness': ('hard', 'soft')}
ADJ_ALL = sorted({a for p in PAIR.values() for a in p})
# preregistered ground truth: +1 positive, -1 negative, 0 ambiguous
TRUTH = {
    'elephant': {'size': 1, 'weight': 1, 'temperature': 0, 'speed': -1,
                 'hardness': 0},
    'mouse': {'size': -1, 'weight': -1, 'temperature': 0, 'speed': 1,
              'hardness': 0},
    'lion': {'size': 1, 'weight': 1, 'temperature': 0, 'speed': 1,
             'hardness': 0},
    'cherry': {'size': -1, 'weight': -1, 'temperature': 0, 'speed': 0,
               'hardness': 0},
    'grape': {'size': -1, 'weight': -1, 'temperature': 0, 'speed': 0,
              'hardness': 0},
    'banana': {'size': -1, 'weight': -1, 'temperature': 0, 'speed': 0,
               'hardness': -1},
    'rock': {'size': 0, 'weight': 1, 'temperature': 0, 'speed': -1,
             'hardness': 1},
    'feather': {'size': -1, 'weight': -1, 'temperature': 0, 'speed': 0,
                'hardness': -1},
    'pillow': {'size': 0, 'weight': 0, 'temperature': 0, 'speed': 0,
               'hardness': -1},
    'sponge': {'size': 0, 'weight': -1, 'temperature': 0, 'speed': 0,
               'hardness': -1},
    'steel': {'size': 0, 'weight': 1, 'temperature': 0, 'speed': 0,
              'hardness': 1},
    'ice': {'size': 0, 'weight': 0, 'temperature': -1, 'speed': 0,
            'hardness': 1},
    'sun': {'size': 1, 'weight': 0, 'temperature': 1, 'speed': 0,
            'hardness': 0},
    'car': {'size': 0, 'weight': 1, 'temperature': 0, 'speed': 1,
            'hardness': 0},
    'plane': {'size': 1, 'weight': 1, 'temperature': 0, 'speed': 1,
              'hardness': 0},
    'rocket': {'size': 1, 'weight': 1, 'temperature': 0, 'speed': 1,
               'hardness': 0},
    'cloud': {'size': 0, 'weight': 0, 'temperature': 0, 'speed': -1,
              'hardness': -1},
    'tree': {'size': 1, 'weight': 0, 'temperature': 0, 'speed': -1,
             'hardness': 0},
    'mountain': {'size': 1, 'weight': 1, 'temperature': 0, 'speed': -1,
                 'hardness': 1},
    'cup': {'size': -1, 'weight': -1, 'temperature': 0, 'speed': 0,
            'hardness': 1},
}
COLOR_WORDS = ['red', 'green', 'blue', 'black', 'white', 'yellow',
               'brown', 'pink', 'gray', 'purple', 'orange', 'crimson']

PREREG = {
    'A0': 'semantic_truth: sign agreement with TRUTH table '
          '>= 0.80 over confident (nonzero) cells',
    'A1': 'additive_binding: two-way ANOVA additive share '
          '1 - ||resid||^2/||M_centered||^2 >= 0.80 AND > '
          'q95(200 column-wise row-permutation nulls)',
    'A2': 'domain_separation: leakage(d->d\') = median |cos| of '
          "domain d's top-5 columns (its own best layer) with dD_d' "
          '< 0.30 for ALL d != d\'',
    'A3': 'late_stage: mean layer of each domain top-5 columns >= 28',
    'B1': 'shared_write_path_size: r_mlp = |dM(cherry)|/|dM(elephant)| '
          '>= 0.5 after size-col redirect big->small',
    'B2': 'private_entity_row_size: emb-edit elephant big->small b=2 '
          'leaves max |dM(other 19, size)| < 0.1',
    'verdict': 'property system = factored computation iff A0/A1 hold '
               'and B1/B2 replicate 2820; domain census A2/A3 maps '
               'the shared write hardware',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def additive_share(M):
    """Two-way ANOVA additive share of a (n, k) matrix."""
    total = M.mean()
    row = M.mean(axis=1, keepdims=True)
    col = M.mean(axis=0, keepdims=True)
    resid = M - row - col + total
    cen = M - total
    denom = float((cen ** 2).sum())
    if denom < 1e-30:
        return 0.0
    return 1.0 - float((resid ** 2).sum()) / denom


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
        'prereg': PREREG, 'seed': SEED, 'n_perm': N_PERM,
        'census_layers': [CENSUS_L0, CENSUS_L1], 'topk': TOPK,
        'entities': ENTS, 'domains': DOMAINS, 'pairs': PAIR,
        'truth_table': TRUTH, 'color_words': COLOR_WORDS,
        'note': 'property-binding global mechanism: additive '
                'factorisation of entity x domain margin matrix; '
                'domain census of down_proj columns L26-35; '
                'cross-domain causal replication of 2820 edits'}
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
    rng = np.random.default_rng(SEED)

    def zw(t):
        e = Etab[tid(t)].astype(np.float64)
        return e / np.sqrt((e ** 2).mean() + eps) * g

    for w in ADJ_ALL + COLOR_WORDS + ['The', ' is']:
        tid(w)

    # domain directions: unit(z(adj+) - z(adj-)); colour ref = 2819
    dD = {d: unit(zw(p[0]) - zw(p[1])) for d, p in PAIR.items()}
    dWc = {}
    for c in ['red', 'blue']:
        others = [w for w in COLOR_WORDS if w != c]
        dWc[c] = unit(zw(c) - np.stack([zw(w) for w in others]).mean(0))

    # ---------- CUDA ----------
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(mdir), dtype=torch.bfloat16, device_map='auto')
    model.eval()
    dev = model.device

    ent_tok = {s: tok(' ' + s, add_special_tokens=False)['input_ids']
               for s in ENTS}
    k_seqs = [[tid('The')] + ent_tok[s] + [tid(' is')] for s in ENTS]
    assert all(len(s) == 3 for s in k_seqs)
    k_ids = torch.tensor(k_seqs, dtype=torch.long)
    k_mask = torch.ones_like(k_ids)
    adj_id = {a: tid(a) for a in ADJ_ALL}

    def run_batch():
        with torch.no_grad():
            out = model(input_ids=k_ids.to(dev),
                        attention_mask=k_mask.to(dev))
        lg = out.logits[:, -1, :].float().cpu().numpy()
        return np.stack([[lg[i, adj_id[PAIR[d][0]]]
                          - lg[i, adj_id[PAIR[d][1]]]
                          for d in DOMAINS] for i in range(len(ENTS))])

    def emb_edit(ent, d, beta):
        """Shift entity row toward the domain's negative pole
        (unit(zw(adj-) - zw(adj+)) = -dD[d] direction)."""
        e = Etab[tid(ent)].astype(np.float64)
        s0 = np.sqrt((e ** 2).mean() + eps)
        step = unit(zw(PAIR[d][1]) - zw(PAIR[d][0]))
        v = torch.tensor(
            (e + beta * s0 * step / g)
            .astype(np.float32), device=dev, dtype=next(
                model.model.embed_tokens.parameters()).dtype)

        def hook(mod, args, output):
            out = output.clone()
            out[k_ids.to(dev) == tid(ent)] = v
            return out
        h = model.model.embed_tokens.register_forward_hook(hook)
        with torch.no_grad():
            o = model(input_ids=k_ids.to(dev),
                      attention_mask=k_mask.to(dev))
        h.remove()
        lg = o.logits[:, -1, :].float().cpu().numpy()
        return np.stack([[lg[i, adj_id[PAIR[d][0]]]
                          - lg[i, adj_id[PAIR[d][1]]]
                          for d in DOMAINS] for i in range(len(ENTS))])

    def mlp_flip(cols, d):
        """Flip each column's write component along dD[d]:
        w' = w - 2 (w.dD) dD  (big-write becomes small-write)."""
        backups = []
        dd = dD[d].astype(np.float32).astype(np.float64)
        for (L, j) in cols:
            W = model.model.layers[L].mlp.down_proj.weight.data
            backups.append((L, j, W[:, j].clone()))
            w = W[:, j].float().cpu().numpy().astype(np.float64)
            comp = float(w.astype(np.float32) @ dd.astype(np.float32))
            w_new = w - 2.0 * comp * dd
            W[:, j] = torch.tensor(w_new.astype(np.float32),
                                   device=W.device)
        with torch.no_grad():
            o = model(input_ids=k_ids.to(dev),
                      attention_mask=k_mask.to(dev))
        for (L, j, orig) in backups:
            model.model.layers[L].mlp.down_proj.weight.data[:, j] = orig
        lg = o.logits[:, -1, :].float().cpu().numpy()
        return np.stack([[lg[i, adj_id[PAIR[d][0]]]
                          - lg[i, adj_id[PAIR[d][1]]]
                          for d in DOMAINS] for i in range(len(ENTS))])

    # ===== Arm A: margin matrix + additivity =====
    M0 = run_batch()
    a0_cells = [(s, d) for s in ENTS for d in DOMAINS
                if TRUTH[s][d] != 0]
    agree = [int(np.sign(M0[i, DOMAINS.index(d)]) == TRUTH[s][d])
             for (s, d) in a0_cells for i in [ENTS.index(s)]]
    p_a0 = float(np.mean(agree))
    n_cells = len(a0_cells)
    share0 = additive_share(M0)
    nulls = np.empty(N_PERM)
    for k in range(N_PERM):
        Mp = M0.copy()
        for j in range(Mp.shape[1]):
            Mp[:, j] = Mp[rng.permutation(len(ENTS)), j]
        nulls[k] = additive_share(Mp)
    q95 = float(np.quantile(nulls, 0.95))
    p_a1 = bool(share0 >= 0.80 and share0 > q95)
    sv = np.linalg.svd(M0 - M0.mean(), compute_uv=False)
    print('P2821 Arm A A0=%.3f (%d cells) share=%.4f q95=%.4f '
          'A1=%s sv=%s' % (p_a0, n_cells, share0, q95, p_a1,
                           np.round(sv[:4], 2).tolist()), flush=True)

    # ===== Arm B: domain census of down_proj columns =====
    dirs = np.stack([dD[d] for d in DOMAINS] +
                    [dWc['red'], dWc['blue']]).astype(np.float32)
    dn = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    names = DOMAINS + ['red_ref', 'blue_ref']
    L = CENSUS_L0
    census = np.zeros((CENSUS_L1 - CENSUS_L0 + 1,
                       model.config.intermediate_size
                       if hasattr(model.config, 'intermediate_size')
                       else 9728, len(names)), dtype=np.float32)
    for L in range(CENSUS_L0, CENSUS_L1 + 1):
        name = 'model.layers.%d.mlp.down_proj.weight' % L
        with safe_open(str(mdir / index[name]), framework='pt') as f:
            W = f.get_tensor(name).float().numpy().astype(np.float32)
        Wn = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True),
                            1e-12)
        census[L - CENSUS_L0] = Wn.T @ dn.T
        del W, Wn
        print('P2821 census L%d done' % L, flush=True)
    # per-domain top-5 columns at its own best layer
    top = {}
    for di, d in enumerate(DOMAINS):
        prof = np.abs(census[:, :, di]).max(axis=1)
        best_layer = int(np.argmax(prof)) + CENSUS_L0
        cl = census[best_layer - CENSUS_L0]
        order = np.argsort(-np.abs(cl[:, di]))[:TOPK]
        top[d] = {'layer': best_layer,
                  'cols': [int(c) for c in order],
                  'signed_cos': [float(cl[c, di]) for c in order]}
    # leakage matrix: domain d top-5 cols vs other domains
    leak = np.zeros((len(DOMAINS), len(names)))
    for di, d in enumerate(DOMAINS):
        cl = census[top[d]['layer'] - CENSUS_L0]
        cols = top[d]['cols']
        for dj in range(len(names)):
            leak[di, dj] = float(np.median(
                np.abs(cl[cols, dj])))
    p_a2 = bool(all(leak[i, j] < 0.30
                    for i in range(len(DOMAINS))
                    for j in range(len(names))
                    if i != j or names[j] not in DOMAINS))
    p_a3 = bool(np.mean([top[d]['layer'] for d in DOMAINS]) >= 28)
    print('P2821 Arm B top %s' % json.dumps(
        {d: {'L': top[d]['layer'], 'cols': top[d]['cols'],
             'cos': [round(c, 3) for c in top[d]['signed_cos']]}
         for d in DOMAINS}), flush=True)
    print('P2821 Arm B leak %s A2=%s A3=%s' % (
        np.round(leak, 3).tolist(), p_a2, p_a3), flush=True)

    # ===== Arm C: cross-domain edit replication (size) =====
    size_cols = [(top['size']['layer'], c) for c in top['size']['cols']]
    ML = mlp_flip(size_cols, 'size')
    dM_mlp = ML[:, DOMAINS.index('size')] - M0[:, DOMAINS.index('size')]
    i_e, i_c = ENTS.index('elephant'), ENTS.index('cherry')
    r_mlp = (float(abs(dM_mlp[i_c]) / max(abs(dM_mlp[i_e]), 1e-9))
             if abs(dM_mlp[i_e]) > 1e-9 else 0.0)
    p_b1 = bool(r_mlp >= 0.5)
    ME = emb_edit('elephant', 'size', 2.0)
    dM_emb = ME[:, DOMAINS.index('size')] - M0[:, DOMAINS.index('size')]
    others = [abs(dM_emb[i]) for i in range(len(ENTS)) if i != i_e]
    p_b2 = bool(max(others) < 0.1)
    print('P2821 Arm C mlp dM %s emb dM %s B1=%s B2=%s' % (
        np.round(dM_mlp, 3).tolist(), np.round(dM_emb, 3).tolist(),
        p_b1, p_b2), flush=True)

    verdict = {
        'M0': {s: {d: round(float(M0[i, j]), 3)
                   for j, d in enumerate(DOMAINS)}
               for i, s in enumerate(ENTS)},
        'A0': {'agreement': round(p_a0, 3), 'n_cells': n_cells,
               'semantic_truth': bool(p_a0 >= 0.80)},
        'A1': {'additive_share': round(share0, 4), 'null_q95': round(
            q95, 4), 'additive_binding': p_a1,
            'sv_top4': [round(float(x), 2) for x in sv[:4]]},
        'A2': {'leak': np.round(leak, 3).tolist(), 'names': names,
               'domain_separation': p_a2},
        'A3': {'top_layers': {d: top[d]['layer'] for d in DOMAINS},
               'late_stage': p_a3},
        'top_cols': top,
        'B1': {'dM_mlp_size': np.round(dM_mlp, 3).tolist(),
               'r_mlp': round(r_mlp, 3),
               'shared_write_path': p_b1},
        'B2': {'dM_emb_size': np.round(dM_emb, 3).tolist(),
               'max_other': round(float(max(others)), 3),
               'private_row': p_b2},
        'gate': {'dW': gate_dW, 'Zeval': gate_Z},
        'load_run_seconds': round(time.monotonic() - t0, 1),
    }
    result = {'phase': 2821, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'binding.npz',
           M0=M0.astype(np.float32),
           census_absmax=np.abs(census).max(axis=1).astype(np.float32),
           leak=leak.astype(np.float32),
           dD=np.stack([dD[d] for d in DOMAINS]).astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2821', elapsed)
    print('P2821 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2821 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
