"""Phase 2823 (LPF-36): OV-HEAD EDIT + SCALE-UP UNIVERSALITY.

Two tasks (user 2026-09-17 05:57): continue (2822 follow-up) + widen
domains/entities to find UNIVERSAL features of the mechanisms.

Arm O (OV head edit, colour main-carrier test): 2819 K2 registered
1152 OV heads; take top-5 late (L>=30) colour heads.  Capture a_h
(o_proj input, head slice) on apple contexts; redirect the head's
actual write delta from red to black via rank-1 update
  W_O^h += (delta' - delta) key^T / |key|^2,  key = a_h(apple)
rank-1 key = apple's a_h pattern -> entity-conditional by
construction.  Measure spill ratios kappa_s = <key, a_h(s)>/|key|^2.
  O1: d margin(apple, black-red) >= +0.5 after editing 5 heads
  O2: |d margin(sky/grass/coal)| < 0.5 (key specificity)
  O3: per-head effect > per-40-column effect (2822 V1 top20 +0.50)

Arm B (scale-up universality): 42 single-token entities x 8 domains
(size/weight/temperature/speed/hardness/taste/loudness/shape;
adj pairs big-small ... round-sharp).  Census signed cos top-5/layer
x8 directions; margin matrix ANOVA + SVD; preregistered truth cells.
  U1: leakage(d,d') < 0.30 for ALL pairs (8x8, incl colour refs)
  U2: all 8 domains' top-column layer >= 28
  U3: every domain's top-5 columns sign-mixed (antagonism universal)
  U5: truth-cell sign agreement >= 0.72 (92 preregistered cells)
  U6: size flip (2822 operator) transfers: d margin(whale) < 0 AND
      d margin(elephant) < 0 at top20/layer
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
OUT = BASE / 'phase2823' / 'ov_edit_universality'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SRC_2819 = BASE / 'phase2819' / 'knowledge_edit_locus'
SRC_2822 = BASE / 'phase2822' / 'balanced_edit'
SEED = 2823
CENSUS_L0, CENSUS_L1 = 26, 35
TOPK = 5
N_OV_HEADS = 5
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
DOMAINS = ['size', 'weight', 'temperature', 'speed', 'hardness',
           'taste', 'loudness', 'shape']
PAIR = {'size': ('big', 'small'), 'weight': ('heavy', 'light'),
        'temperature': ('hot', 'cold'), 'speed': ('fast', 'slow'),
        'hardness': ('hard', 'soft'), 'taste': ('sweet', 'bitter'),
        'loudness': ('loud', 'quiet'), 'shape': ('round', 'sharp')}
ADJ_ALL = sorted({a for p in PAIR.values() for a in p})
T2821 = {
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
T_NEW = {
    'lemon': {'size': -1, 'weight': -1, 'taste': 1},
    'banana': {'taste': 1},
    'grape': {'taste': 1},
    'bread': {'hardness': -1, 'taste': 1},
    'dog': {'speed': 1},
    'cat': {'speed': 1},
    'bee': {'size': -1, 'speed': 1, 'loudness': 1},
    'horse': {'size': 1, 'weight': 1, 'speed': 1},
    'ox': {'size': 1, 'weight': 1, 'speed': -1},
    'whale': {'size': 1, 'weight': 1},
    'shark': {'size': 1, 'speed': 1},
    'kitten': {'size': -1, 'weight': -1, 'speed': 1},
    'puppy': {'size': -1, 'weight': -1},
    'stone': {'weight': 1, 'hardness': 1},
    'brick': {'hardness': 1, 'shape': 1},
    'balloon': {'weight': -1, 'hardness': -1, 'shape': 1},
    'bubble': {'weight': -1, 'hardness': -1, 'shape': 1},
    'hammer': {'weight': 1, 'hardness': 1},
    'nail': {'hardness': 1, 'shape': 1},
    'trumpet': {'loudness': 1},
    'piano': {'loudness': 1, 'weight': 1},
    'moon': {'shape': 1},
    'river': {'size': 1},
    'apple': {'taste': 1},
    'cherry': {'taste': 1},
    'strawberry': {'size': -1, 'weight': -1, 'taste': 1},
    'tomato': {'weight': -1, 'taste': 1},
    'blood': {'temperature': 1},
}

PREREG = {
    'O1': 'ov_edit: rank-1 red->black redirect of top-5 late colour '
          'heads gives d margin(apple) >= +0.5',
    'O2': 'ov_isolation: |d margin(sky/grass/coal)| < 0.5 (rank-1 '
          'key specificity)',
    'O3': 'per-head effect of 5 heads >= per-40-column effect '
          '(2822 V1 top20 d apple +0.50)',
    'U1': 'domain_separation_8: leakage < 0.30 all pairs incl '
          'colour refs',
    'U2': 'late_stage_8: all top-column layers >= 28',
    'U3': 'antagonism_8: every domain top-5 sign-mixed',
    'U5': 'truth_8: sign agreement >= 0.72 over preregistered cells',
    'U6': 'flip_transfers: size flip top20 d margin(whale) < 0 AND '
          'd margin(elephant) < 0',
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
    res2822 = json.loads((SRC_2822 / 'result.json').read_text(
        encoding='utf-8'))
    v2822 = res2822['verdict']['condition_deltas']['V1_top20']

    execution = {
        'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
        'prereg': PREREG, 'seed': SEED,
        'census_layers': [CENSUS_L0, CENSUS_L1], 'topk': TOPK,
        'n_ov_heads': N_OV_HEADS, 'entities': ENTS48,
        'domains': DOMAINS, 'pairs': PAIR,
        'truth_new': T_NEW,
        'note': 'OV rank-1 head edit (colour main carrier) + 8-domain '
                '42-entity universality scale-up'}
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

    for w in ADJ_ALL + COLOR_WORDS + COLOURS + ['The', ' is']:
        tid(w)
    dWc = {}
    for c in ['red', 'black']:
        others = [w for w in COLOR_WORDS if w != c]
        dWc[c] = unit(zw(c) - np.stack([zw(w) for w in others]).mean(0))
    dD = {d: unit(zw(p[0]) - zw(p[1])) for d, p in PAIR.items()}

    # ---------- CUDA ----------
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(mdir), dtype=torch.bfloat16, device_map='auto')
    model.eval()
    dev = model.device
    n_heads = int(cfg.get('num_attention_heads', 32))

    ent_tok = {s: tok(' ' + s, add_special_tokens=False)['input_ids']
               for s in ENTS48}
    k_seqs = [[tid('The')] + ent_tok[s] + [tid(' is')] for s in ENTS48]
    k_ids = torch.tensor(k_seqs, dtype=torch.long)
    k_mask = torch.ones_like(k_ids)
    adj_id = {a: tid(a) for a in ADJ_ALL}
    col_id = {c: tid(c) for c in COLOURS}

    def forward():
        with torch.no_grad():
            out = model(input_ids=k_ids.to(dev),
                        attention_mask=k_mask.to(dev))
        return out.logits[:, -1, :].float().cpu().numpy()

    def readout(lg):
        # colour margins for the 10 red-ctrl entities
        red_i = [ENTS48.index(s) for s in RED_ENTS + CTRL_ENTS]
        cmar = {s: float(lg[i, col_id['black']] - lg[i, col_id['red']])
                for i, s in zip(red_i, RED_ENTS + CTRL_ENTS)}
        # 8-domain margins for all 42 entities
        M = np.stack([[lg[i, adj_id[PAIR[d][0]]]
                       - lg[i, adj_id[PAIR[d][1]]]
                       for d in DOMAINS] for i in range(len(ENTS48))])
        return cmar, M

    # ---------- Arm B: 8-domain margin matrix ----------
    cmar0, M0 = readout(forward())
    print('P2823 baseline colour %s' % json.dumps(
        {k: round(v, 2) for k, v in cmar0.items()}), flush=True)

    # truth cells (merged)
    cells = []
    for s in ENTS48:
        tab = {}
        tab.update(T2821.get(s, {}))
        tab.update(T_NEW.get(s, {}))
        for d, v in tab.items():
            if v != 0:
                cells.append((s, d, v))
    agree = [int(np.sign(M0[ENTS48.index(s), DOMAINS.index(d)]) == v)
             for (s, d, v) in cells]
    u5_rate = float(np.mean(agree))
    total_mean = M0.mean()
    row = M0.mean(axis=1, keepdims=True)
    col = M0.mean(axis=0, keepdims=True)
    resid = M0 - row - col + total_mean
    cen = M0 - total_mean
    add_share = 1.0 - float((resid ** 2).sum()) / float(
        (cen ** 2).sum())
    sv = np.linalg.svd(cen, compute_uv=False)
    print('P2823 ArmB cells %d agree %.3f add_share %.4f sv %s'
          % (len(cells), u5_rate, add_share,
             np.round(sv[:5], 2).tolist()), flush=True)

    # ---------- 8-direction column census ----------
    dirs = np.stack([dD[d] for d in DOMAINS] +
                    [dWc['red'], dWc['black']]).astype(np.float32)
    dn = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)
    names = DOMAINS + ['red_ref', 'black_ref']
    census = np.zeros((CENSUS_L1 - CENSUS_L0 + 1,
                       model.config.intermediate_size, len(names)),
                      dtype=np.float32)
    for L in range(CENSUS_L0, CENSUS_L1 + 1):
        name = 'model.layers.%d.mlp.down_proj.weight' % L
        with safe_open(str(mdir / index[name]), framework='pt') as f:
            W = f.get_tensor(name).float().numpy().astype(np.float32)
        Wn = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True),
                            1e-12)
        census[L - CENSUS_L0] = Wn.T @ dn.T
        del W, Wn
    top = {}
    for di, d in enumerate(DOMAINS):
        prof = np.abs(census[:, :, di]).max(axis=1)
        best_layer = int(np.argmax(prof)) + CENSUS_L0
        cl = census[best_layer - CENSUS_L0]
        order = np.argsort(-np.abs(cl[:, di]))[:TOPK]
        top[d] = {'layer': best_layer,
                  'cols': [int(c) for c in order],
                  'signed_cos': [float(cl[c, di]) for c in order]}
    leak = np.zeros((len(DOMAINS), len(names)))
    for di, d in enumerate(DOMAINS):
        cl = census[top[d]['layer'] - CENSUS_L0]
        cols = top[d]['cols']
        for dj in range(len(names)):
            leak[di, dj] = float(np.median(np.abs(cl[cols, dj])))
    u1 = bool(all(leak[i, j] < 0.30
                  for i in range(len(DOMAINS))
                  for j in range(len(names))
                  if not (i == j and names[j] in DOMAINS)))
    u2 = bool(all(top[d]['layer'] >= 28 for d in DOMAINS))
    u3 = bool(all(min(top[d]['signed_cos']) < 0 < max(
        top[d]['signed_cos']) for d in DOMAINS))
    print('P2823 top %s' % json.dumps(
        {d: {'L': top[d]['layer'],
             'cos': [round(c, 3) for c in top[d]['signed_cos']]}
         for d in DOMAINS}), flush=True)
    print('P2823 leak A2 %s U1=%s U2=%s U3=%s' % (
        np.round(leak, 3).tolist(), u1, u2, u3), flush=True)

    # ---------- U6: size flip transfer (2822 operator) ----------
    size_cols = []
    for L in range(CENSUS_L0, CENSUS_L1 + 1):
        cl = census[L - CENSUS_L0, :, DOMAINS.index('size')]
        order = np.argsort(-np.abs(cl))[:20]
        size_cols += [(L, int(j), float(cl[j])) for j in order]
    dd = dD['size'].astype(np.float32).astype(np.float64)
    backups = []
    for (L, j, comp) in size_cols:
        if comp <= 0:
            continue
        W = model.model.layers[L].mlp.down_proj.weight.data
        backups.append((L, j, W[:, j].clone()))
        w = W[:, j].float().cpu().numpy().astype(np.float64)
        w = w - 2.0 * comp * dd
        W[:, j] = torch.tensor(w.astype(np.float32), device=W.device)
    cmar_f, M_f = readout(forward())
    for (L, j, orig) in backups:
        model.model.layers[L].mlp.down_proj.weight.data[:, j] = orig
    i_wh = ENTS48.index('whale')
    i_el = ENTS48.index('elephant')
    d_wh = float(M_f[i_wh, 0] - M0[i_wh, 0])
    d_el = float(M_f[i_el, 0] - M0[i_el, 0])
    u6 = bool(d_wh < 0 and d_el < 0)
    print('P2823 U6 flip dwhale %.3f delephant %.3f U6=%s'
          % (d_wh, d_el, u6), flush=True)

    # ---------- Arm O: OV rank-1 head edit ----------
    ov_rows = res2819['verdict']['K2']['ov_rows']
    late = [r for r in ov_rows if r['layer'] >= 30
            and r['best_val'] > r['rand_q95']]
    late.sort(key=lambda r: -r['best_val'])
    heads = [(int(r['layer']), int(r['head']), r['best_kind'])
             for r in late[:N_OV_HEADS]]
    print('P2823 OV heads %s' % json.dumps(heads), flush=True)

    hd = None
    a_store = {}

    def cap_hook(mod, args):
        a_store['a'] = args[0].detach()
        return None

    hdL = {L: model.model.layers[L].self_attn.o_proj for (L, _, _) in
           heads}
    hs = [mod.register_forward_pre_hook(cap_hook) for mod in
          hdL.values()]
    with torch.no_grad():
        model(input_ids=k_ids.to(dev), attention_mask=k_mask.to(dev))
    for h in hs:
        h.remove()
    A = a_store['a'].float().cpu().numpy()  # (B, L, 4096)
    ap = ENTS48.index('apple')
    # apple context key: mean of entity+is positions
    key_full = A[ap, 1:3, :].mean(axis=0)  # (4096,)

    W_ref = read_tensor(
        'model.layers.%d.self_attn.o_proj.weight' % heads[0][0])
    d_model, hd = W_ref.shape[0], W_ref.shape[1] // n_heads
    del W_ref
    r_np = dWc['red'].astype(np.float32).astype(np.float64)
    b_np = dWc['black'].astype(np.float32).astype(np.float64)
    edits = []
    ov_backups = []
    for (L, h, kind) in heads:
        W = model.model.layers[L].self_attn.o_proj.weight.data
        sl = slice(h * hd, (h + 1) * hd)
        key = key_full[h * hd:(h + 1) * hd]
        kn2 = float(key @ key)
        if kn2 < 1e-12:
            continue
        delta = W[:, sl].float().cpu().numpy().astype(
            np.float64) @ key
        c = float(delta @ r_np)
        delta_new = delta - c * r_np + c * b_np
        d = delta_new - delta
        ov_backups.append((L, sl, W[:, sl].clone()))
        upd = torch.tensor(
            np.outer(d.astype(np.float32),
                     (key / kn2).astype(np.float32)),
            device=W.device, dtype=W.dtype)
        W[:, sl] = W[:, sl] + upd
        edits.append({'L': L, 'h': h, 'c_red': round(c, 4),
                      'delta_norm': round(
                          float(np.linalg.norm(delta)), 4)})
    print('P2823 OV edits %s' % json.dumps(edits), flush=True)
    cmar_o, M_o = readout(forward())
    for (L, sl, orig) in ov_backups:
        model.model.layers[L].self_attn.o_proj.weight.data[:, sl] = \
            orig
    cmar_r, _ = readout(forward())
    ok_restore = max(abs(cmar_r[s] - cmar0[s])
                     for s in RED_ENTS + CTRL_ENTS) < 0.1
    d_apple = cmar_o['apple'] - cmar0['apple']
    d_sky = abs(cmar_o['sky'] - cmar0['sky'])
    d_grass = abs(cmar_o['grass'] - cmar0['grass'])
    d_coal = abs(cmar_o['coal'] - cmar0['coal'])
    o1 = bool(d_apple >= 0.5)
    o2 = bool(d_sky < 0.5 and d_grass < 0.5 and d_coal < 0.5)
    o3 = bool(d_apple >= abs(v2822['apple']))
    # spill ratios kappa
    kappas = {}
    for s in ['sky', 'grass', 'coal', 'cherry']:
        i = ENTS48.index(s)
        ks = []
        for (L, h, kind) in heads:
            key = key_full[h * hd:(h + 1) * hd]
            a_s = A[i, 1:3, h * hd:(h + 1) * hd].mean(axis=0)
            ks.append(float(key @ a_s) / max(float(key @ key), 1e-12))
        kappas[s] = round(float(np.mean(ks)), 3)
    print('P2823 OV dapple %.3f dsky %.3f restore_ok %s O1=%s O2=%s '
          'O3=%s kappas %s'
          % (d_apple, d_sky, ok_restore, o1, o2, o3,
             json.dumps(kappas)), flush=True)

    verdict = {
        'baseline_colour': {k: round(v, 3)
                            for k, v in cmar0.items()},
        'ArmB': {'n_cells': len(cells), 'agreement': round(u5_rate, 3),
                 'additive_share': round(add_share, 4),
                 'sv_top5': [round(float(x), 2) for x in sv[:5]],
                 'leak': np.round(leak, 3).tolist(),
                 'names': names,
                 'top': {d: {'L': top[d]['layer'],
                             'cos': [round(c, 3) for c in
                                     top[d]['signed_cos']]}
                         for d in DOMAINS},
                 'U1': u1, 'U2': u2, 'U3': u3,
                 'U5_semantic': bool(u5_rate >= 0.72)},
        'U6': {'d_whale': round(d_wh, 3), 'd_elephant': round(d_el, 3),
               'flip_transfers': u6},
        'ArmO': {'heads': edits, 'd_apple': round(d_apple, 3),
                 'd_sky': round(d_sky, 3), 'd_grass': round(d_grass, 3),
                 'd_coal': round(d_coal, 3),
                 'restore_ok': bool(ok_restore),
                 'kappas': kappas,
                 'O1': o1, 'O2': o2, 'O3': o3},
        'gate': {'dW': gate_dW, 'Zeval': gate_Z},
        'load_run_seconds': round(time.monotonic() - t0, 1),
    }
    result = {'phase': 2823, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'uni.npz',
           M0=M0.astype(np.float32),
           leak=leak.astype(np.float32),
           dD=np.stack([dD[d] for d in DOMAINS]).astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2823', elapsed)
    print('P2823 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2823 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
