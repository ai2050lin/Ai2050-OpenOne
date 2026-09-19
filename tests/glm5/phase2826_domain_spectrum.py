"""Phase 2826 (LPF-39): DOMAIN x HEAD FUNCTIONAL MAP (1152 x 9).

User goal (2026-09-17 07:16 "hao de, jixu"): continue.  2824/2825 built
the apple->red head circuit and achieved entity-conditioned true flip.
Open question: is the write-head hardware colour-specific or shared
across attribute domains?  This phase projects the SAME captured OV
writes (apple context, all 1152 heads) onto 9 directions -- 8 attribute
domains (2823 word pairs) + the red colour direction -- producing the
head-level functional map, with random-direction null controls
(2809 discipline) and a cross-domain behavioural edit (size).

Deliverables:
  spec9[L, h, dom]  : measured write spectrum, 9 directions
  null q95          : random-direction projection pool (100 dirs)
  Jaccard matrix    : top-20 head sets per domain (specialisation?)
  L35 h0 profile    : universal write head or colour-specific?
  size head edit    : elephant-context rank-1 big->small on size top-10

Prereg (frozen):
  P1: head_signif: for ALL 9 domains, mean(c of top-20 heads) >
      q95 of the random-direction null pool (same delta set)
  P2: late_layers: for ALL 9 domains, positive write mass
      (sum of max(c,0)) with L>=30 exceeds 0.5 of total
  P3: specialisation: mean pairwise Jaccard of domain top-20 head
      sets < 0.35
  P4: universal_write_head: L35 h0 has c > 0 for ALL 9 domains
  P5: size_head_edit: elephant-context top-10 size-head rank-1
      big->small moves d_margin_size(elephant) < -0.5
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
OUT = BASE / 'phase2826' / 'domain_spectrum'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SEED = 2826
N_HEADS_EDIT = 20
N_EDIT_SIZE = 10
N_NULL = 100
COLOR_WORDS = ['red', 'green', 'blue', 'black', 'white', 'yellow',
               'brown', 'pink', 'gray', 'purple', 'orange', 'crimson']
COLOURS = ['red', 'black', 'purple', 'blue', 'green', 'yellow']
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
    'P1': 'head_signif: for ALL 9 domains, mean(c of top-20 heads) > '
          'q95 of random-direction null pool',
    'P2': 'late_layers: for ALL 9 domains, positive write mass with '
          'L>=30 exceeds 0.5 of total',
    'P3': 'specialisation: mean pairwise Jaccard of domain top-20 '
          'head sets < 0.35',
    'P4': 'universal_write_head: L35 h0 has c > 0 for ALL 9 domains',
    'P5': 'size_head_edit: elephant-context top-10 size-head rank-1 '
          'big->small moves d_margin_size(elephant) < -0.5',
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
        'n_heads_edit': N_HEADS_EDIT, 'n_edit_size': N_EDIT_SIZE,
        'n_null': N_NULL, 'dir_names': DIR_NAMES, 'pairs': PAIR,
        'entities': ENTS48,
        'note': '9-direction head functional map (apple ctx) with '
                'random null; cross-domain size head edit'}
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

    for w in COLOR_WORDS + COLOURS + ADJ_ALL + ['The', ' is']:
        tid(w)
    dWc = {}
    for c in ['red', 'black']:
        others = [w for w in COLOR_WORDS if w != c]
        dWc[c] = unit(zw(c) - np.stack([zw(w) for w in others]).mean(0))
    dD = {d: unit(zw(p[0]) - zw(p[1])) for d, p in PAIR.items()}
    dirs = [dD[d] for d in DOMAINS] + [dWc['red']]

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
    ap_row = ENTS48.index('apple')
    el_row = ENTS48.index('elephant')

    def forward():
        with torch.no_grad():
            out = model(input_ids=k_ids.to(dev),
                        attention_mask=k_mask.to(dev))
        return out.logits[:, -1, :].float().cpu().numpy()

    # ---------- capture + spectrum ----------
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
    print('P2826 captured %d layers' % len(store), flush=True)

    Wo0 = read_tensor('model.layers.0.self_attn.o_proj.weight')
    d_model, hd_total = Wo0.shape
    hd = hd_total // n_heads
    del Wo0
    spec9 = np.zeros((n_layers, n_heads, len(DIR_NAMES)),
                     dtype=np.float64)
    for L in range(n_layers):
        W = read_tensor('model.layers.%d.self_attn.o_proj.weight' % L)
        W3 = W.astype(np.float64).reshape(d_model, n_heads, hd)
        K3 = store[L].mean(axis=1).reshape(len(ENTS48), n_heads, hd)
        delta_ap = np.einsum('dhk,ehk->dh', W3,
                             K3[ap_row:ap_row + 1]).T  # (32, 2560)
        for di, r in enumerate(dirs):
            spec9[L, :, di] = delta_ap @ r
        del W, W3
    print('P2826 spectrum done', flush=True)

    # ---------- random null ----------
    rng = np.random.default_rng(SEED)
    U = rng.standard_normal((N_NULL, d_model))
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    nulls = []
    for L in range(n_layers):
        W = read_tensor('model.layers.%d.self_attn.o_proj.weight' % L)
        W3 = W.astype(np.float64).reshape(d_model, n_heads, hd)
        K3 = store[L].mean(axis=1).reshape(len(ENTS48), n_heads, hd)
        delta_ap = np.einsum('dhk,ehk->dh', W3,
                             K3[ap_row:ap_row + 1]).T
        nulls.append(delta_ap @ U.T)  # (32, N_NULL)
        del W, W3
    null_pool = np.concatenate(nulls, axis=0).ravel()  # 1152*100
    null_q95 = float(np.quantile(null_pool, 0.95))
    print('P2826 null q95 %.4f' % null_q95, flush=True)

    # ---------- per-domain stats ----------
    top20 = {}
    for di, name in enumerate(DIR_NAMES):
        c = spec9[:, :, di]
        top = sorted([(int(L), int(h), float(c[L, h]))
                      for L in range(n_layers)
                      for h in range(n_heads) if c[L, h] > 0],
                     key=lambda t: -t[2])[:N_HEADS_EDIT]
        top20[name] = (top, [(L, h) for (L, h, _) in top])
    jacc = {}
    for i, a in enumerate(DIR_NAMES):
        for b in DIR_NAMES[i + 1:]:
            sa = set(top20[a][1])
            sb = set(top20[b][1])
            jacc['%s~%s' % (a, b)] = round(
                len(sa & sb) / max(len(sa | sb), 1), 3)
    jacc_mean = float(np.mean(list(jacc.values())))
    late = {}
    for di, name in enumerate(DIR_NAMES):
        c = spec9[:, :, di]
        pos = np.maximum(c, 0.0)
        late[name] = round(float(pos[30:].sum()
                                 / max(pos.sum(), 1e-30)), 3)
    sig = {}
    for name in DIR_NAMES:
        top, _ = top20[name]
        sig[name] = bool(np.mean([t[2] for t in top]) > null_q95)
    c35h0 = {name: round(float(spec9[35, 0, di]), 3)
             for di, name in enumerate(DIR_NAMES)}
    print('P2826 c35h0 %s' % json.dumps(c35h0), flush=True)
    print('P2826 jacc_mean %.3f late %s sig %s'
          % (jacc_mean, json.dumps(late), json.dumps(sig)), flush=True)

    # ---------- cross-domain behavioural edit: size ----------
    def margins_size2(lg):
        return {s: float(lg[i, adj_id['big']]
                         - lg[i, adj_id['small']])
                for i, s in enumerate(ENTS48)}

    def rank1_size(heads):
        dbig = dD['size']
        dsm = -dD['size']
        backups = []
        for (L, h) in heads:
            W = model.model.layers[L].self_attn.o_proj.weight.data
            sl = slice(h * hd, (h + 1) * hd)
            key = store[L][el_row, :, h * hd:(h + 1) * hd].mean(axis=0)
            kn2 = float(key @ key)
            if kn2 < 1e-12:
                continue
            delta = W[:, sl].float().cpu().numpy().astype(
                np.float64) @ key
            cbig = float(delta @ dbig)
            d = cbig * (dsm - dbig)
            backups.append((L, sl, W[:, sl].clone()))
            upd = torch.tensor(
                np.outer(d.astype(np.float32),
                         (key / kn2).astype(np.float32)),
                device=W.device, dtype=W.dtype)
            W[:, sl] = W[:, sl] + upd
        return backups

    csize = spec9[:, :, DOMAINS.index('size')]
    top_size = sorted([(int(L), int(h), float(csize[L, h]))
                       for L in range(n_layers)
                       for h in range(n_heads) if csize[L, h] > 0],
                      key=lambda t: -t[2])[:N_EDIT_SIZE]
    ms0 = margins_size2(forward())
    bk = rank1_size([(L, h) for (L, h, _) in top_size])
    ms1 = margins_size2(forward())
    for (L, sl, orig) in bk:
        model.model.layers[L].self_attn.o_proj.weight.data[:, sl] = \
            orig
    ms2 = margins_size2(forward())
    ok_rest = max(abs(ms2[s] - ms0[s]) for s in ENTS48) < 0.1
    dsize = {s: round(ms1[s] - ms0[s], 3) for s in ENTS48}
    focus = ['elephant', 'mouse', 'whale', 'bee', 'rock', 'apple']
    print('P2826 size-edit %s' % json.dumps(
        {s: dsize[s] for s in focus}), flush=True)

    p1 = bool(all(sig.values()))
    p2 = bool(all(v > 0.5 for v in late.values()))
    p3 = bool(jacc_mean < 0.35)
    p4 = bool(all(v > 0 for v in c35h0.values()))
    p5 = bool(dsize['elephant'] < -0.5)

    verdict = {
        'null_q95': round(null_q95, 4),
        'per_domain': {
            name: {'top20_mean_c': round(float(np.mean(
                [t[2] for t in top20[name][0]])), 3),
                'late_mass': late[name],
                'top_heads': [{'L': L, 'h': h, 'c': round(c, 3)}
                              for (L, h, c) in top20[name][0][:8]]}
            for name in DIR_NAMES},
        'c35h0_nine_domains': c35h0,
        'jaccard': jacc, 'jaccard_mean': round(jacc_mean, 3),
        'late_mass': late, 'null_signif': sig,
        'size_edit': {'top_size_heads': [
            {'L': L, 'h': h, 'c': round(c, 3)}
            for (L, h, c) in top_size],
            'd_margin': {s: dsize[s] for s in focus},
            'd_margin_full': dsize},
        'P1_head_signif': p1,
        'P2_late_layers': p2,
        'P3_specialisation': p3,
        'P4_universal_write_head': p4,
        'P5_size_head_edit': p5,
        'restore_ok': bool(ok_rest),
        'gate': {'dW': gate_dW, 'Zeval': gate_Z},
        'load_run_seconds': round(time.monotonic() - t0, 1),
    }
    result = {'phase': 2826, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'spec9.npz',
           spec9=spec9.astype(np.float32))

    print('P2826 VERDICT %s' % json.dumps(verdict), flush=True)
    elapsed = time.monotonic() - t0
    cc.ledger('phase2826', elapsed)
    print('P2826 elapsed %.1fs P1=%s P2=%s P3=%s P4=%s P5=%s'
          % (elapsed, p1, p2, p3, p4, p5), flush=True)


if __name__ == '__main__':
    main()
