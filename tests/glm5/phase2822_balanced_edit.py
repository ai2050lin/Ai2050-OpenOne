"""Phase 2822 (LPF-35): BALANCED ANTAGONISTIC EDIT — toward a WORKING
knowledge editor.

User question (2026-09-17 05:48): can we actually EDIT apple's colour
/ size by adjusting parameters?  2820 failed with top-4 columns
(sign-mixed, net cancel); 2821 mapped the full architecture: domain-
separated late-layer column groups, sign-antagonistic gating.  This
phase tests the balanced edit at scale.

Census: L26-35 down_proj columns, signed cos with dW_red (colour,
2819 protocol) and dD_big (size axis).  top-20 per layer per domain
(signs REGISTERED, 2820/2821 lesson).

Edit variants (all at ' is' readout, margins black-red / big-small):
  V1 uniform_redirect (2820 formula, scaled): w' = w - c*r + c*b
  V2 amplitude_balanced:               w' = w - c*r + |c|*b
     (every column writes black with its full magnitude; positive
      and negative columns cannot cancel on the target side)
  V3 pure_ablation:                    w' = w - c*r
  Scales: top4/layer (2820 replicate), top10, top20.
Size domain: V2-flip (w' = w - 2c*dD for c>0; single-sided flip) and
V3-ablation at top20.

Prereg (frozen):
  E1 flip_works: V2 top20 gives margin(apple, black-red) > 0
  E2 isolation: V2 top20 leaves |d margin(sky/grass/coal)| < 0.5
  E3 dose_response: |d margin(apple)| monotone across top4<top10<
     top20 for V2
  E4 cross_domain: size V2-flip top20 gives margin(elephant) < 0
  E5 balance_vs_uniform: |d apple(V2)| > |d apple(V1)| at top20
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
OUT = BASE / 'phase2822' / 'balanced_edit'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SEED = 2822
CENSUS_L0, CENSUS_L1 = 26, 35
TOPK = 20
COLOR_WORDS = ['red', 'green', 'blue', 'black', 'white', 'yellow',
               'brown', 'pink', 'gray', 'purple', 'orange', 'crimson']
RED_ENTS = ['apple', 'cherry', 'strawberry', 'tomato', 'blood',
            'streetlight']
CTRL_ENTS = ['sky', 'grass', 'coal', 'banana']
SIZE_ENTS = ['elephant', 'mouse', 'rock', 'feather', 'cherry']
ALL_ENTS = RED_ENTS + CTRL_ENTS + SIZE_ENTS
COLOURS = ['red', 'black', 'purple', 'blue', 'green', 'yellow']
SIZES = ['big', 'small']

PREREG = {
    'E1': 'flip_works: V2 amplitude_balanced top20/layer gives '
          'margin(apple, black-red) > 0',
    'E2': 'isolation: V2 top20 leaves |d margin(sky/grass/coal, '
          'black-red)| < 0.5 each',
    'E3': 'dose_response: |d margin(apple)| monotone top4<top10<top20 '
          'for V2',
    'E4': 'cross_domain: size V2-flip top20 gives margin(elephant, '
          'big-small) < 0',
    'E5': 'balance_vs_uniform: |d apple(V2)| > |d apple(V1)| at top20',
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
        'census_layers': [CENSUS_L0, CENSUS_L1], 'topk': TOPK,
        'entities': ALL_ENTS, 'colours': COLOURS,
        'variants': ['V1_uniform', 'V2_balanced', 'V3_ablate'],
        'note': 'balanced antagonistic edit at scale: colour red->'
                'black and size big->small via late-layer column '
                'groups with sign registration'}
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

    for w in COLOR_WORDS + COLOURS + SIZES + ['The', ' is']:
        tid(w)
    dWc = {}
    for c in ['red', 'black']:
        others = [w for w in COLOR_WORDS if w != c]
        dWc[c] = unit(zw(c) - np.stack([zw(w) for w in others]).mean(0))
    dD_big = unit(zw('big') - zw('small'))

    # ---------- CUDA ----------
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(mdir), dtype=torch.bfloat16, device_map='auto')
    model.eval()
    dev = model.device

    # ---------- census: signed cos, top-20 per layer ----------
    red_cols, size_cols = [], []
    for L in range(CENSUS_L0, CENSUS_L1 + 1):
        name = 'model.layers.%d.mlp.down_proj.weight' % L
        with safe_open(str(mdir / index[name]), framework='pt') as f:
            W = f.get_tensor(name).float().numpy().astype(np.float32)
        Wn = W / np.maximum(np.linalg.norm(W, axis=0, keepdims=True),
                            1e-12)
        cr = Wn.T @ dWc['red'].astype(np.float32)
        cs = Wn.T @ dD_big.astype(np.float32)
        orr = np.argsort(-np.abs(cr))[:TOPK]
        oss = np.argsort(-np.abs(cs))[:TOPK]
        red_cols += [(L, int(j), float(cr[j])) for j in orr]
        size_cols += [(L, int(j), float(cs[j])) for j in oss]
        del W, Wn
    print('P2822 census red %d cols (%d pos) size %d cols (%d pos)'
          % (len(red_cols), sum(1 for c in red_cols if c[2] > 0),
             len(size_cols), sum(1 for c in size_cols if c[2] > 0)),
          flush=True)

    # ---------- batch + readout ----------
    ent_tok = {s: tok(' ' + s, add_special_tokens=False)['input_ids']
               for s in ALL_ENTS}
    k_seqs = [[tid('The')] + ent_tok[s] + [tid(' is')]
              for s in ALL_ENTS]
    k_maxlen = max(len(s) for s in k_seqs)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None \
        else tok.eos_token_id
    k_ids = torch.full((len(k_seqs), k_maxlen), int(pad_id),
                       dtype=torch.long)
    k_mask = torch.zeros((len(k_seqs), k_maxlen), dtype=torch.long)
    for i, s in enumerate(k_seqs):
        k_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        k_mask[i, :len(s)] = 1
    col_id = {c: tid(c) for c in COLOURS + SIZES}

    def readout(lg):
        cm = np.stack([[lg[i, col_id[c]] for c in COLOURS]
                       for i in range(len(RED_ENTS + CTRL_ENTS))])
        ci = {c: i for i, c in enumerate(COLOURS)}
        cmar = {s: float(cm[i, ci['black']] - cm[i, ci['red']])
                for i, s in enumerate(RED_ENTS + CTRL_ENTS)}
        smar = {s: float(lg[len(RED_ENTS + CTRL_ENTS) + i, col_id['big']]
                         - lg[len(RED_ENTS + CTRL_ENTS) + i,
                              col_id['small']])
                for i, s in enumerate(SIZE_ENTS)}
        return cmar, smar

    def forward():
        with torch.no_grad():
            out = model(input_ids=k_ids.to(dev),
                        attention_mask=k_mask.to(dev))
        return out.logits[:, -1, :].float().cpu().numpy()

    def apply_edit(cols, variant):
        """cols = [(L, j, comp)]; variant V1/V2/V3; colour r->b."""
        backups = []
        r = dWc['red'].astype(np.float32).astype(np.float64)
        b = dWc['black'].astype(np.float32).astype(np.float64)
        for (L, j, comp) in cols:
            W = model.model.layers[L].mlp.down_proj.weight.data
            backups.append((L, j, W[:, j].clone()))
            w = W[:, j].float().cpu().numpy().astype(np.float64)
            if variant == 'V1':
                w_new = w - comp * r + comp * b
            elif variant == 'V2':
                w_new = w - comp * r + abs(comp) * b
            else:
                w_new = w - comp * r
            W[:, j] = torch.tensor(w_new.astype(np.float32),
                                   device=W.device)
        return backups

    def restore(backups):
        for (L, j, orig) in backups:
            model.model.layers[L].mlp.down_proj.weight.data[:, j] = orig

    def run_colour(cols, variant):
        bk = apply_edit(cols, variant)
        lg = forward()
        restore(bk)
        return readout(lg)

    def apply_size(cols, variant):
        """variant 'flip' (V2 analogue, single-sided) or 'ablate'."""
        backups = []
        dd = dD_big.astype(np.float32).astype(np.float64)
        for (L, j, comp) in cols:
            W = model.model.layers[L].mlp.down_proj.weight.data
            backups.append((L, j, W[:, j].clone()))
            w = W[:, j].float().cpu().numpy().astype(np.float64)
            if variant == 'flip':
                if comp > 0:
                    w = w - 2.0 * comp * dd
            else:
                w = w - comp * dd
            W[:, j] = torch.tensor(w.astype(np.float32),
                                   device=W.device)
        return backups

    def run_size(cols, variant):
        bk = apply_size(cols, variant)
        lg = forward()
        restore(bk)
        return readout(lg)

    cmar0, smar0 = readout(forward())
    print('P2822 baseline colour %s' % json.dumps(
        {k: round(v, 2) for k, v in cmar0.items()}), flush=True)
    print('P2822 baseline size %s' % json.dumps(
        {k: round(v, 2) for k, v in smar0.items()}), flush=True)

    def subset(cols, k):
        """top-k per layer, flattened in layer order."""
        by_layer = {}
        for (L, j, c) in cols:
            by_layer.setdefault(L, []).append((L, j, c))
        out = []
        for L in sorted(by_layer):
            out += sorted(by_layer[L], key=lambda t: -abs(t[2]))[:k]
        return out

    red4 = subset(red_cols, 4)
    red10 = subset(red_cols, 10)
    red20 = subset(red_cols, 20)
    size20 = subset(size_cols, 20)

    conds = {}
    for var in ['V1', 'V2', 'V3']:
        for name, cols in [('top4', red4), ('top10', red10),
                           ('top20', red20)]:
            if var == 'V1' and name == 'top10':
                continue
            cmar, _ = run_colour(cols, var)
            conds['%s_%s' % (var, name)] = cmar
    _, s_flip = run_size(size20, 'flip')
    _, s_abl = run_size(size20, 'ablate')

    def dm(cmar):
        return {s: round(cmar[s] - cmar0[s], 3)
                for s in RED_ENTS + CTRL_ENTS}

    tbl = {k: dm(v) for k, v in conds.items()}

    # ---------- prereg judgements ----------
    v2t = conds['V2_top20']
    v1t = conds['V1_top20']
    e1 = bool(v2t['apple'] > 0)
    e2 = bool(all(abs(v2t[s] - cmar0[s]) < 0.5
                  for s in ['sky', 'grass', 'coal']))
    d_a = [abs(conds['V2_%s' % n]['apple'] - cmar0['apple'])
           for n in ['top4', 'top10', 'top20']]
    e3 = bool(d_a[0] <= d_a[1] <= d_a[2])
    e4 = bool(s_flip['elephant'] < 0)
    e5 = bool(abs(v2t['apple'] - cmar0['apple']) >
              abs(v1t['apple'] - cmar0['apple']))

    verdict = {
        'baseline_colour': {k: round(v, 3) for k, v in cmar0.items()},
        'baseline_size': {k: round(v, 3) for k, v in smar0.items()},
        'condition_deltas': tbl,
        'size_flip_deltas': {s: round(s_flip[s] - smar0[s], 3)
                             for s in SIZE_ENTS},
        'size_ablate_deltas': {s: round(s_abl[s] - smar0[s], 3)
                               for s in SIZE_ENTS},
        'E1_flip_works': e1, 'E2_isolation': e2,
        'E3_dose_response': e3, 'E4_cross_domain': e4,
        'E5_balance_vs_uniform': e5,
        'gate': {'dW': gate_dW, 'Zeval': gate_Z},
        'load_run_seconds': round(time.monotonic() - t0, 1),
    }
    result = {'phase': 2822, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'edit.npz',
           dW_red=dWc['red'].astype(np.float32),
           dW_black=dWc['black'].astype(np.float32),
           dD_big=dD_big.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2822', elapsed)
    print('P2822 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2822 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
