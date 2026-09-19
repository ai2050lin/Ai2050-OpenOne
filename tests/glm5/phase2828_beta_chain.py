"""Phase 2828 = BETA (LPF-41): KEY-DIRECTION CHAIN PROPAGATION.

User plan (2026-09-17 07:56 "hao de, jixu"): Phase Beta.  Question: in
multi-hop knowledge chains (apple -> fruit -> food), how does the key
direction propagate hop by hop?

Paradigm: two-sentence chain prompt, token-aligned across variants
  "The <E1> is a fruit. The fruit is a food."
  positions: e1=1, fruit1=4, fruit2=7, food=10
Chain directions from 2806 class directions: dW_fruit, dW_food.
Variants (8 rows): true-chain starters apple/cherry/lemon/banana;
broken-chain starters rock/steel/hammer/ocean.

Measured spectrum: c[L, variant, pos, head, dir] for dir in
{fruit, food}.  Random-direction null pool.

Arms:
  E1 chain edit   : rank-1 remove fruit-component at pos=1 (top-5
                    heads by summed c_fruit, key = apple-row pos-1
                    slice), re-measure c_food(pos=7) follow-through
  E3 repair inject: add beta*dW_fruit to residual stream at pos=1 of
                    broken rows (beta in {2,5,10}), re-measure
                    recovery of c_food(pos=7) toward true-chain level

Prereg (frozen):
  B1: hop1_real: mean top-10 c_fruit(pos=1, true) > null q95
      AND > same for broken chain
  B2: hop2_follows: mean top-10 c_food(pos=7, true) > broken chain
  B3: edit_follow_through: E1 lowers true-chain c_food(pos=7) by
      >= 20%
  B4: repair: exists beta in {2,5,10} with broken c_food(pos=7) after
      inject >= 0.5 x true-chain before-value
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
OUT = BASE / 'phase2828' / 'beta_chain'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SEED = 2828
N_EDIT_HEADS = 5
N_NULL = 100
BETAS = [2.0, 5.0, 10.0]
TRUE_STARTS = ['apple', 'cherry', 'lemon', 'banana']
BROKEN_STARTS = ['rock', 'steel', 'hammer', 'ocean']
VARIANTS = TRUE_STARTS + BROKEN_STARTS
POS = {'e1': 1, 'fruit1': 4, 'fruit2': 7, 'food': 10}
KEEP_POS = [1, 4, 7, 10]
CHAIN_SENT = 'The {} is a fruit. The fruit is a food.'

PREREG = {
    'B1': 'hop1_real: mean top-10 c_fruit(pos=1, true) > null q95 AND '
          '> broken chain same',
    'B2': 'hop2_follows: mean top-10 c_food(pos=7, true) > broken same',
    'B3': 'edit_follow_through: E1 lowers true c_food(pos=7) >= 20%',
    'B4': 'repair: exists beta in {2,5,10} with broken c_food(pos=7) '
          'after inject >= 0.5 x true before-value',
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
        'n_edit_heads': N_EDIT_HEADS,
        'true_starts': TRUE_STARTS, 'broken_starts': BROKEN_STARTS,
        'chain_sent': CHAIN_SENT, 'positions': POS,
        'note': 'BETA: two-hop chain key-direction propagation; '
                'hop1 removal edit; broken-chain repair injection'}
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

    ci = {c: i for i, c in enumerate(CAT_WORDS)}
    d_fruit = dW[ci['fruit']].astype(np.float64)
    d_food = dW[ci['food']].astype(np.float64)
    DMAT2 = np.stack([d_fruit, d_food], axis=1)  # (2560, 2)

    from transformers import AutoModelForCausalLM
    import torch
    model = AutoModelForCausalLM.from_pretrained(
        str(mdir), dtype=torch.bfloat16, device_map='auto')
    model.eval()
    dev = model.device

    for s in VARIANTS:
        tid(s)
    sents = [CHAIN_SENT.format(s) for s in VARIANTS]
    enc = [tok(s, add_special_tokens=False)['input_ids'] for s in sents]
    L_seq = len(enc[0])
    assert all(len(e) == L_seq for e in enc), 'variant length mismatch'
    k_ids = torch.tensor(enc, dtype=torch.long)
    k_mask = torch.ones_like(k_ids)
    print('P2828 seq_len %d variants %d' % (L_seq, len(VARIANTS)),
          flush=True)

    def forward():
        with torch.no_grad():
            out = model(input_ids=k_ids.to(dev),
                        attention_mask=k_mask.to(dev))
        return out.logits.float().cpu().numpy()

    # ---------- capture helper ----------
    store = {}

    def make_cap(Lr):
        def cap(mod, args):
            store[Lr] = args[0][:, KEEP_POS, :].detach().float() \
                .cpu().numpy()
            return None
        return cap

    def capture_once():
        hs = [model.model.layers[i].self_attn.o_proj
              .register_forward_pre_hook(make_cap(i))
              for i in range(n_layers)]
        lg = forward()
        for h in hs:
            h.remove()
        return lg

    nv = len(VARIANTS)
    np_pos = len(KEEP_POS)

    Wo0 = read_tensor('model.layers.0.self_attn.o_proj.weight')
    d_model, hd_total = Wo0.shape
    hd = hd_total // n_heads
    del Wo0

    def spec_from_store():
        sp = np.zeros((n_layers, nv, np_pos, n_heads, 2),
                      dtype=np.float32)
        for Lr in range(n_layers):
            W = read_tensor(
                'model.layers.%d.self_attn.o_proj.weight' % Lr)
            W3 = W.astype(np.float64).reshape(d_model, n_heads, hd)
            K3 = store[Lr].reshape(nv, np_pos, n_heads, hd)
            delta = np.einsum('dhk,bphk->bpdh', W3, K3)
            sp[Lr] = np.einsum('bpdh,dr->bphr', delta, DMAT2) \
                .astype(np.float32)
            del W, W3, delta
        return sp

    capture_once()
    spec = spec_from_store()
    print('P2828 spec done %s' % (spec.shape,), flush=True)

    # ---------- random null (base deltas) ----------
    rng = np.random.default_rng(SEED)
    U = rng.standard_normal((N_NULL, d_model))
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    null_parts = []
    for Lr in range(n_layers):
        W = read_tensor('model.layers.%d.self_attn.o_proj.weight' % Lr)
        W3 = W.astype(np.float64).reshape(d_model, n_heads, hd)
        K3 = store[Lr].reshape(nv * np_pos, n_heads, hd)
        delta = np.einsum('dhk,rhk->rdh', W3, K3)
        c = np.einsum('rdh,dq->rhq', delta, U.T.astype(np.float64))
        null_parts.append(c.ravel())
        del W, W3, delta
    null_pool = np.concatenate(null_parts)
    null_q95 = float(np.quantile(null_pool, 0.95))
    print('P2828 null q95 %.4f' % null_q95, flush=True)

    true_idx = list(range(len(TRUE_STARTS)))
    broken_idx = list(range(len(TRUE_STARTS), nv))

    def top10_mean(v):
        return float(np.sort(v)[::-1][:10].mean())

    PIDX = {'e1': 0, 'fruit1': 1, 'fruit2': 2, 'food': 3}

    def grp_stat(sp, pos_name, di, rows):
        s = sp[:, rows, PIDX[pos_name], :, di].astype(np.float64)
        vals = [top10_mean(s[:, r, :].sum(axis=0))
                for r in range(s.shape[1])]
        return float(np.mean(vals))

    b1_true = grp_stat(spec, 'e1', 0, true_idx)
    b1_broken = grp_stat(spec, 'e1', 0, broken_idx)
    b2_true = grp_stat(spec, 'fruit2', 1, true_idx)
    b2_broken = grp_stat(spec, 'fruit2', 1, broken_idx)
    aux = {
        'c_fruit_e1': {'true': round(b1_true, 3),
                       'broken': round(b1_broken, 3)},
        'c_food_fruit1': {
            'true': round(grp_stat(spec, 'fruit1', 1, true_idx), 3),
            'broken': round(grp_stat(spec, 'fruit1', 1, broken_idx),
                            3)},
        'c_food_food_pos': {
            'true': round(grp_stat(spec, 'food', 1, true_idx), 3),
            'broken': round(grp_stat(spec, 'food', 1, broken_idx), 3)},
        'c_fruit_fruit2': {
            'true': round(grp_stat(spec, 'fruit2', 0, true_idx), 3),
            'broken': round(grp_stat(spec, 'fruit2', 0, broken_idx),
                            3)},
    }
    b1 = bool(b1_true > null_q95 and b1_true > b1_broken)
    b2 = bool(b2_true > b2_broken)
    print('P2828 B1 true %.3f broken %.3f | B2 true %.3f broken %.3f'
          % (b1_true, b1_broken, b2_true, b2_broken), flush=True)

    # ---------- E1: remove hop-1 fruit write (apple row key) ----------
    c_ap = spec[:, 0, POS['e1'], :, 0].astype(np.float64)  # (36, 32)
    flat = c_ap.sum(axis=0)
    top5 = list(np.argsort(-flat)[:N_EDIT_HEADS])
    print('P2828 E1 heads %s' % json.dumps(
        [{'L': int(Lh // n_heads), 'h': int(Lh % n_heads),
          'c': round(float(flat[Lh]), 3)} for Lh in top5]), flush=True)

    def e1_edit():
        backups = []
        for Lh in top5:
            Lr, h = Lh // n_heads, Lh % n_heads
            W = model.model.layers[Lr].self_attn.o_proj.weight.data
            sl = slice(h * hd, (h + 1) * hd)
            key = store[Lr][0, 0, h * hd:(h + 1) * hd]
            kn2 = float(key @ key)
            if kn2 < 1e-12:
                continue
            delta = W[:, sl].float().cpu().numpy().astype(
                np.float64) @ key
            c = float(delta @ d_fruit)
            d = -c * d_fruit
            backups.append((Lr, sl, W[:, sl].clone()))
            upd = torch.tensor(
                np.outer(d.astype(np.float32),
                         (key / kn2).astype(np.float32)),
                device=W.device, dtype=W.dtype)
            W[:, sl] = W[:, sl] + upd
        return backups

    def restore(bk):
        for (Lr, sl, orig) in bk:
            model.model.layers[Lr].self_attn.o_proj.weight.data[:, sl] \
                = orig

    b3_before = b2_true
    bk = e1_edit()
    capture_once()
    spec_e1 = spec_from_store()
    restore(bk)
    b3_after = grp_stat(spec_e1, 'fruit2', 1, true_idx)
    b3_broken_after = grp_stat(spec_e1, 'fruit2', 1, broken_idx)
    drop = (b3_before - b3_after) / max(abs(b3_before), 1e-9)
    b3 = bool(drop >= 0.20)
    print('P2828 E1 c_food(fruit2) true %.3f -> %.3f (drop %.1f%%) '
          'broken %.3f -> %.3f'
          % (b3_before, b3_after, 100 * drop, b2_broken,
             b3_broken_after), flush=True)

    # ---------- E3: repair injection at pos=1 of broken rows ----------
    d_t = torch.tensor(d_fruit.astype(np.float32))

    def make_inject(beta, rows):
        def inj(mod, args):
            h = args[0]
            out = h.clone()
            idx = torch.tensor(rows, device=h.device)
            out[idx, POS['e1'], :] = out[idx, POS['e1'], :] \
                + (beta * d_t).to(device=h.device, dtype=h.dtype)
            return out
        return inj

    repair_log = []
    b4 = False
    best_beta = None
    for beta in BETAS:
        hnd = model.model.layers[0].register_forward_pre_hook(
            make_inject(beta, broken_idx))
        capture_once()
        hnd.remove()
        sp_i = spec_from_store()
        v = grp_stat(sp_i, 'fruit2', 1, broken_idx)
        ok = bool(v >= 0.5 * b2_true)
        repair_log.append({'beta': beta,
                           'c_food_fruit2_broken': round(v, 3),
                           'recovers_half': ok})
        if ok and best_beta is None:
            best_beta = beta
            b4 = True
    print('P2828 repair %s' % json.dumps(repair_log), flush=True)

    # ---------- restore check ----------
    capture_once()
    spec_rest = spec_from_store()
    ok_rest = bool(np.abs(spec_rest - spec).max() < 0.05)

    verdict = {
        'null_q95': round(null_q95, 4),
        'aux_spectrum': aux,
        'B1_hop1_real': b1,
        'B2_hop2_follows': b2,
        'B3_edit_follow_through': b3,
        'B4_repair': b4,
        'b1_true': round(b1_true, 3), 'b1_broken': round(b1_broken, 3),
        'b2_true': round(b2_true, 3),
        'b2_broken': round(b2_broken, 3),
        'b3_before': round(b3_before, 3),
        'b3_after': round(b3_after, 3),
        'b3_broken_after': round(b3_broken_after, 3),
        'b3_drop_pct': round(100 * drop, 1),
        'repair_log': repair_log, 'best_repair_beta': best_beta,
        'e1_heads': [{'L': int(Lh // n_heads),
                      'h': int(Lh % n_heads),
                      'c': round(float(flat[Lh]), 3)}
                     for Lh in top5],
        'restore_ok': ok_rest,
        'gate': {'dW': gate_dW, 'Zeval': gate_Z},
        'load_run_seconds': round(time.monotonic() - t0, 1),
    }
    result = {'phase': 2828, 'plan': 'BETA', 'prereg': PREREG,
              'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'chain_spec.npz', spec=spec)
    print('P2828 VERDICT %s' % json.dumps(verdict), flush=True)
    elapsed = time.monotonic() - t0
    cc.ledger('phase2828', elapsed)
    print('P2828 elapsed %.1fs B1=%s B2=%s B3=%s B4=%s'
          % (elapsed, b1, b2, b3, b4), flush=True)


if __name__ == '__main__':
    main()
