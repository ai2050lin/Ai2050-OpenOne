"""Phase 2830 = IN-SENTENCE CHAIN RETEST (LPF-42).

Beta (2828) falsified causal chain propagation (B3=false, removal of
hop-1 fruit write lowers c_food by only 0.2%) in a TWO-SENTENCE
paradigm.  Hypothesis: the two sentences are processed in parallel;
chain coherence comes from within-sentence semantics.  Test: same
chain INSIDE ONE SENTENCE (comma-linked), forcing shared inference
flow.

Paradigm (13 tokens, aligned):
  "An <E1> is a fruit, and a fruit is a food."
  positions: e1=1, fruit1=4, fruit2=8, food=11
  variants 8: true apple/cherry/lemon/banana; broken rock/steel/
  hammer/ocean.  Directions dW_fruit / dW_food from 2806 (gate-checked).

Arms:
  spec capture (8x4 positions x 36L x 32H x 2 dirs) + random null
  E1a removal @e1 (top-5 c_fruit@e1 heads, apple-row key)  [Beta
      replication control]
  E1b removal @fruit1 (top-5 c_fruit@fruit1 heads, fruit1-row key)
      [in-sentence specific intervention point]

Prereg (frozen):
  C1 hop1_real_insent: c_fruit@e1 true > null q95 AND > broken
  C2 hop2_follows_insent: c_food@fruit2 top-10 true > broken
  C3 causal_chain: E1b lowers true c_food@fruit2 >= 20%
  C4 activation_flow: insent true c_food@fruit2 > 2828 two-sentence
     value 0.124
  C5 (control, expect small): E1a effect on c_food@fruit2
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
OUT = BASE / 'phase2830' / 'insent_chain'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SRC_2828 = BASE / 'phase2828' / 'beta_chain' / 'result.json'
SEED = 2830
N_EDIT_HEADS = 5
N_NULL = 100
TRUE_STARTS = ['apple', 'cherry', 'lemon', 'banana']
BROKEN_STARTS = ['rock', 'steel', 'hammer', 'ocean']
VARIANTS = TRUE_STARTS + BROKEN_STARTS
POS = {'e1': 1, 'fruit1': 4, 'fruit2': 8, 'food': 11}
KEEP_POS = [1, 4, 8, 11]
PIDX = {'e1': 0, 'fruit1': 1, 'fruit2': 2, 'food': 3}
BASE_2828_FOOD_FRUIT2 = 0.124
BETA_2828_DROP_PCT = 0.2

PREREG = {
    'C1': 'hop1_real_insent: c_fruit@e1 true > null q95 AND > broken',
    'C2': 'hop2_follows_insent: c_food@fruit2 top-10 true > broken',
    'C3': 'causal_chain: E1b removal @fruit1 lowers true c_food@fruit2 '
          '>= 20%',
    'C4': 'activation_flow: insent true c_food@fruit2 > %s (2828 '
          'two-sentence b2_true)' % BASE_2828_FOOD_FRUIT2,
    'C5': 'control: E1a removal @e1 effect replicates Beta B3 '
          '(expected small; Beta drop %s%%)' % BETA_2828_DROP_PCT,
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
    res2828 = json.loads(SRC_2828.read_text(encoding='utf-8'))
    b2_2828 = float(res2828['verdict']['b2_true'])
    assert b2_2828 == BASE_2828_FOOD_FRUIT2, b2_2828

    execution = {
        'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
        'prereg': PREREG, 'seed': SEED, 'n_edit_heads': N_EDIT_HEADS,
        'true_starts': TRUE_STARTS, 'broken_starts': BROKEN_STARTS,
        'chain_sent': 'An/A {E1} is a fruit, and a fruit is a food.',
        'positions': POS, 'b2_2828_two_sentence': b2_2828,
        'note': 'in-sentence comma-linked chain retest of Beta B3 '
                'causal falsification; E1a@e1 control, E1b@fruit1 '
                'in-sentence intervention'}
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
    sents = [('An ' if s[0] in 'aeiou' else 'A ')
             + '%s is a fruit, and a fruit is a food.' % s
             for s in VARIANTS]
    enc = [tok(s, add_special_tokens=False)['input_ids'] for s in sents]
    L_seq = len(enc[0])
    assert all(len(e) == L_seq for e in enc), 'variant length mismatch'
    assert enc[0][POS['e1']] == tid('apple')
    assert enc[0][POS['fruit1']] == tid('fruit')
    assert enc[0][POS['fruit2']] == tid('fruit')
    assert enc[0][POS['food']] == tid('food')
    k_ids = torch.tensor(enc, dtype=torch.long)
    k_mask = torch.ones_like(k_ids)
    print('P2830 seq_len %d variants %d' % (L_seq, len(VARIANTS)),
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
    print('P2830 spec done %s' % (spec.shape,), flush=True)

    # ---------- random null ----------
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
    print('P2830 null q95 %.4f' % null_q95, flush=True)

    true_idx = list(range(len(TRUE_STARTS)))
    broken_idx = list(range(len(TRUE_STARTS), nv))

    def top10_mean(v):
        return float(np.sort(v)[::-1][:10].mean())

    def grp_stat(sp, pos_name, di, rows):
        vals = []
        for r in rows:
            s = sp[:, r, PIDX[pos_name], :, di].astype(np.float64) \
                .sum(axis=0)
            vals.append(top10_mean(s))
        return float(np.mean(vals))

    c1_true = grp_stat(spec, 'e1', 0, true_idx)
    c1_broken = grp_stat(spec, 'e1', 0, broken_idx)
    c2_true = grp_stat(spec, 'fruit2', 1, true_idx)
    c2_broken = grp_stat(spec, 'fruit2', 1, broken_idx)
    aux = {
        'c_fruit_e1': {'true': round(c1_true, 3),
                       'broken': round(c1_broken, 3)},
        'c_fruit_fruit1': {
            'true': round(grp_stat(spec, 'fruit1', 0, true_idx), 3),
            'broken': round(grp_stat(spec, 'fruit1', 0, broken_idx),
                            3)},
        'c_fruit_fruit2': {
            'true': round(grp_stat(spec, 'fruit2', 0, true_idx), 3),
            'broken': round(grp_stat(spec, 'fruit2', 0, broken_idx),
                            3)},
        'c_food_food_pos': {
            'true': round(grp_stat(spec, 'food', 1, true_idx), 3),
            'broken': round(grp_stat(spec, 'food', 1, broken_idx), 3)},
    }
    c1 = bool(c1_true > null_q95 and c1_true > c1_broken)
    c2 = bool(c2_true > c2_broken)
    c4 = bool(c2_true > b2_2828)
    print('P2830 C1 true %.3f broken %.3f | C2 true %.3f broken %.3f'
          % (c1_true, c1_broken, c2_true, c2_broken), flush=True)

    # ---------- removal edits (rank-1 on o_proj head slices) ----------
    def removal(pos_name, di=0, rows=(0,)):
        heads = []
        for r in rows:
            c_map = spec[:, r, PIDX[pos_name], :, di] \
                .astype(np.float64).sum(axis=0)
            for Lh in np.argsort(-c_map)[:N_EDIT_HEADS]:
                heads.append((int(Lh // n_heads), int(Lh % n_heads)))
        heads = sorted(set(heads))

        def edit():
            backups = []
            for (Lr, h) in heads:
                W = model.model.layers[Lr].self_attn.o_proj.weight.data
                sl = slice(h * hd, (h + 1) * hd)
                key = store[Lr][rows[0], PIDX[pos_name],
                                h * hd:(h + 1) * hd]
                kn2 = float(key @ key)
                if kn2 < 1e-12:
                    continue
                delta = W[:, sl].float().cpu().numpy().astype(
                    np.float64) @ key
                c = float(delta @ (d_fruit if di == 0 else d_food))
                d = -c * (d_fruit if di == 0 else d_food)
                backups.append((Lr, sl, W[:, sl].clone()))
                upd = torch.tensor(
                    np.outer(d.astype(np.float32),
                             (key / kn2).astype(np.float32)),
                    device=W.device, dtype=W.dtype)
                W[:, sl] = W[:, sl] + upd
            return backups

        def restore(bk):
            for (Lr, sl, orig) in bk:
                model.model.layers[Lr].self_attn.o_proj.weight.data[
                    :, sl] = orig

        return edit, restore

    def refood(rows):
        capture_once()
        sp2 = spec_from_store()
        vals = []
        for r in rows:
            s = sp2[:, r, PIDX['fruit2'], :, 1].astype(np.float64) \
                .sum(axis=0)
            vals.append(top10_mean(s))
        return float(np.mean(vals))

    # E1a control: removal @e1 (apple row), replicating 2828 edit arm
    print('P2830 arm E1a removal @e1...', flush=True)
    e_a, r_a = removal('e1', 0, rows=(0,))
    b_a = e_a()
    c2_after_e1a = refood(true_idx)
    r_a(b_a)

    # E1b: removal @fruit1 (same-sentence specific point)
    print('P2830 arm E1b removal @fruit1...', flush=True)
    e_b, r_b = removal('fruit1', 0, rows=(0,))
    b_b = e_b()
    c2_after_e1b = refood(true_idx)
    r_b(b_b)

    # broken-chain control: removal @fruit1 on rock row
    print('P2830 arm CTRL removal @fruit1 broken...', flush=True)
    e_c, r_c = removal('fruit1', 0, rows=(4,))
    b_c = e_c()
    c2_after_ctrl = refood(broken_idx)
    r_c(b_c)

    # restore check
    capture_once()
    spec_rest = spec_from_store()
    restore_ok = bool(np.allclose(
        spec_rest[:, :, :, :, :], spec[:, :, :, :, :], atol=2e-3))

    drop_e1a = (c2_true - c2_after_e1a) / max(c2_true, 1e-30)
    drop_e1b = (c2_true - c2_after_e1b) / max(c2_true, 1e-30)
    drop_ctrl = (c2_broken - c2_after_ctrl) / max(max(c2_broken, 1e-30),
                                                  1e-30)
    c3 = bool(drop_e1b >= 0.20)

    verdict = {
        'null_q95': round(null_q95, 4),
        'aux_spectrum': aux,
        'C1_hop1_real_insent': c1,
        'C2_hop2_follows_insent': c2,
        'C3_causal_chain': c3,
        'C4_activation_flow': c4,
        'c1_true': round(c1_true, 3), 'c1_broken': round(c1_broken, 3),
        'c2_true': round(c2_true, 3), 'c2_broken': round(c2_broken, 3),
        'b2_2828_two_sentence': b2_2828,
        'c2_after_e1a': round(c2_after_e1a, 3),
        'c2_after_e1b': round(c2_after_e1b, 3),
        'c2_broken_after_ctrl': round(c2_after_ctrl, 3),
        'drop_e1a_pct': round(float(drop_e1a) * 100, 1),
        'drop_e1b_pct': round(float(drop_e1b) * 100, 1),
        'drop_ctrl_pct': round(float(drop_ctrl) * 100, 1),
        'restore_ok': restore_ok,
        'gate': {'dW': gate_dW, 'Zeval': gate_Z},
        'load_run_seconds': round(time.monotonic() - t0, 1),
    }
    fc.save(OUT / 'result.json', verdict)
    np.savez(OUT / 'insent_spec.npz', spec=spec, null_pool=null_pool,
             enc=np.array(enc))
    print('P2830 DONE verdict %s' % json.dumps(
        {k: v for k, v in verdict.items()
         if k.startswith('C') and isinstance(v, bool)}, indent=1),
        flush=True)
    print('P2830 seconds %.1f' % (time.monotonic() - t0), flush=True)


if __name__ == '__main__':
    main()
