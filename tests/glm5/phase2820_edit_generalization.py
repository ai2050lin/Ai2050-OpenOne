"""Phase 2820 (LPF-33): EDIT GENERALIZATION + CROSS-TARGET MATRIX.

User follow-up (2026-09-17): after apple red->black, do OTHER red
fruits need the same neurons?  Streetlight red->black?  Apple
red->purple?  Find the general law.

Arm H (closure): L31 h4 (the 4th actually-writing head, animal)
causal ablation under the exact 2818 protocol.
  H-P1: Delta prof_animal > 0 AND > q95(5 random heads)

Arm E/G (edit matrix).  Edit operators (both reversible, no
checkpoint):
  emb-edit(s, src->tgt, beta): shift the entity's embedding row(s) in
     z-space: e_new = e + beta*s0*(dW_tgt - dW_src)/g   (exact z-space
     shift mapped back through rms-norm, s0/g from the original row)
  mlp-edit(src->tgt, cols): for top colour-aligned down_proj columns j
     (registered 2819 census), redirect the column's write component:
     w' = w - (w.dW_src)*dW_src + (w.dW_src)*dW_tgt
 Behaviour endpoint: logit matrix colours x entities for
 'The {s} is'; margin(s) = logit(tgt) - logit(src).
  E-P1 edit_works: exists beta with margin(apple, black-red) > 0
       (baseline strongly negative) AND |Delta margin(sky/grass,
       black-red)| < 1.0
  G-P1 storage_law: emb-edit spillover ratio
       |Dm(cherry)| / |Dm(apple)| < 0.2 (entity-independent storage)
       vs mlp-edit spillover ratio >= 0.5 (shared write columns);
       law = which tier carries the generalization
  G-P2 target_independence: mlp-edit red->black vs red->purple give
       same-sign, comparable apple margin shifts (write columns are
       target-agnostic; target supplied by the embedding tier)
  G-P3 isolation: joint best edit leaves sky/grass margins within
       +/-0.5
 Entities: apple cherry strawberry tomato blood streetlight (red
 cluster) + sky grass coal banana (controls).

Prereg frozen before any readout.
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
OUT = BASE / 'phase2820' / 'edit_generalization'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SRC_2816 = BASE / 'phase2816' / 'layers_semantics_position'
SRC_2817 = BASE / 'phase2817' / 'ov_subspace_natural_position'
SRC_2819 = BASE / 'phase2819' / 'knowledge_edit_locus'
SEED = 2820
N_RAND_SITES = 5
BETAS = [1.0, 2.0, 4.0]

COLOR_WORDS = ['red', 'green', 'blue', 'black', 'white', 'yellow',
               'brown', 'pink', 'gray', 'purple', 'orange', 'crimson']
RED_ENTS = ['apple', 'cherry', 'strawberry', 'tomato', 'blood',
            'streetlight']
CTRL_ENTS = ['sky', 'grass', 'coal', 'banana']
ALL_ENTS = RED_ENTS + CTRL_ENTS
COLOURS = ['red', 'black', 'purple', 'blue', 'green', 'yellow']
TARGETS2818 = ['apple', 'gold', 'eagle', 'sedan', 'Spain', 'soup',
               'cave', 'sock']
PREFIX_WORDS = {
    0: [], 1: ['the'], 2: ['and', ' the'], 3: ['and', ' then', ' the'],
    4: ['and', ' then', ' a', ' small'],
    5: ['and', ' then', ' a', ' very', ' small'],
    6: ['and', ' then', ' a', ' very', ' small', ' shiny'],
}
N_POS = 7

PREREG = {
    'H-P1': 'head L31 h4 causal animal: Delta prof_animal (2818 '
            'protocol, 56 sentences) > 0 AND > q95(5 random heads)',
    'E-P1': 'edit_works iff exists beta with margin(apple, black-red) '
            '> 0 after emb-edit AND |Delta margin(sky, black-red)| '
            '< 1.0 AND |Delta margin(grass, black-red)| < 1.0 '
            '(baseline margin strongly negative)',
    'G-P1': 'storage_law: emb-edit spillover ratio '
            'r_emb = |Dm(cherry, black-red)| / |Dm(apple, black-red)| '
            'at best beta; mlp-edit ratio r_mlp likewise; law = '
            'r_emb < 0.2 (entity rows independent) and/or '
            'r_mlp >= 0.5 (write columns shared)',
    'G-P2': 'target_independence iff mlp-edit red->black and '
            'red->purple give same-sign apple margin shifts with '
            'min(|Dm_black|, |Dm_purple|) >= 0.5 * max(...)',
    'G-P3': 'isolation iff best joint edit leaves |Delta margin(sky, '
            'black-red)| < 0.5 AND |Delta margin(grass, black-red)| '
            '< 0.5',
    'verdict': 'edit law from E-P1/G-P1/G-P2/G-P3; H-P1 closes the '
               '4/4 write-head causal map',
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

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED, 'betas': BETAS,
                 'color_words': COLOR_WORDS,
                 'red_entities': RED_ENTS, 'ctrl_entities': CTRL_ENTS,
                 'colours': COLOURS,
                 'note': 'Arm H closes the 4/4 write-head causal map; '
                         'Arm E/G edit matrix: emb-edit (z-space row '
                         'shift) vs mlp-edit (down_proj column write '
                         'redirect) across red fruits / streetlight / '
                         'purple target; endpoint = colour logit '
                         'margins after "The {s} is"'}
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
    Wo0 = read_tensor('model.layers.0.self_attn.o_proj.weight')
    d_model, hd = Wo0.shape[0], Wo0.shape[1] // n_heads
    del Wo0

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
    unitD = np.stack([unit(dW[i]) for i in range(10)])
    cls_idx = {c: i for i, c in enumerate(CAT_WORDS)}
    rng = np.random.default_rng(SEED)

    def zw(t):
        e = Etab[tid(t)].astype(np.float64)
        return e / np.sqrt((e ** 2).mean() + eps) * g

    dWc = {}
    for c in COLOURS:
        others = [w for w in COLOR_WORDS if w != c]
        dWc[c] = unit(zw(c) - np.stack([zw(w) for w in others]).mean(0))
    # prefill token cache (tid path only: colours/census words;
    # entities go through tok() multi-token path, not tid)
    for w in COLOURS + COLOR_WORDS + ['The']:
        tid(w)

    # ---------- CUDA ----------
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(mdir), dtype=torch.bfloat16, device_map='auto')
    model.eval()
    dev = model.device

    # colour census columns from 2819 (registered census, not refit)
    mlp_rows = res2819['verdict']['K2']['mlp_rows']
    red_cols = sorted([r for r in mlp_rows if r['best_kind'] == 'red'],
                      key=lambda r: -r['best_val'])[:4]
    color_cols = sorted([r for r in mlp_rows
                         if r['best_kind'] == 'color'],
                        key=lambda r: -r['best_val'])[:3]
    red_col_ids = [(r['layer'], r['best_col']) for r in red_cols]
    color_col_ids = [(r['layer'], r['best_col']) for r in color_cols]
    print('P2820 mlp-edit columns: red %s color %s'
          % (red_col_ids, color_col_ids), flush=True)

    # ===== Arm H: L31 h4 animal (2818 protocol) =====
    z17 = np.load(SRC_2817 / 'ovpos.npz')
    head_proj = z17['head_proj'].astype(np.float64)
    head_rq = z17['head_rand_q95'].astype(np.float64)
    r_seqs, r_meta = [], []
    for w in TARGETS2818:
        for p in range(N_POS):
            words = PREFIX_WORDS[p]
            prefix_str = ' '.join(t.strip() for t in words)
            pre_ids = (tok(prefix_str, add_special_tokens=False)
                       ['input_ids']) if prefix_str else []
            assert len(pre_ids) == p
            full_str = ((prefix_str + ' ' if prefix_str else '')
                        + w + ' is here')
            ids = tok(full_str, add_special_tokens=False)['input_ids']
            assert ids[:len(pre_ids)] == pre_ids
            r_seqs.append(ids)
            r_meta.append((w, p, len(pre_ids), len(ids) - 2))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None \
        else tok.eos_token_id
    maxlen = max(len(s) for s in r_seqs)
    r_input = torch.full((len(r_seqs), maxlen), int(pad_id),
                         dtype=torch.long)
    r_mask = torch.zeros((len(r_seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(r_seqs):
        r_input[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        r_mask[i, :len(s)] = 1

    def run_r(ablate=None):
        handles = []
        if ablate is not None:
            _, L, h = ablate

            def abl(mod, args, h=h):
                x = args[0].clone()
                x[..., h * hd:(h + 1) * hd] = 0
                return (x,)
            handles.append(
                model.model.layers[L].self_attn.o_proj
                .register_forward_pre_hook(abl))
        with torch.no_grad():
            out = model(input_ids=r_input.to(dev),
                        attention_mask=r_mask.to(dev),
                        output_hidden_states=True)
        for hh in handles:
            hh.remove()
        return out.hidden_states[-1].float().cpu().numpy()

    def span_profiles(HS_last):
        profs = np.zeros((len(TARGETS2818), 10))
        for wi in range(len(TARGETS2818)):
            vecs = np.stack([HS_last[wi * N_POS + p,
                                     r_meta[wi * N_POS + p][2]:
                                     r_meta[wi * N_POS + p][3], :]
                             .mean(0) for p in range(N_POS)])
            profs[wi] = vecs.mean(0) @ unitD.T
        return profs

    prof0 = span_profiles(run_r())
    d_h4 = float((prof0[:, cls_idx['animal']] -
                  span_profiles(run_r(ablate=('ov', 31, 4)))
                  [:, cls_idx['animal']]).mean())
    rand_h = []
    for _ in range(N_RAND_SITES):
        hh = int(rng.integers(0, n_heads))
        if hh == 4:
            hh = 5
        rand_h.append(float((prof0[:, cls_idx['animal']] -
                             span_profiles(run_r(ablate=('ov', 31, hh)))
                             [:, cls_idx['animal']]).mean()))
    q_h4 = float(np.quantile(rand_h, 0.95))
    p_h1 = bool(d_h4 > 0 and d_h4 > q_h4)
    print('P2820 Arm H L31 h4 animal delta=%.4f q95=%.4f P-H1=%s'
          % (d_h4, q_h4, p_h1), flush=True)

    # ===== Arm E/G: edit matrix =====
    k_seqs = [[tid('The')] + tok(' ' + s, add_special_tokens=False)
              ['input_ids'] + [tid(' is')] for s in ALL_ENTS]
    assert all(s[-1] == tid(' is') for s in k_seqs)
    k_maxlen = max(len(s) for s in k_seqs)
    k_ids = torch.full((len(k_seqs), k_maxlen), int(pad_id),
                       dtype=torch.long)
    k_mask = torch.zeros((len(k_seqs), k_maxlen), dtype=torch.long)
    for i, s in enumerate(k_seqs):
        k_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        k_mask[i, :len(s)] = 1
    col_idx = {c: tid(' ' + c) for c in COLOURS}
    ent_tok = {s: tok(' ' + s, add_special_tokens=False)['input_ids']
               for s in ALL_ENTS}

    def run_k(emb_edit=None, mlp_edit=None):
        """emb_edit = (ent, src, tgt, beta) | None; mlp_edit = (src,
        tgt, cols) | None.  Returns (n_ents, n_colours) logits."""
        handles = []
        if emb_edit is not None:
            ent, src, tgt, beta = emb_edit
            rows = {}
            for t in ent_tok[ent]:
                e = Etab[t].astype(np.float64)
                s0 = np.sqrt((e ** 2).mean() + eps)
                rows[t] = torch.tensor(
                    (e + beta * s0 * (dWc[tgt] - dWc[src]) / g)
                    .astype(np.float32))

            def emb_hook(mod, args, output, rows=rows):
                out = output.clone()
                for t, v in rows.items():
                    out[k_ids.to(dev) == t] = v.to(
                        device=out.device, dtype=out.dtype)
                return out
            handles.append(
                model.model.embed_tokens.register_forward_hook(emb_hook))
        if mlp_edit is not None:
            src, tgt, cols = mlp_edit
            backups = []
            for (L, j) in cols:
                W = model.model.layers[L].mlp.down_proj.weight.data
                backups.append((L, j, W[:, j].clone()))
                w = W[:, j].float().cpu().numpy().astype(np.float64)
                ds = dWc[src].astype(np.float32).astype(np.float64)
                dt = dWc[tgt].astype(np.float32).astype(np.float64)
                comp = float(w.astype(np.float32) @ ds.astype(
                    np.float32))
                w_new = w - comp * ds + comp * dt
                W[:, j] = torch.tensor(w_new.astype(np.float32),
                                       device=W.device)
                # keep dtype/device via assignment above
            def fin():
                for (L, j, orig) in backups:
                    model.model.layers[L].mlp.down_proj.weight.data\
                        [:, j] = orig
            # restore handled by caller via returned closure
        with torch.no_grad():
            out = model(input_ids=k_ids.to(dev),
                        attention_mask=k_mask.to(dev))
        for hh in handles:
            hh.remove()
        if mlp_edit is not None:
            fin()
        lg = out.logits[:, -1, :].float().cpu().numpy()
        return np.stack([[lg[i, col_idx[c]] for c in COLOURS]
                         for i in range(len(ALL_ENTS))])

    ci = {c: i for i, c in enumerate(COLOURS)}

    def margin(LG, ent, tgt='black', src='red'):
        i = ALL_ENTS.index(ent)
        return float(LG[i, ci[tgt]] - LG[i, ci[src]])

    LG0 = run_k()
    m0 = {s: margin(LG0, s) for s in ALL_ENTS}
    print('P2820 baseline margins (black-red) %s'
          % json.dumps({s: round(m, 2) for s, m in m0.items()}),
          flush=True)

    cond = {}
    # emb-edit apple red->black beta sweep
    for beta in BETAS:
        LG = run_k(emb_edit=('apple', 'red', 'black', beta))
        cond['emb_apple_black_b%.0f' % beta] = LG
    # emb-edit apple red->purple (beta 2)
    cond['emb_apple_purple_b2'] = run_k(
        emb_edit=('apple', 'red', 'purple', 2.0))
    # emb-edit streetlight / cherry red->black (beta 2)
    cond['emb_streetlight_black_b2'] = run_k(
        emb_edit=('streetlight', 'red', 'black', 2.0))
    cond['emb_cherry_black_b2'] = run_k(
        emb_edit=('cherry', 'red', 'black', 2.0))
    # mlp-edit red cols -> black / -> purple (no emb change)
    cond['mlp_red_black'] = run_k(
        mlp_edit=('red', 'black', red_col_ids))
    cond['mlp_red_purple'] = run_k(
        mlp_edit=('red', 'purple', red_col_ids))
    # joint: emb apple b2 + mlp red->black
    cond['joint_apple_black_b2'] = run_k(
        emb_edit=('apple', 'red', 'black', 2.0),
        mlp_edit=('red', 'black', red_col_ids))
    # pseudo-edit control: apple blue->black b2
    cond['ctrl_apple_blueblack_b2'] = run_k(
        emb_edit=('apple', 'blue', 'black', 2.0))

    def margins_of(LG):
        return {s: margin(LG, s) for s in ALL_ENTS}

    tbl = {k: {s: round(v, 3) for s, v in margins_of(LG).items()}
           for k, LG in cond.items()}

    # E-P1: apple edit works with isolation at some beta
    e_ok, best_beta, best_row = False, None, None
    for beta in BETAS:
        LG = cond['emb_apple_black_b%.0f' % beta]
        ma = margin(LG, 'apple')
        ds = abs(margin(LG, 'sky') - m0['sky'])
        dg = abs(margin(LG, 'grass') - m0['grass'])
        if ma > 0 and ds < 1.0 and dg < 1.0:
            e_ok, best_beta = True, beta
            best_row = {'beta': beta, 'apple': round(ma, 3),
                        'sky_shift': round(ds, 3),
                        'grass_shift': round(dg, 3)}
            break
    p_e1 = bool(e_ok)

    # G-P1 spillover ratios
    emb_best = cond['emb_apple_black_b%.0f' % (best_beta or 2.0)]
    dm_apple_e = margin(emb_best, 'apple') - m0['apple']
    dm_cherry_e = margin(emb_best, 'cherry') - m0['cherry']
    dm_street_e = margin(emb_best, 'streetlight') - m0['streetlight']
    r_emb = (abs(dm_cherry_e) / max(abs(dm_apple_e), 1e-9)
             if abs(dm_apple_e) > 1e-9 else 0.0)
    LG_mb = cond['mlp_red_black']
    dm_apple_m = margin(LG_mb, 'apple') - m0['apple']
    dm_cherry_m = margin(LG_mb, 'cherry') - m0['cherry']
    dm_street_m = margin(LG_mb, 'streetlight') - m0['streetlight']
    r_mlp = (abs(dm_cherry_m) / max(abs(dm_apple_m), 1e-9)
             if abs(dm_apple_m) > 1e-9 else 0.0)
    g1 = {'dm_apple_emb': round(dm_apple_e, 3),
          'dm_cherry_emb': round(dm_cherry_e, 3),
          'dm_streetlight_emb': round(dm_street_e, 3),
          'r_emb': round(r_emb, 3),
          'dm_apple_mlp': round(dm_apple_m, 3),
          'dm_cherry_mlp': round(dm_cherry_m, 3),
          'dm_streetlight_mlp': round(dm_street_m, 3),
          'r_mlp': round(r_mlp, 3),
          'entity_rows_independent': bool(r_emb < 0.2),
          'write_cols_shared': bool(r_mlp >= 0.5)}

    # G-P2 target independence
    dm_purple = margin(cond['mlp_red_purple'], 'apple') - m0['apple']
    g2 = {'dm_apple_mlp_black': round(dm_apple_m, 3),
          'dm_apple_mlp_purple': round(dm_purple, 3),
          'target_independence': bool(
              np.sign(dm_apple_m) == np.sign(dm_purple) and
              min(abs(dm_apple_m), abs(dm_purple)) >=
              0.5 * max(abs(dm_apple_m), abs(dm_purple), 1e-9))}

    # G-P3 isolation of joint edit
    LG_j = cond['joint_apple_black_b2']
    g3 = {'sky_shift': round(abs(margin(LG_j, 'sky') - m0['sky']), 3),
          'grass_shift': round(abs(margin(LG_j, 'grass') - m0['grass']),
                               3),
          'apple': round(margin(LG_j, 'apple'), 3),
          'isolation': bool(abs(margin(LG_j, 'sky') - m0['sky']) < 0.5
                            and abs(margin(LG_j, 'grass') - m0['grass'])
                            < 0.5)}

    # other red fruits under apple emb-edit (user question)
    fruits = {s: round(margin(emb_best, s) - m0[s], 3)
              for s in ['cherry', 'strawberry', 'tomato']}
    street = round(margin(emb_best, 'streetlight') - m0['streetlight'], 3)

    verdict = {
        'H': {'delta': round(d_h4, 4), 'rand_q95': round(q_h4, 4),
              'head31h4_causal_animal': p_h1,
              'write_heads_causal': '4/4' if p_h1 else '3/4'},
        'baseline_margins': {s: round(m, 3) for s, m in m0.items()},
        'condition_margins': tbl,
        'E-P1': {'edit_works': p_e1, 'best': best_row},
        'G-P1': g1,
        'G-P2': g2,
        'G-P3': g3,
        'fruit_spillover_emb': fruits,
        'streetlight_spillover_emb': street,
        'load_run_seconds': round(time.monotonic() - t0, 1),
    }
    result = {'phase': 2820, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'edit.npz',
           baseline=LG0.astype(np.float32),
           unitD=unitD.astype(np.float32),
           dW_red=dWc['red'].astype(np.float32),
           dW_black=dWc['black'].astype(np.float32),
           dW_purple=dWc['purple'].astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2820', elapsed)
    print('P2820 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2820 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
