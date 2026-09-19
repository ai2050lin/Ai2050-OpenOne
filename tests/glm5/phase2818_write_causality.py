"""Phase 2818 (LPF-31): WRITE-HEAD CENSUS, ACTUAL WRITE, CAUSAL ABLATION.

Closes 2817's registered limitations (1)(2): proj_len measures write
CAPACITY, not actual writes; only the best head was read out.

Arm V (actual write verification, CUDA): for every pass head (L,h)
from 2817 (proj >= 0.40, rand q95 < 0.40), capture its real input a_h
(the 128-dim slice of o_proj input) on 56 natural sentences (2817
templates), form the mean actual write delta_bar = W_h @ mean(a_h over
target spans), and measure cos(delta_bar, unitD_i) for the head's
argmax class; random non-pass heads give the shape-matched null with
max-over-classes cos (conservative, same aggregation as the 2817 pass
rule).
  P-V1 ov_write_actual: median over pass heads of
       cos(delta_bar_h, unitD_argmax(h)) >= 0.10 AND random-head
       max-class cos q95 < 0.10
  (0.10 ~= 5 sigma: random cos to a fixed unit direction has
  std 1/sqrt(2560) ~ 0.02; frozen before readout.)

Arm C (causal ablation, CUDA): two registered write sites from the
aligned-census chain:
  MLP:  L35 neuron 1936 (2816 P-W1 best, metal, cos 0.6101)
  OV:   L32 head 0   (2817 P-A1 best, nature, proj 0.6805)
Ablation = zero the write path at its input (down_proj input channel /
o_proj input head slice), one forward per condition on the same 56
sentences.  Effect = Delta prof_i = prof_i(intact) - prof_i(ablated)
at the final hidden state, target-span mean, averaged over words.
  P-C1 mlp_write_causal: Delta_metal(neuron 1936) > 0 AND > q95 of
       10 random same-layer neurons' Delta_metal
  P-C2 ov_write_causal: Delta_nature(head 0) > 0 AND > q95 of 10
       random same-layer heads' Delta_nature
Logits top-8 shifts recorded descriptively.

Prereg frozen before any readout; verdicts only from P-V1/P-C1/P-C2.
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
OUT = BASE / 'phase2818' / 'write_causality'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SRC_2816 = BASE / 'phase2816' / 'layers_semantics_position'
SRC_2817 = BASE / 'phase2817' / 'ov_subspace_natural_position'
SEED = 2818
N_RAND = 10
PASS_THRESH = 0.40
SHARE_THRESH = 0.10

TARGETS = ['apple', 'gold', 'eagle', 'sedan', 'Spain', 'soup', 'cave',
           'sock']
PREFIX_WORDS = {
    0: [],
    1: ['the'],
    2: ['and', ' the'],
    3: ['and', ' then', ' the'],
    4: ['and', ' then', ' a', ' small'],
    5: ['and', ' then', ' a', ' very', ' small'],
    6: ['and', ' then', ' a', ' very', ' small', ' shiny'],
}
N_POS = 7

PREREG = {
    'P-V1': 'ov_write_actual iff median over 2817-pass heads of '
            'cos(delta_bar_h, unitD_argmax(h)) >= 0.10 AND random '
            'non-pass head max-class cos q95 < 0.10 (delta_bar = mean '
            'actual o_proj-path write over target spans, 56 natural '
            'sentences)',
    'P-C1': 'mlp_write_causal iff Delta prof_metal (intact - ablated, '
            'L35 neuron 1936 zeroed at down_proj input, final hidden '
            'state, span mean, word-mean) > 0 AND > q95 of 10 random '
            'same-layer neurons',
    'P-C2': 'ov_write_causal iff Delta prof_nature (L32 head 0 zeroed '
            'at o_proj input) > 0 AND > q95 of 10 random same-layer '
            'heads',
    'verdict': 'ov_write_actual = P-V1; mlp_write_causal = P-C1; '
               'ov_write_causal = P-C2',
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
    res2816 = json.loads((SRC_2816 / 'result.json').read_text(
        encoding='utf-8'))

    z17 = np.load(SRC_2817 / 'ovpos.npz')
    head_proj = z17['head_proj'].astype(np.float64)
    head_rq = z17['head_rand_q95'].astype(np.float64)
    n_layers, n_heads = head_proj.shape[:2]
    pass_list = []
    for L in range(n_layers):
        for h in range(n_heads):
            if head_rq[L, h] < PASS_THRESH:
                i = int(np.argmax(head_proj[L, h]))
                if head_proj[L, h, i] >= PASS_THRESH:
                    pass_list.append({'layer': L, 'head': h,
                                      'class': CAT_WORDS[i],
                                      'proj': round(
                                          float(head_proj[L, h, i]),
                                          4)})
    pass_layers = sorted({p['layer'] for p in pass_list})
    print('P2818 pass heads from 2817: %d across layers %s'
          % (len(pass_list), pass_layers), flush=True)

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED,
                 'pass_thresh': PASS_THRESH, 'share_thresh': SHARE_THRESH,
                 'n_rand': N_RAND,
                 'pass_heads': pass_list,
                 'mlp_site': res2816['verdict'][
                     'best_neuron_readouts']['down']['info'],
                 'targets': TARGETS,
                 'note': 'Arm V actual-write verification via o_proj '
                         'input capture; Arm C causal ablation at '
                         'write-path inputs (down_proj channel / '
                         'o_proj head slice), final-hidden-state class '
                         'profile effect vs random-site nulls'}
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
    hd = Wo0.shape[1] // n_heads
    assert Wo0.shape[1] % n_heads == 0
    cfg_hd = int(cfg.get('head_dim', 0) or 0)
    if cfg_hd:
        assert cfg_hd == hd, (cfg_hd, hd)
    del Wo0
    inter = int(cfg.get('intermediate_size', 0) or 0)

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
    print('P2818 gates: dW=%.2e Zeval=%.2e' % (gate_dW, gate_Z),
          flush=True)
    assert gate_dW < 1e-6 and gate_Z < 1e-4

    unitD = np.stack([unit(dW[i]) for i in range(10)])
    cls_idx = {c: i for i, c in enumerate(CAT_WORDS)}
    rng = np.random.default_rng(SEED)

    # ---------- CUDA setup + sentences (same as 2817) ----------
    import torch
    from transformers import AutoModelForCausalLM
    for w in TARGETS:
        tid(w)
    tail_ids = tok(' is here', add_special_tokens=False)['input_ids']
    assert len(tail_ids) == 2, tail_ids
    seqs, meta = [], []
    for w in TARGETS:
        for p in range(N_POS):
            words = PREFIX_WORDS[p]
            prefix_str = ' '.join(t.strip() for t in words)
            pre_ids = (tok(prefix_str, add_special_tokens=False)
                       ['input_ids']) if prefix_str else []
            assert len(pre_ids) == p, (w, p, prefix_str, pre_ids)
            full_str = ((prefix_str + ' ' if prefix_str else '')
                        + w + ' is here')
            ids = tok(full_str, add_special_tokens=False)['input_ids']
            assert ids[:len(pre_ids)] == pre_ids, (w, p)
            span_lo = len(pre_ids)
            span_hi = len(ids) - len(tail_ids)
            assert 1 <= span_hi - span_lo <= 2, (w, p)
            assert ids[span_hi:] == tail_ids, (w, p)
            seqs.append(ids)
            meta.append((w, p, span_lo, span_hi))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None \
        else tok.eos_token_id
    maxlen = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), maxlen), int(pad_id),
                           dtype=torch.long)
    attn_mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        attn_mask[i, :len(s)] = 1

    model = AutoModelForCausalLM.from_pretrained(
        str(mdir), dtype=torch.bfloat16, device_map='auto')
    model.eval()
    dev = model.device

    def run_forward(capture_layers=(), ablate=None):
        """One batched forward; capture o_proj inputs at capture_layers;
        ablate = None | ('mlp', L, j) | ('ov', L, h)."""
        handles, store = [], {}
        for L in capture_layers:
            def cap_hook(mod, args, L=L):
                store[L] = args[0].detach().float().cpu().numpy()
            handles.append(
                model.model.layers[L].self_attn.o_proj
                .register_forward_pre_hook(cap_hook))
        if ablate is not None:
            kind, L, j = ablate
            if kind == 'mlp':
                def abl(mod, args, j=j):
                    x = args[0].clone()
                    x[..., j] = 0
                    return (x,)
                handles.append(
                    model.model.layers[L].mlp.down_proj
                    .register_forward_pre_hook(abl))
            else:
                def abl(mod, args, j=j):
                    x = args[0].clone()
                    x[..., j * hd:(j + 1) * hd] = 0
                    return (x,)
                handles.append(
                    model.model.layers[L].self_attn.o_proj
                    .register_forward_pre_hook(abl))
        with torch.no_grad():
            out = model(input_ids=input_ids.to(dev),
                        attention_mask=attn_mask.to(dev),
                        output_hidden_states=True)
        for hh in handles:
            hh.remove()
        HS_last = out.hidden_states[-1].float().cpu().numpy()
        logits = out.logits.float().cpu().numpy()
        return HS_last, logits, store

    def span_profiles(HS_last):
        profs = np.zeros((len(TARGETS), 10))
        for wi in range(len(TARGETS)):
            vecs = np.stack([HS_last[wi * N_POS + p,
                                     meta[wi * N_POS + p][2]:
                                     meta[wi * N_POS + p][3], :]
                             .mean(0) for p in range(N_POS)])
            profs[wi] = vecs.mean(0) @ unitD.T
        return profs

    # ---------- intact forward (captures pass-head inputs) ----------
    HS0, LG0, store = run_forward(capture_layers=pass_layers)
    prof0 = span_profiles(HS0)
    print('P2818 intact forward done (%.1fs)'
          % (time.monotonic() - t0), flush=True)

    # ---------- Arm V: actual write of pass heads ----------
    Wh_cache = {}
    for p in pass_list:
        L, h = p['layer'], p['head']
        if (L, h) not in Wh_cache:
            Wo = read_tensor('model.layers.%d.self_attn.o_proj.weight'
                             % L).astype(np.float32)
            Wh_cache[(L, h)] = Wo[:, h * hd:(h + 1) * hd].copy()
            del Wo
        if (L, 'store') not in Wh_cache:
            Wh_cache[(L, 'store')] = store[L]
    arm_v_rows = []
    rand_cos = []
    for p in pass_list:
        L, h = p['layer'], p['head']
        X = Wh_cache[(L, 'store')]         # (B, T, n_heads*hd)
        a = X[:, :, h * hd:(h + 1) * hd]   # (B, T, hd)
        W_h = Wh_cache[(L, h)]             # (2560, hd)
        deltas = []
        for bi in range(len(seqs)):
            lo, hi = meta[bi][2], meta[bi][3]
            deltas.append(a[bi, lo:hi].mean(0) @ W_h.T)
        db = unit(np.mean(np.stack(deltas), axis=0).astype(np.float64))
        # abs, consistent with the 2816/2817 census protocol
        c = float(abs(db @ unitD[cls_idx[p['class']]]))
        arm_v_rows.append({**p, 'actual_cos': round(c, 4)})
        # same-layer random non-pass heads, max-over-classes |cos|
        free = [hh for hh in range(n_heads)
                if hh != h and head_rq[L, hh] < PASS_THRESH
                and head_proj[L, hh].max() < PASS_THRESH]
        picks = rng.choice(free, size=min(N_RAND, len(free)),
                           replace=False)
        for hh in picks:
            if (L, hh) not in Wh_cache:
                Wo = read_tensor('model.layers.%d.self_attn.o_proj.weight'
                                 % L).astype(np.float32)
                Wh_cache[(L, hh)] = Wo[:, hh * hd:(hh + 1) * hd].copy()
                del Wo
            dbrs = []
            for bi in range(len(seqs)):
                lo, hi = meta[bi][2], meta[bi][3]
                ahh = X[bi, lo:hi, hh * hd:(hh + 1) * hd].mean(0)
                dbrs.append(ahh @ Wh_cache[(L, hh)].T)
            dbr = unit(np.mean(np.stack(dbrs), axis=0).astype(np.float64))
            rand_cos.append(float(np.max(np.abs(dbr @ unitD.T))))
    med_pass = float(np.median([r['actual_cos'] for r in arm_v_rows]))
    rq95 = float(np.quantile(rand_cos, 0.95)) if rand_cos else 0.0
    p_v1 = bool(med_pass >= SHARE_THRESH and rq95 < SHARE_THRESH)
    print('P2818 Arm V median actual_cos=%.4f rand q95=%.4f P-V1=%s'
          % (med_pass, rq95, p_v1), flush=True)
    print('P2818 Arm V rows %s' % json.dumps(arm_v_rows), flush=True)

    # ---------- Arm C: causal ablation ----------
    site_mlp = execution['mlp_site']
    Lm, jm = int(site_mlp['layer']), int(site_mlp['neuron'])
    best_ov = [p for p in pass_list]
    best_ov.sort(key=lambda p: -p['proj'])
    Lo, ho = best_ov[0]['layer'], best_ov[0]['head']
    cls_mlp = site_mlp['class']
    cls_ov = best_ov[0]['class']
    print('P2818 Arm C sites: mlp L%d n%d (%s), ov L%d h%d (%s)'
          % (Lm, jm, cls_mlp, Lo, ho, cls_ov), flush=True)

    def delta(prof_abl, cls):
        i = cls_idx[cls]
        d = prof0[:, i] - prof_abl[:, i]
        return float(d.mean()), d.tolist()

    # MLP site + random neurons (same layer, j != jm)
    Wd_shape = read_tensor('model.layers.%d.mlp.down_proj.weight'
                           % Lm).shape
    assert inter == Wd_shape[1], (inter, Wd_shape)
    del Wd_shape
    d_mlp, d_mlp_per_word = delta(span_profiles(
        run_forward(ablate=('mlp', Lm, jm))[0]), cls_mlp)
    rand_neu = []
    js = rng.choice([j for j in range(inter) if j != jm],
                    size=N_RAND, replace=False)
    for j in js:
        dj, _ = delta(span_profiles(
            run_forward(ablate=('mlp', Lm, int(j)))[0]), cls_mlp)
        rand_neu.append(dj)
    q_mlp = float(np.quantile(rand_neu, 0.95))
    p_c1 = bool(d_mlp > 0 and d_mlp > q_mlp)

    d_ov, d_ov_per_word = delta(span_profiles(
        run_forward(ablate=('ov', Lo, ho))[0]), cls_ov)
    rand_head = []
    hs_pool = [h for h in range(n_heads) if h != ho]
    hsel = rng.choice(hs_pool, size=N_RAND, replace=False)
    for h in hsel:
        dh, _ = delta(span_profiles(
            run_forward(ablate=('ov', Lo, int(h)))[0]), cls_ov)
        rand_head.append(dh)
    q_ov = float(np.quantile(rand_head, 0.95))
    p_c2 = bool(d_ov > 0 and d_ov > q_ov)
    print('P2818 Arm C: d_mlp=%.4f rand_q95=%.4f P-C1=%s | '
          'd_ov=%.4f rand_q95=%.4f P-C2=%s'
          % (d_mlp, q_mlp, p_c1, d_ov, q_ov, p_c2), flush=True)

    # descriptive logits shifts at last target token
    def top8(lg):
        row = lg[-1, meta[-1][3] - 1, :]
        ids = np.argsort(-row)[:8].tolist()
        return [tok.decode([t]).strip() for t in ids]
    li = meta[-1][3] - 1
    d_logit_mlp = float(np.abs(LG0[-1, li] -
                               run_forward(ablate=('mlp', Lm, jm))[1][-1, li]).max())

    verdict = {
        'pass_heads': pass_list,
        'pass_layers': pass_layers,
        'arm_v_rows': arm_v_rows,
        'median_actual_cos': round(med_pass, 4),
        'rand_head_cos_q95': round(rq95, 4),
        'ov_write_actual': p_v1,
        'mlp_site': site_mlp, 'ov_site': best_ov[0],
        'd_mlp': round(d_mlp, 4),
        'd_mlp_per_word': [round(x, 4) for x in d_mlp_per_word],
        'rand_neuron_deltas': [round(x, 4) for x in rand_neu],
        'rand_neuron_q95': round(q_mlp, 4),
        'mlp_write_causal': p_c1,
        'd_ov': round(d_ov, 4),
        'd_ov_per_word': [round(x, 4) for x in d_ov_per_word],
        'rand_head_deltas': [round(x, 4) for x in rand_head],
        'rand_head_q95': round(q_ov, 4),
        'ov_write_causal': p_c2,
        'top8_intact': top8(LG0),
        'max_logit_shift_mlp': round(d_logit_mlp, 4),
        'load_run_seconds': round(time.monotonic() - t0, 1),
    }
    result = {'phase': 2818, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'causal.npz',
           prof_intact=prof0.astype(np.float32),
           unitD=unitD.astype(np.float32),
           head_proj=head_proj.astype(np.float32),
           head_rand_q95=head_rq.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2818', elapsed)
    print('P2818 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2818 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
