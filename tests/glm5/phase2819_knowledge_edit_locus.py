"""Phase 2819 (LPF-32): CAUSAL COMPLETION + KNOWLEDGE-EDIT LOCUS.

Arm R (2818 hard-gap repair): causal ablation for the two remaining
actually-writing OV heads (L34 h0 metal, L32 h3 clothing) under the
exact 2818 protocol (56 natural sentences, span-mean final-hidden
class profile, 5+5 random-head nulls), plus descriptive logit shifts
of class words at the sentence end (logit lens).
  R-P1: head L34 h0 causally writes metal: Delta prof_metal > 0 AND
        > q95(5 random heads)
  R-P2: head L32 h3 causally writes clothing: same with clothing
  (logit shifts recorded descriptively)

Arm K (user question, 2026-09-17: "to edit knowledge - apple's colour
from red to black - WHICH parameters must change?"): localize where
the fact 'apple -> red' lives, by three independent probes with the
same behaviour endpoint = logit of the target colour word after
'The {s} is'.
  K1 layer locus: zero ALL attention outputs (o_proj input) and ALL
     MLP outputs (down_proj input) layer by layer (36+36 forwards);
     Delta logit_target(subj-mean) per layer.  Random-word logits
     ('table','lamp','book') as specificity null.
     K-P1 knowledge_layer_locus: exists L with Delta <= -1.0 AND
          random-word median delta at that L > -1.0
  K2 direction census (zero-forward): 2807-protocol contrast
     directions for the colour family:
       dW_color = unit(mean_z(colour words) - mean of 10 class centres)
       dW_red   = unit(z(red) - mean_z(other colour words))
     census vs MLP down_proj columns (max |cos|) and OV head column
     spaces (projection length), random 30-vec q95 per layer.
     K-P2 color_params_present: exists MLP column or OV head with
          alignment >= 0.30 AND that layer random q95 < 0.30
  K3 causal sites: ablate top-3 aligned MLP columns, top-3 aligned OV
     heads (by the census), individually and jointly; endpoint as K1.
     K-P3 site_causal: best-site ablation Delta logit_target <= -0.3
          AND < random-site q95 (5 random same-type sites)
  Also: zeroing the apple embedding row (where does the fact enter),
     and intact full-sentence check 'The apple is red'.

Subjects (typical colour): apple->red, sky->blue, grass->green,
coal->black, banana->yellow, blood->red.

Prereg frozen before any readout; verdicts only from the P-* above.
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
OUT = BASE / 'phase2819' / 'knowledge_edit_locus'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SRC_2816 = BASE / 'phase2816' / 'layers_semantics_position'
SRC_2817 = BASE / 'phase2817' / 'ov_subspace_natural_position'
SEED = 2819
N_RAND_VEC = 30
N_RAND_SITES = 5
THRESH_ALIGN = 0.30
THRESH_LAYER = -1.0
THRESH_SITE = -0.3
LAYER_LO, LAYER_HI = 4, 35

COLOR_WORDS = ['red', 'green', 'blue', 'black', 'white', 'yellow',
               'brown', 'pink', 'gray', 'purple', 'orange', 'crimson']
SUBJ = [('apple', 'red'), ('sky', 'blue'), ('grass', 'green'),
        ('coal', 'black'), ('banana', 'yellow'), ('blood', 'red')]
RAND_WORDS = ['table', 'lamp', 'book']
PREFIX_WORDS = {  # 2817 templates (Arm R only)
    0: [], 1: ['the'], 2: ['and', ' the'], 3: ['and', ' then', ' the'],
    4: ['and', ' then', ' a', ' small'],
    5: ['and', ' then', ' a', ' very', ' small'],
    6: ['and', ' then', ' a', ' very', ' small', ' shiny'],
}
TARGETS2818 = ['apple', 'gold', 'eagle', 'sedan', 'Spain', 'soup',
               'cave', 'sock']
N_POS = 7

PREREG = {
    'R-P1': 'head L34 h0 causal metal: Delta prof_metal (intact - '
            'ablated, o_proj head-slice zero, final hidden, span mean, '
            'word-mean, 56 sentences) > 0 AND > q95(5 random heads)',
    'R-P2': 'head L32 h3 causal clothing: same with clothing class',
    'K-P1': 'knowledge_layer_locus iff exists layer L with '
            'Delta logit_target (subj-mean, output-zeroed at L) '
            '<= -1.0 AND random-word median delta at that L > -1.0 '
            '(MLP and attention scans judged separately)',
    'K-P2': 'color_params_present iff exists MLP column or OV head '
            'with max(|cos(col,dW_color)|, |cos(col,dW_red)|) >= 0.30 '
            'or OV projection >= 0.30 AND that layer random q95 < 0.30',
    'K-P3': 'site_causal iff best aligned-site ablation '
            'Delta logit_target <= -0.3 AND < q95 of 5 random '
            'same-type sites',
    'verdict': 'knowledge_edit_locus answered by K-P1 (layers), '
               'K-P2 (direction-aligned params), K-P3 (causal sites); '
               'R-P1/R-P2 complete the 2818 causal map',
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

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED,
                 'color_words': COLOR_WORDS,
                 'subjects': SUBJ, 'rand_words': RAND_WORDS,
                 'thresh_align': THRESH_ALIGN,
                 'thresh_layer': THRESH_LAYER,
                 'thresh_site': THRESH_SITE,
                 'n_rand_vec': N_RAND_VEC,
                 'n_rand_sites': N_RAND_SITES,
                 'note': 'Arm R completes the 2818 causal map '
                         '(L34 h0 metal, L32 h3 clothing) with logit '
                         'lens; Arm K localizes the apple->red fact '
                         'via layer-output zeroing scans, colour '
                         'direction census, and causal site ablation; '
                         'endpoint = target colour logit after '
                         '"The {s} is"'}
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
    assert Wo0.shape[1] % n_heads == 0 and d_model == Etab.shape[1]
    del Wo0
    inter = read_tensor('model.layers.0.mlp.down_proj.weight').shape[1]

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
        return (Etab[tid(t)].astype(np.float64)
                / np.sqrt((Etab[tid(t)] ** 2).mean() + eps) * g)

    # colour contrast directions (2807-style, registered here)
    color_cent = np.stack([zw(w) for w in COLOR_WORDS]).mean(0)
    dW_color = unit(color_cent - Cm.mean(0))
    red_dir = unit(zw('red') - np.stack(
        [zw(w) for w in COLOR_WORDS if w != 'red']).mean(0))
    for w in COLOR_WORDS + RAND_WORDS + ['The'] + [s for s, _ in SUBJ]:
        tid(w)

    # ---------- CUDA ----------
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        str(mdir), dtype=torch.bfloat16, device_map='auto')
    model.eval()
    dev = model.device

    # ===== Arm K sentence batch (uniform 3 tokens) =====
    k_seqs = [[tid('The'), tid(' ' + s), tid(' is')] for s, _ in SUBJ]
    k_ids = torch.tensor(k_seqs, dtype=torch.long)
    k_apple_pos = 1                       # ' apple' index in each seq

    def run_k(ablate=None):
        """Forward on the 6-subject batch; returns last-position
        logits (6, vocab).  ablate: None | ('attn', L) | ('mlp', L) |
        ('mlp_col', L, j) | ('ov', L, h) | 'emb_apple'."""
        handles = []
        if ablate == 'emb_apple':
            aid = tid(' apple')

            def emb_hook(mod, args, output, aid=aid):
                out = output.clone()
                pos = (k_ids.to(dev) == aid)
                out[pos] = 0
                return out
            handles.append(
                model.model.embed_tokens.register_forward_hook(emb_hook))
        elif ablate is not None:
            kind = ablate[0]
            if kind == 'attn':
                def abl(mod, args):
                    return (torch.zeros_like(args[0]),)
                handles.append(
                    model.model.layers[ablate[1]].self_attn.o_proj
                    .register_forward_pre_hook(abl))
            elif kind == 'mlp':
                def abl(mod, args):
                    return (torch.zeros_like(args[0]),)
                handles.append(
                    model.model.layers[ablate[1]].mlp.down_proj
                    .register_forward_pre_hook(abl))
            elif kind == 'mlp_col':
                def abl(mod, args, j=ablate[2]):
                    x = args[0].clone()
                    x[..., j] = 0
                    return (x,)
                handles.append(
                    model.model.layers[ablate[1]].mlp.down_proj
                    .register_forward_pre_hook(abl))
            elif kind == 'ov':
                def abl(mod, args, h=ablate[2]):
                    x = args[0].clone()
                    x[..., h * hd:(h + 1) * hd] = 0
                    return (x,)
                handles.append(
                    model.model.layers[ablate[1]].self_attn.o_proj
                    .register_forward_pre_hook(abl))
        with torch.no_grad():
            out = model(input_ids=k_ids.to(dev),
                        output_hidden_states=False)
        for hh in handles:
            hh.remove()
        return out.logits[:, -1, :].float().cpu().numpy()

    LG0 = run_k()
    tgt_idx = [tid(' ' + c) for _, c in SUBJ]
    rand_idx = [tid(' ' + w) for w in RAND_WORDS]

    def k_delta(lg):
        """subject-mean target-colour logit delta vs intact."""
        d = lg - LG0
        return float(np.mean([d[i, tgt_idx[i]] for i in range(len(SUBJ))]))

    def k_delta_rand(lg):
        d = lg - LG0
        return d[:, rand_idx].mean(axis=0).tolist()

    # K1: layer-output zeroing scans
    attn_scan, mlp_scan = [], []
    attn_rand, mlp_rand = [], []
    for L in range(n_layers):
        lgA = run_k(ablate=('attn', L))
        attn_scan.append(k_delta(lgA))
        attn_rand.append(k_delta_rand(lgA))
        lgM = run_k(ablate=('mlp', L))
        mlp_scan.append(k_delta(lgM))
        mlp_rand.append(k_delta_rand(lgM))
    attn_scan, mlp_scan = np.array(attn_scan), np.array(mlp_scan)
    attn_rand, mlp_rand = np.array(attn_rand), np.array(mlp_rand)
    ml_best = int(np.argmin(mlp_scan))
    at_best = int(np.argmin(attn_scan))
    p_k1_mlp = bool(mlp_scan[ml_best] <= THRESH_LAYER and
                    np.median(mlp_rand[ml_best]) > THRESH_LAYER)
    p_k1_attn = bool(attn_scan[at_best] <= THRESH_LAYER and
                     np.median(attn_rand[at_best]) > THRESH_LAYER)
    p_k1 = p_k1_mlp or p_k1_attn
    print('P2819 K1 mlp scan min=%.3f@L%d attn min=%.3f@L%d '
          'rand_med@best mlp=%.3f P-K1=%s'
          % (mlp_scan[ml_best], ml_best, attn_scan[at_best], at_best,
             np.median(mlp_rand[ml_best]), p_k1), flush=True)

    # K2: direction census (zero-forward)
    rand_vs = np.stack([unit(rng.standard_normal(d_model))
                        for _ in range(N_RAND_VEC)])
    mlp_rows, ov_rows = [], []
    for L in range(n_layers):
        Wd = read_tensor('model.layers.%d.mlp.down_proj.weight'
                         % L).astype(np.float32)
        cn = np.maximum(np.linalg.norm(Wd, axis=0), 1e-12)
        Wn = Wd / cn
        cc_cos = np.abs(Wn.T @ dW_color.astype(np.float32))
        cr_cos = np.abs(Wn.T @ red_dir.astype(np.float32))
        rnd = [float(np.abs(Wn.T @ v.astype(np.float32)).max())
               for v in rand_vs]
        rq = float(np.quantile(rnd, 0.95))
        j = int(np.argmax(np.maximum(cc_cos, cr_cos)))
        mlp_rows.append({'layer': L,
                         'color_max': round(float(cc_cos.max()), 4),
                         'red_max': round(float(cr_cos.max()), 4),
                         'best_col': j,
                         'best_kind': 'color' if cc_cos[j] >= cr_cos[j]
                         else 'red',
                         'best_val': round(float(max(cc_cos[j],
                                                     cr_cos[j])), 4),
                         'rand_q95': round(rq, 4)})
        del Wd, Wn
        Wo = read_tensor('model.layers.%d.self_attn.o_proj.weight'
                         % L).astype(np.float32)
        eye = np.eye(hd, dtype=np.float32)
        for h in range(n_heads):
            Wh = Wo[:, h * hd:(h + 1) * hd]
            G = Wh.T @ Wh
            Gj = G + eye * (1e-6 * float(np.trace(G)) / hd)
            row = {'layer': L, 'head': h}
            best_v, best_k = 0.0, None
            for name, dv in (('color', dW_color), ('red', red_dir)):
                c = Wh.T @ dv.astype(np.float32)
                X = np.linalg.solve(Gj, c)
                p2 = float(np.maximum((c * X).sum(), 0.0))
                v = float(np.sqrt(min(p2, 1.0)))
                row[name + '_proj'] = round(v, 4)
                if v > best_v:
                    best_v, best_k = v, name
            row['best_val'], row['best_kind'] = round(best_v, 4), best_k
            Cr = Wh.T @ np.stack([v.astype(np.float32) for v in rand_vs]).T
            Xr = np.linalg.solve(Gj, Cr)
            pr2 = np.maximum((Cr * Xr).sum(0), 0.0)
            row['rand_q95'] = round(float(np.quantile(
                np.sqrt(np.minimum(pr2, 1.0)), 0.95)), 4)
            ov_rows.append(row)
        del Wo
    mlp_pass = [r for r in mlp_rows if r['best_val'] >= THRESH_ALIGN
                and r['rand_q95'] < THRESH_ALIGN]
    ov_pass = [r for r in ov_rows if r['best_val'] >= THRESH_ALIGN
               and r['rand_q95'] < THRESH_ALIGN]
    p_k2 = bool(mlp_pass or ov_pass)
    print('P2819 K2 mlp_pass=%d ov_pass=%d P-K2=%s'
          % (len(mlp_pass), len(ov_pass), p_k2), flush=True)

    # K3: causal ablation of top aligned sites
    mlp_top = sorted(mlp_rows, key=lambda r: -r['best_val'])[:3]
    ov_top = sorted(ov_rows, key=lambda r: -r['best_val'])[:3]
    site_res = []
    for r in mlp_top:
        lg = run_k(ablate=('mlp_col', r['layer'], r['best_col']))
        site_res.append({'kind': 'mlp_col', 'layer': r['layer'],
                         'index': r['best_col'],
                         'dir': r['best_kind'],
                         'delta': round(k_delta(lg), 4)})
    for r in ov_top:
        lg = run_k(ablate=('ov', r['layer'], r['head']))
        site_res.append({'kind': 'ov_head', 'layer': r['layer'],
                         'index': r['head'], 'dir': r['best_kind'],
                         'delta': round(k_delta(lg), 4)})
    # random same-type nulls
    rand_site = []
    for _ in range(N_RAND_SITES):
        L = int(rng.integers(LAYER_LO, n_layers))
        j = int(rng.integers(0, inter))
        rand_site.append(k_delta(run_k(ablate=('mlp_col', L, j))))
    rand_site_q95 = float(np.quantile(rand_site, 0.95))
    best_site = min(site_res, key=lambda r: r['delta'])
    p_k3 = bool(best_site['delta'] <= THRESH_SITE and
                best_site['delta'] < rand_site_q95)
    print('P2819 K3 sites %s rand_q95=%.4f best=%s P-K3=%s'
          % (json.dumps(site_res), rand_site_q95,
            json.dumps(best_site), p_k3), flush=True)

    # emb apple ablation
    d_emb = k_delta(run_k(ablate='emb_apple'))
    d_emb_rand = k_delta_rand(run_k(ablate='emb_apple'))

    # ===== Arm R: 2818 protocol completion =====
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
            span_lo = len(pre_ids)
            span_hi = len(ids) - 2
            assert 1 <= span_hi - span_lo <= 2
            r_seqs.append(ids)
            r_meta.append((w, p, span_lo, span_hi))
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
            kind, L, h = ablate

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
        return (out.hidden_states[-1].float().cpu().numpy(),
                out.logits.float().cpu().numpy())

    def span_profiles(HS_last):
        profs = np.zeros((len(TARGETS2818), 10))
        for wi in range(len(TARGETS2818)):
            vecs = np.stack([HS_last[wi * N_POS + p,
                                     r_meta[wi * N_POS + p][2]:
                                     r_meta[wi * N_POS + p][3], :]
                             .mean(0) for p in range(N_POS)])
            profs[wi] = vecs.mean(0) @ unitD.T
        return profs

    HS0, LG0r = run_r()
    prof0 = span_profiles(HS0)

    def r_delta(lg_cls, prof_abl):
        d = prof0[:, cls_idx[lg_cls]] - prof_abl[:, cls_idx[lg_cls]]
        return float(d.mean())

    # R sites: L34 h0 (metal), L32 h3 (clothing)
    r_sites = [(34, 0, 'metal'), (32, 3, 'clothing')]
    r_res = []
    for (L, h, cls) in r_sites:
        d = r_delta(cls, span_profiles(run_r(ablate=('ov', L, h))[0]))
        rand_d = []
        for _ in range(N_RAND_SITES):
            hh = int(rng.integers(0, n_heads))
            if hh == h:
                hh = (hh + 1) % n_heads
            rand_d.append(r_delta(cls, span_profiles(
                run_r(ablate=('ov', L, hh))[0])))
        q95 = float(np.quantile(rand_d, 0.95))
        r_res.append({'layer': L, 'head': h, 'class': cls,
                      'delta': round(d, 4), 'rand_q95': round(q95, 4),
                      'pass': bool(d > 0 and d > q95)})
        print('P2818-R L%d h%d %s delta=%.4f q95=%.4f'
              % (L, h, cls, d, q95), flush=True)
    p_r1 = r_res[0]['pass']
    p_r2 = r_res[1]['pass']

    # descriptive logit lens: class-word logits at sentence end
    cls_words = {'metal': ['gold', 'silver', 'iron'],
                 'clothing': ['shirt', 'sock', 'hat']}
    lens = {}
    for (L, h, cls) in r_sites:
        _, LGa = run_r(ablate=('ov', L, h))
        wi = [tid(' ' + w) for w in cls_words[cls]]
        lens['%d_%d_%s' % (L, h, cls)] = round(float(np.mean(
            LG0r[:, -1, :][:, wi] - LGa[:, -1, :][:, wi])), 4)

    verdict = {
        'K1': {'mlp_scan': [round(x, 4) for x in mlp_scan],
               'attn_scan': [round(x, 4) for x in attn_scan],
               'mlp_best_layer': ml_best,
               'attn_best_layer': at_best,
               'mlp_rand_med_at_best': round(
                   float(np.median(mlp_rand[ml_best])), 4),
               'knowledge_layer_locus': p_k1,
               'knowledge_layer_locus_mlp': p_k1_mlp,
               'knowledge_layer_locus_attn': p_k1_attn},
        'K2': {'mlp_rows': mlp_rows, 'ov_rows': ov_rows,
               'mlp_pass_n': len(mlp_pass), 'ov_pass_n': len(ov_pass),
               'color_params_present': p_k2},
        'K3': {'sites': site_res, 'rand_site_q95': round(rand_site_q95,
                                                         4),
               'best_site': best_site, 'site_causal': p_k3,
               'emb_apple_delta': round(d_emb, 4),
               'emb_apple_rand_delta': [round(x, 4)
                                        for x in d_emb_rand]},
        'R': {'sites': r_res, 'r_p1_metal': p_r1,
              'r_p2_clothing': p_r2, 'logit_lens': lens},
        'load_run_seconds': round(time.monotonic() - t0, 1),
    }
    result = {'phase': 2819, 'prereg': PREREG, 'verdict': verdict}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'locus.npz',
           mlp_scan=mlp_scan.astype(np.float32),
           attn_scan=attn_scan.astype(np.float32),
           mlp_rand=mlp_rand.astype(np.float32),
           attn_rand=attn_rand.astype(np.float32),
           unitD=unitD.astype(np.float32),
           dW_color=dW_color.astype(np.float32),
           dW_red=red_dir.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2819', elapsed)
    print('P2819 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2819 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
