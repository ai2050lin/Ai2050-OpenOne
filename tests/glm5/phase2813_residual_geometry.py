"""Phase 2813 (LPF-26): RESIDUAL GEOMETRY CENSUS — what IS the 91%?

2811: class spike 9%, residual near-full-rank 91%.  2812: adjective-
anchor attribute axes fail to structure the residual (share ~ pool
baseline, split-half 0/14).  Now attack the residual directly:
  Arm R  PCA of the class-removed residual (283 words) — spectrum,
         top-20 PCs read out through the unembed (Wu=Etab, tied),
         cos against class directions (dW) and 2812 attribute axes (dA)
  Arm N  noun-pole attribute axes (POS-mismatch fix for 2812):
         size/temperature/speed/danger/weight/brightness from noun
         pairs (mountain/ant style), share + split-half stability with
         the 2812 lesson applied — identity permutation EXCLUDED from
         the stability null

Zero-forward, 2808 protocol (tie-aware); battery rebuilt identical to
2811 (gates: dW vs 2807 <1e-6, eval_words exact vs 2811, Z vs 2811
battery.npz <1e-4); dA loaded from 2812 axes.npz (read-only).

Prereg (frozen before any readout):
  P-R1  residual_lowdim iff top-20 PC cumulative variance (centered,
        class-removed, 283 words) >= 0.50
  P-R2  pc_class_decoupled iff max |cos(PC_i, dW_j)| over i<=20,
        j<=9 < 0.30  (validity gate: PCs are not re-deriving class axes)
  P-R3a nounpole_share_real iff noun-pole attribute subspace share on
        eval > N1 pole-shuffle x1000 q95 AND > 3 x median
  P-R3b nounpole_stable iff >= 60% of noun-pole axes have split-half
        |cos| > N2 exclude-identity pole-shuffle x300 q95
  P-R3  nounpole_axes_real = P-R3a AND P-R3b
  verdict: residual_geometry = 'lowdim' iff P-R1 AND P-R2;
        nounpole_support = P-R3
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
OUT = BASE / 'phase2813' / 'residual_geometry'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_2807 = BASE / 'phase2807' / 'qwen4_heldout'
SRC_2811 = BASE / 'phase2811' / 'noun_superposition'
SRC_2812 = BASE / 'phase2812' / 'attribute_subspace'
SEED = 2813
N_SHUFFLE = 1000
N_HALF = 300
TOP_PC = 20

NPOLE = {
    'size': (['mountain', 'giant', 'mammoth', 'tower', 'whale',
              'elephant'],
             ['ant', 'atom', 'dwarf', 'mite', 'grain', 'pebble']),
    'temperature': (['fire', 'lava', 'flame', 'furnace', 'desert',
                     'bonfire'],
                    ['ice', 'snow', 'frost', 'winter', 'icicle',
                     'freezer']),
    'speed': (['lightning', 'rocket', 'bullet', 'jet', 'express',
               'sprint'],
              ['snail', 'tortoise', 'statue', 'sloth', 'stone',
               'anchor']),
    'danger': (['venom', 'grenade', 'piranha', 'wolf', 'scorpion',
                'quicksand'],
               ['pillow', 'cotton', 'bubble', 'feather', 'sponge',
                'marshmallow']),
    'weight': (['boulder', 'anvil', 'truck', 'piano', 'fridge', 'vault'],
               ['balloon', 'paper', 'soap', 'cloudlet', 'confetti',
                'cobweb']),
    'brightness': (['sun', 'lamp', 'torch', 'neon', 'flashlight',
                    'candle'],
                   ['cave', 'coal', 'midnight', 'dungeon', 'shadow',
                    'eclipse']),
}

PREREG = {
    'P-R1': 'residual_lowdim iff top-20 PC cumulative variance '
            '(centered, class-removed, 283 words) >= 0.50',
    'P-R2': 'pc_class_decoupled iff max |cos(PC_i, dW_j)| over i<=20, '
            'j<=9 < 0.30 (validity gate)',
    'P-R3a': 'nounpole_share_real iff noun-pole attribute subspace '
             'share on eval > N1 pole-shuffle q95 AND > 3 x median',
    'P-R3b': 'nounpole_stable iff >= 60% of noun-pole axes have '
             'split-half |cos| > N2 exclude-identity pole-shuffle q95',
    'P-R3': 'nounpole_axes_real = P-R3a AND P-R3b',
    'verdict': "residual_geometry = 'lowdim' iff P-R1 AND P-R2; "
               'nounpole_support = P-R3',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())
    exec2807 = json.loads((SRC_2807 / 'execution.json').read_text(
        encoding='utf-8'))
    HELD = exec2807['held']
    assert list(HELD.keys()) == CAT_WORDS
    res2811 = json.loads((SRC_2811 / 'result.json').read_text(
        encoding='utf-8'))

    execution = {'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
                 'prereg': PREREG, 'seed': SEED,
                 'n_shuffle': N_SHUFFLE, 'n_half': N_HALF,
                 'top_pc': TOP_PC, 'npole_candidates': NPOLE,
                 'note': 'residual = centered Z_all minus class-subspace '
                         'projection; noun-pole axes fix 2812 POS '
                         'mismatch; stability null excludes identity '
                         'assignment (2812 lesson)'}
    fc.save(OUT / 'execution.json', execution)

    # ---------- tensors (2808 protocol, tie-aware) ----------
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
        tie = False
    except KeyError:
        Wu = Etab
        tie = True
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

    atlas_words = [w for v in CATS.values() for w in v]
    new_kept = res2811['verdict']['new_survivors']
    eval_pairs = [(w, c) for c in CAT_WORDS for w in HELD[c]] \
        + [(w, c) for c in CAT_WORDS for w in new_kept[c]]
    eval_words = [w for w, _ in eval_pairs]
    ytrue = np.array([CAT_WORDS.index(c) for _, c in eval_pairs])

    # ---------- gates ----------
    assert eval_words == res2811['eval_words'], 'eval battery drift'
    h7 = np.load(SRC_2807 / 'heldout.npz')
    cent = {c: np.stack([Wu[tid(w)].astype(np.float64)
                         for w in CATS[c]]).mean(0) for c in CAT_WORDS}
    Cm = np.stack([cent[c] for c in CAT_WORDS])
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    gate_dW = float(np.abs(dW - h7['dW_class'].astype(np.float64)).max())
    b11 = np.load(SRC_2811 / 'battery.npz')

    def zrow(t):
        e = Etab[tid(t)].astype(np.float64)
        return e / np.sqrt((e ** 2).mean() + eps) * g

    Z_eval = np.stack([zrow(w) for w in eval_words])
    Z_atlas = np.stack([zrow(w) for w in atlas_words])
    gate_Z = float(np.abs(Z_eval - b11['Z_eval'].astype(np.float64)).max())
    dA12 = np.load(SRC_2812 / 'axes.npz')['dA'].astype(np.float64)
    print('P2813 gates: dW_vs_2807=%.2e Z_vs_2811=%.2e dA12=%s tie=%s'
          % (gate_dW, gate_Z, dA12.shape, tie), flush=True)
    assert gate_dW < 1e-6 and gate_Z < 1e-4 and dA12.shape == (14, 2560)

    # ---------- Arm R: residual PCA ----------
    unitD = np.stack([unit(dW[i]) for i in range(10)])
    classQ, _ = np.linalg.qr(unitD.T)
    Z_all = np.vstack([Z_atlas, Z_eval])
    all_words = atlas_words + eval_words
    Zc = Z_all - Z_all.mean(0, keepdims=True)
    resid = Zc - (Zc @ classQ) @ classQ.T
    U, s, Vt = np.linalg.svd(resid, full_matrices=False)
    var = s ** 2 / (s ** 2).sum()
    cum20 = float(var[:TOP_PC].sum())
    p_r1 = bool(cum20 >= 0.50)
    print('P2813 residual spectrum: cum10=%.3f cum20=%.3f cum50=%.3f '
          'P-R1=%s' % (float(var[:10].sum()), cum20,
                       float(var[:50].sum()), p_r1), flush=True)

    unitdW = unitD
    pc_rows = []
    max_cos_dW = 0.0
    for i in range(TOP_PC):
        pc = Vt[i]
        cos_dW = unitdW @ pc
        cos_dA = dA12 @ pc
        top_ids = np.argsort(-(Wu @ pc))[:10].tolist()
        toks = [tok.decode([t]).strip() for t in top_ids]
        m1 = float(np.abs(cos_dW).max())
        max_cos_dW = max(max_cos_dW, m1)
        pc_rows.append({'pc': i, 'var': round(float(var[i]), 4),
                        'maxcos_dW': round(m1, 4),
                        'argmax_dW': CAT_WORDS[int(np.argmax(
                            np.abs(cos_dW)))],
                        'maxcos_dA': round(float(np.abs(cos_dA).max()), 4),
                        'top_unembed': toks})
    p_r2 = bool(max_cos_dW < 0.30)
    print('P2813 max|cos(PC,dW)|=%.4f P-R2=%s (NOTE: resid is '
          'orthogonal to the class span by construction, so this is '
          'exactly 0 — algebraic, 2809-type tautology, registered as '
          'erratum)' % (max_cos_dW, p_r2), flush=True)

    # exploratory (not preregged): PC-class association eta^2
    scores = U[:, :TOP_PC] * s[:TOP_PC]
    y_all = np.concatenate([np.array([ci for ci, c in
                                      enumerate(CAT_WORDS) for _ in
                                      CATS[c]]), ytrue])
    eta2 = []
    for i in range(TOP_PC):
        p_ = scores[:, i]
        grand = float(p_.mean())
        ss_tot = float(((p_ - grand) ** 2).sum())
        ss_bet = sum(float((p_[y_all == ci].mean() - grand) ** 2)
                     * int((y_all == ci).sum()) for ci in range(10)
                     if int((y_all == ci).sum()) > 0)
        eta2.append(round(ss_bet / max(ss_tot, 1e-30), 4))
    print('P2813 PC-class eta2 %s' % eta2, flush=True)
    print('P2813 top PCs %s' % json.dumps(pc_rows[:8]), flush=True)

    # ---------- Arm N: noun-pole axes ----------
    lower_known = set(w.lower() for w in atlas_words + eval_words
                      + CAT_WORDS)
    kept_np, np_rej = {}, {}
    used = set()
    for a, (pos, neg) in NPOLE.items():
        kp, kn, rj = [], [], []
        for pole, pool in (('pos', pos), ('neg', neg)):
            for w in pool:
                if w.lower() in lower_known or w.lower() in used:
                    rj.append(w + ' dup')
                    continue
                try:
                    tid(w)
                except AssertionError:
                    rj.append(w + ' multi-tok')
                    continue
                (kp if pole == 'pos' else kn).append(w)
                used.add(w.lower())
        np_rej[a] = rj
        if len(kp) >= 2 and len(kn) >= 2:
            kept_np[a] = (kp, kn)
    K2 = len(kept_np)
    print('P2813 noun-pole axes kept %d/%d %s'
          % (K2, len(NPOLE), list(kept_np)), flush=True)
    print('P2813 noun-pole rejections %s' % json.dumps(np_rej),
          flush=True)
    assert K2 >= 5, 'too few noun-pole axes survived'

    Znp = {a: (np.stack([zrow(w) for w in kp]),
               np.stack([zrow(w) for w in kn]))
           for a, (kp, kn) in kept_np.items()}
    dA2 = np.stack([unit(Znp[a][0].mean(0) - Znp[a][1].mean(0))
                    for a in kept_np])
    QA2, _ = np.linalg.qr(dA2.T)

    def share_mean(Z, Q):
        num = (Z @ Q) ** 2
        return float((num.sum(1)
                      / np.maximum((Z ** 2).sum(1), 1e-30)).mean())

    share_eval = share_mean(Z_eval, QA2)
    null_shares = np.empty(N_SHUFFLE)
    for it in range(N_SHUFFLE):
        rows = []
        for a in kept_np:
            kp, kn = kept_np[a]
            words = kp + kn
            perm = rng.permutation(len(words))
            p_sel = [words[i] for i in perm[:len(kp)]]
            n_sel = [words[i] for i in perm[len(kp):]]
            zp = np.stack([zrow(w) for w in p_sel]).mean(0)
            zn = np.stack([zrow(w) for w in n_sel]).mean(0)
            rows.append(unit(zp - zn))
        QA_n, _ = np.linalg.qr(np.stack(rows).T)
        null_shares[it] = share_mean(Z_eval, QA_n)
    n1_med = float(np.median(null_shares))
    n1_q95 = float(np.quantile(null_shares, 0.95))
    p_r3a = bool(share_eval > n1_q95 and share_eval > 3 * n1_med)
    print('P2813 nounpole share eval=%.4f | N1 med=%.4f q95=%.4f '
          'P-R3a=%s' % (share_eval, n1_med, n1_q95, p_r3a), flush=True)

    stab_rows = []
    for a, (kp, kn) in kept_np.items():
        ps, ns = sorted(kp), sorted(kn)
        cutp, cutn = (len(ps) + 1) // 2, (len(ns) + 1) // 2
        pa, pb = ps[:cutp], ps[cutp:]
        na, nb = ns[:cutn], ns[cutn:]
        dA_a = unit(np.stack([zrow(w) for w in pa]).mean(0)
                    - np.stack([zrow(w) for w in na]).mean(0))
        dA_b = unit(np.stack([zrow(w) for w in pb]).mean(0)
                    - np.stack([zrow(w) for w in nb]).mean(0))
        cosv = float(abs(unit(dA_a) @ unit(dA_b)))
        pool = pb + nb
        m = len(pb)
        ident = set(range(m))
        null_cos = np.empty(N_HALF)
        for it in range(N_HALF):
            for _try in range(100):
                perm = rng.permutation(len(pool))
                if set(perm[:m].tolist()) != ident:
                    break
            p_sel = [pool[i] for i in perm[:m]]
            n_sel = [pool[i] for i in perm[m:]]
            dA_s = unit(np.stack([zrow(w) for w in p_sel]).mean(0)
                        - np.stack([zrow(w) for w in n_sel]).mean(0))
            null_cos[it] = abs(float(unit(dA_a) @ unit(dA_s)))
        q95 = float(np.quantile(null_cos, 0.95))
        stab_rows.append({'attr': a, 'cos_half': round(cosv, 4),
                          'q95': round(q95, 4), 'pass': bool(cosv > q95)})
    frac_pass = float(np.mean([r['pass'] for r in stab_rows]))
    p_r3b = bool(frac_pass >= 0.60)
    print('P2813 nounpole stability pass-frac=%.2f P-R3b=%s %s'
          % (frac_pass, p_r3b, json.dumps(stab_rows)), flush=True)
    p_r3 = bool(p_r3a and p_r3b)

    # ---------- extras ----------
    per_np_share = {a: round(float(np.mean(
        (Z_eval @ unit(Znp[a][0].mean(0) - Znp[a][1].mean(0))) ** 2
        / np.maximum((Z_eval ** 2).sum(1), 1e-30))), 5) for a in kept_np}
    per_np_top = {a: [eval_words[i] for i in np.argsort(
        -(Z_eval @ unit(Znp[a][0].mean(0) - Znp[a][1].mean(0))))[:8]]
        for a in kept_np}
    pc_vs_dA = [[round(float(abs(unit(Vt[i]) @ dA12[j])), 4)
                 for j in range(14)] for i in range(TOP_PC)]

    verdict = {
        'n_words_total': len(all_words), 'n_eval': len(eval_words),
        'cum_var_top10': round(float(var[:10].sum()), 4),
        'cum_var_top20': round(cum20, 4),
        'cum_var_top50': round(float(var[:50].sum()), 4),
        'spectrum_head': [round(float(v), 5) for v in var[:20]],
        'residual_lowdim': p_r1,
        'max_cos_pc_dW': round(max_cos_dW, 4), 'pc_class_decoupled': p_r2,
        'pc_class_eta2_top20': eta2,
        'pc_table': pc_rows, 'pc_vs_dA_matrix': pc_vs_dA,
        'n_npole_axes': K2, 'kept_npole': list(kept_np),
        'npole_rejected': np_rej,
        'npole_share_eval': round(share_eval, 5),
        'n1_median': round(n1_med, 5), 'n1_q95': round(n1_q95, 5),
        'npole_share_real': p_r3a,
        'npole_stability': stab_rows,
        'npole_stability_pass_frac': round(frac_pass, 3),
        'npole_stable': p_r3b, 'npole_axes_real': p_r3,
        'per_npole_share': per_np_share, 'per_npole_top': per_np_top,
        'residual_geometry': ('lowdim' if (p_r1 and p_r2) else 'flat'),
        'nounpole_support': p_r3,
    }
    result = {'phase': 2813, 'prereg': PREREG, 'verdict': verdict,
              'eval_words': eval_words}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'residual.npz',
           Vt20=Vt[:TOP_PC].astype(np.float32),
           spectrum=var.astype(np.float32),
           dA2=dA2.astype(np.float32),
           Z_all=Z_all.astype(np.float32),
           ytrue=ytrue)

    elapsed = time.monotonic() - t0
    cc.ledger('phase2813', elapsed)
    print('P2813 VERDICT %s' % json.dumps(verdict), flush=True)
    print('P2813 elapsed %.1fs' % elapsed, flush=True)


if __name__ == '__main__':
    main()
