"""Phase 2862 (LPF MA3 / MASTER_PLAN II1 closeout #1): front-edge
mechanism anatomy -- what carries the causal drop of the top-64 heads?

Zero-forward phase.  Inputs (all immutable):
  - 2846 census_full.npz: mean_drop / mean_s0 / mean_s1 over 1152 heads
    (align spectrum = s0+s1 direct cdir writes; causal spectrum = drops)
  - 2859 atlas_stability.npz: top10_full (front-edge legality check)
  - live weights W_V / W_O (GQA-aware) + W_U + SEED=2855 vocab (same
    construction as 2860/2861, words only, no forward pass)

Per-head end-to-end class-direction gain through the OV channel:
  g[h, c]    = cdir_c . W_O^h W_V^{kv(h)} . cdir_c      (cdir->cdir)
  wn[h, c]   = || W_O^h W_V^{kv(h)} . cdir_c ||_2        (total write)
with W_V^{kv(h)} the kv-slice of v_proj (group=4), W_O^h the per-query
head slice of o_proj.  Pure algebra, no activations.

Prereg (frozen before readout):
  M1  ov_carries_causal iff BOTH
      (a) Spearman(mean_drop, max_c|g[h,c]|) over all 1152 heads > 0.15
          (2846 C3 direct-drop full-set scale 0.1496 as reference),
      (b) mean over top-64 (by mean_drop desc) of max_c|g| >
          p95 of 200 random-64 null draws of the same statistic
  M2  descriptive: role mix in top-64 vs all heads.  Roles by frozen
    quantiles of the full set: formatter = drop>=p75 AND direct<=p50;
    amplifier = direct>=p75 AND drop<=p50; direct = mean_s0+mean_s1.
  M3  descriptive: layer thirds (L0-11 / L12-23 / L24-35) occupancy of
    top-64.
  M4  descriptive: top-1 head (L13H30) per-class g and wn spectrum --
    first single-head mechanism dossier of the response atlas.
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
OUT = BASE / 'phase2862' / 'frontedge_anatomy'
SRC_2806_EXEC = BASE / 'phase2806' / 'qwen4_hierarchy' / 'execution.json'
SRC_CENSUS = BASE / 'phase2846' / 'fullhead_census' / 'census_full.npz'
SRC_2859 = BASE / 'phase2859' / 'atlas_stability' / 'atlas_stability.npz'
SEED = 2855
MAX_WORDS = 8
NL, NH, HD, GROUP = 36, 32, 128, 4
N_TOP = 64
N_NULL = 200
NULL_SEED = 2862

PREREG = {
    'M1': 'ov_carries_causal iff Spearman(mean_drop, max_c|g|) > 0.15 '
          'AND mean top64(max_c|g|) > p95 of 200 random-64 nulls',
    'M2': 'descriptive role mix (quantiles p75/p50 of full set): '
          'formatter drop>=p75 & direct<=p50; amplifier direct>=p75 & '
          'drop<=p50; direct=s0+s1',
    'M3': 'descriptive: top-64 layer-thirds occupancy',
    'M4': 'descriptive: L13H30 per-class g/wn spectrum',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def spearman(a, b):
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)
    exec2806 = json.loads(SRC_2806_EXEC.read_text(encoding='utf-8'))
    CATS = exec2806['cats']
    CAT_WORDS = list(CATS.keys())

    execution_path = OUT / 'execution.json'
    if not execution_path.exists():
        execution = {'timestamp': fc.stamp(),
                     'source': cc.snapshot(__file__),
                     'prereg': PREREG, 'seed': SEED,
                     'design': 'zero-forward front-edge anatomy: OV '
                               'class-direction gain g/wn over 1152 '
                               'heads x 10 classes vs 2846 census '
                               'spectra; top-64 by mean_drop'}
        fc.save(execution_path, execution)

    # ---------- immutable census ----------
    cz = np.load(SRC_CENSUS)
    mean_drop = cz['mean_drop'].astype(np.float64)   # (1152,)
    mean_s0 = cz['mean_s0'].astype(np.float64)
    mean_s1 = cz['mean_s1'].astype(np.float64)
    assert mean_drop.shape == (NL * NH,)

    az = np.load(SRC_2859)
    top10_2859 = az['top10_full'].astype(np.int64)   # (10,)

    order = np.argsort(mean_drop)[::-1]
    top64 = order[:N_TOP]
    c0_ok = bool(set(top10_2859.tolist()).issubset(set(top64.tolist())))

    # ---------- vocab -> class directions (words only, no forward) ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(ROOT / 'models' / 'hf' / 'qwen3-4b'), local_files_only=True,
        trust_remote_code=True, use_fast=True)
    from phase2662_symmetric_mapping_contract import load_native
    model, _ = load_native('qwen4')
    model.eval()

    W_U = model.lm_head.weight.detach().float().cpu().numpy()
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(' ' + t, add_special_tokens=False)['input_ids']
            if len(ids) != 1:
                ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, '%s -> %s' % (t, ids)
            tc[t] = int(ids[0])
        return tc[t]

    all_words = [w for v in CATS.values() for w in v]
    single_tok = []
    for w in all_words:
        try:
            tid(w)
            single_tok.append(w)
        except AssertionError:
            pass
    Erows = {w: W_U[tid(w)].astype(np.float64) for w in single_tok}
    cents = []
    for cat in CAT_WORDS:
        ws = [w for w in CATS[cat] if w in single_tok]
        cents.append(np.stack([Erows[w] for w in ws]).mean(0))
    Cm = np.stack(cents)
    dW = Cm - (Cm.sum(0, keepdims=True) - Cm) / 9.0
    dW_unit = np.stack([unit(dW[i]) for i in range(10)]).astype(np.float32)

    # ---------- OV algebra: g / wn over 1152 heads x 10 classes -------
    g = np.zeros((NL * NH, 10))
    wn = np.zeros((NL * NH, 10))
    layers = model.model.layers
    for li, layer in enumerate(layers):
        W_V_all = layer.self_attn.v_proj.weight.detach().float() \
            .cpu().numpy()                       # (n_kv*HD, 2560)
        W_O_all = layer.self_attn.o_proj.weight.detach().float() \
            .cpu().numpy()                       # (2560, NH*HD)
        for h in range(NH):
            kv = h // GROUP
            Wv = W_V_all[kv * HD:(kv + 1) * HD, :]      # (128, 2560)
            Wo = W_O_all[:, h * HD:(h + 1) * HD]        # (2560, 128)
            idx = li * NH + h
            for c in range(10):
                v_in = Wv @ dW_unit[c]                   # (128,)
                out = Wo @ v_in                          # (2560,)
                g[idx, c] = float(out @ dW_unit[c])
                wn[idx, c] = float(np.linalg.norm(out))
        if (li + 1) % 12 == 0:
            print('P2862 layers [%d/%d]' % (li + 1, NL), flush=True)

    g_absmax = np.max(np.abs(g), axis=1)
    direct = mean_s0 + mean_s1

    # ---------- M1 ----------
    rho = spearman(mean_drop, g_absmax)
    rng = np.random.default_rng(NULL_SEED)
    nulls = np.array([float(np.mean(g_absmax[
        rng.choice(NL * NH, size=N_TOP, replace=False)]))
        for _ in range(N_NULL)])
    top64_stat = float(np.mean(g_absmax[top64]))
    null_p95 = float(np.percentile(nulls, 95))
    m1 = bool(rho > 0.15 and top64_stat > null_p95)

    # ---------- M2 ----------
    drop_p75 = float(np.percentile(mean_drop, 75))
    drop_p50 = float(np.percentile(mean_drop, 50))
    dir_p75 = float(np.percentile(direct, 75))
    dir_p50 = float(np.percentile(direct, 50))
    is_form = (mean_drop >= drop_p75) & (direct <= dir_p50)
    is_ampl = (direct >= dir_p75) & (mean_drop <= drop_p50)
    m2 = {
        'form_share_all': round(float(is_form.mean()), 4),
        'ampl_share_all': round(float(is_ampl.mean()), 4),
        'form_share_top64': round(float(is_form[top64].mean()), 4),
        'ampl_share_top64': round(float(is_ampl[top64].mean()), 4),
        'n_form_top64': int(is_form[top64].sum()),
        'n_ampl_top64': int(is_ampl[top64].sum()),
    }

    # ---------- M3 ----------
    lay = top64 // NH
    m3 = {
        'early_L0_11': int(((lay >= 0) & (lay <= 11)).sum()),
        'mid_L12_23': int(((lay >= 12) & (lay <= 23)).sum()),
        'late_L24_35': int(((lay >= 24) & (lay <= 35)).sum()),
    }

    # ---------- M4 ----------
    t1_idx = int(top64[0])
    t1_l, t1_h = t1_idx // NH, t1_idx % NH
    m4 = {
        'head': 'L%dH%d' % (t1_l, t1_h),
        'matches_2859_top1': bool(t1_idx == int(top10_2859[0])),
        'drop': round(float(mean_drop[t1_idx]), 5),
        'direct_write': round(float(direct[t1_idx]), 5),
        'g_per_class': [round(float(x), 4)
                        for x in g[t1_idx]],
        'wn_per_class': [round(float(x), 3)
                         for x in wn[t1_idx]],
        'class_of_max_abs_g': CAT_WORDS[int(np.argmax(
            np.abs(g[t1_idx])))],
    }

    v = {
        'n_heads': NL * NH,
        'c0_top64_covers_2859_top10': c0_ok,
        'M1_ov_carries_causal': m1,
        'm1_rho_drop_vs_gabsmax': round(rho, 4),
        'm1_top64_stat': round(top64_stat, 4),
        'm1_null_p95': round(null_p95, 4),
        'M2_role_mix': m2,
        'M3_layer_thirds': m3,
        'M4_top1_dossier': m4,
        'g_absmax_overall_mean': round(float(g_absmax.mean()), 4),
        'final_verdict': ('ov_carries_causal' if m1
                          else 'ov_uncorrelated'),
    }

    result = {'phase': 2862, 'prereg': PREREG, 'verdict': v}
    fc.save(OUT / 'result.json', result)
    fc.npz(OUT / 'frontedge.npz',
           g=g.astype(np.float32), wn=wn.astype(np.float32),
           top64=top64.astype(np.int64),
           mean_drop=mean_drop.astype(np.float32),
           direct=direct.astype(np.float32))

    elapsed = time.monotonic() - t0
    cc.ledger('phase2862', elapsed)
    print('P2862 VERDICT %s' % json.dumps(v), flush=True)
    print('P2862 elapsed %.1fs' % elapsed, flush=True)

    del model
    import torch
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
