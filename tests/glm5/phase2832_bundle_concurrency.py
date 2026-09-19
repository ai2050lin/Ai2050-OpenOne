"""Phase 2832 = ATTRIBUTE BUNDLE CONCURRENCY (LPF-44).

User question 2: when processing '红色的圆形的甜的苹果' (the red round
sweet apple), how are multiple attribute bundles invoked
simultaneously?

Design (all-Chinese, 8-direction set from 2831):
  conds: base '苹果' (apple@0); single '红色的苹果' / '圆形的苹果' /
  '甜的苹果' (apple@2); triple '红色的圆形的甜的苹果' (apple@6).
  Capture full-token spectra c[L, pos, head, 8 dirs] per condition.

Prereg (frozen):
  A1_additive: triple apple-position spectrum on the 3 modified
      directions (red/round/sweet) matches the SUM of the three
      single-condition spectra: residual rate
      ||c_tri - sum(c_i)|| / ||sum(c_i)|| < 0.3
  A2_writer_locality: in single conditions, each modified direction's
      spectrum peaks at its own modifier token position (writer sits
      at the adjective), not at unrelated positions
  A3_head_reuse: top-5 head sets of the triple condition across
      red/round/sweet share heads (Jaccard > 0, bundle head reuses
      the same hardware)
  A4_saturation: triple top10 mean per direction vs single: classify
      super/linear/sub-additive (report; pass = all three directions
      non-negative net contribution)
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
OUT = BASE / 'phase2832' / 'bundle_concurrency'
SEED = 2832
N_NULL = 100
COLOR7 = ['红色', '黑色', '绿色', '蓝色', '黄色', '白色', '紫色']
PAIRS = {
    'big': ('大', '小'),
    'heavy': ('重', '轻'),
    'sweet': ('甜', '苦'),
    'hard': ('硬', '软'),
    'hot': ('热', '冷'),
    'fast': ('快', '慢'),
    'round': ('圆', '方'),
}
DIR_ORDER = ['red'] + list(PAIRS.keys())
CONDS = ['base', 'red', 'round', 'sweet', 'triple']
SENTS = {
    'base': '苹果',
    'red': '红色的苹果',
    'round': '圆形的苹果',
    'sweet': '甜的苹果',
    'triple': '红色的圆形的甜的苹果',
}
APPLE_POS = {'base': 0, 'red': 2, 'round': 2, 'sweet': 2, 'triple': 6}
MOD_DIR = {'red': 'red', 'round': 'round', 'sweet': 'sweet'}
BUNDLE_DIRS = ['red', 'round', 'sweet']

PREREG = {
    'A1': 'additive: residual ||c_tri - sum(c_i)|| / ||sum(c_i)|| '
          'over bundle dirs at apple position < 0.3',
    'A2': 'writer_locality: each single condition peaks its own '
          'direction at its modifier position',
    'A3': 'head_reuse: triple top-5 head sets across 3 bundle dirs '
          'have nonzero pairwise Jaccard',
    'A4': 'saturation: all 3 bundle dirs keep non-negative net '
          'contribution in triple vs singles (report class)',
}


def unit(x):
    return x / max(np.linalg.norm(x), 1e-30)


def main():
    t0 = time.monotonic()
    OUT.mkdir(parents=True, exist_ok=True)

    execution = {
        'timestamp': fc.stamp(), 'source': cc.snapshot(__file__),
        'prereg': PREREG, 'seed': SEED, 'conds': CONDS,
        'sents': SENTS, 'apple_pos': APPLE_POS,
        'dir_order': DIR_ORDER,
        'note': 'attribute bundle concurrency: single vs triple '
                'modifier sentences, additivity + writer locality + '
                'head reuse'}
    fc.save(OUT / 'execution.json', execution)

    # ---------- tensors + directions ----------
    from safetensors import safe_open
    mdir = ROOT / 'models' / 'hf' / 'qwen3-4b'
    index = json.loads((mdir / 'model.safetensors.index.json')
                       .read_text(encoding='utf-8'))['weight_map']

    def read_tensor(name):
        with safe_open(str(mdir / index[name]), framework='pt') as f:
            return f.get_tensor(name).float().numpy()

    try:
        Wu = read_tensor('lm_head.weight')
    except KeyError:
        Wu = read_tensor('model.embed_tokens.weight')
    cfg = json.loads((mdir / 'config.json').read_text(encoding='utf-8'))
    n_layers = int(cfg.get('num_hidden_layers', 36))
    n_heads = int(cfg.get('num_attention_heads', 32))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        str(mdir), local_files_only=True, trust_remote_code=True,
        use_fast=True)
    tc = {}

    def tid(t):
        if t not in tc:
            ids = tok(t, add_special_tokens=False)['input_ids']
            assert len(ids) == 1, (t, len(ids))
            tc[t] = int(ids[0])
        return tc[t]

    cent_red = np.stack([Wu[tid(w)].astype(np.float64)
                         for w in COLOR7])
    dirs = {'red': unit(cent_red[0] - cent_red[1:].mean(0))}
    for d, (w1, w2) in PAIRS.items():
        dirs[d] = unit(Wu[tid(w1)].astype(np.float64)
                       - Wu[tid(w2)].astype(np.float64))
    DMAT = np.stack([dirs[d] for d in DIR_ORDER], axis=1)

    # ---------- model ----------
    from transformers import AutoModelForCausalLM
    import torch
    model = AutoModelForCausalLM.from_pretrained(
        str(mdir), dtype=torch.bfloat16, device_map='auto')
    model.eval()
    dev = model.device

    enc = {c: tok(SENTS[c], add_special_tokens=False)['input_ids']
           for c in CONDS}
    for c in CONDS:
        assert enc[c][APPLE_POS[c]] == tid('苹果'), c
    if 'triple' in enc:
        assert len(enc['triple']) == 7

    Wo0 = read_tensor('model.layers.0.self_attn.o_proj.weight')
    d_model, hd_total = Wo0.shape
    hd = hd_total // n_heads
    del Wo0

    store = {}

    def make_cap(cid, Lr):
        def cap(mod, args):
            store[(cid, Lr)] = args[0][0].detach().float() \
                .cpu().numpy()
            return None
        return cap

    def run_cond(cid):
        hs = [model.model.layers[i].self_attn.o_proj
              .register_forward_pre_hook(make_cap(cid, i))
              for i in range(n_layers)]
        ids = torch.tensor([enc[cid]], dtype=torch.long)
        with torch.no_grad():
            model(input_ids=ids.to(dev))
        for h in hs:
            h.remove()

    for c in CONDS:
        run_cond(c)
    print('P2832 forward done', flush=True)

    spec = {}
    for c in CONDS:
        sp = np.zeros((n_layers, len(enc[c]), n_heads, 8),
                      dtype=np.float32)
        for Lr in range(n_layers):
            W = read_tensor(
                'model.layers.%d.self_attn.o_proj.weight' % Lr)
            W3 = W.astype(np.float64).reshape(d_model, n_heads, hd)
            K3 = store[(c, Lr)].reshape(len(enc[c]), n_heads, hd)
            delta = np.einsum('dhk,phk->pdh', W3, K3)
            sp[Lr] = np.einsum('pdh,dq->phq', delta, DMAT) \
                .astype(np.float32)
            del W, W3, delta
        spec[c] = sp
    print('P2832 spectra done', flush=True)

    # ---------- random null ----------
    rng = np.random.default_rng(SEED)
    U = rng.standard_normal((N_NULL, d_model))
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    null_parts = []
    for Lr in range(0, n_layers, 3):
        W = read_tensor('model.layers.%d.self_attn.o_proj.weight' % Lr)
        W3 = W.astype(np.float64).reshape(d_model, n_heads, hd)
        K3 = store[('base', Lr)].reshape(len(enc['base']), n_heads, hd)
        delta = np.einsum('dhk,phk->pdh', W3, K3)
        c = np.einsum('pdh,dq->phq', delta, U.T.astype(np.float64))
        null_parts.append(c.ravel())
        del W, W3, delta
    null_q95 = float(np.quantile(np.concatenate(null_parts), 0.95))
    print('P2832 null q95 %.4f' % null_q95, flush=True)

    def top10_sum(c, pos, di):
        return float(np.sort(spec[c][:, pos, :, di]
                             .astype(np.float64).sum(axis=0))
                     [::-1][:10].mean())

    # A1: additivity at apple position over bundle dirs
    di_b = [DIR_ORDER.index(d) for d in BUNDLE_DIRS]
    c_tri = np.array([top10_sum('triple', APPLE_POS['triple'], di)
                      for di in di_b])
    c_sum = np.array([sum(top10_sum(c, APPLE_POS[c], di)
                          for c in ['red', 'round', 'sweet'])
                      for di in di_b])
    resid = float(np.linalg.norm(c_tri - c_sum)
                  / max(np.linalg.norm(c_sum), 1e-30))
    a1 = bool(resid < 0.3)

    # A2: writer locality (peak head-sum position per direction)
    peak_report = {}
    a2_checks = []
    for c in ['red', 'round', 'sweet']:
        di = DIR_ORDER.index(MOD_DIR[c])
        pos_sums = np.array([spec[c][:, p, :, di].astype(np.float64)
                             .sum().sum()
                             for p in range(len(enc[c]))])
        peak = int(np.argmax(pos_sums))
        peak_report[c] = {'peak_pos': peak,
                          'peak_tok': tok.decode([enc[c][peak]]),
                          'pos_sums': [round(float(x), 2)
                                       for x in pos_sums]}
        a2_checks.append(peak in (0, 1))
    a2 = bool(all(a2_checks))

    # A3: head reuse in triple condition
    top5 = {}
    for d in BUNDLE_DIRS:
        di = DIR_ORDER.index(d)
        hs = spec['triple'][:, APPLE_POS['triple'], :, di] \
            .astype(np.float64).sum(axis=0)
        top5[d] = set(int(i) for i in np.argsort(-hs)[:5])
    jac = {}
    for i, d1 in enumerate(BUNDLE_DIRS):
        for d2 in BUNDLE_DIRS[i + 1:]:
            jac['%s-%s' % (d1, d2)] = round(
                len(top5[d1] & top5[d2])
                / max(len(top5[d1] | top5[d2]), 1), 3)
    a3 = bool(any(v > 0 for v in jac.values()))

    # A4: saturation class
    sat = {}
    for k, d in enumerate(BUNDLE_DIRS):
        singles = sum(top10_sum(c, APPLE_POS[c], di_b[k])
                      for c in ['red', 'round', 'sweet'])
        tri = c_tri[k]
        sat[d] = {'triple': round(float(tri), 3),
                  'sum_singles': round(float(singles), 3),
                  'ratio': round(float(tri / max(singles, 1e-30)), 3)}
    a4 = bool(all(v['ratio'] > 0 for v in sat.values()))

    verdict = {
        'null_q95': round(null_q95, 4),
        'A1_additive': a1, 'residual_rate': round(resid, 3),
        'A2_writer_locality': a2, 'peak_report': peak_report,
        'A3_head_reuse': a3, 'top5_jaccard': jac,
        'A4_nonneg': a4, 'saturation': sat,
        'apple_base_top10': {d: round(top10_sum('base', 0, di), 3)
                             for di, d in enumerate(DIR_ORDER)},
        'load_run_seconds': round(time.monotonic() - t0, 1),
    }
    fc.save(OUT / 'result.json', verdict)
    np.savez(OUT / 'bundle_spec.npz',
             **{'spec_%s' % c: spec[c] for c in CONDS},
             enc=np.array([enc[c] + [-1] * (8 - len(enc[c]))
                           for c in CONDS]))
    print('P2832 verdict %s' % json.dumps(
        {k: v for k, v in verdict.items()
         if k.startswith('A') and isinstance(v, bool)}), flush=True)
    print('P2832 seconds %.1f' % (time.monotonic() - t0), flush=True)


if __name__ == '__main__':
    main()
