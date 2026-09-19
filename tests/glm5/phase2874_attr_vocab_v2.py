# -*- coding: utf-8 -*-
"""Phase 2874: attribute-axis vocabulary expansion v2 (zero forward).

Goal: fix the power bottleneck diagnosed in 2873 (2-5 words/axis < the
28-80 word detection window).  Expand every attr axis to >= 5
single-token polar words, then gate the vocabulary with the 2858-style
geometry criteria BEFORE any census forward is spent.

Pool (per axis, polarity-ordered pairs (+pole, -pole)):
  size       big/small large/little huge/tiny
  speed      fast/slow quick/slow rapid/slow swift/slow
  temp       hot/cold warm/cool boiling/freezing
  age        old/new young/old ancient/modern
  weight     heavy/light hefty/light heavy/lightweight heavy/weightless
  strength   strong/weak powerful/weak sturdy/weak strong/feeble
  brightness bright/dark luminous/dim brilliant/dark
  moisture   wet/dry damp/arid humid/dry
  height     tall/short high/low towering/short
  fullness   full/empty packed/hollow vacant/empty
Note: 'high/low' is broader than height (polysemy risk logged as
limitation, not a gate).

Prereg (frozen before any unembed statistic is computed):
  VG1  coverage: every axis keeps >= 5 single-token words; overall pass
       iff >= 8/10 axes pass (failing axes quarantined from census v2).
  VG2a axis-direction orthogonality: max cross-axis |cos(dir_a,dir_b)|
       < 0.488 (the 2858 G2 gate, frozen unchanged - no goalpost move).
  VG2b within-axis pair reproducibility: for axes with >= 2 valid
       pairs, mean same-axis pair-direction cos > 0.2 (2870 measured
       0.46-0.59 for existing pairs).
  VG3  norm health: max per-axis word-unembed norm ratio (max/min) < 3.0.
  Verdict vocab_legal iff VG1 ∧ VG2a ∧ VG2b ∧ VG3.
Output: vocab npz in the exact schema AttrCensus consumes (targets,
target_list, dW_unit, null_tids, func_tid, pairs_json) so phase2875
runs the census without re-deriving anything.
"""
import hashlib
import io
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2874', 'attr_vocab_v2')
MODEL_DIR = r'D:\AI2050\Ai2050-OpenOne\models\hf\qwen3-4b'
SEED = 2874

AXES = {
    'size': [('size1', 'big', 'small'), ('size2', 'large', 'little'),
             ('size3', 'huge', 'tiny')],
    'speed': [('speed1', 'fast', 'slow'), ('speed2', 'quick', 'slow'),
              ('speed3', 'rapid', 'slow'), ('speed4', 'swift', 'slow')],
    'temp': [('temp1', 'hot', 'cold'), ('temp2', 'warm', 'cool'),
             ('temp3', 'boiling', 'freezing')],
    'age': [('age1', 'old', 'new'), ('age2', 'young', 'old'),
            ('age3', 'ancient', 'modern')],
    'weight': [('weight1', 'heavy', 'light'),
               ('weight2', 'hefty', 'light'),
               ('weight3', 'heavy', 'lightweight'),
               ('weight4', 'ponderous', 'light')],
    'strength': [('strength1', 'strong', 'weak'),
                 ('strength2', 'powerful', 'weak'),
                 ('strength3', 'sturdy', 'weak'),
                 ('strength4', 'mighty', 'weak')],
    'brightness': [('brightness1', 'bright', 'dark'),
                   ('brightness2', 'brilliant', 'dim'),
                   ('brightness3', 'radiant', 'dark')],
    'moisture': [('moisture1', 'wet', 'dry'),
                 ('moisture2', 'damp', 'dry'),
                 ('moisture3', 'soggy', 'dry')],
    'height': [('height1', 'tall', 'short'), ('height2', 'high', 'low'),
               ('height3', 'towering', 'short')],
    'fullness': [('fullness1', 'full', 'empty'),
                 ('fullness2', 'packed', 'hollow'),
                 ('fullness3', 'vacant', 'empty')],
}

PREREG = {
    'VG1': 'every axis >= 5 single-token words; overall pass iff >= 8/10 '
           'axes pass (failures quarantined)',
    'VG2a': 'max cross-axis |cos(dir_a,dir_b)| < 0.488 (2858 G2 gate)',
    'VG2b': 'mean same-axis pair-pair dir cos > 0.2 (axes with >=2 pairs)',
    'VG3': 'max per-axis word norm ratio max/min < 3.0',
    'verdict': 'vocab_legal iff VG1 and VG2a and VG2b and VG3',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def load_embed(model_dir):
    idx_path = os.path.join(model_dir, 'model.safetensors.index.json')
    if os.path.exists(idx_path):
        idx = json.load(io.open(idx_path, encoding='utf-8'))
        shard = idx['weight_map']['model.embed_tokens.weight']
    else:
        shard = 'model.safetensors'
    from safetensors import safe_open
    with safe_open(os.path.join(model_dir, shard), framework='pt') as f:
        emb = f.get_tensor('model.embed_tokens.weight').float().numpy()
    return emb


def log(msg):
    print(msg, flush=True)


def main():
    t0 = time.monotonic()
    os.makedirs(OUT, exist_ok=True)

    # ---- execution.json frozen BEFORE any unembed statistic ----
    script = os.path.abspath(__file__)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2874, 'name': 'attr_vocab_v2',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(script),
                   'prereg': PREREG, 'seed': SEED,
                   'n_pool_pairs': sum(len(v) for v in AXES.values())},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=True,
        use_fast=True)

    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from rdc_atlas_census import single_token_id

    emb = load_embed(MODEL_DIR)
    log('embed loaded %s' % (emb.shape,))

    # ---------- VG1: single-token filtering ----------
    tc = {}
    axis_words = {}
    for axis, pairs in AXES.items():
        ws = []
        for _, wp, wm in pairs:
            for w in (wp, wm):
                if w not in ws:
                    ws.append(w)
        kept = []
        for w in ws:
            try:
                single_token_id(tok, w, tc)
                kept.append(w)
            except AssertionError:
                pass
        axis_words[axis] = kept
        log('VG1 %-11s kept %d: %s' % (axis, len(kept), ','.join(kept)))

    g1_axes = {a: len(ws) >= 5 for a, ws in axis_words.items()}
    n_g1 = sum(g1_axes.values())
    vg1 = bool(n_g1 >= 8)
    axes = [a for a in AXES if g1_axes[a]]
    log('VG1 pass axes %d/10 -> %s' % (n_g1, vg1))

    # ---------- axis directions from valid pairs ----------
    axis_dirs = {}
    pair_dirs = {}
    for a in axes:
        dirs = []
        for _, wp, wm in AXES[a]:
            if wp in axis_words[a] and wm in axis_words[a]:
                d = unit(emb[tc[wp]] - emb[tc[wm]])
                dirs.append(d)
                pair_dirs.setdefault(a, []).append(d)
        axis_dirs[a] = unit(np.stack(dirs).mean(axis=0))
    axes_ok = list(axis_dirs)
    dW = np.stack([axis_dirs[a] for a in axes_ok])

    # ---------- VG2a: cross-axis orthogonality ----------
    C = dW @ dW.T
    off = C[np.triu_indices(len(axes_ok), 1)]
    max_off = float(np.abs(off).max())
    vg2a = bool(max_off < 0.488)

    # ---------- VG2b: same-axis pair reproducibility ----------
    pp = []
    for a, dirs in pair_dirs.items():
        if len(dirs) >= 2:
            D = np.stack(dirs)
            S = D @ D.T
            iu = np.triu_indices(len(dirs), 1)
            pp.append(float(S[iu].mean()))
    mean_pp = float(np.mean(pp)) if pp else float('nan')
    vg2b = bool(mean_pp > 0.2)

    # ---------- VG3: norm health ----------
    worst_ratio = 0.0
    for a in axes_ok:
        norms = [float(np.linalg.norm(emb[tc[w]]))
                 for w in axis_words[a]]
        worst_ratio = max(worst_ratio, max(norms) / max(min(norms), 1e-30))
    vg3 = bool(worst_ratio < 3.0)

    verdict = bool(vg1 and vg2a and vg2b and vg3)

    # ---------- save census-ready vocab ----------
    target_list = [(a, w) for a in axes_ok for w in axis_words[a]]
    n_words = len(target_list)
    rng = np.random.default_rng(SEED)
    word_tids = set(tc.values())
    null_tids = {}
    while len(null_tids) < n_words:
        r = int(rng.integers(0, emb.shape[0]))
        if r not in word_tids and r > 0:
            null_tids[len(null_tids)] = r
    func_tid = single_token_id(tok, 'the', tc)

    res = {
        'phase': 2874,
        'prereg': PREREG,
        'axis_word_counts': {a: len(axis_words[a]) for a in axes_ok},
        'quarantined_axes': [a for a in AXES if a not in axes_ok],
        'VG1': {'pass_axes': n_g1, 'verdict': vg1},
        'VG2a': {'max_cross_axis_cos': round(max_off, 4),
                 'gate': 0.488, 'verdict': vg2a},
        'VG2b': {'mean_pair_pair_cos': round(mean_pp, 4),
                 'gate': 0.2, 'n_axes_eval': len(pp), 'verdict': vg2b},
        'VG3': {'worst_norm_ratio': round(worst_ratio, 4),
                'gate': 3.0, 'verdict': vg3},
        'vocab_legal': verdict,
        'n_words_v2': n_words,
        'n_pairs_v2': sum(len(pair_dirs[a]) for a in axes_ok),
        'final_verdict': 'vocab_legal=%s (%d axes, %d words, %d pairs)'
                         % (verdict, len(axes_ok), n_words,
                            sum(len(pair_dirs[a]) for a in axes_ok)),
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    np.savez_compressed(
        os.path.join(OUT, 'attr_vocab_v2.npz'),
        targets=np.array(json.dumps(axis_words), dtype=object),
        pairs_json=np.array(json.dumps(AXES), dtype=object),
        axes=np.array(axes_ok, dtype=object),
        target_list=np.array(['%s:%s' % t for t in target_list],
                             dtype=object),
        dW_unit=dW.astype(np.float32),
        null_tids=np.array([null_tids[i] for i in range(n_words)],
                           dtype=np.int64),
        func_tid=np.array(func_tid, dtype=np.int64),
        tid_map=np.array(json.dumps(tc), dtype=object),
        n_words=np.array(n_words, dtype=np.int64))

    log('==== VERDICT ====')
    log(json.dumps({k: res[k] for k in
                    ('VG1', 'VG2a', 'VG2b', 'VG3', 'vocab_legal',
                     'final_verdict')}, ensure_ascii=False))
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
