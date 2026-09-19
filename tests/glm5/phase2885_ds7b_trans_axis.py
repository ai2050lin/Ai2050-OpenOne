# -*- coding: utf-8 -*-
"""Phase 2885: DS7B translation-axis discrimination test (untied lm_head).

2878 (qwen3-4b, TIED unembed) found a real negative: no unified
cross-lingual axis direction (VG2b ~ 0 for all three translation axes
after homograph decontamination).  The attachment analysis (2884 MEMO)
flagged a testable confound the attachment itself raised but never
tested: tied-embedding geometric distortion.  DS7B has an UNTIED
lm_head -> running the identical 2878 translation-family protocol on
DS7B discriminates:

  D-  attributed_to_tied  iff >= 2 translation axes survive
      (VG2b > 0.2) on DS7B -> 2878 negative attributed to tying;
      unembed-level translation axis exists when output rows are free.
  D+  model_general_negative iff >= 2 translation axes have valid
      pairs (VG1 pass) but ALL VG2b <= 0.2 -> no unified cross-lingual
      axis in the untied unembed either; "translation != static axis"
      becomes a model-general fact (independent of tying), and the
      hourglass hypothesis (language info lives mid-stream, not at the
      endpoints) gains indirect support.

Protocol (verbatim 2878 Gen2, deviations declared):
  T1  translation family only (lang_fr/lang_de/lang_es pools,
      homograph exclusion list frozen verbatim); syntax family not
      needed for the discrimination.
  T2  E rows from lm_head.weight (DS7B untied; safetensors direct
      read, 2883 lesson) instead of embed_tokens.
  T3  gates verbatim: VG1 >= 5 pairs/axis, >= 3/3 axes; VG2a max
      cross-axis |cos| < 0.488; VG2b per-axis mean pair-pair cos
      > 0.2 for survival; VG3 norm ratio < 3.0.
  T4  SEED=2885; zero forward (embed/lm_head rows only).
Output: result/rdc_query_construction_20260913/phase2885/
        ds7b_trans_axis/{execution.json, result.json,
        ds7b_trans_axis.npz}
"""
import hashlib
import io
import json
import os
import time

import numpy as np

BASE = (r'D:\AI2050\Ai2050-OpenOne\tests\glm5\result'
        r'\rdc_query_construction_20260913')
OUT = os.path.join(BASE, 'phase2885', 'ds7b_trans_axis')
MODEL_DIR = (r'D:\AI2050\Ai2050-OpenOne\models\hf'
             r'\deepseek-r1-distill-qwen-7b')
SEED = 2885

TRANS_AXES = {
    'lang_fr': [('f1', 'cat', 'chat'), ('f2', 'house', 'maison'),
                ('f3', 'book', 'livre'), ('f4', 'night', 'nuit'),
                ('f5', 'roi', 'king'), ('f6', 'femme', 'woman'),
                ('f7', 'homme', 'man'), ('f8', 'arbre', 'tree'),
                ('f9', 'vin', 'wine'), ('f10', 'lune', 'moon'),
                ('f11', 'mer', 'sea'), ('f12', 'ciel', 'sky'),
                ('f13', 'fleur', 'flower'), ('f14', 'ville', 'city'),
                ('f15', 'guerre', 'war'), ('f16', 'pied', 'foot'),
                ('f17', 'water', 'eau'), ('f18', 'sun', 'soleil'),
                ('f19', 'milk', 'lait'), ('f20', 'dog', 'chien')],
    'lang_de': [('d1', 'cat', 'Katze'), ('d2', 'dog', 'Hund'),
                ('d3', 'water', 'Wasser'), ('d4', 'house', 'Haus'),
                ('d5', 'book', 'Buch'), ('d6', 'night', 'Nacht'),
                ('d7', 'sun', 'Sonne'), ('d8', 'milk', 'Milch'),
                ('d9', 'Vogel', 'bird'), ('d10', 'Fisch', 'fish'),
                ('d11', 'Brot', 'bread'), ('d12', 'Wein', 'wine'),
                ('d13', 'Mond', 'moon'), ('d14', 'Stern', 'star'),
                ('d15', 'Baum', 'tree'), ('d16', 'Feuer', 'fire'),
                ('d17', 'Berg', 'mountain'), ('d18', 'Schnee', 'snow'),
                ('d19', 'Frau', 'woman'), ('d20', 'Wolf', 'wolf'),
                ('d21', 'Koenig', 'king'), ('d22', 'Sommer', 'summer')],
    'lang_es': [('e1', 'water', 'agua'), ('e2', 'house', 'casa'),
                ('e3', 'book', 'libro'), ('e4', 'night', 'noche'),
                ('e5', 'cat', 'gato'), ('e6', 'dog', 'perro'),
                ('e7', 'milk', 'leche'), ('e8', 'mesa', 'table'),
                ('e9', 'luna', 'moon'), ('e10', 'mar', 'sea'),
                ('e11', 'cielo', 'sky'), ('e12', 'vino', 'wine'),
                ('e13', 'flor', 'flower'), ('e14', 'rey', 'king'),
                ('e15', 'mujer', 'woman'), ('e16', 'hombre', 'man'),
                ('e17', 'madre', 'mother'), ('e18', 'padre', 'father'),
                ('e19', 'guerra', 'war'), ('e20', 'luz', 'light'),
                ('e21', 'fuego', 'fire'), ('e22', 'tierra', 'earth'),
                ('e23', 'puerta', 'door'), ('e24', 'caballo', 'horse'),
                ('e25', 'queso', 'cheese'), ('e26', 'huevo', 'egg'),
                ('e27', 'lobo', 'wolf'), ('e28', 'nieve', 'snow'),
                ('e29', 'cama', 'bed'), ('e30', 'estrella', 'star')],
}
TRANS_FAMILY = ['lang_fr', 'lang_de', 'lang_es']

HOMOGRAPH_EXCLUDE = {
    'chat', 'sol', 'pain', 'chef', 'table', 'main', 'train', 'route',
    'lion', 'grand', 'mine', 'Mann', 'Gold', 'Winter', 'Hand', 'Bank',
    'Wind', 'Kind', 'Sommer', 'Morgen', 'Boot', 'Arm', 'pan', 'plaza',
    'real', 'color', 'metal', 'hotel', 'piano', 'radio', 'rosa',
}

PREREG = {
    'T1': 'translation family only, 2878 Gen2 pools + homograph '
          'exclusion list verbatim',
    'T2': 'E rows from lm_head.weight (DS7B untied), safetensors direct '
          'read; this is the discrimination lever vs qwen3-4b tied',
    'T3': 'gates verbatim 2878: VG1 >=5 pairs/axis and >=3/3 axes; VG2a '
          'max cross-axis |cos| < 0.488; VG2b per-axis mean pair-pair '
          'cos > 0.2 for survival; VG3 norm ratio < 3.0',
    'T4': 'zero forward; SEED=2885',
    'decision': 'D- attributed_to_tied iff >=2 translation axes survive; '
                'D+ model_general_negative iff >=2 axes pass VG1 but '
                'all VG2b <= 0.2; VG1 fail => vocab quarantine, '
                'inconclusive',
    'reference': 'qwen3-4b 2878 Gen2: VG2b fr -0.0007 / de -0.0029 / '
                 'es 0.0011 (all dead), trans_legal=false',
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:8]


def unit(v):
    return v / max(float(np.linalg.norm(v)), 1e-30)


def load_unembed(model_dir):
    idx = json.load(io.open(os.path.join(model_dir,
                                         'model.safetensors.index.json'),
                            encoding='utf-8'))
    wmap = idx['weight_map']
    key = None
    for k in ('lm_head.weight', 'model.lm_head.weight',
              'model.embed_tokens.weight'):
        if k in wmap:
            key = k
            break
    assert key is not None
    from safetensors import safe_open
    with safe_open(os.path.join(model_dir, wmap[key]),
                   framework='pt') as f:
        w = f.get_tensor(key).float().numpy()
    return w, key


def log(msg):
    print(msg, flush=True)


def main():
    t0 = time.monotonic()
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, 'execution.json'), 'w',
              encoding='utf-8') as f:
        json.dump({'phase': 2885, 'name': 'ds7b_trans_axis',
                   'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
                   'script_sha256_8': sha8(os.path.abspath(__file__)),
                   'prereg': PREREG, 'seed': SEED,
                   'n_pool_pairs': sum(len(v) for v in TRANS_AXES.values()),
                   'model': 'deepseek-r1-distill-qwen-7b'},
                  f, indent=2, ensure_ascii=False)
    log('execution.json frozen')

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=True,
        use_fast=True)

    import sys
    sys.path.insert(0, r'D:\AI2050\Ai2050-OpenOne\tests\glm5')
    from rdc_atlas_census import single_token_id

    W_U, ukey = load_unembed(MODEL_DIR)
    log('unembed rows from %s %s' % (ukey, W_U.shape,))

    # ---------- VG0 ----------
    n_excl = 0
    eff_axes = {}
    for axis, pairs in TRANS_AXES.items():
        keep = []
        for pid, wp, wm in pairs:
            if wp in HOMOGRAPH_EXCLUDE or wm in HOMOGRAPH_EXCLUDE:
                n_excl += 1
                continue
            keep.append((pid, wp, wm))
        eff_axes[axis] = keep
    log('VG0 excluded %d homograph pairs' % n_excl)

    # ---------- VG1 ----------
    tc = {}
    axis_words = {}
    axis_pairs = {}
    for axis, pairs in eff_axes.items():
        words = []
        valid = []
        for _, wp, wm in pairs:
            ok_p = ok_m = False
            try:
                single_token_id(tok, wp, tc)
                ok_p = True
            except AssertionError:
                pass
            try:
                single_token_id(tok, wm, tc)
                ok_m = True
            except AssertionError:
                pass
            if ok_p and ok_m:
                valid.append((wp, wm))
            for w, ok in ((wp, ok_p), (wm, ok_m)):
                if ok and w not in words:
                    words.append(w)
        axis_words[axis] = words
        axis_pairs[axis] = valid
        log('VG1 %-8s %d/%d pairs: %s'
            % (axis, len(valid), len(pairs),
               '; '.join('%s/%s' % p for p in valid)))

    g1 = {a: len(axis_pairs[a]) >= 5 for a in TRANS_AXES}
    tr_ok = sum(g1[a] for a in TRANS_FAMILY)
    vg1 = bool(tr_ok >= 3)
    log('VG1 translation %d/3 -> %s' % (tr_ok, vg1))

    # ---------- pair directions ----------
    axis_dirs = {}
    pair_dirs = {}
    for a in TRANS_AXES:
        dirs = []
        for wp, wm in axis_pairs[a]:
            d = unit(W_U[tc[wp]] - W_U[tc[wm]])
            dirs.append(d)
            pair_dirs.setdefault(a, []).append(d)
        if dirs:
            axis_dirs[a] = unit(np.stack(dirs).mean(axis=0))

    # ---------- VG2b ----------
    vg2b = {}
    for a, dirs in pair_dirs.items():
        if len(dirs) >= 2:
            D = np.stack(dirs)
            S = D @ D.T
            iu = np.triu_indices(len(dirs), 1)
            vg2b[a] = float(S[iu].mean())
        else:
            vg2b[a] = float('nan')
    surviving = [a for a in axis_dirs
                 if g1[a] and len(pair_dirs[a]) >= 2
                 and vg2b[a] > 0.2]
    for a in axis_dirs:
        log('VG2b %-8s %.4f -> %s' % (a, vg2b[a],
                                      'survive' if a in surviving
                                      else 'quarantined'))

    # ---------- VG2a / VG3 ----------
    vg2a = True
    max_off = float('nan')
    if len(surviving) >= 2:
        dW = np.stack([axis_dirs[a] for a in surviving])
        C = dW @ dW.T
        off = C[np.triu_indices(len(surviving), 1)]
        max_off = float(np.abs(off).max())
        vg2a = bool(max_off < 0.488)
    worst_ratio = 0.0
    worst_axis = ''
    for a in surviving:
        norms = [float(np.linalg.norm(W_U[tc[w]]))
                 for w in axis_words[a]]
        r = max(norms) / max(min(norms), 1e-30)
        if r > worst_ratio:
            worst_ratio, worst_axis = r, a
    vg3 = bool(worst_ratio < 3.0) if surviving else True

    n_tr = len(surviving)
    n_valid_axes = sum(1 for a in TRANS_FAMILY
                       if len(axis_pairs[a]) >= 5)
    if n_tr >= 2:
        decision = 'D-_attributed_to_tied'
    elif n_valid_axes >= 2 and all(
            not (vg2b.get(a, float('nan')) > 0.2)
            for a in TRANS_FAMILY if len(axis_pairs[a]) >= 2):
        decision = 'D+_model_general_negative'
    else:
        decision = 'inconclusive_vocab'
    trans_legal = bool(n_tr >= 2 and vg2a and vg3)

    res = {
        'phase': 2885,
        'model': 'deepseek-r1-distill-qwen-7b',
        'unembed_key': ukey,
        'prereg': PREREG,
        'n_homograph_excluded': n_excl,
        'VG1': {'translation_pass': tr_ok, 'verdict': vg1,
                'per_axis_pairs': {a: len(axis_pairs[a])
                                   for a in TRANS_AXES}},
        'VG2b_per_axis': {a: round(v, 4) for a, v in vg2b.items()},
        'surviving_axes': surviving,
        'VG2a': {'max_cross_axis_cos': (round(max_off, 4)
                                        if max_off == max_off else None),
                 'gate': 0.488, 'verdict': vg2a},
        'VG3': {'worst_norm_ratio': round(worst_ratio, 4),
                'worst_axis': worst_axis, 'gate': 3.0, 'verdict': vg3},
        'trans_legal': trans_legal,
        'decision': decision,
        'final_verdict': 'ds7b_trans_axis=%s (VG2b %s; qwen ref '
                         'fr -0.0007/de -0.0029/es 0.0011)'
                         % (decision,
                            {a: round(vg2b[a], 4) for a in vg2b}),
        'runtime_s': round(time.monotonic() - t0, 1),
    }
    with open(os.path.join(OUT, 'result.json'), 'w',
              encoding='utf-8') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    np.savez_compressed(
        os.path.join(OUT, 'ds7b_trans_axis.npz'),
        targets=np.array(json.dumps({a: axis_words[a]
                                     for a in surviving}),
                         dtype=object),
        pairs_json=np.array(json.dumps(
            {a: TRANS_AXES[a] for a in surviving}), dtype=object),
        axes=np.array(surviving, dtype=object),
        dW_unit=(np.stack([axis_dirs[a] for a in surviving]).astype(
            np.float32) if surviving else np.zeros((0, 1),
                                                   dtype=np.float32)),
        pair_dirs_json=np.array(json.dumps(
            {a: [d.tolist() for d in pair_dirs.get(a, [])]
             for a in TRANS_AXES}), dtype=object),
        tid_map=np.array(json.dumps(tc), dtype=object))

    log('==== VERDICT: %s ====' % res['final_verdict'])
    log('runtime %.1fs' % (time.monotonic() - t0))


if __name__ == '__main__':
    main()
